"""Shared helper: download a list of HF dataset files into a shard directory."""

from __future__ import annotations

from pathlib import Path

from ucc.logging_utils import get_logger
from ucc.sources.base import DownloadError

log = get_logger("sources.download")


def download_hf_files(
    repo_id: str,
    files: list[tuple[str, int]],
    revision: str | None,
    dest_dir: Path,
    token: str | None,
    stop_check=None,
) -> None:
    """Download each (remote_path, expected_size) into dest_dir, preserving
    the remote relative path. Size-validates every file."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(size for _, size in files)
    done_bytes = 0
    for idx, (remote_path, expected_size) in enumerate(files, 1):
        if stop_check is not None and stop_check():
            raise DownloadError("download interrupted by pipeline shutdown")
        log.info("downloading %s [file %d/%d] ...", remote_path, idx, len(files))
        try:
            # hf_hub_download renders its own live byte-level progress bar on
            # a TTY; the log lines below add per-shard percentages.
            local = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
                local_dir=str(dest_dir),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001 - hub/network layer
            raise DownloadError(
                f"failed downloading {repo_id}/{remote_path}: {exc.__class__.__name__}: {exc}"
            ) from exc
        actual = Path(local).stat().st_size
        if expected_size and actual != expected_size:
            raise DownloadError(
                f"size mismatch for {remote_path}: expected {expected_size}, got {actual}"
            )
        done_bytes += actual
        pct = (
            100.0 * done_bytes / total_bytes
            if total_bytes
            else 100.0 * idx / len(files)
        )
        log.info(
            "downloaded %s (%.1f MB) — shard download %5.1f%% (file %d/%d)",
            remote_path, actual / 1e6, pct, idx, len(files),
        )
