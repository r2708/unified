"""Shared helper: download a list of HF dataset files into a shard directory.

Executes the same way as the fineweb_pipeline shard downloader: each file is
fetched with hf_hub_download (hf_transfer fast path when installed) and a
transient failure retries the SAME file with exponential backoff + jitter
instead of failing the whole shard and re-downloading everything.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from ucc.logging_utils import get_logger
from ucc.sources.base import DownloadError

log = get_logger("sources.download")


def _download_with_retry(
    repo_id: str,
    remote_path: str,
    expected_size: int,
    revision: str | None,
    dest_dir: Path,
    token: str | None,
    stop_check,
    retries: int,
    backoff_base: float,
    backoff_max: float,
) -> Path:
    """Download one file, retrying transient failures (network errors and
    size mismatches) with exponential backoff + jitter. Returns the local path."""
    from huggingface_hub import hf_hub_download

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if stop_check is not None and stop_check():
            raise DownloadError("download interrupted by pipeline shutdown")
        try:
            local = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(dest_dir),
                    token=token,
                )
            )
            actual = local.stat().st_size
            if expected_size and actual != expected_size:
                # A truncated/partial file: drop it so the next attempt
                # re-fetches instead of trusting the local copy.
                local.unlink(missing_ok=True)
                raise OSError(
                    f"size mismatch: expected {expected_size}, got {actual}"
                )
            return local
        except Exception as exc:  # noqa: BLE001 - hub/network layer
            last_exc = exc
            if attempt == retries:
                break
            delay = min(backoff_max, backoff_base * (2**attempt))
            delay *= 0.5 + random.random() / 2.0  # jitter
            log.warning(
                "download failed for %s (attempt %d/%d): %s — retrying in %.1fs",
                remote_path, attempt + 1, retries, exc.__class__.__name__, delay,
            )
            time.sleep(delay)
    raise DownloadError(
        f"failed downloading {repo_id}/{remote_path} after {retries + 1} "
        f"attempts: {last_exc.__class__.__name__}: {last_exc}"
    ) from last_exc


def download_hf_files(
    repo_id: str,
    files: list[tuple[str, int]],
    revision: str | None,
    dest_dir: Path,
    token: str | None,
    stop_check=None,
    retries: int = 5,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
) -> None:
    """Download each (remote_path, expected_size) into dest_dir, preserving
    the remote relative path. Size-validates every file; transient failures
    retry per file with exponential backoff + jitter."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(size for _, size in files)
    done_bytes = 0
    for idx, (remote_path, expected_size) in enumerate(files, 1):
        if stop_check is not None and stop_check():
            raise DownloadError("download interrupted by pipeline shutdown")
        log.info("downloading %s [file %d/%d] ...", remote_path, idx, len(files))
        # hf_hub_download renders its own live byte-level progress bar on
        # a TTY; the log lines below add per-shard percentages.
        local = _download_with_retry(
            repo_id, remote_path, expected_size, revision, dest_dir, token,
            stop_check, retries, backoff_base, backoff_max,
        )
        actual = local.stat().st_size
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
