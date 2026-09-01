"""Upload processed shard outputs to the Hub and verify them.

Idempotent by construction: before uploading a file, the remote is checked —
if it already exists with a matching checksum the upload is skipped, so a
crashed-and-resumed upload never duplicates files or overwrites valid output.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ucc.hf_remote import HubClient, verify_remote_file
from ucc.io_utils import read_json
from ucc.logging_utils import get_logger

log = get_logger("uploader")


def _run_parallel(fn, outputs: list[dict], workers: int):
    """Yield (output, result_or_exception) as each finishes. Order follows
    completion when parallel; input order when workers == 1."""
    workers = max(1, min(int(workers), len(outputs)))
    if workers == 1:
        for out in outputs:
            try:
                yield out, fn(out)
            except Exception as exc:  # noqa: BLE001 - reported by the caller
                yield out, exc
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, out): out for out in outputs}
        for fut in as_completed(futures):
            try:
                yield futures[fut], fut.result()
            except Exception as exc:  # noqa: BLE001 - reported by the caller
                yield futures[fut], exc


class UploadFailure(Exception):
    pass


class VerificationFailure(Exception):
    pass


def load_processed_outputs(processed_dir: str | Path) -> list[dict]:
    manifest_path = Path(processed_dir) / "processed_files.json"
    if not manifest_path.exists():
        raise UploadFailure(f"processed_files.json missing in {processed_dir}")
    blob = read_json(manifest_path)
    return list(blob["files"])


def upload_outputs(
    hub: HubClient, outputs: list[dict], shard_id: str, workers: int = 4
) -> int:
    """Upload every output file that isn't already present-and-matching, up
    to `workers` files in parallel (each upload is independent + idempotent:
    already-matching remotes are skipped, and the hub client retries
    transient failures internally). Logs live per-shard upload percentages.
    Returns how many files were actually transferred."""
    total_bytes = sum(o["size"] for o in outputs) or 1

    def _upload_one(out: dict) -> bool:
        """True when bytes were transferred, False when skipped as matching."""
        local = Path(out["local"])
        if not local.exists():
            raise UploadFailure(f"local processed file missing: {local}")
        ok, why = verify_remote_file(
            hub, local, out["dest"],
            expected_sha256=out.get("sha256"), expected_size=out.get("size"),
        )
        if ok:
            out["_skip_reason"] = why
            return False
        log.info("uploading %s (%.1f MB) ...", out["dest"], out["size"] / 1e6)
        hub.upload_file(local, out["dest"], message=f"ucc: shard {shard_id}")
        return True

    transferred = 0
    done_bytes = 0
    done_files = 0
    first_error: Exception | None = None
    for out, result in _run_parallel(_upload_one, outputs, workers):
        if isinstance(result, Exception):
            first_error = first_error or result
            log.error("upload failed for %s: %s", out["dest"], result)
            continue
        done_files += 1
        done_bytes += out["size"]
        if result:
            transferred += 1
            log.info(
                "uploaded %s — shard upload %5.1f%% (%.1f/%.1f MB, file %d/%d)",
                out["dest"], 100.0 * done_bytes / total_bytes,
                done_bytes / 1e6, total_bytes / 1e6, done_files, len(outputs),
            )
        else:
            log.info(
                "skip upload (already on hub, %s): %s — shard upload %5.1f%% "
                "(file %d/%d)",
                out.pop("_skip_reason", ""), out["dest"],
                100.0 * done_bytes / total_bytes, done_files, len(outputs),
            )
    if first_error is not None:
        raise first_error
    return transferred


def verify_outputs(hub: HubClient, outputs: list[dict], workers: int = 4) -> None:
    """Verify EVERY output file on the Hub against its recorded checksum and
    size (read-only metadata calls, up to `workers` in parallel). Raises
    VerificationFailure listing what failed."""

    def _verify_one(out: dict) -> tuple[bool, str]:
        local = Path(out["local"])
        return verify_remote_file(
            hub, local if local.exists() else None, out["dest"],
            expected_sha256=out.get("sha256"), expected_size=out.get("size"),
        )

    failures: list[str] = []
    for out, result in _run_parallel(_verify_one, outputs, workers):
        if isinstance(result, Exception):
            failures.append(f"{out['dest']}: {result}")
            continue
        ok, why = result
        if not ok:
            failures.append(f"{out['dest']}: {why}")
        else:
            log.info("verified %s (%s)", out["dest"], why)
    if failures:
        raise VerificationFailure("; ".join(sorted(failures)))


def verify_outputs_remote_only(hub: HubClient, outputs: list[dict]) -> bool:
    """Verification using only recorded checksums (local files may be gone).
    Used by resume logic after a crash that lost local state."""
    try:
        verify_outputs(hub, outputs)
        return True
    except VerificationFailure:
        return False


def upload_shard_stats(hub: HubClient, shard: dict) -> None:
    """Best-effort per-shard stats upload (small JSON, deterministic path)."""
    stats = {}
    if shard.get("stats_json"):
        try:
            stats = json.loads(shard["stats_json"])
        except json.JSONDecodeError:
            stats = {}
    payload = {
        "shard_id": shard["shard_id"],
        "source_dataset": shard["source_dataset"],
        "seq_index": shard["seq_index"],
        "records_in": shard.get("records_in"),
        "records_out": shard.get("records_out"),
        "token_count": shard.get("token_count"),
        "pipeline_version": shard.get("pipeline_version"),
        "config_hash": shard.get("config_hash"),
        "stats": stats,
    }
    try:
        hub.upload_json(
            payload,
            f"stats/shards/{shard['shard_id']}.json",
            message=f"ucc: stats for {shard['shard_id']}",
        )
    except Exception as exc:  # noqa: BLE001 - stats are best-effort
        log.warning("stats upload for %s failed (non-fatal): %s", shard["shard_id"], exc)
