"""Startup reconciliation: resume only incomplete work, never redo verified work.

Implements the spec's resume rules:

    Load persistent manifest
      -> check Hugging Face uploaded files
      -> validate local files
      -> determine the latest true state of every shard
      -> resume only incomplete work

- uploaded & verified            -> skip entirely (completed)
- processed but upload failed    -> retry upload, never re-download
- downloaded but not processed   -> resume processing from the raw shard
- crashed mid-stage              -> resume from the last stage checkpoint
- crashed during upload          -> check the Hub, validate checksums,
                                    mark verified if valid, retry safely if not
- raw file missing               -> check Hub + checkpoints first; re-download
                                    only if no verified output exists
"""

from __future__ import annotations

import json
from pathlib import Path

from ucc.io_utils import safe_rmtree
from ucc.logging_utils import get_logger
from ucc.states import RAW_ON_DISK_STATES, ShardState
from ucc.uploader import load_processed_outputs, verify_outputs_remote_only

log = get_logger("resume")


def _raw_intact(raw_dir: Path) -> bool:
    listing = raw_dir / ".ucc_raw_files.json"
    if not raw_dir.exists() or not listing.exists():
        return False
    try:
        listed = json.loads(listing.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    for rel, size in zip(listed["files"], listed.get("sizes", [])):
        path = raw_dir / rel
        if not path.exists() or (size and path.stat().st_size != size):
            return False
    return True


def _processed_intact(processed_dir: Path) -> bool:
    try:
        outputs = load_processed_outputs(processed_dir)
    except Exception:  # noqa: BLE001
        return False
    for out in outputs:
        path = Path(out["local"])
        if not path.exists() or path.stat().st_size != out["size"]:
            return False
    return True


def reconcile(manifest, hub, cfg, paths) -> dict:
    """Normalize every shard to its true resumable state. Returns a summary
    including how many raw shards remain on disk (to prime the capacity
    gauge)."""
    manifest.clear_all_claims()
    summary = {"checked": 0, "reset_to_pending": 0, "resumed_processing": 0,
               "retry_upload": 0, "marked_verified": 0, "cleaned": 0}

    for shard in manifest.all_shards():
        summary["checked"] += 1
        state = ShardState(shard["state"])
        shard_id = shard["shard_id"]
        raw_dir = paths.raw / shard_id
        work_dir = paths.work / shard_id
        processed_dir = paths.processed / shard_id

        def to_pending() -> None:
            safe_rmtree(raw_dir, paths.workspace)
            safe_rmtree(work_dir, paths.workspace)
            safe_rmtree(processed_dir, paths.workspace)
            manifest.set_fields(
                shard_id, stage_checkpoint=None, checkpoint_path=None,
                local_raw_dir=None, raw_checksum=None,
            )
            manifest.transition(shard_id, [state], ShardState.PENDING)
            summary["reset_to_pending"] += 1

        if state in (ShardState.COMPLETED, ShardState.SKIPPED):
            # Terminal: make sure leftovers are gone (idempotent cleanup).
            for leftover in (raw_dir, work_dir, processed_dir):
                if leftover.exists():
                    safe_rmtree(leftover, paths.workspace)
                    summary["cleaned"] += 1
            continue

        if state == ShardState.PENDING:
            if raw_dir.exists():
                safe_rmtree(raw_dir, paths.workspace)
            continue

        if state == ShardState.DOWNLOADING:
            # Crashed mid-download: partial data is untrustworthy.
            log.info("%s: crashed while downloading — re-queueing", shard_id)
            to_pending()
            continue

        if state in (ShardState.DOWNLOADED, ShardState.PROCESSING):
            if _raw_intact(raw_dir):
                if state == ShardState.PROCESSING:
                    log.info(
                        "%s: crashed mid-processing — will resume from stage "
                        "checkpoint '%s'", shard_id, shard.get("stage_checkpoint"),
                    )
                    summary["resumed_processing"] += 1
                continue
            log.warning("%s: raw shard missing/incomplete — re-downloading", shard_id)
            to_pending()
            continue

        if state in (
            ShardState.PROCESSED,
            ShardState.UPLOADING,
            ShardState.UPLOADED,
            ShardState.UPLOAD_FAILED,
            ShardState.VERIFICATION_FAILED,
        ):
            if _processed_intact(processed_dir):
                # Retry the upload path; already-uploaded files are skipped by
                # checksum, so this is safe and cheap.
                manifest.transition(shard_id, [state], ShardState.PROCESSED)
                summary["retry_upload"] += 1
                continue
            # Local processed output lost. Check the Hub against RECORDED
            # checksums before assuming anything.
            outputs = None
            try:
                outputs = load_processed_outputs(processed_dir)
            except Exception:  # noqa: BLE001
                outputs = None
            if outputs and verify_outputs_remote_only(hub, outputs):
                log.info("%s: all outputs already verified on the Hub", shard_id)
                manifest.transition(shard_id, [state], ShardState.VERIFIED)
                summary["marked_verified"] += 1
                continue
            if _raw_intact(raw_dir):
                log.info("%s: processed output lost, raw intact — reprocessing", shard_id)
                manifest.transition(shard_id, [state], ShardState.DOWNLOADED)
                continue
            log.warning(
                "%s: no verified remote output, no local files — re-downloading",
                shard_id,
            )
            to_pending()
            continue

        if state == ShardState.VERIFIED:
            continue  # consumer will finish cleanup -> completed

        if state == ShardState.DOWNLOAD_FAILED:
            safe_rmtree(raw_dir, paths.workspace)
            continue

        if state == ShardState.PROCESSING_FAILED:
            if not _raw_intact(raw_dir):
                to_pending()
            continue

    # Raw shards still on disk occupy queue slots.
    on_disk = 0
    for shard in manifest.shards_in_states(RAW_ON_DISK_STATES):
        if (paths.raw / shard["shard_id"]).exists():
            on_disk += 1
    summary["raw_on_disk"] = on_disk
    log.info("reconcile summary: %s", summary)
    return summary


def check_config_hash(manifest, cfg, allow_change: bool) -> None:
    """Refuse to mix configurations silently: shards already enumerated under
    a different data-affecting config hash abort the run unless explicitly
    allowed."""
    mismatched = sorted(
        {
            s["config_hash"]
            for s in manifest.all_shards()
            if s["config_hash"] and s["config_hash"] != cfg.config_hash
            and ShardState(s["state"]) not in (ShardState.COMPLETED, ShardState.SKIPPED)
        }
    )
    if mismatched and not allow_change:
        raise SystemExit(
            f"config hash changed (manifest has {mismatched}, current is "
            f"{cfg.config_hash}). Data-affecting settings differ from the ones "
            "used for incomplete shards. Re-run with --allow-config-change to "
            "proceed anyway (new shards use the new config; existing shards "
            "keep their recorded hash)."
        )
    if mismatched:
        log.warning("proceeding across config-hash change: %s -> %s",
                    mismatched, cfg.config_hash)
