"""Per-shard stage runner.

Stage order (mirrors the spec):
    validate raw -> normalize -> provenance -> dedup_exact -> dedup_near
    -> repo_reconstruct -> license_filter -> secrets_scan -> quality_filter
    -> complexity -> classify -> finalize/output

Two execution modes (processing.upload_per_batch):

SHARD MODE (false) — the whole shard flows through the stages once; after
each stage listed in processing.checkpoint_after the full working set (rows +
excluded report + a stats snapshot) is checkpointed so a crash resumes from
the last completed stage.

PER-BATCH MODE (true) — records are processed in deterministic batches of
processing.batch_size; as soon as a batch completes ALL stages, its parquet
outputs are UPLOADED TO THE HUB AND VERIFIED immediately, recorded in
work_dir/batch_progress.json, and only then does the next batch start. A
crash resumes at the first batch not yet uploaded (checkpoint_after is
ignored — the batch itself is the checkpoint unit). The raw shard is still
deleted only after the whole shard is verified.

Every stage with global side effects is idempotent (batch-scoped
self-recognition, exactly-once repo deltas), so re-running after a crash is
always safe in both modes.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ucc.crash import maybe_crash
from ucc.hashing import sha256_file
from ucc.io_utils import atomic_write_json, ensure_dir
from ucc.logging_utils import get_logger
from ucc.processing.base import ShardContext
from ucc.processing.classify import ClassifyStage
from ucc.processing.complexity import ComplexityStage
from ucc.processing.dedup_exact import ExactDedupStage
from ucc.processing.dedup_near import NearDedupStage
from ucc.processing.finalize import (
    FinalizeOutputStage,
    write_processed_manifest,
    write_subset_outputs,
)
from ucc.processing.license_filter import LicenseFilterStage
from ucc.processing.normalize import NormalizeStage, iter_normalized_batches
from ucc.processing.provenance import ProvenanceStage
from ucc.processing.quality_filter import QualityFilterStage
from ucc.processing.repo_reconstruct import RepoReconstructStage
from ucc.processing.secrets_scan import SecretsScanStage
from ucc.schema import EXCLUDED_SCHEMA, load_parquet_rows, write_rows_parquet
from ucc.uploader import upload_outputs, verify_outputs

log = get_logger("runner")

_STAGE_CLASSES = [
    NormalizeStage,
    ProvenanceStage,
    ExactDedupStage,
    NearDedupStage,
    RepoReconstructStage,
    LicenseFilterStage,
    SecretsScanStage,
    QualityFilterStage,
    ComplexityStage,
    ClassifyStage,
    FinalizeOutputStage,
]
STAGE_NAMES = [cls.name for cls in _STAGE_CLASSES]

# Per-batch mode runs everything between normalize (the batch iterator) and
# finalize (the per-batch output writer).
_BATCH_STAGE_CLASSES = _STAGE_CLASSES[1:-1]


class RawValidationError(Exception):
    pass


def validate_raw(ctx: ShardContext) -> None:
    """'Validate Download' step: the raw shard's files must exist and their
    combined checksum must match what was recorded at download time."""
    recorded = ctx.shard.get("raw_checksum")
    file_list_path = ctx.raw_dir / ".ucc_raw_files.json"
    if not ctx.raw_dir.exists() or not file_list_path.exists():
        raise RawValidationError(f"raw shard dir incomplete: {ctx.raw_dir}")
    file_hashes = {}
    listed = json.loads(file_list_path.read_text())
    for rel in listed["files"]:
        path = ctx.raw_dir / rel
        if not path.exists():
            raise RawValidationError(f"raw file missing: {rel}")
        file_hashes[rel] = sha256_file(path)
    from ucc.hashing import combined_raw_checksum

    actual = combined_raw_checksum(file_hashes)
    if recorded and actual != recorded:
        raise RawValidationError(
            f"raw checksum mismatch for {ctx.shard['shard_id']}"
        )


def _run_stage(stage, rows: list[dict], ctx: ShardContext) -> list[dict]:
    started = time.monotonic()
    rows = stage.run(rows, ctx)
    ctx.stats[f"timing.{stage.name}_s"] = round(
        ctx.stats.get(f"timing.{stage.name}_s", 0) + time.monotonic() - started, 2
    )
    return rows


# ---------------------------------------------------------------- shard mode
def _checkpoint_paths(work_dir: Path, stage_name: str) -> tuple[Path, Path, Path]:
    return (
        work_dir / f"after-{stage_name}.parquet",
        work_dir / f"excluded-after-{stage_name}.parquet",
        work_dir / f"after-{stage_name}.stats.json",
    )


def _write_checkpoint(ctx: ShardContext, stage_name: str, rows: list[dict]) -> None:
    ensure_dir(ctx.work_dir)
    rows_path, excl_path, stats_path = _checkpoint_paths(ctx.work_dir, stage_name)
    tmp = str(rows_path) + ".tmp"
    write_rows_parquet(iter(rows), tmp)
    os.replace(tmp, rows_path)
    tmp = str(excl_path) + ".tmp"
    write_rows_parquet(iter(ctx.excluded), tmp, schema=EXCLUDED_SCHEMA)
    os.replace(tmp, excl_path)
    # Stats snapshot taken AT the checkpoint: resuming restores exactly this
    # state, so accumulating counters never double-count re-run stages.
    atomic_write_json(stats_path, ctx.stats)
    ctx.manifest.set_fields(
        ctx.shard["shard_id"],
        stage_checkpoint=stage_name,
        checkpoint_path=str(rows_path),
    )
    ctx.manifest.update_shard_stats(ctx.shard["shard_id"], ctx.stats)


def _try_resume(ctx: ShardContext) -> tuple[int, list[dict]]:
    """Return (start_stage_index, rows) — resuming from the last completed
    checkpointed stage when its files are intact, else from the beginning."""
    stage_name = ctx.shard.get("stage_checkpoint")
    if not stage_name or stage_name not in STAGE_NAMES:
        return 0, []
    rows_path, excl_path, stats_path = _checkpoint_paths(ctx.work_dir, stage_name)
    if not rows_path.exists():
        ctx.log.warning(
            "checkpoint after '%s' recorded but file missing — restarting stages "
            "(global side effects are idempotent, so this is safe)", stage_name
        )
        return 0, []
    try:
        rows = load_parquet_rows(str(rows_path))
        ctx.excluded = (
            load_parquet_rows(str(excl_path)) if excl_path.exists() else []
        )
        if stats_path.exists():
            ctx.stats = json.loads(stats_path.read_text())
    except Exception as exc:  # noqa: BLE001 - corrupt checkpoint
        ctx.log.warning("checkpoint unreadable (%s) — restarting stages", exc)
        ctx.stats = {}
        ctx.excluded = []
        return 0, []
    ctx.log.info(
        "resuming after stage '%s' with %d rows (%d excluded so far)",
        stage_name, len(rows), len(ctx.excluded),
    )
    return STAGE_NAMES.index(stage_name) + 1, rows


def _run_shard_mode(ctx: ShardContext) -> ShardContext:
    checkpoint_after = set(ctx.cfg.processing.checkpoint_after or [])
    start_index, rows = _try_resume(ctx)

    for stage_cls in _STAGE_CLASSES[start_index:]:
        stage = stage_cls()
        rows = _run_stage(stage, rows, ctx)
        ctx.manifest.update_shard_stats(ctx.shard["shard_id"], ctx.stats)
        if stage.name in checkpoint_after and stage.name != "finalize":
            _write_checkpoint(ctx, stage.name, rows)
        maybe_crash(f"after_stage:{stage.name}")

    return ctx


# ------------------------------------------------------------ per-batch mode
def _run_batched_mode(ctx: ShardContext) -> ShardContext:
    """Process in batches of processing.batch_size; upload + verify each
    batch's outputs on the Hub the moment its processing completes."""
    if ctx.hub is None:
        raise RuntimeError(
            "processing.upload_per_batch requires a hub client on the context"
        )
    shard_id = ctx.shard["shard_id"]
    seq = int(ctx.shard["seq_index"])
    batch_size = int(ctx.cfg.processing.batch_size)
    ensure_dir(ctx.work_dir)
    ensure_dir(ctx.processed_dir)

    progress_path = ctx.work_dir / "batch_progress.json"
    done: dict[str, dict] = {}
    if progress_path.exists():
        try:
            done = json.loads(progress_path.read_text()).get("batches", {})
        except (json.JSONDecodeError, OSError):
            done = {}
    if done:
        # Stats were last persisted right after the newest completed batch,
        # so restoring them keeps the accumulating counters exact.
        if ctx.shard.get("stats_json"):
            try:
                ctx.stats = json.loads(ctx.shard["stats_json"])
            except json.JSONDecodeError:
                pass
        ctx.log.info(
            "per-batch resume: %d batches already uploaded & verified — "
            "skipping straight past them", len(done),
        )

    for batch_idx, counts, rows in iter_normalized_batches(
        ctx, batch_size, done_batches=done
    ):
        key = str(batch_idx)
        if key in done:  # non-contiguous holes only; the prefix is fast-skipped
            continue

        ctx.excluded = []
        ctx.scratch["dedup_scope"] = f"{shard_id}#b{batch_idx}"
        ctx.scratch["repo_delta_key"] = f"{shard_id}#b{batch_idx}"
        ctx.bump("records_in", counts["raw"])
        ctx.bump("normalize.unmappable", counts["unmappable"])
        ctx.bump("normalize.intra_shard_id_collision", counts["id_collisions"])
        ctx.bump("normalize.records_out", len(rows))
        ctx.log.info(
            "batch %d: %d records entering stages (%d raw read)",
            batch_idx, len(rows), counts["raw"],
        )

        for stage_cls in _BATCH_STAGE_CLASSES:
            rows = _run_stage(stage_cls(), rows, ctx)
            maybe_crash(f"after_stage:{stage_cls.name}")

        part_name = f"part-{seq:06d}-b{batch_idx:04d}.parquet"
        outputs = write_subset_outputs(ctx, rows, part_name)

        # The whole processed batch goes to the Hub right now.
        upload_outputs(ctx.hub, outputs, f"{shard_id} batch {batch_idx}")
        verify_outputs(ctx.hub, outputs)

        done[key] = {
            "files": outputs,
            "records_in": counts["raw"],
            "records_out": len(rows),
            "token_count": sum(r.get("token_count") or 0 for r in rows),
        }
        atomic_write_json(progress_path, {"batches": done})
        ctx.manifest.update_shard_stats(shard_id, ctx.stats)
        ctx.log.info(
            "batch %d uploaded & verified (%d files, %d records) — "
            "%d batches done for this shard",
            batch_idx, len(outputs), len(rows), len(done),
        )
        maybe_crash("after_batch_upload")

    all_outputs = [
        out
        for key in sorted(done, key=int)
        for out in done[key]["files"]
    ]
    records_out = sum(b["records_out"] for b in done.values())
    token_count = sum(b["token_count"] for b in done.values())
    write_processed_manifest(ctx, all_outputs, records_out=records_out,
                             token_count=token_count)
    ctx.manifest.update_shard_stats(shard_id, ctx.stats)
    ctx.log.info(
        "per-batch shard complete: %d batches, %d records, %d output files "
        "(all already on the hub)",
        len(done), records_out, len(all_outputs),
    )
    return ctx


def run_shard_pipeline(ctx: ShardContext) -> ShardContext:
    """Run (or resume) the full stage pipeline for one downloaded shard."""
    validate_raw(ctx)
    ensure_dir(ctx.work_dir)
    ensure_dir(ctx.processed_dir)
    if ctx.cfg.path("processing.upload_per_batch", False):
        return _run_batched_mode(ctx)
    return _run_shard_mode(ctx)
