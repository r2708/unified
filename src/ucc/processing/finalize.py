"""Stage 10 — finalize: write the processed shard outputs.

Deterministic output layout (idempotent re-runs produce identical paths):

    data/full/{source}/part-{seq:06d}.parquet              shard mode
    data/full/{source}/part-{seq:06d}-b{batch:04d}.parquet per-batch mode
    data/commits|issues|high_quality/{source}/...          same pattern
    excluded/{source}/...                                  audit report

The record_type subsets are disjoint (no duplication); high_quality is an
extra filtered copy (subsets.materialize_high_quality). The writer helpers
are shared by the shard-mode FinalizeOutputStage and the per-batch runner
(processing.upload_per_batch), which uploads each batch's files as soon as
that batch finishes processing.
"""

from __future__ import annotations

import json
import os

from ucc import constants as C
from ucc.hashing import combined_raw_checksum, sha256_file
from ucc.io_utils import atomic_write_json, ensure_dir
from ucc.processing.base import ShardContext, Stage
from ucc.schema import EXCLUDED_SCHEMA, UNIFIED_SCHEMA, write_rows_parquet

# Flags that do NOT disqualify a record from the high_quality subset.
_HQ_BENIGN_FLAGS = {"license_source_level", "lockfile"}


def _is_high_quality(rec: dict, min_score: float | None) -> bool:
    if rec.get("license_status") != C.LIC_PERMISSIVE:
        return False
    if rec.get("is_near_duplicate"):
        return False
    if set(rec.get("quality_flags") or []) - _HQ_BENIGN_FLAGS:
        return False
    if min_score is not None and rec.get("quality_score") is not None:
        if float(rec["quality_score"]) < float(min_score):
            return False
    return True


def write_subset_outputs(ctx: ShardContext, rows: list[dict], part_name: str) -> list[dict]:
    """Write the parquet outputs for one set of processed rows (a whole shard
    or one batch) under deterministic dest paths, plus the excluded-records
    audit file for whatever ctx.excluded currently holds. Returns the output
    descriptors: {local, dest, sha256, size, records, subset}."""
    source = ctx.shard["source_dataset"]
    compression = ctx.cfg.processing.parquet_compression
    row_group_size = int(ctx.cfg.processing.parquet_row_group_size)

    by_subset: dict[str, list[dict]] = {}
    for rec in rows:
        subset = C.RECORD_TYPE_TO_SUBSET.get(rec["record_type"], C.SUBSET_FULL)
        by_subset.setdefault(subset, []).append(rec)

    if ctx.cfg.path("subsets.materialize_high_quality", True):
        min_score = ctx.cfg.path("subsets.high_quality_min_score")
        hq = [r for r in rows if _is_high_quality(r, min_score)]
        if hq:
            by_subset[C.SUBSET_HIGH_QUALITY] = hq

    outputs: list[dict] = []

    def _write(dest: str, out_rows: list[dict], schema, subset: str) -> None:
        local = ctx.processed_dir / dest
        ensure_dir(local.parent)
        tmp = str(local) + ".tmp"
        records = write_rows_parquet(
            iter(out_rows), tmp, schema=schema,
            compression=compression, row_group_size=row_group_size,
        )
        os.replace(tmp, local)
        outputs.append(
            {
                "local": str(local),
                "dest": dest,
                "sha256": sha256_file(local),
                "size": local.stat().st_size,
                "records": records,
                "subset": subset,
            }
        )

    for subset in sorted(by_subset):
        _write(f"data/{subset}/{source}/{part_name}", by_subset[subset],
               UNIFIED_SCHEMA, subset)

    if ctx.excluded and ctx.cfg.path("hf.upload_excluded_reports", True):
        _write(f"excluded/{source}/{part_name}", ctx.excluded,
               EXCLUDED_SCHEMA, C.SUBSET_EXCLUDED)

    return outputs


def write_processed_manifest(
    ctx: ShardContext, outputs: list[dict], records_out: int, token_count: int
) -> None:
    """Persist the authoritative list of this shard's output files
    (processed_files.json) and the shard's manifest fields. Derived per-subset
    stats are ASSIGNED from the output list (idempotent on re-runs)."""
    blob = {
        "shard_id": ctx.shard["shard_id"],
        "files": outputs,
        "pipeline_version": ctx.cfg.pipeline_version,
        "config_hash": ctx.cfg.config_hash,
    }
    atomic_write_json(ctx.processed_dir / "processed_files.json", blob)

    subset_records: dict[str, int] = {}
    for out in outputs:
        subset_records[out["subset"]] = subset_records.get(out["subset"], 0) + out["records"]
    for subset, records in subset_records.items():
        ctx.stats[f"output.{subset}.records"] = records
    ctx.stats["records_out"] = records_out
    ctx.stats["token_count"] = token_count

    ctx.manifest.set_fields(
        ctx.shard["shard_id"],
        records_out=records_out,
        token_count=token_count,
        processed_size_bytes=sum(o["size"] for o in outputs),
        processed_checksum=combined_raw_checksum(
            {o["dest"]: o["sha256"] for o in outputs}
        ),
        hf_dest_paths=json.dumps([o["dest"] for o in outputs]),
        local_processed_dir=str(ctx.processed_dir),
    )


class FinalizeOutputStage(Stage):
    name = "finalize"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        part_name = f"part-{int(ctx.shard['seq_index']):06d}.parquet"
        outputs = write_subset_outputs(ctx, rows, part_name)
        token_count = sum(r.get("token_count") or 0 for r in rows)
        write_processed_manifest(ctx, outputs, records_out=len(rows),
                                 token_count=token_count)
        ctx.log.info(
            "finalize: wrote %d output files (%d records kept, %d excluded)",
            len(outputs), len(rows), len(ctx.excluded),
        )
        return rows
