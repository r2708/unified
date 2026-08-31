"""Stage 3 — exact deduplication (SHA-256), within-shard and global.

When identical content appears in multiple datasets, one canonical copy is
retained and every source dataset is preserved: same-shard duplicates merge
their sources onto the surviving record directly; cross-shard duplicates are
merged into the manifest's exact_hashes provenance table, which `finalize`
exports as provenance/sources.parquet (an already-uploaded canonical record
cannot be rewritten in place — the provenance table is the durable merge).
"""

from __future__ import annotations

from ucc.processing.base import ShardContext, Stage


class ExactDedupStage(Stage):
    name = "dedup_exact"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        if not ctx.cfg.path("dedup.exact.enabled", True):
            return rows

        # In per-batch mode the scope is "<shard>#b<n>": the self-recognition
        # guard (a crashed attempt re-seeing its own insertions) must be
        # batch-local so a duplicate in a LATER batch of the same shard is
        # still caught as a duplicate.
        scope = ctx.scratch.get("dedup_scope") or ctx.shard["shard_id"]
        source = ctx.shard["source_dataset"]

        # ---- within-shard (rows are sorted by id -> deterministic canonical)
        first_by_hash: dict[str, dict] = {}
        survivors: list[dict] = []
        for rec in rows:
            h = rec["content_sha256"]
            canonical = first_by_hash.get(h)
            if canonical is None:
                first_by_hash[h] = rec
                survivors.append(rec)
            else:
                merged = set(canonical["source_datasets"]) | set(rec["source_datasets"])
                canonical["source_datasets"] = sorted(merged)
                ctx.exclude(rec, "exact_duplicate_intra", detail=canonical["id"])

        # ---- global (cross-shard, cross-source) via the manifest index
        out: list[dict] = []
        batch_size = int(ctx.cfg.processing.batch_size)
        live = ctx.progress("dedup_exact (global index)", total=len(survivors))
        for start in range(0, len(survivors), batch_size):
            batch = survivors[start : start + batch_size]
            live.update(len(batch))
            items = [
                (rec["content_sha256"], rec["id"], scope, source) for rec in batch
            ]
            verdicts = ctx.manifest.exact_seen_or_add_many(items)
            for rec in batch:
                is_new, canonical_id = verdicts[rec["content_sha256"]]
                if is_new:
                    out.append(rec)
                else:
                    ctx.exclude(rec, "exact_duplicate_global", detail=canonical_id)

        live.close()
        removed = len(rows) - len(out)
        ctx.bump("dedup_exact.removed", removed)
        ctx.bump("dedup_exact.records_out", len(out))
        ctx.log.info("dedup_exact: %d -> %d (-%d duplicates)", len(rows), len(out), removed)
        return out
