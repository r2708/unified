"""Stage 1 — normalize: raw source records -> unified schema records.

Two consumers share the normalization logic:
- NormalizeStage (shard mode): normalizes the whole raw shard at once.
- iter_normalized_batches (per-batch upload mode): yields deterministic,
  sorted batches of `batch_size` normalized records so each batch can run
  the remaining stages and be uploaded as soon as it completes.
"""

from __future__ import annotations

from typing import Iterator

from ucc.hashing import make_record_id, sha256_text
from ucc.processing.base import ShardContext, Stage
from ucc.tokens import count_tokens


class RecordNormalizer:
    """Stateless-per-record normalization; counts nothing (callers own the
    bookkeeping so resumed re-reads never double-count stats)."""

    def __init__(self, ctx: ShardContext):
        self.adapter = ctx.adapter
        self.spec_ref = ctx.spec_ref
        self.token_mode = ctx.cfg.processing.token_counter
        self.source = ctx.shard["source_dataset"]
        self.shard_id = ctx.shard["shard_id"]
        self.pipeline_version = ctx.cfg.pipeline_version
        self.config_hash = ctx.cfg.config_hash

    def normalize(self, raw: dict) -> dict | None:
        rec = self.adapter.normalize_record(raw, self.spec_ref)
        if rec is None:
            return None
        content = rec["content"]
        rec["content_sha256"] = sha256_text(content)
        rec["size_bytes"] = len(content.encode("utf-8", errors="replace"))
        rec["token_count"] = count_tokens(content, self.token_mode)
        rec["id"] = make_record_id(
            rec.get("repo_name"), rec.get("path"), rec["content_sha256"]
        )
        rec["source_dataset"] = self.source
        rec["source_datasets"] = [self.source]
        rec["source_shard"] = self.shard_id
        rec["pipeline_version"] = self.pipeline_version
        rec["config_hash"] = self.config_hash
        return rec


class NormalizeStage(Stage):
    name = "normalize"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        del rows  # normalize reads from the raw shard, not from prior rows
        batch_size = min(int(ctx.cfg.processing.batch_size), 8192)
        norm = RecordNormalizer(ctx)

        out: list[dict] = []
        seen_ids: set[str] = set()
        raw_count = 0
        # Total record count is unknown until the raw shard is fully read, so
        # this stage reports a live running count + rate instead of a %.
        live = ctx.progress("normalize (reading raw shard)")
        for batch in ctx.adapter.iter_raw_batches(ctx.spec_ref, ctx.raw_dir, batch_size):
            for raw in batch:
                raw_count += 1
                live.update()
                rec = norm.normalize(raw)
                if rec is None:
                    ctx.bump("normalize.unmappable")
                    continue
                # Identical (repo, path, content) inside one shard collapses
                # here so downstream ids stay unique.
                if rec["id"] in seen_ids:
                    ctx.bump("normalize.intra_shard_id_collision")
                    continue
                seen_ids.add(rec["id"])
                out.append(rec)

        live.close()
        ctx.bump("records_in", raw_count)
        ctx.bump("normalize.records_out", len(out))
        # Deterministic processing order — canonical-copy selection in the
        # dedup stages depends on it.
        out.sort(key=lambda r: r["id"])
        ctx.log.info("normalize: %d raw -> %d unified records", raw_count, len(out))
        return out


def done_prefix(done_batches: dict) -> tuple[int, int]:
    """(next_batch_index, raw_records_to_skip) for the CONTIGUOUS prefix of
    already-completed batches: batch 0..k-1 done -> resume at batch k after
    fast-skipping the sum of their recorded raw record counts."""
    k = 0
    skip_raw = 0
    while str(k) in done_batches:
        skip_raw += int(done_batches[str(k)].get("records_in") or 0)
        k += 1
    return k, skip_raw


def iter_normalized_batches(
    ctx: ShardContext, batch_size: int, done_batches: dict | None = None
) -> Iterator[tuple[int, dict, list[dict]]]:
    """Yield (batch_index, counts, rows) with len(rows) <= batch_size.

    Batch boundaries follow the deterministic raw read order (same shard +
    same batch_size => identical batches), rows are sorted by id within each
    batch, and `counts` carries the batch's raw/unmappable/collision numbers
    WITHOUT touching ctx.stats — the caller adds them only for batches it
    actually processes, so resuming past already-uploaded batches never
    double-counts.

    Resume goes DIRECTLY to the last checkpoint, never back to the start:
    the contiguous prefix of `done_batches` is fast-skipped at the reader
    level (parquet row-group skips / line counting — no re-parsing, no
    re-normalization) and iteration begins at the first incomplete batch.
    """
    done_batches = done_batches or {}
    start_batch, skip_raw = done_prefix(done_batches)
    if len(done_batches) != start_batch:
        # Non-contiguous completion history (should not happen): no fast
        # skip; the caller's per-batch done-guard still skips the holes.
        ctx.log.warning(
            "batch progress is non-contiguous — resuming without reader-level fast-skip"
        )
        start_batch, skip_raw = 0, 0
    if skip_raw:
        ctx.log.info(
            "fast-skip: jumping directly past %d completed batches "
            "(%s raw records skipped at the reader level — no re-normalization); "
            "resuming at batch %d",
            start_batch, f"{skip_raw:,}", start_batch,
        )

    norm = RecordNormalizer(ctx)
    read_size = min(batch_size, 8192)
    live = ctx.progress("normalize (reading raw shard)")

    buf: list[dict] = []
    seen_ids: set[str] = set()
    counts = {"raw": 0, "unmappable": 0, "id_collisions": 0}
    batch_idx = start_batch

    def flush():
        buf.sort(key=lambda r: r["id"])
        return buf, dict(counts)

    for batch in ctx.adapter.iter_raw_batches(
        ctx.spec_ref, ctx.raw_dir, read_size, skip_records=skip_raw
    ):
        for raw in batch:
            counts["raw"] += 1
            live.update()
            rec = norm.normalize(raw)
            if rec is None:
                counts["unmappable"] += 1
                continue
            if rec["id"] in seen_ids:
                # Within-batch collapse; cross-batch duplicates are caught by
                # the global exact-dedup index.
                counts["id_collisions"] += 1
                continue
            seen_ids.add(rec["id"])
            buf.append(rec)
            if len(buf) >= batch_size:
                rows, batch_counts = flush()
                yield batch_idx, batch_counts, rows
                buf, seen_ids = [], set()
                counts = {"raw": 0, "unmappable": 0, "id_collisions": 0}
                batch_idx += 1

    if buf or counts["raw"]:
        rows, batch_counts = flush()
        yield batch_idx, batch_counts, rows
    live.close()
