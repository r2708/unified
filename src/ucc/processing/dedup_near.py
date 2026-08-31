"""Stage 4 — near deduplication (MinHash + LSH), within-shard and global.

Catches forks, copied repositories, mirrors, generated boilerplate and
lightly-edited copies that exact hashing misses.

Design:
- MinHash over 5-gram word shingles, `num_perm` permutations, banded LSH.
- The LSH band index and full signatures persist in the manifest DB, so
  near-dedup works ACROSS shards and survives restarts.
- All DB writes are INSERT OR IGNORE keyed by deterministic record ids, and
  candidates from the record's own shard are ignored (they are resolved
  in-memory each run), so re-running the stage after a crash is idempotent.
- Default mode is `annotate`: near-duplicates stay in the `full` subset with
  is_near_duplicate=true + near_dup_cluster (duplicates are not hidden) and
  are excluded from high_quality. `drop` mode removes them entirely.
- Commits and issues are historical records and are exempt — meaningful
  historical versions are never removed.
"""

from __future__ import annotations

import re

import numpy as np

from ucc.processing.base import ShardContext, Stage

_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")


def _minhash_signature(text: str, num_perm: int, shingle_size: int) -> np.ndarray | None:
    from datasketch import MinHash
    import xxhash

    tokens = _TOKEN_RX.findall(text.lower())
    if len(tokens) < shingle_size:
        return None
    shingles = {
        " ".join(tokens[i : i + shingle_size])
        for i in range(len(tokens) - shingle_size + 1)
    }
    if not shingles:
        return None
    mh = MinHash(num_perm=num_perm, hashfunc=lambda b: xxhash.xxh32_intdigest(b))
    mh.update_batch([s.encode("utf-8") for s in shingles])
    return mh.hashvalues.astype("<u4")


def _band_hashes(sig: np.ndarray, bands: int) -> list[bytes]:
    import xxhash

    rows = len(sig) // bands
    raw = sig.tobytes()
    row_bytes = rows * 4
    return [
        xxhash.xxh64(raw[band * row_bytes : (band + 1) * row_bytes]).digest()
        for band in range(bands)
    ]


def _jaccard_est(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        return 0.0
    return float(np.count_nonzero(a == b)) / len(a)


class NearDedupStage(Stage):
    name = "dedup_near"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        ncfg = ctx.cfg.dedup.near
        if not ncfg.get("enabled", True):
            return rows

        num_perm = int(ncfg["num_perm"])
        bands = int(ncfg["bands"])
        if num_perm % bands != 0:
            raise ValueError("dedup.near.num_perm must be divisible by bands")
        threshold = float(ncfg["jaccard_threshold"])
        shingle_size = int(ncfg["shingle_size"])
        min_tokens = int(ncfg["min_tokens"])
        exempt = set(ncfg.get("exempt_record_types") or [])
        mode = ncfg.get("mode", "annotate")
        # Batch-local scope in per-batch mode (see dedup_exact): only the
        # current batch's own crashed-attempt rows are excluded from the
        # candidate index; earlier batches of the same shard stay visible.
        scope = ctx.scratch.get("dedup_scope") or ctx.shard["shard_id"]

        # In-memory structures for this shard (also make crash re-runs
        # self-consistent: own-shard DB rows from a crashed attempt are
        # excluded from candidate queries).
        local_bands: dict[tuple[int, bytes], list[str]] = {}
        local_sigs: dict[str, np.ndarray] = {}

        pending_sig_rows: list[tuple[str, str, bytes]] = []
        pending_band_rows: list[tuple[int, bytes, str, str]] = []

        def flush_pending() -> None:
            if pending_sig_rows or pending_band_rows:
                ctx.manifest.add_minhash_batch(pending_sig_rows, pending_band_rows)
                pending_sig_rows.clear()
                pending_band_rows.clear()

        out: list[dict] = []
        annotated = 0
        dropped = 0
        skipped = 0

        live = ctx.progress("dedup_near (MinHash/LSH)", total=len(rows))
        for rec in rows:  # rows sorted by id -> deterministic canonicals
            live.update()
            if rec["record_type"] in exempt or rec["token_count"] < min_tokens:
                skipped += 1
                out.append(rec)
                continue

            sig = _minhash_signature(rec["content"], num_perm, shingle_size)
            if sig is None:
                skipped += 1
                out.append(rec)
                continue
            band_list = _band_hashes(sig, bands)

            # Gather candidates: same-shard from memory, cross-shard from DB.
            candidate_ids: set[str] = set()
            for band_idx, band_hash in enumerate(band_list):
                candidate_ids.update(local_bands.get((band_idx, band_hash), ()))
                candidate_ids.update(
                    ctx.manifest.band_candidates(band_idx, band_hash, scope)
                )
            candidate_ids.discard(rec["id"])

            canonical_id: str | None = None
            if candidate_ids:
                need_db = [cid for cid in candidate_ids if cid not in local_sigs]
                db_sigs = ctx.manifest.get_sigs(sorted(need_db)) if need_db else {}
                passing: list[str] = []
                for cid in sorted(candidate_ids):
                    other = local_sigs.get(cid)
                    if other is None:
                        blob = db_sigs.get(cid)
                        if blob is None:
                            continue
                        other = np.frombuffer(blob, dtype="<u4")
                    if _jaccard_est(sig, other) >= threshold:
                        passing.append(cid)
                if passing:
                    canonical_id = min(passing)

            if canonical_id is not None:
                if mode == "drop":
                    ctx.exclude(rec, "near_duplicate", detail=canonical_id)
                    dropped += 1
                else:
                    rec["is_near_duplicate"] = True
                    rec["near_dup_cluster"] = canonical_id
                    annotated += 1
                    out.append(rec)
                continue

            # Unique: index it (memory now, DB in batched idempotent writes).
            local_sigs[rec["id"]] = sig
            sig_bytes = sig.tobytes()
            pending_sig_rows.append((rec["id"], scope, sig_bytes))
            for band_idx, band_hash in enumerate(band_list):
                local_bands.setdefault((band_idx, band_hash), []).append(rec["id"])
                pending_band_rows.append((band_idx, band_hash, rec["id"], scope))
            if len(pending_band_rows) >= 20_000:
                flush_pending()
            out.append(rec)

        flush_pending()
        live.close()
        ctx.bump("dedup_near.annotated", annotated)
        ctx.bump("dedup_near.dropped", dropped)
        ctx.bump("dedup_near.skipped_short_or_exempt", skipped)
        ctx.bump("dedup_near.records_out", len(out))
        ctx.log.info(
            "dedup_near: %d records, %d annotated, %d dropped", len(rows), annotated, dropped
        )
        return out
