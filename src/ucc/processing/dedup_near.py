"""Stage 4 — near deduplication (MinHash + LSH), within-shard and global.

Catches forks, copied repositories, mirrors, generated boilerplate and
lightly-edited copies that exact hashing misses.

Design:
- MinHash over 5-gram word shingles, `num_perm` permutations, banded LSH.
- The permutation arrays are seed-deterministic, so they are built once and
  shared across records (datasketch regenerates them per MinHash() otherwise
  — ~20x slower per record for identical signatures).
- Signatures for a chunk of records are computed up front (optionally on the
  processing.cpu_workers process pool) and the chunk's cross-shard LSH
  candidates are fetched from the manifest in batched IN-queries instead of
  one query per record per band. Candidate resolution stays sequential, so
  canonical selection is exactly the id-ordered behavior it always was.
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
from functools import partial

import numpy as np

from ucc.parallel import parallel_map, resolve_cpu_workers
from ucc.processing.base import ShardContext, Stage

_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")

# Records per signature/prefetch chunk. Bounds transient memory
# (chunk sigs + band hashes) to a few MB regardless of batch_size.
_SIG_CHUNK = 4096

# The MinHash permutation scheme is pinned EXPLICITLY (and datasketch is
# pinned to one major in pyproject): signatures persist in the manifest
# across runs, and datasketch changes the default scheme between majors
# (2.0 -> affine32), which would silently mix incomparable signatures in
# one corpus if left implicit.
_SCHEME = "affine32"

# num_perm -> the datasketch permutation arrays (seed-deterministic, so
# signatures are identical to per-record regeneration — and identical to
# whatever an earlier run persisted in the manifest).
_PERMUTATIONS: dict[int, object] = {}


def _permutations_for(num_perm: int):
    perms = _PERMUTATIONS.get(num_perm)
    if perms is None:
        from datasketch import MinHash

        perms = MinHash(num_perm=num_perm, scheme=_SCHEME).permutations
        _PERMUTATIONS[num_perm] = perms
    return perms


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
    mh = MinHash(
        num_perm=num_perm,
        hashfunc=xxhash.xxh32_intdigest,
        permutations=_permutations_for(num_perm),
        scheme=_SCHEME,
    )
    mh.update_batch([s.encode("utf-8") for s in shingles])
    return mh.hashvalues.astype("<u4")


def _signature_bytes(text: str, num_perm: int, shingle_size: int) -> bytes | None:
    """Process-pool worker: pure function of the content, returns the packed
    signature (or None) so only ~num_perm*4 bytes travel back."""
    sig = _minhash_signature(text, num_perm, shingle_size)
    return None if sig is None else sig.tobytes()


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
        cpu_workers = resolve_cpu_workers(ctx.cfg.path("processing.cpu_workers", 1))
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
        sig_fn = partial(_signature_bytes, num_perm=num_perm, shingle_size=shingle_size)

        live = ctx.progress("dedup_near (MinHash/LSH)", total=len(rows))
        for start in range(0, len(rows), _SIG_CHUNK):
            chunk = rows[start : start + _SIG_CHUNK]

            # Phase 1 — signatures for the chunk (parallelizable: pure per-
            # record work; exempt/short records are skipped exactly as before).
            eligible = [
                i for i, rec in enumerate(chunk)
                if rec["record_type"] not in exempt and rec["token_count"] >= min_tokens
            ]
            sig_blobs = parallel_map(
                sig_fn, [chunk[i]["content"] for i in eligible], cpu_workers
            )
            sigs: dict[int, np.ndarray] = {}
            bands_of: dict[int, list[bytes]] = {}
            band_to_hashes: dict[int, list[bytes]] = {b: [] for b in range(bands)}
            for i, blob in zip(eligible, sig_blobs):
                if blob is None:
                    continue
                sig = np.frombuffer(blob, dtype="<u4")
                sigs[i] = sig
                band_list = _band_hashes(sig, bands)
                bands_of[i] = band_list
                for band_idx, band_hash in enumerate(band_list):
                    band_to_hashes[band_idx].append(band_hash)

            # Phase 2 — one batched cross-shard candidate lookup for the whole
            # chunk (own-scope rows are excluded by the query, so the result
            # is identical to the previous per-record queries).
            db_bands = ctx.manifest.band_candidates_many(band_to_hashes, scope)

            # Phase 3 — sequential resolution in id order (canonical-copy
            # selection and within-shard precedence unchanged).
            for i, rec in enumerate(chunk):
                live.update()
                sig = sigs.get(i)
                if sig is None:
                    skipped += 1
                    out.append(rec)
                    continue
                band_list = bands_of[i]

                # Gather candidates: same-shard from memory, cross-shard from
                # the prefetched map.
                candidate_ids: set[str] = set()
                for band_idx, band_hash in enumerate(band_list):
                    candidate_ids.update(local_bands.get((band_idx, band_hash), ()))
                    candidate_ids.update(db_bands.get((band_idx, band_hash), ()))
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
