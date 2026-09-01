"""Shared raw-file readers (parquet / jsonl / jsonl.gz) with resume fast-skip.

Every reader accepts an optional `skip` budget — a single-element list cell
`[n]` that is decremented as records are skipped — so a resumed shard can
jump straight past already-processed records instead of re-parsing and
re-normalizing them from the beginning:

- Parquet skipping is EXACT and cheap: whole row groups are skipped without
  decoding, and the first surviving batch is sliced.
- JSONL(.gz) skipping counts non-empty lines without JSON-parsing them. A
  malformed line inside the skipped region (which the original pass would
  not have counted) can only make the resume start a few records EARLY —
  the replayed records are then dropped by the global exact-dedup index —
  never late, so no record can be lost to skipping.

The cell is shared across a shard's files, so multi-file shards skip whole
leading files with just a footer read / line count.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from ucc.logging_utils import get_logger

log = get_logger("sources.readers")

try:
    import orjson as _orjson

    def _json_loads(line: str):
        # orjson is several times faster on the per-line hot path, but is
        # stricter than the stdlib (bare NaN/Infinity, >64-bit ints). Fall
        # back to the stdlib for exactly those lines so accepted input is
        # byte-for-byte identical with and without orjson installed.
        try:
            return _orjson.loads(line)
        except json.JSONDecodeError:
            return json.loads(line)
except ImportError:  # pragma: no cover - optional dependency
    _json_loads = json.loads


def iter_parquet_batches(
    path: Path, batch_size: int, skip: list[int] | None = None
) -> Iterator[list[dict]]:
    pf = pq.ParquetFile(path)
    row_groups: list[int] | None = None
    if skip and skip[0] > 0:
        group_rows = [
            pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)
        ]
        start = 0
        while start < len(group_rows) and skip[0] >= group_rows[start]:
            skip[0] -= group_rows[start]
            start += 1
        if start >= len(group_rows):
            return  # the whole file is consumed by the skip budget
        if start > 0:
            row_groups = list(range(start, len(group_rows)))
    iterator = (
        pf.iter_batches(batch_size=batch_size, row_groups=row_groups)
        if row_groups is not None
        else pf.iter_batches(batch_size=batch_size)
    )
    for batch in iterator:
        if skip and skip[0] > 0:
            if skip[0] >= batch.num_rows:
                skip[0] -= batch.num_rows
                continue
            batch = batch.slice(skip[0])
            skip[0] = 0
        yield batch.to_pylist()


def iter_jsonl_gz_batches(
    path: Path, batch_size: int, skip: list[int] | None = None
) -> Iterator[list[dict]]:
    """Stream a gzipped JSONL file in batches.

    Known gotcha (observed on Common Pile shards): the transport layer may
    close the underlying stream before GzipFile finishes its EOF bookkeeping,
    surfacing as ValueError/EOFError/OSError on the very last read. Treat
    those as a clean end-of-shard, but log how many rows were recovered.
    """
    batch: list[dict] = []
    rows = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if skip and skip[0] > 0:
                    if line.strip():  # empty lines were never counted as records
                        skip[0] -= 1
                        rows += 1
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    batch.append(_json_loads(line))
                except json.JSONDecodeError:
                    log.warning("undecodable JSON line in %s (skipped)", path.name)
                    continue
                rows += 1
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    except (EOFError, ValueError, OSError) as exc:
        log.warning(
            "gzip stream for %s ended abruptly after %d rows (%s) — treating as end-of-shard",
            path.name, rows, exc.__class__.__name__,
        )
    if batch:
        yield batch


def iter_jsonl_batches(
    path: Path, batch_size: int, skip: list[int] | None = None
) -> Iterator[list[dict]]:
    batch: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if skip and skip[0] > 0:
                if line.strip():
                    skip[0] -= 1
                continue
            line = line.strip()
            if not line:
                continue
            try:
                batch.append(_json_loads(line))
            except json.JSONDecodeError:
                continue
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def iter_file_batches(
    path: Path, batch_size: int, skip: list[int] | None = None
) -> Iterator[list[dict]]:
    name = path.name.lower()
    if name.endswith(".parquet"):
        yield from iter_parquet_batches(path, batch_size, skip=skip)
    elif name.endswith((".jsonl.gz", ".json.gz")):
        yield from iter_jsonl_gz_batches(path, batch_size, skip=skip)
    elif name.endswith((".jsonl", ".json")):
        yield from iter_jsonl_batches(path, batch_size, skip=skip)
    else:
        raise ValueError(f"unsupported raw file format: {path.name}")
