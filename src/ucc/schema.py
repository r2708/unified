"""Unified record schema shared by every source adapter and pipeline stage.

Records flow through the stages as plain dicts (one per record) and are only
converted to Arrow tables at checkpoints and final output, which keeps stage
code simple and lets 1–2 GB shards stream in bounded batches.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

UNIFIED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("record_type", pa.string()),          # code | commit | issue | doc
        ("content", pa.large_string()),
        ("content_sha256", pa.string()),
        ("size_bytes", pa.int64()),
        ("token_count", pa.int64()),
        ("line_count", pa.int32()),
        ("avg_line_length", pa.float32()),
        ("max_line_length", pa.int32()),
        ("alnum_ratio", pa.float32()),
        ("repo_name", pa.string()),
        ("repo_url", pa.string()),
        ("path", pa.string()),
        ("language", pa.string()),
        ("license", pa.string()),              # normalized SPDX ids, comma-joined
        ("detected_licenses", pa.list_(pa.string())),
        ("license_status", pa.string()),
        ("commit_id", pa.string()),
        ("stars", pa.int64()),
        ("created_at", pa.string()),
        ("source_dataset", pa.string()),
        ("source_datasets", pa.list_(pa.string())),
        ("source_shard", pa.string()),
        ("source_record_id", pa.string()),
        ("layer", pa.string()),
        ("technologies", pa.list_(pa.string())),
        ("repo_category", pa.string()),
        ("file_complexity", pa.float32()),
        ("repo_complexity", pa.float32()),
        ("quality_score", pa.float32()),       # e.g. stackv2_edu score
        ("quality_flags", pa.list_(pa.string())),
        ("secrets_redacted", pa.int32()),
        ("pii_redacted", pa.int32()),
        ("is_near_duplicate", pa.bool_()),
        ("near_dup_cluster", pa.string()),
        ("pipeline_version", pa.string()),
        ("config_hash", pa.string()),
    ]
)

# Excluded-record audit report: metadata + reason only. Content is NEVER
# stored for excluded records (a license-excluded file must not be
# redistributed through the audit trail either).
EXCLUDED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("content_sha256", pa.string()),
        ("record_type", pa.string()),
        ("source_dataset", pa.string()),
        ("source_shard", pa.string()),
        ("repo_name", pa.string()),
        ("path", pa.string()),
        ("language", pa.string()),
        ("size_bytes", pa.int64()),
        ("reason", pa.string()),
        ("detail", pa.string()),
    ]
)

REPO_SCHEMA = pa.schema(
    [
        ("repo_name", pa.string()),
        ("repo_url", pa.string()),
        ("n_files", pa.int64()),
        ("n_tokens", pa.int64()),
        ("n_commits", pa.int64()),
        ("n_issues", pa.int64()),
        ("n_deps", pa.int64()),
        ("languages", pa.list_(pa.string())),
        ("layers", pa.list_(pa.string())),
        ("technologies", pa.list_(pa.string())),
        ("licenses", pa.list_(pa.string())),
        ("has_tests", pa.bool_()),
        ("has_ci", pa.bool_()),
        ("has_infrastructure", pa.bool_()),
        ("category", pa.string()),
        ("complexity", pa.float32()),
        ("capture", pa.string()),   # partial_stream_sample | consolidated
    ]
)


def new_record(**overrides) -> dict:
    """A unified record with every field present and defaulted."""
    rec = {
        "id": None,
        "record_type": "code",
        "content": None,
        "content_sha256": None,
        "size_bytes": 0,
        "token_count": 0,
        "line_count": 0,
        "avg_line_length": 0.0,
        "max_line_length": 0,
        "alnum_ratio": 0.0,
        "repo_name": None,
        "repo_url": None,
        "path": None,
        "language": None,
        "license": None,
        "detected_licenses": [],
        "license_status": None,
        "commit_id": None,
        "stars": None,
        "created_at": None,
        "source_dataset": None,
        "source_datasets": [],
        "source_shard": None,
        "source_record_id": None,
        "layer": None,
        "technologies": [],
        "repo_category": None,
        "file_complexity": None,
        "repo_complexity": None,
        "quality_score": None,
        "quality_flags": [],
        "secrets_redacted": 0,
        "pii_redacted": 0,
        "is_near_duplicate": False,
        "near_dup_cluster": None,
        "pipeline_version": None,
        "config_hash": None,
    }
    rec.update(overrides)
    return rec


def rows_to_table(rows: list[dict], schema: pa.Schema = UNIFIED_SCHEMA) -> pa.Table:
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def write_rows_parquet(
    rows: Iterable[dict],
    path: str,
    schema: pa.Schema = UNIFIED_SCHEMA,
    compression: str = "zstd",
    row_group_size: int = 2048,
) -> int:
    """Stream rows to a parquet file in bounded batches. Returns row count."""
    writer = pq.ParquetWriter(path, schema, compression=compression)
    count = 0
    batch: list[dict] = []
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= row_group_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                count += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            count += len(batch)
    finally:
        writer.close()
    return count


def iter_parquet_rows(path: str, batch_size: int = 2048) -> Iterator[dict]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def load_parquet_rows(path: str, batch_size: int = 2048) -> list[dict]:
    return list(iter_parquet_rows(path, batch_size=batch_size))
