"""Resume must jump DIRECTLY to the last checkpoint — never redo work."""

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from ucc.processing.normalize import done_prefix
from ucc.sources.readers import iter_file_batches, iter_jsonl_gz_batches, iter_parquet_batches


def test_done_prefix_contiguous_and_holes():
    assert done_prefix({}) == (0, 0)
    done = {"0": {"records_in": 100}, "1": {"records_in": 250}}
    assert done_prefix(done) == (2, 350)
    # a hole stops the prefix (caller falls back to per-batch guards)
    done = {"0": {"records_in": 100}, "2": {"records_in": 50}}
    assert done_prefix(done) == (1, 100)


def _write_parquet(path, n, row_group_size):
    table = pa.table({"v": list(range(n))})
    pq.write_table(table, path, row_group_size=row_group_size)


def test_parquet_skip_is_exact_across_row_groups(tmp_path):
    path = tmp_path / "a.parquet"
    _write_parquet(path, 10, row_group_size=3)  # groups: 3+3+3+1
    skip = [4]  # crosses the first group boundary, lands mid-group
    rows = [r["v"] for b in iter_parquet_batches(path, batch_size=4, skip=skip) for r in b]
    assert rows == [4, 5, 6, 7, 8, 9]
    assert skip[0] == 0


def test_parquet_skip_spans_multiple_files(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _write_parquet(a, 10, row_group_size=5)
    _write_parquet(b, 5, row_group_size=5)
    skip = [13]  # consumes all of file a (10) + 3 of file b
    rows = []
    for path in (a, b):  # shared cell, exactly like the adapters
        rows += [r["v"] for batch in iter_file_batches(path, 4, skip=skip) for r in batch]
    assert rows == [3, 4]
    assert skip[0] == 0


def test_jsonl_gz_skip_counts_records_not_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl.gz"
    lines = []
    for i in range(3):
        lines.append(json.dumps({"i": i}))
    lines.append("")                      # blank: never counted as a record
    for i in range(3, 6):
        lines.append(json.dumps({"i": i}))
    lines.append("{not json")             # malformed: skipped with a warning
    lines.append(json.dumps({"i": 6}))
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))

    skip = [3]
    rows = [r["i"] for b in iter_jsonl_gz_batches(path, batch_size=2, skip=skip) for r in b]
    assert rows == [3, 4, 5, 6]
    assert skip[0] == 0
