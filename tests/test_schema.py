"""Parquet writer row-group sizing: row cap AND uncompressed-byte cap."""

import pyarrow.parquet as pq

from ucc.schema import new_record, write_rows_parquet


def _rec(i: int, content: str) -> dict:
    return new_record(id=f"r{i:04d}", content=content,
                      size_bytes=len(content.encode()))


def test_row_group_closes_on_byte_cap(tmp_path):
    # 10 rows of ~1 MB with a 2 MB cap -> a new group every 2 rows.
    rows = [_rec(i, "x" * 1_000_000) for i in range(10)]
    path = tmp_path / "bytes.parquet"
    n = write_rows_parquet(iter(rows), str(path), row_group_size=100_000,
                           row_group_max_bytes=2_000_000)
    pf = pq.ParquetFile(path)
    assert n == 10
    assert pf.metadata.num_rows == 10
    assert pf.num_row_groups == 5


def test_row_group_closes_on_row_cap(tmp_path):
    rows = [_rec(i, f"tiny {i}") for i in range(1000)]
    path = tmp_path / "rows.parquet"
    n = write_rows_parquet(iter(rows), str(path), row_group_size=300,
                           row_group_max_bytes=64 * 1024 * 1024)
    pf = pq.ParquetFile(path)
    assert n == 1000
    assert pf.num_row_groups == 4  # 300+300+300+100
    assert [pf.metadata.row_group(i).num_rows for i in range(4)] == [300, 300, 300, 100]
