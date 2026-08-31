"""bigcode/the-stack-v2 adapter.

The Stack v2 publishes METADATA ONLY (blob ids + provenance) under
data/{Language}/*.parquet; the file contents live in Software Heritage's S3
bucket (s3://softwareheritage/content/{blob_id}, gzip-compressed) and require
BOTH an accepted-terms HF token AND AWS credentials. Gating is never
bypassed: without either credential this source reports itself unavailable
and the pipeline skips it with an explanatory message.

Shard construction: metadata parquet files are split by row groups (footer
reads only — a few KB over HTTP range requests) into units whose estimated
downloaded-content volume lands in the 1–2 GB target. Downloading a shard
means: read the unit's row groups, then fetch each blob from SWH S3 with a
bounded thread pool and write a single local content.parquet — that file is
the raw shard.
"""

from __future__ import annotations

import gzip
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ucc.constants import RT_CODE
from ucc.logging_utils import get_logger
from ucc.schema import new_record
from ucc.sources.base import (
    DownloadError,
    ShardSpec,
    SourceAdapter,
    SourceStatus,
    hf_list_files,
    pinned_revision,
)
from ucc.sources.readers import iter_parquet_batches

log = get_logger("sources.stack_v2")

RAW_CONTENT_SCHEMA = pa.schema(
    [
        ("blob_id", pa.string()),
        ("repo_name", pa.string()),
        ("path", pa.string()),
        ("language", pa.string()),
        ("revision_id", pa.string()),
        ("visit_date", pa.string()),
        ("detected_licenses", pa.list_(pa.string())),
        ("license_type", pa.string()),
        ("star_events_count", pa.int64()),
        ("is_vendor", pa.bool_()),
        ("is_generated", pa.bool_()),
        ("src_encoding", pa.string()),
        ("content", pa.large_string()),
    ]
)

_META_WANTED = [f.name for f in RAW_CONTENT_SCHEMA if f.name != "content"]


def _decode_blob(data: bytes, src_encoding: str | None) -> str | None:
    for enc in (src_encoding, "utf-8"):
        if not enc:
            continue
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    text = data.decode("utf-8", errors="replace")
    if len(text) and text.count("�") / len(text) > 0.05:
        return None  # effectively binary / wrong encoding
    return text


class TheStackV2Adapter(SourceAdapter):
    requires_token = True

    # ------------------------------------------------------------ availability
    def status(self) -> SourceStatus:
        base = super().status()
        if not base.available:
            return base
        try:
            import boto3

            creds = boto3.Session().get_credentials()
        except Exception as exc:  # noqa: BLE001
            return SourceStatus(False, f"boto3 unavailable/misconfigured: {exc}")
        if creds is None:
            return SourceStatus(
                False,
                "the-stack-v2 content lives in Software Heritage S3 and requires "
                "AWS credentials (and acceptance of the SWH/Inria terms). "
                "Configure AWS credentials to enable this source — gating is "
                "never bypassed.",
            )
        return SourceStatus(True, "ok")

    # -------------------------------------------------------------- enumerate
    def _hf_fs(self):
        from huggingface_hub import HfFileSystem

        return HfFileSystem(token=self.token)

    def _fs_path(self, remote_path: str, revision: str | None) -> str:
        """HfFileSystem path with the revision pinned inline
        (datasets/{repo}@{revision}/{path})."""
        if revision:
            return f"datasets/{self.repo_id}@{revision}/{remote_path}"
        return f"datasets/{self.repo_id}/{remote_path}"

    def enumerate_shards(self) -> Iterator[ShardSpec]:
        revision = pinned_revision(self.repo_id, self.token)

        # Shard size is expressed in GB (this source's `shard_gb`, falling
        # back to the global shard.target_gb). Metadata rows carry no content,
        # so the row budget per shard is derived from the GB target and an
        # average source-file size estimate (`est_file_size_kb`, default 12).
        shard_gb = self.source_cfg.get("shard_gb")
        target_bytes = (
            int(float(shard_gb) * 1_000_000_000)
            if shard_gb is not None
            else int(self.cfg.shard.target_bytes)
        )
        est_bytes_per_row = max(1, int(float(self.source_cfg.get("est_file_size_kb", 12)) * 1000))
        rows_per_shard = max(1000, target_bytes // est_bytes_per_row)
        log.info(
            "%s: targeting %.2f GB/shard ≈ %d files/shard (est. %.1f KB/file)",
            self.name, target_bytes / 1e9, rows_per_shard, est_bytes_per_row / 1000,
        )
        max_files_per_language = self.source_cfg.get("max_files_per_language")
        languages = self.source_cfg.get("languages") or ["Python"]
        cap = self.max_shards()
        fs = self._hf_fs()

        seq = 0
        for language in languages:
            files = [
                (path, size)
                for path, size in hf_list_files(
                    self.repo_id, f"data/{language}", revision, self.token
                )
                if path.endswith(".parquet")
            ]
            if max_files_per_language:
                files = files[: int(max_files_per_language)]
            for remote_path, _size in files:
                if cap is not None and seq >= cap:
                    return
                # Footer-only read (HTTP range request) for row-group layout.
                with fs.open(self._fs_path(remote_path, revision), "rb") as fh:
                    meta = pq.ParquetFile(fh).metadata
                    group_rows = [
                        meta.row_group(i).num_rows for i in range(meta.num_row_groups)
                    ]
                start = 0
                while start < len(group_rows):
                    end, rows = start, 0
                    while end < len(group_rows) and rows < rows_per_shard:
                        rows += group_rows[end]
                        end += 1
                    if cap is not None and seq >= cap:
                        return
                    yield ShardSpec(
                        shard_id=self.shard_id(seq),
                        source=self.name,
                        seq_index=seq,
                        ref={
                            "repo_id": self.repo_id,
                            "revision": revision,
                            "file": remote_path,
                            "row_groups": [start, end],
                            "rows": rows,
                            "language": language,
                        },
                        est_bytes=rows * est_bytes_per_row,
                        record_type_hint=RT_CODE,
                    )
                    seq += 1
                    start = end

    # --------------------------------------------------------------- download
    def _s3_client(self):
        import boto3
        from botocore.config import Config

        threads = int(self.source_cfg.path("swh.max_fetch_threads", 16))
        return boto3.client(
            "s3",
            config=Config(
                max_pool_connections=max(threads, 10),
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def _fetch_blob(self, client, bucket: str, prefix: str, blob_id: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            obj = client.get_object(Bucket=bucket, Key=f"{prefix}/{blob_id}")
            body = obj["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        try:
            return gzip.GzipFile(fileobj=io.BytesIO(body)).read()
        except (OSError, EOFError):
            return body  # rare: stored uncompressed

    def download(self, spec_ref: dict, dest_dir: Path, stop_check=None) -> None:
        bucket = self.source_cfg.path("swh.bucket", "softwareheritage")
        prefix = self.source_cfg.path("swh.prefix", "content")
        threads = int(self.source_cfg.path("swh.max_fetch_threads", 16))
        max_missing = float(self.source_cfg.path("swh.max_missing_ratio", 0.05))

        fs = self._hf_fs()
        with fs.open(self._fs_path(spec_ref["file"], spec_ref.get("revision")),
                     "rb") as fh:
            pf = pq.ParquetFile(fh)
            start, end = spec_ref["row_groups"]
            meta_table = pf.read_row_groups(list(range(start, end)))
        available_cols = [c for c in _META_WANTED if c in meta_table.column_names]
        meta_rows = meta_table.select(available_cols).to_pylist()
        del meta_table

        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / "content.parquet"
        tmp_path = dest_dir / "content.parquet.part"
        client = self._s3_client()

        from ucc.progress import Progress

        live = Progress(
            log,
            f"SWH fetch {spec_ref['file'].split('/')[-1]}[rg {start}-{end}]",
            total=len(meta_rows),
            min_interval_s=float(self.cfg.path("queue.progress_log_interval_s", 5)),
            check_every=1,
            unit="blobs",
        )
        fetched = 0
        missing = 0
        try:
            writer = pq.ParquetWriter(tmp_path, RAW_CONTENT_SCHEMA, compression="zstd")
            try:
                chunk = 2000
                for offset in range(0, len(meta_rows), chunk):
                    if stop_check is not None and stop_check():
                        raise DownloadError("download interrupted by pipeline shutdown")
                    part = meta_rows[offset : offset + chunk]
                    with ThreadPoolExecutor(max_workers=threads) as pool:
                        blobs = list(
                            pool.map(
                                lambda row: self._fetch_blob(
                                    client, bucket, prefix, row.get("blob_id")
                                )
                                if row.get("blob_id")
                                else None,
                                part,
                            )
                        )
                    out_rows = []
                    for row, blob in zip(part, blobs):
                        if blob is None:
                            missing += 1
                            continue
                        text = _decode_blob(blob, row.get("src_encoding"))
                        if text is None:
                            missing += 1
                            continue
                        licenses = row.get("detected_licenses") or []
                        if isinstance(licenses, str):
                            licenses = [licenses]
                        out_rows.append(
                            {
                                "blob_id": str(row.get("blob_id")),
                                "repo_name": _s(row.get("repo_name")),
                                "path": _s(row.get("path")),
                                "language": _s(row.get("language"))
                                or spec_ref.get("language"),
                                "revision_id": _s(row.get("revision_id")),
                                "visit_date": _s(row.get("visit_date")),
                                "detected_licenses": [str(x) for x in licenses],
                                "license_type": _s(row.get("license_type")),
                                "star_events_count": _i(row.get("star_events_count")),
                                "is_vendor": bool(row.get("is_vendor") or False),
                                "is_generated": bool(row.get("is_generated") or False),
                                "src_encoding": _s(row.get("src_encoding")),
                                "content": text,
                            }
                        )
                        fetched += 1
                    if out_rows:
                        writer.write_table(
                            pa.Table.from_pylist(out_rows, schema=RAW_CONTENT_SCHEMA)
                        )
                    live.update(len(part))
            finally:
                writer.close()
        except DownloadError:
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001
            tmp_path.unlink(missing_ok=True)
            raise DownloadError(
                f"SWH content fetch failed: {exc.__class__.__name__}: {exc}"
            ) from exc

        total = fetched + missing
        if total == 0 or (missing / max(total, 1)) > max_missing:
            tmp_path.unlink(missing_ok=True)
            raise DownloadError(
                f"too many unfetchable blobs: {missing}/{total} "
                f"(max ratio {max_missing})"
            )
        tmp_path.replace(out_path)
        live.close()
        log.info(
            "stack-v2 shard content written: %d records, %d missing blobs", fetched, missing
        )

    # ------------------------------------------------------------------ read
    def iter_raw_batches(self, spec_ref: dict, raw_dir: Path, batch_size: int,
                         skip_records: int = 0) -> Iterator[list[dict]]:
        local = raw_dir / "content.parquet"
        if not local.exists():
            raise FileNotFoundError(f"raw file missing: {local}")
        yield from iter_parquet_batches(
            local, batch_size, skip=[max(int(skip_records), 0)]
        )

    def normalize_record(self, raw: dict, spec_ref: dict) -> dict | None:
        content = raw.get("content")
        if not isinstance(content, str) or not content:
            return None
        repo = raw.get("repo_name") or None
        flags: list[str] = []
        if raw.get("is_vendor"):
            flags.append("vendored")
        if raw.get("is_generated"):
            flags.append("generated_metadata")
        return new_record(
            record_type=RT_CODE,
            content=content,
            repo_name=str(repo) if repo else None,
            repo_url=f"https://github.com/{repo}" if repo and "/" in str(repo) else None,
            path=raw.get("path"),
            language=raw.get("language") or spec_ref.get("language"),
            detected_licenses=[str(x) for x in (raw.get("detected_licenses") or [])],
            commit_id=raw.get("revision_id"),
            stars=raw.get("star_events_count"),
            created_at=raw.get("visit_date"),
            source_record_id=raw.get("blob_id"),
            quality_flags=flags,
        )


def _s(value) -> str | None:
    return str(value) if value is not None else None


def _i(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
