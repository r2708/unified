"""Common Pile Stack v2 adapters (public, Dolma-format JSONL.gz shards).

Layout facts (verified against the published repos):
- common-pile/stackv2: documents/{00000..00763}_stackv2.jsonl.gz, ~1.5 GB each
  (a natural fit for the 1–2 GB shard target). Rows are
  {id, text, source, added, created, metadata} where metadata carries the full
  Stack v2 provenance row (blob_id, repo_name, path, revision_id,
  detected_licenses, star_events_count, is_vendor, is_generated, url, ...).
  metadata.license is a comma-joined string — prefer the detected_licenses list.
- common-pile/stackv2 only covers languages alphabetically up to ~Markdown;
  the major languages (Python, TypeScript, ...) live in
  common-pile/stackv2_edu_filtered (stack-edu-*.json.gz, adds score/int_score).
- Rows are NOT grouped by repository, so per-shard repository reconstruction
  is a partial capture; cross-shard consolidation happens in the manifest's
  repos table and the `finalize` step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ucc.constants import RT_CODE
from ucc.logging_utils import get_logger
from ucc.schema import new_record
from ucc.sources.base import (
    ShardSpec,
    SourceAdapter,
    group_files_into_units,
    hf_list_files,
    match_any,
    pinned_revision,
)
from ucc.sources.hf_download import download_hf_files
from ucc.sources.readers import iter_file_batches

log = get_logger("sources.common_pile")


class CommonPileStackV2Adapter(SourceAdapter):
    requires_token = False
    default_patterns = ["documents/*_stackv2.jsonl.gz", "*_stackv2.jsonl.gz"]

    def _patterns(self) -> list[str]:
        return self.source_cfg.get("file_patterns") or self.default_patterns

    def enumerate_shards(self) -> Iterator[ShardSpec]:
        revision = pinned_revision(self.repo_id, self.token)
        files = [
            (path, size)
            for path, size in hf_list_files(self.repo_id, "", revision, self.token)
            if match_any(path, self._patterns())
        ]
        log.info("%s: %d matching files at revision %.12s", self.name, len(files), revision)
        units = group_files_into_units(
            files, self.cfg.shard.target_bytes, self.cfg.shard.max_bytes
        )
        cap = self.max_shards()
        for seq, unit in enumerate(units):
            if cap is not None and seq >= cap:
                break
            yield ShardSpec(
                shard_id=self.shard_id(seq),
                source=self.name,
                seq_index=seq,
                ref={
                    "repo_id": self.repo_id,
                    "revision": revision,
                    "files": [[path, size] for path, size in unit],
                },
                est_bytes=sum(size for _, size in unit),
                record_type_hint=RT_CODE,
            )

    def download(self, spec_ref: dict, dest_dir: Path, stop_check=None) -> None:
        download_hf_files(
            spec_ref["repo_id"],
            [(path, size) for path, size in spec_ref["files"]],
            spec_ref.get("revision"),
            dest_dir,
            self.token,
            stop_check,
        )

    def iter_raw_batches(self, spec_ref: dict, raw_dir: Path, batch_size: int,
                         skip_records: int = 0) -> Iterator[list[dict]]:
        skip = [max(int(skip_records), 0)]
        for path, _size in spec_ref["files"]:
            local = raw_dir / path
            if not local.exists():
                raise FileNotFoundError(f"raw file missing: {local}")
            yield from iter_file_batches(local, batch_size, skip=skip)

    def normalize_record(self, raw: dict, spec_ref: dict) -> dict | None:
        content = raw.get("text")
        if not isinstance(content, str) or not content:
            return None
        meta = raw.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        detected = meta.get("detected_licenses")
        if not detected:
            lic = meta.get("license")
            detected = (
                [part.strip() for part in lic.split(",") if part.strip()]
                if isinstance(lic, str) and lic
                else []
            )
        detected = [str(x) for x in detected]

        repo = meta.get("repo_name") or None
        repo_url = meta.get("url") or (
            f"https://github.com/{repo}" if repo and "/" in str(repo) else None
        )
        flags: list[str] = []
        if meta.get("is_vendor"):
            flags.append("vendored")
        if meta.get("is_generated"):
            flags.append("generated_metadata")

        score = raw.get("score", meta.get("score"))
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        stars = meta.get("star_events_count")
        try:
            stars = int(stars) if stars is not None else None
        except (TypeError, ValueError):
            stars = None

        return new_record(
            record_type=RT_CODE,
            content=content,
            repo_name=str(repo) if repo else None,
            repo_url=str(repo_url) if repo_url else None,
            path=str(meta.get("path")) if meta.get("path") else None,
            language=str(meta.get("language")) if meta.get("language") else None,
            detected_licenses=detected,
            commit_id=str(meta.get("revision_id")) if meta.get("revision_id") else None,
            stars=stars,
            created_at=str(raw.get("created")) if raw.get("created") else None,
            source_record_id=str(raw.get("id")) if raw.get("id") else None,
            quality_score=score,
            quality_flags=flags,
        )


class CommonPileStackV2EduAdapter(CommonPileStackV2Adapter):
    """common-pile/stackv2_edu_filtered — same Dolma rows plus an
    educational-quality score; carries the major languages missing from
    common-pile/stackv2."""

    default_patterns = ["stack-edu-*.json.gz", "*/stack-edu-*.json.gz"]
