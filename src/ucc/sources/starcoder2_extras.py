"""bigcode/starcoder2data-extras adapter (public).

The task brief listed "https://huggingface.co/starcoder2data", which is not a
dataset repo; the real StarCoder2 companion data is
bigcode/starcoder2data-extras. It is organized as one directory per config;
the `issues` config holds 15.5M real GitHub issue threads with schema
{repo_name, content, issue_id} (usernames pre-anonymized as username_N).
Other configs (documentation, kaggle, stackoverflow, ...) are optional and
disabled by default because they are not repository code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ucc.constants import RT_CODE, RT_DOC, RT_ISSUE
from ucc.logging_utils import get_logger
from ucc.schema import new_record
from ucc.sources.base import (
    ShardSpec,
    SourceAdapter,
    group_files_into_units,
    hf_list_files,
    pinned_revision,
)
from ucc.sources.hf_download import download_hf_files
from ucc.sources.readers import iter_file_batches

log = get_logger("sources.sc2extras")

_CONFIG_RECORD_TYPE = {
    "issues": RT_ISSUE,
    "documentation": RT_DOC,
}
_DATA_EXTS = (".parquet", ".jsonl", ".jsonl.gz", ".json.gz")


class StarCoder2ExtrasAdapter(SourceAdapter):
    requires_token = False

    def _configs(self) -> list[str]:
        return self.source_cfg.get("include_configs") or ["issues"]

    def enumerate_shards(self) -> Iterator[ShardSpec]:
        revision = pinned_revision(self.repo_id, self.token)
        seq = 0
        cap = self.max_shards()
        for config_name in self._configs():
            files = [
                (path, size)
                for path, size in hf_list_files(self.repo_id, config_name, revision, self.token)
                if path.lower().endswith(_DATA_EXTS)
            ]
            log.info("%s/%s: %d data files", self.name, config_name, len(files))
            units = group_files_into_units(
                files, self.cfg.shard.target_bytes, self.cfg.shard.max_bytes
            )
            record_type = _CONFIG_RECORD_TYPE.get(config_name, RT_CODE)
            for unit in units:
                if cap is not None and seq >= cap:
                    return
                yield ShardSpec(
                    shard_id=self.shard_id(seq),
                    source=self.name,
                    seq_index=seq,
                    ref={
                        "repo_id": self.repo_id,
                        "revision": revision,
                        "config": config_name,
                        "record_type": record_type,
                        "files": [[path, size] for path, size in unit],
                    },
                    est_bytes=sum(size for _, size in unit),
                    record_type_hint=record_type,
                )
                seq += 1

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
        record_type = spec_ref.get("record_type", RT_CODE)
        content = raw.get("content") or raw.get("text")
        if not isinstance(content, str) or not content:
            return None
        repo = raw.get("repo_name") or raw.get("repo") or None
        source_record_id = raw.get("issue_id", raw.get("id"))
        return new_record(
            record_type=record_type,
            content=content,
            repo_name=str(repo) if repo else None,
            repo_url=f"https://github.com/{repo}" if repo and "/" in str(repo) else None,
            source_record_id=str(source_record_id) if source_record_id is not None else None,
        )
