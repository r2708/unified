"""Source adapter framework.

Each adapter knows how to:
  1. enumerate deterministic 1–2 GB shard specs (streaming — never lists or
     downloads the full dataset content, only file metadata / parquet footers),
  2. download exactly one shard into a local directory,
  3. iterate the raw records of a downloaded shard in bounded batches,
  4. normalize a raw record into the unified schema.

Enumeration pins the dataset revision (commit sha) so shard ids and shard
contents stay deterministic even if the upstream dataset changes later.
"""

from __future__ import annotations

import fnmatch
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ucc.constants import ENV_HF_TOKEN, RT_CODE
from ucc.logging_utils import get_logger

log = get_logger("sources.base")


class SourceUnavailable(Exception):
    """Source cannot be used right now (gated without token, etc.).
    The pipeline skips it and says why — it never bypasses gating."""


class DownloadError(Exception):
    pass


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    source: str
    seq_index: int
    ref: dict                      # files / row-group ranges / revision
    est_bytes: int | None = None
    record_type_hint: str = RT_CODE


@dataclass
class SourceStatus:
    available: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


def hf_token() -> str | None:
    token = os.environ.get(ENV_HF_TOKEN)
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def hf_list_files(repo_id: str, prefix: str = "", revision: str | None = None,
                  token: str | None = None) -> list[tuple[str, int]]:
    """List (path, size) for dataset repo files under a prefix, sorted by path.
    Uses the tree API (public even for gated=auto repos)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    entries = api.list_repo_tree(
        repo_id, path_in_repo=prefix or None, repo_type="dataset",
        revision=revision, recursive=True,
    )
    files: list[tuple[str, int]] = []
    for entry in entries:
        if entry.__class__.__name__ == "RepoFile" or hasattr(entry, "size"):
            size = getattr(entry, "size", None)
            if size is not None:
                files.append((entry.path, int(size)))
    return sorted(files)


def pinned_revision(repo_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    info = HfApi(token=token).dataset_info(repo_id)
    return info.sha


def group_files_into_units(
    files: list[tuple[str, int]], target_bytes: int, max_bytes: int
) -> list[list[tuple[str, int]]]:
    """Greedy, deterministic grouping of source files into ~target-sized
    processing units. A single file larger than max_bytes becomes its own
    unit (splitting a remote file is handled by row-group sharding where the
    format allows it)."""
    units: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_bytes = 0
    for path, size in files:
        if size >= max_bytes:
            if current:
                units.append(current)
                current, current_bytes = [], 0
            units.append([(path, size)])
            continue
        if current and current_bytes + size > max_bytes:
            units.append(current)
            current, current_bytes = [], 0
        current.append((path, size))
        current_bytes += size
        if current_bytes >= target_bytes:
            units.append(current)
            current, current_bytes = [], 0
    if current:
        units.append(current)
    return units


def match_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


class SourceAdapter(ABC):
    #: True when downloads require an accepted-terms HF token.
    requires_token: bool = False

    def __init__(self, name: str, cfg, priority: int = 100):
        from ucc.config import Cfg

        self.name = name
        self.cfg = cfg
        self.source_cfg = Cfg(cfg.sources[name])
        self.priority = priority
        self.repo_id = self.source_cfg["repo_id"]
        self.token = hf_token()

    # ------------------------------------------------------------- helpers
    def shard_id(self, seq_index: int) -> str:
        return f"{self.name}-{seq_index:06d}"

    def enumeration_signature(self, revision: str) -> str:
        return f"{self.repo_id}@{revision}"

    def max_shards(self) -> int | None:
        return self.source_cfg.get("max_shards")

    # ----------------------------------------------------------- interface
    def status(self) -> SourceStatus:
        """Can this source be used right now? Never bypass gating: a gated
        source without an accepted-terms token is reported unavailable."""
        if self.requires_token and not self.token:
            return SourceStatus(
                False,
                f"{self.repo_id} is gated: run `hf auth login` after accepting "
                "the dataset terms on huggingface.co — skipping (never bypassed)",
            )
        return SourceStatus(True, "ok")

    @abstractmethod
    def enumerate_shards(self) -> Iterator[ShardSpec]:
        """Yield deterministic shard specs (ordered by seq_index)."""

    @abstractmethod
    def download(self, spec_ref: dict, dest_dir: Path, stop_check=None) -> None:
        """Download one shard's raw files into dest_dir. Must raise
        DownloadError on failure. Must not touch anything outside dest_dir."""

    @abstractmethod
    def iter_raw_batches(self, spec_ref: dict, raw_dir: Path, batch_size: int,
                         skip_records: int = 0) -> Iterator[list[dict]]:
        """Iterate raw records of a downloaded shard in bounded batches.

        skip_records: resume fast-skip — jump past this many leading records
        at the reader level (row-group skips / line counting) instead of
        re-parsing them, so a resumed shard starts where it left off rather
        than from the beginning."""

    @abstractmethod
    def normalize_record(self, raw: dict, spec_ref: dict) -> dict | None:
        """Map a raw record to a partial unified record
        (content/repo/path/language/licenses/provenance fields).
        Return None to skip a structurally unusable record."""
