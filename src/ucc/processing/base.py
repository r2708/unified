"""Stage framework: the per-shard processing context and Stage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ucc.logging_utils import get_logger


@dataclass
class ShardContext:
    shard: dict                      # manifest row
    spec_ref: dict                   # decoded source_ref
    cfg: Any                         # Cfg
    manifest: Any                    # Manifest
    adapter: Any                     # SourceAdapter
    raw_dir: Path
    work_dir: Path
    processed_dir: Path
    workspace_root: Path
    hub: Any = None                                # HubClient (per-batch uploads)
    stats: dict = field(default_factory=dict)
    excluded: list = field(default_factory=list)   # excluded-report rows
    scratch: dict = field(default_factory=dict)    # intra-run stage handoffs
    log: Any = None

    def __post_init__(self):
        if self.log is None:
            self.log = get_logger(f"shard.{self.shard['shard_id']}")

    def bump(self, key: str, n: int | float = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def progress(self, label: str, total: int | None = None, unit: str = "records"):
        """Live percentage logger for this shard's terminal output, throttled
        by queue.progress_log_interval_s."""
        from ucc.progress import Progress

        return Progress(
            self.log, label, total=total,
            min_interval_s=float(self.cfg.path("queue.progress_log_interval_s", 5)),
            unit=unit,
        )

    def exclude(self, rec: dict, reason: str, detail: str = "") -> None:
        """Record an excluded record in the audit report — metadata + reason
        only, never the content (excluded content must not be redistributed
        through the audit trail)."""
        self.excluded.append(
            {
                "id": rec.get("id"),
                "content_sha256": rec.get("content_sha256"),
                "record_type": rec.get("record_type"),
                "source_dataset": rec.get("source_dataset"),
                "source_shard": rec.get("source_shard"),
                "repo_name": rec.get("repo_name"),
                "path": rec.get("path"),
                "language": rec.get("language"),
                "size_bytes": rec.get("size_bytes") or 0,
                "reason": reason,
                "detail": detail[:500] if detail else "",
            }
        )
        self.bump(f"excluded.{reason}")
        self.bump("excluded.total")


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        """Transform the shard's records. May drop rows (recording each drop
        via ctx.exclude) but must stay deterministic and idempotent: given
        the same input rows and manifest state, re-running after a crash
        yields the same output and the same global side effects."""
