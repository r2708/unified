"""fineweb_pipeline-style live terminal status table.

Renders pipeline progress as a live-updating rich table (Metric | Value)
when ``rich`` is installed and stdout is a TTY. While the table is live the
console log level is raised to WARNING so the display stays clean — the full
INFO log always keeps going to the rotating file. Headless / CI runs (no
TTY, or rich missing) are unaffected: the classic OVERALL log lines remain
the terminal output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ucc import system_stats
from ucc.progress import fmt_eta

try:  # optional, professional TUI
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    _HAS_RICH = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_RICH = False


def render_status_table(snap: dict, workspace: str | Path, elapsed: float):
    """Build the rich Table for one progress snapshot (fineweb layout)."""
    done = int(snap.get("done", 0))
    total = int(snap.get("total", 0))
    records = int(snap.get("records_out", 0) or 0)
    eta = None
    if done and total > done:
        eta = (total - done) * (elapsed / done)
    _used_mb, free_mb = system_stats.disk_usage_mb(workspace)
    counts = snap.get("counts") or {}

    table = Table(title="Unified Code Corpus Pipeline", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Active Shards", snap.get("active_shards") or "-")
    table.add_row(
        "Shards Done",
        f"{done:,}/{total:,} ({100.0 * done / max(total, 1):.1f}%)",
    )
    table.add_row(
        "State Counts",
        " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-",
    )
    table.add_row("Pending", f"{snap.get('pending', 0):,}")
    table.add_row("In Flight", f"{snap.get('in_flight', 0):,}")
    table.add_row(
        "Retryable / Given Up",
        f"{snap.get('retryable', 0):,} / {snap.get('given_up', 0):,}",
    )
    table.add_row(
        "Queue Slots", f"{snap.get('slots_used', 0)}/{snap.get('slots_max', 0)}"
    )
    table.add_row("Records Out", f"{records:,}")
    table.add_row("Tokens", f"≈{int(snap.get('tokens', 0) or 0):,}")
    table.add_row(
        "Processing Speed", f"{records / max(elapsed, 1e-6):,.1f} rec/s"
    )
    table.add_row("Memory Usage", f"{system_stats.memory_usage_mb():,.1f} MB")
    table.add_row("Disk Free", f"{free_mb:,.0f} MB")
    table.add_row("ETA", fmt_eta(eta) if eta is not None else "n/a")
    table.add_row("Elapsed", fmt_eta(elapsed))
    return table


class LiveStatusTable:
    """Live table when possible; inert (no-op) otherwise."""

    def __init__(self, workspace: str | Path, enabled: bool = True):
        self.workspace = workspace
        self.enabled = bool(enabled) and _HAS_RICH and sys.stdout.isatty()
        self._live = None
        self._t0 = time.time()

    def start(self) -> None:
        if not self.enabled or self._live is not None:
            return
        self._t0 = time.time()
        self._live = Live(
            render_status_table({}, self.workspace, 0.0),
            console=Console(),
            refresh_per_second=4,
        )
        self._live.start()

    def update(self, snap: dict) -> None:
        if self._live is not None:
            self._live.update(
                render_status_table(snap, self.workspace, time.time() - self._t0)
            )

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
