"""Throttled live progress for terminal logs.

Emits lines like

    [starcoderdata-000002] secrets_scan:  62.4% (127,488/204,310, 8,412/s, eta 9s)

rate-limited (default: one line every few seconds per task) so multi-hour
runs stay readable while still showing live percentages. Count-only mode
(total=None) is used where the total isn't known upfront.
"""

from __future__ import annotations

import time


def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class Progress:
    def __init__(self, log, label: str, total: int | None = None,
                 min_interval_s: float = 5.0, check_every: int = 1024,
                 unit: str = ""):
        self.log = log
        self.label = label
        self.total = int(total) if total else None
        self.min_interval = float(min_interval_s)
        self.check_every = max(int(check_every), 1)
        self.unit = f" {unit}" if unit else ""
        self.done = 0
        self._since_check = 0
        self._t0 = time.monotonic()
        self._last_emit = self._t0

    def update(self, n: int = 1) -> None:
        self.done += n
        self._since_check += n
        if self._since_check < self.check_every:
            return
        self._since_check = 0
        now = time.monotonic()
        if now - self._last_emit >= self.min_interval:
            self._emit(now)

    def _emit(self, now: float) -> None:
        self._last_emit = now
        elapsed = max(now - self._t0, 1e-9)
        rate = self.done / elapsed
        if self.total:
            pct = min(100.0, 100.0 * self.done / self.total)
            remaining = max(self.total - self.done, 0)
            eta = remaining / rate if rate > 0 else 0.0
            self.log.info(
                "%s: %5.1f%% (%s/%s%s, %s/s, eta %s)",
                self.label, pct, f"{self.done:,}", f"{self.total:,}",
                self.unit, f"{rate:,.0f}", fmt_eta(eta),
            )
        else:
            self.log.info(
                "%s: %s%s done (%s/s)",
                self.label, f"{self.done:,}", self.unit, f"{rate:,.0f}",
            )

    def close(self) -> None:
        """Emit the final line (100% / final count) unconditionally."""
        self._emit(time.monotonic())
