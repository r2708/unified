"""Local raw-shard capacity gauge.

Enforces the hard cap: at most `max_local_shards` (default 5) raw shards may
exist locally at any moment — downloading, waiting, processing or awaiting
verified upload. The downloader must acquire a slot BEFORE starting a
download and the slot is only released after the raw shard is deleted
(post-verification) or a failed download is cleaned up.
"""

from __future__ import annotations

import threading

from ucc.logging_utils import get_logger

log = get_logger("queue")


class CapacityGauge:
    def __init__(self, max_slots: int):
        if max_slots < 1:
            raise ValueError("max_local_shards must be >= 1")
        self.max_slots = max_slots
        self._lock = threading.Lock()
        self._available = threading.Semaphore(max_slots)
        self._occupied = 0

    @property
    def occupied(self) -> int:
        with self._lock:
            return self._occupied

    def prime(self, already_on_disk: int) -> None:
        """Account for raw shards that survived a crash and are already on
        disk when the pipeline restarts."""
        for _ in range(min(already_on_disk, self.max_slots)):
            acquired = self._available.acquire(blocking=False)
            if not acquired:  # pragma: no cover - defensive
                break
            with self._lock:
                self._occupied += 1
        if already_on_disk > self.max_slots:
            log.warning(
                "found %d raw shards on disk, above the configured cap of %d; "
                "no new downloads will start until the backlog drains",
                already_on_disk,
                self.max_slots,
            )
            with self._lock:
                self._occupied = already_on_disk

    def acquire(self, stop_event: threading.Event, poll_s: float = 1.0) -> bool:
        """Block until a slot frees up (pausing further downloads while the
        queue is full). Returns False if the pipeline is stopping."""
        while not stop_event.is_set():
            if self._available.acquire(timeout=poll_s):
                with self._lock:
                    self._occupied += 1
                return True
        return False

    def release(self) -> None:
        with self._lock:
            if self._occupied == 0:
                log.warning("capacity release with zero occupancy — ignoring")
                return
            self._occupied -= 1
            over_cap = self._occupied >= self.max_slots
        # Occupancy above max_slots (crash backlog) drains before the
        # semaphore is credited again.
        if not over_cap:
            self._available.release()
