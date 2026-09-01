"""Optional multi-core execution for per-record CPU-bound stage work.

The stage pipeline is pure-Python regex/hash work that holds the GIL, so a
single process worker caps throughput at one core. With
processing.cpu_workers != 1 the embarrassingly-parallel per-record
computations (MinHash signatures, the secrets battery, quality metrics) are
chunked out to a shared process pool. Guarantees:

- results come back in input order (`ProcessPoolExecutor.map`), and all
  record mutation, exclusion and stats bookkeeping stays in the parent, so
  parallel and serial runs are bit-identical;
- worker functions are pure top-level functions of picklable arguments —
  no manifest/DB access ever happens in a worker;
- small inputs always run serially (tests and tiny batches never pay pool
  startup), and a broken pool (e.g. a worker OOM-killed) falls back to the
  serial path for that call instead of failing the shard.

cpu_workers: 1 = serial, N > 1 = pool of N processes, 0 = auto
(cpu_count - 2, capped at 8).
"""

from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Sequence, TypeVar

from ucc.logging_utils import get_logger

log = get_logger("parallel")

T = TypeVar("T")
R = TypeVar("R")

# Below this many items the IPC + pool overhead outweighs the parallel gain.
MIN_PARALLEL_ITEMS = 2048

_pool: ProcessPoolExecutor | None = None
_pool_size = 0
_pool_lock = threading.Lock()


def resolve_cpu_workers(value: object) -> int:
    """processing.cpu_workers -> effective worker count (0 = auto)."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    if n == 0:
        return max(1, min(8, (os.cpu_count() or 2) - 2))
    return max(1, n)


def _shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def _get_pool(workers: int) -> ProcessPoolExecutor:
    """Lazily create (and reuse) one process pool for the whole run."""
    global _pool, _pool_size
    with _pool_lock:
        if _pool is None or _pool_size != workers:
            if _pool is not None:
                _pool.shutdown(wait=False, cancel_futures=True)
            # spawn (the macOS default, and the only fork-safe choice in this
            # multithreaded process) — workers import ucc fresh.
            import multiprocessing

            _pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            _pool_size = workers
        return _pool


atexit.register(_shutdown_pool)


def parallel_map(
    fn: Callable[[T], R], items: Sequence[T], cpu_workers: int
) -> list[R]:
    """Order-preserving map of a pure, top-level function over items.

    Serial when cpu_workers <= 1 or the input is small; otherwise runs on the
    shared process pool. `fn` must be a module-level function or a
    functools.partial of one (picklable), and must not touch shared state.
    """
    if cpu_workers <= 1 or len(items) < MIN_PARALLEL_ITEMS:
        return [fn(item) for item in items]
    chunksize = max(1, min(512, len(items) // (cpu_workers * 4)))
    try:
        pool = _get_pool(cpu_workers)
        return list(pool.map(fn, items, chunksize=chunksize))
    except BrokenProcessPool:
        log.warning(
            "process pool broke (worker killed?) — finishing this call "
            "serially; the pool is rebuilt on next use"
        )
        _shutdown_pool()
        return [fn(item) for item in items]
