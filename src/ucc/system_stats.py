"""Lightweight system-resource sampling (memory / disk) with graceful
fallback when ``psutil`` is not installed. Mirrors fineweb_pipeline's
utils/system_stats.py."""

from __future__ import annotations

import shutil
from pathlib import Path

try:  # optional dependency
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_PSUTIL = False


def memory_usage_mb() -> float:
    """Resident memory of the current process in MB (0.0 if unavailable)."""
    if not _HAS_PSUTIL:
        return 0.0
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover
        return 0.0


def disk_usage_mb(path: str | Path = ".") -> tuple[float, float]:
    """Return ``(used_mb, free_mb)`` for the filesystem containing ``path``."""
    try:
        total, used, free = shutil.disk_usage(str(path))
        return used / (1024 * 1024), free / (1024 * 1024)
    except Exception:  # pragma: no cover
        return 0.0, 0.0
