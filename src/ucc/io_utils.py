"""Atomic filesystem helpers. All destructive operations are workspace-guarded."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_rmtree(path: str | Path, must_be_under: str | Path) -> None:
    """Delete a directory tree only if it lives inside the workspace root."""
    path = Path(path).resolve()
    root = Path(must_be_under).resolve()
    if not path.exists():
        return
    if root not in path.parents and path != root:
        raise RuntimeError(f"refusing to delete {path}: outside workspace {root}")
    shutil.rmtree(path, ignore_errors=True)


def safe_unlink(path: str | Path, must_be_under: str | Path) -> None:
    path = Path(path).resolve()
    root = Path(must_be_under).resolve()
    if not path.exists():
        return
    if root not in path.parents:
        raise RuntimeError(f"refusing to delete {path}: outside workspace {root}")
    path.unlink(missing_ok=True)


def dir_size_bytes(path: str | Path) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def free_disk_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / 1e9
