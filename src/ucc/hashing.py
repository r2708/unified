"""Hash helpers: content hashes, file hashes, git blob sha1, stable ids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """git's blob object id: sha1(b"blob <size>\\0" + content).

    The Hugging Face Hub reports this id for small (non-LFS) files, which lets
    us verify uploads without re-downloading.
    """
    size = Path(path).stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make_record_id(repo_name: str | None, path: str | None, content_sha256: str) -> str:
    """Deterministic, source-independent record id.

    The same logical file (same repo, path and content) yields the same id no
    matter which source dataset it came from, which keeps re-runs and
    cross-source exact dedup idempotent.
    """
    key = f"{repo_name or ''}\x00{path or ''}\x00{content_sha256}"
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:32]


def combined_raw_checksum(file_hashes: dict[str, str]) -> str:
    """One checksum for a multi-file raw shard: hash of sorted 'relpath:sha256' lines."""
    lines = "\n".join(f"{rel}:{h}" for rel, h in sorted(file_hashes.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()
