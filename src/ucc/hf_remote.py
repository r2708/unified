"""Hugging Face Hub client with upload verification, plus a local mock hub.

The mock hub mirrors the real client's interface against a local directory so
the prototype's crash/resume tests can exercise the full
upload -> verify -> cleanup path without touching the real Hub.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ucc.constants import ENV_HF_TOKEN
from ucc.hashing import git_blob_sha1_file, sha256_file
from ucc.io_utils import atomic_write_json, ensure_dir
from ucc.logging_utils import get_logger

log = get_logger("hub")


@dataclass
class RemoteFileInfo:
    exists: bool
    size: int | None = None
    sha256: str | None = None       # LFS files
    git_sha1: str | None = None     # small (non-LFS) files


class HubError(RuntimeError):
    pass


def retry_call(fn, what: str, retries: int = 5, base_delay: float = 2.0):
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network layer
            last_exc = exc
            delay = min(base_delay * (2**attempt), 120.0) + random.uniform(0, 1)
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what, attempt + 1, retries, exc.__class__.__name__, delay,
            )
            time.sleep(delay)
    raise HubError(f"{what} failed after {retries} attempts: {last_exc}") from last_exc


class HubClient:
    """Interface shared by RealHub and MockHub."""

    def ensure_repo(self) -> None:
        raise NotImplementedError

    def file_info(self, dest_path: str) -> RemoteFileInfo:
        raise NotImplementedError

    def upload_file(self, local_path: str | Path, dest_path: str, message: str) -> None:
        raise NotImplementedError

    def upload_json(self, obj: object, dest_path: str, message: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            atomic_write_json(path, obj)
            self.upload_file(path, dest_path, message)

    def list_files(self, prefix: str = "") -> list[str]:
        raise NotImplementedError


class RealHub(HubClient):
    def __init__(self, repo_id: str, private: bool = True, max_retries: int = 5):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.private = private
        self.max_retries = max_retries
        self.api = HfApi(token=os.environ.get(ENV_HF_TOKEN) or None)

    def ensure_repo(self) -> None:
        retry_call(
            lambda: self.api.create_repo(
                self.repo_id, repo_type="dataset", private=self.private, exist_ok=True
            ),
            f"create_repo({self.repo_id})",
            retries=self.max_retries,
        )

    def file_info(self, dest_path: str) -> RemoteFileInfo:
        def _fetch():
            return self.api.get_paths_info(
                self.repo_id, paths=[dest_path], repo_type="dataset"
            )

        infos = retry_call(_fetch, f"get_paths_info({dest_path})", retries=self.max_retries)
        for info in infos:
            if getattr(info, "path", None) != dest_path:
                continue
            lfs = getattr(info, "lfs", None)
            sha256 = None
            if lfs is not None:
                sha256 = getattr(lfs, "sha256", None) or (
                    lfs.get("sha256") if isinstance(lfs, dict) else None
                )
            return RemoteFileInfo(
                exists=True,
                size=getattr(info, "size", None),
                sha256=sha256,
                git_sha1=getattr(info, "blob_id", None),
            )
        return RemoteFileInfo(exists=False)

    def upload_file(self, local_path: str | Path, dest_path: str, message: str) -> None:
        retry_call(
            lambda: self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=dest_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=message,
            ),
            f"upload_file({dest_path})",
            retries=self.max_retries,
        )

    def list_files(self, prefix: str = "") -> list[str]:
        files = retry_call(
            lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset"),
            f"list_repo_files({self.repo_id})",
            retries=self.max_retries,
        )
        return [f for f in files if f.startswith(prefix)]


class MockHub(HubClient):
    """Filesystem-backed stand-in with identical verification semantics."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def ensure_repo(self) -> None:
        ensure_dir(self.root)

    def _target(self, dest_path: str) -> Path:
        target = (self.root / dest_path).resolve()
        if self.root.resolve() not in target.parents:
            raise HubError(f"mock hub path escapes root: {dest_path}")
        return target

    def file_info(self, dest_path: str) -> RemoteFileInfo:
        target = self._target(dest_path)
        if not target.exists():
            return RemoteFileInfo(exists=False)
        return RemoteFileInfo(
            exists=True,
            size=target.stat().st_size,
            sha256=sha256_file(target),
            git_sha1=git_blob_sha1_file(target),
        )

    def upload_file(self, local_path: str | Path, dest_path: str, message: str) -> None:
        target = self._target(dest_path)
        ensure_dir(target.parent)
        tmp = target.with_suffix(target.suffix + ".uploading")
        shutil.copyfile(local_path, tmp)
        os.replace(tmp, target)

    def list_files(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        out = []
        for path in self.root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self.root).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)


def build_hub(cfg) -> HubClient:
    from ucc.config import workspace_paths

    if cfg.hf.mode == "mock":
        return MockHub(workspace_paths(cfg).mock_hub)
    return RealHub(cfg.hf.target_repo, private=cfg.hf.private, max_retries=cfg.hf.max_retries)


def verify_remote_file(
    hub: HubClient, local_path: str | Path, dest_path: str,
    expected_sha256: str | None = None, expected_size: int | None = None,
) -> tuple[bool, str]:
    """Verify a remote file against the local file (or recorded checksums when
    the local file no longer exists). Prefers content hashes; falls back to
    size-only with an explicit note."""
    info = hub.file_info(dest_path)
    if not info.exists:
        return False, "remote file missing"

    local = Path(local_path) if local_path else None
    local_exists = local is not None and local.exists()

    size = expected_size if expected_size is not None else (
        local.stat().st_size if local_exists else None
    )
    if size is not None and info.size is not None and info.size != size:
        return False, f"size mismatch (remote={info.size}, expected={size})"

    sha = expected_sha256 if expected_sha256 else (sha256_file(local) if local_exists else None)
    if info.sha256 is not None:
        if sha is None:
            return (info.size == size and size is not None), "size-only (no local checksum)"
        return (info.sha256 == sha), (
            "sha256 verified" if info.sha256 == sha else "sha256 mismatch"
        )
    if info.git_sha1 is not None and local_exists:
        local_git = git_blob_sha1_file(local)
        return (info.git_sha1 == local_git), (
            "git-sha1 verified" if info.git_sha1 == local_git else "git-sha1 mismatch"
        )
    if size is not None and info.size == size:
        return True, "size-only verification"
    return False, "no verifiable checksum available"
