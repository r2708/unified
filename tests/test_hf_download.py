"""download_hf_files retry semantics: transient failures retry the SAME file
with backoff instead of failing the whole shard."""

from __future__ import annotations

import pytest

import ucc.sources.hf_download as hf_download
from ucc.sources.base import DownloadError
from ucc.sources.hf_download import download_hf_files


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(hf_download.time, "sleep", lambda _s: None)


def _fake_hub(tmp_path, fail_times: int, content: bytes = b"x" * 10):
    """hf_hub_download stand-in that fails `fail_times` times, then succeeds."""
    calls = {"n": 0}

    def fake(repo_id, filename, repo_type, revision, local_dir, token):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ConnectionError("transient network error")
        local = tmp_path / "dest" / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        return str(local)

    return fake, calls


def _patch_hub(monkeypatch, fake):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda **kw: fake(**kw),
    )


def test_transient_failure_retries_and_succeeds(tmp_path, monkeypatch):
    fake, calls = _fake_hub(tmp_path, fail_times=2)
    _patch_hub(monkeypatch, fake)
    download_hf_files(
        "org/repo", [("data/a.parquet", 10)], None, tmp_path / "dest", None,
        retries=3,
    )
    assert calls["n"] == 3  # two failures + one success
    assert (tmp_path / "dest" / "data" / "a.parquet").read_bytes() == b"x" * 10


def test_retries_exhausted_raises_download_error(tmp_path, monkeypatch):
    fake, calls = _fake_hub(tmp_path, fail_times=100)
    _patch_hub(monkeypatch, fake)
    with pytest.raises(DownloadError, match="after 3 attempts"):
        download_hf_files(
            "org/repo", [("data/a.parquet", 10)], None, tmp_path / "dest", None,
            retries=2,
        )
    assert calls["n"] == 3


def test_size_mismatch_is_retried_then_fails(tmp_path, monkeypatch):
    # Always returns 10 bytes while 99 are expected: every attempt discards
    # the partial file and retries, then the shard download fails.
    fake, calls = _fake_hub(tmp_path, fail_times=0)
    _patch_hub(monkeypatch, fake)
    with pytest.raises(DownloadError, match="size mismatch"):
        download_hf_files(
            "org/repo", [("data/a.parquet", 99)], None, tmp_path / "dest", None,
            retries=2,
        )
    assert calls["n"] == 3
    assert not (tmp_path / "dest" / "data" / "a.parquet").exists()


def test_stop_check_interrupts_between_attempts(tmp_path, monkeypatch):
    fake, calls = _fake_hub(tmp_path, fail_times=100)
    _patch_hub(monkeypatch, fake)
    # per-file check, first attempt runs (and fails), then shutdown
    stops = iter([False, False, True])
    with pytest.raises(DownloadError, match="interrupted"):
        download_hf_files(
            "org/repo", [("data/a.parquet", 10)], None, tmp_path / "dest", None,
            stop_check=lambda: next(stops), retries=5,
        )
    assert calls["n"] == 1
