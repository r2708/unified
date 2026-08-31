"""Shared fixtures. Test snippets below are unit-test fixtures only — the
pipeline itself never generates corpus content."""

from __future__ import annotations

import pytest

from ucc.config import DEFAULTS, Cfg, _deep_merge, compute_config_hash
from ucc.hashing import make_record_id, sha256_text
from ucc.manifest import Manifest
from ucc.processing.base import ShardContext
from ucc.schema import new_record
from ucc.tokens import count_tokens


@pytest.fixture
def cfg(tmp_path):
    raw = _deep_merge(DEFAULTS, {"paths": {"workspace": str(tmp_path / "ws")}})
    raw["config_hash"] = compute_config_hash(raw)
    raw["config_file"] = "<test>"
    return Cfg(raw)


@pytest.fixture
def manifest(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    yield m
    m.close()


@pytest.fixture
def make_ctx(cfg, manifest, tmp_path):
    def _make(shard_id="testsource-000001", source="testsource", seq=1):
        manifest.upsert_shard_spec(
            shard_id, source, {"files": []}, seq, 0, "0.1.0", cfg.config_hash
        )
        shard = manifest.get_shard(shard_id)
        return ShardContext(
            shard=shard,
            spec_ref={},
            cfg=cfg,
            manifest=manifest,
            adapter=None,
            raw_dir=tmp_path / "raw" / shard_id,
            work_dir=tmp_path / "work" / shard_id,
            processed_dir=tmp_path / "proc" / shard_id,
            workspace_root=tmp_path,
        )

    return _make


def make_rec(content: str, source="testsource", shard="testsource-000001", **kw):
    """Build a unified record the way NormalizeStage would."""
    rec = new_record(content=content, **kw)
    rec["content_sha256"] = sha256_text(content)
    rec["size_bytes"] = len(content.encode("utf-8"))
    rec["token_count"] = count_tokens(content)
    rec["id"] = make_record_id(rec.get("repo_name"), rec.get("path"), rec["content_sha256"])
    rec["source_dataset"] = source
    rec["source_datasets"] = [source]
    rec["source_shard"] = shard
    return rec
