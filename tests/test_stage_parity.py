"""End-to-end parity: the full stage chain must produce identical rows,
exclusions and stats whether the per-record CPU work runs serially
(cpu_workers=1) or on the process pool — the pool is forced on here by
lowering the size threshold."""

from __future__ import annotations

import pytest

import ucc.parallel as parallel
from tests.conftest import make_rec
from ucc.config import DEFAULTS, Cfg, _deep_merge, compute_config_hash
from ucc.manifest import Manifest
from ucc.processing.base import ShardContext


def _make_rows() -> list[dict]:
    base = "\n".join(f"def handler_{i}(x):\n    return process(x, retries={i})"
                     for i in range(40))
    rows = [
        make_rec(base, repo_name="org/app", path="src/app.py", language="Python"),
        make_rec(base.replace("retries=7", "retries=9") + "\n# fork\n",
                 repo_name="org/fork", path="src/app.py"),
        make_rec("import argparse\nimport os\n" + base,
                 repo_name="org/app", path="src/cli.py"),
        make_rec("# maintainer dev@example-corp.net\nHOST = '203.51.44.99'\n" + base,
                 repo_name="org/app", path="src/net.py"),
        make_rec("aws_key = 'AKIAIOSFODNN7EXAMPLE'\n" + base,
                 repo_name="org/app", path="src/cfg.py"),
        make_rec("SECRET=1\n", repo_name="org/app", path="ops/.env"),
        make_rec("var a=1;" * 3000, repo_name="org/app",
                 path="static/x.min.js", language="JavaScript"),
        make_rec("x", repo_name="org/app", path="src/t.py"),
        make_rec("int main() { return 0; }\n" + base, repo_name="org/gpl",
                 path="main.c", detected_licenses=["GPL-3.0"]),
        make_rec(base, repo_name="org/app", path=None, record_type="commit"),
    ]
    rows.sort(key=lambda r: r["id"])
    return rows


def _run_chain(tmp_path, tag: str, cpu_workers: int):
    from ucc.processing.classify import ClassifyStage
    from ucc.processing.complexity import ComplexityStage
    from ucc.processing.dedup_exact import ExactDedupStage
    from ucc.processing.dedup_near import NearDedupStage
    from ucc.processing.license_filter import LicenseFilterStage
    from ucc.processing.provenance import ProvenanceStage
    from ucc.processing.quality_filter import QualityFilterStage
    from ucc.processing.repo_reconstruct import RepoReconstructStage
    from ucc.processing.secrets_scan import SecretsScanStage

    raw = _deep_merge(DEFAULTS, {
        "paths": {"workspace": str(tmp_path / tag)},
        "processing": {"cpu_workers": cpu_workers},
    })
    raw["config_hash"] = compute_config_hash(raw)
    cfg = Cfg(raw)
    manifest = Manifest(tmp_path / f"{tag}.db")
    manifest.upsert_shard_spec(
        "testsource-000001", "testsource", {}, 1, 0, "0.1.0", cfg.config_hash
    )
    ctx = ShardContext(
        shard=manifest.get_shard("testsource-000001"),
        spec_ref={}, cfg=cfg, manifest=manifest, adapter=None,
        raw_dir=tmp_path / tag / "raw", work_dir=tmp_path / tag / "work",
        processed_dir=tmp_path / tag / "proc", workspace_root=tmp_path / tag,
    )
    rows = _make_rows()
    for stage_cls in (ProvenanceStage, ExactDedupStage, NearDedupStage,
                      RepoReconstructStage, LicenseFilterStage,
                      SecretsScanStage, QualityFilterStage, ComplexityStage,
                      ClassifyStage):
        rows = stage_cls().run(rows, ctx)
    manifest.close()
    return rows, ctx.excluded, ctx.stats


def test_stage_chain_serial_vs_pool_identical(tmp_path, monkeypatch):
    pytest.importorskip("datasketch")
    pytest.importorskip("xxhash")
    # Force the pool even for this tiny row count.
    monkeypatch.setattr(parallel, "MIN_PARALLEL_ITEMS", 1)

    serial_rows, serial_excl, serial_stats = _run_chain(tmp_path, "serial", 1)
    pooled_rows, pooled_excl, pooled_stats = _run_chain(tmp_path, "pooled", 2)

    assert pooled_rows == serial_rows
    assert pooled_excl == serial_excl
    assert pooled_stats == serial_stats
    # Sanity: the chain actually did interesting work.
    assert any(r.get("is_near_duplicate") for r in serial_rows)
    assert any(r.get("secrets_redacted") for r in serial_rows)
    reasons = {e["reason"] for e in serial_excl}
    assert {"secret_carrier_file", "quality", "license_copyleft_strong"} <= reasons
