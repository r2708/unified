"""Dataset statistics: aggregation, status rendering, and the `finalize`
export (global stats + consolidated repository table + cross-source
provenance + dataset card) uploaded to the Hub."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pyarrow as pa

from ucc.config import Cfg, workspace_paths
from ucc.io_utils import ensure_dir
from ucc.logging_utils import get_logger
from ucc.schema import REPO_SCHEMA
from ucc.states import RETRY_RECYCLE, ShardState

log = get_logger("stats")

PROVENANCE_SCHEMA = pa.schema(
    [
        ("content_sha256", pa.string()),
        ("canonical_record_id", pa.string()),
        ("source_datasets", pa.list_(pa.string())),
    ]
)


def aggregate_global(manifest) -> dict:
    """Global statistics = sum of idempotent per-shard stat blobs (never
    incremented counters, so crashes/re-runs cannot double count)."""
    shards = manifest.all_shards()
    counts = manifest.counts_by_state()
    agg: dict = {
        "shards": {"total": len(shards), "by_state": counts},
        "sources": {},
        "records_in": 0,
        "records_out": 0,
        "token_count": 0,
        "duplicates_removed_exact": 0,
        "near_duplicates_annotated": 0,
        "near_duplicates_dropped": 0,
        "secrets_redacted": 0,
        "pii_redacted": 0,
        "excluded_records": 0,
        "processing_failures": sum(
            counts.get(s.value, 0) for s in RETRY_RECYCLE.keys()
        ),
        "excluded_by_reason": {},
        "licenses": {},
        "layers": {},
    }
    for shard in shards:
        src = shard["source_dataset"]
        source_agg = agg["sources"].setdefault(
            src, {"shards_completed": 0, "records_out": 0, "token_count": 0}
        )
        if shard["state"] == ShardState.COMPLETED.value:
            source_agg["shards_completed"] += 1
            agg["records_in"] += shard.get("records_in") or 0
            agg["records_out"] += shard.get("records_out") or 0
            agg["token_count"] += shard.get("token_count") or 0
            source_agg["records_out"] += shard.get("records_out") or 0
            source_agg["token_count"] += shard.get("token_count") or 0
        if not shard.get("stats_json"):
            continue
        try:
            stats = json.loads(shard["stats_json"])
        except json.JSONDecodeError:
            continue
        agg["duplicates_removed_exact"] += stats.get("dedup_exact.removed", 0)
        agg["near_duplicates_annotated"] += stats.get("dedup_near.annotated", 0)
        agg["near_duplicates_dropped"] += stats.get("dedup_near.dropped", 0)
        agg["secrets_redacted"] += stats.get("secrets.redacted_total", 0)
        agg["pii_redacted"] += stats.get("secrets.pii_redacted_total", 0)
        agg["excluded_records"] += stats.get("excluded.total", 0)
        for key, value in stats.items():
            if key.startswith("excluded.") and key != "excluded.total":
                reason = key.split(".", 1)[1]
                agg["excluded_by_reason"][reason] = (
                    agg["excluded_by_reason"].get(reason, 0) + value
                )
            elif key.startswith("license."):
                bucket = key.split(".", 1)[1]
                if bucket != "records_out":
                    agg["licenses"][bucket] = agg["licenses"].get(bucket, 0) + value
            elif key.startswith("classify.layer."):
                layer = key.split(".", 2)[2]
                agg["layers"][layer] = agg["layers"].get(layer, 0) + value
    return agg


def render_status(manifest) -> str:
    agg = aggregate_global(manifest)
    lines = ["=== unified-code-corpus status ==="]
    lines.append("shards by state:")
    for state, count in sorted(agg["shards"]["by_state"].items()):
        lines.append(f"  {state:22s} {count}")
    lines.append(
        f"records: in={agg['records_in']:,} out={agg['records_out']:,} "
        f"tokens≈{agg['token_count']:,}"
    )
    lines.append(
        f"dedup: exact removed={agg['duplicates_removed_exact']:,} "
        f"near annotated={agg['near_duplicates_annotated']:,} "
        f"near dropped={agg['near_duplicates_dropped']:,}"
    )
    lines.append(
        f"security: secrets redacted={agg['secrets_redacted']:,} "
        f"pii redacted={agg['pii_redacted']:,}"
    )
    lines.append(f"excluded: {agg['excluded_records']:,}")
    for reason, count in sorted(agg["excluded_by_reason"].items()):
        lines.append(f"  {reason:30s} {count:,}")
    lines.append("per source:")
    for src, sagg in sorted(agg["sources"].items()):
        lines.append(
            f"  {src:28s} shards={sagg['shards_completed']} "
            f"records={sagg['records_out']:,} tokens≈{sagg['token_count']:,}"
        )
    return "\n".join(lines)


def _export_repos_parquet(manifest, out_path: Path) -> int:
    import pyarrow.parquet as pq

    writer = pq.ParquetWriter(str(out_path), REPO_SCHEMA, compression="zstd")
    count = 0
    batch: list[dict] = []
    try:
        for repo in manifest.iter_repos():
            batch.append(
                {
                    "repo_name": repo["repo_key"],
                    "repo_url": repo.get("repo_url"),
                    "n_files": repo.get("n_files", 0),
                    "n_tokens": repo.get("n_tokens", 0),
                    "n_commits": repo.get("n_commits", 0),
                    "n_issues": repo.get("n_issues", 0),
                    "n_deps": repo.get("n_deps", 0),
                    "languages": sorted((repo.get("languages") or {}).keys()),
                    "layers": sorted((repo.get("layers") or {}).keys()),
                    "technologies": repo.get("technologies") or [],
                    "licenses": repo.get("licenses") or [],
                    "has_tests": bool(repo.get("has_tests")),
                    "has_ci": bool(repo.get("has_ci")),
                    "has_infrastructure": bool(repo.get("has_infrastructure")),
                    "category": repo.get("category"),
                    "complexity": repo.get("complexity"),
                    "capture": "partial_stream_sample",
                }
            )
            count += 1
            if len(batch) >= 2000:
                writer.write_table(pa.Table.from_pylist(batch, schema=REPO_SCHEMA))
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=REPO_SCHEMA))
    finally:
        writer.close()
    return count


def _export_provenance_parquet(manifest, out_path: Path) -> int:
    import pyarrow.parquet as pq

    writer = pq.ParquetWriter(str(out_path), PROVENANCE_SCHEMA, compression="zstd")
    count = 0
    batch: list[dict] = []
    try:
        for sha, canonical, sources in manifest.iter_multi_source_hashes():
            batch.append(
                {
                    "content_sha256": sha,
                    "canonical_record_id": canonical,
                    "source_datasets": sources,
                }
            )
            count += 1
            if len(batch) >= 5000:
                writer.write_table(pa.Table.from_pylist(batch, schema=PROVENANCE_SCHEMA))
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=PROVENANCE_SCHEMA))
    finally:
        writer.close()
    return count


def _dataset_card(cfg: Cfg, agg: dict) -> str:
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    sources_md = "\n".join(
        f"| {src} | {s['shards_completed']} | {s['records_out']:,} "
        f"| {s['token_count']:,} |"
        for src, s in sorted(agg["sources"].items())
    )
    return f"""---
pretty_name: Unified Real-World Code Corpus
configs:
- config_name: full
  data_files: data/full/*/*.parquet
- config_name: high_quality
  data_files: data/high_quality/*/*.parquet
- config_name: commits
  data_files: data/commits/*/*.parquet
- config_name: issues
  data_files: data/issues/*/*.parquet
- config_name: repository_level
  data_files: repos/repositories.parquet
---

# Unified Real-World Code Corpus

Deduplicated, provenance-aware, license-aware, repository-level real-world
software engineering corpus, built by the unified-code-corpus pipeline
(v{cfg.pipeline_version}, config hash `{cfg.config_hash}`, updated {generated}).
100% real data from The Stack v2 (via Software Heritage), StarCoderData,
Common Pile Stack v2 (+ edu-filtered) and StarCoder2 extras — nothing
synthetic, nothing fabricated.

## Subsets

- **full** — all kept code/doc records. Near-duplicates are RETAINED here,
  annotated via `is_near_duplicate` / `near_dup_cluster` (duplicates are
  never hidden).
- **high_quality** — permissive-license, flag-free, near-dup-free records.
- **commits** / **issues** — real git-commit and GitHub-issue records
  (never near-deduplicated; history is preserved).
- **repository_level** — consolidated per-repository table (files, tokens,
  languages, layers, technologies, dependencies, tests/CI/infra presence,
  category, complexity 0–100). Streaming sources interleave repos across
  shards, so per-repo captures may be partial samples of the true repo.
- **frontend / backend / database / infrastructure / full_stack** views:
  filter `full` on the `layer` and `repo_category` columns.
- **excluded/** — audit report of every removed record (metadata + reason
  only, never content).
- **provenance/multi_source.parquet** — content found in more than one
  source dataset, with all sources preserved.

## Current statistics

records: {agg['records_out']:,} kept / {agg['records_in']:,} ingested ·
tokens ≈ {agg['token_count']:,} ·
exact duplicates removed: {agg['duplicates_removed_exact']:,} ·
near-duplicates annotated: {agg['near_duplicates_annotated']:,} ·
secrets redacted: {agg['secrets_redacted']:,} ·
excluded records: {agg['excluded_records']:,}

| source | shards | records | tokens (approx.) |
|---|---|---|---|
{sources_md}

## Licensing & provenance

Every record keeps its detected licenses (`detected_licenses`, normalized in
`license`) and a `license_status` bucket; strong-copyleft records are
excluded by default and unknown-license records are flagged. Retention of a
record here is NOT a redistribution grant — consult the upstream terms
(BigCode/The Stack v2, Software Heritage/Inria, Common Pile) before
redistributing. Secrets and obvious PII were redacted
(`secrets_redacted` / `pii_redacted` counters per record).
"""


def run_finalize(cfg: Cfg, manifest, hub) -> dict:
    """Export consolidated artifacts and upload them: global stats JSON,
    repositories.parquet, multi-source provenance, dataset card."""
    paths = workspace_paths(cfg)
    export_dir = ensure_dir(paths.workspace / "exports")
    agg = aggregate_global(manifest)

    repos_path = export_dir / "repositories.parquet"
    n_repos = _export_repos_parquet(manifest, repos_path)
    agg["repositories"] = n_repos

    prov_path = export_dir / "multi_source.parquet"
    n_prov = _export_provenance_parquet(manifest, prov_path)
    agg["multi_source_contents"] = n_prov

    hub.upload_json(agg, "stats/global.json", message="ucc: global statistics")
    hub.upload_file(repos_path, "repos/repositories.parquet",
                    message="ucc: consolidated repository table")
    if n_prov:
        hub.upload_file(prov_path, "provenance/multi_source.parquet",
                        message="ucc: cross-source provenance")
    card_path = export_dir / "README.md"
    card_path.write_text(_dataset_card(cfg, agg), encoding="utf-8")
    hub.upload_file(card_path, "README.md", message="ucc: dataset card")
    log.info("finalize: %d repos, %d multi-source contents exported", n_repos, n_prov)
    return agg
