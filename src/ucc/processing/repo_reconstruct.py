"""Stage 5 — repository reconstruction.

Files are not isolated samples: this stage
- orders the shard's records so each repository's files are contiguous
  (repo, path) — repository-level layout in the output parquet;
- computes per-repo aggregates (files, tokens, languages, layers,
  technologies, licenses, tests/CI/infra presence, dependency counts,
  commit/issue counts) and merges them EXACTLY ONCE into the manifest's
  cross-shard repos table (idempotent via a per-shard applied-flag written
  in the same transaction);
- computes an in-shard interconnectedness estimate (files importing sibling
  modules) used by the complexity stage.

Streaming caveat: sources like Common Pile interleave repositories across
shards, so a shard sees a PARTIAL capture of most repos. Cross-shard
consolidation lives in the repos table and is exported by `finalize`.
"""

from __future__ import annotations

import posixpath
import re

from ucc import constants as C
from ucc.processing.base import ShardContext, Stage
from ucc.processing.classify import (
    detect_technologies,
    file_layer,
    infer_language,
)

_DEP_PATTERNS = {
    "package.json": re.compile(r'"[^"\n]+"\s*:\s*"[~^><=0-9]'),
    "requirements.txt": re.compile(r"^\s*[A-Za-z0-9_.\-]+\s*(?:[=<>~!\[]|$)", re.M),
    "pyproject.toml": re.compile(r"^\s*[A-Za-z0-9_.\-]+\s*=\s*[\"'{\[]", re.M),
    "go.mod": re.compile(r"^\s*(?:require\s+)?[\w.\-/]+\s+v[\d.]", re.M),
    "cargo.toml": re.compile(r"^\s*[A-Za-z0-9_\-]+\s*=\s*[\"{]", re.M),
    "pom.xml": re.compile(r"<dependency>", re.I),
    "build.gradle": re.compile(r"\b(?:implementation|api|compile)\b"),
    "gemfile": re.compile(r"^\s*gem\s+['\"]", re.M),
    "composer.json": re.compile(r'"[^"\n]+"\s*:\s*"[~^><=0-9]'),
}

_IMPORT_LINE_RX = re.compile(
    r"^\s*(?:import|from|require|include|use)\b.*$", re.MULTILINE
)


def _count_dependencies(path: str | None, content: str) -> int:
    if not path:
        return 0
    base = posixpath.basename(path).lower()
    pattern = _DEP_PATTERNS.get(base)
    if pattern is None:
        return 0
    return len(pattern.findall(content[:100_000]))


class RepoReconstructStage(Stage):
    name = "repo_reconstruct"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        # Repository-level ordering: files of one repo are contiguous.
        rows.sort(
            key=lambda r: (r.get("repo_name") or "￿", r.get("path") or "", r["id"])
        )

        deltas: dict[str, dict] = {}
        stems_by_repo: dict[str, set[str]] = {}
        rows_by_repo: dict[str, list[dict]] = {}

        for rec in rows:
            repo = rec.get("repo_name")
            if not repo:
                ctx.bump("repo.records_without_repo")
                continue
            rows_by_repo.setdefault(repo, []).append(rec)

            language = infer_language(rec.get("path"), rec.get("language"))
            head = (rec.get("content") or "")[:4000]
            layer = file_layer(rec.get("path"), language, head)
            techs = detect_technologies(rec.get("path"), head)

            delta = deltas.setdefault(
                repo,
                {
                    "repo_url": rec.get("repo_url"),
                    "n_files": 0, "n_tokens": 0, "n_commits": 0, "n_issues": 0,
                    "n_deps": 0, "languages": {}, "layers": {},
                    "technologies": [], "licenses": [],
                    "has_tests": False, "has_ci": False, "has_infrastructure": False,
                },
            )
            delta["n_tokens"] += rec.get("token_count") or 0
            if rec["record_type"] == C.RT_COMMIT:
                delta["n_commits"] += 1
            elif rec["record_type"] == C.RT_ISSUE:
                delta["n_issues"] += 1
            else:
                delta["n_files"] += 1
            if language:
                delta["languages"][language] = delta["languages"].get(language, 0) + 1
            delta["layers"][layer] = delta["layers"].get(layer, 0) + 1
            delta["technologies"] = sorted(set(delta["technologies"]) | set(techs))
            delta["licenses"] = sorted(
                set(delta["licenses"]) | {str(x) for x in rec.get("detected_licenses") or []}
            )
            if layer == C.LAYER_TESTING:
                delta["has_tests"] = True
            if "ci_cd" in techs:
                delta["has_ci"] = True
            if layer == C.LAYER_INFRA:
                delta["has_infrastructure"] = True
            delta["n_deps"] += _count_dependencies(rec.get("path"), rec.get("content") or "")

            if rec.get("path"):
                stem = posixpath.splitext(posixpath.basename(rec["path"]))[0]
                if len(stem) >= 3:
                    stems_by_repo.setdefault(repo, set()).add(stem.lower())

        # Interconnectedness: share of a repo's in-shard files whose import
        # lines mention a sibling file's module name.
        interconnect: dict[str, float] = {}
        for repo, repo_rows in rows_by_repo.items():
            stems = stems_by_repo.get(repo) or set()
            if len(repo_rows) < 2 or not stems:
                interconnect[repo] = 0.0
                continue
            linked = 0
            for rec in repo_rows:
                own_stem = (
                    posixpath.splitext(posixpath.basename(rec["path"]))[0].lower()
                    if rec.get("path")
                    else ""
                )
                imports = " ".join(
                    _IMPORT_LINE_RX.findall((rec.get("content") or "")[:20_000])
                ).lower()
                if any(s in imports for s in stems if s != own_stem):
                    linked += 1
            interconnect[repo] = round(linked / len(repo_rows), 4)
        ctx.scratch["repo_interconnect"] = interconnect

        # Exactly-once key is batch-scoped in per-batch mode so every batch's
        # deltas merge once and a crashed batch re-run stays a no-op.
        delta_key = ctx.scratch.get("repo_delta_key") or ctx.shard["shard_id"]
        applied = ctx.manifest.apply_repo_deltas(delta_key, deltas)
        if not applied:
            ctx.log.info("repo aggregates for %s already merged (resume) — skipped", delta_key)
        ctx.bump("repo.repos_in_shard", len(deltas))
        ctx.bump("repo.records_out", len(rows))
        return rows
