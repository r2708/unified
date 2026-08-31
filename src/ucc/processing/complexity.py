"""Stage 9 — complexity metrics and filtering.

File-level: an approximate cyclomatic count (1 + branch keywords/operators).
Repository-level: a 0–100 score computed from ACTUAL repository data held in
the cross-shard aggregates — files, tokens, languages, dependencies,
software layers, tests, infrastructure, configuration, commits/issues, and
the in-shard interconnectedness estimate from repo_reconstruct:

    score = 14*log10(files+1) + 4*log10(tokens+1) + 6*min(langs,6)
          + 5*layers_present + 6*has_tests + 5*has_infra + 4*has_ci
          + 6*log10(deps+1) + 6*log10(commits+issues+1) + 10*interconnect
    (capped at 100)

Trivial records (token_count below complexity.min_file_tokens) are dropped
when configured, recorded in the excluded report.
"""

from __future__ import annotations

import math
import re

from ucc.constants import RT_CODE
from ucc.processing.base import ShardContext, Stage

_BRANCH_RX = re.compile(
    r"\b(if|elif|else if|for|while|case|when|catch|except|match)\b|&&|\|\||\?\s*:"
)


def file_complexity(content: str) -> float:
    return float(1 + len(_BRANCH_RX.findall(content[:200_000])))


def repo_complexity_score(agg: dict, interconnect: float) -> float:
    n_files = agg.get("n_files", 0)
    n_tokens = agg.get("n_tokens", 0)
    n_langs = len(agg.get("languages") or {})
    n_deps = agg.get("n_deps", 0)
    n_hist = agg.get("n_commits", 0) + agg.get("n_issues", 0)
    layers_present = len(agg.get("layers") or {})
    score = (
        14.0 * math.log10(n_files + 1)
        + 4.0 * math.log10(n_tokens + 1)
        + 6.0 * min(n_langs, 6)
        + 5.0 * min(layers_present, 6)
        + (6.0 if agg.get("has_tests") else 0.0)
        + (5.0 if agg.get("has_infrastructure") else 0.0)
        + (4.0 if agg.get("has_ci") else 0.0)
        + 6.0 * math.log10(n_deps + 1)
        + 6.0 * math.log10(n_hist + 1)
        + 10.0 * max(0.0, min(interconnect, 1.0))
    )
    return round(min(score, 100.0), 2)


class ComplexityStage(Stage):
    name = "complexity"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        ccfg = ctx.cfg.complexity
        min_tokens = int(ccfg.get("min_file_tokens", 4))
        drop_trivial = bool(ccfg.get("drop_below_min_tokens", True))
        interconnect: dict[str, float] = ctx.scratch.get("repo_interconnect") or {}

        out: list[dict] = []
        repo_scores: dict[str, float] = {}
        for rec in rows:
            if (
                drop_trivial
                and rec["record_type"] == RT_CODE
                and rec["token_count"] < min_tokens
            ):
                ctx.exclude(rec, "trivial", detail=f"{rec['token_count']} tokens")
                continue
            rec["file_complexity"] = file_complexity(rec.get("content") or "")

            repo = rec.get("repo_name")
            if repo:
                if repo not in repo_scores:
                    agg = ctx.manifest.get_repo(repo) or {}
                    repo_scores[repo] = repo_complexity_score(
                        agg, interconnect.get(repo, 0.0)
                    )
                    ctx.manifest.update_repo_computed(repo, complexity=repo_scores[repo])
                rec["repo_complexity"] = repo_scores[repo]
            out.append(rec)

        ctx.bump("complexity.records_out", len(out))
        return out
