"""Stage 2 — provenance: enforce that every record carries usable provenance.

Provenance is preserved from the sources (repository, path, commit id,
license list, source dataset + shard); this stage validates it and drops
records that cannot be attributed at all.
"""

from __future__ import annotations

from ucc.processing.base import ShardContext, Stage


class ProvenanceStage(Stage):
    name = "provenance"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        out: list[dict] = []
        for rec in rows:
            if not rec.get("content") or not rec.get("content_sha256"):
                ctx.exclude(rec, "no_content")
                continue
            if not rec.get("source_dataset") or not rec.get("source_shard"):
                ctx.exclude(rec, "no_provenance")
                continue
            # Repo-less records are allowed only for non-code types (some
            # issue/doc sources have no repo attribution).
            if rec.get("record_type") == "code" and not rec.get("repo_name"):
                ctx.exclude(rec, "code_without_repo")
                continue
            licenses = rec.get("detected_licenses") or []
            rec["detected_licenses"] = [str(x) for x in licenses if x]
            repo = rec.get("repo_name")
            if repo and not rec.get("repo_url") and "/" in str(repo):
                rec["repo_url"] = f"https://github.com/{repo}"
            out.append(rec)
        ctx.bump("provenance.records_out", len(out))
        return out
