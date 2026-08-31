"""bigcode/starcoderdata adapter (gated=auto — token with accepted terms
required for downloads; the file tree itself is public).

Layout: one directory per language ({lang}/train-*.parquet) with the content
inline, plus special directories:
  git-commits-cleaned                real commit history samples
  github-issues-filtered-structured  real GitHub issue threads
  jupyter-scripts-dedup-filtered / jupyter-structured-clean-dedup

Code rows carry max_stars_repo_name / max_stars_repo_path /
max_stars_repo_licenses / max_stars_count / content / id. The dataset is
already filtered to permissive licenses upstream, so records without an
explicit license list fall back to source-level permissive handling (see
licenses.source_level_permissive in the config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ucc.constants import RT_CODE, RT_COMMIT, RT_ISSUE
from ucc.logging_utils import get_logger
from ucc.schema import new_record
from ucc.sources.base import (
    ShardSpec,
    SourceAdapter,
    group_files_into_units,
    hf_list_files,
    pinned_revision,
)
from ucc.sources.hf_download import download_hf_files
from ucc.sources.readers import iter_file_batches

log = get_logger("sources.starcoderdata")

_DIR_RECORD_TYPE = {
    "git-commits-cleaned": RT_COMMIT,
    "github-issues-filtered-structured": RT_ISSUE,
}

_DIR_LANGUAGE = {
    "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
    "sql": "SQL", "go": "Go", "rust": "Rust", "java": "Java", "c": "C",
    "cpp": "C++", "c-sharp": "C#", "php": "PHP", "ruby": "Ruby",
    "kotlin": "Kotlin", "swift": "Swift", "scala": "Scala", "shell": "Shell",
    "dockerfile": "Dockerfile", "yaml": "YAML", "json": "JSON",
    "markdown": "Markdown", "html": "HTML", "css": "CSS", "lua": "Lua",
    "perl": "Perl", "r": "R", "tex": "TeX", "makefile": "Makefile",
    "cmake": "CMake", "haskell": "Haskell", "julia": "Julia", "dart": "Dart",
    "elixir": "Elixir", "erlang": "Erlang", "fortran": "Fortran",
    "groovy": "Groovy", "hcl": "HCL", "powershell": "PowerShell",
    "protocol-buffer": "Protocol Buffer", "restructuredtext": "reStructuredText",
    "jupyter-scripts-dedup-filtered": "Python",
    "jupyter-structured-clean-dedup": "Jupyter Notebook",
}


class StarCoderDataAdapter(SourceAdapter):
    requires_token = True

    def _dirs(self) -> list[str]:
        return self.source_cfg.get("include_dirs") or ["python", "sql", "dockerfile"]

    def enumerate_shards(self) -> Iterator[ShardSpec]:
        revision = pinned_revision(self.repo_id, self.token)
        seq = 0
        cap = self.max_shards()
        for dir_name in self._dirs():
            files = [
                (path, size)
                for path, size in hf_list_files(self.repo_id, dir_name, revision, self.token)
                if path.endswith(".parquet")
            ]
            log.info("%s/%s: %d parquet files", self.name, dir_name, len(files))
            units = group_files_into_units(
                files, self.cfg.shard.target_bytes, self.cfg.shard.max_bytes
            )
            record_type = _DIR_RECORD_TYPE.get(dir_name, RT_CODE)
            language = _DIR_LANGUAGE.get(dir_name, dir_name.replace("-", " ").title())
            for unit in units:
                if cap is not None and seq >= cap:
                    return
                yield ShardSpec(
                    shard_id=self.shard_id(seq),
                    source=self.name,
                    seq_index=seq,
                    ref={
                        "repo_id": self.repo_id,
                        "revision": revision,
                        "dir": dir_name,
                        "record_type": record_type,
                        "language": None if record_type != RT_CODE else language,
                        "files": [[path, size] for path, size in unit],
                    },
                    est_bytes=sum(size for _, size in unit),
                    record_type_hint=record_type,
                )
                seq += 1

    def download(self, spec_ref: dict, dest_dir: Path, stop_check=None) -> None:
        download_hf_files(
            spec_ref["repo_id"],
            [(path, size) for path, size in spec_ref["files"]],
            spec_ref.get("revision"),
            dest_dir,
            self.token,
            stop_check,
        )

    def iter_raw_batches(self, spec_ref: dict, raw_dir: Path, batch_size: int,
                         skip_records: int = 0) -> Iterator[list[dict]]:
        skip = [max(int(skip_records), 0)]
        for path, _size in spec_ref["files"]:
            local = raw_dir / path
            if not local.exists():
                raise FileNotFoundError(f"raw file missing: {local}")
            yield from iter_file_batches(local, batch_size, skip=skip)

    # ------------------------------------------------------------ normalize
    def normalize_record(self, raw: dict, spec_ref: dict) -> dict | None:
        record_type = spec_ref.get("record_type", RT_CODE)
        if record_type == RT_COMMIT:
            return self._normalize_commit(raw)
        if record_type == RT_ISSUE:
            return self._normalize_issue(raw)
        return self._normalize_code(raw, spec_ref)

    def _normalize_code(self, raw: dict, spec_ref: dict) -> dict | None:
        content = raw.get("content")
        if not isinstance(content, str) or not content:
            return None
        repo = raw.get("max_stars_repo_name") or raw.get("repo_name") or None
        path = raw.get("max_stars_repo_path") or raw.get("path") or None
        licenses = raw.get("max_stars_repo_licenses") or []
        if isinstance(licenses, str):
            licenses = [licenses]
        stars = raw.get("max_stars_count")
        try:
            stars = int(stars) if stars is not None else None
        except (TypeError, ValueError):
            stars = None
        return new_record(
            record_type=RT_CODE,
            content=content,
            repo_name=str(repo) if repo else None,
            repo_url=f"https://github.com/{repo}" if repo and "/" in str(repo) else None,
            path=str(path) if path else None,
            language=spec_ref.get("language"),
            detected_licenses=[str(x) for x in licenses],
            stars=stars,
            source_record_id=str(raw.get("id")) if raw.get("id") is not None else None,
        )

    def _normalize_commit(self, raw: dict) -> dict | None:
        """git-commits-cleaned rows: commit, old_file, new_file, old_contents,
        new_contents, subject, message, lang, license, repos.

        The output content is a deterministic serialization of those REAL
        fields (subject/message + before/after file contents) — never
        synthesized history."""
        new_contents = raw.get("new_contents")
        message = raw.get("message") or raw.get("subject")
        if not isinstance(new_contents, str) and not isinstance(message, str):
            return None
        repos = raw.get("repos") or ""
        repo = str(repos).split(",")[0].strip() or None
        commit_id = raw.get("commit")

        parts: list[str] = []
        if commit_id:
            parts.append(f"commit {commit_id}")
        if repo:
            parts.append(f"repo: {repo}")
        if raw.get("subject"):
            parts.append(f"subject: {raw['subject']}")
        if raw.get("message") and raw.get("message") != raw.get("subject"):
            parts.append(str(raw["message"]))
        old_file, new_file = raw.get("old_file"), raw.get("new_file")
        if isinstance(raw.get("old_contents"), str):
            parts.append(f"--- a/{old_file or new_file or ''}\n{raw['old_contents']}")
        if isinstance(new_contents, str):
            parts.append(f"+++ b/{new_file or old_file or ''}\n{new_contents}")
        content = "\n\n".join(parts)
        if not content:
            return None

        license_field = raw.get("license")
        return new_record(
            record_type=RT_COMMIT,
            content=content,
            repo_name=repo,
            repo_url=f"https://github.com/{repo}" if repo and "/" in repo else None,
            path=str(new_file or old_file) if (new_file or old_file) else None,
            language=str(raw.get("lang")) if raw.get("lang") else None,
            detected_licenses=[str(license_field)] if license_field else [],
            commit_id=str(commit_id) if commit_id else None,
        )

    def _normalize_issue(self, raw: dict) -> dict | None:
        content = raw.get("content") or raw.get("text")
        if not isinstance(content, str) or not content:
            return None
        repo = raw.get("repo") or raw.get("repo_name") or None
        return new_record(
            record_type=RT_ISSUE,
            content=content,
            repo_name=str(repo) if repo else None,
            repo_url=f"https://github.com/{repo}" if repo and "/" in str(repo) else None,
            source_record_id=str(raw.get("issue_id", raw.get("id")))
            if raw.get("issue_id", raw.get("id")) is not None
            else None,
        )
