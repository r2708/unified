"""Language / layer / technology / repository-category classification.

The helper functions here are also used earlier in the pipeline
(repo_reconstruct, complexity); the ClassifyStage itself runs last and stamps
the final fields onto every record and the repos table.
"""

from __future__ import annotations

import posixpath
import re

from ucc import constants as C
from ucc.processing.base import ShardContext, Stage

# ----------------------------------------------------------- language by ext
EXT_LANGUAGE = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".vue": "Vue", ".svelte": "Svelte",
    ".java": "Java", ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".hpp": "C++",
    ".cs": "C#", ".kt": "Kotlin", ".swift": "Swift", ".scala": "Scala",
    ".sql": "SQL", ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".ps1": "PowerShell", ".yaml": "YAML", ".yml": "YAML", ".json": "JSON",
    ".toml": "TOML", ".ini": "INI", ".cfg": "INI", ".xml": "XML",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".less": "Less", ".md": "Markdown", ".rst": "reStructuredText",
    ".tf": "HCL", ".tfvars": "HCL", ".hcl": "HCL", ".proto": "Protocol Buffer",
    ".lua": "Lua", ".r": "R", ".pl": "Perl", ".ex": "Elixir", ".exs": "Elixir",
    ".erl": "Erlang", ".hs": "Haskell", ".jl": "Julia", ".dart": "Dart",
    ".groovy": "Groovy", ".gradle": "Groovy", ".ipynb": "Jupyter Notebook",
    ".dockerfile": "Dockerfile",
}

_FRONTEND_LANGS = {"JavaScript", "TypeScript", "Vue", "Svelte", "HTML", "CSS", "SCSS", "Less"}
_BACKEND_LANGS = {"Python", "Java", "Go", "Rust", "Ruby", "PHP", "C#", "Kotlin",
                  "Scala", "Elixir", "Erlang", "Haskell", "C", "C++", "Perl"}
_CONFIG_LANGS = {"YAML", "JSON", "TOML", "INI", "XML"}

_TEST_PATH_RX = re.compile(
    r"(^|/)(tests?|__tests__|spec|specs)(/|$)|_test\.[a-z]+$|\.test\.[a-z]+$"
    r"|\.spec\.[a-z]+$|(^|/)test_[^/]+$",
    re.IGNORECASE,
)
_CI_PATH_RX = re.compile(
    r"(^|/)\.github/workflows/|(^|/)\.gitlab-ci\.ya?ml$|(^|/)Jenkinsfile"
    r"|(^|/)\.circleci/|(^|/)\.travis\.ya?ml$|(^|/)azure-pipelines\.ya?ml$"
    r"|(^|/)bitbucket-pipelines\.ya?ml$",
    re.IGNORECASE,
)
_INFRA_PATH_RX = re.compile(
    r"(^|/)(Dockerfile[^/]*|docker-compose[^/]*\.ya?ml|Makefile|nginx[^/]*\.conf)$"
    r"|\.(tf|tfvars)$|(^|/)(k8s|kubernetes|helm|charts|terraform|ansible)(/|$)"
    r"|(^|/)(Vagrantfile|Procfile)$",
    re.IGNORECASE,
)
_DB_PATH_RX = re.compile(
    r"(^|/)(migrations?|alembic|db/migrate)(/|$)|\.sql$", re.IGNORECASE
)
_MANIFEST_BASENAMES = {
    "package.json", "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg",
    "go.mod", "cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts",
    "gemfile", "composer.json", "mix.exs", "pubspec.yaml", "environment.yml",
    "pipfile",
}

_K8S_CONTENT_RX = re.compile(r"^\s*apiVersion\s*:.*$\n(?:.*\n)*?^\s*kind\s*:", re.MULTILINE)

# ---------------------------------------------------- technology detectors
# (name, path regex or None, content regex or None) — a hit on either counts.
_TECH_DETECTORS: list[tuple[str, re.Pattern | None, re.Pattern | None]] = [
    ("react", None, re.compile(r"""(from\s+['"]react['"]|require\(['"]react['"]\))""")),
    ("nextjs", re.compile(r"(^|/)next\.config\.[a-z]+$"),
     re.compile(r"""from\s+['"]next(/|['"])""")),
    ("vue", re.compile(r"\.vue$"), re.compile(r"""from\s+['"]vue['"]""")),
    ("angular", None, re.compile(r"""from\s+['"]@angular/""")),
    ("svelte", re.compile(r"\.svelte$"), None),
    ("express", None, re.compile(r"""require\(['"]express['"]\)|from\s+['"]express['"]""")),
    ("nodejs", re.compile(r"(^|/)package\.json$"), None),
    ("django", None, re.compile(r"\bfrom\s+django|\bimport\s+django\b")),
    ("flask", None, re.compile(r"\bfrom\s+flask\s+import|\bimport\s+flask\b")),
    ("fastapi", None, re.compile(r"\bfrom\s+fastapi\s+import|\bimport\s+fastapi\b")),
    ("spring", None, re.compile(r"\borg\.springframework\b")),
    ("rails", re.compile(r"(^|/)config/routes\.rb$"), re.compile(r"\bRails\.application\b")),
    ("postgresql", None, re.compile(r"\bpostgres(ql)?://|\bpsycopg2?\b|\bpg_dump\b", re.I)),
    ("mysql", None, re.compile(r"\bmysql(://|\.connector)|\bpymysql\b", re.I)),
    ("mongodb", None, re.compile(r"\bmongodb(\+srv)?://|\bmongoose\b|\bpymongo\b", re.I)),
    ("redis", None, re.compile(r"\bredis://|\bimport\s+redis\b|\bioredis\b", re.I)),
    ("sqlite", None, re.compile(r"\bsqlite3?\b", re.I)),
    ("docker", re.compile(r"(^|/)(Dockerfile[^/]*|docker-compose[^/]*\.ya?ml|\.dockerignore)$", re.I), None),
    ("kubernetes", re.compile(r"(^|/)(k8s|kubernetes|helm|charts)(/|$)", re.I), _K8S_CONTENT_RX),
    ("terraform", re.compile(r"\.(tf|tfvars)$"), None),
    ("nginx", re.compile(r"(^|/)nginx[^/]*\.conf$", re.I),
     re.compile(r"^\s*server\s*\{(?:.*\n)*?\s*listen\s+\d+", re.MULTILINE)),
    ("ci_cd", _CI_PATH_RX, None),
    ("graphql", re.compile(r"\.(graphql|gql)$"), re.compile(r"\bgraphql\b", re.I)),
    ("grpc", re.compile(r"\.proto$"), None),
    ("kafka", None, re.compile(r"\bkafka\b", re.I)),
    ("rabbitmq", None, re.compile(r"\bamqp://|\brabbitmq\b", re.I)),
    ("elasticsearch", None, re.compile(r"\belasticsearch\b", re.I)),
    ("pytorch", None, re.compile(r"\bimport\s+torch\b|\bfrom\s+torch\b")),
    ("tensorflow", None, re.compile(r"\bimport\s+tensorflow\b|\bfrom\s+tensorflow\b")),
    ("sklearn", None, re.compile(r"\bfrom\s+sklearn\b|\bimport\s+sklearn\b")),
    ("pandas", None, re.compile(r"\bimport\s+pandas\b|\bfrom\s+pandas\b")),
]

_ML_TECHS = {"pytorch", "tensorflow", "sklearn", "pandas"}
_DB_TECHS = {"postgresql", "mysql", "mongodb", "redis", "sqlite"}
_INFRA_TECHS = {"docker", "kubernetes", "terraform", "nginx", "ci_cd"}
_FRONTEND_TECHS = {"react", "nextjs", "vue", "angular", "svelte"}
_BACKEND_TECHS = {"express", "django", "flask", "fastapi", "spring", "rails", "nodejs"}

_CLI_CONTENT_RX = re.compile(
    r"\bimport\s+argparse\b|\bimport\s+click\b|\bfrom\s+click\b|\bcobra\.Command\b"
    r"|\byargs\b|\bcommander\b|\bclap::"
)


def infer_language(path: str | None, declared: str | None) -> str | None:
    if declared:
        return declared
    if not path:
        return None
    base = posixpath.basename(path).lower()
    if base.startswith("dockerfile"):
        return "Dockerfile"
    if base == "makefile":
        return "Makefile"
    _, ext = posixpath.splitext(base)
    return EXT_LANGUAGE.get(ext)


def file_layer(path: str | None, language: str | None, content_head: str = "") -> str:
    p = path or ""
    if p and _TEST_PATH_RX.search(p):
        return C.LAYER_TESTING
    if p and (_CI_PATH_RX.search(p) or _INFRA_PATH_RX.search(p)):
        return C.LAYER_INFRA
    if p and _DB_PATH_RX.search(p):
        return C.LAYER_DATABASE
    if language == "SQL":
        return C.LAYER_DATABASE
    if language in ("Dockerfile", "HCL", "Makefile"):
        return C.LAYER_INFRA
    if p and posixpath.basename(p).lower() in _MANIFEST_BASENAMES:
        return C.LAYER_CONFIG
    if language in _CONFIG_LANGS:
        if language == "YAML" and content_head and _K8S_CONTENT_RX.search(content_head):
            return C.LAYER_INFRA
        return C.LAYER_CONFIG
    if language in ("Markdown", "reStructuredText"):
        return C.LAYER_DOCS
    if language in _FRONTEND_LANGS:
        return C.LAYER_FRONTEND
    if language in _BACKEND_LANGS:
        return C.LAYER_BACKEND
    return C.LAYER_OTHER


def detect_technologies(path: str | None, content_head: str) -> list[str]:
    found: list[str] = []
    p = path or ""
    for name, path_rx, content_rx in _TECH_DETECTORS:
        if path_rx is not None and p and path_rx.search(p):
            found.append(name)
            continue
        if content_rx is not None and content_head and content_rx.search(content_head):
            found.append(name)
    return sorted(set(found))


def layer_and_technologies(
    item: tuple[str | None, str | None, str],
) -> tuple[str, list[str]]:
    """Pure per-record classification (process-pool worker): item is
    (path, language, content head). The ~30-regex tech/layer pass dominates
    repo_reconstruct, so it is farmed out via processing.cpu_workers."""
    path, language, head = item
    return file_layer(path, language, head), detect_technologies(path, head)


def repo_category(agg: dict) -> str:
    """Categorize a repository from its (cross-shard) aggregates."""
    layers: dict[str, int] = agg.get("layers") or {}
    techs = set(agg.get("technologies") or [])
    total = sum(layers.values()) or 1

    def share(layer: str) -> float:
        return layers.get(layer, 0) / total

    fe = share(C.LAYER_FRONTEND) + (0.15 if techs & _FRONTEND_TECHS else 0)
    be = share(C.LAYER_BACKEND) + (0.15 if techs & _BACKEND_TECHS else 0)
    db = share(C.LAYER_DATABASE) + (0.15 if techs & _DB_TECHS else 0)
    infra = share(C.LAYER_INFRA) + (0.15 if techs & _INFRA_TECHS else 0)

    if infra >= 0.6:
        return "infrastructure"
    if db >= 0.5:
        return "database"
    if techs & _ML_TECHS and (techs & _ML_TECHS or share(C.LAYER_BACKEND) > 0):
        if len(techs & _ML_TECHS) >= 2 or share(C.LAYER_BACKEND) >= 0.3:
            return "ml_data"
    if fe >= 0.2 and be >= 0.2:
        return "full_stack"
    if fe >= 0.5:
        return "frontend"
    if be >= 0.5:
        if agg.get("is_cli"):
            return "cli"
        if agg.get("is_library"):
            return "library"
        return "backend"
    if agg.get("is_cli"):
        return "cli"
    if agg.get("is_library"):
        return "library"
    return "mixed"


def looks_like_cli(content_head: str) -> bool:
    return bool(_CLI_CONTENT_RX.search(content_head))


class ClassifyStage(Stage):
    name = "classify"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        # Per-record layer + technologies (heads only — cheap and robust).
        # repo_reconstruct already classified each record and cached the
        # results on it; reuse them unless the secrets stage redacted the
        # content in between (the caches are head-derived) or a checkpoint
        # resume dropped the underscore keys from the rows.
        repo_seen: dict[str, dict] = {}
        live = ctx.progress("classify", total=len(rows))
        for rec in rows:
            live.update()
            head = (rec.get("content") or "")[:4000]
            content_unchanged = (
                not rec.get("secrets_redacted") and not rec.get("pii_redacted")
            )
            cached_language = rec.pop("_ucc_language", None)
            cached_layer = rec.pop("_ucc_layer", None)
            cached_techs = rec.pop("_ucc_technologies", None)

            language = cached_language or infer_language(
                rec.get("path"), rec.get("language")
            )
            rec["language"] = language
            rec["layer"] = (
                cached_layer
                if cached_layer is not None and content_unchanged
                else file_layer(rec.get("path"), language, head)
            )
            rec["technologies"] = (
                cached_techs
                if cached_techs is not None and content_unchanged
                else detect_technologies(rec.get("path"), head)
            )
            repo = rec.get("repo_name")
            if repo:
                info = repo_seen.setdefault(repo, {"cli": False})
                if not info["cli"] and looks_like_cli(head):
                    info["cli"] = True
        live.close()

        # Repository category from cross-shard aggregates in the manifest
        # (merged by repo_reconstruct, including this shard's contribution).
        # One batched read + one batched write instead of two statements (and
        # one fsync) per distinct repo.
        aggs = ctx.manifest.get_repos(sorted(repo_seen))
        categories: dict[str, str] = {}
        updates: list[tuple[str, float | None, str | None]] = []
        for repo, info in repo_seen.items():
            agg = aggs.get(repo) or {}
            agg["is_cli"] = info["cli"]
            layers_hist = agg.get("layers") or {}
            agg["is_library"] = (
                layers_hist.get(C.LAYER_CONFIG, 0) > 0
                and not agg.get("has_infrastructure")
                and 0 < agg.get("n_files", 0) <= 50
            )
            category = repo_category(agg)
            categories[repo] = category
            updates.append((repo, None, category))
        ctx.manifest.update_repo_computed_many(updates)

        for rec in rows:
            repo = rec.get("repo_name")
            if repo:
                rec["repo_category"] = categories.get(repo)

        hist: dict[str, int] = {}
        for rec in rows:
            hist[rec["layer"]] = hist.get(rec["layer"], 0) + 1
        for layer, count in hist.items():
            ctx.bump(f"classify.layer.{layer}", count)
        ctx.bump("classify.records_out", len(rows))
        return rows
