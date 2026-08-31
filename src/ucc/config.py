"""Configuration loading, defaults, env overrides and config hashing.

The config hash covers every setting that changes *record content or record
selection* (sources, dedup, licenses, secrets, quality, complexity, shard
sizing, pipeline version). Paths, queue sizes, retry limits, HF repo names and
prototype caps (max_shards) are excluded so that operational tweaks do not
invalidate already-processed shards.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from ucc import constants


class Cfg(dict):
    """dict with attribute access and dotted-path lookup."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Cfg(value) if isinstance(value, dict) else value

    def path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


DEFAULTS: dict[str, Any] = {
    "pipeline_version": constants.PIPELINE_VERSION,
    "paths": {
        "workspace": "./workspace",
    },
    "hf": {
        "target_repo": "CHANGE-ME/unified-real-world-code",
        "private": True,
        "mode": "real",          # real | mock  (mock = local dir, for crash tests)
        "mock_dir": None,         # defaults to <workspace>/mock_hub
        "upload_excluded_reports": True,
        "max_retries": 5,
    },
    "queue": {
        "max_local_shards": constants.DEFAULT_MAX_LOCAL_SHARDS,
        "download_workers": 3,
        "process_workers": 1,
        "max_retries": 4,
        "retry_backoff_base_s": 30,
        "retry_backoff_max_s": 3600,
        "min_free_disk_gb": 20,
        "status_interval_s": 30,
        "progress_log_interval_s": 5,
    },
    "shard": {
        "target_bytes": constants.DEFAULT_SHARD_TARGET_BYTES,
        "min_bytes": constants.DEFAULT_SHARD_MIN_BYTES,
        "max_bytes": constants.DEFAULT_SHARD_MAX_BYTES,
    },
    "processing": {
        "batch_size": 2048,
        # True: every processed batch of batch_size records is uploaded to
        # the Hub (and verified) as its own parquet the moment its stages
        # finish; the batch is also the crash-resume unit.
        "upload_per_batch": False,
        # Stages after which a full intermediate parquet checkpoint is
        # written (stage-level resume). All global side effects are
        # idempotent, so re-running a stage after a crash is always safe;
        # checkpoints only save repeated work.
        "checkpoint_after": ["normalize", "dedup_near", "secrets_scan"],
        "parquet_compression": "zstd",
        "parquet_row_group_size": 2048,
        "token_counter": "heuristic",  # heuristic | tiktoken
    },
    "dedup": {
        "exact": {"enabled": True},
        "near": {
            "enabled": True,
            # annotate: keep near-dups in `full` with is_near_duplicate=true
            #           and exclude them from high_quality (spec: do not hide
            #           duplicates in the full subset).
            # drop:     remove them entirely (recorded in excluded report).
            "mode": "annotate",
            "num_perm": 128,
            "bands": 16,        # rows_per_band = num_perm / bands
            "jaccard_threshold": 0.75,
            "shingle_size": 5,
            "min_tokens": 30,
            # commits/issues are historical records — never near-deduped.
            "exempt_record_types": ["commit", "issue"],
        },
    },
    "licenses": {
        # Action per bucket: keep | flag | drop
        "permissive_action": "keep",
        "weak_copyleft_action": "flag",
        "strong_copyleft_action": "drop",
        "unknown_action": "flag",
        # Sources whose records are already license-filtered upstream; a
        # missing per-record license there is treated as source-level
        # permissive instead of unknown.
        "source_level_permissive": ["starcoderdata", "starcoder2data_extras"],
    },
    "secrets": {
        "enabled": True,
        "redact_emails": True,
        "redact_ips": True,
        "max_hits_per_file": 20,       # more than this -> drop the file
        "drop_env_like_files": True,   # .env, id_rsa, *.pem, .netrc, ...
    },
    "quality": {
        "min_content_chars": 2,
        "max_content_bytes": 1_048_576,
        "minified_avg_line_len": 250.0,
        "minified_max_line_len": 10_000,
        "max_base64_run": 2048,
        "repeated_line_ratio": 0.10,
        "min_alnum_ratio": 0.20,
        "lockfile_max_bytes": 204_800,
        "drop_flags": [
            "binary_like", "empty", "too_large", "minified", "vendored",
            "data_blob_heavy", "repeated_content", "generated_metadata",
            "lockfile_large", "low_alnum",
        ],
        "flag_only": ["generated_marker", "lockfile", "suspicious_obfuscation"],
    },
    "complexity": {
        "min_file_tokens": 4,
        "drop_below_min_tokens": True,
    },
    "subsets": {
        # high_quality is materialized as an extra filtered copy per shard.
        "materialize_high_quality": True,
        "high_quality_min_score": None,   # e.g. 3 for stackv2_edu int_score
    },
    "sources": {},   # filled by the YAML config
    "prototype": {
        "max_total_shards": None,
    },
}

# Keys excluded from the config hash (operational, not data-affecting).
# Underscore-prefixed keys and the derived config_* keys are always excluded.
_HASH_EXCLUDE_TOP = {"paths", "hf", "queue", "prototype", "config_hash", "config_file"}
_HASH_EXCLUDE_SOURCE_KEYS = {"max_shards", "enabled"}
# Nested keys that only change chunking / resume granularity, never records.
_HASH_EXCLUDE_NESTED = {("processing", "batch_size"), ("processing", "checkpoint_after")}


def load_dotenv_files(*candidates: str | Path) -> list[str]:
    """Minimal .env loader (no extra dependency).

    Accepts `KEY=VALUE` lines, `#` comments and an optional `export ` prefix.
    Variables already present in the real environment are NEVER overridden
    (shell env > .env > YAML), empty values are ignored, and values are never
    logged. Returns the names of the keys it set.
    """
    loaded: list[str] = []
    for candidate in candidates:
        candidate = Path(candidate)
        if not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if not key or not value or key in os.environ or key in loaded:
                continue
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


GB = 1_000_000_000  # decimal gigabyte, matching the spec's "1 GB – 2 GB"


def _normalize_units(cfg: dict) -> None:
    """User-facing GB keys -> canonical byte keys, BEFORE hashing, so
    `target_gb: 1.5` and `target_bytes: 1500000000` produce the same config
    hash. GB values may be fractional (e.g. 0.5)."""
    shard = cfg.get("shard") or {}
    for gb_key, bytes_key in (
        ("target_gb", "target_bytes"),
        ("min_gb", "min_bytes"),
        ("max_gb", "max_bytes"),
    ):
        if shard.get(gb_key) is not None:
            shard[bytes_key] = int(float(shard.pop(gb_key)) * GB)
    cfg["shard"] = shard

    if shard["min_bytes"] > shard["target_bytes"]:
        raise SystemExit(
            f"invalid shard sizing: min ({shard['min_bytes'] / GB:.2f} GB) is "
            f"larger than target ({shard['target_bytes'] / GB:.2f} GB)"
        )
    if shard["target_bytes"] > shard["max_bytes"]:
        raise SystemExit(
            f"invalid shard sizing: target ({shard['target_bytes'] / GB:.2f} GB) "
            f"exceeds max ({shard['max_bytes'] / GB:.2f} GB) — raise shard.max_gb"
        )


def compute_config_hash(cfg: dict) -> str:
    hashable: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in _HASH_EXCLUDE_TOP or key.startswith("_"):
            continue
        if key == "sources":
            sources = {}
            for name, scfg in (value or {}).items():
                sources[name] = {
                    k: v for k, v in (scfg or {}).items()
                    if k not in _HASH_EXCLUDE_SOURCE_KEYS
                }
            hashable[key] = sources
        else:
            hashable[key] = value
    for section, sub_key in _HASH_EXCLUDE_NESTED:
        if isinstance(hashable.get(section), dict) and sub_key in hashable[section]:
            filtered = dict(hashable[section])   # copy — never mutate cfg
            filtered.pop(sub_key)
            hashable[section] = filtered
    blob = json.dumps(hashable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> Cfg:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    cfg = _deep_merge(DEFAULTS, user_cfg)
    _normalize_units(cfg)

    # .env files (cwd, then next to the config / repo root). Loaded into
    # os.environ so huggingface_hub and boto3 pick them up too.
    cfg["_dotenv_loaded_keys"] = load_dotenv_files(
        Path.cwd() / ".env",
        config_path.parent / ".env",
        config_path.parent.parent / ".env",
    )

    # Environment overrides (operational knobs only).
    if os.environ.get(constants.ENV_HF_MODE):
        cfg["hf"]["mode"] = os.environ[constants.ENV_HF_MODE]
    if os.environ.get(constants.ENV_TARGET_REPO):
        cfg["hf"]["target_repo"] = os.environ[constants.ENV_TARGET_REPO]
    if os.environ.get(constants.ENV_WORKSPACE):
        cfg["paths"]["workspace"] = os.environ[constants.ENV_WORKSPACE]

    cfg["config_hash"] = compute_config_hash(cfg)
    cfg["config_file"] = str(config_path)
    return Cfg(cfg)


def workspace_paths(cfg: Cfg) -> Cfg:
    ws = Path(cfg.paths.workspace).resolve()
    return Cfg(
        {
            "workspace": ws,
            "raw": ws / "raw",
            "work": ws / "work",
            "processed": ws / "processed",
            "logs": ws / "logs",
            "manifest_db": ws / "manifest.db",
            "mock_hub": Path(cfg.hf.mock_dir).resolve() if cfg.hf.mock_dir else ws / "mock_hub",
        }
    )
