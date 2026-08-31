"""Global constants for the unified-code-corpus pipeline."""

PIPELINE_VERSION = "0.1.0"
MANIFEST_SCHEMA_VERSION = 1

# Hard requirement from the spec: at most 5 raw shards may exist locally at
# any time (downloading + waiting + processing + awaiting verified upload).
DEFAULT_MAX_LOCAL_SHARDS = 5

# Raw shard sizing targets (bytes).
DEFAULT_SHARD_TARGET_BYTES = 1_500_000_000   # aim ~1.5 GB
DEFAULT_SHARD_MIN_BYTES = 1_000_000_000      # 1 GB
DEFAULT_SHARD_MAX_BYTES = 2_000_000_000      # 2 GB

# Record types.
RT_CODE = "code"
RT_COMMIT = "commit"
RT_ISSUE = "issue"
RT_DOC = "doc"

# Output subsets that are always materialized as their own directories
# (disjoint by record_type, so no data duplication).
SUBSET_FULL = "full"
SUBSET_COMMITS = "commits"
SUBSET_ISSUES = "issues"
SUBSET_HIGH_QUALITY = "high_quality"
SUBSET_EXCLUDED = "excluded"

RECORD_TYPE_TO_SUBSET = {
    RT_CODE: SUBSET_FULL,
    RT_DOC: SUBSET_FULL,
    RT_COMMIT: SUBSET_COMMITS,
    RT_ISSUE: SUBSET_ISSUES,
}

# Software layers assigned per record.
LAYER_FRONTEND = "frontend"
LAYER_BACKEND = "backend"
LAYER_DATABASE = "database"
LAYER_INFRA = "infrastructure"
LAYER_CONFIG = "configuration"
LAYER_TESTING = "testing"
LAYER_DOCS = "docs"
LAYER_OTHER = "other"

# Repository categories.
REPO_CATEGORIES = (
    "frontend",
    "backend",
    "database",
    "full_stack",
    "infrastructure",
    "library",
    "cli",
    "ml_data",
    "mixed",
)

# License status buckets.
LIC_PERMISSIVE = "permissive"
LIC_WEAK_COPYLEFT = "copyleft_weak"
LIC_STRONG_COPYLEFT = "copyleft_strong"
LIC_UNKNOWN = "unknown"

ENV_HF_TOKEN = "HF_TOKEN"
ENV_CRASH_AT = "UCC_CRASH_AT"
ENV_HF_MODE = "UCC_HF_MODE"
ENV_TARGET_REPO = "UCC_TARGET_REPO"
ENV_WORKSPACE = "UCC_WORKSPACE"
