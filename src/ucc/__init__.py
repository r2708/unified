"""unified-code-corpus (ucc).

Shard-streamed, crash-resumable pipeline that unifies real-world code from
The Stack v2, StarCoderData, Common Pile Stack v2 and StarCoder2 extras into a
deduplicated, provenance-aware, license-aware, repository-level Hugging Face
dataset.

Real data only — this package never generates synthetic corpus content.
"""

import os

# Enable the fast Rust download path BEFORE huggingface_hub is imported
# anywhere — the flags are read at import time, so setting them later (inside
# a function) has no effect. All ucc imports of huggingface_hub are lazy, so
# running this at package import time always precedes them.
#
# huggingface_hub < 1.0 uses hf_transfer (HF_HUB_ENABLE_HF_TRANSFER);
# huggingface_hub >= 1.0 replaced it with Xet (HF_XET_HIGH_PERFORMANCE), and
# setting the old flag there only emits a deprecation warning.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
try:  # pragma: no cover - depends on installed versions
    from importlib.metadata import version as _pkg_version

    if int(_pkg_version("huggingface_hub").split(".")[0]) < 1:
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:  # noqa: BLE001 - optional fast path, never block import
    pass

from ucc.constants import PIPELINE_VERSION

__version__ = PIPELINE_VERSION
