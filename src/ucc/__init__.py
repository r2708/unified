"""unified-code-corpus (ucc).

Shard-streamed, crash-resumable pipeline that unifies real-world code from
The Stack v2, StarCoderData, Common Pile Stack v2 and StarCoder2 extras into a
deduplicated, provenance-aware, license-aware, repository-level Hugging Face
dataset.

Real data only — this package never generates synthetic corpus content.
"""

from ucc.constants import PIPELINE_VERSION

__version__ = PIPELINE_VERSION
