"""Shard lifecycle state machine (names mirror the pipeline spec)."""

from __future__ import annotations

from enum import Enum


class ShardState(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    VERIFIED = "verified"
    COMPLETED = "completed"

    DOWNLOAD_FAILED = "download_failed"
    PROCESSING_FAILED = "processing_failed"
    UPLOAD_FAILED = "upload_failed"
    VERIFICATION_FAILED = "verification_failed"

    # Source unavailable (gated dataset without accepted terms/token, etc.).
    SKIPPED = "skipped"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# States in which the raw downloaded shard may exist on local disk and
# therefore occupies one of the MAX_LOCAL_SHARDS queue slots. The raw shard
# is only deleted (and the slot freed) after the processed output has been
# uploaded AND verified.
RAW_ON_DISK_STATES = {
    ShardState.DOWNLOADING,
    ShardState.DOWNLOADED,
    ShardState.PROCESSING,
    ShardState.PROCESSED,
    ShardState.UPLOADING,
    ShardState.UPLOADED,
    ShardState.VERIFIED,
    ShardState.PROCESSING_FAILED,
    ShardState.UPLOAD_FAILED,
    ShardState.VERIFICATION_FAILED,
}

# Terminal states: never picked up by workers again.
TERMINAL_STATES = {ShardState.COMPLETED, ShardState.SKIPPED}

# States the processing/upload consumer may claim work from.
CONSUMER_CLAIMABLE_STATES = {
    ShardState.DOWNLOADED,
    ShardState.PROCESSING,
    ShardState.PROCESSED,
    ShardState.UPLOADING,
    ShardState.UPLOADED,
    ShardState.VERIFIED,
}

# Failure state -> state it is recycled to when retried.
RETRY_RECYCLE = {
    ShardState.DOWNLOAD_FAILED: ShardState.PENDING,
    ShardState.PROCESSING_FAILED: ShardState.DOWNLOADED,
    ShardState.UPLOAD_FAILED: ShardState.PROCESSED,
    ShardState.VERIFICATION_FAILED: ShardState.PROCESSED,
}
