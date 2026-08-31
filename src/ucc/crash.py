"""Crash-point injection for prototype resilience testing.

Set UCC_CRASH_AT to a comma-separated list of crash points and the process
hard-exits (os._exit) when it reaches one — simulating a power failure at
that exact moment. Used by scripts/crash_test.py to prove that resume never
duplicates work.

Known crash points:
    after_download
    after_stage:<stage_name>       e.g. after_stage:dedup_near
    after_batch_upload             per-batch mode: after a batch is verified
    after_processed
    after_upload_before_verify
    after_verify_before_cleanup
"""

from __future__ import annotations

import os

from ucc.constants import ENV_CRASH_AT
from ucc.logging_utils import get_logger

log = get_logger("crash")


def maybe_crash(point: str) -> None:
    configured = os.environ.get(ENV_CRASH_AT, "")
    if not configured:
        return
    points = {p.strip() for p in configured.split(",") if p.strip()}
    if point in points:
        log.error("CRASH INJECTION: simulating hard crash at '%s'", point)
        os._exit(137)
