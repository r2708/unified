#!/usr/bin/env python3
"""Crash/resume validation harness for prototype-v0.1 (spec point 12).

Simulates hard crashes (os._exit) at every pipeline phase, restarts, and
asserts that the pipeline resumes WITHOUT duplicating work:

  1. run with UCC_CRASH_AT=<point>  -> expect the injected crash (exit 137)
  2. re-run                          -> pipeline resumes from the manifest
  3. after all points: assert every shard is completed, every recorded
     Hub file exists, and a further idempotent re-run changes NOTHING
     (file hashes on the hub stay identical, no shard is reprocessed).

Uses REAL source data (network + disk required) but a MOCK hub
(UCC_HF_MODE=mock) so nothing is pushed to the real Hugging Face Hub.

Usage:
    python scripts/crash_test.py [--config configs/prototype.yaml]
                                 [--workspace ./workspace-crashtest]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CRASH_POINTS = [
    "after_download",
    "after_stage:normalize",
    "after_stage:dedup_near",
    "after_stage:secrets_scan",
    "after_batch_upload",        # fires only when upload_per_batch is true
    "after_stage:finalize",
    "after_processed",
    "after_upload_before_verify",
    "after_verify_before_cleanup",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hub_snapshot(mock_hub: Path) -> dict[str, str]:
    if not mock_hub.exists():
        return {}
    return {
        str(p.relative_to(mock_hub)): sha256_file(p)
        for p in sorted(mock_hub.rglob("*"))
        if p.is_file()
    }


def run_pipeline(config: str, workspace: Path, crash_at: str | None,
                 extra_args: list[str] | None = None) -> int:
    env = dict(os.environ)
    env["UCC_HF_MODE"] = "mock"
    env["UCC_WORKSPACE"] = str(workspace)
    env.pop("UCC_CRASH_AT", None)
    if crash_at:
        env["UCC_CRASH_AT"] = crash_at
    cmd = [sys.executable, "-m", "ucc", "run", "--config", config]
    cmd += extra_args or []
    print(f"\n>>> {' '.join(cmd)}  (UCC_CRASH_AT={crash_at or '-'})")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


def manifest_rows(workspace: Path) -> list[dict]:
    db = workspace / "manifest.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM shards")]
    conn.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--workspace", default="./workspace-crashtest")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    mock_hub = workspace / "mock_hub"
    if workspace.exists():
        print(f"ERROR: workspace {workspace} already exists — use a fresh one "
              "so the test starts from a clean slate")
        return 2

    failures: list[str] = []

    # Phase 1: crash at every point in sequence; each restart must resume.
    for point in CRASH_POINTS:
        before = hub_snapshot(mock_hub)
        code = run_pipeline(args.config, workspace, crash_at=point)
        if code == 137:
            print(f"    crash injected at {point} (as designed)")
        elif code in (0, 2, 3):
            print(f"    pipeline finished before reaching {point} "
                  f"(exit {code}) — point already passed on a resumed shard")
        else:
            failures.append(f"unexpected exit code {code} at crash point {point}")
        after = hub_snapshot(mock_hub)
        for path, digest in before.items():
            if path in after and after[path] != digest:
                failures.append(
                    f"hub file {path} CHANGED after crash at {point} — "
                    "verified uploads must never be rewritten"
                )

    # Phase 2: clean run to completion.
    code = run_pipeline(args.config, workspace, crash_at=None)
    if code not in (0, 2):
        failures.append(f"final resume run exited {code}")

    rows = manifest_rows(workspace)
    incomplete = [r["shard_id"] for r in rows
                  if r["state"] not in ("completed", "skipped")]
    if incomplete:
        failures.append(f"shards not completed after final run: {incomplete}")

    for row in rows:
        if row["state"] != "completed":
            continue
        for dest in json.loads(row["hf_dest_paths"] or "[]"):
            if not (mock_hub / dest).exists():
                failures.append(f"{row['shard_id']}: uploaded file missing on hub: {dest}")
        raw_dir = workspace / "raw" / row["shard_id"]
        if raw_dir.exists():
            failures.append(f"{row['shard_id']}: raw shard not deleted after completion")

    # Phase 3: idempotency — a re-run must be a no-op.
    before = hub_snapshot(mock_hub)
    rows_before = {r["shard_id"]: (r["state"], r["updated_at"]) for r in rows}
    code = run_pipeline(args.config, workspace, crash_at=None)
    if code not in (0, 2):
        failures.append(f"idempotent re-run exited {code}")
    after = hub_snapshot(mock_hub)
    if before != after:
        changed = {k for k in before.keys() ^ after.keys()} | {
            k for k in before.keys() & after.keys() if before[k] != after[k]
        }
        failures.append(f"idempotent re-run modified hub files: {sorted(changed)}")
    for row in manifest_rows(workspace):
        prev = rows_before.get(row["shard_id"])
        if prev and prev[0] == "completed" and row["state"] != "completed":
            failures.append(f"{row['shard_id']}: completed shard was reopened")

    print("\n=== crash test result ===")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {len(CRASH_POINTS)} crash points survived; all shards "
          "completed exactly once; re-run was a no-op")
    return 0


if __name__ == "__main__":
    sys.exit(main())
