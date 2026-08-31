"""Command-line interface.

    ucc run            start / resume the pipeline (fully idempotent)
    ucc status         state + statistics from the persistent manifest
    ucc enumerate      enumerate source shards only (no downloads)
    ucc verify-remote  audit uploaded files against recorded checksums
    ucc retry-failed   reset retry budgets for failed shards
    ucc finalize       export + upload global stats / repos table / card
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from ucc.config import load_config, workspace_paths
from ucc.logging_utils import get_logger, setup_logging

log = get_logger("cli")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", "-c", default="config.yaml",
        help="pipeline YAML config (default: ./config.yaml; presets: "
             "configs/prototype.yaml, configs/full.yaml). Credentials and the "
             "dataset name are read from .env",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ucc",
        description="unified-code-corpus: shard-streamed, resumable code-dataset pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start or resume the pipeline")
    _add_common(run)
    run.add_argument("--max-shards", type=int, default=None,
                     help="override prototype.max_total_shards for this run")
    run.add_argument("--allow-config-change", action="store_true",
                     help="proceed although data-affecting config changed")
    run.add_argument("--re-enumerate", action="store_true",
                     help="refresh source enumeration (new upstream files are appended)")

    status = sub.add_parser("status", help="show shard states and statistics")
    _add_common(status)

    enum = sub.add_parser("enumerate", help="enumerate source shards only")
    _add_common(enum)
    enum.add_argument("--re-enumerate", action="store_true")

    verify = sub.add_parser("verify-remote",
                            help="audit uploaded shards against recorded checksums")
    _add_common(verify)

    retry = sub.add_parser("retry-failed", help="reset retry budget of failed shards")
    _add_common(retry)

    fin = sub.add_parser("finalize",
                         help="export + upload global stats, repositories table, dataset card")
    _add_common(fin)

    return parser


def _cmd_run(args) -> int:
    from ucc.orchestrator import Orchestrator

    cfg = load_config(args.config)
    if args.max_shards is not None:
        cfg["prototype"]["max_total_shards"] = args.max_shards
    orch = Orchestrator(
        cfg,
        allow_config_change=args.allow_config_change,
        re_enumerate=args.re_enumerate,
    )

    def _on_signal(signum, frame):  # noqa: ARG001
        if orch.stop_event.is_set():
            log.error("second interrupt — hard exit (state is crash-safe)")
            import os

            os._exit(130)
        orch.request_stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    return orch.run()


def _open_manifest(cfg):
    from ucc.manifest import Manifest

    paths = workspace_paths(cfg)
    if not Path(paths.manifest_db).exists():
        log.error("no manifest at %s — run the pipeline first", paths.manifest_db)
        raise SystemExit(1)
    return Manifest(paths.manifest_db), paths


def _cmd_status(args) -> int:
    from ucc.stats import render_status

    cfg = load_config(args.config)
    manifest, _ = _open_manifest(cfg)
    print(render_status(manifest))
    return 0


def _cmd_enumerate(args) -> int:
    from ucc.orchestrator import Orchestrator

    cfg = load_config(args.config)
    orch = Orchestrator(cfg, re_enumerate=getattr(args, "re_enumerate", False))
    orch.enumerate_sources()
    print(json.dumps(orch.manifest.counts_by_state(), indent=2))
    return 0


def _cmd_verify_remote(args) -> int:
    """Audit: every output file the manifest believes is uploaded must still
    exist on the Hub. When the local processed copy still exists (shards not
    yet cleaned up), checksums are compared too."""
    from ucc.hf_remote import build_hub, verify_remote_file
    from ucc.states import ShardState
    from ucc.uploader import load_processed_outputs

    cfg = load_config(args.config)
    manifest, paths = _open_manifest(cfg)
    hub = build_hub(cfg)
    problems = 0
    checked = 0
    for shard in manifest.shards_in_states(
        [ShardState.VERIFIED, ShardState.COMPLETED]
    ):
        try:
            outputs = load_processed_outputs(paths.processed / shard["shard_id"])
            recorded = {o["dest"]: o for o in outputs}
        except Exception:  # noqa: BLE001 - cleaned up after completion
            recorded = {}
        for dest in json.loads(shard.get("hf_dest_paths") or "[]"):
            checked += 1
            rec = recorded.get(dest)
            if rec:
                ok, why = verify_remote_file(
                    hub, rec["local"], dest,
                    expected_sha256=rec.get("sha256"), expected_size=rec.get("size"),
                )
            else:
                info = hub.file_info(dest)
                ok, why = info.exists, ("exists" if info.exists else "missing on hub")
            if not ok:
                problems += 1
                print(f"PROBLEM {shard['shard_id']} {dest}: {why}")
    print(f"checked {checked} remote files, {problems} problems")
    return 1 if problems else 0


def _cmd_retry_failed(args) -> int:
    from ucc.states import RETRY_RECYCLE

    cfg = load_config(args.config)
    manifest, _ = _open_manifest(cfg)
    reset = 0
    for failure_state, recycle_state in RETRY_RECYCLE.items():
        for shard in manifest.shards_in_states([failure_state]):
            if manifest.transition(
                shard["shard_id"], [failure_state], recycle_state,
                retry_count=0, next_retry_at=None, claimed_by=None, error=None,
            ):
                reset += 1
    print(f"reset {reset} failed shards — run `ucc run` to resume")
    return 0


def _cmd_finalize(args) -> int:
    from ucc.hf_remote import build_hub
    from ucc.stats import run_finalize

    cfg = load_config(args.config)
    manifest, _ = _open_manifest(cfg)
    hub = build_hub(cfg)
    hub.ensure_repo()
    agg = run_finalize(cfg, manifest, hub)
    print(json.dumps({k: v for k, v in agg.items() if k != "sources"}, indent=2,
                     default=str))
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "status": _cmd_status,
    "enumerate": _cmd_enumerate,
    "verify-remote": _cmd_verify_remote,
    "retry-failed": _cmd_retry_failed,
    "finalize": _cmd_finalize,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(
            f"config file not found: {cfg_path}\n"
            "Expected ./config.yaml (main config) — or pass a preset with "
            "-c configs/prototype.yaml / -c configs/full.yaml",
            file=sys.stderr,
        )
        sys.exit(1)
    log_dir = None
    try:
        log_dir = workspace_paths(load_config(cfg_path)).logs
    except Exception:  # noqa: BLE001 - logging must never block startup
        log_dir = None
    setup_logging(log_dir)
    sys.exit(_COMMANDS[args.command](args))


if __name__ == "__main__":
    main()
