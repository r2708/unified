"""Producer–consumer orchestrator.

    SOURCE DATASETS -> DOWNLOAD PRODUCERS -> LOCAL SHARD QUEUE (max 5)
    -> PROCESSING CONSUMER -> UPLOAD -> VERIFY -> DELETE RAW -> FREE SLOT
    -> next download starts automatically

Downloads run concurrently with processing; the downloader pauses
automatically when the local queue holds the configured maximum of raw
shards and resumes as soon as a verified shard's raw data is deleted.
Every transition goes through the manifest state machine, so a hard crash at
any point resumes without duplicating work.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import traceback

from ucc.config import Cfg, workspace_paths
from ucc.crash import maybe_crash
from ucc.hashing import combined_raw_checksum, sha256_file
from ucc.hf_remote import build_hub
from ucc.io_utils import atomic_write_json, ensure_dir, free_disk_gb, safe_rmtree
from ucc.logging_utils import get_logger, setup_logging
from ucc.manifest import Manifest
from ucc.processing.base import ShardContext
from ucc.processing.runner import RawValidationError, run_shard_pipeline
from ucc.resume import check_config_hash, reconcile
from ucc.shard_queue import CapacityGauge
from ucc.sources import build_adapters
from ucc.sources.base import DownloadError
from ucc.states import RETRY_RECYCLE, ShardState
from ucc.uploader import (
    UploadFailure,
    VerificationFailure,
    load_processed_outputs,
    upload_outputs,
    upload_shard_stats,
    verify_outputs,
)

log = get_logger("orchestrator")

_ACTIVE_STATES = {
    ShardState.DOWNLOADING, ShardState.DOWNLOADED, ShardState.PROCESSING,
    ShardState.PROCESSED, ShardState.UPLOADING, ShardState.UPLOADED,
    ShardState.VERIFIED, ShardState.PENDING,
}


class Orchestrator:
    def __init__(self, cfg: Cfg, allow_config_change: bool = False,
                 re_enumerate: bool = False):
        self.cfg = cfg
        self.allow_config_change = allow_config_change
        self.re_enumerate = re_enumerate
        self.paths = workspace_paths(cfg)
        for key in ("workspace", "raw", "work", "processed", "logs"):
            ensure_dir(self.paths[key])
        setup_logging(self.paths.logs)
        self.manifest = Manifest(self.paths.manifest_db)
        self.hub = build_hub(cfg)
        self.adapters = build_adapters(cfg)
        self.gauge = CapacityGauge(int(cfg.queue.max_local_shards))
        self.stop_event = threading.Event()
        self.exit_code = 0
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------ enumerate
    def enumerate_sources(self) -> None:
        global_cap = self.cfg.path("prototype.max_total_shards")
        existing = len(self.manifest.all_shards())
        budget = None if global_cap is None else max(0, int(global_cap) - existing)

        for name, adapter in self.adapters.items():
            status = adapter.status()
            self.manifest.set_kv(f"source_status:{name}", status.reason)
            if not status.available:
                log.warning("source %s unavailable: %s", name, status.reason)
                continue
            enum_key = f"enum:{name}"
            if self.manifest.get_kv(enum_key) and not self.re_enumerate:
                log.info("source %s already enumerated — skipping (use "
                         "--re-enumerate to refresh)", name)
                continue
            log.info("enumerating source %s ...", name)
            try:
                inserted = 0
                revision = None
                for spec in adapter.enumerate_shards():
                    if budget is not None and budget <= 0:
                        log.warning(
                            "prototype.max_total_shards reached — stopped "
                            "enumerating %s early (coverage truncated, not silent: "
                            "%d shards inserted)", name, inserted,
                        )
                        break
                    revision = spec.ref.get("revision", revision)
                    if self.manifest.upsert_shard_spec(
                        spec.shard_id, name, spec.ref, spec.seq_index,
                        adapter.priority, self.cfg.pipeline_version,
                        self.cfg.config_hash,
                    ):
                        inserted += 1
                        if budget is not None:
                            budget -= 1
                self.manifest.set_kv(
                    enum_key,
                    json.dumps({"revision": revision, "inserted": inserted}),
                )
                log.info("source %s: %d new shards enumerated", name, inserted)
            except Exception as exc:  # noqa: BLE001 - one source must not kill the run
                log.error("enumeration failed for %s: %s", name, exc)

    # ------------------------------------------------------------- download
    def _stop_check(self) -> bool:
        return self.stop_event.is_set()

    def _downloader_loop(self, idx: int) -> None:
        worker_id = f"downloader-{idx}"
        min_free = float(self.cfg.queue.min_free_disk_gb)
        while not self.stop_event.is_set():
            if free_disk_gb(self.paths.workspace) < min_free:
                log.warning("low disk space (<%.0f GB free) — downloads paused", min_free)
                self.stop_event.wait(30)
                continue
            # Queue cap: block here until a raw-shard slot frees up.
            if not self.gauge.acquire(self.stop_event):
                return
            shard = self.manifest.claim_next(
                [ShardState.PENDING], worker_id, to_state=ShardState.DOWNLOADING
            )
            if shard is None:
                self.gauge.release()
                self.stop_event.wait(3)
                continue
            try:
                self._download_one(shard)
                # Success: slot stays occupied by the raw shard on disk.
            except Exception as exc:  # noqa: BLE001
                self._fail_download(shard, exc)
                self.gauge.release()

    def _download_one(self, shard: dict) -> None:
        shard_id = shard["shard_id"]
        adapter = self.adapters[shard["source_dataset"]]
        ref = json.loads(shard["source_ref"])
        raw_dir = self.paths.raw / shard_id
        safe_rmtree(raw_dir, self.paths.workspace)  # clear any partial attempt
        ensure_dir(raw_dir)
        log.info("[%s] downloading ...", shard_id)
        adapter.download(ref, raw_dir, stop_check=self._stop_check)

        rel_files = sorted(
            str(p.relative_to(raw_dir))
            for p in raw_dir.rglob("*")
            if p.is_file() and p.name != ".ucc_raw_files.json"
            and ".cache" not in p.parts
        )
        if not rel_files:
            raise DownloadError("download produced no files")
        sizes = [(raw_dir / rel).stat().st_size for rel in rel_files]
        hashes = {rel: sha256_file(raw_dir / rel) for rel in rel_files}
        atomic_write_json(
            raw_dir / ".ucc_raw_files.json", {"files": rel_files, "sizes": sizes}
        )
        raw_checksum = combined_raw_checksum(hashes)
        maybe_crash("after_download")
        ok = self.manifest.transition(
            shard_id, [ShardState.DOWNLOADING], ShardState.DOWNLOADED,
            local_raw_dir=str(raw_dir), raw_checksum=raw_checksum,
            raw_size_bytes=sum(sizes), error=None, claimed_by=None,
        )
        if not ok:  # pragma: no cover - state-machine guard
            raise RuntimeError(f"illegal transition for {shard_id}")
        log.info("[%s] downloaded: %d files, %.2f GB", shard_id,
                 len(rel_files), sum(sizes) / 1e9)

    def _fail_download(self, shard: dict, exc: Exception) -> None:
        shard_id = shard["shard_id"]
        log.error("[%s] download failed: %s", shard_id, exc)
        safe_rmtree(self.paths.raw / shard_id, self.paths.workspace)
        retry = int(shard.get("retry_count") or 0) + 1
        self.manifest.transition(
            shard_id, [ShardState.DOWNLOADING], ShardState.DOWNLOAD_FAILED,
            error=str(exc)[:1000], retry_count=retry,
            next_retry_at=self._next_retry(retry), claimed_by=None,
            local_raw_dir=None, raw_checksum=None,
        )

    # ------------------------------------------------------------- consumer
    def _consumer_loop(self, idx: int) -> None:
        worker_id = f"processor-{idx}"
        claimable = [
            ShardState.DOWNLOADED, ShardState.PROCESSING, ShardState.PROCESSED,
            ShardState.UPLOADING, ShardState.UPLOADED, ShardState.VERIFIED,
        ]
        while not self.stop_event.is_set():
            shard = self.manifest.claim_next(claimable, worker_id)
            if shard is None:
                self.stop_event.wait(3)
                continue
            try:
                self._advance(shard)
            except Exception:  # noqa: BLE001 - never kill the worker thread
                log.error("[%s] unexpected worker error:\n%s",
                          shard["shard_id"], traceback.format_exc())
            finally:
                current = self.manifest.get_shard(shard["shard_id"])
                if current and current.get("claimed_by") == worker_id:
                    self.manifest.release_claim(shard["shard_id"])

    def _advance(self, shard: dict) -> None:
        """Drive one claimed shard as far as possible:
        process -> upload -> verify -> cleanup -> completed."""
        shard_id = shard["shard_id"]
        while not self.stop_event.is_set():
            state = ShardState(shard["state"])
            if state in (ShardState.DOWNLOADED, ShardState.PROCESSING):
                if not self._process(shard):
                    return
            elif state in (ShardState.PROCESSED, ShardState.UPLOADING):
                if not self._upload(shard):
                    return
            elif state == ShardState.UPLOADED:
                if not self._verify(shard):
                    return
            elif state == ShardState.VERIFIED:
                self._cleanup_and_complete(shard)
                return
            else:
                return
            shard = self.manifest.get_shard(shard_id)
            if shard is None:
                return

    def _build_ctx(self, shard: dict) -> ShardContext:
        shard_id = shard["shard_id"]
        return ShardContext(
            shard=shard,
            spec_ref=json.loads(shard["source_ref"]),
            cfg=self.cfg,
            manifest=self.manifest,
            adapter=self.adapters[shard["source_dataset"]],
            raw_dir=self.paths.raw / shard_id,
            work_dir=self.paths.work / shard_id,
            processed_dir=self.paths.processed / shard_id,
            workspace_root=self.paths.workspace,
            hub=self.hub,
        )

    def _process(self, shard: dict) -> bool:
        shard_id = shard["shard_id"]
        self.manifest.transition(
            shard_id, [ShardState.DOWNLOADED, ShardState.PROCESSING],
            ShardState.PROCESSING,
        )
        shard = self.manifest.get_shard(shard_id)
        log.info("[%s] processing ...", shard_id)
        try:
            ctx = self._build_ctx(shard)
            run_shard_pipeline(ctx)
            self.manifest.update_shard_stats(shard_id, ctx.stats)
            self.manifest.set_fields(shard_id, records_in=ctx.stats.get("records_in"))
            ok = self.manifest.transition(
                shard_id, [ShardState.PROCESSING], ShardState.PROCESSED, error=None
            )
            maybe_crash("after_processed")
            return ok
        except RawValidationError as exc:
            # Raw data is unusable: re-download only this shard.
            log.warning("[%s] raw validation failed (%s) — re-queueing", shard_id, exc)
            for d in (self.paths.raw / shard_id, self.paths.work / shard_id,
                      self.paths.processed / shard_id):
                safe_rmtree(d, self.paths.workspace)
            self.manifest.set_fields(
                shard_id, stage_checkpoint=None, checkpoint_path=None,
                local_raw_dir=None, raw_checksum=None,
            )
            self.manifest.transition(
                shard_id, [ShardState.PROCESSING], ShardState.PENDING,
                error=str(exc)[:1000], claimed_by=None,
            )
            self.gauge.release()
            return False
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] processing failed:\n%s", shard_id, traceback.format_exc())
            retry = int(shard.get("retry_count") or 0) + 1
            self.manifest.transition(
                shard_id, [ShardState.PROCESSING], ShardState.PROCESSING_FAILED,
                error=str(exc)[:1000], retry_count=retry,
                next_retry_at=self._next_retry(retry), claimed_by=None,
            )
            return False

    def _upload(self, shard: dict) -> bool:
        shard_id = shard["shard_id"]
        try:
            outputs = load_processed_outputs(self.paths.processed / shard_id)
            self.manifest.transition(
                shard_id, [ShardState.PROCESSED, ShardState.UPLOADING],
                ShardState.UPLOADING,
            )
            transferred = upload_outputs(
                self.hub, outputs, shard_id,
                workers=int(self.cfg.path("hf.upload_workers", 4)),
            )
            log.info("[%s] upload done (%d files transferred)", shard_id, transferred)
            maybe_crash("after_upload_before_verify")
            return self.manifest.transition(
                shard_id, [ShardState.UPLOADING], ShardState.UPLOADED, error=None
            )
        except Exception as exc:  # noqa: BLE001 - includes UploadFailure/HubError
            log.error("[%s] upload failed: %s", shard_id, exc)
            retry = int(shard.get("retry_count") or 0) + 1
            self.manifest.transition(
                shard_id,
                [ShardState.PROCESSED, ShardState.UPLOADING],
                ShardState.UPLOAD_FAILED,
                error=str(exc)[:1000], retry_count=retry,
                next_retry_at=self._next_retry(retry), claimed_by=None,
            )
            return False

    def _verify(self, shard: dict) -> bool:
        shard_id = shard["shard_id"]
        try:
            outputs = load_processed_outputs(self.paths.processed / shard_id)
            verify_outputs(
                self.hub, outputs,
                workers=int(self.cfg.path("hf.upload_workers", 4)),
            )
            ok = self.manifest.transition(
                shard_id, [ShardState.UPLOADED], ShardState.VERIFIED, error=None
            )
            log.info("[%s] upload verified", shard_id)
            upload_shard_stats(self.hub, self.manifest.get_shard(shard_id))
            maybe_crash("after_verify_before_cleanup")
            return ok
        except (VerificationFailure, UploadFailure) as exc:
            log.error("[%s] verification failed: %s", shard_id, exc)
            retry = int(shard.get("retry_count") or 0) + 1
            self.manifest.transition(
                shard_id, [ShardState.UPLOADED], ShardState.VERIFICATION_FAILED,
                error=str(exc)[:1000], retry_count=retry,
                next_retry_at=self._next_retry(retry), claimed_by=None,
            )
            return False

    def _cleanup_and_complete(self, shard: dict) -> None:
        """Only after successful verification: delete the raw shard and all
        temporary processing files, then free the queue slot."""
        shard_id = shard["shard_id"]
        safe_rmtree(self.paths.raw / shard_id, self.paths.workspace)
        safe_rmtree(self.paths.work / shard_id, self.paths.workspace)
        if not self.cfg.path("paths.keep_local_processed", False):
            safe_rmtree(self.paths.processed / shard_id, self.paths.workspace)
        ok = self.manifest.transition(
            shard_id, [ShardState.VERIFIED], ShardState.COMPLETED,
            claimed_by=None, local_raw_dir=None,
        )
        if ok:
            self.gauge.release()
            log.info("[%s] COMPLETED — raw shard deleted, queue slot freed", shard_id)

    # -------------------------------------------------------------- retries
    def _next_retry(self, retry_count: int) -> str:
        base = float(self.cfg.queue.retry_backoff_base_s)
        cap = float(self.cfg.queue.retry_backoff_max_s)
        delay = min(base * (2 ** max(retry_count - 1, 0)), cap)
        when = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=delay)
        return when.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _janitor_loop(self) -> None:
        max_retries = int(self.cfg.queue.max_retries)
        while not self.stop_event.wait(15):
            try:
                for failure_state, recycle_state in RETRY_RECYCLE.items():
                    for shard in self.manifest.shards_in_states([failure_state]):
                        if int(shard.get("retry_count") or 0) > max_retries:
                            continue
                        due = shard.get("next_retry_at")
                        now = dt.datetime.now(dt.timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"
                        )
                        if due and due > now:
                            continue
                        if self.manifest.transition(
                            shard["shard_id"], [failure_state], recycle_state,
                            claimed_by=None,
                        ):
                            log.info("[%s] retrying (%s -> %s, attempt %s)",
                                     shard["shard_id"], failure_state.value,
                                     recycle_state.value, shard.get("retry_count"))
            except Exception:  # noqa: BLE001 - the janitor must never die silently
                log.error("janitor error (continuing):\n%s", traceback.format_exc())

    # ------------------------------------------------------------ lifecycle
    def _progress_snapshot(self) -> dict:
        counts = self.manifest.counts_by_state()
        max_retries = int(self.cfg.queue.max_retries)
        failed = self.manifest.shards_in_states(RETRY_RECYCLE.keys())
        given_up = sum(1 for s in failed if int(s.get("retry_count") or 0) > max_retries)
        retryable = len(failed) - given_up
        pending = counts.get(ShardState.PENDING.value, 0)
        state_active = sum(counts.get(s.value, 0) for s in _ACTIVE_STATES)
        return {
            "counts": counts,
            "pending": pending,
            "in_flight": state_active - pending,
            "retryable": retryable,        # failed, janitor will recycle them
            "given_up": given_up,          # failed, retries exhausted
            "active": state_active + retryable,
        }

    def _log_overall_progress(self, snap: dict) -> None:
        tot = self.manifest.progress_totals()
        done = snap["counts"].get(ShardState.COMPLETED.value, 0) + snap[
            "counts"
        ].get(ShardState.SKIPPED.value, 0)
        total = max(tot["total_shards"], 1)
        log.info(
            "OVERALL %5.1f%% — %d/%d shards done | slots %d/%d | %s | "
            "records out %s | tokens ≈%s | retryable %d | given up %d",
            100.0 * done / total, done, tot["total_shards"],
            self.gauge.occupied, self.gauge.max_slots,
            {k: v for k, v in sorted(snap["counts"].items())},
            f"{tot['records_out']:,}", f"{tot['tokens']:,}",
            snap["retryable"], snap["given_up"],
        )

    def _monitor_loop(self) -> None:
        interval = float(self.cfg.queue.status_interval_s)
        while not self.stop_event.wait(interval):
            try:
                snap = self._progress_snapshot()
            except Exception:  # noqa: BLE001 - progress lines must never stop
                log.error("monitor error (continuing):\n%s", traceback.format_exc())
                continue
            self._log_overall_progress(snap)
            if snap["active"] == 0:
                log.info("all shards reached a terminal state — shutting down")
                self.stop_event.set()
            elif (
                snap["pending"] > 0
                and snap["in_flight"] == 0
                and snap["retryable"] == 0
                and snap["given_up"] > 0
                and self.gauge.occupied >= self.gauge.max_slots
            ):
                log.error(
                    "queue blocked: %d pending shards but all %d slots are "
                    "held by shards that exhausted retries. Inspect errors, "
                    "then run `ucc retry-failed` to retry them.",
                    snap["pending"], self.gauge.max_slots,
                )
                self.exit_code = 3
                self.stop_event.set()

    def run(self) -> int:
        log.info(
            "unified-code-corpus pipeline v%s | config %s | hash %s | hub mode %s",
            self.cfg.pipeline_version, self.cfg.config_file,
            self.cfg.config_hash, self.cfg.hf.mode,
        )
        dotenv_keys = self.cfg.get("_dotenv_loaded_keys") or []
        if dotenv_keys:
            log.info(".env loaded (%s)", ", ".join(dotenv_keys))  # names only
        if self.cfg.hf.mode == "real" and str(self.cfg.hf.target_repo).startswith(
            "CHANGE-ME"
        ):
            raise SystemExit(
                "No target dataset configured. Set UCC_TARGET_REPO="
                "<hf-username>/<dataset-name> in .env (or hf.target_repo in "
                "config.yaml), or use UCC_HF_MODE=mock for a dry run."
            )
        check_config_hash(self.manifest, self.cfg, self.allow_config_change)
        self.hub.ensure_repo()

        summary = reconcile(self.manifest, self.hub, self.cfg, self.paths)
        self.enumerate_sources()
        self.gauge.prime(summary["raw_on_disk"])

        snap = self._progress_snapshot()
        if snap["active"] == 0:
            log.info("nothing to do: %s", snap["counts"])
            return 0

        workers: list[threading.Thread] = []
        for i in range(int(self.cfg.queue.download_workers)):
            workers.append(threading.Thread(
                target=self._downloader_loop, args=(i,), name=f"downloader-{i}",
                daemon=True,
            ))
        for i in range(int(self.cfg.queue.process_workers)):
            workers.append(threading.Thread(
                target=self._consumer_loop, args=(i,), name=f"processor-{i}",
                daemon=True,
            ))
        workers.append(threading.Thread(target=self._janitor_loop, name="janitor", daemon=True))
        workers.append(threading.Thread(target=self._monitor_loop, name="monitor", daemon=True))
        for w in workers:
            w.start()
        self._threads = workers

        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        finally:
            self.stop_event.set()
            for w in workers:
                w.join(timeout=60)

        snap = self._progress_snapshot()
        self._log_overall_progress(snap)
        log.info("final state counts: %s | given up: %d",
                 snap["counts"], snap["given_up"])
        if snap["given_up"] and self.exit_code == 0:
            self.exit_code = 2
        return self.exit_code

    def request_stop(self) -> None:
        log.warning("shutdown requested — finishing current stage, then "
                    "checkpointing (state is resumable at any point)")
        self.stop_event.set()
