# unified-code-corpus

Shard-streamed, crash-resumable pipeline that builds a **deduplicated,
provenance-aware, license-aware, repository-level real-world code corpus**
from four sources and publishes it to the Hugging Face Hub:

| source | access | what it contributes |
|---|---|---|
| `common-pile/stackv2` | public | Dolma-format Stack v2 slice with full provenance metadata (`documents/*_stackv2.jsonl.gz`, ~1.5 GB each — a natural shard size). Covers languages alphabetically up to ~Markdown. |
| `common-pile/stackv2_edu_filtered` | public | The edu-filtered Stack v2 shards that carry the major languages missing from the above (Python, TypeScript, JS, Go, Rust, …) plus a per-record quality score. |
| `bigcode/starcoder2data-extras` | public | 15.5M **real GitHub issues** (`{repo_name, content, issue_id}`), documentation, and more. *(The task brief's `huggingface.co/starcoder2data` URL is not a dataset repo; this is the actual StarCoder2 companion data — adjust `repo_id` in the config if you meant something else.)* |
| `bigcode/starcoderdata` | gated (accept terms + `hf auth login`) | Per-language code with content inline, plus **real git commits** (`git-commits-cleaned`) and issue threads. |
| `bigcode/the-stack-v2` | gated + AWS | Metadata only on HF; file contents come from Software Heritage S3 (`s3://softwareheritage/content/{blob_id}`) and require **both** an accepted-terms HF token **and** AWS credentials. |

**Real data only.** The pipeline never generates synthetic corpus content,
never fabricates history, and **never bypasses gating** — sources whose
credentials/terms are missing are skipped with an explanatory log line and
activate automatically once credentials exist.

---

## Architecture

```
                    SOURCE DATASETS
                          │  (enumerate: file listings + parquet footers only,
                          ▼   revision-pinned → deterministic shard specs)
                   DOWNLOAD PRODUCERS (3 threads)
                          │
                          ▼
                  LOCAL SHARD QUEUE  ←── HARD CAP: 5 raw shards on disk
                          │              (downloading + waiting + processing
                          ▼               + awaiting verified upload)
                 PROCESSING CONSUMER
     validate → normalize → provenance → exact dedup → near dedup
     → repo reconstruction → license filter → secrets/PII removal
     → quality filter → complexity → classification → finalize
                          │
                          ▼
                   HUGGING FACE UPLOAD  (skip files already present+matching)
                          │
                          ▼
                     VERIFY UPLOAD      (size + sha256 / git-sha1)
                          │
                          ▼
                   DELETE RAW SHARD + temp files   ← only AFTER verification
                          │
                          ▼
                   FREE QUEUE SLOT ────────────→ next download starts
```

Downloads run **concurrently** with processing; the producer pauses
automatically when 5 raw shards are local and resumes the moment a verified
shard's raw data is deleted. Every step transitions a SQLite state machine
(`workspace/manifest.db`, WAL + `synchronous=FULL`), so a crash at any
instant — mid-download, mid-stage, mid-upload, mid-verify — resumes without
duplicating work.

### Per-batch upload mode (`processing.upload_per_batch: true` — on in config.yaml)

Records are processed in deterministic batches of `processing.batch_size`;
**the moment a batch finishes all stages, its parquet is uploaded to the Hub
and verified** (`part-{shard}-b{batch:04d}.parquet`), recorded in
`work_dir/batch_progress.json`, and only then does the next batch start —
data reaches the Hub continuously instead of once per shard. The batch is
also the crash-resume unit: on restart, already-uploaded batches are skipped
outright (dedup self-recognition and repo-delta exactly-once keys are
batch-scoped, so a crashed batch re-runs safely). The raw shard is still
deleted only after the *whole shard* passes the final verification sweep.
With `false`, uploads happen once per shard and `checkpoint_after`
stage-checkpoints drive resume instead.

### Shard states

```
pending → downloading → downloaded → processing → processed
       → uploading → uploaded → verified → completed
```
plus `download_failed / processing_failed / upload_failed /
verification_failed` (auto-retried with exponential backoff, recycled to the
right earlier state) and `skipped` (source unavailable).

### Resume rules (enforced by `ucc/resume.py` at every startup)

| crash situation | on restart |
|---|---|
| uploaded & verified | skipped entirely |
| processed, upload failed/crashed | retry upload only — **never re-download**; files already on the Hub with matching checksums are skipped |
| downloaded, processing not finished | resume processing from the raw shard, from the **last completed stage checkpoint** (`processing.checkpoint_after`) |
| crashed during upload | Hub checked first: valid checksum ⇒ marked verified; incomplete ⇒ safe retry |
| raw file missing | Hub + recorded checksums checked first; re-download **only** if no verified output exists |
| completed | never touched again (idempotent: deterministic shard ids, output paths, content hashes) |

Every global side effect (exact-hash index, MinHash LSH index, repo
aggregates) is idempotent — `INSERT OR IGNORE` on deterministic ids, repo
deltas applied exactly-once per shard inside a single transaction — so
re-running any stage after a crash is always safe even without a checkpoint.

---

## Setup

```bash
cd unified-code-corpus
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -e '.[tokens,dev]'
```

**1) Credentials & dataset name → `.env`** (gitignored; `.env.example` is the
committed template):

```bash
HF_TOKEN=hf_...                                # write-access token
UCC_TARGET_REPO=your-username/your-dataset     # output dataset repo
# optional: UCC_WORKSPACE, UCC_HF_MODE=mock, AWS_* (for the-stack-v2 content)
```

The file is loaded automatically by every `ucc` command; variables already
set in your shell win over `.env`, and `.env` wins over `config.yaml`.
Accept the terms of `bigcode/the-stack-v2` + `bigcode/starcoderdata` on
huggingface.co if you want the gated sources; AWS credentials (env or
`~/.aws/credentials`) additionally enable the-stack-v2 content.

**2) Settings → `config.yaml`** (the default config for every command):
shard size in GB (`shard.target_gb/min_gb/max_gb`, fractions allowed), queue
depth and workers, dedup thresholds, license actions, secret/quality/complexity rules,
per-source enable/caps, run scale (`prototype.max_total_shards`). Every knob
is commented in the file. Settings marked `[DATA-AFFECTING]` are part of the
config hash — changing them mid-corpus requires `--allow-config-change`;
operational knobs (queue, caps, hf.*, paths) can be changed freely between
runs. Presets: `configs/prototype.yaml`, `configs/full.yaml`.

macOS note: python.org builds can lack SSL certs for `urllib`; this pipeline
uses `requests`/`huggingface_hub`+`certifi` throughout, so no extra step is
needed.

## Running

```bash
ucc run             # start OR resume — same command (uses ./config.yaml)
ucc status          # states + statistics
ucc verify-remote   # audit uploads vs checksums
ucc retry-failed    # reset retry budgets
ucc finalize        # global stats + repos table + dataset card

ucc run -c configs/prototype.yaml   # or run a preset instead
```

`Ctrl-C` once = graceful stop (finish current stage, checkpoint); twice =
hard exit — both are fully resumable. `kill -9`, OOM, power failure: also
resumable, that's what the crash tests prove.

### Live terminal output

Everything streams to the terminal (and to `workspace/logs/pipeline.log`)
with live percentages:

```
OVERALL  37.5% — 3/8 shards done | slots 4/5 | {...} | records out 1,204,331 | tokens ≈482,110,905 | ...
[starcoderdata-000001] downloaded python/train-00003.parquet (412.0 MB) — shard download  61.2% (file 2/4)
[the_stack_v2-000000] SWH fetch train-00000[rg 0-4]:  48.0% (60,000/125,000 blobs, 1,912/s, eta 34s)
[starcoderdata-000001] dedup_near (MinHash/LSH):  71.3% (145,720/204,310 records, 8,412/s, eta 7s)
[starcoderdata-000001] uploaded data/full/... — shard upload 100.0% (531.8/531.8 MB, file 3/3)
```

- `OVERALL` line cadence: `queue.status_interval_s` (default 10 s in
  config.yaml). Its percentage is of currently-enumerated shards.
- Per-task line cadence: `queue.progress_log_interval_s` (default 5 s).
- HF transfers additionally render byte-level tqdm bars on a TTY; silence
  them with `HF_HUB_DISABLE_PROGRESS_BARS=1` when piping logs to a file.

## Prototype-v0.1 checklist → how each point is validated

1. **Streaming source loading** — enumeration reads file listings/parquet
   footers only; content arrives shard-by-shard, never the whole dataset.
2. **1–2 GB shards** — `shard.target_gb/max_gb` grouping (file-level) and
   row-group splitting (the-stack-v2 derives its per-shard file budget from
   the same GB target).
3. **Max 5-shard local queue** — `CapacityGauge` (`queue.max_local_shards`),
   primed from disk state after a crash. `tests/test_gauge.py`.
4. **Concurrent download + processing** — 3 producer threads + consumer
   thread, watch the interleaved log lines / `ucc status`.
5. **Automatic HF upload** — immediately after `finalize`, per shard.
6. **Upload verification** — size + LFS sha256 (or git blob sha1) compared
   before a shard may advance.
7. **Automatic raw-file cleanup** — only on `verified → completed`;
   `scripts/crash_test.py` asserts raw dirs are gone afterwards.
8. **Deduplication** — exact SHA-256 (global index) + MinHash/LSH near-dedup
   across shards. `tests/test_dedup.py`.
9. **Repository reconstruction** — repo-contiguous output ordering +
   cross-shard `repos` table + `repos/repositories.parquet` at finalize.
10. **License filtering** — SPDX normalization, most-restrictive-wins,
    configurable keep/flag/drop. Strong copyleft dropped by default.
11. **Secret detection** — pattern battery + entropy + carrier-file drops;
    `tests/test_secrets.py`. Logs are scrubbed too (`logging_utils`).
12. **Checkpoint/resume after forced interruption** —
    `python scripts/crash_test.py` injects hard crashes at 8 points
    (download / stages / post-process / mid-upload / pre-cleanup), restarts,
    and asserts completion **without duplicated work** and byte-identical
    hub state on an idempotent re-run (mock hub, real source data).

Scale to `configs/full.yaml` only after all of the above pass.

## Output layout (deterministic, idempotent)

```
data/full/{source}/part-000001.parquet         code+docs; near-dups RETAINED,
                                               annotated (is_near_duplicate,
                                               near_dup_cluster)
                                               (per-batch mode appends -b0000,
                                                -b0001, ... per uploaded batch)
data/high_quality/{source}/part-*.parquet      permissive, flag-free, dedup'd
data/commits/{source}/part-*.parquet           real git commits
data/issues/{source}/part-*.parquet            real GitHub issues
excluded/{source}/part-*.parquet               audit report — metadata+reason
                                               only, never content
stats/shards/{shard_id}.json · stats/global.json
repos/repositories.parquet                     consolidated repo-level table
provenance/multi_source.parquet                content seen in >1 source
README.md                                      generated dataset card
```

Frontend / backend / database / infrastructure / full-stack / testing /
configuration views are column-addressable (`layer`, `repo_category`,
`technologies`, `repo_complexity`) — no data duplication. The dataset card
declares HF `configs` for `full`, `high_quality`, `commits`, `issues`,
`repository_level`.

### Record schema (main columns)

`id, record_type, content, content_sha256, size_bytes, token_count,
line/length metrics, repo_name, repo_url, path, language, license,
detected_licenses, license_status, commit_id, stars, created_at,
source_dataset, source_datasets, source_shard, source_record_id, layer,
technologies, repo_category, file_complexity, repo_complexity,
quality_score, quality_flags, secrets_redacted, pii_redacted,
is_near_duplicate, near_dup_cluster, pipeline_version, config_hash`

## Design notes & caveats

- **Provenance on cross-source duplicates.** The canonical copy may already
  be uploaded when a later source re-encounters its content, so the merged
  source list lives in the manifest and is exported as
  `provenance/multi_source.parquet` (uploaded parquet files are never
  rewritten).
- **Partial repo captures.** Streaming sources interleave repositories
  across shards; per-shard repo views are partial samples. The cross-shard
  `repos` table consolidates counts; rows are marked
  `capture=partial_stream_sample`.
- **Commits/issues are exempt from near-dedup** — meaningful historical
  versions are never removed. Commit records are a deterministic
  serialization of the source's real fields (subject/message/before/after),
  never synthesized.
- **Licensing ≠ redistribution.** Records keep their license provenance and
  strong-copyleft/unknown handling is configurable, but inclusion here is
  not a redistribution grant — review BigCode, Software Heritage/Inria and
  Common Pile terms before publishing the output repo publicly
  (`hf.private: true` by default).
- **Config hash.** Data-affecting settings are hashed into every shard;
  changing them mid-corpus aborts unless `--allow-config-change` is passed.
- **Known layout quirk:** the Common Pile gzip streams can end abruptly
  (transport closes before gzip EOF bookkeeping); the reader treats
  `EOFError/ValueError/OSError` at the tail as end-of-shard and logs it.

## Repo layout

```
.env               your HF token + dataset name (gitignored; template: .env.example)
config.yaml        main config — shard size, queue, dedup/license/quality knobs
configs/           presets: prototype.yaml, full.yaml
scripts/           crash_test.py (spec point 12), run_prototype.sh
src/ucc/
  cli.py           ucc run/status/enumerate/verify-remote/retry-failed/finalize
  orchestrator.py  producer-consumer core, retries, stall detection
  manifest.py      SQLite state machine + dedup/LSH/repo indexes (crash-safe)
  shard_queue.py   the 5-slot capacity gauge
  resume.py        startup reconciliation (the resume rules table above)
  uploader.py      idempotent upload + checksum verification
  hf_remote.py     RealHub / MockHub with identical verify semantics
  sources/         one adapter per dataset + shared readers/downloader
  processing/      the 11 pipeline stages + checkpointing runner
  stats.py         aggregation, status, finalize exports, dataset card
tests/             unit tests (manifest, gauge, dedup, secrets, quality)
```
