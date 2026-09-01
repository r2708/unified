"""Persistent, crash-safe pipeline state: the checkpoint/manifest database.

SQLite in WAL mode with synchronous=NORMAL. A single connection guarded by a
process-wide lock serves every thread; every multi-statement mutation runs in
an explicit BEGIN IMMEDIATE transaction, so a hard kill (OOM, SIGKILL) can
never leave a half-written or lost state. Power loss / OS crash can drop the
most recent commits but never corrupts the DB (WAL guarantee) — and every
consumer re-derives from re-checkable ground truth (Hub checksums, fsynced
progress files, idempotent INSERT OR IGNORE indexes), so the worst case is
redoing a little work. The one place a lost commit could silently matter —
marking an uploaded batch done in the fsynced batch_progress.json while its
dedup-index commits die with the page cache — is closed by the runner calling
sync() (a WAL checkpoint, i.e. one real fsync per batch) before writing the
progress file.

Tables
------
shards          one row per shard: full lifecycle + checksums + stats
exact_hashes    global exact-dedup index + cross-source provenance merge
minhash_bands   persistent LSH band index for near-dedup across shards
minhash_sigs    full MinHash signatures for candidate verification
repos           incremental repository-level aggregates (cross-shard)
kv              pipeline metadata (schema version, enumeration flags, ...)
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

from ucc.constants import MANIFEST_SCHEMA_VERSION
from ucc.states import ShardState

_SHARD_COLUMNS = """
    shard_id TEXT PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    seq_index INTEGER NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    state TEXT NOT NULL,
    claimed_by TEXT,
    stage_checkpoint TEXT,
    checkpoint_path TEXT,
    local_raw_dir TEXT,
    local_work_dir TEXT,
    local_processed_dir TEXT,
    raw_size_bytes INTEGER,
    raw_checksum TEXT,
    processed_checksum TEXT,
    processed_size_bytes INTEGER,
    records_in INTEGER,
    records_out INTEGER,
    token_count INTEGER,
    hf_dest_paths TEXT,
    error TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    pipeline_version TEXT,
    config_hash TEXT,
    stats_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
"""

_DDL = [
    f"CREATE TABLE IF NOT EXISTS shards ({_SHARD_COLUMNS})",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_shards_source_seq"
    " ON shards(source_dataset, seq_index)",
    "CREATE INDEX IF NOT EXISTS idx_shards_state ON shards(state)",
    """CREATE TABLE IF NOT EXISTS exact_hashes (
        content_sha256 TEXT PRIMARY KEY,
        canonical_record_id TEXT NOT NULL,
        canonical_shard_id TEXT NOT NULL,
        sources TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS minhash_bands (
        band INTEGER NOT NULL,
        band_hash BLOB NOT NULL,
        record_id TEXT NOT NULL,
        shard_id TEXT NOT NULL,
        PRIMARY KEY (band, band_hash, record_id)
    ) WITHOUT ROWID""",
    """CREATE TABLE IF NOT EXISTS minhash_sigs (
        record_id TEXT PRIMARY KEY,
        shard_id TEXT NOT NULL,
        sig BLOB NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS repos (
        repo_key TEXT PRIMARY KEY,
        repo_url TEXT,
        n_files INTEGER NOT NULL DEFAULT 0,
        n_tokens INTEGER NOT NULL DEFAULT 0,
        n_commits INTEGER NOT NULL DEFAULT 0,
        n_issues INTEGER NOT NULL DEFAULT 0,
        n_deps INTEGER NOT NULL DEFAULT 0,
        languages TEXT NOT NULL DEFAULT '{}',
        layers TEXT NOT NULL DEFAULT '{}',
        technologies TEXT NOT NULL DEFAULT '[]',
        licenses TEXT NOT NULL DEFAULT '[]',
        has_tests INTEGER NOT NULL DEFAULT 0,
        has_ci INTEGER NOT NULL DEFAULT 0,
        has_infrastructure INTEGER NOT NULL DEFAULT 0,
        category TEXT,
        complexity REAL,
        updated_at TEXT
    )""",
    "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Manifest:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None, timeout=60.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL in WAL mode: commits skip the per-transaction fsync (large
        # I/O win, especially on network/cloud disks) while staying corruption
        # -proof; durability barriers happen at batch boundaries via sync().
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        with self._txn():
            for ddl in _DDL:
                self._conn.execute(ddl)
        self.set_kv("manifest_schema_version", str(MANIFEST_SCHEMA_VERSION))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def sync(self) -> None:
        """Durability barrier: fsync the WAL so every commit made so far
        survives power loss. Called once per uploaded batch / stage
        checkpoint — the batched replacement for synchronous=FULL's
        per-transaction fsync."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    @contextmanager
    def _txn(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ------------------------------------------------------------------ kv
    def get_kv(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # -------------------------------------------------------------- shards
    def upsert_shard_spec(
        self,
        shard_id: str,
        source_dataset: str,
        source_ref: dict,
        seq_index: int,
        priority: int,
        pipeline_version: str,
        config_hash: str,
    ) -> bool:
        """Insert a newly enumerated shard. Existing rows are left untouched
        (idempotent re-enumeration). Returns True if inserted."""
        now = now_iso()
        with self._txn() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO shards
                   (shard_id, source_dataset, source_ref, seq_index, priority,
                    state, pipeline_version, config_hash, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    shard_id,
                    source_dataset,
                    json.dumps(source_ref, sort_keys=True),
                    seq_index,
                    priority,
                    ShardState.PENDING.value,
                    pipeline_version,
                    config_hash,
                    now,
                    now,
                ),
            )
        return cur.rowcount == 1

    def get_shard(self, shard_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM shards WHERE shard_id=?", (shard_id,)
            ).fetchone()
        return dict(row) if row else None

    def all_shards(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shards ORDER BY priority, seq_index, source_dataset"
            ).fetchall()
        return [dict(r) for r in rows]

    def shards_in_states(self, states: Iterable[ShardState | str]) -> list[dict]:
        vals = [s.value if isinstance(s, ShardState) else s for s in states]
        marks = ",".join("?" * len(vals))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM shards WHERE state IN ({marks})"
                " ORDER BY priority, seq_index, source_dataset",
                vals,
            ).fetchall()
        return [dict(r) for r in rows]

    def counts_by_state(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state, COUNT(*) AS n FROM shards GROUP BY state"
            ).fetchall()
        return {r["state"]: r["n"] for r in rows}

    def progress_totals(self) -> dict:
        """Cheap aggregates for the live progress line."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS done,"
                " SUM(COALESCE(records_out, 0)) AS records,"
                " SUM(COALESCE(token_count, 0)) AS tokens"
                " FROM shards WHERE state='completed'"
            ).fetchone()
            total = self._conn.execute(
                "SELECT COUNT(*) AS n FROM shards"
            ).fetchone()["n"]
        return {
            "completed": row["done"] or 0,
            "records_out": row["records"] or 0,
            "tokens": row["tokens"] or 0,
            "total_shards": total,
        }

    def set_fields(self, shard_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._txn() as conn:
            conn.execute(
                f"UPDATE shards SET {cols} WHERE shard_id=?",
                (*fields.values(), shard_id),
            )

    def transition(
        self,
        shard_id: str,
        allowed_from: Sequence[ShardState],
        to_state: ShardState,
        **fields: Any,
    ) -> bool:
        """Atomically move a shard between states; fails (returns False) if
        the shard is not in one of `allowed_from` — the exactly-once guard."""
        from_vals = [s.value for s in allowed_from]
        marks = ",".join("?" * len(from_vals))
        fields["updated_at"] = now_iso()
        extra_cols = "".join(f", {k}=?" for k in fields)
        with self._txn() as conn:
            cur = conn.execute(
                f"UPDATE shards SET state=?{extra_cols}"
                f" WHERE shard_id=? AND state IN ({marks})",
                (to_state.value, *fields.values(), shard_id, *from_vals),
            )
        return cur.rowcount == 1

    def claim_next(
        self,
        from_states: Sequence[ShardState],
        worker_id: str,
        to_state: ShardState | None = None,
    ) -> dict | None:
        """Atomically claim the next unclaimed shard in deterministic order."""
        from_vals = [s.value for s in from_states]
        marks = ",".join("?" * len(from_vals))
        now = now_iso()
        with self._txn() as conn:
            row = conn.execute(
                f"""SELECT * FROM shards
                    WHERE state IN ({marks}) AND claimed_by IS NULL
                      AND (next_retry_at IS NULL OR next_retry_at <= ?)
                    ORDER BY priority, seq_index, source_dataset LIMIT 1""",
                (*from_vals, now),
            ).fetchone()
            if row is None:
                return None
            new_state = (to_state.value if to_state else row["state"])
            conn.execute(
                "UPDATE shards SET claimed_by=?, state=?, updated_at=? WHERE shard_id=?",
                (worker_id, new_state, now, row["shard_id"]),
            )
        shard = dict(row)
        shard["claimed_by"] = worker_id
        shard["state"] = new_state
        return shard

    def release_claim(self, shard_id: str) -> None:
        self.set_fields(shard_id, claimed_by=None)

    def clear_all_claims(self) -> None:
        with self._txn() as conn:
            conn.execute("UPDATE shards SET claimed_by=NULL")

    def update_shard_stats(self, shard_id: str, stats: dict) -> None:
        """Overwrite (idempotent) the per-shard stats blob. Global statistics
        are always computed by summing these blobs, never by incrementing
        global counters — so a re-run after a crash can never double count."""
        self.set_fields(shard_id, stats_json=json.dumps(stats, sort_keys=True, default=str))

    # -------------------------------------------------- exact deduplication
    def exact_seen_or_add(
        self, content_sha256: str, record_id: str, shard_id: str, source: str
    ) -> tuple[bool, str]:
        """Returns (is_new, canonical_record_id). When the hash was already
        seen, the new source is merged into the canonical provenance set
        (set semantics — idempotent on re-run)."""
        return self.exact_seen_or_add_many(
            [(content_sha256, record_id, shard_id, source)]
        )[content_sha256]

    def exact_seen_or_add_many(
        self, items: list[tuple[str, str, str, str]]
    ) -> dict[str, tuple[bool, str]]:
        """Batch exact-dedup lookup/insert: one transaction (one fsync) per
        record batch, and one bulk pre-fetch per ~500 hashes instead of one
        SELECT round trip per record.

        Semantics match processing the items one at a time, in order: the
        first sighting of a hash (in the DB or earlier in this same batch)
        is canonical, a record re-encountering itself (same shard re-run
        after a crash) is not a duplicate, and every duplicate's source is
        merged into the canonical row's provenance set.

        items: (content_sha256, record_id, shard_id, source)
        returns: {content_sha256: (is_new, canonical_record_id)}
        """
        results: dict[str, tuple[bool, str]] = {}
        if not items:
            return results
        with self._txn() as conn:
            # Bulk pre-fetch of every hash in the batch (chunked IN-queries).
            # Probes and inserts run in SORTED key order: on an index bigger
            # than the page cache, ascending B-tree sweeps beat random point
            # probes; order has no effect on the results.
            existing: dict[str, tuple[str, str, list[str]]] = {}
            hashes = sorted(dict.fromkeys(item[0] for item in items))
            for i in range(0, len(hashes), 500):
                chunk = hashes[i : i + 500]
                marks = ",".join("?" * len(chunk))
                for row in conn.execute(
                    "SELECT content_sha256, canonical_record_id,"
                    " canonical_shard_id, sources FROM exact_hashes"
                    f" WHERE content_sha256 IN ({marks})",
                    chunk,
                ):
                    existing[row[0]] = (row[1], row[2], json.loads(row[3]))

            inserts: list[tuple[str, str, str, str]] = []
            merged: set[str] = set()  # hashes whose provenance gained a source
            for content_sha256, record_id, shard_id, source in items:
                entry = existing.get(content_sha256)
                if entry is None:
                    # First sighting anywhere: this record becomes canonical
                    # (also visible to later duplicates in this same batch).
                    existing[content_sha256] = (record_id, shard_id, [source])
                    inserts.append(
                        (content_sha256, record_id, shard_id, json.dumps([source]))
                    )
                    results[content_sha256] = (True, record_id)
                    continue
                canonical_id, canonical_shard, sources = entry
                if canonical_id == record_id and canonical_shard == shard_id:
                    # A record re-encountering itself (same shard re-run
                    # after a crash) is not a duplicate.
                    results[content_sha256] = (True, record_id)
                    continue
                if source not in sources:
                    sources.append(source)
                    merged.add(content_sha256)
                results[content_sha256] = (False, canonical_id)

            if inserts:
                inserts.sort()
                conn.executemany(
                    "INSERT INTO exact_hashes VALUES (?,?,?,?)", inserts
                )
            if merged:
                # After the inserts, so a source merged onto a row first seen
                # in this very batch lands too (same as sequential order).
                conn.executemany(
                    "UPDATE exact_hashes SET sources=? WHERE content_sha256=?",
                    [
                        (json.dumps(sorted(existing[h][2])), h)
                        for h in sorted(merged)
                    ],
                )
        return results

    def iter_multi_source_hashes(self, batch: int = 5000):
        """Yield (content_sha256, canonical_record_id, sources) for content
        seen in more than one source dataset — the durable cross-source
        provenance merge exported by `finalize`.

        Streams with keyset pagination: exact_hashes can reach hundreds of
        millions of rows, so the table is never fetchall()'d into RAM."""
        last = ""
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT content_sha256, canonical_record_id, sources"
                    " FROM exact_hashes WHERE content_sha256 > ?"
                    " ORDER BY content_sha256 LIMIT ?",
                    (last, int(batch)),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                sources = json.loads(row["sources"])
                if len(sources) > 1:
                    yield row["content_sha256"], row["canonical_record_id"], sources
            last = rows[-1]["content_sha256"]

    def exact_sources_for(self, content_sha256: str) -> list[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT sources FROM exact_hashes WHERE content_sha256=?",
                (content_sha256,),
            ).fetchone()
        return json.loads(row["sources"]) if row else []

    # --------------------------------------------------- near deduplication
    def band_candidates_many(
        self, band_to_hashes: dict[int, Sequence[bytes]], exclude_shard: str
    ) -> dict[tuple[int, bytes], list[str]]:
        """Cross-shard LSH candidate lookup for a whole chunk of records at
        once: one IN-query per band per ~500 hashes instead of one query per
        record per band. Only (band, hash) pairs with hits appear in the
        result."""
        out: dict[tuple[int, bytes], list[str]] = {}
        with self._lock:
            for band, hashes in band_to_hashes.items():
                uniq = list(dict.fromkeys(hashes))
                for i in range(0, len(uniq), 500):
                    chunk = uniq[i : i + 500]
                    marks = ",".join("?" * len(chunk))
                    rows = self._conn.execute(
                        "SELECT band_hash, record_id FROM minhash_bands"
                        f" WHERE band=? AND band_hash IN ({marks}) AND shard_id != ?",
                        (band, *chunk, exclude_shard),
                    ).fetchall()
                    for r in rows:
                        out.setdefault((band, r["band_hash"]), []).append(r["record_id"])
        return out

    def get_sigs(self, record_ids: Sequence[str]) -> dict[str, bytes]:
        if not record_ids:
            return {}
        out: dict[str, bytes] = {}
        with self._lock:
            for i in range(0, len(record_ids), 500):
                chunk = record_ids[i : i + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT record_id, sig FROM minhash_sigs WHERE record_id IN ({marks})",
                    chunk,
                ).fetchall()
                out.update({r["record_id"]: r["sig"] for r in rows})
        return out

    def add_minhash_batch(
        self, sig_rows: list[tuple[str, str, bytes]], band_rows: list[tuple[int, bytes, str, str]]
    ) -> None:
        """INSERT OR IGNORE everywhere -> re-running a stage is idempotent."""
        with self._txn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO minhash_sigs(record_id, shard_id, sig) VALUES (?,?,?)",
                [(rid, sid, sig) for rid, sid, sig in sig_rows],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO minhash_bands(band, band_hash, record_id, shard_id)"
                " VALUES (?,?,?,?)",
                band_rows,
            )

    # ------------------------------------------------------ repo aggregates
    def _merge_repo_in_txn(self, conn, repo_key: str, delta: dict) -> None:
        """Merge one repo delta (counts summed, histograms merged, sets
        unioned, booleans OR-ed). Must be called inside an open transaction."""
        row = conn.execute("SELECT * FROM repos WHERE repo_key=?", (repo_key,)).fetchone()
        if row is None:
            current = {
                "repo_url": None, "n_files": 0, "n_tokens": 0, "n_commits": 0,
                "n_issues": 0, "n_deps": 0, "languages": {}, "layers": {},
                "technologies": [], "licenses": [], "has_tests": 0, "has_ci": 0,
                "has_infrastructure": 0, "category": None, "complexity": None,
            }
        else:
            current = dict(row)
            for key in ("languages", "layers", "technologies", "licenses"):
                current[key] = json.loads(current[key])

        for key in ("n_files", "n_tokens", "n_commits", "n_issues", "n_deps"):
            current[key] += int(delta.get(key, 0))
        for key in ("languages", "layers"):
            for name, count in (delta.get(key) or {}).items():
                current[key][name] = current[key].get(name, 0) + count
        for key in ("technologies", "licenses"):
            current[key] = sorted(set(current[key]) | set(delta.get(key) or []))
        for key in ("has_tests", "has_ci", "has_infrastructure"):
            current[key] = int(bool(current[key]) or bool(delta.get(key)))
        if delta.get("repo_url"):
            current["repo_url"] = delta["repo_url"]
        if delta.get("category"):
            current["category"] = delta["category"]
        if delta.get("complexity") is not None:
            current["complexity"] = float(delta["complexity"])

        conn.execute(
            """INSERT INTO repos (repo_key, repo_url, n_files, n_tokens, n_commits,
                   n_issues, n_deps, languages, layers, technologies, licenses,
                   has_tests, has_ci, has_infrastructure, category, complexity,
                   updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(repo_key) DO UPDATE SET
                   repo_url=excluded.repo_url, n_files=excluded.n_files,
                   n_tokens=excluded.n_tokens, n_commits=excluded.n_commits,
                   n_issues=excluded.n_issues, n_deps=excluded.n_deps,
                   languages=excluded.languages, layers=excluded.layers,
                   technologies=excluded.technologies, licenses=excluded.licenses,
                   has_tests=excluded.has_tests, has_ci=excluded.has_ci,
                   has_infrastructure=excluded.has_infrastructure,
                   category=excluded.category, complexity=excluded.complexity,
                   updated_at=excluded.updated_at""",
            (
                repo_key, current["repo_url"], current["n_files"],
                current["n_tokens"], current["n_commits"], current["n_issues"],
                current["n_deps"],
                json.dumps(current["languages"], sort_keys=True),
                json.dumps(current["layers"], sort_keys=True),
                json.dumps(current["technologies"]),
                json.dumps(current["licenses"]),
                current["has_tests"], current["has_ci"],
                current["has_infrastructure"], current["category"],
                current["complexity"], now_iso(),
            ),
        )

    def apply_repo_deltas(self, shard_id: str, deltas: dict[str, dict]) -> bool:
        """Apply one shard's repository deltas EXACTLY ONCE.

        The merge of every delta and the setting of the applied-flag happen
        in one transaction, so a crash mid-merge either applies all of them
        or none — re-running the stage after a crash can never double count.
        Returns False if this shard's deltas were already applied.
        """
        flag_key = f"repos_merged:{shard_id}"
        with self._txn() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (flag_key,)).fetchone()
            if row is not None:
                return False
            for repo_key, delta in deltas.items():
                self._merge_repo_in_txn(conn, repo_key, delta)
            conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?)", (flag_key, now_iso())
            )
        return True

    def update_repo_computed(
        self, repo_key: str, complexity: float | None = None, category: str | None = None
    ) -> None:
        """Overwrite derived repo fields (idempotent — safe to re-run)."""
        self.update_repo_computed_many([(repo_key, complexity, category)])

    def update_repo_computed_many(
        self, rows: list[tuple[str, float | None, str | None]]
    ) -> None:
        """Batch variant of update_repo_computed: one transaction (one fsync)
        for a whole batch's repos instead of one per repo.

        rows: (repo_key, complexity or None, category or None)
        """
        if not rows:
            return
        now = now_iso()
        with self._txn() as conn:
            conn.executemany(
                "UPDATE repos SET complexity=COALESCE(?, complexity),"
                " category=COALESCE(?, category), updated_at=? WHERE repo_key=?",
                [(c, cat, now, key) for key, c, cat in rows],
            )

    @staticmethod
    def _decode_repo_row(row) -> dict:
        repo = dict(row)
        for key in ("languages", "layers", "technologies", "licenses"):
            repo[key] = json.loads(repo[key])
        return repo

    def get_repo(self, repo_key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM repos WHERE repo_key=?", (repo_key,)
            ).fetchone()
        return self._decode_repo_row(row) if row else None

    def get_repos(self, repo_keys: Sequence[str]) -> dict[str, dict]:
        """Batch variant of get_repo: chunked IN-queries instead of one query
        per repo. Missing keys are simply absent from the result."""
        out: dict[str, dict] = {}
        if not repo_keys:
            return out
        with self._lock:
            for i in range(0, len(repo_keys), 500):
                chunk = list(repo_keys[i : i + 500])
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT * FROM repos WHERE repo_key IN ({marks})", chunk
                ).fetchall()
                for row in rows:
                    out[row["repo_key"]] = self._decode_repo_row(row)
        return out

    def iter_repos(self, batch: int = 1000):
        """Stream every repo row (one ordered scan, decoded in batches —
        not one query per repo)."""
        last_key = ""
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM repos WHERE repo_key > ?"
                    " ORDER BY repo_key LIMIT ?",
                    (last_key, int(batch)),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._decode_repo_row(row)
            last_key = rows[-1]["repo_key"]
