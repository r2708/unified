from ucc.states import ShardState


def test_transition_guard(manifest, cfg):
    manifest.upsert_shard_spec("s-000001", "src", {}, 1, 0, "0.1.0", cfg.config_hash)
    assert manifest.transition("s-000001", [ShardState.PENDING], ShardState.DOWNLOADING)
    # Illegal from-state must be rejected (exactly-once guard).
    assert not manifest.transition("s-000001", [ShardState.PENDING], ShardState.DOWNLOADED)
    assert manifest.transition("s-000001", [ShardState.DOWNLOADING], ShardState.DOWNLOADED)
    assert manifest.get_shard("s-000001")["state"] == "downloaded"


def test_upsert_is_idempotent(manifest, cfg):
    assert manifest.upsert_shard_spec("s-000001", "src", {}, 1, 0, "0.1.0", cfg.config_hash)
    manifest.transition("s-000001", [ShardState.PENDING], ShardState.COMPLETED)
    # Re-enumeration must never reset an existing shard.
    assert not manifest.upsert_shard_spec("s-000001", "src", {}, 1, 0, "0.1.0", cfg.config_hash)
    assert manifest.get_shard("s-000001")["state"] == "completed"


def test_claim_next_orders_and_excludes_claimed(manifest, cfg):
    for i in (2, 1, 3):
        manifest.upsert_shard_spec(f"s-{i:06d}", "src", {}, i, 0, "0.1.0", cfg.config_hash)
    first = manifest.claim_next([ShardState.PENDING], "w1", to_state=ShardState.DOWNLOADING)
    assert first["shard_id"] == "s-000001"
    second = manifest.claim_next([ShardState.PENDING], "w2")
    assert second["shard_id"] == "s-000002"
    # Claimed shards are invisible to other workers.
    third = manifest.claim_next([ShardState.PENDING], "w3")
    assert third["shard_id"] == "s-000003"
    assert manifest.claim_next([ShardState.PENDING], "w4") is None


def test_exact_dedup_index(manifest):
    res = manifest.exact_seen_or_add_many([("h1", "rec-a", "shard-1", "src-A")])
    assert res["h1"] == (True, "rec-a")
    # Same record re-runs after a crash: not a duplicate of itself.
    res = manifest.exact_seen_or_add_many([("h1", "rec-a", "shard-1", "src-A")])
    assert res["h1"] == (True, "rec-a")
    # Different record, same content: duplicate; source provenance merged.
    res = manifest.exact_seen_or_add_many([("h1", "rec-b", "shard-2", "src-B")])
    assert res["h1"] == (False, "rec-a")
    assert manifest.exact_sources_for("h1") == ["src-A", "src-B"]
    merged = list(manifest.iter_multi_source_hashes())
    assert merged == [("h1", "rec-a", ["src-A", "src-B"])]


def test_exact_dedup_within_one_batch(manifest):
    """Sequential semantics inside a single batched call: the first sighting
    of a hash is canonical, later items with the same hash are duplicates,
    and their sources merge onto the row inserted in that same batch."""
    res = manifest.exact_seen_or_add_many([
        ("h2", "rec-a", "shard-1", "src-A"),   # first sighting -> canonical
        ("h2", "rec-b", "shard-1", "src-A"),   # same batch, same source
        ("h2", "rec-c", "shard-2", "src-B"),   # same batch, new source
        ("h3", "rec-d", "shard-1", "src-A"),   # unrelated new hash
    ])
    assert res["h2"] == (False, "rec-a")       # last verdict for the hash
    assert res["h3"] == (True, "rec-d")
    assert manifest.exact_sources_for("h2") == ["src-A", "src-B"]
    assert manifest.exact_sources_for("h3") == ["src-A"]
    # The single-record variant delegates to the batch path.
    assert manifest.exact_seen_or_add("h2", "rec-e", "shard-3", "src-C") == (
        False, "rec-a",
    )
    assert manifest.exact_sources_for("h2") == ["src-A", "src-B", "src-C"]
    # Self-recognition still holds after the batch insert.
    assert manifest.exact_seen_or_add("h2", "rec-a", "shard-1", "src-A") == (
        True, "rec-a",
    )


def test_batched_repo_reads_writes_and_iteration(manifest):
    for i in range(7):
        manifest.apply_repo_deltas(
            f"shard-{i}", {f"org/repo{i}": {"n_files": i + 1, "languages": {"Go": 1}}}
        )
    got = manifest.get_repos([f"org/repo{i}" for i in range(7)] + ["org/missing"])
    assert set(got) == {f"org/repo{i}" for i in range(7)}
    assert got["org/repo3"]["n_files"] == 4
    assert got["org/repo3"]["languages"] == {"Go": 1}

    manifest.update_repo_computed_many(
        [("org/repo0", 42.5, None), ("org/repo1", None, "backend")]
    )
    assert manifest.get_repo("org/repo0")["complexity"] == 42.5
    assert manifest.get_repo("org/repo1")["category"] == "backend"
    # COALESCE semantics: None never clears an existing value.
    manifest.update_repo_computed_many([("org/repo0", None, "cli")])
    repo0 = manifest.get_repo("org/repo0")
    assert repo0["complexity"] == 42.5 and repo0["category"] == "cli"

    # iter_repos streams every row in key order (batch smaller than count).
    keys = [r["repo_key"] for r in manifest.iter_repos(batch=3)]
    assert keys == sorted(f"org/repo{i}" for i in range(7))


def test_iter_multi_source_hashes_paginates(manifest):
    items = []
    for i in range(12):
        items.append((f"h-{i:02d}", f"rec-{i}", "shard-1", "src-A"))
        if i % 2 == 0:  # every even hash also seen from a second source
            items.append((f"h-{i:02d}", f"other-{i}", "shard-2", "src-B"))
    manifest.exact_seen_or_add_many(items)
    # batch smaller than the table forces multiple pagination pages.
    got = list(manifest.iter_multi_source_hashes(batch=5))
    assert [h for h, _, _ in got] == [f"h-{i:02d}" for i in range(0, 12, 2)]
    assert all(srcs == ["src-A", "src-B"] for _, _, srcs in got)


def test_band_candidates_many(manifest):
    manifest.add_minhash_batch(
        sig_rows=[("rec-a", "shard-1", b"\x01" * 8), ("rec-b", "shard-2", b"\x02" * 8)],
        band_rows=[(0, b"h0", "rec-a", "shard-1"), (0, b"h0", "rec-b", "shard-2"),
                   (1, b"h1", "rec-a", "shard-1")],
    )
    # Own-shard rows are excluded (crash re-run self-recognition).
    got = manifest.band_candidates_many({0: [b"h0", b"h0"], 1: [b"h1", b"hX"]}, "shard-1")
    assert got == {(0, b"h0"): ["rec-b"]}
    got = manifest.band_candidates_many({0: [b"h0"]}, "shard-3")
    assert sorted(got[(0, b"h0")]) == ["rec-a", "rec-b"]
    assert manifest.get_sigs(["rec-a"]) == {"rec-a": b"\x01" * 8}


def test_apply_repo_deltas_exactly_once(manifest):
    delta = {"org/repo": {"n_files": 3, "n_tokens": 100, "languages": {"Python": 3}}}
    assert manifest.apply_repo_deltas("shard-1", delta)
    # A crashed-and-resumed stage re-applies: must be a no-op.
    assert not manifest.apply_repo_deltas("shard-1", delta)
    repo = manifest.get_repo("org/repo")
    assert repo["n_files"] == 3
    assert repo["languages"] == {"Python": 3}
    # A different shard's delta merges additively.
    assert manifest.apply_repo_deltas("shard-2", {"org/repo": {"n_files": 2}})
    assert manifest.get_repo("org/repo")["n_files"] == 5
