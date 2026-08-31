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
