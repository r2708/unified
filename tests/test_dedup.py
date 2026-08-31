import pytest

from tests.conftest import make_rec
from ucc.processing.dedup_exact import ExactDedupStage


def test_exact_dedup_merges_sources(make_ctx):
    ctx = make_ctx()
    content = "def add(a, b):\n    return a + b\n"
    a = make_rec(content, repo_name="org/one", path="add.py")
    b = make_rec(content, repo_name="org/two", path="copy_of_add.py")
    out = ExactDedupStage().run(sorted([a, b], key=lambda r: r["id"]), ctx)
    assert len(out) == 1
    assert len(ctx.excluded) == 1
    assert ctx.excluded[0]["reason"].startswith("exact_duplicate")

    # Cross-shard, cross-source duplicate: dropped, provenance merged.
    ctx2 = make_ctx(shard_id="other-000002", source="othersource", seq=2)
    c = make_rec(content, repo_name="org/three", path="third.py",
                 source="othersource", shard="other-000002")
    out2 = ExactDedupStage().run([c], ctx2)
    assert out2 == []
    sources = ctx.manifest.exact_sources_for(c["content_sha256"])
    assert sources == ["othersource", "testsource"]


def test_near_dedup_annotates_and_exempts_history(make_ctx):
    pytest.importorskip("datasketch")
    pytest.importorskip("xxhash")
    from ucc.processing.dedup_near import NearDedupStage

    ctx = make_ctx()
    base = "\n".join(
        f"def handler_{i}(request):\n    payload = request.json\n"
        f"    return process(payload, retries={i})" for i in range(40)
    )
    near_copy = base.replace("retries=7", "retries=9") + "\n# fork tweak\n"
    a = make_rec(base, repo_name="org/one", path="app.py")
    b = make_rec(near_copy, repo_name="org/fork", path="app.py")
    commit = make_rec(base, repo_name="org/one", path=None, record_type="commit")

    rows = sorted([a, b], key=lambda r: r["id"]) + [commit]
    out = NearDedupStage().run(rows, ctx)
    assert len(out) == 3  # annotate mode keeps everything

    flagged = [r for r in out if r["is_near_duplicate"]]
    assert len(flagged) == 1
    canonical_id = min(a["id"], b["id"])
    assert flagged[0]["near_dup_cluster"] == canonical_id
    # historical record types are never near-deduped
    assert all(not r["is_near_duplicate"] for r in out if r["record_type"] == "commit")
