from tests.conftest import make_rec
from ucc.processing.quality_filter import QualityFilterStage


def run_stage(rows, ctx):
    return QualityFilterStage().run(rows, ctx)


def test_minified_js_dropped_but_keeplisted_file_survives(make_ctx):
    ctx = make_ctx()
    long_line = "var a=1;" * 2000  # single ~16k-char line
    minified = make_rec(long_line, path="static/app.min.js", language="JavaScript")
    dockerfile = make_rec(
        "RUN apt-get update && apt-get install -y " + " ".join(f"pkg{i}" for i in range(200)),
        path="Dockerfile", language="Dockerfile",
    )
    out = run_stage([minified, dockerfile], ctx)
    assert [r["path"] for r in out] == ["Dockerfile"]
    assert any(e["reason"] == "quality" and "minified" in e["detail"] for e in ctx.excluded)


def test_vendored_path_dropped(make_ctx):
    ctx = make_ctx()
    rec = make_rec("module.exports = 1;\n", path="node_modules/lib/index.js",
                   language="JavaScript")
    assert run_stage([rec], ctx) == []


def test_empty_dropped_lockfile_flagged_not_dropped(make_ctx):
    ctx = make_ctx()
    empty = make_rec("  \n", path="src/__init__.py", language="Python")
    lock = make_rec('{"lockfileVersion": 2, "dependencies": {}}\n',
                    path="package-lock.json", language="JSON")
    out = run_stage([empty, lock], ctx)
    assert len(out) == 1 and out[0]["path"] == "package-lock.json"
    assert "lockfile" in out[0]["quality_flags"]


def test_metrics_computed(make_ctx):
    ctx = make_ctx()
    rec = make_rec("def f(x):\n    return x + 1\n", path="a.py", language="Python")
    out = run_stage([rec], ctx)
    assert out[0]["line_count"] == 2
    assert out[0]["max_line_length"] == len("    return x + 1")
    assert out[0]["alnum_ratio"] > 0.9
