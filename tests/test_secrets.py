"""Secret/PII redaction tests. All 'secrets' below are the standard public
documentation examples / obviously fake values — never real credentials."""

from tests.conftest import make_rec
from ucc.processing.secrets_scan import SecretsScanStage


def run_stage(rows, ctx):
    return SecretsScanStage().run(rows, ctx)


def test_aws_key_redacted(make_ctx):
    ctx = make_ctx()
    # canonical AWS docs example key id
    rec = make_rec("aws_key = 'AKIAIOSFODNN7EXAMPLE'\nprint('hi')\n", path="config.py")
    out = run_stage([rec], ctx)
    assert len(out) == 1
    assert "AKIAIOSFODNN7EXAMPLE" not in out[0]["content"]
    assert "<REDACTED_AWS_ACCESS_KEY_ID>" in out[0]["content"]
    assert out[0]["secrets_redacted"] == 1
    # content hash must track the redacted content
    from ucc.hashing import sha256_text

    assert out[0]["content_sha256"] == sha256_text(out[0]["content"])


def test_private_key_block_redacted(make_ctx):
    ctx = make_ctx()
    fake_key = (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEfakefakefake\n"
        "-----END RSA PRIVATE KEY-----"
    )
    rec = make_rec(f"# deploy helper\nKEY = '''{fake_key}'''\n", path="deploy.py")
    out = run_stage([rec], ctx)
    assert "BEGIN RSA PRIVATE KEY" not in out[0]["content"]
    assert "<REDACTED_PRIVATE_KEY_BLOCK>" in out[0]["content"]


def test_connection_string_password_redacted(make_ctx):
    ctx = make_ctx()
    rec = make_rec(
        "DATABASE_URL = 'postgres://appuser:hunter2pass@db.internal:5432/app'\n",
        path="settings.py",
    )
    out = run_stage([rec], ctx)
    assert "hunter2pass" not in out[0]["content"]
    assert "<REDACTED_PASSWORD>" in out[0]["content"]


def test_email_and_public_ip_redacted_private_ip_kept(make_ctx):
    ctx = make_ctx()
    rec = make_rec(
        "# maintainer: dev@example-corp.com\nHOST = '203.51.44.99'\nLOCAL = '192.168.1.10'\n",
        path="net.py",
    )
    out = run_stage([rec], ctx)
    content = out[0]["content"]
    assert "dev@example-corp.com" not in content and "<EMAIL>" in content
    assert "203.51.44.99" not in content and "<IP_ADDRESS>" in content
    assert "192.168.1.10" in content  # private ranges stay
    assert out[0]["pii_redacted"] == 2


def test_env_file_dropped(make_ctx):
    ctx = make_ctx()
    rec = make_rec("API_KEY=abc123\n", path="backend/.env")
    keep = make_rec("API_KEY=changeme\n", path="backend/.env.example")
    out = run_stage([rec, keep], ctx)
    assert len(out) == 1 and out[0]["path"] == "backend/.env.example"
    assert ctx.excluded and ctx.excluded[0]["reason"] == "secret_carrier_file"


def test_secret_dense_file_dropped(make_ctx):
    ctx = make_ctx()
    lines = "\n".join(f"key{i} = 'AKIA{'A' * 12}{i:04d}'" for i in range(30))
    rec = make_rec(lines, path="keys.py")
    out = run_stage([rec], ctx)
    assert out == []
    assert any(e["reason"] == "secret_dense" for e in ctx.excluded)


# One synthetic (obviously fake) example per battery pattern, used to prove
# the anchor prefilter can never skip a record the battery would redact.
_PATTERN_EXAMPLES = {
    "private_key_block": "-----BEGIN RSA PRIVATE KEY-----\nMIIEfake\n-----END RSA PRIVATE KEY-----",
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "github_token": "ghp_" + "a1" * 18,
    "github_pat": "github_pat_" + "A" * 22,
    "gitlab_pat": "glpat-" + "x" * 20,
    "hf_token": "hf_" + "Q" * 30,
    "slack_token": "xoxb-1234567890",
    "google_api_key": "AIza" + "B" * 35,
    "stripe_key": "sk_live_" + "c" * 24,
    "sendgrid_key": "SG." + "d" * 22 + "." + "e" * 43,
    "twilio_key": "SK" + "0f" * 16,
    "npm_token": "npm_" + "g" * 36,
    "pypi_token": "pypi-AgEIcHlwaS5vcmc" + "h" * 24,
    "openai_key": "sk-" + "i" * 40,
    "anthropic_key": "sk-ant-" + "j" * 24,
    "azure_account_key": "AccountKey=" + "K" * 64,
    "jwt": "eyJ" + "a" * 12 + ".eyJ" + "b" * 12 + "." + "c" * 12,
    "url_credentials": "postgres://appuser:hunter2pass@db.internal/app",
}


def test_anchor_prefilter_covers_every_battery_pattern():
    """The battery only runs when _ANCHOR_RX hits, so every pattern must be
    covered by an anchor — this guards future additions to the battery."""
    from ucc.processing.secrets_scan import _ANCHOR_RX, _SECRET_PATTERNS

    assert set(_PATTERN_EXAMPLES) == {name for name, _ in _SECRET_PATTERNS}
    for name, pattern in _SECRET_PATTERNS:
        sample = f"some code before {_PATTERN_EXAMPLES[name]} and after"
        assert pattern.search(sample), f"example for {name} must match its pattern"
        assert _ANCHOR_RX.search(sample), f"anchor regex does not cover {name}"


def test_anchor_skip_still_redacts_pii(make_ctx):
    """A record with no battery anchors takes the fast path but email/IP
    redaction must still run."""
    ctx = make_ctx()
    rec = make_rec("# contact ops@example-corp.net if 203.51.44.99 is down\n",
                   path="notes.py")
    out = run_stage([rec], ctx)
    content = out[0]["content"]
    assert "<EMAIL>" in content and "<IP_ADDRESS>" in content
    assert out[0]["secrets_redacted"] == 0 and out[0]["pii_redacted"] == 2
