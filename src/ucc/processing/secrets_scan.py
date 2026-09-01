"""Stage 7 — secret and PII removal.

Redacts credentials (API keys, tokens, private keys, connection-string
passwords, high-entropy assignments) and obvious PII (emails, public IPs)
in place; drops credential-carrier files (.env, id_rsa, *.pem, ...) and
files saturated with secrets. Only aggregate counts are ever logged or kept
in stats — never the matched values.

Performance shape: every pattern in the battery starts with a distinctive
literal, so a single combined anchor regex pre-screens each record and the
19-pattern battery only runs on records that could possibly match (~3x on
clean files, which are the overwhelming majority). The per-record scan is a
pure function of the content, so it can run on the processing.cpu_workers
process pool; all record mutation, drop decisions and stats stay in the
parent and results are identical to a serial run.
"""

from __future__ import annotations

import math
import posixpath
import re
from functools import partial

from ucc.hashing import sha256_bytes
from ucc.parallel import parallel_map, resolve_cpu_workers
from ucc.processing.base import ShardContext, Stage
from ucc.tokens import count_tokens

_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_key_block", re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----.*?-----END [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----",
        re.DOTALL)),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b")),
    ("gitlab_pat", re.compile(r"\bglpat-[A-Za-z0-9\-_]{20,}\b")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,250}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_key", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{20,247}\b")),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b")),
    ("twilio_key", re.compile(r"\bSK[0-9a-fA-F]{32}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("pypi_token", re.compile(r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("azure_account_key", re.compile(r"\bAccountKey=[A-Za-z0-9+/=]{60,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("url_credentials", re.compile(
        r"\b(?P<scheme>[a-z][a-z0-9+.\-]*)://(?P<user>[^/\s:@'\"]{1,64}):(?P<pw>[^/\s@'\"]{1,128})@")),
]

# One literal anchor per battery pattern: any string the battery can match is
# guaranteed to contain a match of this regex, so a single cheap scan skips
# the whole battery on clean files. Keep in sync with _SECRET_PATTERNS —
# tests/test_secrets.py asserts the coverage for every pattern.
_ANCHOR_RX = re.compile(
    r"-----BEGIN"
    r"|AKIA|ASIA|ABIA|ACCA"
    r"|gh[opusr]_|github_pat_|glpat-"
    r"|hf_|xox[baprs]-|AIza|k_live_|SG\.|SK[0-9a-fA-F]"
    r"|npm_|pypi-AgEI|sk-|AccountKey=|eyJ"
    r"|://[^/\s@]+@"
)

_ASSIGNED_SECRET_RX = re.compile(
    r"""(?ix)\b(secret|token|passwd|password|api[_\-]?key|apikey|auth[_\-]?key|
        access[_\-]?key|private[_\-]?key|client[_\-]?secret)\b
        \s*[:=]\s*["']([A-Za-z0-9+/=_\-]{20,128})["']""",
    re.VERBOSE,
)

_EMAIL_RX = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_IPV4_RX = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")

_SECRET_FILE_BASENAMES_RX = re.compile(
    r"(?i)^(\.env(\..+)?|id_rsa[^/]*|id_ed25519[^/]*|id_dsa[^/]*|\.netrc"
    r"|credentials(\.json)?|service[_\-]?account.*\.json|.*\.(pem|p12|pfx|key|keystore|jks))$"
)
_SECRET_FILE_ALLOW_RX = re.compile(r"(?i)(example|sample|template|test|mock|fixture|dummy)")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _is_nonroutable_or_versionlike(octets: tuple[str, ...]) -> bool:
    try:
        a, b, c, d = (int(x) for x in octets)
    except ValueError:
        return True
    if any(x > 255 for x in (a, b, c, d)):
        return True  # not a real IP (version strings like 10.1.2.300)
    if a in (0, 10, 127) or a == 255:
        return True
    if a == 192 and b == 168:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 169 and b == 254:
        return True
    if (a, b, c, d) in ((8, 8, 8, 8), (1, 1, 1, 1)):
        return True  # ubiquitous public resolvers, useless to redact
    if max(a, b, c, d) <= 20:
        return True  # likely a version string (e.g. 1.2.3.4)
    return False


def _url_repl(m: re.Match) -> str:
    return f"{m.group('scheme')}://{m.group('user')}:<REDACTED_PASSWORD>@"


def _scan_content(
    content: str, redact_emails: bool, redact_ips: bool
) -> tuple[str | None, list[tuple[str, int]], int, int, int]:
    """Pure per-record scan (process-pool worker).

    Returns (redacted content — or None when nothing was hit, per-battery-type
    counts, high-entropy-assignment hits, email hits, ip hits). Never touches
    shared state; only counts and the (possibly) redacted text travel back.
    """
    type_counts: list[tuple[str, int]] = []
    if _ANCHOR_RX.search(content) is not None:
        for name, pattern in _SECRET_PATTERNS:
            if name == "url_credentials":
                content, n = pattern.subn(_url_repl, content)
            else:
                content, n = pattern.subn(f"<REDACTED_{name.upper()}>", content)
            if n:
                type_counts.append((name, n))

    assign_hits = 0

    def _assign_repl(m: re.Match) -> str:
        nonlocal assign_hits
        value = m.group(2)
        if _shannon_entropy(value) >= 4.0:
            assign_hits += 1
            return m.group(0).replace(value, "<REDACTED_SECRET>")
        return m.group(0)

    content = _ASSIGNED_SECRET_RX.sub(_assign_repl, content)

    email_hits = 0
    if redact_emails:
        content, email_hits = _EMAIL_RX.subn("<EMAIL>", content)

    ip_hits = 0
    if redact_ips:
        ip_cell = [0]

        def _ip_repl(m: re.Match) -> str:
            if _is_nonroutable_or_versionlike(m.groups()):
                return m.group(0)
            ip_cell[0] += 1
            return "<IP_ADDRESS>"

        content = _IPV4_RX.sub(_ip_repl, content)
        ip_hits = ip_cell[0]

    hit_anything = bool(type_counts or assign_hits or email_hits or ip_hits)
    return (content if hit_anything else None, type_counts, assign_hits,
            email_hits, ip_hits)


class SecretsScanStage(Stage):
    name = "secrets_scan"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        scfg = ctx.cfg.secrets
        if not scfg.get("enabled", True):
            return rows
        max_hits = int(scfg.get("max_hits_per_file", 20))
        redact_emails = bool(scfg.get("redact_emails", True))
        redact_ips = bool(scfg.get("redact_ips", True))
        drop_env_like = bool(scfg.get("drop_env_like_files", True))
        token_mode = ctx.cfg.processing.token_counter
        cpu_workers = resolve_cpu_workers(ctx.cfg.path("processing.cpu_workers", 1))

        # Pass 1 — carrier-file drops (path-only, cheap, stays in the parent).
        survivors: list[dict] = []
        for rec in rows:
            path = rec.get("path") or ""
            base = posixpath.basename(path)
            if (
                drop_env_like
                and base
                and _SECRET_FILE_BASENAMES_RX.match(base)
                and not _SECRET_FILE_ALLOW_RX.search(path)
            ):
                ctx.exclude(rec, "secret_carrier_file", detail=base)
                continue
            survivors.append(rec)

        # Pass 2 — the content scans (pure per-record work, parallelizable).
        scan_fn = partial(
            _scan_content, redact_emails=redact_emails, redact_ips=redact_ips
        )
        results = parallel_map(
            scan_fn, [rec["content"] for rec in survivors], cpu_workers
        )

        # Pass 3 — apply results: drops, redactions, stats (parent only).
        out: list[dict] = []
        live = ctx.progress("secrets_scan", total=len(rows))
        if len(rows) > len(survivors):
            live.update(len(rows) - len(survivors))
        for rec, (changed, type_counts, assign_hits, email_hits, ip_hits) in zip(
            survivors, results
        ):
            live.update()
            secret_hits = assign_hits
            for name, n in type_counts:
                secret_hits += n
                ctx.bump(f"secrets.type.{name}", n)
            if assign_hits:
                ctx.bump("secrets.type.high_entropy_assignment", assign_hits)
            pii_hits = email_hits + ip_hits

            if secret_hits > max_hits:
                ctx.exclude(rec, "secret_dense", detail=f"{secret_hits} redactions")
                ctx.bump("secrets.files_dropped_dense")
                continue

            if changed is not None:
                data = changed.encode("utf-8", errors="replace")
                rec["content"] = changed
                rec["content_sha256"] = sha256_bytes(data)
                rec["size_bytes"] = len(data)
                rec["token_count"] = count_tokens(changed, token_mode,
                                                  size_bytes=len(data))
            rec["secrets_redacted"] = secret_hits
            rec["pii_redacted"] = pii_hits
            if secret_hits:
                ctx.bump("secrets.redacted_total", secret_hits)
                ctx.bump("secrets.files_with_secrets")
            if pii_hits:
                ctx.bump("secrets.pii_redacted_total", pii_hits)
            out.append(rec)

        live.close()
        ctx.bump("secrets.records_out", len(out))
        return out
