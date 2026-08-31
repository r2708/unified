"""Logging setup with a secret-scrubbing formatter.

Spec requirement: never expose secrets in logs. Every formatted log line is
passed through a battery of token-shaped regexes before it reaches any
handler; matches are replaced with <REDACTED:type>.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

_SCRUB_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("stripe_key", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{16,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{16,}=*")),
    ("basic_auth_url", re.compile(r"://[^/\s:@]{1,64}:[^/\s@]{1,128}@")),
    ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
]


def scrub(text: str) -> str:
    for name, pattern in _SCRUB_PATTERNS:
        text = pattern.sub(f"<REDACTED:{name}>", text)
    return text


class ScrubbingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("ucc")
    if getattr(root, "_ucc_configured", False):
        return root
    root.setLevel(level)
    root.propagate = False

    fmt = ScrubbingFormatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Third-party warnings/errors (huggingface_hub, boto3/botocore, urllib3,
    # fsspec, ...) propagate to the ABSOLUTE root logger — attach the same
    # scrubbed handlers there so they also reach the terminal and the log
    # file. The absolute root's default WARNING level keeps libraries quiet
    # below warnings, and the "ucc" logger has propagate=False so nothing
    # prints twice.
    absolute_root = logging.getLogger()
    for handler in root.handlers:
        absolute_root.addHandler(handler)

    root._ucc_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ucc.{name}")
