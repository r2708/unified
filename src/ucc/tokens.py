"""Approximate token counting.

Default is a fast heuristic (code averages ~4 bytes/token for BPE
tokenizers). Set processing.token_counter: tiktoken in the config (and
install the `tokens` extra) for exact cl100k counts.
"""

from __future__ import annotations

_ENCODER = None
_TIKTOKEN_FAILED = False


def _get_encoder():
    global _ENCODER, _TIKTOKEN_FAILED
    if _ENCODER is None and not _TIKTOKEN_FAILED:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_FAILED = True
    return _ENCODER


def count_tokens(text: str, mode: str = "heuristic") -> int:
    if not text:
        return 0
    if mode == "tiktoken":
        enc = _get_encoder()
        if enc is not None:
            return len(enc.encode(text, disallowed_special=()))
    # Heuristic: bytes/4, floor 1. Close enough for corpus statistics.
    return max(1, len(text.encode("utf-8", errors="replace")) // 4)
