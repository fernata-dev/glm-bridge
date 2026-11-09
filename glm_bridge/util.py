"""Small shared helpers."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token estimate (the upstream reports no usage): ~4 chars/token for
    ASCII, ~2 chars/token for other scripts such as Arabic or Chinese."""
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return max(1, round(ascii_chars / 4 + (len(text) - ascii_chars) / 2))
