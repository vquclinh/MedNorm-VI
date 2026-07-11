"""Small deterministic Unicode helpers for L1.

No transformation of ``original_text`` happens here — these are pure predicates
and constants used by loading, normalization, and tokenization.
"""

from __future__ import annotations

import unicodedata

BOM = "﻿"

# Bullet marker code points recognized by list detection.
BULLET_CHARS: frozenset[str] = frozenset("-–—*•·▪‣◦")


def has_bom(text: str) -> bool:
    return text.startswith(BOM)


def detect_newline_style(text: str) -> str:
    """Return ``LF`` | ``CRLF`` | ``MIXED`` | ``NONE`` without altering text."""
    crlf = text.count("\r\n")
    total_lf = text.count("\n")
    lone_lf = total_lf - crlf  # every CRLF contains one LF
    lone_cr = text.count("\r") - crlf
    styles = set()
    if crlf:
        styles.add("CRLF")
    if lone_lf > 0:
        styles.add("LF")
    if lone_cr > 0:
        styles.add("CR")
    if not styles:
        return "NONE"
    if len(styles) == 1:
        return next(iter(styles))
    return "MIXED"


def is_whitespace(ch: str) -> bool:
    return ch.isspace()


def is_punctuation(ch: str) -> bool:
    """True for Unicode punctuation/symbol code points (category P* or S*)."""
    if not ch:
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def strip_accents(text: str) -> str:
    """Decompose and drop combining marks (auxiliary accent-insensitive view)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


__all__ = [
    "BOM",
    "BULLET_CHARS",
    "has_bom",
    "detect_newline_style",
    "is_whitespace",
    "is_punctuation",
    "strip_accents",
]
