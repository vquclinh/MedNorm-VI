"""Governed ICD label quality.

The competition ICD source was extracted from a PDF, and some concept titles arrived as
instructions, mid-phrase fragments or single glyphs. `is_damaged_title` is the predicate the
title-repair work settled on, kept here because the ontology facts shown to the reranker must
never present a damaged label as if it were a concept name.
"""

from __future__ import annotations

import re

NOTE_MARKERS: tuple[str, ...] = (
    "bao gồm",
    "loại trừ",
    "incl.",
    "excl.",
    "includes",
    "excludes",
    "lưu ý",
    "tham khảo",
    "ghi chú",
    "chú ý",
    "note:",
)

_LEADING_GLYPHS = re.compile(r"^[\s\-+*†‡•·\[\]()]+")

_LEADING_GLYPHS = re.compile(r"^[\s\-+*†‡•·\[\]()]+")


def strip_glyphs(value: str) -> str:
    return _LEADING_GLYPHS.sub("", value or "").strip()


def is_note(value: str) -> bool:
    """True when the string is an ICD instruction rather than a concept title."""
    bare = strip_glyphs(value).casefold()
    return any(bare.startswith(marker) for marker in NOTE_MARKERS)


def is_damaged_title(value: str) -> bool:
    """Instruction, mid-phrase fragment, or too short to be a concept name.

    Deliberately conservative: a clean title is never rewritten for style.
    """
    text = (value or "").strip()
    if not text:
        return True
    if is_note(text):
        return True
    if text.rstrip().endswith((",", ":", "(", "-", "+", "/")):
        return True
    return len(strip_glyphs(text)) <= 6


__all__ = ["NOTE_MARKERS", "is_damaged_title", "is_note", "strip_glyphs"]
