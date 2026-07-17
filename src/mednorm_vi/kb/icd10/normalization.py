"""ICD-10 code normalization (dotted <-> undotted, specificity).

Whether the organizer expects dotted or undotted output, and what specificity is
required, are UNRESOLVED (see UNRES-ICD-DOTTED, UNRES-ICD-SPECIFICITY). This
module only provides reversible format conversions; it picks no output policy.
"""

from __future__ import annotations

import re

_CATEGORY = re.compile(r"^[A-Z]\d{2}")


def clean(code: str) -> str:
    """Uppercase and strip whitespace (keeps dots)."""
    return code.strip().upper().replace(" ", "")


def to_undotted(code: str) -> str:
    """Remove the dot: ``A09.9`` -> ``A099``."""
    return clean(code).replace(".", "")


def to_dotted(code: str) -> str:
    """Insert the dot after the 3-char category: ``A099`` -> ``A09.9``.

    A 3-char (or shorter) category is returned unchanged. Idempotent on already
    dotted input.
    """
    u = to_undotted(code)
    if len(u) <= 3:
        return u
    return f"{u[:3]}.{u[3:]}"


def is_category(code: str) -> bool:
    """True for a bare 3-character category (e.g. ``A09``)."""
    return len(to_undotted(code)) == 3


def specificity(code: str) -> int:
    """Number of characters beyond the 3-char category (0 for a category)."""
    return max(0, len(to_undotted(code)) - 3)


def parent_code(code: str) -> str | None:
    """The immediate parent (one char less specific), or None for a category."""
    u = to_undotted(code)
    if len(u) <= 3:
        return None
    return u[:-1]


def chapter_letter(code: str) -> str:
    """The leading letter (chapter proxy), or '' if malformed."""
    u = to_undotted(code)
    return u[0] if u and u[0].isalpha() else ""


def is_wellformed(code: str) -> bool:
    return _CATEGORY.match(to_undotted(code)) is not None


__all__ = [
    "clean",
    "to_undotted",
    "to_dotted",
    "is_category",
    "specificity",
    "parent_code",
    "chapter_letter",
    "is_wellformed",
]
