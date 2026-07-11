"""Table-like and key-value row detection.

At L1 these are labelled ONLY as structural ``table_like`` or ``key_value_like``
rows. This module never emits ``TÊN_XÉT_NGHIỆM`` / ``KẾT_QUẢ_XÉT_NGHIỆM`` (or any)
medical entities — that is a later layer's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import L1Config


@dataclass(frozen=True, slots=True)
class RowCell:
    kind: str  # "key" | "value"
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RowMatch:
    row_kind: str  # "table_like" | "key_value_like"
    cells: tuple[RowCell, ...]


def _trim(content: str, start: int, end: int) -> tuple[int, int]:
    while start < end and content[start] in " \t":
        start += 1
    while end > start and content[end - 1] in " \t":
        end -= 1
    return start, end


def detect_row(content: str, content_start: int, config: L1Config) -> RowMatch | None:
    """Detect a structural row within one line's content."""
    stripped = content.strip()
    if not stripped:
        return None

    if "\t" in content:
        cells: list[RowCell] = []
        pos = 0
        for chunk in content.split("\t"):
            s, e = _trim(content, pos, pos + len(chunk))
            if e > s:
                cells.append(RowCell("value", content_start + s, content_start + e))
            pos += len(chunk) + 1
        return RowMatch("table_like", tuple(cells))

    if re.match(config.kv_regex, content):
        colon = content.find(":")
        ks, ke = _trim(content, 0, colon)
        vs, ve = _trim(content, colon + 1, len(content))
        cells = [RowCell("key", content_start + ks, content_start + ke)]
        if ve > vs:
            cells.append(RowCell("value", content_start + vs, content_start + ve))
        return RowMatch("key_value_like", tuple(cells))

    if content.count(";") >= 2:
        return RowMatch("table_like", ())

    return None


__all__ = ["RowCell", "RowMatch", "detect_row"]
