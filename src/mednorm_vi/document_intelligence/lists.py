"""Numbered and bulleted list-marker detection.

Detects a leading marker (``1.``, ``1)``, ``(1)``, ``a.``, ``a)``, ``-``, ``–``,
``—``, ``*``, ``•`` …) and separates the marker span from the item-content span.
List numbers/bullets are kept as their own structure and must NOT be folded into
later entity spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import L1Config


@dataclass(frozen=True, slots=True)
class ListMatch:
    marker_start: int  # absolute
    marker_end: int
    content_start: int
    content_end: int
    indent: int


def _leading_ws(text: str) -> int:
    i = 0
    while i < len(text) and text[i] in " \t":
        i += 1
    return i


def detect_list_item(content: str, content_start: int, config: L1Config) -> ListMatch | None:
    """Detect a list marker at the start of one line's content."""
    indent = _leading_ws(content)
    rest = content[indent:]
    if not rest:
        return None

    marker_len = 0
    # Ordered markers (regex, anchored).
    for pattern in config.ordered_marker_patterns:
        m = re.match(pattern, rest)
        if m and m.end() > 0:
            # Marker must be followed by whitespace or end-of-line to count.
            after = rest[m.end():]
            if after == "" or after[0] in " \t":
                marker_len = m.end()
                break
    # Bullet markers (single char followed by whitespace).
    if marker_len == 0 and rest[0] in set(config.bullet_markers):
        if len(rest) == 1 or rest[1] in " \t":
            marker_len = 1

    if marker_len == 0:
        return None

    marker_start = content_start + indent
    marker_end = marker_start + marker_len
    # Content begins after the marker's trailing whitespace.
    c = indent + marker_len
    while c < len(content) and content[c] in " \t":
        c += 1
    return ListMatch(
        marker_start=marker_start,
        marker_end=marker_end,
        content_start=content_start + c,
        content_end=content_start + len(content),
        indent=indent,
    )


__all__ = ["ListMatch", "detect_list_item"]
