"""Deterministic reconstruction of wrapped ICD table rows (Audit 0069 §3).

Audit 0068 recovered 673 of 2,777 damaged titles and left 2,104. The dominant reason was
not a missing title but a *wrapped* one: ``pdftotext -layout`` renders one logical table row
across several physical lines, and the previous recovery only ever read the row's first
line. C14.2 illustrates it exactly - the anchor line ends with ``U ác tính ở vòng bạch`` and
the remainder, ``huyết Waldeyer``, sits on the next physical line.

The layout is recoverable because it is columnar and the column origins are exact. In the
C14.2 block every segment of the Vietnamese title column begins at x=260 and nothing else
does, so a title can be rejoined by x-position alone - no semantics, no guessing, no model.

Two rules keep the join honest:

* **Exact x-alignment.** A continuation segment joins a column only when it starts at that
  column's exact origin. Neighbouring columns (the chapter and block breadcrumbs at x=92 and
  x=181) therefore cannot bleed into a concept title.
* **Stop at the first gap.** A column closes the moment a line carries nothing at its
  origin. A later line that happens to reuse the x belongs to a different logical row, and
  resuming across the gap is exactly how a title would acquire another concept's words.

Anything that cannot be joined under those rules stays damaged. A lower recovery count with
trustworthy titles is worth more than a high one with uncertain mappings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A column segment: a run of text with no internal double space. ``pdftotext -layout``
#: separates columns by two or more spaces, so this is the column granularity.
_SEGMENT = re.compile(r"\S+(?: \S+)*")

#: A logical row always opens with `<sequence> <chapter numeral>`.
_ROW_START = re.compile(r"^\s*(\d+)\s+([IVXLC]+)\s")

_CODE_TOKEN = re.compile(r"^[A-Z]\d{2}(\.\d+)?[†*]?$")

#: How far a wrapped segment may sit from its column origin. Zero: the observed layout puts
#: continuations at the exact origin, and any tolerance is a chance to absorb a neighbour.
X_TOLERANCE = 0


@dataclass(frozen=True, slots=True)
class Column:
    """One reconstructed table column with the provenance of every line it drew from."""

    x: int
    text: str
    source_lines: tuple[int, ...]

    @property
    def wrapped(self) -> bool:
        return len(self.source_lines) > 1


def segments(line: str) -> list[tuple[int, str]]:
    """``(x_origin, text)`` for every column segment on one physical line."""
    return [
        (match.start(), match.group().strip())
        for match in _SEGMENT.finditer(line)
        if match.group().strip()
    ]


def is_row_start(line: str) -> bool:
    return bool(_ROW_START.match(line))


def reconstruct_row(text_lines: list[str], index: int) -> list[Column]:
    """Rejoin the wrapped continuation lines of the logical row anchored at ``index``.

    ``index`` is 0-based into ``text_lines``; ``Column.source_lines`` are 1-based line
    numbers so they can be quoted directly as source provenance.
    """
    anchor = segments(text_lines[index])
    if not anchor:
        return []

    parts: dict[int, list[str]] = {x: [text] for x, text in anchor}
    lines_used: dict[int, list[int]] = {x: [index + 1] for x, _ in anchor}
    # A code column is never wrapped - it is one short token - and joining anything onto it
    # would dissolve the `<dotted> <undotted>` pair that identifies whose row this is.
    open_columns = {x for x, text in anchor if not _CODE_TOKEN.match(text)}

    for offset in range(index + 1, len(text_lines)):
        if not open_columns:
            break
        line = text_lines[offset]
        # A blank line ends the block; a new row-start line belongs to the next concept.
        if not line.strip() or is_row_start(line):
            break
        found = {x: text for x, text in segments(line)}
        for x in sorted(open_columns):
            match = _aligned(found, x)
            # No segment at this origin: the column is finished. Resuming after a gap is
            # how a title would pick up an unrelated row's words, so it never resumes.
            if match is None or _CODE_TOKEN.match(match):
                open_columns.discard(x)
                continue
            parts[x].append(match)
            lines_used[x].append(offset + 1)

    return [
        Column(x=x, text=" ".join(parts[x]).strip(), source_lines=tuple(lines_used[x]))
        for x in sorted(parts)
    ]


def _aligned(found: dict[int, str], x: int) -> str | None:
    if x in found:
        return found[x]
    for delta in range(1, X_TOLERANCE + 1):
        for candidate in (x - delta, x + delta):
            if candidate in found:
                return found[candidate]
    return None


__all__ = ["X_TOLERANCE", "Column", "is_row_start", "reconstruct_row", "segments"]
