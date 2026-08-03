"""Stable line-addressed view of a source document (0081).

The model is never allowed to produce a character offset. It is given lines with stable
identifiers and returns `(line_id, text, occurrence)`; this module is the only thing that
turns that triple into offsets, and it does so by finding the text **in the source**. A
proposal whose text is not literally present cannot be aligned and is therefore rejected -
that is what makes hallucinated spans structurally impossible rather than merely unlikely.

Line ids are derived from the document, so they are stable across runs and independent of
anything the model says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LINE_ID_WIDTH = 3


def line_id_for(index: int) -> str:
    """`0 -> "L001"`. One-based so the first line is not `L000`."""
    return f"L{index + 1:0{LINE_ID_WIDTH}d}"


@dataclass(frozen=True, slots=True)
class LineView:
    """One source line and where it lives in the document."""

    line_id: str
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentView:
    """Immutable line-addressed view. `source` is never modified, only sliced."""

    source: str
    lines: tuple[LineView, ...]

    @staticmethod
    def build(source: str) -> DocumentView:
        lines: list[LineView] = []
        offset = 0
        for index, raw in enumerate(source.split("\n")):
            lines.append(
                LineView(
                    line_id=line_id_for(index), index=index,
                    start=offset, end=offset + len(raw), text=raw,
                )
            )
            offset += len(raw) + 1  # the newline that split() consumed
        return DocumentView(source=source, lines=tuple(lines))

    @property
    def by_id(self) -> dict[str, LineView]:
        return {line.line_id: line for line in self.lines}

    def render(self, *, skip_blank: bool = True) -> str:
        """`L001| text` per line. What the proposer sees, and nothing else."""
        return "\n".join(
            f"{line.line_id}| {line.text}"
            for line in self.lines
            if line.text.strip() or not skip_blank
        )

    def line_of(self, offset: int) -> LineView | None:
        for line in self.lines:
            if line.start <= offset <= line.end:
                return line
        return None

    def slice(self, start: int, end: int) -> str:
        return self.source[start:end]


#: Why an alignment failed. Counted, never silently swallowed.
REJECT_EMPTY = "empty_text"
REJECT_UNKNOWN_LINE = "unknown_line_id"
REJECT_NOT_IN_SOURCE = "text_not_in_source"
REJECT_AMBIGUOUS = "ambiguous_without_line"
REJECT_OCCURRENCE = "occurrence_out_of_range"


@dataclass(frozen=True, slots=True)
class Alignment:
    """A resolved span, or a reason there is none. Offsets come from the source alone."""

    start: int = -1
    end: int = -1
    line_id: str = ""
    how: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.start >= 0 and self.end > self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start, "end": self.end, "line_id": self.line_id,
            "how": self.how, "reason": self.reason,
        }


ALIGN_LINE = "line_occurrence"
ALIGN_UNIQUE = "unique_in_document"


def _occurrences(haystack: str, needle: str, base: int = 0) -> list[int]:
    found: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        found.append(base + start)
        start = haystack.find(needle, start + 1)
    return found


def align(
    document: DocumentView, line_id: str, text: str, occurrence: int = 0
) -> Alignment:
    """Resolve `(line_id, text, occurrence)` to offsets, or explain why it cannot be.

    The named line is authoritative. If the model named a line that does not contain the
    text, one fallback is allowed: the text occurring **exactly once** in the whole
    document, which is unambiguous and therefore still deterministic. Anything else is a
    rejection - guessing between several possible positions would be inventing a span.
    """
    if not text or not text.strip():
        return Alignment(reason=REJECT_EMPTY)
    if text != text.strip():
        # Leading/trailing whitespace is the model's formatting, not part of the mention.
        return align(document, line_id, text.strip(), occurrence)

    lines = document.by_id
    line = lines.get(line_id)
    if line is not None:
        hits = _occurrences(line.text, text, line.start)
        if hits:
            if occurrence < 0 or occurrence >= len(hits):
                return Alignment(reason=REJECT_OCCURRENCE, line_id=line_id)
            start = hits[occurrence]
            return Alignment(
                start=start, end=start + len(text), line_id=line_id, how=ALIGN_LINE
            )
    elif line_id:
        # A named-but-unknown line is still allowed the unique-text fallback below; the
        # reason is recorded if that fails too.
        pass

    document_hits = _occurrences(document.source, text)
    if not document_hits:
        return Alignment(
            reason=REJECT_NOT_IN_SOURCE if line is not None else REJECT_UNKNOWN_LINE,
            line_id=line_id,
        )
    if len(document_hits) != 1:
        return Alignment(reason=REJECT_AMBIGUOUS, line_id=line_id)
    start = document_hits[0]
    resolved = document.line_of(start)
    return Alignment(
        start=start, end=start + len(text),
        line_id=resolved.line_id if resolved else "", how=ALIGN_UNIQUE,
    )


__all__ = [
    "ALIGN_LINE",
    "ALIGN_UNIQUE",
    "LINE_ID_WIDTH",
    "REJECT_AMBIGUOUS",
    "REJECT_EMPTY",
    "REJECT_NOT_IN_SOURCE",
    "REJECT_OCCURRENCE",
    "REJECT_UNKNOWN_LINE",
    "Alignment",
    "DocumentView",
    "LineView",
    "align",
    "line_id_for",
]
