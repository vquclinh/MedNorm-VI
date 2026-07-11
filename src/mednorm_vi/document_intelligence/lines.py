"""Line + paragraph segmentation that preserves newline delimiters and offsets.

A line's content span EXCLUDES its terminator; the terminator (``\\n`` or
``\\r\\n``) is preserved as an explicit delimiter. Content lines plus newline
delimiters tile the whole document, so no region is lost. Blank lines are kept as
zero-length content lines.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinePiece:
    """One line: content ``[content_start, content_end)`` + optional terminator."""

    index: int
    content_start: int
    content_end: int
    newline_start: int
    newline_end: int  # == newline_start when the last line has no terminator

    @property
    def has_newline(self) -> bool:
        return self.newline_end > self.newline_start


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A maximal run of non-blank lines (blank lines separate paragraphs)."""

    start: int
    end: int
    line_indices: tuple[int, ...]


def split_lines(text: str) -> list[LinePiece]:
    """Split into lines preserving exact terminators (LF and CRLF)."""
    pieces: list[LinePiece] = []
    pos = 0
    idx = 0
    n = len(text)
    while pos < n:
        nl = text.find("\n", pos)
        if nl == -1:
            pieces.append(LinePiece(idx, pos, n, n, n))
            pos = n
            idx += 1
            break
        # CRLF: the terminator starts at the preceding \r if present.
        term_start = nl - 1 if nl > pos and text[nl - 1] == "\r" else nl
        pieces.append(LinePiece(idx, pos, term_start, term_start, nl + 1))
        pos = nl + 1
        idx += 1
    if not pieces:
        # Empty document → a single empty content line, no terminator.
        pieces.append(LinePiece(0, 0, 0, 0, 0))
    elif text.endswith("\n"):
        # A trailing newline implies a final empty line after it.
        pieces.append(LinePiece(idx, n, n, n, n))
    return pieces


def is_blank(text: str, piece: LinePiece) -> bool:
    return text[piece.content_start : piece.content_end].strip() == ""


def paragraphs(
    text: str, pieces: list[LinePiece], break_before: frozenset[int] = frozenset()
) -> list[Paragraph]:
    """Group consecutive non-blank lines into paragraphs.

    Blank lines separate paragraphs. ``break_before`` (e.g. section-header line
    indices) forces a paragraph boundary so a paragraph never crosses a section
    start.
    """
    result: list[Paragraph] = []
    run: list[LinePiece] = []

    def flush() -> None:
        if run:
            result.append(Paragraph(
                start=run[0].content_start,
                end=run[-1].content_end,
                line_indices=tuple(p.index for p in run),
            ))
            run.clear()

    for piece in pieces:
        if is_blank(text, piece):
            flush()
        else:
            if piece.index in break_before:
                flush()
            run.append(piece)
    flush()
    return result


__all__ = ["LinePiece", "Paragraph", "split_lines", "is_blank", "paragraphs"]
