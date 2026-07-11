"""Conservative sentence-like segmentation within a single line's content.

Does NOT assume every period ends a sentence: a period between digits (decimal)
or after a known abbreviation/initial is not a boundary. Sentence spans keep
their terminating punctuation; inter-sentence whitespace belongs to no sentence
(only lines are required to tile the document).
"""

from __future__ import annotations

from ..schemas.spans import Span

# Representative abbreviations (normalized, no trailing dot). Not exhaustive.
ABBREVIATIONS: frozenset[str] = frozenset({
    "mg", "ml", "mcg", "g", "kg", "l", "mmol", "mmhg", "u", "ui", "iu",
    "po", "iv", "im", "sc", "bid", "tid", "qid", "qd", "qhs", "qam", "prn",
    "dr", "mr", "mrs", "ms", "prof", "no", "vs", "etc", "approx",
    "bs", "bn", "tp", "th", "vd", "vv",
})


def _preceding_word(text: str, dot_index: int) -> str:
    j = dot_index
    while j > 0 and (text[j - 1].isalnum()):
        j -= 1
    return text[j:dot_index].casefold()


def _is_hard_boundary(text: str, k: int, terminator: str) -> bool:
    if terminator != ".":
        return True
    # Decimal: digit . digit
    if 0 < k < len(text) - 1 and text[k - 1].isdigit() and text[k + 1].isdigit():
        return False
    word = _preceding_word(text, k)
    if word in ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isalpha():  # single-letter initial, e.g. "A."
        return False
    return True


def segment_sentences(
    content: str, content_start: int, terminators: tuple[str, ...]
) -> list[Span]:
    """Return absolute sentence-like spans within one line's content."""
    term_set = set(terminators)
    spans: list[Span] = []
    seg_start: int | None = None
    for k, ch in enumerate(content):
        if seg_start is None and not ch.isspace():
            seg_start = k
        if ch in term_set and seg_start is not None and _is_hard_boundary(content, k, ch):
            end = k + 1
            spans.append(Span(content_start + seg_start, content_start + end))
            seg_start = None
    if seg_start is not None:
        end = len(content)
        while end > seg_start and content[end - 1].isspace():
            end -= 1
        if end > seg_start:
            spans.append(Span(content_start + seg_start, content_start + end))
    return spans


__all__ = ["ABBREVIATIONS", "segment_sentences"]
