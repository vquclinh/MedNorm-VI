"""Deterministic laboratory value/unit/flag/reference extraction (local offsets)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import LabLexicon

_STANDALONE_FLAG = re.compile(r"(?<![\w])(H|L)(?![\w])")
_FLAG_WORD = re.compile(r"(?<![\w])(high|low|cao|thấp)(?![\w])", re.IGNORECASE | re.UNICODE)
_ARROW = re.compile(r"[↑↓]")


@dataclass(frozen=True, slots=True)
class LocalMatch:
    role: str
    start: int
    end: int
    text: str
    normalized: str | None = None


def _norm_number(text: str) -> str:
    return text.replace(",", ".")


def _within(pos: int, span: tuple[int, int] | None) -> bool:
    return span is not None and span[0] <= pos < span[1]


def find_reference(seg: str, lex: LabLexicon) -> LocalMatch | None:
    for pat in lex.reference_res:
        m = pat.search(seg)
        if m is not None:
            return LocalMatch("reference_range", m.start(), m.end(), m.group(0))
    return None


# Preference among value roles that begin at the SAME offset. A percent match is
# a number plus the unit "%", and "%" is a configured lab unit, so the bare number
# must win: that keeps the value-only / value+unit boundary alternatives for "%"
# identical to the ones produced for mmol/L, instead of folding the unit into the
# value and leaving no value-only alternative at all (found by the synthetic
# laboratory stress suite).
_ROLE_PRIORITY: dict[str, int] = {
    "result_inequality": 0,
    "result_qualitative": 0,
    "result_value": 0,
    "result_percent": 1,
}


def find_value(
    seg: str, lex: LabLexicon, exclude: tuple[int, int] | None = None
) -> LocalMatch | None:
    """Find the observed result value (not inside a reference range)."""
    # (start, role priority, -length, role, text)
    candidates: list[tuple[int, int, int, str, str]] = []
    for role, pat in (("result_inequality", lex.inequality_re),
                      ("result_percent", lex.percent_re),
                      ("result_qualitative", lex.qualitative_re),
                      ("result_value", lex.number_re)):
        for m in pat.finditer(seg):
            if _within(m.start(), exclude):
                continue
            candidates.append((m.start(), _ROLE_PRIORITY[role],
                               -(m.end() - m.start()), role, m.group(0)))
    if not candidates:
        return None
    candidates.sort()
    start, _priority, neg_len, role, text = candidates[0]
    end = start + (-neg_len)
    norm = _norm_number(text) if role in ("result_value", "result_percent") else None
    return LocalMatch(role, start, end, text, norm)


def find_unit(seg: str, lex: LabLexicon, start: int = 0) -> LocalMatch | None:
    m = lex.unit_re.search(seg, start)
    if m is None:
        return None
    return LocalMatch("unit", m.start(), m.end(), m.group(0))


def find_flag(seg: str, lex: LabLexicon, start: int = 0) -> LocalMatch | None:
    best: tuple[int, int, str] | None = None
    for pat in (_STANDALONE_FLAG, _FLAG_WORD, _ARROW):
        m = pat.search(seg, start)
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), m.end(), m.group(0))
    if best is None:
        return None
    return LocalMatch("flag", best[0], best[1], best[2])


def hard_negative(seg: str, lex: LabLexicon) -> str | None:
    for pat in lex.hard_negative_res:
        m = pat.search(seg)
        if m is not None:
            return m.group(0)
    return None


__all__ = ["LocalMatch", "find_reference", "find_value", "find_unit", "find_flag",
           "hard_negative"]
