"""Deterministic medication component scanning within a text segment.

All offsets returned are LOCAL to the segment; the parser converts them to
absolute original-text coordinates. Original text is never rewritten; a
normalized numeric may be recorded separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import MedicationLexicon

_LEADING_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?(?:\s?[-–—]\s?\d+(?:[.,]\d+)?)?")


@dataclass(frozen=True, slots=True)
class LocalMatch:
    role: str
    start: int
    end: int
    text: str
    normalized: str | None = None


def _norm_number(text: str) -> str:
    return text.replace(",", ".")


def _first(pattern: re.Pattern[str], seg: str, start: int = 0) -> re.Match[str] | None:
    return pattern.search(seg, start)


def scan_strength(seg: str, lex: MedicationLexicon, start: int = 0) -> list[LocalMatch]:
    """Return strength_value + strength_unit (split) or concentration/percent."""
    conc = _first(lex.concentration_re, seg, start)
    if conc is not None:
        return [LocalMatch("concentration", conc.start(), conc.end(), conc.group(0),
                           _norm_number(conc.group(0)))]
    m = _first(lex.strength_re, seg, start)
    if m is None:
        pct = _first(lex.percent_re, seg, start)
        if pct is not None:
            return [LocalMatch("strength_value", pct.start(), pct.end(), pct.group(0),
                               _norm_number(pct.group(0)))]
        return []
    whole = m.group(0)
    num = _LEADING_NUMBER.match(whole)
    out: list[LocalMatch] = []
    if num is not None:
        v_start = m.start()
        v_end = m.start() + num.end()
        out.append(LocalMatch("strength_value", v_start, v_end, seg[v_start:v_end],
                              _norm_number(seg[v_start:v_end])))
        # unit is the remainder of the match, trimmed of leading whitespace
        u_start = v_end
        while u_start < m.end() and seg[u_start].isspace():
            u_start += 1
        if u_start < m.end():
            out.append(LocalMatch("strength_unit", u_start, m.end(), seg[u_start:m.end()]))
    else:
        out.append(LocalMatch("strength_value", m.start(), m.end(), whole))
    return out


def scan_first_role(role: str, pattern: re.Pattern[str], seg: str,
                    start: int = 0) -> LocalMatch | None:
    m = _first(pattern, seg, start)
    if m is None:
        return None
    return LocalMatch(role, m.start(), m.end(), m.group(0))


def scan_duration(seg: str, lex: MedicationLexicon, start: int = 0) -> LocalMatch | None:
    for pat in lex.duration_res:
        m = _first(pat, seg, start)
        if m is not None:
            return LocalMatch("duration", m.start(), m.end(), m.group(0))
    return None


def hard_negative(seg: str, lex: MedicationLexicon) -> str | None:
    for pat in lex.hard_negative_res:
        m = pat.search(seg)
        if m is not None:
            return m.group(0)
    return None


__all__ = ["LocalMatch", "scan_strength", "scan_first_role", "scan_duration", "hard_negative"]
