"""Deterministic laboratory evidence scoring (NOT calibrated probabilities)."""

from __future__ import annotations

from .lexicon import LabLexicon


def score_test_name(
    lex: LabLexicon, *, known: bool, structured_row: bool, key_value: bool,
) -> float:
    s = lex.scoring
    total = (s.get("test_lexicon_match", 0.45) if known
             else s.get("unknown_test_with_structure", 0.20))
    if key_value:
        total += s.get("key_value_structure", 0.40)
    elif structured_row:
        total += s.get("row_structure", 0.35)
    return round(min(1.0, total), 6)


def score_test_result(
    lex: LabLexicon, *, structured_row: bool, key_value: bool, has_unit: bool,
    has_reference: bool,
) -> float:
    s = lex.scoring
    total = s.get("value_pattern", 0.30)
    if key_value:
        total += s.get("key_value_structure", 0.40)
    elif structured_row:
        total += s.get("row_structure", 0.35)
    if has_unit:
        total += s.get("unit_evidence", 0.20)
    if has_reference:
        total += s.get("reference_range_evidence", 0.15)
    return round(min(1.0, total), 6)


__all__ = ["score_test_name", "score_test_result"]
