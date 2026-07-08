"""Duplicate-detection tests (spec sections 7.3, 11, C7).

Key rules:
* Duplicate text at DIFFERENT positions must remain distinct (never merged).
* An exact duplicate at the SAME position must be rejected or merged
  deterministically.
"""

from __future__ import annotations

from mednorm_vi.validator import (
    deduplicate,
    find_duplicate_groups,
    validate_no_duplicates,
)


def _entity(text: str, start: int, end: int, etype: str = "SYMPTOM") -> dict:
    return {
        "text": text,
        "type": etype,
        "assertions": [],
        "candidates": [],
        "position": [start, end],
    }


def test_same_text_different_position_stays_distinct() -> None:
    # "táo bón" (constipation) appears twice at different offsets -> two entities.
    entities = [_entity("táo bón", 10, 17), _entity("táo bón", 40, 47)]
    assert find_duplicate_groups(entities) == []
    assert validate_no_duplicates(entities).ok
    kept, result = deduplicate(entities)
    assert len(kept) == 2
    assert result.ok  # no warnings, nothing merged


def test_exact_duplicate_same_position_is_rejected() -> None:
    entities = [_entity("ho", 12, 14), _entity("ho", 12, 14)]
    groups = find_duplicate_groups(entities)
    assert groups == [[0, 1]]
    result = validate_no_duplicates(entities)
    assert not result.ok
    assert any(i.code == "duplicate.exact" for i in result.errors)


def test_exact_duplicate_merged_deterministically() -> None:
    entities = [_entity("ho", 12, 14), _entity("ho", 12, 14), _entity("ho", 12, 14)]
    kept, result = deduplicate(entities)
    assert len(kept) == 1  # first occurrence kept
    assert kept[0] is entities[0]
    assert len(result.warnings) == 2  # two duplicates recorded


def test_same_position_different_type_is_not_a_duplicate() -> None:
    # Same span but competing types are distinct hypotheses, not a duplicate.
    entities = [
        _entity("sốt", 0, 3, etype="SYMPTOM"),
        _entity("sốt", 0, 3, etype="DIAGNOSIS"),
    ]
    assert find_duplicate_groups(entities) == []
    assert validate_no_duplicates(entities).ok
