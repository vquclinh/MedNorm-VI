"""Entity schema / type / assertion-label tests (spec sections 1, 7, 8)."""

from __future__ import annotations

from mednorm_vi.validator import (
    validate_assertion_labels,
    validate_entity_schema,
    validate_entity_type,
)


def _entity(**overrides: object) -> dict:
    base = {
        "text": "viêm phổi",
        "type": "DIAGNOSIS",
        "assertions": [],
        "candidates": ["J18.9"],
        "position": [0, 9],
    }
    base.update(overrides)
    return base


def test_valid_entity_passes_schema() -> None:
    assert validate_entity_schema(_entity()).ok


def test_invalid_entity_type_fails() -> None:
    result = validate_entity_type(_entity(type="DRUGGGG"))
    assert not result.ok
    assert any(i.code == "entity.invalid_type" for i in result.errors)


def test_all_five_entity_types_valid() -> None:
    cands_by_type = {"DIAGNOSIS": ["J18.9"], "MEDICATION": ["1049640"]}
    for etype in ("MEDICATION", "DIAGNOSIS", "SYMPTOM", "TEST_NAME", "TEST_RESULT"):
        cands = cands_by_type.get(etype, [])
        assert validate_entity_type(_entity(type=etype, candidates=cands)).ok


def test_invalid_assertion_label_fails() -> None:
    result = validate_assertion_labels(_entity(assertions=["isNegated", "isChronic"]))
    assert not result.ok
    assert any(i.code == "assertion.invalid_label" for i in result.errors)


def test_valid_assertion_labels_pass() -> None:
    result = validate_assertion_labels(
        _entity(assertions=["isNegated", "isHistorical", "isFamily"])
    )
    assert result.ok


def test_duplicate_assertion_label_flagged() -> None:
    result = validate_assertion_labels(_entity(assertions=["isNegated", "isNegated"]))
    assert not result.ok
    assert any(i.code == "assertion.duplicate_label" for i in result.errors)


def test_missing_required_field_fails() -> None:
    e = _entity()
    del e["candidates"]
    result = validate_entity_schema(e)
    assert not result.ok
    assert any(i.code == "schema.missing_field" for i in result.errors)


def test_candidates_forbidden_for_symptom() -> None:
    # SYMPTOM has no ontology; carrying candidates violates cross-link rule.
    result = validate_entity_schema(
        _entity(type="SYMPTOM", candidates=["J18.9"])
    )
    assert not result.ok
    assert any(i.code == "schema.candidates_not_allowed_for_type" for i in result.errors)


def test_medication_may_carry_candidates() -> None:
    assert validate_entity_schema(
        _entity(type="MEDICATION", candidates=["1049640"])
    ).ok
