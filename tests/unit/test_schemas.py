"""Schema contract tests (spec sections 1, 4, 19.1)."""

from __future__ import annotations

import pytest

from mednorm_vi.schemas import (
    ASSERTION_LABELS,
    CANDIDATE_ONTOLOGY_BY_TYPE,
    ENTITY_TYPES,
    EntityPrediction,
    Span,
    SpanCoordinates,
)


def test_fixed_vocabularies() -> None:
    assert ENTITY_TYPES == frozenset(
        {"MEDICATION", "DIAGNOSIS", "SYMPTOM", "TEST_NAME", "TEST_RESULT"}
    )
    assert ASSERTION_LABELS == frozenset({"isNegated", "isHistorical", "isFamily"})
    assert CANDIDATE_ONTOLOGY_BY_TYPE["DIAGNOSIS"] == "ICD10"
    assert CANDIDATE_ONTOLOGY_BY_TYPE["MEDICATION"] == "RXNORM"
    assert CANDIDATE_ONTOLOGY_BY_TYPE["SYMPTOM"] is None


def test_span_validates_bounds() -> None:
    with pytest.raises(ValueError):
        Span(-1, 3)
    with pytest.raises(ValueError):
        Span(5, 2)
    s = Span(2, 7)
    assert s.length == 5
    assert s.slice_of("0123456789") == "23456"


def test_span_overlap() -> None:
    assert Span(0, 5).overlaps(Span(4, 8))
    assert not Span(0, 5).overlaps(Span(5, 8))  # half-open: touching != overlap


def test_entity_prediction_output_dict() -> None:
    coords = SpanCoordinates(absolute=Span(3, 8))
    pred = EntityPrediction(
        text="viêm",
        type="DIAGNOSIS",
        coords=coords,
        assertions=("isHistorical",),
        candidates=("J18.9",),
    )
    out = pred.to_output_dict()
    assert out == {
        "text": "viêm",
        "type": "DIAGNOSIS",
        "assertions": ["isHistorical"],
        "candidates": ["J18.9"],
        "position": [3, 8],
    }
    assert pred.position == (3, 8)
