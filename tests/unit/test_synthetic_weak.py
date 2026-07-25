"""Synthetic generators + weak-supervision contracts (Audit 0017). Deterministic,
offline; no neural pseudo-labeling."""

from __future__ import annotations

from mednorm_vi.data_engine.synthetic import synthetic_fixture_examples
from mednorm_vi.data_engine.weak_labeling import (
    ABSTAIN,
    LabelingFunctionVote,
    aggregate_votes,
    is_valid_weak_label,
)
from mednorm_vi.schemas.spans import Span


def test_synthetic_fixtures_offset_valid_and_cover_missing_types() -> None:
    docs = synthetic_fixture_examples()
    types = set()
    for d in docs:
        assert d.metadata.get("provenance") == "synthetic"
        for a in d.annotations:
            assert d.text[a.span.start:a.span.end] == a.text     # half-open invariant
            types.add(a.entity_type)
    # The governed corpus v1 lacks TEST_NAME/TEST_RESULT — synthetic supplies them.
    assert {"TEST_NAME", "TEST_RESULT", "MEDICATION"} <= types


def test_synthetic_assertions_are_valid_labels() -> None:
    docs = synthetic_fixture_examples()
    seen = set()
    for d in docs:
        for a in d.annotations:
            for asrt in a.assertions:
                assert asrt in {"isNegated", "isFamily", "isHistorical"}
                seen.add(asrt)
    assert seen == {"isNegated", "isFamily", "isHistorical"}


def test_weak_vote_aggregation_conflict_and_abstain() -> None:
    span = Span(0, 3)
    votes = (
        LabelingFunctionVote("lf_med_grammar", span, "MEDICATION", 0.9, "route+strength"),
        LabelingFunctionVote("lf_alias", span, "MEDICATION", 0.7, "rxnorm_alias"),
        LabelingFunctionVote("lf_noise", span, ABSTAIN, 0.0, "no_signal"),
    )
    agg = aggregate_votes(votes, min_confidence=0.5)
    assert len(agg) == 1
    assert agg[0].label == "MEDICATION" and agg[0].support == 2
    assert agg[0].conflict is False
    assert agg[0].voters == ("lf_alias", "lf_med_grammar")


def test_weak_vote_conflict_flagged() -> None:
    span = Span(5, 12)
    votes = (
        LabelingFunctionVote("lf_a", span, "DIAGNOSIS", 0.8, "section_prior"),
        LabelingFunctionVote("lf_b", span, "SYMPTOM", 0.8, "cue"),
    )
    agg = aggregate_votes(votes)
    assert agg[0].conflict is True


def test_is_valid_weak_label() -> None:
    assert is_valid_weak_label("MEDICATION")
    assert is_valid_weak_label("isNegated")
    assert not is_valid_weak_label("PATIENT_INFO")
