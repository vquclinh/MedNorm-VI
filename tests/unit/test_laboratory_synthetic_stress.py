"""Synthetic laboratory stress suite: exact offsets by construction (Audit 0034).

Nothing here is a claim about real TEST_NAME / TEST_RESULT performance — the
governed corpus has no supervision for those types.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mednorm_vi.deterministic_baseline.models import Phase1BConfig
from mednorm_vi.evaluation.laboratory_stress import run_case, run_suite
from mednorm_vi.mention_factory.laboratory.synthetic import (
    FAMILIES,
    FAMILY_DECIMAL_COMMA,
    FAMILY_DECIMAL_POINT,
    FAMILY_DISTRACTOR,
    FAMILY_FLAG,
    FAMILY_MISSING_UNIT,
    FAMILY_MULTILINE,
    FAMILY_REFERENCE_RANGE,
    FAMILY_REPEATED,
    FAMILY_SEMICOLON,
    FAMILY_TABLE,
    FAMILY_UNIT,
    build_cases,
)
from mednorm_vi.resolution.config_v1 import DEFAULT_CONFIG_PATH, load_resolver_v1_config

REPO = Path(__file__).resolve().parents[2]
PHASE1B = Phase1BConfig.load(
    REPO / "configs" / "case_router" / "base.yaml",
    REPO / "configs" / "medication" / "grammar_v1.yaml",
    REPO / "configs" / "laboratory" / "parser_v1.yaml")
RESOLVER = load_resolver_v1_config(REPO / DEFAULT_CONFIG_PATH)
CASES = {case.case_id: case for case in build_cases()}


def _run(case_id: str):
    return run_case(CASES[case_id], PHASE1B, RESOLVER)[0]


# -- construction ---------------------------------------------------------------


def test_every_case_has_exact_offsets_by_construction() -> None:
    for case in build_cases():
        case.validate()
        for entity in case.entities:
            assert case.text[entity.start:entity.end] == entity.text
            assert entity.end > entity.start


def test_every_declared_family_is_covered() -> None:
    covered = {case.family for case in build_cases()}
    assert covered == set(FAMILIES)


def test_gold_pairs_reference_real_entities() -> None:
    for case in build_cases():
        for (name_span, result_span) in case.gold_pairs():
            assert case.text[name_span[0]:name_span[1]]
            assert case.text[result_span[0]:result_span[1]]


def test_a_broken_constructed_offset_is_rejected() -> None:
    from mednorm_vi.mention_factory.laboratory.synthetic import (
        SyntheticEntity,
        SyntheticLabCase,
    )
    bad = SyntheticLabCase(
        "bad", "x", "WBC: 1",
        (SyntheticEntity(0, 3, "TEST_NAME", "XXX"),))
    with pytest.raises(ValueError):
        bad.validate()


# -- per-family behaviour -------------------------------------------------------


@pytest.mark.parametrize("case_id", [
    "lab-colon-01", "lab-unit-01", "lab-decimal-comma-01", "lab-decimal-point-01",
    "lab-flag-01", "lab-reference-01", "lab-missing-unit-01", "lab-multiline-01",
    "lab-semicolon-01", "lab-table-01", "lab-repeated-01", "lab-qualitative-02",
])
def test_case_is_recovered_exactly(case_id: str) -> None:
    outcome = _run(case_id)
    assert outcome.exact_matches == len(outcome.gold), (
        f"{case_id}: gold {outcome.gold} predicted {outcome.predicted}")
    assert len(outcome.predicted) == len(outcome.gold), (
        f"{case_id}: extra predictions {outcome.predicted}")


def test_decimal_comma_value_is_not_split() -> None:
    outcome = _run("lab-decimal-comma-01")
    assert outcome.family == FAMILY_DECIMAL_COMMA
    result = [span for span in outcome.predicted if span[2] == "TEST_RESULT"][0]
    assert CASES["lab-decimal-comma-01"].text[result[0]:result[1]] == "88,5"


def test_decimal_point_value_is_not_split() -> None:
    outcome = _run("lab-decimal-point-01")
    assert outcome.family == FAMILY_DECIMAL_POINT
    result = [span for span in outcome.predicted if span[2] == "TEST_RESULT"][0]
    assert CASES["lab-decimal-point-01"].text[result[0]:result[1]] == "76.4"


def test_unit_is_not_swallowed_into_the_value() -> None:
    outcome = _run("lab-unit-01")
    assert outcome.family == FAMILY_UNIT
    result = [span for span in outcome.predicted if span[2] == "TEST_RESULT"][0]
    assert CASES["lab-unit-01"].text[result[0]:result[1]] == "5.6"


def test_flags_are_not_part_of_the_value_span() -> None:
    outcome = _run("lab-flag-01")
    assert outcome.family == FAMILY_FLAG
    text = CASES["lab-flag-01"].text
    for span in outcome.predicted:
        if span[2] == "TEST_RESULT":
            assert not text[span[0]:span[1]].strip().endswith(("H", "L"))


def test_reference_range_is_not_swallowed() -> None:
    outcome = _run("lab-reference-01")
    assert outcome.family == FAMILY_REFERENCE_RANGE
    text = CASES["lab-reference-01"].text
    for span in outcome.predicted:
        assert "(" not in text[span[0]:span[1]]


def test_missing_unit_still_yields_a_result() -> None:
    outcome = _run("lab-missing-unit-01")
    assert outcome.family == FAMILY_MISSING_UNIT
    assert any(span[2] == "TEST_RESULT" for span in outcome.predicted)


def test_multiline_rows_all_pair() -> None:
    outcome = _run("lab-multiline-01")
    assert outcome.family == FAMILY_MULTILINE
    assert outcome.pairing.correct == outcome.pairing.gold == 3


def test_semicolon_and_table_layouts_pair() -> None:
    for case_id, family in (("lab-semicolon-01", FAMILY_SEMICOLON),
                            ("lab-table-01", FAMILY_TABLE)):
        outcome = _run(case_id)
        assert outcome.family == family
        assert outcome.pairing.correct == outcome.pairing.gold == 2


def test_repeated_test_name_stays_two_distinct_mentions() -> None:
    outcome = _run("lab-repeated-01")
    assert outcome.family == FAMILY_REPEATED
    names = [span for span in outcome.predicted if span[2] == "TEST_NAME"]
    assert len(names) == 2
    text = CASES["lab-repeated-01"].text
    assert text[names[0][0]:names[0][1]] == text[names[1][0]:names[1][1]]
    assert names[0][0] != names[1][0]


def test_distractors_produce_no_laboratory_mention() -> None:
    outcome = _run("lab-distractor-01")
    assert outcome.family == FAMILY_DISTRACTOR
    assert outcome.gold == ()
    assert outcome.predicted == ()


def test_bare_qualitative_row_is_a_known_routing_gap() -> None:
    """A bare qualitative row carries no lab-specific evidence, so C2 never fires.

    This is recorded as a measured gap, not asserted as correct behaviour: the
    same rows under a laboratory section header are recovered exactly
    (``lab-qualitative-02``).
    """
    bare = _run("lab-qualitative-01")
    sectioned = _run("lab-qualitative-02")
    assert bare.predicted == ()
    assert sectioned.exact_matches == len(sectioned.gold)


# -- suite level ----------------------------------------------------------------


def test_suite_report_is_labelled_synthetic_and_claims_nothing_real() -> None:
    report = run_suite(PHASE1B, RESOLVER)
    assert report["synthetic"] is True
    assert report["real_performance_claim"] is False
    assert "no test_name or test_result supervision" in report["warning"].lower()
    assert report["reproduces_complete_organizer_score"] is False
    assert "SYNTHETIC" in report["label"]


def test_suite_pairing_precision_is_perfect_and_recall_is_reported() -> None:
    report = run_suite(PHASE1B, RESOLVER)
    pairing = report["has_result_pairing"]
    assert pairing["gold"] > 0
    assert pairing["precision"] == 1.0
    assert 0.0 < pairing["recall"] <= 1.0


def test_suite_is_deterministic() -> None:
    first = run_suite(PHASE1B, RESOLVER)
    second = run_suite(PHASE1B, RESOLVER)
    assert first["micro"] == second["micro"]
    assert [c["predicted"] for c in first["cases"]] == [
        c["predicted"] for c in second["cases"]]


def test_resolver_reduces_spurious_boundary_alternatives() -> None:
    with_resolver = run_suite(PHASE1B, RESOLVER)
    without = run_suite(PHASE1B, RESOLVER, apply_resolver=False)
    assert (with_resolver["micro"]["false_positive"]
            < without["micro"]["false_positive"])
