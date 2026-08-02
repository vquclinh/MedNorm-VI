"""Essential guards for the constrained 8B reasoner (sprint 0075). No weights, no network."""

from __future__ import annotations

import pytest

from mednorm_vi.reasoner import budget
from mednorm_vi.reasoner.pool import Candidate
from mednorm_vi.reasoner.prompt import candidate_prompt, entity_prompt
from mednorm_vi.reasoner.validator import (
    CANDIDATE_TYPES,
    ORGANIZER_TYPES,
    ValidationReport,
    validate,
)

NOTE = "Bệnh nhân sốt cao, không ho. Mẹ bị đái tháo đường. Tiền sử sốt."


def test_budget_allows_8b_with_e3_and_refuses_co_deploying_the_4b_pair() -> None:
    assert budget.assert_within_cap() == 8_325_737_477
    with pytest.raises(budget.ParameterBudgetExceeded):
        budget.assert_within_cap(with_semantic_s1=True)


def test_only_five_organizer_types_are_accepted() -> None:
    assert set(ORGANIZER_TYPES) == {
        "TRIỆU_CHỨNG",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
        "CHẨN_ĐOÁN",
        "THUỐC",
    }
    _, report = validate(NOTE, [{"text": "sốt", "type": "SYMPTOM"}])
    assert report.accepted == 0 and report.rejected["unknown_type"] == 1


def test_hallucinated_span_is_rejected() -> None:
    _, report = validate(NOTE, [{"text": "ung thư phổi giai đoạn 4", "type": "CHẨN_ĐOÁN"}])
    assert report.accepted == 0 and report.rejected["span_not_in_source"] == 1


def test_positions_are_computed_from_source_not_taken_from_the_model() -> None:
    entities, _ = validate(NOTE, [{"text": "ho", "type": "TRIỆU_CHỨNG", "position": [999, 1000]}])
    start, end = entities[0].position
    assert NOTE[start:end] == "ho"


def test_duplicate_surface_forms_resolve_to_distinct_occurrences() -> None:
    entities, _ = validate(
        NOTE, [{"text": "sốt", "type": "TRIỆU_CHỨNG"}, {"text": "sốt", "type": "TRIỆU_CHỨNG"}]
    )
    positions = [e.position for e in entities]
    assert len(set(positions)) == 2
    assert all(NOTE[s:e] == "sốt" for s, e in positions)


def test_more_occurrences_than_exist_are_rejected() -> None:
    proposals = [{"text": "ho", "type": "TRIỆU_CHỨNG"} for _ in range(3)]
    _, report = validate(NOTE, proposals)
    assert report.accepted == 1
    assert report.rejected["occurrence_exhausted"] == 2


def test_model_can_never_emit_a_code_outside_the_offered_pool() -> None:
    entities, report = validate(
        NOTE,
        [
            {
                "text": "đái tháo đường",
                "type": "CHẨN_ĐOÁN",
                "pool_key": "0",
                "candidates": ["E11.9", "MADE.UP", "Z99.9"],
            }
        ],
        {"0": {"E11.9"}},
    )
    assert entities[0].candidates == ("E11.9",)
    assert report.rejected["ungoverned_candidate"] == 2


def test_empty_candidate_list_is_a_valid_answer() -> None:
    entities, _ = validate(
        NOTE,
        [{"text": "đái tháo đường", "type": "CHẨN_ĐOÁN", "pool_key": "0", "candidates": []}],
        {"0": {"E11.9"}},
    )
    assert entities[0].candidates == ()
    assert "candidates" in entities[0].as_organizer_json()


def test_assertions_are_strict_booleans_and_default_false() -> None:
    entities, _ = validate(
        NOTE,
        [
            {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": {"isNegated": True, "bogus": True}},
            {"text": "sốt", "type": "TRIỆU_CHỨNG"},
        ],
    )
    by_text = {(e.text, e.position[0]): e for e in entities}
    negated = next(e for e in entities if e.text == "ho")
    assert negated.assertions == ("isNegated",)
    assert all("bogus" not in e.assertions for e in entities)
    assert next(e for e in entities if e.text == "sốt").assertions == ()
    assert by_text


def test_only_diagnosis_and_medication_carry_candidates() -> None:
    entities, _ = validate(NOTE, [{"text": "sốt", "type": "TRIỆU_CHỨNG"}])
    assert "candidates" not in entities[0].as_organizer_json()
    assert CANDIDATE_TYPES == {"CHẨN_ĐOÁN", "THUỐC"}


def test_output_is_deterministic_and_position_sorted() -> None:
    proposals = [{"text": "ho", "type": "TRIỆU_CHỨNG"}, {"text": "sốt", "type": "TRIỆU_CHỨNG"}]
    first = [e.as_organizer_json() for e in validate(NOTE, proposals)[0]]
    assert all(
        [e.as_organizer_json() for e in validate(NOTE, proposals)[0]] == first for _ in range(3)
    )
    assert [e["position"][0] for e in first] == sorted(e["position"][0] for e in first)


def test_prompts_state_the_constraints_the_validator_enforces() -> None:
    text = entity_prompt(NOTE)
    assert "exact substring" in text
    assert "Do NOT output character offsets" in text
    for organizer_type in ORGANIZER_TYPES:
        assert organizer_type in text
    selection = candidate_prompt(NOTE, "sốt", "CHẨN_ĐOÁN", [Candidate("A", "A", "n")])
    assert "Never write a code that is not listed" in selection
    assert "Return [] if none of them is correct" in selection


def test_validation_report_counts_every_rejection_reason() -> None:
    report = ValidationReport()
    assert report.as_dict() == {"accepted": 0, "rejected": {}}
