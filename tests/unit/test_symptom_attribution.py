"""SYMPTOM error attribution rules and privacy guarantees (Audit 0034)."""

from __future__ import annotations

import json

from mednorm_vi.evaluation.exact_mention import (
    BOTH_BOUNDARY,
    LEFT_BOUNDARY,
    MISSED,
    RIGHT_BOUNDARY,
    SPURIOUS,
    WRONG_TYPE,
)
from mednorm_vi.evaluation.symptom_attribution import (
    ATTRIBUTION_CATEGORIES,
    BOUNDARY_TOO_LONG,
    BOUNDARY_TOO_SHORT,
    COMPLETE_MISS,
    DETERMINISTIC_EVIDENCE_ABSENT,
    DIAGNOSIS_SYMPTOM_CONFUSION,
    LOW_NEURAL_CONFIDENCE,
    OVERLAP_COMPETITION,
    SECTION_ROUTER_ERROR,
    TREATMENT_PURPOSE_PHRASE,
    UNKNOWN_OTHER,
    SymptomAttribution,
    SymptomContext,
    attribute_one,
    render_markdown,
    summarize,
    treatment_cue_before,
)


def _context(**kwargs) -> SymptomContext:
    base = dict(privacy_safe_example_id="abcd1234", gold_span=(10, 20),
                evaluator_category=MISSED)
    base.update(kwargs)
    return SymptomContext(**base)  # type: ignore[arg-type]


def test_exact_span_typed_diagnosis_is_confusion() -> None:
    attribution = attribute_one(_context(
        evaluator_category=WRONG_TYPE, predicted_span=(10, 20),
        predicted_type="DIAGNOSIS"))
    assert attribution.category == DIAGNOSIS_SYMPTOM_CONFUSION


def test_short_prediction_is_boundary_too_short() -> None:
    attribution = attribute_one(_context(
        evaluator_category=RIGHT_BOUNDARY, predicted_span=(10, 15)))
    assert attribution.category == BOUNDARY_TOO_SHORT


def test_long_prediction_is_boundary_too_long() -> None:
    attribution = attribute_one(_context(
        evaluator_category=LEFT_BOUNDARY, predicted_span=(5, 20)))
    assert attribution.category == BOUNDARY_TOO_LONG


def test_boundary_error_with_many_competitors_is_overlap_competition() -> None:
    attribution = attribute_one(_context(
        evaluator_category=BOTH_BOUNDARY, predicted_span=(5, 15),
        overlapping_competitors=4))
    assert attribution.category == OVERLAP_COMPETITION


def test_gold_symptom_in_a_non_symptom_section_is_a_router_error() -> None:
    attribution = attribute_one(_context(section="laboratory"))
    assert attribution.category == SECTION_ROUTER_ERROR


def test_treatment_purpose_cue_is_attributed() -> None:
    attribution = attribute_one(_context(treatment_cue_nearby=True))
    assert attribution.category == TREATMENT_PURPOSE_PHRASE


def test_nothing_proposed_at_all_is_a_complete_miss() -> None:
    attribution = attribute_one(_context(lattice_covered=False))
    assert attribution.category == COMPLETE_MISS


def test_low_confidence_symptom_proposal_is_attributed() -> None:
    attribution = attribute_one(_context(
        lattice_covered=True, symptom_proposed=True, neural_confidence=0.2))
    assert attribution.category == LOW_NEURAL_CONFIDENCE


def test_covered_but_never_proposed_as_symptom() -> None:
    attribution = attribute_one(_context(
        lattice_covered=True, symptom_proposed=False))
    assert attribution.category == DETERMINISTIC_EVIDENCE_ABSENT


def test_confident_spurious_prediction_is_reported_as_unknown_other() -> None:
    attribution = attribute_one(_context(
        evaluator_category=SPURIOUS, predicted_span=(10, 20),
        predicted_type="SYMPTOM", neural_confidence=0.95,
        overlapping_competitors=1, lattice_covered=True, symptom_proposed=True))
    assert attribution.category == UNKNOWN_OTHER


def test_treatment_cue_detection_window() -> None:
    text = "bệnh nhân được điều trị ho khan"
    assert treatment_cue_before(text, text.index("ho khan"))
    assert not treatment_cue_before("bệnh nhân bị ho khan", 13)


def test_every_category_is_declared() -> None:
    categories = {
        attribute_one(_context(section="laboratory")).category,
        attribute_one(_context(treatment_cue_nearby=True)).category,
        attribute_one(_context(lattice_covered=False)).category,
    }
    assert categories <= set(ATTRIBUTION_CATEGORIES)


# -- privacy --------------------------------------------------------------------


def test_attribution_records_carry_no_clinical_text() -> None:
    attribution = attribute_one(_context(
        evaluator_category=RIGHT_BOUNDARY, predicted_span=(10, 15),
        route_tags=("C3",), section="unknown"))
    payload = json.dumps(attribution.as_dict(), ensure_ascii=False)
    assert "sốt" not in payload
    for value in attribution.as_dict().values():
        if isinstance(value, str):
            assert value in {
                attribution.category, attribution.evaluator_category,
                attribution.privacy_safe_example_id, attribution.section,
                attribution.detail}


def test_privacy_safe_id_is_a_hash_handle_not_a_verbatim_id() -> None:
    attribution = SymptomAttribution(
        "0123456789abcdef", COMPLETE_MISS, MISSED, (0, 5), None, (), "", "d")
    assert len(attribution.privacy_safe_example_id) == 16
    assert ":" not in attribution.privacy_safe_example_id


def test_summary_and_markdown_contain_counts_only() -> None:
    attributions = [
        attribute_one(_context(lattice_covered=False)),
        attribute_one(_context(treatment_cue_nearby=True)),
    ]
    summary = summarize(attributions)
    assert summary["total"] == 2
    assert summary["categories"][COMPLETE_MISS] == 1
    assert summary["categories"][TREATMENT_PURPOSE_PHRASE] == 1
    assert set(summary["categories"]) == set(ATTRIBUTION_CATEGORIES)
    assert "structural_note" in summary
    markdown = render_markdown(summary)
    assert "SYMPTOM error attribution" in markdown
    for category in ATTRIBUTION_CATEGORIES:
        assert category in markdown
