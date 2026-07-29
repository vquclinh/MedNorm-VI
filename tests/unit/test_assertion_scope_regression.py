"""Assertion scope regression: the Audit-0051 false positives (Audit 0052).

The defect, measured on ``tests/fixtures/phase1b/synthetic_medical_document.txt``
before the fix: the Hydra asked whether *any* cue of a family appeared anywhere in a
symmetric ±80-character window, so **ten** entities in that one fixture were assigned
all three labels — ``isNegated + isHistorical + isFamily`` — because the document
mentions "âm tính", "Tiền sử" and "gia đình" somewhere.

Three independent causes, each with its own tests below:

* **A** no direction and no §8.1 boundary: a cue in another sentence or clause
  counted;
* **B** no contrast handling: "không A nhưng B" negated B as well as A;
* **C** bare substring matching: the family cue "ông" (grandfather) matches inside
  "kh**ông**" (not), so every negated sentence also read as family context.

The consequence was masked, not absent: TEST_NAME/TEST_RESULT carry no ``assertions``
field, so the wrong decisions were computed and silently dropped. Activating E3 puts
narrative DIAGNOSIS and SYMPTOM entities beside lab blocks, which is exactly where
they would have reached L9. Assertions are 30% of the organizer metric, scored by
Jaccard, so a spurious label against an empty gold set scores zero for that entity
(spec §13.3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mednorm_vi.resolution.models import (
    BoundaryEvidence,
    EntityHypothesis,
    TypeEvidence,
)
from mednorm_vi.specialists.assertion.cues import (
    ASSERTION_ELIGIBLE_TYPES,
    CONTRAST_MARKERS,
    IS_FAMILY,
    IS_HISTORICAL,
    IS_NEGATED,
    find_cue_occurrences,
    scope_start,
    section_priors,
)
from mednorm_vi.specialists.assertion.hydra import resolve_assertions

REPO = Path(__file__).resolve().parents[2]
DEFECT_FIXTURE = REPO / "tests" / "fixtures" / "phase1b" / "synthetic_medical_document.txt"


def _hypothesis(
    text: str, mention: str, organizer_type: str, *, occurrence: int = 0,
) -> EntityHypothesis:
    """A minimal hypothesis at the nth occurrence of ``mention``."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(mention, start + 1)
    return EntityHypothesis(
        hypothesis_id=f"h-{organizer_type}-{start}",
        document_id="doc", start=start, end=start + len(mention), text=mention,
        entity_type=organizer_type, status="accepted", chosen_proposal_id="p",
        source_proposal_ids=("p",),
        boundary_evidence=BoundaryEvidence(
            policy="test", chosen_kind="full", chosen_proposal_id="p",
            considered_kinds=("full",)),
        type_evidence=TypeEvidence(
            entity_type=organizer_type, source_specialist="test",
            proposed_types=(organizer_type,)))


def _labels(text: str, mention: str, organizer_type: str, **kw: int) -> tuple[str, ...]:
    hypothesis = _hypothesis(text, mention, organizer_type, **kw)
    return resolve_assertions(text, (hypothesis,))[0].labels


# ---------------------------------------------------------------------------
# The exact defect, on the exact fixture that exposed it
# ---------------------------------------------------------------------------


def test_no_entity_receives_all_three_labels_on_the_defect_fixture() -> None:
    """The headline regression. Before the fix: 10 entities with all three."""
    from mednorm_vi.deterministic_baseline.models import Phase1BConfig
    from mednorm_vi.deterministic_baseline.pipeline import run_phase1b
    from mednorm_vi.document_intelligence import analyze_document
    from mednorm_vi.document_intelligence.builder import load_config
    from mednorm_vi.inference.config import PipelineConfig
    from mednorm_vi.lattice.builder import build_span_lattice
    from mednorm_vi.resolution.canonical import resolve_lattice_to_hypotheses
    from mednorm_vi.resolution.config_v1 import load_resolver_v1_config

    config = PipelineConfig.load(REPO / "configs" / "pipeline" / "full_v1.yaml")
    l1_config, lexicon = load_config(config.l1_config)
    graph = analyze_document(DEFECT_FIXTURE, config=l1_config, lexicon=lexicon)
    phase1b = run_phase1b(graph, Phase1BConfig.load(
        config.router_config, config.medication_config, config.laboratory_config))
    lattice = build_span_lattice(
        graph.document_id, graph.original_text, routings=phase1b.routings,
        specialist_proposals=phase1b.proposals, relations=phase1b.relations)
    resolved = resolve_lattice_to_hypotheses(
        lattice, load_resolver_v1_config(config.l4_config),
        relations=phase1b.relations)
    accepted = resolved.accepted()
    decisions = resolve_assertions(graph.original_text, accepted)

    all_three = [d for d in decisions if len(d.labels) == 3]
    assert not all_three, (
        f"{len(all_three)} entities received all three assertion labels; the "
        "±80-character window defect has returned")

    # And a lab value must not be given assertions at all.
    for hypothesis, decision in zip(accepted, decisions, strict=True):
        if hypothesis.entity_type in ("TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
            assert decision.labels == ()
            assert decision.source.startswith("type_not_assertion_eligible")


# ---------------------------------------------------------------------------
# Cause A: direction, sentence and clause boundaries (spec §8.1)
# ---------------------------------------------------------------------------


def test_a_cue_after_the_mention_does_not_apply() -> None:
    text = "Bệnh nhân có sốt, không rõ nguyên nhân."
    assert IS_NEGATED not in _labels(text, "sốt", "TRIỆU_CHỨNG")


def test_a_cue_in_a_previous_sentence_does_not_apply() -> None:
    text = "Không đau ngực. Có ho khan nhiều ngày."
    assert _labels(text, "ho khan", "TRIỆU_CHỨNG") == ()


def test_a_cue_in_a_previous_clause_does_not_apply() -> None:
    """This is the case that produced the fixture's false positives."""
    text = "Bệnh nhân không sốt, ho khan nhiều ngày."
    assert _labels(text, "ho khan", "TRIỆU_CHỨNG") == ()
    # The negated mention in the SAME clause still gets its label.
    assert IS_NEGATED in _labels(text, "sốt", "TRIỆU_CHỨNG")


def test_scope_start_stops_at_the_nearest_boundary() -> None:
    text = "Không đau ngực. Bệnh nhân ho khan."
    mention = text.index("ho khan")
    boundary = scope_start(text, mention, scope=80)
    assert boundary > text.index("Không"), (
        "the scope must not reach back past the sentence terminator")


def test_a_cue_in_scope_still_applies() -> None:
    text = "Bệnh nhân không có tiểu đường."
    assert IS_NEGATED in _labels(text, "tiểu đường", "CHẨN_ĐOÁN")


# ---------------------------------------------------------------------------
# Cause B: contrast markers (spec §8.1)
# ---------------------------------------------------------------------------


def test_negation_stops_at_a_contrast_marker() -> None:
    text = "Bệnh nhân không sốt nhưng có ho khan."
    assert _labels(text, "ho khan", "TRIỆU_CHỨNG") == (), (
        'spec §8.1: "không A nhưng B" negates A, never B')
    assert IS_NEGATED in _labels(text, "sốt", "TRIỆU_CHỨNG")


@pytest.mark.parametrize("marker", ["nhưng", "tuy nhiên", "song", "ngược lại"])
def test_each_contrast_marker_blocks_a_negation(marker: str) -> None:
    text = f"Bệnh nhân không sốt {marker} có ho khan."
    assert _labels(text, "ho khan", "TRIỆU_CHỨNG") == ()


def test_the_contrast_marker_list_is_not_empty() -> None:
    assert CONTRAST_MARKERS


# ---------------------------------------------------------------------------
# Cause C: word-boundary cue matching
# ---------------------------------------------------------------------------


def test_the_family_cue_ong_does_not_match_inside_khong() -> None:
    """"ông" (grandfather) inside "không" (not) — the substring bug."""
    assert find_cue_occurrences("không có sốt", "ông") == ()
    text = "Bệnh nhân không có sốt."
    labels = _labels(text, "sốt", "TRIỆU_CHỨNG")
    assert IS_FAMILY not in labels, (
        '"ông" matched inside "không"; cue matching must respect word boundaries')
    assert labels == (IS_NEGATED,)


def test_a_real_family_cue_still_matches() -> None:
    text = "Tiền sử gia đình: bố bị tăng huyết áp."
    assert IS_FAMILY in _labels(text, "tăng huyết áp", "CHẨN_ĐOÁN")


def test_word_boundary_matching_finds_standalone_cues() -> None:
    assert find_cue_occurrences("tiền sử gia đình", "gia đình")
    assert find_cue_occurrences("bố bị bệnh", "bố")
    # ...and rejects them inside longer words.
    assert find_cue_occurrences("khống chế", "ông") == ()


# ---------------------------------------------------------------------------
# Entity-type eligibility
# ---------------------------------------------------------------------------


def test_only_eligible_types_carry_assertions() -> None:
    assert ASSERTION_ELIGIBLE_TYPES == {"MEDICATION", "DIAGNOSIS", "SYMPTOM"}


@pytest.mark.parametrize(
    "organizer_type", ["TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"])
def test_laboratory_types_never_receive_assertions(organizer_type: str) -> None:
    """They have no ``assertions`` field, and computing one hid the defect."""
    text = "Xét nghiệm: HIV: âm tính. Tiền sử gia đình: bố bị lao."
    hypothesis = _hypothesis(text, "âm tính", organizer_type)
    decision = resolve_assertions(text, (hypothesis,))[0]
    assert decision.labels == ()
    assert decision.source.startswith("type_not_assertion_eligible")


@pytest.mark.parametrize(
    "organizer_type", ["THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG"])
def test_eligible_types_do_receive_assertions(organizer_type: str) -> None:
    text = "Bệnh nhân không có biểu hiện nào."
    hypothesis = _hypothesis(text, "biểu hiện", organizer_type)
    decision = resolve_assertions(text, (hypothesis,))[0]
    assert IS_NEGATED in decision.labels


# ---------------------------------------------------------------------------
# Repeated entities, multiple clauses, family vs patient-current
# ---------------------------------------------------------------------------


def test_repeated_entities_are_judged_independently() -> None:
    """Spec §5 case C7: the same surface form at two offsets is two entities."""
    text = "Bệnh nhân không sốt. Sau đó bệnh nhân có sốt cao."
    first = _labels(text, "sốt", "TRIỆU_CHỨNG", occurrence=0)
    second = _labels(text, "sốt", "TRIỆU_CHỨNG", occurrence=1)
    assert IS_NEGATED in first
    assert IS_NEGATED not in second, (
        "the second occurrence is in a different sentence and is not negated")


def test_two_cues_in_separate_clauses_do_not_both_apply() -> None:
    text = "Tiền sử: đái tháo đường. Hiện tại bệnh nhân không sốt."
    negated = _labels(text, "sốt", "TRIỆU_CHỨNG")
    assert IS_NEGATED in negated
    assert IS_HISTORICAL not in negated, (
        "the history cue belongs to a previous sentence")


def test_family_history_and_patient_current_are_distinguished() -> None:
    text = (
        "Tiền sử gia đình: bố bị tăng huyết áp.\n"
        "Bệnh nhân hiện tại có ho khan."
    )
    family = _labels(text, "tăng huyết áp", "CHẨN_ĐOÁN")
    current = _labels(text, "ho khan", "TRIỆU_CHỨNG")
    assert IS_FAMILY in family
    assert IS_FAMILY not in current, (
        "the family header must not reach the patient-current sentence")


# ---------------------------------------------------------------------------
# Section priors (spec §4.2) — evidence, not a rule
# ---------------------------------------------------------------------------


def test_a_colon_header_establishes_a_prior_for_the_lines_beneath_it() -> None:
    text = "Thuốc tại nhà:\n1. metformin 500 mg\n"
    assert IS_HISTORICAL in section_priors(text, text.index("metformin"))
    assert IS_HISTORICAL in _labels(text, "metformin", "THUỐC")


def test_a_non_header_line_does_not_carry_its_prior_forward() -> None:
    text = "Tiền sử gia đình: bố tăng huyết áp.\nBệnh nhân có ho khan."
    assert section_priors(text, text.index("ho khan")) == ()


def test_a_prior_only_label_is_flagged_uncertain() -> None:
    """Spec §4.2: a local sentence may override a section prior, so flag it."""
    text = "Thuốc tại nhà:\n1. metformin 500 mg\n"
    hypothesis = _hypothesis(text, "metformin", "THUỐC")
    decision = resolve_assertions(text, (hypothesis,))[0]
    assert decision.labels == (IS_HISTORICAL,)
    assert decision.source == "section_prior_only"
    assert decision.uncertain is True


def test_a_cue_backed_label_is_not_flagged_uncertain() -> None:
    text = "Bệnh nhân không có sốt."
    hypothesis = _hypothesis(text, "sốt", "TRIỆU_CHỨNG")
    decision = resolve_assertions(text, (hypothesis,))[0]
    assert decision.labels == (IS_NEGATED,)
    assert decision.uncertain is False


def test_cue_and_prior_evidence_combine() -> None:
    """Spec §8: A1 and A2/A3 are contributing stages, not alternatives."""
    text = "Tiền sử gia đình: bố bị tăng huyết áp."
    decision = resolve_assertions(
        text, (_hypothesis(text, "tăng huyết áp", "CHẨN_ĐOÁN"),))[0]
    assert IS_FAMILY in decision.labels
    assert IS_HISTORICAL in decision.labels
    assert decision.source == "cue_in_scope_and_section_prior"


# ---------------------------------------------------------------------------
# Conservative default
# ---------------------------------------------------------------------------


def test_no_cue_and_no_prior_yields_a_confident_empty_set() -> None:
    text = "Chẩn đoán: viêm phổi thùy dưới phải."
    decision = resolve_assertions(
        text, (_hypothesis(text, "viêm phổi", "CHẨN_ĐOÁN"),))[0]
    assert decision.labels == ()
    assert decision.uncertain is False
    assert decision.source == "deterministic_no_cue"


def test_a_blocked_cue_yields_empty_labels_and_flags_uncertainty() -> None:
    text = "Bệnh nhân không sốt nhưng có ho khan."
    decision = resolve_assertions(
        text, (_hypothesis(text, "ho khan", "TRIỆU_CHỨNG"),))[0]
    assert decision.labels == ()
    assert decision.uncertain is True, (
        "a cue that exists but could not be scoped is the ambiguous case, and L7 "
        "should see it")


def test_an_empty_set_never_means_false() -> None:
    text = "Chẩn đoán: viêm phổi."
    decision = resolve_assertions(
        text, (_hypothesis(text, "viêm phổi", "CHẨN_ĐOÁN"),))[0]
    payload = decision.as_dict()
    assert payload["assertions"] == []
    assert payload["empty_means_insufficient_evidence_not_false"] is True
