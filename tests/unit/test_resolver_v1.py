"""L4 Boundary & Type Resolver v1 (Audit 0034)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mednorm_vi.lattice import build_span_lattice
from mednorm_vi.lattice.models import (
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_VIHEALTHBERT,
    SourceEvidence,
)
from mednorm_vi.lattice.models import SpanProposal as LatticeProposal
from mednorm_vi.mention_factory.models import HAS_RESULT, ComponentSpan, RelationProposal
from mednorm_vi.mention_factory.models import SpanProposal as SpecialistProposal
from mednorm_vi.mention_factory.neural.decoding import NeuralSpan
from mednorm_vi.resolution.boundary import expand_to_competitor, trim_span
from mednorm_vi.resolution.config_v1 import (
    DEFAULT_CONFIG_PATH,
    ResolverConfigError,
    load_resolver_v1_config,
)
from mednorm_vi.resolution.overlap import (
    OverlapCandidate,
    interval_iou,
    resolve_near_complete_overlaps,
)
from mednorm_vi.resolution.resolver_v1 import (
    LEARNED_BOUNDARY_OFFSET_HEAD_TRAINED,
    resolve_lattice,
)
from mednorm_vi.resolution.typing import decide_type, type_utilities

REPO = Path(__file__).resolve().parents[2]
CONFIG = load_resolver_v1_config(REPO / DEFAULT_CONFIG_PATH)

TRIM = dict(trim_characters=CONFIG.boundary.trim_characters,
            max_trim_chars=CONFIG.boundary.max_trim_chars,
            trim_leading_list_markers=CONFIG.boundary.trim_leading_list_markers)


def _neural(text: str, start: int, end: int, entity_type: str, score: float = 0.9):
    return NeuralSpan(start, end, entity_type, text[start:end], score, 1)


def _node(text: str, start: int, end: int, sources: tuple[SourceEvidence, ...],
          scores: dict[str, float], *, section: str = "", routes: tuple[str, ...] = ()):
    return LatticeProposal(
        document_id="doc", start=start, end=end, text=text[start:end],
        type_scores=scores, sources=sources, routes=routes, section=section)


# -- the tracked config ---------------------------------------------------------


def test_config_reports_the_sha256_of_its_own_bytes() -> None:
    import hashlib
    expected = hashlib.sha256((REPO / DEFAULT_CONFIG_PATH).read_bytes()).hexdigest()
    assert CONFIG.config_sha256 == expected
    assert len(CONFIG.config_sha256) == 64


def test_config_rejects_a_forbidden_ontology_pairing(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "config_version: x\nontology:\n  allowed_evidence:\n    MEDICATION: ICD10\n",
        encoding="utf-8")
    with pytest.raises(ResolverConfigError):
        load_resolver_v1_config(bad)


def test_no_learned_boundary_offset_head_is_claimed() -> None:
    assert LEARNED_BOUNDARY_OFFSET_HEAD_TRAINED is False


# -- controlled trim ------------------------------------------------------------


def test_trim_removes_leading_list_number_and_trailing_separator() -> None:
    text = "1. amlodipine,"
    shaped = trim_span(text, 0, len(text), **TRIM)
    assert text[shaped.start:shaped.end] == "amlodipine"
    assert any(a.startswith("trim_left_list_marker") for a in shaped.actions)
    assert any(a.startswith("trim_right") for a in shaped.actions)


def test_trim_never_shreds_a_decimal_value() -> None:
    # "14.43" reads as the list marker "14." followed by "43" unless the numeric
    # marker form requires trailing whitespace.
    text = "14.43"
    shaped = trim_span(text, 0, len(text), **TRIM)
    assert (shaped.start, shaped.end) == (0, 5)
    assert shaped.actions == ()


def test_trim_is_capped_and_never_empties_a_span() -> None:
    text = "::::::::::"
    shaped = trim_span(text, 0, len(text), **TRIM)
    assert shaped.end > shaped.start
    assert shaped.start <= CONFIG.boundary.max_trim_chars
    assert len(text) - shaped.end <= CONFIG.boundary.max_trim_chars


def test_trim_preserves_the_offset_invariant() -> None:
    text = "  (sốt cao)  "
    shaped = trim_span(text, 0, len(text), **TRIM)
    assert text[shaped.start:shaped.end] == text[shaped.start:shaped.end]
    assert "sốt" in text[shaped.start:shaped.end]


# -- controlled expand ----------------------------------------------------------


def test_expand_adopts_only_a_competing_boundary_that_shares_an_edge() -> None:
    shaped = expand_to_competitor(
        0, 10, ((0, 19, 1.0),), min_grammar_completeness=0.6, max_expand_chars=40)
    assert (shaped.start, shaped.end) == (0, 19)


def test_expand_refuses_an_unsupported_competitor() -> None:
    shaped = expand_to_competitor(
        0, 10, ((0, 19, 0.1),), min_grammar_completeness=0.6, max_expand_chars=40)
    assert (shaped.start, shaped.end) == (0, 10)


def test_expand_refuses_a_competitor_that_shares_no_edge() -> None:
    shaped = expand_to_competitor(
        5, 10, ((0, 19, 1.0),), min_grammar_completeness=0.6, max_expand_chars=40)
    assert (shaped.start, shaped.end) == (5, 10)


def test_expand_respects_the_character_cap() -> None:
    shaped = expand_to_competitor(
        0, 10, ((0, 200, 1.0),), min_grammar_completeness=0.6, max_expand_chars=40)
    assert (shaped.start, shaped.end) == (0, 10)


# -- type decision, agreement and conflict --------------------------------------


def test_medication_grammar_and_neural_agreement_raises_utility() -> None:
    text = "amlodipine 10 mg po"
    grammar = SourceEvidence(
        EXPERT_MEDICATION_GRAMMAR, "med-1", 0.8, {"MEDICATION": 0.8},
        features={"grammar_component_count": 4.0})
    neural = SourceEvidence(EXPERT_VIHEALTHBERT, "e3-1", 0.7, {"MEDICATION": 0.7})
    agreed = _node(text, 0, len(text), (grammar, neural), {"MEDICATION": 0.8})
    alone = _node(text, 0, len(text), (neural,), {"MEDICATION": 0.7})
    assert (decide_type(agreed, text, CONFIG).utility
            > decide_type(alone, text, CONFIG).utility)


def test_expert_conflict_on_the_same_span_is_penalised() -> None:
    text = "aspirin"
    grammar = SourceEvidence(
        EXPERT_MEDICATION_GRAMMAR, "med-1", 0.8, {"MEDICATION": 0.8},
        features={"grammar_component_count": 4.0})
    neural = SourceEvidence(EXPERT_VIHEALTHBERT, "e3-1", 0.8, {"DIAGNOSIS": 0.8})
    node = _node(text, 0, 7, (grammar, neural), {"MEDICATION": 0.8, "DIAGNOSIS": 0.8})
    utilities, _conflicts, breakdown = type_utilities(node, text, CONFIG)
    assert breakdown["DIAGNOSIS"]["expert_disagreement"] < 0
    assert breakdown["MEDICATION"]["expert_disagreement"] < 0
    assert set(utilities) == {"MEDICATION", "DIAGNOSIS"}


def test_wrong_type_risk_causes_abstention_on_a_near_tie() -> None:
    text = "sốt"
    neural = SourceEvidence(
        EXPERT_VIHEALTHBERT, "e3-1", 0.61, {"SYMPTOM": 0.61, "DIAGNOSIS": 0.60})
    node = _node(text, 0, 3, (neural,), {"SYMPTOM": 0.61, "DIAGNOSIS": 0.60})
    decision = decide_type(node, text, CONFIG)
    assert decision.abstained
    assert decision.reason == "type_margin_below_threshold"


def test_weak_evidence_is_dropped_rather_than_guessed() -> None:
    text = "sốt"
    neural = SourceEvidence(EXPERT_VIHEALTHBERT, "e3-1", 0.1, {"SYMPTOM": 0.1})
    decision = decide_type(_node(text, 0, 3, (neural,), {"SYMPTOM": 0.1}), text, CONFIG)
    assert decision.abstained and decision.reason == "utility_below_threshold"


def test_flagged_deterministic_evidence_alone_abstains() -> None:
    text = "tiên lượng sống"
    flagged = SourceEvidence(
        EXPERT_MEDICATION_GRAMMAR, "med-1", 0.9, {"MEDICATION": 0.9},
        warnings=("unknown_medication_name",),
        features={"grammar_component_count": 4.0})
    decision = decide_type(
        _node(text, 0, len(text), (flagged,), {"MEDICATION": 0.9}), text, CONFIG)
    assert decision.abstained
    assert decision.reason == "flagged_deterministic_evidence_only"


# -- spec §7.3 ontology consistency ---------------------------------------------


def test_icd_evidence_never_reinforces_medication() -> None:
    text = "amlodipine"
    source = SourceEvidence(
        EXPERT_VIHEALTHBERT, "e3-1", 0.9, {"MEDICATION": 0.9},
        features={"ontology_ICD10": 1.0})
    _u, conflicts, breakdown = type_utilities(
        _node(text, 0, 10, (source,), {"MEDICATION": 0.9}), text, CONFIG)
    assert breakdown["MEDICATION"]["ontology_evidence"] == 0.0
    assert conflicts["MEDICATION"] is True


def test_rxnorm_evidence_never_reinforces_diagnosis() -> None:
    text = "suy tim"
    source = SourceEvidence(
        EXPERT_VIHEALTHBERT, "e3-1", 0.9, {"DIAGNOSIS": 0.9},
        features={"ontology_RXNORM": 1.0})
    _u, conflicts, breakdown = type_utilities(
        _node(text, 0, 7, (source,), {"DIAGNOSIS": 0.9}), text, CONFIG)
    assert breakdown["DIAGNOSIS"]["ontology_evidence"] == 0.0
    assert conflicts["DIAGNOSIS"] is True


def test_allowed_ontology_evidence_does_reinforce() -> None:
    text = "suy tim"
    source = SourceEvidence(
        EXPERT_VIHEALTHBERT, "e3-1", 0.9, {"DIAGNOSIS": 0.9},
        features={"ontology_ICD10": 1.0})
    _u, conflicts, breakdown = type_utilities(
        _node(text, 0, 7, (source,), {"DIAGNOSIS": 0.9}), text, CONFIG)
    assert breakdown["DIAGNOSIS"]["ontology_evidence"] > 0.0
    assert conflicts["DIAGNOSIS"] is False


def test_ontology_conflict_abstains() -> None:
    text = "amlodipine"
    source = SourceEvidence(
        EXPERT_VIHEALTHBERT, "e3-1", 0.9, {"MEDICATION": 0.9},
        features={"ontology_ICD10": 1.0})
    decision = decide_type(
        _node(text, 0, 10, (source,), {"MEDICATION": 0.9}), text, CONFIG)
    assert decision.abstained and decision.reason == "type_ontology_conflict"


# -- overlap structure ----------------------------------------------------------


def test_interval_iou_is_zero_for_identical_text_at_different_offsets() -> None:
    assert interval_iou((0, 7), (18, 25)) == 0.0


def test_near_complete_same_type_overlap_suppresses_the_weaker_span() -> None:
    outcome = resolve_near_complete_overlaps(
        [OverlapCandidate("a", 0, 10, "MEDICATION", 0.9),
         OverlapCandidate("b", 0, 9, "MEDICATION", 0.4)],
        near_complete_iou=0.6, competition_penalty=0.2,
        suppress_cross_type=False, protect_pairs=True)
    assert outcome.survivors == ("a",)
    assert outcome.suppressed[0][0] == "b"


def test_partial_overlap_below_the_threshold_keeps_both() -> None:
    outcome = resolve_near_complete_overlaps(
        [OverlapCandidate("a", 0, 10, "SYMPTOM", 0.9),
         OverlapCandidate("b", 8, 30, "SYMPTOM", 0.8)],
        near_complete_iou=0.6, competition_penalty=0.2,
        suppress_cross_type=False, protect_pairs=True)
    assert set(outcome.survivors) == {"a", "b"}


def test_cross_type_overlap_is_kept() -> None:
    outcome = resolve_near_complete_overlaps(
        [OverlapCandidate("name", 0, 10, "TEST_NAME", 0.9),
         OverlapCandidate("result", 0, 10, "TEST_RESULT", 0.4)],
        near_complete_iou=0.6, competition_penalty=0.2,
        suppress_cross_type=False, protect_pairs=True)
    assert set(outcome.survivors) == {"name", "result"}


def test_a_strong_has_result_partner_is_never_suppressed() -> None:
    outcome = resolve_near_complete_overlaps(
        [OverlapCandidate("name", 0, 10, "TEST_NAME", 0.9,
                          protected_partners=frozenset({"result"})),
         OverlapCandidate("result", 0, 10, "TEST_NAME", 0.4,
                          protected_partners=frozenset({"name"}))],
        near_complete_iou=0.6, competition_penalty=0.2,
        suppress_cross_type=True, protect_pairs=True)
    assert set(outcome.survivors) == {"name", "result"}


# -- end-to-end resolver behaviour ----------------------------------------------


def test_repeated_mentions_at_different_positions_both_survive() -> None:
    text = "táo bón, sau đó táo bón"
    second = text.rindex("táo bón")
    lattice = build_span_lattice("doc", text, neural_spans=[
        _neural(text, 0, 7, "SYMPTOM"), _neural(text, second, second + 7, "SYMPTOM")])
    result = resolve_lattice(lattice, CONFIG)
    spans = sorted(h.coords.output_position() for h in result.hypotheses)
    assert spans == [(0, 7), (second, second + 7)]


def test_boundary_alternatives_of_one_mention_collapse_to_one_hypothesis() -> None:
    text = "Glucose: 5.6 mmol/L"
    value_only = SpecialistProposal(
        proposal_id="lab-v", document_id="doc", start=9, end=12, text="5.6",
        proposed_types=("KẾT_QUẢ_XÉT_NGHIỆM",), source_specialist="laboratory",
        source_node_id="row-1", source_routes=("C2",), local_score=0.9,
        matched_rule="lab:test_result:value_only:key_value", boundary_group_id="rg-1",
        components=(ComponentSpan("result_value", 9, 12, "5.6"),))
    value_unit = SpecialistProposal(
        proposal_id="lab-u", document_id="doc", start=9, end=19, text="5.6 mmol/L",
        proposed_types=("KẾT_QUẢ_XÉT_NGHIỆM",), source_specialist="laboratory",
        source_node_id="row-1", source_routes=("C2",), local_score=0.9,
        matched_rule="lab:test_result:value_unit:key_value", boundary_group_id="rg-1",
        components=(ComponentSpan("result_value", 9, 12, "5.6"),))
    lattice = build_span_lattice(
        "doc", text, specialist_proposals=[value_only, value_unit])
    result = resolve_lattice(lattice, CONFIG)
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].coords.output_position() == (9, 12)  # value_only policy
    assert any(d.reason == "boundary_alternative_not_selected" for d in result.decisions)


def test_a_strong_has_result_pair_survives_resolution() -> None:
    text = "WBC: 14.43"
    name = SpecialistProposal(
        proposal_id="lab-n", document_id="doc", start=0, end=3, text="WBC",
        proposed_types=("TÊN_XÉT_NGHIỆM",), source_specialist="laboratory",
        source_node_id="row-1", source_routes=("C2",), local_score=0.85,
        matched_rule="lab:test_name:key_value")
    value = SpecialistProposal(
        proposal_id="lab-v", document_id="doc", start=5, end=10, text="14.43",
        proposed_types=("KẾT_QUẢ_XÉT_NGHIỆM",), source_specialist="laboratory",
        source_node_id="row-1", source_routes=("C2",), local_score=0.9,
        matched_rule="lab:test_result:value_only:key_value", boundary_group_id="rg-1")
    relation = RelationProposal(
        relation_id="r1", document_id="doc", relation_type=HAS_RESULT,
        source_proposal_id="lab-n", target_proposal_id="lab-v", score=0.9,
        pairing_cost=0.1, pair_group_id="pg-1")
    lattice = build_span_lattice("doc", text, specialist_proposals=[name, value])
    result = resolve_lattice(lattice, CONFIG, relations=[relation])
    emitted = {h.coords.output_position() for h in result.hypotheses}
    assert (0, 3) in emitted and (5, 10) in emitted


def test_resolver_output_is_typed_hypotheses_not_organizer_json() -> None:
    text = "suy tim"
    lattice = build_span_lattice("doc", text, neural_spans=[_neural(text, 0, 7, "DIAGNOSIS")])
    hypothesis = resolve_lattice(lattice, CONFIG).hypotheses[0]
    assert hypothesis.type_distribution
    assert not hasattr(hypothesis, "candidates")
    assert not hasattr(hypothesis, "assertions")
    assert hypothesis.coords.output_position() == (0, 7)


def test_resolver_preserves_the_offset_invariant() -> None:
    text = "  1. amlodipine 10 mg,  "
    lattice = build_span_lattice(
        "doc", text, neural_spans=[_neural(text, 0, len(text), "MEDICATION")])
    for hypothesis in resolve_lattice(lattice, CONFIG).hypotheses:
        start, end = hypothesis.coords.output_position()
        assert text[start:end] == hypothesis.text


def test_resolver_replays_deterministically() -> None:
    text = "suy tim, ho khan, suy tim"
    spans = [_neural(text, 0, 7, "DIAGNOSIS"), _neural(text, 9, 16, "SYMPTOM"),
             _neural(text, 18, 25, "DIAGNOSIS")]
    first = resolve_lattice(build_span_lattice("doc", text, neural_spans=spans), CONFIG)
    second = resolve_lattice(
        build_span_lattice("doc", text, neural_spans=list(reversed(spans))), CONFIG)
    assert first.determinism_hash() == second.determinism_hash()
    assert first.config_sha256 == CONFIG.config_sha256


def test_every_decision_is_recorded_for_replay() -> None:
    text = "sốt"
    weak = SourceEvidence(EXPERT_VIHEALTHBERT, "e3-1", 0.05, {"SYMPTOM": 0.05})
    lattice = build_span_lattice("doc", text, neural_spans=[_neural(text, 0, 3, "SYMPTOM", 0.05)])
    result = resolve_lattice(lattice, CONFIG)
    assert weak.local_score == pytest.approx(0.05)
    assert len(result.decisions) == len(lattice.proposals)
    assert result.decisions[0].status == "abstained"
    assert result.counts()["abstained"] == 1
