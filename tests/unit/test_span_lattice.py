"""Unified L3 span-lattice invariants (Audit 0034)."""

from __future__ import annotations

import unicodedata

import pytest

from mednorm_vi.case_router.models import CaseScore, NodeRouting
from mednorm_vi.lattice import (
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_VIHEALTHBERT,
    LatticeError,
    build_span_lattice,
    validate_lattice,
)
from mednorm_vi.lattice.models import SourceEvidence, SpanProposal
from mednorm_vi.mention_factory.models import ComponentSpan
from mednorm_vi.mention_factory.models import SpanProposal as SpecialistProposal
from mednorm_vi.mention_factory.neural.decoding import NeuralSpan


def _neural(text: str, start: int, end: int, entity_type: str, score: float = 0.9):
    return NeuralSpan(start, end, entity_type, text[start:end], score, 1)


def _specialist(text: str, start: int, end: int, label: str, *, specialist: str = "medication",
                proposal_id: str = "p1", score: float = 0.8, rule: str = "med_grammar:full",
                components: tuple[ComponentSpan, ...] = (), warnings: tuple[str, ...] = ()):
    return SpecialistProposal(
        proposal_id=proposal_id, document_id="doc", start=start, end=end,
        text=text[start:end], proposed_types=(label,), source_specialist=specialist,
        source_node_id="line-0001", source_routes=("C1",), local_score=score,
        matched_rule=rule, components=components, warnings=warnings)


# -- spec §4: exact, end-exclusive offsets --------------------------------------


def test_span_slices_back_to_its_own_text() -> None:
    text = "Bệnh nhân bị suy tim."
    lattice = build_span_lattice("doc", text, neural_spans=[_neural(text, 13, 20, "DIAGNOSIS")])
    node = lattice.proposals[0]
    assert text[node.start:node.end] == node.text == "suy tim"
    validate_lattice(lattice)


def test_end_exclusive_span_may_touch_the_end_of_the_text() -> None:
    text = "suy tim"
    lattice = build_span_lattice("doc", text, neural_spans=[_neural(text, 0, 7, "DIAGNOSIS")])
    assert lattice.proposals[0].end == len(text)


def test_span_past_the_end_of_the_text_is_rejected() -> None:
    text = "suy tim"
    bad = NeuralSpan(0, 99, "DIAGNOSIS", text, 0.9, 1)
    with pytest.raises(LatticeError):
        build_span_lattice("doc", text, neural_spans=[bad])


def test_mismatched_text_is_rejected_not_repaired() -> None:
    text = "suy tim"
    bad = NeuralSpan(0, 3, "DIAGNOSIS", "XXX", 0.9, 1)
    with pytest.raises(LatticeError):
        build_span_lattice("doc", text, neural_spans=[bad])


def test_empty_span_is_rejected() -> None:
    text = "suy tim"
    with pytest.raises(LatticeError):
        build_span_lattice("doc", text, neural_spans=[NeuralSpan(3, 3, "DIAGNOSIS", "", 0.9, 1)])


def test_unsupported_type_is_rejected() -> None:
    with pytest.raises(LatticeError):
        SpanProposal(
            document_id="doc", start=0, end=1, text="a",
            type_scores={"NOT_A_TYPE": 1.0},
            sources=(SourceEvidence(EXPERT_VIHEALTHBERT, "p", 1.0, {}),))


def test_node_without_provenance_is_rejected() -> None:
    with pytest.raises(LatticeError):
        SpanProposal(document_id="doc", start=0, end=1, text="a",
                     type_scores={"DIAGNOSIS": 1.0}, sources=())


# -- spec §18.3 offset stress: Unicode, whitespace ------------------------------


def test_decomposed_unicode_offsets_are_exact() -> None:
    text = unicodedata.normalize("NFD", "sốt cao")
    assert text != unicodedata.normalize("NFC", text)
    lattice = build_span_lattice(
        "doc", text, neural_spans=[NeuralSpan(0, 4, "SYMPTOM", text[0:4], 0.9, 1)])
    node = lattice.proposals[0]
    assert text[node.start:node.end] == node.text
    validate_lattice(lattice)


def test_normalization_never_mutates_the_original_text() -> None:
    text = unicodedata.normalize("NFD", "sốt")
    lattice = build_span_lattice(
        "doc", text, neural_spans=[NeuralSpan(0, len(text), "SYMPTOM", text, 0.9, 1)],
        normalized_view=unicodedata.normalize("NFC", text))
    assert lattice.original_text == text
    assert lattice.proposals[0].text == text
    assert lattice.proposals[0].normalized_view != text


def test_repeated_spaces_and_newlines_do_not_shift_offsets() -> None:
    text = "sốt  cao\n\nho   khan"
    start = text.index("ho")
    lattice = build_span_lattice(
        "doc", text,
        neural_spans=[NeuralSpan(start, start + 8, "SYMPTOM", text[start:start + 8], 0.9, 1)])
    assert lattice.proposals[0].text == "ho   khan"[:8]
    validate_lattice(lattice)


# -- spec §5 case C7: never deduplicate by text alone ---------------------------


def test_repeated_text_at_different_offsets_stays_two_nodes() -> None:
    text = "táo bón, sau đó táo bón"
    first, second = 0, text.rindex("táo bón")
    lattice = build_span_lattice("doc", text, neural_spans=[
        _neural(text, first, first + 7, "SYMPTOM"),
        _neural(text, second, second + 7, "SYMPTOM"),
    ])
    assert len(lattice.proposals) == 2
    assert lattice.proposals[0].text == lattice.proposals[1].text
    assert lattice.proposals[0].start != lattice.proposals[1].start
    assert lattice.repeated_surface_forms() == ("táo bón",)


def test_exact_coordinate_duplicates_merge_evidence_and_keep_every_source() -> None:
    text = "amlodipine 10 mg"
    lattice = build_span_lattice(
        "doc", text,
        specialist_proposals=[_specialist(text, 0, 16, "THUỐC", proposal_id="med-1")],
        neural_spans=[_neural(text, 0, 16, "MEDICATION", 0.7)])
    assert len(lattice.proposals) == 1
    node = lattice.proposals[0]
    assert node.expert_ids == (EXPERT_MEDICATION_GRAMMAR, EXPERT_VIHEALTHBERT)
    assert len(node.sources) == 2
    assert {s.proposal_id for s in node.sources} == {"med-1", "e3-doc-0001"}
    assert node.type_scores["MEDICATION"] == pytest.approx(0.8)  # max of both
    assert lattice.merged_coordinate_groups == 1


def test_adjacent_spans_both_survive() -> None:
    text = "suy tim ho khan"
    lattice = build_span_lattice("doc", text, neural_spans=[
        _neural(text, 0, 7, "DIAGNOSIS"), _neural(text, 8, 15, "SYMPTOM")])
    assert len(lattice.proposals) == 2
    assert not lattice.proposals[0].overlaps(lattice.proposals[1])


def test_overlapping_spans_both_survive_in_the_lattice() -> None:
    text = "amlodipine 10 mg po"
    lattice = build_span_lattice("doc", text, neural_spans=[
        _neural(text, 0, 10, "MEDICATION"), _neural(text, 0, 19, "MEDICATION")])
    assert len(lattice.proposals) == 2
    assert lattice.proposals[0].overlaps(lattice.proposals[1])


# -- provenance -----------------------------------------------------------------


def test_every_required_provenance_field_is_preserved() -> None:
    text = "WBC: 14,43"
    routing = NodeRouting(
        decision_id="d1", document_id="doc", node_id="row-0001", start=0, end=len(text),
        text=text, cases=(CaseScore("C2", 0.8),), node_kind="table_row",
        section_category="laboratory")
    proposal = _specialist(
        text, 0, 3, "TÊN_XÉT_NGHIỆM", specialist="laboratory", proposal_id="lab-1",
        rule="lab:test_name:key_value",
        components=(ComponentSpan("test_name", 0, 3, "WBC", "wbc"),))
    lattice = build_span_lattice(
        "doc", text, routings=[routing], specialist_proposals=[proposal],
        neural_spans=[_neural(text, 5, 10, "TEST_RESULT", 0.6)])
    name_node = lattice.proposals[0]
    source = name_node.sources[0]
    assert name_node.start == 0 and name_node.end == 3
    assert name_node.text == "WBC"
    assert name_node.type_scores == {"TEST_NAME": pytest.approx(0.8)}
    assert source.expert_id == EXPERT_LABORATORY_PARSER
    assert source.local_score == pytest.approx(0.8)
    assert "C2" in source.routes
    assert name_node.section == "laboratory"
    assert source.matched_rule == "lab:test_name:key_value"
    assert source.features["grammar_component_count"] == 1.0
    assert source.node_id == "line-0001"


def test_route_lookup_prefers_the_narrowest_containing_unit() -> None:
    text = "Xét nghiệm: WBC 14,43"
    wide = NodeRouting("d1", "doc", "line-1", 0, len(text), text,
                       (CaseScore("C3", 0.6),), "line", section_category="laboratory")
    narrow = NodeRouting("d2", "doc", "row-1", 12, len(text), text[12:],
                         (CaseScore("C2", 0.9),), "table_row", section_category="laboratory")
    lattice = build_span_lattice(
        "doc", text, routings=[wide, narrow],
        neural_spans=[_neural(text, 12, 15, "TEST_NAME", 0.9)])
    assert lattice.proposals[0].routes == ("C2",)


def test_unknown_specialist_fails_loudly() -> None:
    text = "abc"
    bad = SpecialistProposal(
        proposal_id="x", document_id="doc", start=0, end=3, text="abc",
        proposed_types=("THUỐC",), source_specialist="mystery", source_node_id="n",
        source_routes=(), local_score=0.5)
    with pytest.raises(LatticeError):
        build_span_lattice("doc", text, specialist_proposals=[bad])


# -- determinism ----------------------------------------------------------------


def test_lattice_replays_deterministically() -> None:
    text = "suy tim, ho khan, suy tim"
    spans = [
        _neural(text, 0, 7, "DIAGNOSIS"), _neural(text, 9, 16, "SYMPTOM"),
        _neural(text, 18, 25, "DIAGNOSIS"),
    ]
    first = build_span_lattice("doc", text, neural_spans=spans)
    second = build_span_lattice("doc", text, neural_spans=list(reversed(spans)))
    assert first.determinism_hash() == second.determinism_hash()
    assert [p.coordinates for p in first.proposals] == [(0, 7), (9, 16), (18, 25)]


def test_summary_reports_expert_breakdown() -> None:
    text = "amlodipine 10 mg"
    lattice = build_span_lattice(
        "doc", text,
        specialist_proposals=[_specialist(text, 0, 10, "THUỐC")],
        neural_spans=[_neural(text, 0, 16, "MEDICATION")])
    summary = lattice.summary()
    assert summary["proposal_count"] == 2
    assert summary["proposals_by_expert"] == {
        EXPERT_MEDICATION_GRAMMAR: 1, EXPERT_VIHEALTHBERT: 1}
