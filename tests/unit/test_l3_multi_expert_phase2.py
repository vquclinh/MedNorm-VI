"""Multi-expert L3 lattice merge and provenance for Phase 2 experts."""

from __future__ import annotations

import unicodedata

import pytest

from mednorm_vi.lattice import (
    EXPERT_GLINER,
    EXPERT_PHOBERT_W2NER,
    EXPERT_VIHEALTHBERT,
    ExpertSpanProposal,
    LatticeError,
    build_span_lattice,
)
from mednorm_vi.mention_factory.neural.decoding import NeuralSpan


def _expert(
    document_id: str,
    text: str,
    start: int,
    end: int,
    expert_id: str,
    proposal_id: str,
) -> ExpertSpanProposal:
    return ExpertSpanProposal(
        document_id=document_id,
        start=start,
        end=end,
        text=text[start:end],
        type_scores={"SYMPTOM": 0.8},
        local_score=0.8,
        expert_id=expert_id,
        proposal_id=proposal_id,
        original_start=start,
        original_end=end,
        model_revision="rev",
        checkpoint_sha256="UNAVAILABLE_UNTRAINED",
        config_sha256="d" * 64,
    )


def test_phase2_experts_merge_only_on_exact_coordinates_and_keep_sources() -> None:
    text = "ho khan và ho khan"
    first = text.index("ho khan")
    second = text.rindex("ho khan")
    lattice = build_span_lattice(
        "doc",
        text,
        neural_spans=(NeuralSpan(first, first + 7, "SYMPTOM", "ho khan", 0.7, 2),),
        expert_spans=(
            _expert("doc", text, first, first + 7, EXPERT_PHOBERT_W2NER, "e4-a"),
            _expert("doc", text, second, second + 7, EXPERT_GLINER, "e6-b"),
        ),
    )
    assert len(lattice.proposals) == 2
    first_node = lattice.proposals[0]
    assert first_node.expert_ids == (EXPERT_VIHEALTHBERT, EXPERT_PHOBERT_W2NER)
    assert {source.proposal_id for source in first_node.sources} == {"e3-doc-0001", "e4-a"}
    assert lattice.repeated_surface_forms() == ("ho khan",)


def test_phase2_lattice_rejects_bad_document_or_bad_offsets() -> None:
    text = "suy tim"
    with pytest.raises(LatticeError):
        build_span_lattice(
            "doc",
            text,
            expert_spans=(_expert("other", text, 0, 7, EXPERT_PHOBERT_W2NER, "bad"),),
        )
    bad = ExpertSpanProposal(
        document_id="doc",
        start=0,
        end=3,
        text="XXX",
        type_scores={"DIAGNOSIS": 0.8},
        local_score=0.8,
        expert_id=EXPERT_PHOBERT_W2NER,
        proposal_id="bad",
    )
    with pytest.raises(LatticeError):
        build_span_lattice("doc", text, expert_spans=(bad,))


def test_phase2_lattice_preserves_decomposed_unicode_offsets() -> None:
    text = unicodedata.normalize("NFD", "sốt cao")
    proposal = _expert("doc", text, 0, len(text), EXPERT_PHOBERT_W2NER, "e4")
    lattice = build_span_lattice("doc", text, expert_spans=(proposal,))
    assert lattice.proposals[0].text == text
    assert lattice.proposals[0].sources[0].model_revision == "rev"
