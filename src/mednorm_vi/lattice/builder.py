"""Build the unified L3 span lattice from L1, L2 and the available experts.

Flow (spec §16 steps 2-4)::

    L1 DocumentGraph  ->  L2 route tags  ->  L3 unified span lattice

Only experts that are enabled and have valid local artifacts contribute. E5,
E6 and E7 use the same generic expert proposal contract as E3 before L4; when a
checkpoint is unavailable, the arm is reported as unavailable rather than
simulated.

Merging policy. Two proposals with the *exact* same coordinates become one node
whose evidence is the union of both sources; every source is kept individually.
Anything else — a different start, a different end, the same text at another
offset — stays a separate node. Text is never a merge key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..case_router.models import NodeRouting
from ..deterministic_baseline.models import Phase1BResult
from ..document_intelligence.models import DocumentGraph
from ..mention_factory.models import RelationProposal
from ..mention_factory.models import SpanProposal as SpecialistProposal
from ..mention_factory.neural.decoding import NeuralSpan
from ..schemas.constants import TYPE_BY_ORGANIZER_LABEL
from .models import (
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_VIHEALTHBERT,
    ExpertSpanProposal,
    LatticeError,
    SourceEvidence,
    SpanLattice,
    SpanProposal,
)
from .validation import validate_lattice

BUILDER_VERSION = "l3-span-lattice-v1"

_SPECIALIST_EXPERT: dict[str, str] = {
    "medication": EXPERT_MEDICATION_GRAMMAR,
    "laboratory": EXPERT_LABORATORY_PARSER,
}


class RouteIndex:
    """Offset -> (route tags, section) lookup over the L2 routing decisions.

    A span is attributed to the *narrowest* routed unit that contains it, so a
    list item wins over the line it sits in. A span that no routed unit contains
    keeps empty route/section evidence rather than being given a guessed one.
    """

    def __init__(self, routings: Sequence[NodeRouting]) -> None:
        self._units = sorted(
            ((r.start, r.end, r) for r in routings),
            key=lambda item: (item[1] - item[0], item[0]))

    def lookup(self, start: int, end: int) -> tuple[tuple[str, ...], str, str, str]:
        """``(route_tags, section, node_id, node_kind)`` for a span."""
        for unit_start, unit_end, routing in self._units:
            if unit_start <= start and end <= unit_end:
                return (tuple(routing.route_tags), routing.section_category or "",
                        routing.node_id, routing.node_kind)
        return ((), "", "", "")

    def routes_by_node(self) -> dict[str, tuple[str, ...]]:
        return {r.node_id: tuple(r.route_tags) for _s, _e, r in self._units}


def _pair_group_ids(
    relations: Sequence[RelationProposal],
) -> dict[str, tuple[str, ...]]:
    """Proposal id -> the relation pair groups it participates in (spec §6.2).

    Both endpoints of a ``has_result`` relation are indexed, so a TEST_NAME and its
    TEST_RESULT carry the same group id into the lattice and L4 can keep the pair
    together instead of resolving each side blind to the other.
    """
    out: dict[str, set[str]] = {}
    for relation in relations:
        group = getattr(relation, "pair_group_id", "")
        if not group:
            continue
        for endpoint in (relation.source_proposal_id, relation.target_proposal_id):
            if endpoint:
                out.setdefault(endpoint, set()).add(group)
    return {key: tuple(sorted(value)) for key, value in out.items()}


def _primary_route(routes: Sequence[str]) -> str:
    """One route label for reporting. Deterministic: the lowest case id."""
    return sorted(routes)[0] if routes else ""


def specialist_evidence(
    proposal: SpecialistProposal, index: RouteIndex,
    *, pair_group_ids: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[SourceEvidence, dict[str, float]]:
    """Adapt one deterministic E1/E2 proposal to lattice evidence.

    ``pair_group_ids`` maps a proposal id to the relation pair groups it belongs
    to, so an E2 TEST_NAME and its TEST_RESULT stay linked through the lattice
    (spec §6.2). It is optional so existing callers keep working; when omitted the
    evidence simply carries no relation references.
    """
    expert_id = _SPECIALIST_EXPERT.get(proposal.source_specialist)
    if expert_id is None:
        raise LatticeError(
            f"unknown deterministic specialist {proposal.source_specialist!r}")
    type_scores: dict[str, float] = {}
    for label in proposal.proposed_types:
        internal = TYPE_BY_ORGANIZER_LABEL.get(label)
        if internal is None:
            raise LatticeError(f"specialist proposed an unknown organizer label {label!r}")
        type_scores[internal] = max(type_scores.get(internal, 0.0),
                                    float(proposal.local_score))
    routes, section, node_id, node_kind = index.lookup(proposal.start, proposal.end)
    declared = tuple(proposal.source_routes)
    merged_routes = tuple(sorted(set(routes) | set(declared)))
    features = dict(proposal.features)
    features["grammar_component_count"] = float(len(proposal.components))
    # Carry the structured field decomposition through verbatim (Audit 0052).
    # The count above is kept for scoring; the components themselves are what
    # spec §10.1 linking needs, and reducing them to a count destroyed the parse.
    components = tuple(
        {
            "role": component.role,
            "start": component.start,
            "end": component.end,
            "text": component.text,
            "normalized": component.normalized or "",
            "detail": component.detail or "",
        }
        for component in proposal.components
    )
    relation_refs = tuple((pair_group_ids or {}).get(proposal.proposal_id, ()))
    evidence = SourceEvidence(
        expert_id=expert_id,
        proposal_id=proposal.proposal_id,
        local_score=float(proposal.local_score),
        type_scores=dict(type_scores),
        route=_primary_route(merged_routes),
        section=section,
        matched_rule=proposal.matched_rule or "",
        normalized_form=proposal.normalized_form or "",
        node_id=proposal.source_node_id or node_id,
        node_kind=proposal.source_node_kind or node_kind,
        parent_line_id=proposal.parent_line_id or "",
        boundary_group_id=proposal.boundary_group_id or "",
        routes=merged_routes,
        evidence_ids=tuple(proposal.evidence_ids),
        rule_ids=(proposal.matched_rule,) if proposal.matched_rule else (),
        features=features,
        warnings=tuple(proposal.warnings),
        config_version=proposal.config_version,
        lexicon_version=proposal.lexicon_version,
        components=components,
        relation_refs=relation_refs,
    )
    return evidence, type_scores


def neural_evidence(
    span: NeuralSpan, index: RouteIndex, *, document_id: str, ordinal: int,
) -> tuple[SourceEvidence, dict[str, float]]:
    """Adapt one decoded E3 span to lattice evidence."""
    type_scores = {span.entity_type: float(span.score)}
    routes, section, node_id, node_kind = index.lookup(span.start, span.end)
    evidence = SourceEvidence(
        expert_id=EXPERT_VIHEALTHBERT,
        proposal_id=f"e3-{document_id}-{ordinal:04d}",
        local_score=float(span.score),
        type_scores=dict(type_scores),
        route=_primary_route(routes),
        section=section,
        matched_rule="vihealthbert:token_run",
        normalized_form="",
        node_id=node_id,
        node_kind=node_kind,
        routes=tuple(routes),
        features={"neural_token_count": float(span.token_count),
                  "neural_mean_probability": float(span.score)},
    )
    return evidence, type_scores


def expert_span_evidence(proposal: ExpertSpanProposal) -> tuple[SourceEvidence, dict[str, float]]:
    """Adapt one trainable expert proposal to lattice evidence."""
    return proposal.as_source_evidence(), dict(proposal.type_scores)


def build_span_lattice(
    document_id: str,
    original_text: str,
    *,
    routings: Sequence[NodeRouting] = (),
    specialist_proposals: Sequence[SpecialistProposal] = (),
    neural_spans: Sequence[NeuralSpan] = (),
    expert_spans: Sequence[ExpertSpanProposal] = (),
    relations: Sequence[RelationProposal] = (),
    normalized_view: str = "",
    config_hash: str = "",
) -> SpanLattice:
    """Merge every expert's proposals into one deterministic span lattice.

    Fails loudly (``LatticeError``) on any offset/text violation — the invalid
    proposal is never trimmed, dropped, or repaired into something plausible.
    """
    index = RouteIndex(routings)
    pair_group_ids = _pair_group_ids(relations)
    warnings: list[str] = []
    merged: dict[tuple[int, int], list[SourceEvidence]] = {}
    scores: dict[tuple[int, int], dict[str, float]] = {}
    experts: set[str] = set()

    def register(
        start: int, end: int, text: str,
        evidence: SourceEvidence, type_scores: Mapping[str, float],
    ) -> None:
        if original_text[start:end] != text:
            raise LatticeError(
                f"{evidence.expert_id} proposed [{start}, {end}) whose text does "
                "not slice out of original_text (spec §4)")
        key = (start, end)
        merged.setdefault(key, []).append(evidence)
        bucket = scores.setdefault(key, {})
        for entity_type, score in type_scores.items():
            bucket[entity_type] = max(bucket.get(entity_type, 0.0), float(score))
        experts.add(evidence.expert_id)

    # Both expert streams are ordered canonically before ids are assigned, so the
    # lattice — and its determinism hash — does not depend on the order the
    # caller happened to collect proposals in.
    for proposal in sorted(
            specialist_proposals,
            key=lambda p: (p.start, p.end, p.source_specialist, p.proposal_id)):
        evidence, type_scores = specialist_evidence(
            proposal, index, pair_group_ids=pair_group_ids)
        register(proposal.start, proposal.end, proposal.text, evidence, type_scores)

    ordered_neural = sorted(
        neural_spans, key=lambda s: (s.start, s.end, s.entity_type))
    for ordinal, span in enumerate(ordered_neural, start=1):
        evidence, type_scores = neural_evidence(
            span, index, document_id=document_id, ordinal=ordinal)
        register(span.start, span.end, span.text, evidence, type_scores)

    ordered_expert_spans = sorted(
        expert_spans,
        key=lambda p: (p.start, p.end, p.expert_id, p.proposal_id),
    )
    for expert_proposal in ordered_expert_spans:
        if expert_proposal.document_id != document_id:
            raise LatticeError(
                f"{expert_proposal.expert_id} proposal {expert_proposal.proposal_id} belongs to "
                f"{expert_proposal.document_id!r}, expected {document_id!r}"
            )
        expert_proposal.validate_against(original_text)
        evidence, type_scores = expert_span_evidence(expert_proposal)
        register(
            expert_proposal.start,
            expert_proposal.end,
            expert_proposal.text,
            evidence,
            type_scores,
        )

    proposals: list[SpanProposal] = []
    merged_groups = 0
    for (start, end) in sorted(merged):
        sources = tuple(sorted(
            merged[(start, end)], key=lambda s: (s.expert_id, s.proposal_id)))
        if len({source.expert_id for source in sources}) > 1:
            merged_groups += 1
        routes = tuple(sorted({route for source in sources for route in source.routes}))
        sections = sorted({source.section for source in sources if source.section})
        if len(sections) > 1:
            warnings.append(f"conflicting_sections:{start}:{end}")
        features = {
            "source_count": float(len(sources)),
            "expert_count": float(len({source.expert_id for source in sources})),
            "deterministic_sources": float(
                sum(1 for source in sources if source.family == "deterministic")),
            "neural_sources": float(
                sum(1 for source in sources if source.family == "neural")),
            "open_type_sources": float(
                sum(1 for source in sources if source.family == "open_type")),
            "llm_interface_sources": float(
                sum(1 for source in sources if source.family == "llm_interface")),
            "span_length": float(end - start),
        }
        proposals.append(SpanProposal(
            document_id=document_id, start=start, end=end,
            text=original_text[start:end],
            type_scores=dict(sorted(scores[(start, end)].items())),
            sources=sources, routes=routes,
            section=sections[0] if sections else "",
            normalized_view=normalized_view,
            features=features))

    lattice = SpanLattice(
        document_id=document_id, original_text=original_text,
        proposals=tuple(proposals), routes_by_node=index.routes_by_node(),
        warnings=tuple(warnings), expert_ids=tuple(sorted(experts)),
        merged_coordinate_groups=merged_groups, config_hash=config_hash)
    validate_lattice(lattice)
    return lattice


def build_from_phase1b(
    graph: DocumentGraph,
    phase1b: Phase1BResult,
    *,
    neural_spans: Sequence[NeuralSpan] = (),
    expert_spans: Sequence[ExpertSpanProposal] = (),
    config_hash: str = "",
) -> SpanLattice:
    """Convenience wiring: an L1 graph plus a Phase 1B run plus decoded E3 spans."""
    return build_span_lattice(
        graph.document_id, graph.original_text,
        routings=phase1b.routings,
        specialist_proposals=phase1b.proposals,
        neural_spans=neural_spans,
        expert_spans=expert_spans,
        relations=phase1b.relations,
        config_hash=config_hash)


def lattice_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "BUILDER_VERSION",
    "RouteIndex",
    "build_from_phase1b",
    "build_span_lattice",
    "expert_span_evidence",
    "lattice_config_hash",
    "neural_evidence",
    "specialist_evidence",
]
