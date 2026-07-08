"""MedNorm-VI typed data contracts (schemas).

These dataclasses define the interfaces passed between the nine layers. They are
contracts, not implementations — no parsing, retrieval, or model logic lives
here. See ``docs/architecture/LAYER_CONTRACTS.md``.
"""

from __future__ import annotations

from .constants import (
    ASSERTION_LABELS,
    CANDIDATE_ONTOLOGY_BY_TYPE,
    ENTITY_TYPES,
    N_SUBMISSION_DOCUMENTS,
    ORGANIZER_FIELDS_BY_TYPE,
    ORGANIZER_LABEL_BY_TYPE,
    ORGANIZER_LABELS,
    POSITION_IS_END_EXCLUSIVE,
    TYPE_BY_ORGANIZER_LABEL,
)
from .document import DocumentGraph, NormalizedView, SectionNode, SegmentNode
from .evidence import AssertionEvidence, CandidateEvidence, EvidenceBundle
from .hypotheses import SpanProposal, TypedHypothesis
from .prediction import EntityPrediction
from .routing import RouteDecision, RouteTag, SegmentPriors
from .spans import OffsetAlignment, Span, SpanCoordinates, SpanProvenance

__all__ = [
    # constants
    "ENTITY_TYPES",
    "ASSERTION_LABELS",
    "CANDIDATE_ONTOLOGY_BY_TYPE",
    "ORGANIZER_LABEL_BY_TYPE",
    "TYPE_BY_ORGANIZER_LABEL",
    "ORGANIZER_LABELS",
    "ORGANIZER_FIELDS_BY_TYPE",
    "POSITION_IS_END_EXCLUSIVE",
    "N_SUBMISSION_DOCUMENTS",
    # spans
    "Span",
    "SpanCoordinates",
    "OffsetAlignment",
    "SpanProvenance",
    # document (L1)
    "DocumentGraph",
    "NormalizedView",
    "SectionNode",
    "SegmentNode",
    # routing (L2)
    "RouteTag",
    "RouteDecision",
    "SegmentPriors",
    # mentions / hypotheses (L3, L4)
    "SpanProposal",
    "TypedHypothesis",
    # evidence (L5)
    "AssertionEvidence",
    "CandidateEvidence",
    "EvidenceBundle",
    # final output (L8/L9)
    "EntityPrediction",
]
