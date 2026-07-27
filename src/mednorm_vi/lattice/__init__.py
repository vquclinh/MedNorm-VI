"""L3 — the unified span lattice (spec §6).

One lattice per document, built from every expert that actually exists. Nodes are
spans with competing type evidence; no expert emits a final entity. Identity is
coordinates, never text, so a repeated mention at a second offset is a second
node.
"""

from __future__ import annotations

from .builder import (
    BUILDER_VERSION,
    RouteIndex,
    build_from_phase1b,
    build_span_lattice,
    lattice_config_hash,
)
from .models import (
    AVAILABLE_EXPERTS,
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_VIHEALTHBERT,
    FAMILY_DETERMINISTIC,
    FAMILY_NEURAL,
    LatticeError,
    SourceEvidence,
    SpanLattice,
    SpanProposal,
    order_proposals,
)
from .validation import validate_lattice, validate_proposal

__all__ = [
    "AVAILABLE_EXPERTS",
    "BUILDER_VERSION",
    "EXPERT_LABORATORY_PARSER",
    "EXPERT_MEDICATION_GRAMMAR",
    "EXPERT_VIHEALTHBERT",
    "FAMILY_DETERMINISTIC",
    "FAMILY_NEURAL",
    "LatticeError",
    "RouteIndex",
    "SourceEvidence",
    "SpanLattice",
    "SpanProposal",
    "build_from_phase1b",
    "build_span_lattice",
    "lattice_config_hash",
    "order_proposals",
    "validate_lattice",
    "validate_proposal",
]
