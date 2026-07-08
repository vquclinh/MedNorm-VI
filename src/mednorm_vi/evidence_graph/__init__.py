"""L6 — Clinical Evidence Graph (spec section 11).

Responsibility: merge all evidence into one graph and compute global consistency
features. Nodes: spans, sections, cues, list items, lab values, ontology
candidates. Edges: has_result, modified_by, in_section, treats, overlaps,
same_surface, candidate_of. Repeated mentions at different offsets stay distinct.

Contract:
    build_graph(hypotheses, bundles, graph) -> EvidenceGraphResult

Status: NOT IMPLEMENTED (bootstrap). Interface only.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import DocumentGraph, EvidenceBundle, TypedHypothesis

# Edge types defined by the spec (section 11).
EDGE_TYPES = (
    "has_result",
    "modified_by",
    "in_section",
    "treats",
    "overlaps",
    "same_surface",
    "candidate_of",
)


class EvidenceGraphBuilder(Protocol):
    def build_graph(
        self,
        hypotheses: list[TypedHypothesis],
        bundles: list[EvidenceBundle],
        graph: DocumentGraph,
    ) -> object:
        ...


def build_graph(
    hypotheses: list[TypedHypothesis],
    bundles: list[EvidenceBundle],
    graph: DocumentGraph,
) -> object:
    """TODO(L6): relation reasoning + consistency features (beam/weighted-interval first)."""
    raise NotImplementedError("L6 Evidence Graph is not implemented yet (bootstrap).")


__all__ = ["EvidenceGraphBuilder", "build_graph", "EDGE_TYPES"]
