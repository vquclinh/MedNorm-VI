"""L3 — Mention Factory (spec section 6).

Responsibility: produce a high-recall span lattice from experts E1-E7. NO expert
emits a final entity directly. Every proposal carries the coordinate triplet and
provenance.

Contract:
    propose_spans(graph, routes) -> list[SpanProposal]

Planned experts (interfaces only):
    E1 medication grammar, E2 lab parser, E3 ViHealthBERT span, E4 PhoBERT W2NER,
    E5 XLM-R MRC-NER, E6 GLiNER, E7 Qwen3-1.7B proposer (proposal-only).

Status: NOT IMPLEMENTED (bootstrap). Interface only. No models are loaded.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import DocumentGraph, RouteDecision, SpanProposal


class SpanExpert(Protocol):
    """Interface for a single Mention Factory expert (E1-E7)."""

    name: str

    def propose(
        self, graph: DocumentGraph, routes: list[RouteDecision]
    ) -> list[SpanProposal]:
        ...


def propose_spans(
    graph: DocumentGraph, routes: list[RouteDecision]
) -> list[SpanProposal]:
    """TODO(L3): fan out to routed experts and merge into a span lattice."""
    raise NotImplementedError("L3 Mention Factory is not implemented yet (bootstrap).")


__all__ = ["SpanExpert", "propose_spans"]
