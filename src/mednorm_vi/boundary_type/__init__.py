"""L4 — Boundary & Type Resolver (spec section 7).

Responsibility: select/trim/merge overlapping proposals, assign types, and
ABSTAIN when wrong-type risk is high. Global span optimization must never
deduplicate by text alone.

Contract:
    resolve(proposals, graph) -> list[TypedHypothesis]

Status: NOT IMPLEMENTED (bootstrap). Interface only.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import DocumentGraph, SpanProposal, TypedHypothesis


class BoundaryTypeResolver(Protocol):
    """Interface for the L4 resolver."""

    def resolve(
        self, proposals: list[SpanProposal], graph: DocumentGraph
    ) -> list[TypedHypothesis]:
        ...


def resolve(
    proposals: list[SpanProposal], graph: DocumentGraph
) -> list[TypedHypothesis]:
    """TODO(L4): boundary ensemble + type scoring + abstention + global optimizer."""
    raise NotImplementedError("L4 Boundary & Type Resolver is not implemented yet (bootstrap).")


__all__ = ["BoundaryTypeResolver", "resolve"]
