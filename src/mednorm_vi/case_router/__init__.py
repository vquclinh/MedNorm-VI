"""L2 — Case Router (spec section 5).

Responsibility: assign one or more cases (C1-C7) to each segment. MULTI-LABEL —
no forced exclusivity. Declarative cues live in ``configs/routes.yaml``.

Contract:
    route_segments(graph: DocumentGraph, routes_config: dict) -> list[RouteDecision]

Status: NOT IMPLEMENTED (bootstrap). Interface only.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..schemas import DocumentGraph, RouteDecision


class CaseRouter(Protocol):
    """Interface for the L2 multi-label router."""

    def route_segments(
        self, graph: DocumentGraph, routes_config: dict[str, Any]
    ) -> list[RouteDecision]:
        ...


def route_segments(
    graph: DocumentGraph, routes_config: dict[str, Any]
) -> list[RouteDecision]:
    """TODO(L2): score C1-C7 per segment from lexical/structural/section signals."""
    raise NotImplementedError("L2 Case Router is not implemented yet (bootstrap).")


__all__ = ["CaseRouter", "route_segments"]
