"""Route-gate diagnostics for the deterministic L3 experts (spec §5, Audit 0053).

E1 and E2 have always been route-gated in code — E2 refuses any node without C2,
E1 requires C1/C5 or a C3 node carrying strong medication evidence. What was missing
was **visibility**: nothing reported which nodes an expert was eligible for, which it
skipped, or why, so a routing defect could only be found by reading parser source.

Audit 0052 measured the cost of that blindness. Narrative sentences were scoring C2 at
0.85 from punctuation and a numeral alone, E2 fired inside its own route exactly as
designed, and the result was 11 false positives and 0 true positives on the governed
validation split. The gate was correct; the route was wrong, and nothing said so.

This module answers the five questions §2 of Milestone 2 asks, per document, without
storing clinical text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..case_router.models import NodeRouting
from ..lattice.models import EXPERT_LABORATORY_PARSER, EXPERT_MEDICATION_GRAMMAR

ROUTE_GATE_DIAGNOSTICS_VERSION = "route-gate-diagnostics-v1"

# The routes each deterministic expert is designed for (spec §5, §6.1, §6.2).
# Multi-label throughout: a node carrying C1 *and* C2 makes both experts eligible,
# and neither forces the other off.
EXPERT_ROUTES: Mapping[str, tuple[str, ...]] = {
    EXPERT_MEDICATION_GRAMMAR: ("C1", "C5", "C3"),
    EXPERT_LABORATORY_PARSER: ("C2",),
}

# The subset of those routes that activates the expert's full grammar. A C3 node
# reaches E1 only with strong deterministic medication evidence, which the parser
# itself decides; the diagnostic records eligibility, not that decision.
EXPERT_STRUCTURED_ROUTES: Mapping[str, tuple[str, ...]] = {
    EXPERT_MEDICATION_GRAMMAR: ("C1", "C5"),
    EXPERT_LABORATORY_PARSER: ("C2",),
}


@dataclass(frozen=True, slots=True)
class RouteGateDiagnostics:
    """Which nodes each expert was eligible for, and why the rest were skipped."""

    document_id: str
    eligible_nodes_by_expert: Mapping[str, int] = field(default_factory=dict)
    skipped_nodes_by_expert: Mapping[str, int] = field(default_factory=dict)
    proposals_by_expert: Mapping[str, int] = field(default_factory=dict)
    active_routes_by_node: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    route_gate_reasons: tuple[str, ...] = field(default_factory=tuple)
    version: str = ROUTE_GATE_DIAGNOSTICS_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_gate_diagnostics_version": self.version,
            "document_id": self.document_id,
            "eligible_nodes_by_expert": dict(sorted(self.eligible_nodes_by_expert.items())),
            "skipped_nodes_by_expert": dict(sorted(self.skipped_nodes_by_expert.items())),
            "proposals_by_expert": dict(sorted(self.proposals_by_expert.items())),
            "active_routes_by_node": {
                node: list(routes)
                for node, routes in sorted(self.active_routes_by_node.items())
            },
            "route_gate_reasons": list(self.route_gate_reasons),
            "contains_clinical_text": False,
        }


def build_route_gate_diagnostics(
    document_id: str,
    routings: Sequence[NodeRouting],
    *,
    proposals_by_expert: Mapping[str, int] | None = None,
) -> RouteGateDiagnostics:
    """Summarise route eligibility for the deterministic experts on one document."""
    eligible: dict[str, int] = {expert: 0 for expert in EXPERT_ROUTES}
    skipped: dict[str, int] = {expert: 0 for expert in EXPERT_ROUTES}
    active_routes: dict[str, tuple[str, ...]] = {}
    reasons: list[str] = []

    for routing in routings:
        tags = frozenset(routing.route_tags)
        active_routes[routing.node_id] = tuple(sorted(tags))
        for expert, routes in EXPERT_ROUTES.items():
            matched = tags & set(routes)
            if matched:
                eligible[expert] += 1
            else:
                skipped[expert] += 1
                reasons.append(
                    f"{routing.node_id}:{expert}:no_matching_route:"
                    f"active={','.join(sorted(tags)) or 'none'}:"
                    f"requires={','.join(routes)}")
        # Routes the router itself considered and withheld, carried through so a
        # suppressed route is visible here too rather than only inside L2.
        for gate_reason in routing.gate_reasons:
            reasons.append(f"{routing.node_id}:router:{gate_reason}")

    return RouteGateDiagnostics(
        document_id=document_id,
        eligible_nodes_by_expert=eligible,
        skipped_nodes_by_expert=skipped,
        proposals_by_expert=dict(proposals_by_expert or {}),
        active_routes_by_node=active_routes,
        route_gate_reasons=tuple(reasons),
    )


__all__ = [
    "EXPERT_ROUTES",
    "EXPERT_STRUCTURED_ROUTES",
    "ROUTE_GATE_DIAGNOSTICS_VERSION",
    "RouteGateDiagnostics",
    "build_route_gate_diagnostics",
]
