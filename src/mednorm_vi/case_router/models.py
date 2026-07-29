"""L2 Case Router contracts (Phase 1B).

A route tag is a PROCESSING CASE (C1-C7), not an entity type and not a final
prediction. Routing is MULTI-LABEL: a node may carry several cases. Route scores
are deterministic heuristic evidence, not probabilities.

Reuses ``schemas.routing.RouteTag`` / ``RouteDecision`` for the multi-label
outcome and adds Phase-1B specifics (signal breakdown, activated specialists,
section priors, decision id).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.routing import RouteDecision, RouteTag


@dataclass(frozen=True, slots=True)
class RouteSignal:
    """One fired routing signal contributing evidence to a case."""

    case: str
    name: str
    kind: str  # "hard" | "soft" | "structural" | "derived"
    weight: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CaseScore:
    """The accumulated evidence for a single case on one node."""

    case: str
    score: float
    fired_signals: tuple[RouteSignal, ...] = field(default_factory=tuple)
    activated_specialists: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class NodeRouting:
    """The multi-label routing outcome for one L1 node."""

    decision_id: str
    document_id: str
    node_id: str
    start: int
    end: int
    text: str
    cases: tuple[CaseScore, ...]
    node_kind: str = "line"  # canonical routable unit kind (list_item/table_row/sentence/line)
    parent_line_id: str | None = None
    section_category: str | None = None
    section_priors: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    router_version: str = ""
    signals_version: str = ""
    # Why a case that scored was NOT attached (Audit 0053). Reported so a
    # suppressed route is auditable rather than silent; a route withheld for lack
    # of positive evidence must be as visible as one that fired.
    gate_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def route_tags(self) -> tuple[str, ...]:
        return tuple(c.case for c in self.cases)

    def has_case(self, case: str) -> bool:
        return any(c.case == case for c in self.cases)

    def activated_specialists(self) -> tuple[str, ...]:
        out: list[str] = []
        for c in self.cases:
            for s in c.activated_specialists:
                if s not in out:
                    out.append(s)
        return tuple(out)

    def as_route_decision(self) -> RouteDecision:
        """Bridge to the schema-level multi-label ``RouteDecision``."""
        tags = tuple(
            RouteTag(
                case=c.case,
                score=c.score,
                fired_signals=tuple(s.name for s in c.fired_signals),
                applied_priors={k: v for k, v in self.section_priors.items()},
            )
            for c in self.cases
        )
        return RouteDecision(segment_id=self.node_id, tags=tags)


__all__ = ["RouteSignal", "CaseScore", "NodeRouting"]
