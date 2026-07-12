"""Deterministic debug serialization for routing decisions."""

from __future__ import annotations

from typing import Any

from .models import NodeRouting


def routing_to_dict(r: NodeRouting) -> dict[str, Any]:
    return {
        "decision_id": r.decision_id,
        "node_id": r.node_id,
        "start": r.start,
        "end": r.end,
        "text": r.text,
        "section_category": r.section_category,
        "section_priors": dict(sorted(r.section_priors.items())),
        "router_version": r.router_version,
        "signals_version": r.signals_version,
        "warnings": list(r.warnings),
        "cases": [
            {
                "case": c.case,
                "score": c.score,
                "activated_specialists": list(c.activated_specialists),
                "fired_signals": [
                    {"name": s.name, "kind": s.kind, "weight": s.weight, "detail": s.detail}
                    for s in c.fired_signals
                ],
            }
            for c in sorted(r.cases, key=lambda c: c.case)
        ],
    }


def routings_to_list(routings: list[NodeRouting]) -> list[dict[str, Any]]:
    return [routing_to_dict(r) for r in routings]


__all__ = ["routing_to_dict", "routings_to_list"]
