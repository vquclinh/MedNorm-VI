"""L2 — Case Router (Phase 1B): deterministic multi-label routing over L1.

A route tag (C1-C7) is a PROCESSING CASE, not an entity type and not a final
prediction. Routing is multi-label; a node may carry several cases. Route scores
are deterministic heuristic evidence. See ``docs/case_router/``.
"""

from __future__ import annotations

from .models import CaseScore, NodeRouting, RouteSignal
from .router import CaseRouter, build_line_contexts
from .serialization import routing_to_dict, routings_to_list
from .signals import LineContext, RouterConfig, load_router_config
from .validation import validate_routings

__all__ = [
    "CaseRouter",
    "RouterConfig",
    "load_router_config",
    "build_line_contexts",
    "LineContext",
    "NodeRouting",
    "CaseScore",
    "RouteSignal",
    "routing_to_dict",
    "routings_to_list",
    "validate_routings",
]
