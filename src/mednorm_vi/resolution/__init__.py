"""Deterministic L4 Boundary & Type Resolver foundation (Phase 1C-A).

Resolves Phase 1B proposals into typed ``EntityHypothesis`` objects (chosen
boundary + retained alternatives + accepted/rejected/unresolved status). No
ontology linking; exact offsets preserved; repeated occurrences kept distinct;
``has_result`` evidence retained without requiring pairing. Not final entities.
"""

from __future__ import annotations

from .models import (
    BoundaryAlternative,
    BoundaryEvidence,
    EntityHypothesis,
    OverlapDecision,
    ResolutionResult,
    TypeEvidence,
)
from .resolver import ResolverConfig, resolve
from .serialization import determinism_hash, to_debug_dict, to_json
from .validation import validate_result

__all__ = [
    "EntityHypothesis",
    "BoundaryAlternative",
    "BoundaryEvidence",
    "TypeEvidence",
    "OverlapDecision",
    "ResolutionResult",
    "ResolverConfig",
    "resolve",
    "validate_result",
    "to_debug_dict",
    "to_json",
    "determinism_hash",
]
