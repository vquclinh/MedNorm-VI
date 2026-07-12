"""L3 — Mention Factory (Phase 1B): deterministic proposal sources.

Two deterministic specialists (medication grammar, laboratory parser) emit
``SpanProposal`` and ``RelationProposal`` objects only — never final entities.
L4 (later) decides which proposals survive and their final types/boundaries.
"""

from __future__ import annotations

from .merge import MergeDiagnostics, collect
from .models import (
    HAS_RESULT,
    ComponentSpan,
    RelationProposal,
    SpanProposal,
    SpecialistRunResult,
)
from .validation import validate_proposals

__all__ = [
    "SpanProposal",
    "RelationProposal",
    "ComponentSpan",
    "SpecialistRunResult",
    "HAS_RESULT",
    "collect",
    "MergeDiagnostics",
    "validate_proposals",
]
