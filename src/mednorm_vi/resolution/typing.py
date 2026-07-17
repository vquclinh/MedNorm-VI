"""Deterministic type assignment for a proposal group (Phase 1C-A).

Phase 1B proposals carry a single organizer label per specialist, so typing is a
direct read here. The resolver does not invent CHẨN_ĐOÁN / TRIỆU_CHỨNG types that
have no proposal source.
"""

from __future__ import annotations

from ..mention_factory.models import SpanProposal
from .models import RESOLVABLE_TYPES, TypeEvidence


def assign_type(group: list[SpanProposal]) -> tuple[str, TypeEvidence]:
    """Return (entity_type, evidence) for a group of boundary alternatives."""
    rep = group[0]
    types = tuple(sorted({t for p in group for t in p.proposed_types}))
    entity_type = rep.proposed_types[0] if rep.proposed_types else ""
    note = ""
    if len(types) > 1:
        note = f"multiple proposed types across group: {types}"
    if entity_type not in RESOLVABLE_TYPES:
        note = f"type {entity_type!r} not resolvable in Phase 1C-A"
    return entity_type, TypeEvidence(
        entity_type=entity_type, source_specialist=rep.source_specialist,
        proposed_types=types, note=note)


__all__ = ["assign_type"]
