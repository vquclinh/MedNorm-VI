"""Deterministic weak-label generation from rule-backed proposals."""

from __future__ import annotations

from ..mention_factory.models import SpanProposal
from ..schemas.constants import TYPE_BY_ORGANIZER_LABEL
from .models import CanonicalAnnotation


def from_span_proposals(
    document_id: str, proposals: tuple[SpanProposal, ...], *, min_score: float = 0.0
) -> tuple[CanonicalAnnotation, ...]:
    annotations: list[CanonicalAnnotation] = []
    for idx, proposal in enumerate(
        sorted(proposals, key=lambda p: (p.start, p.end, p.proposal_id)), 1
    ):
        if proposal.local_score < min_score:
            continue
        label = proposal.proposed_types[0] if proposal.proposed_types else ""
        annotations.append(
            CanonicalAnnotation(
                annotation_id=f"{document_id}-weak-{idx:04d}",
                document_id=document_id,
                span=proposal.span,
                text=proposal.text,
                entity_type=TYPE_BY_ORGANIZER_LABEL.get(label, label),
                source=f"weak:{proposal.source_specialist}:{proposal.matched_rule or 'unknown'}",
                confidence=proposal.local_score,
            )
        )
    return tuple(annotations)


__all__ = ["from_span_proposals"]
