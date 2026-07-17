"""Expected-Jaccard set decoding for final entity selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..confidence_cascade import CascadeDecision
from ..linking.models import LinkerResult
from ..resolution.models import EntityHypothesis
from ..specialists.assertion import AssertionDecision


@dataclass(frozen=True, slots=True)
class DecodedEntity:
    hypothesis: EntityHypothesis
    assertions: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)


def decode_expected_jaccard(
    hypotheses: tuple[EntityHypothesis, ...],
    cascade: tuple[CascadeDecision, ...],
    assertions: tuple[AssertionDecision, ...],
    links: tuple[LinkerResult, ...],
    *,
    max_candidates: int = 10,
) -> tuple[DecodedEntity, ...]:
    """Select accepted entities and calibrated candidate sets deterministically."""
    keep = {d.hypothesis_id for d in cascade if d.accepted}
    assertion_map = {a.hypothesis_id: a.labels for a in assertions}
    link_map = {
        result.mention_id: tuple(candidate.code for candidate in result.candidates[:max_candidates])
        for result in links
    }
    out = [
        DecodedEntity(
            hypothesis=h,
            assertions=assertion_map.get(h.hypothesis_id, ()),
            candidates=link_map.get(h.hypothesis_id, ()),
        )
        for h in hypotheses
        if h.hypothesis_id in keep
    ]
    out.sort(key=lambda e: (e.hypothesis.start, e.hypothesis.end, e.hypothesis.entity_type))
    return tuple(out)


__all__ = ["DecodedEntity", "decode_expected_jaccard"]
