"""Deterministic same-type overlap resolution (Phase 1C-A).

Overlap is resolved ONLY between accepted hypotheses of the SAME entity type
whose spans overlap but are not identical. Repeated occurrences at different
positions never overlap, so they always both survive — repeated mentions are
distinct concepts. Cross-type overlaps (e.g. TEST_NAME vs TEST_RESULT) are kept.
Never deduplicates by text.
"""

from __future__ import annotations

from dataclasses import replace

from .models import (
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNRESOLVED,
    EntityHypothesis,
    OverlapDecision,
)


def _overlaps(a: EntityHypothesis, b: EntityHypothesis) -> bool:
    return a.start < b.end and b.start < a.end and (a.start, a.end) != (b.start, b.end)


def _rank(h: EntityHypothesis) -> tuple[float, int, int, str]:
    """Deterministic preference: higher score, then wider, then earlier, then id."""
    return (-h.score, -(h.end - h.start), h.start, h.hypothesis_id)


def resolve_overlaps(
    hypotheses: list[EntityHypothesis], *, abstain_on_conflict: bool,
) -> list[EntityHypothesis]:
    """Suppress the weaker of two same-type overlapping hypotheses, deterministically."""
    out = {h.hypothesis_id: h for h in hypotheses}
    ids = [h.hypothesis_id for h in hypotheses]
    for i, aid in enumerate(ids):
        for bid in ids[i + 1:]:
            a, b = out[aid], out[bid]
            if a.status != STATUS_ACCEPTED or b.status != STATUS_ACCEPTED:
                continue
            if a.entity_type != b.entity_type or not _overlaps(a, b):
                continue
            # deterministic winner: higher score, then wider, then earlier, then id
            tie = _rank(a)[:2] == _rank(b)[:2]
            if tie and abstain_on_conflict:
                out[aid] = replace(a, status=STATUS_UNRESOLVED,
                                   overlap_decision=OverlapDecision(
                                       "abstained", b.hypothesis_id, "tied same-type overlap"))
                out[bid] = replace(b, status=STATUS_UNRESOLVED,
                                   overlap_decision=OverlapDecision(
                                       "abstained", a.hypothesis_id, "tied same-type overlap"))
                continue
            winner, loser = (a, b) if _rank(a) <= _rank(b) else (b, a)
            out[winner.hypothesis_id] = replace(
                winner, overlap_decision=OverlapDecision(
                    "winner", loser.hypothesis_id, "same-type overlap"))
            out[loser.hypothesis_id] = replace(
                loser, status=STATUS_REJECTED,
                rejection_reason=f"overlap_suppressed_by:{winner.hypothesis_id}",
                overlap_decision=OverlapDecision(
                    "suppressed", winner.hypothesis_id, "same-type overlap"))
    return [out[i] for i in ids]


__all__ = ["resolve_overlaps"]
