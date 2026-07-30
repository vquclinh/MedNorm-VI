"""Deterministic same-type overlap resolution (Phase 1C-A).

Overlap is resolved ONLY between accepted hypotheses of the SAME entity type
whose spans overlap but are not identical. Repeated occurrences at different
positions never overlap, so they always both survive — repeated mentions are
distinct concepts. Cross-type overlaps (e.g. TEST_NAME vs TEST_RESULT) are kept.
Never deduplicates by text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

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


# ------------------------------------------------------------------------------
# v1 near-complete overlap competition (spec §7.3)
# ------------------------------------------------------------------------------


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Jaccard overlap of two half-open character intervals."""
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class OverlapCandidate:
    """One resolved hypothesis entering the global span competition."""

    candidate_id: str
    start: int
    end: int
    entity_type: str
    utility: float
    protected_partners: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class OverlapOutcome:
    """Which candidates survived the competition, and why the others did not."""

    survivors: tuple[str, ...]
    suppressed: tuple[tuple[str, str, float], ...]  # (loser, winner, iou)
    # Tied competitions the caller asked to abstain on rather than decide.
    # Kept separate from `suppressed` because abstention is not rejection: neither
    # span was judged wrong, the resolver simply could not tell them apart.
    abstained_ties: tuple[tuple[str, str, float], ...] = ()


def resolve_near_complete_overlaps(
    candidates: Sequence[OverlapCandidate], *,
    near_complete_iou: float, competition_penalty: float,
    suppress_cross_type: bool, protect_pairs: bool,
    abstain_on_tie: bool = False,
) -> OverlapOutcome:
    """Suppress the weaker of two **near-completely overlapping** same-type spans.

    Spec §7.3, in order:

    * never deduplicate by text alone — this function only ever compares
      coordinates, so identical text at two offsets never competes (their IoU is
      zero);
    * penalise near-complete overlap between two spans of the same type — below
      ``near_complete_iou`` both survive, because a nested or adjacent span may be
      a genuinely different mention;
    * preserve TEST_NAME/TEST_RESULT pairs with a strong ``has_result`` edge —
      a protected partner is never suppressed by the span it is paired with.

    ``abstain_on_tie`` migrates the retired resolver's ``abstain_on_conflict``: when
    two competitors have **equal utility**, marking both unresolved is more honest
    than breaking the tie on span geometry.
    """
    order = sorted(candidates, key=lambda c: (-c.utility, c.start, c.end, c.candidate_id))
    suppressed: dict[str, tuple[str, float]] = {}
    abstained: dict[str, tuple[str, float]] = {}
    for index, winner in enumerate(order):
        if winner.candidate_id in suppressed:
            continue
        for loser in order[index + 1:]:
            if loser.candidate_id in suppressed:
                continue
            if not suppress_cross_type and loser.entity_type != winner.entity_type:
                continue
            if protect_pairs and (
                    loser.candidate_id in winner.protected_partners
                    or winner.candidate_id in loser.protected_partners):
                continue
            iou = interval_iou((winner.start, winner.end), (loser.start, loser.end))
            if iou < near_complete_iou:
                continue
            # The penalty makes the competition auditable: the loser is only
            # suppressed when it still ranks below the winner after it applies.
            if abstain_on_tie and loser.utility == winner.utility:
                abstained[loser.candidate_id] = (winner.candidate_id, round(iou, 6))
                abstained[winner.candidate_id] = (loser.candidate_id, round(iou, 6))
                continue
            if loser.utility - competition_penalty >= winner.utility:
                continue
            suppressed[loser.candidate_id] = (winner.candidate_id, round(iou, 6))
    decided = set(suppressed) | set(abstained)
    survivors = tuple(
        c.candidate_id for c in candidates if c.candidate_id not in decided)
    return OverlapOutcome(
        survivors=survivors,
        suppressed=tuple(
            (loser, winner, iou) for loser, (winner, iou) in sorted(suppressed.items())),
        abstained_ties=tuple(
            (left, right, iou) for left, (right, iou) in sorted(abstained.items())))


__all__ = [
    "OverlapCandidate",
    "OverlapOutcome",
    "interval_iou",
    "resolve_near_complete_overlaps",
    "resolve_overlaps",
]
