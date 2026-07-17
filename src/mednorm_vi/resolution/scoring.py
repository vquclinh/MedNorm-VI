"""Deterministic hypothesis scoring (Phase 1C-A).

Scores are deterministic heuristics — NOT calibrated probabilities. The base is
the chosen proposal's ``local_score``, with a small, fixed adjustment for extra
corroborating boundary alternatives in the same group.
"""

from __future__ import annotations

from ..mention_factory.models import SpanProposal


def score_hypothesis(chosen: SpanProposal, group: list[SpanProposal]) -> float:
    base = float(chosen.local_score)
    corroboration = 0.02 * (len(group) - 1)
    return round(min(1.0, base + corroboration), 6)


__all__ = ["score_hypothesis"]
