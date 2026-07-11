"""`exact-position` matcher: same type AND identical ``[start, end)`` position."""

from __future__ import annotations

from ..models import EvaluationEntity, MatchingDecision, MatchingResult
from .base import Matcher, group_indices_by_type

STRATEGY = "exact-position"


class ExactPositionMatcher(Matcher):
    name = STRATEGY

    def match(
        self,
        gt: tuple[EvaluationEntity, ...],
        pred: tuple[EvaluationEntity, ...],
    ) -> MatchingResult:
        pred_by_type = group_indices_by_type(pred)
        # position -> queue of prediction indices (stable order), per type.
        pred_pos: dict[tuple[str, int, int], list[int]] = {}
        for etype, idxs in pred_by_type.items():
            for pi in idxs:
                key = (etype, pred[pi].start, pred[pi].end)
                pred_pos.setdefault(key, []).append(pi)

        pairs: list[MatchingDecision] = []
        matched_pred: set[int] = set()
        unmatched_gt: list[int] = []
        for gi, ent in enumerate(gt):
            key = (ent.type, ent.start, ent.end)
            queue = pred_pos.get(key)
            if queue:
                pi = queue.pop(0)
                matched_pred.add(pi)
                pairs.append(
                    MatchingDecision(gt_index=gi, pred_index=pi, strategy=STRATEGY, cost=0.0)
                )
            else:
                unmatched_gt.append(gi)
        unmatched_pred = [pi for pi in range(len(pred)) if pi not in matched_pred]
        return MatchingResult(
            pairs=tuple(pairs),
            unmatched_gt=tuple(unmatched_gt),
            unmatched_pred=tuple(unmatched_pred),
            strategy=STRATEGY,
        )


__all__ = ["ExactPositionMatcher", "STRATEGY"]
