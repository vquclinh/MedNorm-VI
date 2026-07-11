"""`exact-text-occurrence` matcher.

Match by (organizer type, exact entity text), pairing occurrences in a stable
order with absolute position as the deterministic tie-breaker. The same surface
form at two different positions produces two distinct mentions that are matched
independently and never collapsed.
"""

from __future__ import annotations

from ..models import EvaluationEntity, MatchingDecision, MatchingResult
from .base import Matcher

STRATEGY = "exact-text-occurrence"


def _ordered_by_position(entities: tuple[EvaluationEntity, ...], indices: list[int]) -> list[int]:
    """Order indices by (start, end, original index) — deterministic tie-break."""
    return sorted(indices, key=lambda i: (entities[i].start, entities[i].end, i))


class ExactTextOccurrenceMatcher(Matcher):
    name = STRATEGY

    def match(
        self,
        gt: tuple[EvaluationEntity, ...],
        pred: tuple[EvaluationEntity, ...],
    ) -> MatchingResult:
        # (type, text) -> position-ordered index lists.
        gt_keyed: dict[tuple[str, str], list[int]] = {}
        pred_keyed: dict[tuple[str, str], list[int]] = {}
        for gi, ent in enumerate(gt):
            gt_keyed.setdefault((ent.type, ent.text), []).append(gi)
        for pi, ent in enumerate(pred):
            pred_keyed.setdefault((ent.type, ent.text), []).append(pi)

        pairs: list[MatchingDecision] = []
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        for key, gis in gt_keyed.items():
            pis = pred_keyed.get(key)
            if not pis:
                continue
            gis_ord = _ordered_by_position(gt, gis)
            pis_ord = _ordered_by_position(pred, pis)
            for gi, pi in zip(gis_ord, pis_ord, strict=False):
                matched_gt.add(gi)
                matched_pred.add(pi)
                pairs.append(
                    MatchingDecision(gt_index=gi, pred_index=pi, strategy=STRATEGY, cost=0.0)
                )
        pairs.sort(key=lambda d: (d.gt_index, d.pred_index))
        unmatched_gt = tuple(i for i in range(len(gt)) if i not in matched_gt)
        unmatched_pred = tuple(i for i in range(len(pred)) if i not in matched_pred)
        return MatchingResult(
            pairs=tuple(pairs),
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
            strategy=STRATEGY,
        )


__all__ = ["ExactTextOccurrenceMatcher", "STRATEGY"]
