"""`min-cost-bipartite` matcher.

Deterministic optimal one-to-one assignment **within each organizer type**,
using a pure-Python Hungarian algorithm (no numerical dependency). Pairs whose
optimal cost exceeds ``max_matching_cost`` are rejected, leaving both entities
unmatched. Cost components are configurable:

  cost = w_wer·min(1, WER)
       + w_overlap·(1 − charIoU)
       + w_pos·min(1, |Δstart| / position_scale)
       + w_len·(|Δlen| / max(len))
       − w_exact·[text is identical]

Wrong types never enter the same cost matrix, so they never match.
"""

from __future__ import annotations

from ...assignment import min_cost_assignment as _hungarian
from ..models import EvaluationConfig, EvaluationEntity, MatchingDecision, MatchingResult
from ..wer import compute_wer
from .base import Matcher, char_iou, group_indices_by_type

STRATEGY = "min-cost-bipartite"
_PAD = 1e9


class MinCostBipartiteMatcher(Matcher):
    name = STRATEGY

    def __init__(self, config: EvaluationConfig) -> None:
        self._cfg = config
        w = config.cost_weights
        self._w_wer = float(w.get("token_wer", 0.5))
        self._w_overlap = float(w.get("char_overlap", 0.3))
        self._w_pos = float(w.get("position_distance", 0.15))
        self._w_len = float(w.get("boundary_length", 0.05))
        self._w_exact = float(w.get("exact_text_bonus", 0.5))
        self._pos_scale = float(w.get("position_scale", 50.0)) or 50.0

    def _pair_cost(self, g: EvaluationEntity, p: EvaluationEntity) -> float:
        wer = compute_wer(
            g.text, p.text, tokenization=self._cfg.tokenization, clipping_enabled=False
        ).raw_wer
        wer_c = min(1.0, wer)
        iou = char_iou(g.position, p.position)
        pos = min(1.0, abs(g.start - p.start) / self._pos_scale)
        len_g, len_p = g.end - g.start, p.end - p.start
        len_norm = abs(len_g - len_p) / max(len_g, len_p, 1)
        exact = 1.0 if g.text == p.text else 0.0
        return (
            self._w_wer * wer_c
            + self._w_overlap * (1.0 - iou)
            + self._w_pos * pos
            + self._w_len * len_norm
            - self._w_exact * exact
        )

    def match(
        self,
        gt: tuple[EvaluationEntity, ...],
        pred: tuple[EvaluationEntity, ...],
    ) -> MatchingResult:
        gt_by_type = group_indices_by_type(gt)
        pred_by_type = group_indices_by_type(pred)
        pairs: list[MatchingDecision] = []
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        for etype in sorted(set(gt_by_type) | set(pred_by_type)):
            gis = gt_by_type.get(etype, [])
            pis = pred_by_type.get(etype, [])
            if not gis or not pis:
                continue
            n = max(len(gis), len(pis))
            cost = [[_PAD] * n for _ in range(n)]
            for a, gi in enumerate(gis):
                for b, pi in enumerate(pis):
                    cost[a][b] = self._pair_cost(gt[gi], pred[pi])
            assignment = _hungarian(cost)
            for a, b in enumerate(assignment):
                if a >= len(gis) or b < 0 or b >= len(pis):
                    continue
                c = cost[a][b]
                if c > self._cfg.max_matching_cost:
                    continue
                gi, pi = gis[a], pis[b]
                matched_gt.add(gi)
                matched_pred.add(pi)
                pairs.append(
                    MatchingDecision(gt_index=gi, pred_index=pi, strategy=STRATEGY, cost=c)
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


__all__ = ["MinCostBipartiteMatcher", "STRATEGY"]
