"""Set-based Jaccard for assertions and candidates, with published empty-set rules.

Confirmed empty-set behavior:
  * GT empty and prediction empty      -> 1.0
  * GT empty and prediction non-empty  -> 0.0
  * otherwise                          -> |intersection| / |union|

Confirmed candidate aggregation weight: ``len(ground_truth_candidates) + 1``.

Assertions/candidates are deterministically deduplicated (order-preserving)
before set conversion; duplicates are surfaced as diagnostics.
"""

from __future__ import annotations

from .models import SetSimilarityBreakdown


def _dedup_ordered(items: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (deduped-in-order, duplicates-in-order)."""
    seen: set[str] = set()
    deduped: list[str] = []
    duplicates: list[str] = []
    for it in items:
        if it in seen:
            duplicates.append(it)
        else:
            seen.add(it)
            deduped.append(it)
    return tuple(deduped), tuple(duplicates)


def jaccard_breakdown(
    kind: str,
    gt_items: tuple[str, ...],
    pred_items: tuple[str, ...],
) -> SetSimilarityBreakdown:
    """Compute a deterministic Jaccard breakdown for one entity's set field."""
    gt_dedup, gt_dups = _dedup_ordered(gt_items)
    pred_dedup, pred_dups = _dedup_ordered(pred_items)
    gt_set = set(gt_dedup)
    pred_set = set(pred_dedup)

    inter = tuple(x for x in gt_dedup if x in pred_set)
    union_list: list[str] = list(gt_dedup)
    for x in pred_dedup:
        if x not in gt_set:
            union_list.append(x)
    union = tuple(union_list)
    missing = tuple(x for x in gt_dedup if x not in pred_set)
    extra = tuple(x for x in pred_dedup if x not in gt_set)

    if not gt_set and not pred_set:
        score = 1.0
    elif not gt_set and pred_set:
        score = 0.0
    else:
        score = len(inter) / len(union) if union else 0.0

    # Confirmed candidate weight: len(GT candidates) + 1. We use the deduped GT
    # count (duplicates are data errors and are surfaced separately).
    weight = float(len(gt_dedup) + 1) if kind == "candidates" else 1.0

    return SetSimilarityBreakdown(
        kind=kind,
        gt_ordered=gt_items,
        pred_ordered=pred_items,
        gt_deduped=gt_dedup,
        pred_deduped=pred_dedup,
        intersection=inter,
        union=union,
        missing=missing,
        extra=extra,
        jaccard=score,
        gt_duplicates=gt_dups,
        pred_duplicates=pred_dups,
        weight=weight,
    )


__all__ = ["jaccard_breakdown"]
