"""Annotation review and active-learning queue helpers."""

from __future__ import annotations

from .models import CanonicalDocument, ReviewItem


def build_review_queue(document: CanonicalDocument) -> tuple[ReviewItem, ...]:
    items: list[ReviewItem] = []
    for ann in document.annotations:
        reasons: list[str] = []
        if ann.confidence < 0.75:
            reasons.append("low_confidence")
        if not ann.candidates and ann.entity_type in {"MEDICATION", "DIAGNOSIS"}:
            reasons.append("missing_candidate")
        for reason in reasons:
            items.append(
                ReviewItem(
                    item_id=f"{ann.annotation_id}:{reason}",
                    document_id=document.document_id,
                    annotation_id=ann.annotation_id,
                    reason=reason,
                    priority=1.0 - ann.confidence,
                )
            )
    return tuple(sorted(items, key=lambda item: (-item.priority, item.item_id)))


__all__ = ["build_review_queue"]
