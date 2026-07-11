"""Matcher base class and shared helpers.

All strategies match entities **only within the same organizer type** — a
prediction can never align to a ground-truth entity of a different type. This is
what produces the organizer-described double error (one missing + one spurious)
for a wrong-type prediction. Every strategy is deterministic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import EvaluationEntity, MatchingResult


class Matcher(ABC):
    """Abstract deterministic one-to-one entity matcher."""

    name: str

    @abstractmethod
    def match(
        self,
        gt: tuple[EvaluationEntity, ...],
        pred: tuple[EvaluationEntity, ...],
    ) -> MatchingResult:
        """Return the matching decisions and unmatched index sets for one document."""


def group_indices_by_type(
    entities: tuple[EvaluationEntity, ...],
) -> dict[str, list[int]]:
    """Map organizer type -> list of entity indices (ascending, stable)."""
    groups: dict[str, list[int]] = {}
    for idx, ent in enumerate(entities):
        groups.setdefault(ent.type, []).append(idx)
    return groups


def char_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Character-span intersection-over-union for two ``[start, end)`` spans."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    len_a = max(0, a[1] - a[0])
    len_b = max(0, b[1] - b[0])
    union = len_a + len_b - inter
    return inter / union if union > 0 else 1.0


__all__ = ["Matcher", "group_indices_by_type", "char_iou"]
