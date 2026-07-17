"""Confidence cascade and critic/adjudicator readiness gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..resolution.models import EntityHypothesis


class MissingCascadeCheckpointError(RuntimeError):
    """Raised when an enabled critic/adjudicator checkpoint is missing."""


@dataclass(frozen=True, slots=True)
class CascadeDecision:
    hypothesis_id: str
    accepted: bool
    calibrated_score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def require_checkpoint(path: str | Path, *, role: str) -> None:
    if not Path(path).exists():
        raise MissingCascadeCheckpointError(f"missing {role} checkpoint: {path}")


def apply_confidence_cascade(
    hypotheses: tuple[EntityHypothesis, ...],
    *,
    threshold: float = 0.15,
) -> tuple[CascadeDecision, ...]:
    """Deterministic fallback cascade over resolver scores."""
    decisions: list[CascadeDecision] = []
    for h in hypotheses:
        accepted = h.status == "accepted" and h.score >= threshold
        reasons = ("resolver_accepted",) if accepted else ("below_threshold_or_rejected",)
        decisions.append(
            CascadeDecision(
                hypothesis_id=h.hypothesis_id,
                accepted=accepted,
                calibrated_score=h.score,
                reasons=reasons,
            )
        )
    return tuple(decisions)


__all__ = [
    "CascadeDecision",
    "MissingCascadeCheckpointError",
    "apply_confidence_cascade",
    "require_checkpoint",
]
