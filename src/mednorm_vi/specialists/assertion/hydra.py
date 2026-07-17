"""Assertion Hydra: independent deterministic assertion-label evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...resolution.models import EntityHypothesis

NEGATION_CUES = ("không", "chưa", "âm tính", "phủ nhận")
HISTORICAL_CUES = ("tiền sử", "đã từng", "trước đây", "cũ")
FAMILY_CUES = ("gia đình", "bố", "mẹ", "anh", "chị", "em")


@dataclass(frozen=True, slots=True)
class AssertionDecision:
    hypothesis_id: str
    labels: tuple[str, ...]
    scores: dict[str, float]
    evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _window(text: str, hypothesis: EntityHypothesis, radius: int = 80) -> str:
    start = max(0, hypothesis.start - radius)
    end = min(len(text), hypothesis.end + radius)
    return text[start:end].casefold()


def resolve_assertions(
    text: str, hypotheses: tuple[EntityHypothesis, ...]
) -> tuple[AssertionDecision, ...]:
    """Resolve assertion labels independently with cue/scope fallback.

    Section priors may add score mass in future calibrated models, but this
    deterministic fallback only emits a label when lexical cue evidence is in
    the local entity window.
    """
    out: list[AssertionDecision] = []
    for h in hypotheses:
        local = _window(text, h)
        labels: list[str] = []
        scores = {"isNegated": 0.0, "isHistorical": 0.0, "isFamily": 0.0}
        evidence: dict[str, tuple[str, ...]] = {}
        for label, cues in (
            ("isNegated", NEGATION_CUES),
            ("isHistorical", HISTORICAL_CUES),
            ("isFamily", FAMILY_CUES),
        ):
            hits = tuple(cue for cue in cues if cue in local)
            if hits:
                labels.append(label)
                scores[label] = 0.75
                evidence[label] = hits
        out.append(
            AssertionDecision(
                hypothesis_id=h.hypothesis_id,
                labels=tuple(labels),
                scores=scores,
                evidence=evidence,
            )
        )
    return tuple(out)


__all__ = ["AssertionDecision", "resolve_assertions"]
