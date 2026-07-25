"""Deterministic weak-label generation from rule-backed proposals.

Also defines lightweight labeling-function / label-model CONTRACTS (no neural
pseudo-labeling, no Snorkel training locally): a labeling function emits a vote
with evidence provenance and may abstain; votes are aggregated deterministically
with explicit conflict detection and a confidence threshold. Large-scale weak-label
aggregation with a trained label model is a Colab/next-step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..mention_factory.models import SpanProposal
from ..schemas.constants import ASSERTION_LABELS, TYPE_BY_ORGANIZER_LABEL
from ..schemas.spans import Span
from .models import CanonicalAnnotation

ABSTAIN = ""


@dataclass(frozen=True, slots=True)
class LabelingFunctionVote:
    """One labeling function's vote on a span (ABSTAIN == no opinion)."""

    lf_id: str
    span: Span
    label: str                       # ENTITY_TYPE or assertion label, or ABSTAIN
    confidence: float
    evidence: str = ""               # rule/cue provenance (no raw restricted text)


@dataclass(frozen=True, slots=True)
class AggregatedLabel:
    span: Span
    label: str
    support: int
    conflict: bool
    mean_confidence: float
    voters: tuple[str, ...] = field(default_factory=tuple)


def aggregate_votes(
    votes: tuple[LabelingFunctionVote, ...], *, min_confidence: float = 0.5
) -> tuple[AggregatedLabel, ...]:
    """Deterministic per-span aggregation: majority label, conflict flag, abstention.

    A span with only ABSTAIN votes (or below the confidence threshold) yields no
    aggregated label. Conflict is flagged when >1 distinct non-abstain label is voted.
    """
    by_span: dict[tuple[int, int], list[LabelingFunctionVote]] = {}
    for v in votes:
        if v.label == ABSTAIN or v.confidence < min_confidence:
            continue
        by_span.setdefault((v.span.start, v.span.end), []).append(v)
    out: list[AggregatedLabel] = []
    for (s, e), vs in sorted(by_span.items()):
        counts = Counter(v.label for v in vs)
        top, support = counts.most_common(1)[0]
        out.append(AggregatedLabel(
            span=Span(s, e), label=top, support=support, conflict=len(counts) > 1,
            mean_confidence=round(sum(v.confidence for v in vs) / len(vs), 4),
            voters=tuple(sorted(v.lf_id for v in vs))))
    return tuple(out)


def is_valid_weak_label(label: str) -> bool:
    from ..schemas.constants import ENTITY_TYPES
    return label in ENTITY_TYPES or label in ASSERTION_LABELS


def from_span_proposals(
    document_id: str, proposals: tuple[SpanProposal, ...], *, min_score: float = 0.0
) -> tuple[CanonicalAnnotation, ...]:
    annotations: list[CanonicalAnnotation] = []
    for idx, proposal in enumerate(
        sorted(proposals, key=lambda p: (p.start, p.end, p.proposal_id)), 1
    ):
        if proposal.local_score < min_score:
            continue
        label = proposal.proposed_types[0] if proposal.proposed_types else ""
        annotations.append(
            CanonicalAnnotation(
                annotation_id=f"{document_id}-weak-{idx:04d}",
                document_id=document_id,
                span=proposal.span,
                text=proposal.text,
                entity_type=TYPE_BY_ORGANIZER_LABEL.get(label, label),
                source=f"weak:{proposal.source_specialist}:{proposal.matched_rule or 'unknown'}",
                confidence=proposal.local_score,
            )
        )
    return tuple(annotations)


__all__ = [
    "from_span_proposals", "LabelingFunctionVote", "AggregatedLabel",
    "aggregate_votes", "is_valid_weak_label", "ABSTAIN",
]
