"""ZS0 conservative zero-shot resolver (Audit 0048).

Neither learned L4 is available here: v1 measured **below** the E3-only baseline
(Audit 0034, exact F1 0.7039 against 0.7103) and v2 has no trained checkpoint. So
ZS0 resolves deterministically, and conservatively: when the evidence does not
decide, it abstains rather than guessing.

Merging is by **coordinate identity** — ``(start, end)`` — never by text. Two
occurrences of the same string at different absolute positions are two mentions
(spec §5 case C7), and collapsing them would silently delete one.

Every threshold lives in :class:`ResolverThresholds`, tracked in one config. No
threshold search is performed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..lattice.models import (
    EXPERT_GLINER,
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_QWEN_PROPOSER,
    ExpertSpanProposal,
)

RESOLVER_CONTRACT_VERSION = "zs0-conservative-resolver-v1"

# Deterministic grammars are structural evidence; the pretrained taggers are
# statistical. In a structured region the grammar's boundary wins, because it
# parsed the construction rather than recognizing a shape.
DETERMINISTIC_EXPERTS: frozenset[str] = frozenset(
    {EXPERT_MEDICATION_GRAMMAR, EXPERT_LABORATORY_PARSER})
PRETRAINED_EXPERTS: frozenset[str] = frozenset({EXPERT_GLINER, EXPERT_QWEN_PROPOSER})

ROUTE_STRUCTURED = "structured"
ROUTE_NARRATIVE = "narrative"

ABSTAIN_LOW_SCORE = "single_pretrained_expert_below_threshold"
ABSTAIN_TYPE_CONFLICT = "irreconcilable_type_disagreement"
ABSTAIN_WRONG_TYPE_RISK = "wrong_type_risk_above_threshold"


class ResolverError(ValueError):
    """Raised when the resolver is given something it cannot resolve safely."""


@dataclass(frozen=True, slots=True)
class ResolverThresholds:
    """Every tuneable number, in one place, tracked in one config.

    These are conservative defaults chosen once, not the product of a search.
    """

    min_single_expert_score: float = 0.50
    min_agreement_score: float = 0.30
    max_wrong_type_risk: float = 0.60
    # Two spans whose overlap exceeds this are treated as competing readings of
    # one mention rather than two mentions.
    near_complete_overlap_iou: float = 0.80

    def __post_init__(self) -> None:
        for name in ("min_single_expert_score", "min_agreement_score",
                     "max_wrong_type_risk", "near_complete_overlap_iou"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ResolverError(f"{name} must lie in [0, 1], got {value}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_single_expert_score": self.min_single_expert_score,
            "min_agreement_score": self.min_agreement_score,
            "max_wrong_type_risk": self.max_wrong_type_risk,
            "near_complete_overlap_iou": self.near_complete_overlap_iou,
            "threshold_search_performed": False,
        }


@dataclass(frozen=True, slots=True)
class ResolvedMention:
    """One surviving mention with every contributing proposal preserved."""

    start: int
    end: int
    text: str
    entity_type: str
    score: float
    provenance: tuple[str, ...]
    proposal_ids: tuple[str, ...]
    agreeing_experts: int
    abstained: bool = False
    abstain_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "score": self.score,
            "provenance": list(self.provenance),
            "proposal_ids": list(self.proposal_ids),
            "agreeing_experts": self.agreeing_experts,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
        }


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union else 0.0


def group_by_coordinate_identity(
    proposals: Sequence[ExpertSpanProposal],
) -> dict[tuple[int, int], list[ExpertSpanProposal]]:
    """Group on ``(start, end)`` only.

    Never on text: "sốt" at 10:13 and "sốt" at 92:95 are two mentions, and a
    text-keyed dictionary would keep one of them.
    """
    groups: dict[tuple[int, int], list[ExpertSpanProposal]] = {}
    for proposal in proposals:
        groups.setdefault((proposal.start, proposal.end), []).append(proposal)
    return groups


def _decide_type(
    group: Sequence[ExpertSpanProposal], *, route: str,
) -> tuple[str, float, int]:
    """``(type, score, agreeing_experts)`` for one coordinate-identical group."""
    scores: dict[str, float] = {}
    voters: dict[str, set[str]] = {}
    for proposal in group:
        for entity_type, score in proposal.type_scores.items():
            weight = float(score)
            if route == ROUTE_STRUCTURED and proposal.expert_id in DETERMINISTIC_EXPERTS:
                # The grammar parsed the construction; prefer its reading.
                weight += 1.0
            elif route == ROUTE_NARRATIVE and proposal.expert_id in PRETRAINED_EXPERTS:
                # Narrative text is what the contextual taggers are for.
                weight += 0.25
            scores[entity_type] = scores.get(entity_type, 0.0) + weight
            voters.setdefault(entity_type, set()).add(proposal.expert_id)
    best = max(sorted(scores), key=lambda t: (scores[t], t))
    total = sum(scores.values())
    normalized = scores[best] / total if total else 0.0
    return best, normalized, len(voters[best])


def resolve(
    proposals: Sequence[ExpertSpanProposal],
    *,
    route: str = ROUTE_NARRATIVE,
    thresholds: ResolverThresholds | None = None,
) -> tuple[ResolvedMention, ...]:
    """Deterministically resolve a document's proposals.

    Order of operations: merge coordinate-identical proposals, decide each
    group's type from the combined evidence, abstain where the evidence is thin,
    then collapse near-complete overlaps in favour of the better-supported span.
    Output is sorted by ``(start, end, type)`` so two runs agree exactly.
    """
    limits = thresholds or ResolverThresholds()
    resolved: list[ResolvedMention] = []

    for (start, end), group in sorted(group_by_coordinate_identity(proposals).items()):
        entity_type, score, agreeing = _decide_type(group, route=route)
        provenance = tuple(sorted({p.expert_id for p in group}))
        proposal_ids = tuple(sorted(p.proposal_id for p in group))
        text = group[0].text

        abstain_reason = ""
        # Wrong-type risk: how much of the evidence pointed somewhere else.
        wrong_type_risk = 1.0 - score
        if agreeing == 1 and len(group) == 1:
            only = group[0]
            if (only.expert_id in PRETRAINED_EXPERTS
                    and only.local_score < limits.min_single_expert_score):
                abstain_reason = ABSTAIN_LOW_SCORE
        if not abstain_reason and score < limits.min_agreement_score:
            abstain_reason = ABSTAIN_TYPE_CONFLICT
        if not abstain_reason and wrong_type_risk > limits.max_wrong_type_risk:
            abstain_reason = ABSTAIN_WRONG_TYPE_RISK

        resolved.append(ResolvedMention(
            start=start, end=end, text=text, entity_type=entity_type,
            score=score, provenance=provenance, proposal_ids=proposal_ids,
            agreeing_experts=agreeing,
            abstained=bool(abstain_reason), abstain_reason=abstain_reason))

    return _collapse_near_complete_overlaps(resolved, limits=limits)


def _collapse_near_complete_overlaps(
    mentions: Sequence[ResolvedMention], *, limits: ResolverThresholds,
) -> tuple[ResolvedMention, ...]:
    """Keep the better-supported of two near-identical spans, deterministically.

    Genuine nesting (a short span inside a much longer one) is left alone: both
    can be real. Only near-complete overlap is treated as one mention proposed
    twice with slightly different edges.
    """
    kept: list[ResolvedMention] = []
    for candidate in sorted(
            mentions, key=lambda m: (m.start, m.end, m.entity_type)):
        replaced = False
        for index, existing in enumerate(kept):
            if _iou((existing.start, existing.end),
                    (candidate.start, candidate.end)) < limits.near_complete_overlap_iou:
                continue
            better = max(
                (existing, candidate),
                key=lambda m: (m.agreeing_experts, m.score,
                               m.end - m.start, -m.start))
            merged = ResolvedMention(
                start=better.start, end=better.end, text=better.text,
                entity_type=better.entity_type, score=better.score,
                provenance=tuple(sorted(
                    set(existing.provenance) | set(candidate.provenance))),
                proposal_ids=tuple(sorted(
                    set(existing.proposal_ids) | set(candidate.proposal_ids))),
                agreeing_experts=max(
                    existing.agreeing_experts, candidate.agreeing_experts),
                abstained=better.abstained, abstain_reason=better.abstain_reason)
            kept[index] = merged
            replaced = True
            break
        if not replaced:
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda m: (m.start, m.end, m.entity_type)))


def emitted(mentions: Sequence[ResolvedMention]) -> tuple[ResolvedMention, ...]:
    """The mentions that survive abstention and are actually emitted."""
    return tuple(m for m in mentions if not m.abstained)


def assert_span_unchanged(
    mention: ResolvedMention, original_text: str,
) -> None:
    """Linking must never move a span (spec §4).

    Called after candidate generation, so a linker that "improved" a boundary to
    match an ontology alias is caught rather than shipped.
    """
    if original_text[mention.start:mention.end] != mention.text:
        raise ResolverError(
            f"span {mention.start}:{mention.end} no longer reproduces its text; "
            "linking must never change a span")


__all__ = [
    "ABSTAIN_LOW_SCORE",
    "ABSTAIN_TYPE_CONFLICT",
    "ABSTAIN_WRONG_TYPE_RISK",
    "DETERMINISTIC_EXPERTS",
    "PRETRAINED_EXPERTS",
    "RESOLVER_CONTRACT_VERSION",
    "ROUTE_NARRATIVE",
    "ROUTE_STRUCTURED",
    "ResolvedMention",
    "ResolverError",
    "ResolverThresholds",
    "assert_span_unchanged",
    "emitted",
    "group_by_coordinate_identity",
    "resolve",
]
