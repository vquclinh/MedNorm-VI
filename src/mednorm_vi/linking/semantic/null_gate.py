"""NO_VALID_CANDIDATE as a first-class output (Audit 0071 §10).

Measured on the 69 high-consensus engineering records, the production linker emits at least
one candidate in **43 of 54** cases where independent adjudication concluded that no valid
concept exists. Broken out by ontology the picture is sharper still:

    ICD-10   20/31 false emissions   NULL F1 0.524
    RxNorm   23/23 false emissions   NULL F1 0.000

RxNorm never declines. A system that always returns its best-scoring candidate cannot express
"this mention has no concept", so a therapeutic class, an unrepresented local brand or a bare
drug category is answered with whatever was nearest - and every one of those is a wrong
candidate set. That is the most likely reason Audit 0070's coverage expansion *lowered*
J_candidates: adding candidates to a record that should emit none can only hurt.

The gate is deliberately evidence-based rather than threshold-tuned, and its thresholds are
derived from the governed KB and the adjudicated engineering records - **never** from public
leaderboard outcomes. It composes with, and does not replace, the Audit-0069 evidence tiers:
a candidate with exact accent-sensitive canonical support is never refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...kb.indexing import evidence as ev

NULL_GATE_VERSION = "null-gate-v1"

DECISION_ACCEPT = "ACCEPT"
DECISION_NO_VALID = "NO_VALID_CANDIDATE"

#: Reasons a candidate set is refused. Recorded per decision so a refusal can be argued with.
REASON_NO_CANDIDATES = "no_candidates_retrieved"
REASON_WEAK_EVIDENCE = "evidence_below_floor"
REASON_NO_MARGIN = "top1_top2_margin_too_small"
REASON_NO_SOURCE_AGREEMENT = "single_weak_source_only"
REASON_UNSUPPORTED_SPECIFICITY = "candidate_asserts_unstated_detail"
REASON_GENERIC_DRUG_CLASS = "mention_is_a_drug_class_not_a_product"

#: Evidence tiers strong enough that a candidate is never refused for weakness alone. These
#: are whole-string, accent-sensitive matches on governed text - the cases Audit 0069 built
#: the tier ladder to protect.
PROTECTED_TIERS: frozenset[str] = frozenset(
    {ev.TIER_A_EXACT_CANONICAL, ev.TIER_B_EXACT_ALIAS, ev.TIER_C_SEMANTIC_SYNONYM}
)

#: Vietnamese heads that name a *therapeutic class* rather than a dispensable product.
#: RxNorm has no concept for "an antibiotic", so a mention that is only a class head has no
#: valid normalization and must be allowed to return nothing.
DRUG_CLASS_HEADS: tuple[str, ...] = (
    "kháng sinh",
    "kháng viêm",
    "giảm đau",
    "hạ sốt",
    "kháng histamin",
    "corticoid",
    "vitamin",
    "thuốc bổ",
    "thuốc nam",
    "thuốc đông y",
    "dịch truyền",
    "thuốc hạ áp",
    "thuốc lợi tiểu",
    "thuốc an thần",
    "thuốc ngủ",
    "thuốc tiểu đường",
)


@dataclass(frozen=True, slots=True)
class NullGateEvidence:
    """What the gate is allowed to look at. All of it is computable without a model."""

    top_tier: str = ""
    top_score: float = 0.0
    second_score: float = 0.0
    dense_score: float | None = None
    reranker_score: float | None = None
    source_count: int = 0
    has_exact_evidence: bool = False
    unsupported_extra_tokens: int = 0
    mention_text: str = ""
    ontology: str = ""
    candidate_count: int = 0

    @property
    def margin(self) -> float:
        return self.top_score - self.second_score


@dataclass(frozen=True, slots=True)
class NullGateDecision:
    decision: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    version: str = NULL_GATE_VERSION

    @property
    def emits(self) -> bool:
        return self.decision == DECISION_ACCEPT

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "null_gate_version": self.version,
        }


@dataclass(frozen=True, slots=True)
class NullGateThresholds:
    """Calibrated on governed KB structure and adjudicated engineering records only.

    Explicitly NOT calibrated from public leaderboard score deltas, which would make the
    leaderboard a training signal.
    """

    minimum_reranker_score: float = 0.35
    minimum_dense_score: float = 0.55
    minimum_margin: float = 0.05
    require_sources_when_weak: int = 2


def is_drug_class_mention(mention: str) -> bool:
    """True when the mention names a therapeutic class rather than a dispensable product.

    A class head followed by a real drug name (`kháng sinh amoxicillin`) is a product mention
    and is NOT refused; only the bare class is.
    """
    value = " ".join((mention or "").split()).casefold().strip(" .,:;()")
    if not value:
        return False
    for head in DRUG_CLASS_HEADS:
        if value == head or value == f"thuốc {head}":
            return True
        if value.startswith(head):
            remainder = value[len(head) :].strip(" .,:;-")
            # Only quantity/generic filler may follow a bare class head.
            if not remainder or remainder in {"khác", "các loại", "liều cao", "đường uống"}:
                return True
    return False


def evaluate(
    evidence: NullGateEvidence, *, thresholds: NullGateThresholds | None = None
) -> NullGateDecision:
    """Decide whether to emit this candidate set at all."""
    limits = thresholds or NullGateThresholds()
    reasons: list[str] = []

    if evidence.candidate_count == 0:
        return NullGateDecision(DECISION_NO_VALID, (REASON_NO_CANDIDATES,))

    # A bare therapeutic class has no RxNorm product concept, however close the nearest
    # ingredient looks. This fires before the strength checks precisely because the nearest
    # ingredient often *does* score well.
    if evidence.ontology == "RXNORM" and is_drug_class_mention(evidence.mention_text):
        return NullGateDecision(DECISION_NO_VALID, (REASON_GENERIC_DRUG_CLASS,))

    # Whole-string accent-sensitive evidence on governed text is never refused for weakness.
    if evidence.top_tier in PROTECTED_TIERS or evidence.has_exact_evidence:
        return NullGateDecision(DECISION_ACCEPT)

    if evidence.reranker_score is not None:
        if evidence.reranker_score < limits.minimum_reranker_score:
            reasons.append(REASON_WEAK_EVIDENCE)
    elif evidence.dense_score is not None and evidence.dense_score < limits.minimum_dense_score:
        reasons.append(REASON_WEAK_EVIDENCE)

    if evidence.candidate_count > 1 and evidence.margin < limits.minimum_margin:
        reasons.append(REASON_NO_MARGIN)
    if evidence.source_count < limits.require_sources_when_weak:
        reasons.append(REASON_NO_SOURCE_AGREEMENT)
    if evidence.unsupported_extra_tokens > 0:
        reasons.append(REASON_UNSUPPORTED_SPECIFICITY)

    # One weak signal is not enough to refuse; two independent ones are. Refusing on a single
    # soft signal would trade the false-emission problem for a coverage problem.
    if len(reasons) >= 2:
        return NullGateDecision(DECISION_NO_VALID, tuple(reasons))
    return NullGateDecision(DECISION_ACCEPT, tuple(reasons))


__all__ = [
    "DECISION_ACCEPT",
    "DECISION_NO_VALID",
    "DRUG_CLASS_HEADS",
    "NULL_GATE_VERSION",
    "PROTECTED_TIERS",
    "REASON_GENERIC_DRUG_CLASS",
    "REASON_NO_CANDIDATES",
    "REASON_NO_MARGIN",
    "REASON_NO_SOURCE_AGREEMENT",
    "REASON_UNSUPPORTED_SPECIFICITY",
    "REASON_WEAK_EVIDENCE",
    "NullGateDecision",
    "NullGateEvidence",
    "NullGateThresholds",
    "evaluate",
    "is_drug_class_mention",
]
