"""Competition candidate top-k policy (Audit 0058 §7).

The organizer scores the ``candidates`` field by Jaccard against the gold set. That
detail decides the whole policy, because Jaccard is symmetric in its punishments: with
a single gold code, emitting one correct code scores ``1/1 = 1.0``, and emitting the
correct code plus one wrong one scores ``1/2 = 0.5``. A speculative extra candidate is
not a free lottery ticket — it costs exactly as much as a miss when it is wrong.

L8's existing band selection was tuned for recall under that misreading: it kept every
candidate within 60% of the best score in the best tier, which produced sets of 20 on
145 of 200 measured documents (Audit 0053 §8). Against Jaccard, a 20-code set
containing the right answer scores ``0.05``.

So this module narrows, and only narrows:

    medication   top-1, plus a second only on a near-exact score tie
    diagnosis    top-1, plus a second on a near tie *or* a genuine
                 specific/unspecified pair that the evidence supports

**It can never add.** :func:`apply_competition_topk` selects a prefix-with-reasons of
the list L8 already chose, so a code absent from L5's offered set cannot appear here —
the L8/L9 invariant is preserved structurally rather than by a downstream check.

The thresholds are fixed and written down rather than searched. No calibrated
probability exists anywhere in this pipeline (Audit 0053 §19), so a tuned threshold
would be fitted to an evidence score that is not a probability; a documented constant
is the more honest instrument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ...schemas.constants import CANDIDATE_ONTOLOGY_BY_TYPE, TYPE_BY_ORGANIZER_LABEL

COMPETITION_TOPK_VERSION = "competition-topk-policy-v1"

#: Hard ceiling. Neither ontology may emit more than this many candidates.
MAX_CANDIDATES = 2

#: A medication runner-up must be within this fraction of the leader to be emitted.
#: Deliberately tight: RxNorm products are distinguished by strength and dose form, so
#: two genuinely comparable products are close to identical under the ranking score.
MEDICATION_TIE_RATIO = 0.98

#: A diagnosis runner-up qualifies on a near tie at this ratio.
DIAGNOSIS_TIE_RATIO = 0.95

#: A specific/unspecified ICD pair is admitted at a looser ratio, because the pair is
#: evidence of a *structural* ambiguity rather than a scoring coincidence — but it must
#: still be supported, so an unrelated low scorer cannot ride in on the exemption.
DIAGNOSIS_SPECIFICITY_RATIO = 0.80

# Reason codes, recorded per candidate.
KEEP_TOP1 = "kept_top1"
KEEP_CLOSE_TIE = "kept_close_score_tie"
KEEP_SPECIFICITY_PAIR = "kept_specific_unspecified_pair"
DROP_BEYOND_TOPK = "dropped_beyond_competition_topk"
DROP_NOT_CLOSE = "dropped_not_within_tie_band"
DROP_DIFFERENT_TIER = "dropped_weaker_evidence_tier"
DROP_TYPE_TAKES_NO_CANDIDATES = "dropped_type_takes_no_candidates"


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One candidate as L8 ranked it. ``score`` is an evidence score, not a probability."""

    code: str
    score: float
    tier: str = ""


@dataclass(frozen=True, slots=True)
class TopKDecision:
    """Why one candidate survived or was cut by the competition policy."""

    code: str
    rank: int
    kept: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "rank": self.rank, "kept": self.kept, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TopKSelection:
    """The narrowed candidate list plus a full audit trail."""

    codes: tuple[str, ...]
    decisions: tuple[TopKDecision, ...] = field(default_factory=tuple)
    policy_version: str = COMPETITION_TOPK_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "codes": list(self.codes),
            "decisions": [d.as_dict() for d in self.decisions],
        }


def _undotted(code: str) -> str:
    return code.replace(".", "").strip().upper()


def is_specific_unspecified_pair(first: str, second: str) -> bool:
    """Whether two ICD codes are the same concept at two levels of specificity.

    ``A09`` against ``A090`` is the classic organizer ambiguity: the text supports the
    category but does not clearly state the detail that separates the subdivision. A
    strict prefix relation is the structural signature of that pair.
    """
    a, b = _undotted(first), _undotted(second)
    if not a or not b or a == b:
        return False
    return a.startswith(b) or b.startswith(a)


def ontology_for(entity_type: str) -> str | None:
    """Candidate ontology for an entity type, given either naming convention.

    Hypotheses carry the organizer's Vietnamese label (``THUỐC``); the constant table
    is keyed by the internal enum (``MEDICATION``). Accepting both means a caller
    cannot silently get ``None`` — and therefore an empty candidate list — by passing
    the label that every runtime object actually holds.
    """
    internal = TYPE_BY_ORGANIZER_LABEL.get(entity_type, entity_type)
    return CANDIDATE_ONTOLOGY_BY_TYPE.get(internal)


def apply_competition_topk(
    entity_type: str,
    candidates: Sequence[RankedCandidate],
) -> TopKSelection:
    """Narrow an L8-selected candidate list to the competition top-k policy.

    ``candidates`` must already be in L8's deterministic order. The result is a
    sub-sequence of that input in the same order, so ordering is inherited rather
    than recomputed and no code can be introduced.
    """
    ontology = ontology_for(entity_type)
    if ontology is None:
        # Spec §7.3: SYMPTOM and the TEST_* types carry no candidates at all.
        return TopKSelection(
            (),
            tuple(
                TopKDecision(c.code, rank, False, DROP_TYPE_TAKES_NO_CANDIDATES)
                for rank, c in enumerate(candidates, start=1)
            ),
        )
    if not candidates:
        return TopKSelection(())

    leader = candidates[0]
    kept = [leader.code]
    decisions = [TopKDecision(leader.code, 1, True, KEEP_TOP1)]

    for rank, candidate in enumerate(candidates[1:], start=2):
        if len(kept) >= MAX_CANDIDATES:
            decisions.append(TopKDecision(candidate.code, rank, False, DROP_BEYOND_TOPK))
            continue
        # A runner-up from a weaker evidence tier is not a tie, however close its
        # score: the tiers are ordinal and the scores are not comparable across them.
        if leader.tier and candidate.tier and candidate.tier != leader.tier:
            decisions.append(TopKDecision(candidate.code, rank, False, DROP_DIFFERENT_TIER))
            continue
        ratio = (candidate.score / leader.score) if leader.score > 0 else 0.0
        if ontology == "RXNORM":
            if ratio >= MEDICATION_TIE_RATIO:
                kept.append(candidate.code)
                decisions.append(TopKDecision(candidate.code, rank, True, KEEP_CLOSE_TIE))
            else:
                decisions.append(TopKDecision(candidate.code, rank, False, DROP_NOT_CLOSE))
            continue
        if ratio >= DIAGNOSIS_TIE_RATIO:
            kept.append(candidate.code)
            decisions.append(TopKDecision(candidate.code, rank, True, KEEP_CLOSE_TIE))
        elif ratio >= DIAGNOSIS_SPECIFICITY_RATIO and is_specific_unspecified_pair(
            leader.code, candidate.code
        ):
            kept.append(candidate.code)
            decisions.append(TopKDecision(candidate.code, rank, True, KEEP_SPECIFICITY_PAIR))
        else:
            decisions.append(TopKDecision(candidate.code, rank, False, DROP_NOT_CLOSE))

    return TopKSelection(tuple(kept), tuple(decisions))


__all__ = [
    "COMPETITION_TOPK_VERSION",
    "DIAGNOSIS_SPECIFICITY_RATIO",
    "DIAGNOSIS_TIE_RATIO",
    "DROP_BEYOND_TOPK",
    "DROP_DIFFERENT_TIER",
    "DROP_NOT_CLOSE",
    "DROP_TYPE_TAKES_NO_CANDIDATES",
    "KEEP_CLOSE_TIE",
    "KEEP_SPECIFICITY_PAIR",
    "KEEP_TOP1",
    "MAX_CANDIDATES",
    "MEDICATION_TIE_RATIO",
    "RankedCandidate",
    "TopKDecision",
    "TopKSelection",
    "apply_competition_topk",
    "is_specific_unspecified_pair",
    "ontology_for",
]
