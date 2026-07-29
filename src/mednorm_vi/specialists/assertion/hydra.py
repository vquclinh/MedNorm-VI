"""Assertion Hydra: deterministic assertion evidence for L5 (spec §8).

This is the wired A1-A3 path: section prior, cue detection, scope resolution. It
delegates every decision to :mod:`.cues`, which owns the lexicons and the spec §8.1
boundary model, and adds only the per-hypothesis plumbing the pipeline needs.

**What Audit 0052 fixed here.** The previous implementation asked, for each
hypothesis, whether *any* cue of a family appeared anywhere in a symmetric
±80-character window::

    local = text[start-80 : end+80].casefold()
    if any(cue in local for cue in NEGATION_CUES): labels.append("isNegated")

Three independent defects, all measured on
``tests/fixtures/phase1b/synthetic_medical_document.txt``:

1. **no direction.** A cue *after* the mention counted, so "sốt không giảm" negated
   a mention that precedes the cue.
2. **no §8.1 boundaries.** A cue in a different sentence or clause counted, so one
   document containing "âm tính", "Tiền sử" and "gia đình" anywhere assigned
   ``isNegated + isHistorical + isFamily`` to essentially every entity in it — ten
   such decisions in that one fixture.
3. **bare substring matching.** The family cue "ông" (grandfather) matches inside
   "kh**ông**" (not), so every negated sentence also read as family context.

Only the third was invisible in output: the first two were masked because
TEST_NAME/TEST_RESULT carry no ``assertions`` field, so ten wrong decisions were
computed and silently discarded. The moment a narrative DIAGNOSIS or SYMPTOM
appeared next to a lab block — which is what activating E3 does — they would have
reached L9. Assertions are 30% of the organizer metric and are scored by Jaccard, so
a spurious label against an empty gold set scores zero for that entity (spec §13.3).

Type eligibility is applied here too: a hypothesis whose type carries no
``assertions`` field is not given one, which both saves work and stops a future
over-firing bug from hiding the same way again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...resolution.models import EntityHypothesis
from ...schemas.constants import TYPE_BY_ORGANIZER_LABEL
from .cues import (
    DEFAULT_SCOPE_CHARACTERS,
    FAMILY_CUES,
    HISTORICAL_CUES,
    NEGATION_CUES,
    decide_from_cues,
)

HYDRA_CONTRACT_VERSION = "assertion-hydra-v2"

# Confidence attached to a label. A cue in scope is firmer evidence than a section
# prior alone, and the two are not averaged into one indistinguishable number.
SCORE_CUE_IN_SCOPE = 0.75
SCORE_SECTION_PRIOR = 0.55


@dataclass(frozen=True, slots=True)
class AssertionDecision:
    hypothesis_id: str
    labels: tuple[str, ...]
    scores: dict[str, float]
    evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Why this decision was reached, and whether L7 should adjudicate it (spec §8
    # stage A5). Both are reported rather than inferred from an empty label set,
    # because "no cue" and "a cue I could not scope" are different answers.
    source: str = ""
    uncertain: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "assertions": list(self.labels),
            "scores": dict(sorted(self.scores.items())),
            "evidence": {k: list(v) for k, v in sorted(self.evidence.items())},
            "source": self.source,
            "uncertain": self.uncertain,
            "contract_version": HYDRA_CONTRACT_VERSION,
            "empty_means_insufficient_evidence_not_false": True,
        }


def resolve_assertions(
    text: str,
    hypotheses: tuple[EntityHypothesis, ...],
    *,
    scope: int = DEFAULT_SCOPE_CHARACTERS,
) -> tuple[AssertionDecision, ...]:
    """Deterministic assertion labels for each hypothesis (spec §8 A1-A3).

    One decision per hypothesis, in input order, so the caller can zip by index as
    well as by ``hypothesis_id``. Insufficient evidence yields an **empty** label
    set, never a guessed ``false``: the governed corpus has zero assertion
    supervision (Audit 0042), so a confident negative would be invented.
    """
    out: list[AssertionDecision] = []
    for hypothesis in hypotheses:
        # `EntityHypothesis.entity_type` carries the organizer label; the cue module
        # reasons in internal enums.
        internal_type = TYPE_BY_ORGANIZER_LABEL.get(
            hypothesis.entity_type, hypothesis.entity_type)
        decision = decide_from_cues(
            text,
            mention_start=hypothesis.start,
            scope=scope,
            entity_type=internal_type,
        )
        prior_only = decision.source == "section_prior_only"
        scores = {label: 0.0 for label in ("isNegated", "isHistorical", "isFamily")}
        evidence: dict[str, tuple[str, ...]] = {}
        for item in decision.evidence:
            if item.label in decision.labels and item.within_scope:
                evidence.setdefault(item.label, ())
                evidence[item.label] = (*evidence[item.label], item.cue)
        for label in decision.labels:
            scores[label] = (
                SCORE_SECTION_PRIOR
                if prior_only or label not in evidence
                else SCORE_CUE_IN_SCOPE
            )
        out.append(AssertionDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            labels=decision.labels,
            scores=scores,
            evidence=evidence,
            source=decision.source,
            uncertain=decision.uncertain,
        ))
    return tuple(out)


__all__ = [
    "FAMILY_CUES",
    "HISTORICAL_CUES",
    "HYDRA_CONTRACT_VERSION",
    "NEGATION_CUES",
    "SCORE_CUE_IN_SCOPE",
    "SCORE_SECTION_PRIOR",
    "AssertionDecision",
    "resolve_assertions",
]
