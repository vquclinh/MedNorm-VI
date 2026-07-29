"""Deterministic cue and scope evidence for the Assertion Hydra (spec §8).

Spec §8 stages A1-A6 resolve an assertion from section prior, cue, scope,
entity-cue relation, adjudication and set calibration. Stages A2 (cue detector) and
A3 (scope resolver) have a deterministic form that needs no trained head, and this
module is it — the single source of truth for the cue families and the scope window
used anywhere in L5.

Three rules from spec §4.3 and §8:

* **a cue only contributes evidence.** "Keyword presence must never label every
  entity indiscriminately", so a cue outside the scope window decides nothing;
* **an entity may carry multiple assertions.** Decoding is multi-label, never an
  exclusive softmax;
* **insufficient evidence yields an empty set, not a guessed ``false``.** The
  governed corpus has zero assertion supervision (Audit 0042), so a confident
  negative here would be thousands of labels invented out of nothing. Under
  Jaccard, an extra predicted label against an empty gold set scores zero
  (spec §13.3) — empty is the cheap error and a guess is the expensive one.

The lexicons are deliberately small and high-precision. Spec §4.3 requires
high-coverage lexicons for production, mined and regression-tested before
promotion; that expansion is a separate, evidence-backed milestone, and inventing
breadth here without regression data would trade precision for nothing measurable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...schemas.constants import ASSERTION_LABELS

CUE_CONTRACT_VERSION = "assertion-cue-scope-v1"

IS_NEGATED = "isNegated"
IS_FAMILY = "isFamily"
IS_HISTORICAL = "isHistorical"
SUPPORTED_ASSERTIONS: tuple[str, ...] = (IS_NEGATED, IS_FAMILY, IS_HISTORICAL)

# Vietnamese cue lexicons, one family per assertion label (spec §4.3 "alias/cue
# family"). Longer forms are listed alongside their heads because the scope
# decision uses the matched cue's own end position.
NEGATION_CUES: tuple[str, ...] = (
    "không", "chưa", "không có", "phủ định", "phủ nhận", "loại trừ", "âm tính",
    "không thấy", "không ghi nhận",
)
FAMILY_CUES: tuple[str, ...] = (
    "gia đình", "tiền sử gia đình", "mẹ", "cha", "bố", "anh", "chị", "em",
    "ông", "bà",
)
HISTORICAL_CUES: tuple[str, ...] = (
    "tiền sử", "trước đây", "đã từng", "cũ", "năm ngoái", "trước đó",
)
CUES_BY_LABEL: Mapping[str, tuple[str, ...]] = {
    IS_NEGATED: NEGATION_CUES,
    IS_FAMILY: FAMILY_CUES,
    IS_HISTORICAL: HISTORICAL_CUES,
}

# A cue governs a mention only within this many characters before it. Without a
# scope window, "không" at the start of a paragraph would negate the paragraph.
DEFAULT_SCOPE_CHARACTERS = 60


class CueContractError(ValueError):
    """Raised when an assertion decision violates its contract."""


@dataclass(frozen=True, slots=True)
class CueEvidence:
    """One matched cue, its distance from the mention, and the scope decision."""

    label: str
    cue: str
    cue_start: int
    distance: int
    within_scope: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "cue": self.cue, "cue_start": self.cue_start,
            "distance": self.distance, "within_scope": self.within_scope,
        }


def find_cues(
    text: str, *, mention_start: int, scope: int = DEFAULT_SCOPE_CHARACTERS,
) -> tuple[CueEvidence, ...]:
    """Cues preceding a mention, with their distance and scope decision.

    Direction matters: a cue is looked for *before* the mention, because a
    Vietnamese negation or history cue scopes forward over the phrase it
    introduces (spec §8.1). Sorted deterministically so two runs agree exactly.
    """
    lowered = text.lower()
    window_start = max(0, mention_start - scope)
    found: list[CueEvidence] = []
    for label, cues in CUES_BY_LABEL.items():
        for cue in cues:
            position = lowered.rfind(cue, 0, mention_start)
            if position < 0:
                continue
            distance = mention_start - (position + len(cue))
            found.append(CueEvidence(
                label=label, cue=cue, cue_start=position, distance=distance,
                within_scope=position >= window_start))
    return tuple(sorted(found, key=lambda e: (e.label, e.distance, e.cue)))


@dataclass(frozen=True, slots=True)
class CueScopeDecision:
    """The assertions asserted for one mention, and how they were reached."""

    labels: tuple[str, ...]
    source: str
    evidence: tuple[CueEvidence, ...] = ()
    uncertain: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.labels) - ASSERTION_LABELS
        if unknown:
            raise CueContractError(f"unsupported assertion labels: {sorted(unknown)}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CUE_CONTRACT_VERSION,
            "assertions": list(self.labels),
            "source": self.source,
            "uncertain": self.uncertain,
            "evidence": [e.as_dict() for e in self.evidence],
            "empty_means_insufficient_evidence_not_false": True,
        }


def decide_from_cues(
    text: str, *, mention_start: int, scope: int = DEFAULT_SCOPE_CHARACTERS,
) -> CueScopeDecision:
    """Cue-and-scope decision. Reports ``uncertain`` when it cannot decide.

    ``uncertain`` is what routes a mention to the L7 adjudicator (spec §8 stage
    A5); it is not a label, and the labels stay empty until something decides.
    """
    evidence = find_cues(text, mention_start=mention_start, scope=scope)
    in_scope = [e for e in evidence if e.within_scope]
    if not evidence:
        # No cue anywhere: confidently no assertion.
        return CueScopeDecision(labels=(), source="deterministic_no_cue")
    if not in_scope:
        # A cue exists but is out of scope. That is exactly the ambiguous case.
        return CueScopeDecision(
            labels=(), source="deterministic_out_of_scope", evidence=evidence,
            uncertain=True)
    labels = tuple(sorted({e.label for e in in_scope}))
    return CueScopeDecision(
        labels=labels, source="deterministic_cue_in_scope", evidence=tuple(in_scope))


# Spec §12.1's constrained output shape for assertion adjudication.
ADJUDICATION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["assertions"],
    "properties": {
        "assertions": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(ASSERTION_LABELS)},
            "uniqueItems": True,
        }
    },
    "additionalProperties": False,
}


def constrain_adjudication(
    payload: str, *, deterministic: CueScopeDecision,
) -> CueScopeDecision:
    """Constrain an adjudicator response to the three allowed labels.

    The adjudicator may only choose labels: it cannot touch the span or the type,
    and it cannot introduce a label outside the three. Every malformed or
    out-of-vocabulary response falls back to the deterministic decision — named by
    reason in ``source`` — rather than to a guess.
    """
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return CueScopeDecision(
            labels=deterministic.labels, source="adjudicator_rejected_malformed_json",
            evidence=deterministic.evidence, uncertain=True)
    if not isinstance(document, dict) or not isinstance(
            document.get("assertions"), list):
        return CueScopeDecision(
            labels=deterministic.labels, source="adjudicator_rejected_malformed_record",
            evidence=deterministic.evidence, uncertain=True)
    proposed = document["assertions"]
    if any(not isinstance(item, str) for item in proposed):
        return CueScopeDecision(
            labels=deterministic.labels, source="adjudicator_rejected_non_string_label",
            evidence=deterministic.evidence, uncertain=True)
    unknown = set(proposed) - ASSERTION_LABELS
    if unknown:
        return CueScopeDecision(
            labels=deterministic.labels, source="adjudicator_rejected_unsupported_label",
            evidence=deterministic.evidence, uncertain=True)
    return CueScopeDecision(
        labels=tuple(sorted(set(proposed))), source="adjudicated",
        evidence=deterministic.evidence)


def build_adjudication_payload(
    *, mention_text: str, entity_type: str, context: str, route: str,
    deterministic: CueScopeDecision,
) -> dict[str, Any]:
    """Exactly what the adjudicator is shown (spec §12.1).

    The span and the type are **inputs**, not outputs, and the payload carries no
    field with which to change either.
    """
    return {
        "task": "assertion_adjudication",
        "allowed_labels": sorted(ASSERTION_LABELS),
        "mention": mention_text,
        "entity_type": entity_type,
        "context": context,
        "route": route,
        "deterministic_evidence": [e.as_dict() for e in deterministic.evidence],
        "instruction": (
            "Choose zero or more of allowed_labels. Do not modify the mention or "
            "its type. Do not add commentary or a medical conclusion. Return an "
            "empty list when the evidence is insufficient."),
        "may_modify_span": False,
        "may_modify_type": False,
        "may_produce_free_text": False,
    }


__all__ = [
    "ADJUDICATION_SCHEMA",
    "CUES_BY_LABEL",
    "CUE_CONTRACT_VERSION",
    "DEFAULT_SCOPE_CHARACTERS",
    "FAMILY_CUES",
    "HISTORICAL_CUES",
    "IS_FAMILY",
    "IS_HISTORICAL",
    "IS_NEGATED",
    "NEGATION_CUES",
    "SUPPORTED_ASSERTIONS",
    "CueContractError",
    "CueEvidence",
    "CueScopeDecision",
    "build_adjudication_payload",
    "constrain_adjudication",
    "decide_from_cues",
    "find_cues",
]
