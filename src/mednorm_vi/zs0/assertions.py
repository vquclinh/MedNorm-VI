"""ZS0 zero-shot assertions (Audit 0048).

Deterministic cue-and-scope logic first; Qwen only where that logic is uncertain.

The governed corpus has **zero** assertion supervision (Audit 0042), so nothing
here is learned and nothing is evaluated against gold. Insufficient evidence
yields an **empty** assertion set — never a guessed `false`, which would be
thousands of confident negatives invented out of nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..schemas.constants import ASSERTION_LABELS

ASSERTION_CONTRACT_VERSION = "zs0-assertion-v1"

IS_NEGATED = "isNegated"
IS_FAMILY = "isFamily"
IS_HISTORICAL = "isHistorical"
SUPPORTED_ASSERTIONS: tuple[str, ...] = (IS_NEGATED, IS_FAMILY, IS_HISTORICAL)

# Vietnamese cue lexicons. Deliberately small and high-precision: a cue that
# fires often but wrongly is worse than no cue, because the fallback (empty) is
# already the safe answer.
NEGATION_CUES: tuple[str, ...] = (
    "không", "chưa", "không có", "phủ định", "loại trừ", "âm tính", "không thấy",
    "không ghi nhận",
)
FAMILY_CUES: tuple[str, ...] = (
    "gia đình", "mẹ", "cha", "bố", "anh", "chị", "em", "ông", "bà",
    "tiền sử gia đình",
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
# scope window "không" at the start of a paragraph would negate the paragraph.
DEFAULT_SCOPE_CHARACTERS = 60


class AssertionError_(ValueError):
    """Raised when an assertion decision violates its contract."""


@dataclass(frozen=True, slots=True)
class CueEvidence:
    """What the deterministic pass found, and how confident it is."""

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
    """Cues preceding a mention, with their distance and scope decision."""
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
class AssertionDecision:
    """The assertions asserted for one mention, and how they were reached."""

    labels: tuple[str, ...]
    source: str
    evidence: tuple[CueEvidence, ...] = ()
    uncertain: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.labels) - ASSERTION_LABELS
        if unknown:
            raise AssertionError_(f"unsupported assertion labels: {sorted(unknown)}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "assertions": list(self.labels),
            "source": self.source,
            "uncertain": self.uncertain,
            "evidence": [e.as_dict() for e in self.evidence],
            "empty_means_insufficient_evidence_not_false": True,
        }


def decide_deterministically(
    text: str, *, mention_start: int, scope: int = DEFAULT_SCOPE_CHARACTERS,
) -> AssertionDecision:
    """Cue-and-scope decision. Returns ``uncertain`` when it cannot decide."""
    evidence = find_cues(text, mention_start=mention_start, scope=scope)
    in_scope = [e for e in evidence if e.within_scope]
    if not evidence:
        # No cue anywhere: confidently no assertion.
        return AssertionDecision(labels=(), source="deterministic_no_cue")
    if not in_scope:
        # A cue exists but is out of scope. That is exactly the ambiguous case.
        return AssertionDecision(
            labels=(), source="deterministic_out_of_scope", evidence=evidence,
            uncertain=True)
    labels = tuple(sorted({e.label for e in in_scope}))
    return AssertionDecision(
        labels=labels, source="deterministic_cue_in_scope", evidence=tuple(in_scope))


QWEN_ASSERTION_SCHEMA: Mapping[str, Any] = {
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


def parse_qwen_assertions(
    payload: str, *, deterministic: AssertionDecision,
) -> AssertionDecision:
    """Constrain a Qwen assertion response to the allowed label set.

    Qwen may only choose labels; it cannot touch the span or the type, and it
    cannot introduce a label outside the three. Anything malformed falls back to
    the deterministic decision rather than to a guess.
    """
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return AssertionDecision(
            labels=deterministic.labels, source="qwen_rejected_malformed_json",
            evidence=deterministic.evidence, uncertain=True)
    if not isinstance(document, dict) or not isinstance(
            document.get("assertions"), list):
        return AssertionDecision(
            labels=deterministic.labels, source="qwen_rejected_malformed_record",
            evidence=deterministic.evidence, uncertain=True)
    proposed = document["assertions"]
    if any(not isinstance(item, str) for item in proposed):
        return AssertionDecision(
            labels=deterministic.labels, source="qwen_rejected_non_string_label",
            evidence=deterministic.evidence, uncertain=True)
    unknown = set(proposed) - ASSERTION_LABELS
    if unknown:
        return AssertionDecision(
            labels=deterministic.labels, source="qwen_rejected_unsupported_label",
            evidence=deterministic.evidence, uncertain=True)
    return AssertionDecision(
        labels=tuple(sorted(set(proposed))), source="qwen_adjudicated",
        evidence=deterministic.evidence)


def build_qwen_prompt_payload(
    *, mention_text: str, entity_type: str, context: str, route: str,
    deterministic: AssertionDecision,
) -> dict[str, Any]:
    """Exactly what Qwen is shown. The span and type are inputs, not outputs."""
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
    "ASSERTION_CONTRACT_VERSION",
    "CUES_BY_LABEL",
    "DEFAULT_SCOPE_CHARACTERS",
    "FAMILY_CUES",
    "HISTORICAL_CUES",
    "IS_FAMILY",
    "IS_HISTORICAL",
    "IS_NEGATED",
    "NEGATION_CUES",
    "QWEN_ASSERTION_SCHEMA",
    "SUPPORTED_ASSERTIONS",
    "AssertionDecision",
    "AssertionError_",
    "CueEvidence",
    "build_qwen_prompt_payload",
    "decide_deterministically",
    "find_cues",
    "parse_qwen_assertions",
]
