"""Constrained LLM disambiguation and deterministic precision gates (GraphCENT 0080).

The public leaderboard says emitting nothing scores better than emitting the linker's best
guess (all-null 14.3749 against 12.4757). So selection is not the last word here: the model
chooses among governed ids or abstains, and a deterministic tier gate then decides whether the
selection is backed by enough independent evidence to be worth emitting at all.

The parser is adversarial by construction. Any id outside the supplied set, any unknown
schema, any prose, any parse failure resolves to NULL - never to a guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .ontology import IcdFacts, RxNormFacts
from .retrieval import CandidateEvidence

DECISION_NULL = "NULL"
DECISION_SELECT = "SELECT"

#: Closed enum. Diagnostic only - reason codes never influence emission.
REASON_CODES: frozenset[str] = frozenset(
    {
        "exact_label_match",
        "alias_match",
        "same_concept_different_wording",
        "ingredient_and_strength_agree",
        "context_supports",
        "multiple_codes_required",
    }
)

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"
TIER_NONE = "NONE"

_JSON = re.compile(r"\{.*\}", re.S)

RUBRIC = (
    "You are linking a Vietnamese clinical mention to a governed medical code.\n\n"
    "DEFAULT ANSWER IS NULL.\n"
    "* Answer NULL unless a listed candidate denotes the SAME medical concept as the "
    "mention.\n"
    "* Similar wording is NOT enough. Sharing a word, a stem or a spelling is the most "
    "common way to be wrong.\n"
    "* Do NOT select a broader, narrower, parent, sibling or otherwise related concept. "
    "Only an exact conceptual match counts.\n"
    "* For medications: if the ingredient differs, answer NULL. If the mention states a "
    "strength, concentration or form and the candidate records a different one, answer "
    "NULL.\n"
    "* Select several ids only when the mention genuinely requires several governed codes. "
    "Otherwise prefer one, or NULL.\n"
    "* You may ONLY return ids from the numbered list. You cannot write a code."
)


def build_prompt(
    mention: str,
    entity_type: str,
    context: str,
    candidates: list[tuple[str, list[str]]],
    *,
    section: str = "",
) -> str:
    """Curated small context, CENT-style. The model never sees the KB or raw scores."""
    listed = "\n".join(
        f"{index}. " + "\n   ".join(lines) for index, (_, lines) in enumerate(candidates)
    )
    ontology = "ICD-10" if entity_type == "CHẨN_ĐOÁN" else "RxNorm"
    heading = f"SECTION: {section}\n" if section else ""
    return (
        f"{RUBRIC}\n\n"
        f"ONTOLOGY: {ontology}\nENTITY TYPE: {entity_type}\n"
        f"MENTION: {mention}\n{heading}CONTEXT: {context}\n\n"
        f"CANDIDATES:\n{listed}\n\n"
        "Return JSON only. Either:\n"
        '  {"decision":"NULL"}\n'
        "or:\n"
        '  {"decision":"SELECT","candidate_ids":["<id from the list>"]}\n'
    )


@dataclass(frozen=True, slots=True)
class Decision:
    decision: str
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    parse_failed: bool = False
    invalid_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def selected(self) -> bool:
        return self.decision == DECISION_SELECT and bool(self.candidate_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "candidate_ids": list(self.candidate_ids),
            "reason_codes": list(self.reason_codes),
            "parse_failed": self.parse_failed,
            "invalid_ids": list(self.invalid_ids),
        }


NULL_DECISION = Decision(DECISION_NULL)


def parse_decision(reply: str, offered: list[str]) -> Decision:
    """Strict parse. Anything unrecognised becomes NULL, never a guess."""
    match = _JSON.search(reply or "")
    if not match:
        return Decision(DECISION_NULL, parse_failed=True)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Decision(DECISION_NULL, parse_failed=True)
    if not isinstance(payload, dict):
        return Decision(DECISION_NULL, parse_failed=True)

    decision = str(payload.get("decision", "")).upper()
    if decision == DECISION_NULL:
        return Decision(DECISION_NULL)
    if decision != DECISION_SELECT:
        return Decision(DECISION_NULL, parse_failed=True)

    allowed = set(offered)
    raw_ids = payload.get("candidate_ids")
    if not isinstance(raw_ids, list):
        return Decision(DECISION_NULL, parse_failed=True)
    kept: list[str] = []
    invalid: list[str] = []
    for value in raw_ids:
        text = str(value).strip()
        if text in allowed and text not in kept:
            kept.append(text)
        elif text:
            invalid.append(text)  # an id the model invented; dropped, never emitted
    if not kept:
        return Decision(DECISION_NULL, invalid_ids=tuple(invalid))
    reasons = payload.get("reason_codes")
    codes = (
        tuple(str(r) for r in reasons if isinstance(reasons, list) and str(r) in REASON_CODES)
        if isinstance(reasons, list)
        else ()
    )
    return Decision(DECISION_SELECT, tuple(kept), codes, invalid_ids=tuple(invalid))


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """Which evidence classes may emit. Explicit and configurable, no hidden thresholds."""

    allow_tier_a: bool = True
    allow_tier_b: bool = True
    allow_tier_c: bool = True
    min_retrievers_for_tier_b: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow_tier_a": self.allow_tier_a,
            "allow_tier_b": self.allow_tier_b,
            "allow_tier_c": self.allow_tier_c,
            "min_retrievers_for_tier_b": self.min_retrievers_for_tier_b,
        }


def classify(
    evidence: CandidateEvidence,
    facts: IcdFacts | RxNormFacts | None,
    mention_has_structure: bool,
    policy: TierPolicy,
) -> str:
    """The evidence tier of one Qwen-selected candidate. A contradiction always demotes."""
    conflicted = isinstance(facts, RxNormFacts) and facts.has_conflict
    if conflicted:
        return TIER_NONE
    if evidence.exact_alias_match:
        return TIER_A
    if evidence.supporting_retrievers >= policy.min_retrievers_for_tier_b:
        return TIER_B
    if (
        isinstance(facts, RxNormFacts)
        and mention_has_structure
        and facts.ingredients
        and not facts.has_conflict
    ):
        return TIER_C
    return TIER_NONE


def emit_for_variant(tier: str, variant: str) -> bool:
    """Which tiers a named output variant emits. One run, four ZIPs."""
    if variant == "allnull":
        return False
    if variant == "tierA":
        return tier == TIER_A
    if variant == "tierAB":
        return tier in (TIER_A, TIER_B)
    if variant == "tierABC":
        return tier in (TIER_A, TIER_B, TIER_C)
    return False


VARIANTS: tuple[str, ...] = ("allnull", "tierA", "tierAB", "tierABC")


__all__ = [
    "DECISION_NULL",
    "DECISION_SELECT",
    "NULL_DECISION",
    "REASON_CODES",
    "RUBRIC",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "TIER_NONE",
    "VARIANTS",
    "Decision",
    "TierPolicy",
    "build_prompt",
    "classify",
    "emit_for_variant",
    "parse_decision",
]
