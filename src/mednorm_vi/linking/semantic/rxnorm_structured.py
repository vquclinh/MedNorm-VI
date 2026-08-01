"""Structured RxNorm compatibility evidence for the S1 linker (Audit 0074 §4).

The S1 architecture is unchanged - lexical Top-20 + dense Top-50 -> governed union ->
Qwen3-Reranker-4B. This module adds *evidence* to that pipeline on the RxNorm side only:
richer document text for retrieval and reranking, and a deterministic compatibility signal
that a candidate's recovered structure either agrees with the mention or contradicts it.

The asymmetry is the point. A mention that states `500 mg` is **contradicted** by a candidate
recorded as `250 mg`, and that is much stronger information than the absence of a match: it
means the candidate is wrong, not merely unsupported. A mention that states no strength is
contradicted by nothing, so a candidate carrying a strength is only unsupported and keeps its
place. Nothing here invents a strength, a form or a brand that the source does not record.

ICD is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...kb.rxnorm.structured import StructuredDrug, normalize_unit, parse_strengths

COMPATIBILITY_VERSION = "rxnorm-structured-compatibility-v1"

#: Deterministic verdicts, ordered best to worst. Ordering is lexicographic, never additive,
#: so no quantity of lexical similarity can promote a contradicted candidate.
MATCH_EXACT = "exact"
MATCH_SUPPORTED = "supported"
MATCH_UNSUPPORTED = "unsupported"
MATCH_CONTRADICTED = "contradicted"
VERDICT_ORDER: tuple[str, ...] = (
    MATCH_EXACT,
    MATCH_SUPPORTED,
    MATCH_UNSUPPORTED,
    MATCH_CONTRADICTED,
)

#: Vietnamese and English dose-form cues that appear in clinical mentions, mapped to the
#: RxNorm form vocabulary. Only forms the governed source actually uses are listed.
FORM_CUES: tuple[tuple[str, str], ...] = (
    ("viên nén", "tablet"),
    ("viên nang", "capsule"),
    ("viên", "tablet"),
    ("nang", "capsule"),
    ("ống", "injection"),
    ("tiêm", "injection"),
    ("truyền", "injection"),
    ("dịch truyền", "injection"),
    ("siro", "syrup"),
    ("hỗn dịch", "suspension"),
    ("dung dịch", "solution"),
    ("kem", "cream"),
    ("mỡ", "ointment"),
    ("gel", "gel"),
    ("xịt", "spray"),
    ("nhỏ mắt", "ophthalmic"),
    ("đặt", "suppository"),
    ("bột", "powder"),
    ("miếng dán", "patch"),
    ("tablet", "tablet"),
    ("capsule", "capsule"),
    ("injection", "injection"),
    ("syrup", "syrup"),
    ("cream", "cream"),
    ("solution", "solution"),
)


@dataclass(frozen=True, slots=True)
class MentionStructure:
    """What the clinical mention itself states. Absent means absent, never assumed."""

    strength_keys: frozenset[str] = frozenset()
    form_cues: frozenset[str] = frozenset()
    raw: str = ""

    @property
    def states_strength(self) -> bool:
        return bool(self.strength_keys)

    @property
    def states_form(self) -> bool:
        return bool(self.form_cues)


def parse_mention(mention: str) -> MentionStructure:
    """Extract only what the text says: strengths it writes and dose-form words it uses."""
    lowered = (mention or "").casefold()
    strengths = frozenset(s.key for s in parse_strengths(mention))
    forms = frozenset(form for cue, form in FORM_CUES if cue in lowered)
    return MentionStructure(strength_keys=strengths, form_cues=forms, raw=mention or "")


@dataclass(frozen=True, slots=True)
class Compatibility:
    """Per-facet verdicts plus the worst one, which is what ranking uses."""

    strength: str = MATCH_UNSUPPORTED
    form: str = MATCH_UNSUPPORTED
    ingredient: str = MATCH_UNSUPPORTED
    worst: str = MATCH_UNSUPPORTED
    reasons: tuple[str, ...] = ()

    @property
    def rank(self) -> int:
        return VERDICT_ORDER.index(self.worst)

    @property
    def contradicted(self) -> bool:
        return self.worst == MATCH_CONTRADICTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "form": self.form,
            "ingredient": self.ingredient,
            "worst": self.worst,
            "reasons": list(self.reasons),
            "compatibility_version": COMPATIBILITY_VERSION,
        }


def assess(mention: MentionStructure, drug: StructuredDrug | None) -> Compatibility:
    """Compare what the mention states against what the source records for the candidate."""
    if drug is None:
        return Compatibility(reasons=("no_structured_record",))

    reasons: list[str] = []

    if not mention.states_strength:
        strength = MATCH_UNSUPPORTED
    elif not drug.strength_keys:
        # The mention gives a strength the candidate has no record of. Unsupported, not
        # contradicted: silence in the source is not disagreement.
        strength = MATCH_UNSUPPORTED
        reasons.append("mention_strength_not_recorded_for_candidate")
    elif mention.strength_keys & drug.strength_keys:
        strength = MATCH_EXACT
        reasons.append("strength_exact")
    else:
        strength = MATCH_CONTRADICTED
        reasons.append("strength_contradicted")

    candidate_forms = {normalize_unit(f) for f in (*drug.dose_forms, drug.rxterm_form) if f}
    if not mention.states_form:
        form = MATCH_UNSUPPORTED
    elif not candidate_forms:
        form = MATCH_UNSUPPORTED
        reasons.append("mention_form_not_recorded_for_candidate")
    elif any(cue in joined for cue in mention.form_cues for joined in candidate_forms):
        form = MATCH_SUPPORTED
        reasons.append("form_supported")
    else:
        form = MATCH_CONTRADICTED
        reasons.append("form_contradicted")

    lowered = mention.raw.casefold()
    if drug.ingredients and any(i.casefold() in lowered for i in drug.ingredients):
        ingredient = MATCH_SUPPORTED
        reasons.append("ingredient_present_in_mention")
    else:
        ingredient = MATCH_UNSUPPORTED

    verdicts = (strength, form, ingredient)
    worst = max(verdicts, key=VERDICT_ORDER.index)
    return Compatibility(
        strength=strength,
        form=form,
        ingredient=ingredient,
        worst=worst,
        reasons=tuple(reasons),
    )


def rank_key(
    concept_id: str, compatibility: Compatibility, base_rank: int
) -> tuple[int, int, int, str]:
    """Lexicographic key: contradiction demotes, exact promotes, base order breaks ties.

    Ordering, not arithmetic: a contradicted candidate cannot climb back by accumulating
    lexical score, and an exact-strength candidate cannot be displaced by a merely similar
    name.
    """
    exactness = 0 if compatibility.strength == MATCH_EXACT else 1
    return (compatibility.rank, exactness, base_rank, concept_id)


def reorder(
    ranked: list[str],
    mention: str,
    structured: dict[str, StructuredDrug],
) -> list[str]:
    """Reorder one candidate list by structured compatibility. Set is preserved exactly."""
    parsed = parse_mention(mention)
    scored = [
        (rank_key(code, assess(parsed, structured.get(code)), position), code)
        for position, code in enumerate(ranked)
    ]
    scored.sort()
    return [code for _, code in scored]


__all__ = [
    "COMPATIBILITY_VERSION",
    "FORM_CUES",
    "MATCH_CONTRADICTED",
    "MATCH_EXACT",
    "MATCH_SUPPORTED",
    "MATCH_UNSUPPORTED",
    "VERDICT_ORDER",
    "Compatibility",
    "MentionStructure",
    "assess",
    "parse_mention",
    "rank_key",
    "reorder",
]
