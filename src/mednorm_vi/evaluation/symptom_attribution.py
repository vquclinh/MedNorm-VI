"""SYMPTOM error attribution over the real governed internal-test split.

Audit 0033 produced the coarse SYMPTOM taxonomy that falls straight out of the
exact evaluator (missed / left / right / both boundary / wrong type). It could not
produce the finer categories the milestone asked for, because attributing them
needs the L2 route tags and L3 lattice provenance that did not exist yet.

They exist now, so this module completes the taxonomy. Every category is decided
from evidence that is actually available for the example — the route the segment
carried, the section L1 assigned, which experts proposed the span, and the neural
confidence attached to it. When no rule fires the failure is reported as
``unknown_other`` rather than being forced into a plausible-looking bucket.

**Privacy.** Tracked summaries carry counts and the salted-free
``privacy_safe_example_id`` handle only. Clinical text never leaves the git-ignored
``reports/`` tree, and the per-error records this module emits contain offsets,
types and category labels — never a substring of a document.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .exact_mention import (
    BOTH_BOUNDARY,
    LEFT_BOUNDARY,
    MISSED,
    RIGHT_BOUNDARY,
    SPURIOUS,
    WRONG_TYPE,
)

# The categories the milestone requires, in report order.
COMPLETE_MISS = "complete_miss"
BOUNDARY_TOO_SHORT = "boundary_too_short"
BOUNDARY_TOO_LONG = "boundary_too_long"
DIAGNOSIS_SYMPTOM_CONFUSION = "diagnosis_symptom_confusion"
TREATMENT_PURPOSE_PHRASE = "treatment_purpose_phrase"
SECTION_ROUTER_ERROR = "section_router_error"
OVERLAP_COMPETITION = "overlap_competition"
LOW_NEURAL_CONFIDENCE = "low_neural_confidence"
DETERMINISTIC_EVIDENCE_ABSENT = "deterministic_evidence_absent"
UNKNOWN_OTHER = "unknown_other"

ATTRIBUTION_CATEGORIES: tuple[str, ...] = (
    COMPLETE_MISS, BOUNDARY_TOO_SHORT, BOUNDARY_TOO_LONG,
    DIAGNOSIS_SYMPTOM_CONFUSION, TREATMENT_PURPOSE_PHRASE, SECTION_ROUTER_ERROR,
    OVERLAP_COMPETITION, LOW_NEURAL_CONFIDENCE, DETERMINISTIC_EVIDENCE_ABSENT,
    UNKNOWN_OTHER,
)

# Spec §7.2: "Treatment purpose — treated for cough — increases SYMPTOM for
# 'cough'". A gold SYMPTOM sitting immediately after a treatment-purpose cue is a
# distinct failure mode from an ordinary miss, because the cue is the evidence the
# resolver should have used.
TREATMENT_PURPOSE_CUES: tuple[str, ...] = (
    "điều trị", "chữa", "dùng để", "để giảm", "giảm", "chống", "trị",
    "treated for", "for the treatment of", "to treat", "indicated for",
)

# How far before a gold span a treatment-purpose cue may sit and still explain it.
TREATMENT_CUE_WINDOW = 40

# Below this mean token probability the neural expert did propose something but
# was not confident; that is a different problem from proposing nothing at all.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Sections whose prior points AWAY from SYMPTOM. A gold SYMPTOM landing in one of
# them is a section/router attribution problem, not a modelling one.
NON_SYMPTOM_SECTIONS: frozenset[str] = frozenset({
    "laboratory", "pre_admission_medications", "home_medications",
    "current_medications", "diagnosis", "admission_information",
    "discharge_information",
})

_CUE_PATTERN = re.compile(
    "|".join(re.escape(cue) for cue in TREATMENT_PURPOSE_CUES), re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SymptomContext:
    """Everything known about one gold SYMPTOM failure, from L2/L3 evidence."""

    privacy_safe_example_id: str
    gold_span: tuple[int, int]
    evaluator_category: str
    predicted_span: tuple[int, int] | None = None
    predicted_type: str = ""
    route_tags: tuple[str, ...] = field(default_factory=tuple)
    section: str = ""
    proposing_experts: tuple[str, ...] = field(default_factory=tuple)
    neural_confidence: float = 0.0
    overlapping_competitors: int = 0
    treatment_cue_nearby: bool = False
    lattice_covered: bool = False
    # True when at least one overlapping lattice node carried a SYMPTOM score at
    # all — i.e. some expert did consider SYMPTOM here, however weakly.
    symptom_proposed: bool = False


@dataclass(frozen=True, slots=True)
class SymptomAttribution:
    """One attributed SYMPTOM failure. Contains no clinical text."""

    privacy_safe_example_id: str
    category: str
    evaluator_category: str
    gold_span: tuple[int, int]
    predicted_span: tuple[int, int] | None
    route_tags: tuple[str, ...]
    section: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "privacy_safe_example_id": self.privacy_safe_example_id,
            "category": self.category,
            "evaluator_category": self.evaluator_category,
            "gold_span": list(self.gold_span),
            "predicted_span": list(self.predicted_span) if self.predicted_span else None,
            "route_tags": list(self.route_tags),
            "section": self.section,
            "detail": self.detail,
        }


def treatment_cue_before(original_text: str, start: int, *, window: int = TREATMENT_CUE_WINDOW,
                         ) -> bool:
    """Is there a treatment-purpose cue in the ``window`` characters before ``start``?"""
    return bool(_CUE_PATTERN.search(original_text[max(0, start - window):start]))


def attribute_one(context: SymptomContext) -> SymptomAttribution:
    """Assign the most specific category the available evidence supports.

    Rules are ordered from most to least specific. Only evidence that is actually
    present decides a category; anything unexplained lands in ``unknown_other``,
    which is reported rather than hidden.
    """
    category, detail = UNKNOWN_OTHER, "no attribution rule matched"

    if context.evaluator_category == WRONG_TYPE and context.predicted_type == "DIAGNOSIS":
        category = DIAGNOSIS_SYMPTOM_CONFUSION
        detail = "exact span, predicted DIAGNOSIS instead of SYMPTOM"
    elif context.evaluator_category in (LEFT_BOUNDARY, RIGHT_BOUNDARY, BOTH_BOUNDARY):
        gold_length = context.gold_span[1] - context.gold_span[0]
        predicted_length = (
            context.predicted_span[1] - context.predicted_span[0]
            if context.predicted_span else 0)
        if predicted_length < gold_length:
            category = BOUNDARY_TOO_SHORT
            detail = f"predicted {predicted_length} of {gold_length} characters"
        elif predicted_length > gold_length:
            category = BOUNDARY_TOO_LONG
            detail = f"predicted {predicted_length} for a {gold_length}-character gold span"
        else:
            category = BOUNDARY_TOO_LONG
            detail = "same length, shifted span"
        if context.overlapping_competitors > 1:
            category = OVERLAP_COMPETITION
            detail = (f"{context.overlapping_competitors} lattice proposals "
                      "competed for this span")
    elif context.evaluator_category == MISSED:
        if context.section in NON_SYMPTOM_SECTIONS:
            category = SECTION_ROUTER_ERROR
            detail = f"gold SYMPTOM inside section {context.section!r}"
        elif context.treatment_cue_nearby:
            category = TREATMENT_PURPOSE_PHRASE
            detail = "a treatment-purpose cue precedes the gold span"
        elif not context.lattice_covered:
            category = COMPLETE_MISS
            detail = "no expert proposed any span overlapping the gold mention"
        elif context.symptom_proposed and context.neural_confidence < LOW_CONFIDENCE_THRESHOLD:
            category = LOW_NEURAL_CONFIDENCE
            detail = (f"SYMPTOM was proposed here at confidence "
                      f"{context.neural_confidence:.3f}, below threshold")
        elif not context.symptom_proposed:
            category = DETERMINISTIC_EVIDENCE_ABSENT
            detail = ("the region was covered, but no expert proposed SYMPTOM for "
                      "it at all")
        elif context.overlapping_competitors > 1:
            category = OVERLAP_COMPETITION
            detail = (f"{context.overlapping_competitors} lattice proposals "
                      "competed for this region")
        else:
            category = COMPLETE_MISS
            detail = "covered and proposed, but no hypothesis matched the gold span"
    elif context.evaluator_category == WRONG_TYPE:
        category = DIAGNOSIS_SYMPTOM_CONFUSION
        detail = f"exact span typed {context.predicted_type or 'unknown'}"
    elif context.evaluator_category == SPURIOUS:
        if context.treatment_cue_nearby:
            category = TREATMENT_PURPOSE_PHRASE
            detail = "spurious SYMPTOM after a treatment-purpose cue"
        elif context.neural_confidence and context.neural_confidence < LOW_CONFIDENCE_THRESHOLD:
            category = LOW_NEURAL_CONFIDENCE
            detail = (f"spurious SYMPTOM at confidence "
                      f"{context.neural_confidence:.3f}, below threshold")
        elif context.overlapping_competitors > 1:
            category = OVERLAP_COMPETITION
            detail = "spurious SYMPTOM competing with other proposals for the region"
        else:
            category = UNKNOWN_OTHER
            detail = "confident spurious SYMPTOM prediction with no gold counterpart"

    return SymptomAttribution(
        privacy_safe_example_id=context.privacy_safe_example_id,
        category=category, evaluator_category=context.evaluator_category,
        gold_span=context.gold_span, predicted_span=context.predicted_span,
        route_tags=context.route_tags, section=context.section, detail=detail)


def attribute_all(contexts: Sequence[SymptomContext]) -> tuple[SymptomAttribution, ...]:
    return tuple(attribute_one(context) for context in contexts)


def summarize(attributions: Sequence[SymptomAttribution]) -> dict[str, Any]:
    """Counts per category plus route/section rollups. Privacy-safe by construction."""
    counts = dict.fromkeys(ATTRIBUTION_CATEGORIES, 0)
    by_route: dict[str, int] = {}
    by_section: dict[str, int] = {}
    for attribution in attributions:
        counts[attribution.category] = counts.get(attribution.category, 0) + 1
        route = "+".join(attribution.route_tags) if attribution.route_tags else "unrouted"
        by_route[route] = by_route.get(route, 0) + 1
        section = attribution.section or "unsectioned"
        by_section[section] = by_section.get(section, 0) + 1
    total = len(attributions)
    return {
        "total": total,
        "categories": dict(counts),
        "category_share": {
            name: round(count / total, 4) if total else 0.0
            for name, count in counts.items()
        },
        "by_route": dict(sorted(by_route.items())),
        "by_section": dict(sorted(by_section.items())),
        "unattributed": counts.get(UNKNOWN_OTHER, 0),
        "structural_note": (
            "E1 (medication grammar) and E2 (laboratory parser) cannot emit "
            "SYMPTOM by construction, so deterministic corroboration is "
            "structurally unavailable for this type. 'deterministic_evidence_"
            "absent' therefore reports where NO expert proposed SYMPTOM at all, "
            "not a deficiency of a specific expert."),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Compact tracked summary. Counts only — never clinical text."""
    lines = [
        "# SYMPTOM error attribution — governed internal_test (real data)",
        "",
        f"Total attributed SYMPTOM failures: **{summary['total']}**",
        "",
        "| category | count | share |",
        "| --- | --- | --- |",
    ]
    for name in ATTRIBUTION_CATEGORIES:
        count = summary["categories"].get(name, 0)
        share = summary["category_share"].get(name, 0.0)
        lines.append(f"| {name} | {count} | {share:.1%} |")
    return "\n".join(lines) + "\n"


__all__ = [
    "ATTRIBUTION_CATEGORIES",
    "BOUNDARY_TOO_LONG",
    "BOUNDARY_TOO_SHORT",
    "COMPLETE_MISS",
    "DETERMINISTIC_EVIDENCE_ABSENT",
    "DIAGNOSIS_SYMPTOM_CONFUSION",
    "LOW_CONFIDENCE_THRESHOLD",
    "LOW_NEURAL_CONFIDENCE",
    "NON_SYMPTOM_SECTIONS",
    "OVERLAP_COMPETITION",
    "SECTION_ROUTER_ERROR",
    "TREATMENT_PURPOSE_CUES",
    "TREATMENT_PURPOSE_PHRASE",
    "UNKNOWN_OTHER",
    "SymptomAttribution",
    "SymptomContext",
    "attribute_all",
    "attribute_one",
    "render_markdown",
    "summarize",
    "treatment_cue_before",
]
