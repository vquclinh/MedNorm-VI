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
import re
import unicodedata
from collections.abc import Mapping, Sequence
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

# ---------------------------------------------------------------------------
# Spec §8.1 scope boundaries
# ---------------------------------------------------------------------------
#
# A character-distance window is not a scope model. Spec §8.1 is explicit: a
# Vietnamese negation cue "usually stops at contrast markers such as 'but' or
# 'however'", and a cue's reach ends at the clause or sentence it belongs to. A
# window that ignores those boundaries assigns every label in a paragraph to every
# entity in it — which is exactly the defect Audit 0051 measured.
#
# Sentence terminators end any cue's scope unconditionally.
SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?", "\n", ";")

# Clause separators end a cue's scope for the *following* clause. A colon is
# included because "Tiền sử gia đình: ..." introduces a new region, and a comma
# because coordinated clauses are where over-assignment does the most damage.
CLAUSE_SEPARATORS: tuple[str, ...] = (",", ":", "–", "—", "|")

# Contrast markers stop a negation outright (spec §8.1). "không A nhưng B" negates
# A, never B; treating both as negated is the single most damaging false positive
# available, because assertions are scored by Jaccard against a frequently empty
# gold set (spec §13.3).
CONTRAST_MARKERS: tuple[str, ...] = (
    "nhưng", "tuy nhiên", "mà", "song", "ngược lại", "trái lại", "còn",
)

# Which entity types may carry an assertion at all. Straight from
# ``ORGANIZER_FIELDS_BY_TYPE``: a TEST_NAME or TEST_RESULT has no ``assertions``
# field in organizer output, so computing one is wasted work that also hides
# over-firing — which is precisely how the Audit-0051 defect stayed invisible.
ASSERTION_ELIGIBLE_TYPES: frozenset[str] = frozenset(
    {"MEDICATION", "DIAGNOSIS", "SYMPTOM"})

# Section headers that establish a *document-level* prior (spec §4.2). A prior is
# evidence, not a rule: a local sentence such as "bắt đầu hôm nay" can override it,
# so a prior alone never labels an entity outside its own section.
SECTION_PRIOR_CUES: Mapping[str, tuple[str, ...]] = {
    IS_HISTORICAL: (
        "tiền sử", "tiền căn", "bệnh sử", "thuốc tại nhà", "thuốc đang dùng",
        "trước khi nhập viện", "thuốc trước nhập viện",
    ),
    IS_FAMILY: ("tiền sử gia đình", "gia đình"),
}


def scope_start(text: str, mention_start: int, *, scope: int) -> int:
    """Where a cue's reach begins for a mention at ``mention_start``.

    The window is clipped at the nearest preceding **sentence terminator** and then
    at the nearest preceding **clause separator**, so a cue in a previous sentence
    or a previous clause cannot reach this mention. This is the §8.1 boundary the
    character window alone was missing.
    """
    window_start = max(0, mention_start - scope)
    region = text[window_start:mention_start]
    cut = window_start
    for terminator in SENTENCE_TERMINATORS:
        position = region.rfind(terminator)
        if position >= 0:
            cut = max(cut, window_start + position + len(terminator))
    # Re-scan only the surviving sentence fragment for a clause separator.
    fragment = text[cut:mention_start]
    for separator in CLAUSE_SEPARATORS:
        position = fragment.rfind(separator)
        if position >= 0:
            cut = max(cut, cut + position + len(separator))
    return cut


def _is_word_character(character: str) -> bool:
    """Whether a character can be part of a Vietnamese word.

    Diacritics are combining marks, so a naive ``str.isalpha`` on a decomposed
    string would treat a mark as a boundary. Category ``Mn`` is included for that
    reason.
    """
    return character.isalpha() or unicodedata.category(character) == "Mn"


def find_cue_occurrences(haystack: str, needle: str) -> tuple[int, ...]:
    """Word-boundary-respecting occurrences of ``needle`` in ``haystack``.

    Bare substring matching is not safe for these lexicons: the family cue "ông"
    (grandfather) occurs inside "kh**ông**" (not), so ``"ông" in text`` labels every
    negated sentence as family context. Audit 0052 measured exactly that. Vietnamese
    words are space-separated, so a match must be bounded by non-word characters on
    both sides.
    """
    if not needle:
        return ()
    out: list[int] = []
    for match in re.finditer(re.escape(needle), haystack):
        start, end = match.start(), match.end()
        before = haystack[start - 1] if start > 0 else " "
        after = haystack[end] if end < len(haystack) else " "
        if _is_word_character(before) or _is_word_character(after):
            continue
        out.append(start)
    return tuple(out)


def contrast_between(text: str, cue_end: int, mention_start: int) -> str:
    """The contrast marker separating a cue from a mention, if any (spec §8.1)."""
    if cue_end >= mention_start:
        return ""
    between = text[cue_end:mention_start].lower()
    for marker in CONTRAST_MARKERS:
        if find_cue_occurrences(between, marker):
            return marker
    return ""


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
    blocked_by: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "cue": self.cue, "cue_start": self.cue_start,
            "distance": self.distance, "within_scope": self.within_scope,
            "blocked_by": self.blocked_by,
        }


def find_cues(
    text: str, *, mention_start: int, scope: int = DEFAULT_SCOPE_CHARACTERS,
) -> tuple[CueEvidence, ...]:
    """Cues that could govern a mention, with the §8.1 scope decision for each.

    Three constraints, all from spec §8.1, and all necessary — dropping any one of
    them reproduces the over-assignment Audit 0051 measured:

    1. **direction** — only cues *preceding* the mention are considered, because a
       Vietnamese negation or history cue scopes forward over what it introduces;
    2. **clause and sentence boundaries** — the window is clipped at the nearest
       preceding sentence terminator and clause separator, so a cue belonging to
       another clause cannot reach this mention;
    3. **contrast markers** — a cue separated from the mention by "nhưng", "tuy
       nhiên" and similar is recorded but marked out of scope, because "không A
       nhưng B" negates A and not B.

    Every considered cue is returned, in scope or not, so the reason a label was
    *withheld* is as inspectable as the reason one was applied. Sorted
    deterministically so two runs agree exactly.
    """
    lowered = text.lower()
    boundary = scope_start(text, mention_start, scope=scope)
    found: list[CueEvidence] = []
    for label, cues in CUES_BY_LABEL.items():
        for cue in cues:
            occurrences = find_cue_occurrences(lowered[:mention_start], cue)
            if not occurrences:
                continue
            position = occurrences[-1]
            cue_end = position + len(cue)
            distance = mention_start - cue_end
            blocked = ""
            if position < boundary:
                blocked = "clause_or_sentence_boundary"
            else:
                marker = contrast_between(text, cue_end, mention_start)
                if marker:
                    blocked = f"contrast:{marker}"
            found.append(CueEvidence(
                label=label, cue=cue, cue_start=position, distance=distance,
                within_scope=not blocked, blocked_by=blocked))
    return tuple(sorted(found, key=lambda e: (e.label, e.distance, e.cue)))


def nearest_in_scope(
    evidence: Sequence[CueEvidence],
) -> dict[str, CueEvidence]:
    """The closest in-scope cue per label.

    Spec §8.1's "nearest eligible entity" read from the cue's side: when several
    cues of one family are in scope, the nearest one is the governing evidence and
    the rest are corroboration, not extra weight.
    """
    best: dict[str, CueEvidence] = {}
    for item in evidence:
        if not item.within_scope:
            continue
        current = best.get(item.label)
        if current is None or item.distance < current.distance:
            best[item.label] = item
    return best


def section_priors(text: str, mention_start: int) -> tuple[str, ...]:
    """Labels the enclosing section makes *more likely* (spec §4.2).

    A section prior is evidence, never a rule, and it is deliberately narrow:

    * it applies when the mention sits **on the header line itself**
      ("Tiền sử: đái tháo đường"), or
    * when the header line **ends with a colon**, which is what makes it a header
      introducing following content ("Thuốc tại nhà:" then a numbered list).

    A line such as "Tiền sử gia đình: bố tăng huyết áp." does **not** carry its prior
    onto the next sentence, because that sentence introduces a new subject
    ("Bệnh nhân ..."). Letting it would repeat, one level up, exactly the
    indiscriminate labelling spec §4.3 forbids.
    """
    paragraph_start = text.rfind("\n\n", 0, mention_start)
    region_start = paragraph_start + 2 if paragraph_start >= 0 else 0
    region = text[region_start:mention_start]
    if not region:
        return ()
    lines = region.split("\n")
    header_line = lines[0]
    mention_on_header_line = len(lines) == 1
    # A colon-terminated header governs the lines beneath it; anything else governs
    # only its own line.
    header_governs_following = header_line.rstrip().endswith(":")
    if not (mention_on_header_line or header_governs_following):
        return ()
    if mention_on_header_line:
        # ...and only up to the end of the header's own sentence. A line may hold a
        # header plus a following sentence ("Tiền sử: X. Hiện tại Y"), and the prior
        # governs X, not Y — the same §8.1 boundary that applies to cues.
        if any(
            terminator in header_line
            for terminator in SENTENCE_TERMINATORS
            if terminator != "\n"
        ):
            first_cut = min(
                (header_line.find(terminator) for terminator in SENTENCE_TERMINATORS
                 if terminator != "\n" and terminator in header_line),
                default=-1)
            if first_cut >= 0 and first_cut < len(header_line) - 1:
                return ()
    header = header_line.lower()
    out: list[str] = []
    for label, headers in SECTION_PRIOR_CUES.items():
        if any(find_cue_occurrences(header, marker) for marker in headers):
            out.append(label)
    return tuple(sorted(out))


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
    text: str,
    *,
    mention_start: int,
    scope: int = DEFAULT_SCOPE_CHARACTERS,
    entity_type: str = "",
    use_section_priors: bool = True,
) -> CueScopeDecision:
    """The deterministic A2/A3 decision for one mention (spec §8, §8.1).

    ``entity_type`` is optional but recommended: a type that carries no
    ``assertions`` field in organizer output is refused outright, which keeps the
    decision honest instead of computing labels that are silently discarded later.

    ``uncertain`` is what routes a mention to the L7 adjudicator (spec §8 stage A5);
    it is not a label, and the labels stay empty until something decides.
    """
    if entity_type and entity_type not in ASSERTION_ELIGIBLE_TYPES:
        return CueScopeDecision(
            labels=(), source=f"type_not_assertion_eligible:{entity_type}")

    evidence = find_cues(text, mention_start=mention_start, scope=scope)
    governing = nearest_in_scope(evidence)
    priors = section_priors(text, mention_start) if use_section_priors else ()

    if not evidence and not priors:
        # No cue anywhere and no section prior: confidently no assertion.
        return CueScopeDecision(labels=(), source="deterministic_no_cue")

    # A1 (section prior) and A2/A3 (cue + scope) are separate contributing stages in
    # spec §8, not alternatives. A colon-headed section can establish isHistorical
    # while a local cue independently establishes isFamily, and reporting only one
    # of them would discard real evidence.
    cue_labels = set(governing)
    prior_labels = set(priors)
    labels = tuple(sorted(cue_labels | prior_labels))

    if not labels:
        # Cues exist but every one is blocked by a clause boundary or a contrast
        # marker, and no section prior applies. That is the ambiguous case, and it
        # yields NO labels — an empty set is the cheap error (spec §13.3).
        return CueScopeDecision(
            labels=(), source="deterministic_out_of_scope", evidence=evidence,
            uncertain=True)

    if cue_labels and prior_labels:
        source = "cue_in_scope_and_section_prior"
    elif cue_labels:
        source = "deterministic_cue_in_scope"
    else:
        source = "section_prior_only"

    # A label resting only on a section prior is weaker than one resting on a local
    # cue: spec §4.2 says a local sentence ("bắt đầu hôm nay") may override the
    # prior, and nothing at this stage has read that sentence. Flag it so L7 can
    # adjudicate rather than pretending the prior settled it.
    uncertain = bool(prior_labels - cue_labels)
    return CueScopeDecision(
        labels=labels, source=source,
        evidence=tuple(governing.values()) or evidence, uncertain=uncertain)


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
    "ASSERTION_ELIGIBLE_TYPES",
    "CLAUSE_SEPARATORS",
    "CONTRAST_MARKERS",
    "SECTION_PRIOR_CUES",
    "SENTENCE_TERMINATORS",
    "contrast_between",
    "find_cue_occurrences",
    "nearest_in_scope",
    "scope_start",
    "section_priors",
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
