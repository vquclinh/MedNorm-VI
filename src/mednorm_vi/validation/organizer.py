"""Deterministic strict validator for reasoner output (sprint 0075).

The model is a proposer with no authority. Everything it returns is re-derived here from the
source text and the governed KB, and anything that cannot be proved is dropped:

* a span survives only if its text is an **exact substring** of the original note;
* offsets are **computed here**, never taken from the model;
* the type must be one of the five organizer types;
* a candidate must be a member of the governed pool that was offered for that mention;
* duplicate surface forms are resolved deterministically by occurrence order.

Rejections are counted by reason so a bad prompt shows up as a number rather than as silent
quality loss.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

#: The only entity types the organizer accepts.
ORGANIZER_TYPES: tuple[str, ...] = (
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
)
#: Types that carry assertions.
ASSERTION_TYPES: frozenset[str] = frozenset({"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"})
ASSERTION_KEYS: tuple[str, ...] = ("isNegated", "isFamily", "isHistorical")
#: Types that may carry ontology candidates.
CANDIDATE_TYPES: frozenset[str] = frozenset({"CHẨN_ĐOÁN", "THUỐC"})

REJECT_UNKNOWN_TYPE = "unknown_type"
REJECT_SPAN_NOT_FOUND = "span_not_in_source"
REJECT_EMPTY_TEXT = "empty_text"
REJECT_OCCURRENCE_EXHAUSTED = "occurrence_exhausted"
REJECT_UNGOVERNED_CANDIDATE = "ungoverned_candidate"
REJECT_DUPLICATE_SPAN = "duplicate_span"


@dataclass(frozen=True, slots=True)
class ValidatedEntity:
    text: str
    type: str
    position: tuple[int, int]
    assertions: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)

    def as_organizer_json(self) -> dict[str, Any]:
        """Exactly the fields the organizer schema allows for this type - no more.

        `assertions` is legal only for TRIỆU_CHỨNG / CHẨN_ĐOÁN / THUỐC and must be ABSENT
        (not empty) for the two laboratory types; `candidates` only for CHẨN_ĐOÁN / THUỐC.
        Emitting `assertions: []` on a lab entity is an unsupported field, which is what
        blocked packaging of the full-8B run on 385 entities. The scored baseline's own
        output confirms the convention: lab entities carry `position`, `text`, `type` alone.
        """
        row: dict[str, Any] = {
            "text": self.text,
            "type": self.type,
            "position": [self.position[0], self.position[1]],
        }
        if self.type in ASSERTION_TYPES:
            row["assertions"] = list(self.assertions)
        if self.type in CANDIDATE_TYPES:
            row["candidates"] = list(self.candidates)
        return row


@dataclass
class ValidationReport:
    accepted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "rejected": dict(sorted(self.rejected.items()))}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def locate(source: str, surface: str, used: Counter[str]) -> tuple[int, int] | None:
    """The next unused occurrence of ``surface`` in ``source``, or None.

    Duplicate surface forms are handled by occurrence order: the first proposal for a repeated
    mention takes the first occurrence, the second takes the second. Deterministic, and it
    never invents an offset the model supplied.
    """
    if not surface:
        return None
    start = 0
    for _ in range(used[surface] + 1):
        index = source.find(surface, start)
        if index < 0:
            return None
        start = index + 1
    used[surface] += 1
    return (index, index + len(surface))


def validate(
    source_text: str,
    proposals: list[dict[str, Any]],
    governed_pool: dict[str, set[str]] | None = None,
) -> tuple[list[ValidatedEntity], ValidationReport]:
    """Turn raw model proposals into organizer-valid entities, dropping the unprovable.

    ``governed_pool`` maps a proposal key to the candidate ids that were offered for it. A
    candidate absent from its own offered set is removed - the model may only choose from
    what it was given, so it can never emit a code the KB does not contain.
    """
    source = _normalize(source_text)
    report = ValidationReport()
    used: Counter[str] = Counter()
    seen_spans: set[tuple[int, int, str]] = set()
    out: list[ValidatedEntity] = []

    for index, proposal in enumerate(proposals):
        entity_type = str(proposal.get("type", ""))
        surface = _normalize(str(proposal.get("text", ""))).strip()
        if not surface:
            report.rejected[REJECT_EMPTY_TEXT] += 1
            continue
        if entity_type not in ORGANIZER_TYPES:
            report.rejected[REJECT_UNKNOWN_TYPE] += 1
            continue
        span = locate(source, surface, used)
        if span is None:
            report.rejected[
                REJECT_OCCURRENCE_EXHAUSTED if surface in source else REJECT_SPAN_NOT_FOUND
            ] += 1
            continue
        key = (span[0], span[1], entity_type)
        if key in seen_spans:
            report.rejected[REJECT_DUPLICATE_SPAN] += 1
            continue
        seen_spans.add(key)

        assertions: tuple[str, ...] = ()
        if entity_type in ASSERTION_TYPES:
            raw = proposal.get("assertions") or {}
            assertions = tuple(name for name in ASSERTION_KEYS if bool(raw.get(name, False)))

        candidates: tuple[str, ...] = ()
        if entity_type in CANDIDATE_TYPES:
            offered = (governed_pool or {}).get(str(proposal.get("pool_key", index)), set())
            chosen: list[str] = []
            for code in proposal.get("candidates") or []:
                code = str(code).strip()
                if code and code in offered and code not in chosen:
                    chosen.append(code)
                elif code and code not in offered:
                    report.rejected[REJECT_UNGOVERNED_CANDIDATE] += 1
            candidates = tuple(chosen)

        out.append(
            ValidatedEntity(
                text=surface,
                type=entity_type,
                position=span,
                assertions=assertions,
                candidates=candidates,
            )
        )
        report.accepted += 1

    out.sort(key=lambda e: (e.position[0], e.position[1], e.type))
    return out, report


__all__ = [
    "ASSERTION_KEYS",
    "ASSERTION_TYPES",
    "CANDIDATE_TYPES",
    "ORGANIZER_TYPES",
    "REJECT_DUPLICATE_SPAN",
    "REJECT_EMPTY_TEXT",
    "REJECT_OCCURRENCE_EXHAUSTED",
    "REJECT_SPAN_NOT_FOUND",
    "REJECT_UNGOVERNED_CANDIDATE",
    "REJECT_UNKNOWN_TYPE",
    "ValidatedEntity",
    "ValidationReport",
    "locate",
    "validate",
]
