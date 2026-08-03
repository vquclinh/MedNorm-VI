"""Contextual assertion pass (0081).

Runs only after spans and types are final, so an assertion is always about a fixed entity
rather than about a span that might still move.

Three assertions exist - `isNegated`, `isFamily`, `isHistorical` - and only three types carry
them. The rules are deliberately narrow, because a wrong assertion is scored against us just
as a missing one is:

**Negation** scopes to the entity's own clause. A negation cue earlier in the sentence stops
at the first clause break, so "no fever, has cough" does not negate the cough.

**Family** requires a family reference in the same clause, and the subject is whoever is
named first: a patient narrating their own history is never family history, however the
sentence is phrased, while "the patient's mother" is.

**Historical** comes from the section the entity sits in, or from an explicit past-time
marker. Chronicity is not history: a condition described as chronic is a *current* chronic
condition unless the text says it is in the past.

The cue lists are ordinary Vietnamese clinical vocabulary - negation words, kinship words,
section headings. They contain no disease, drug or test name, and nothing derived from any
particular test set.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ..reasoner.validator import ASSERTION_TYPES
from .document import DocumentView
from .proposals import Proposal

NEGATED = "isNegated"
FAMILY = "isFamily"
HISTORICAL = "isHistorical"
ASSERTION_KEYS: tuple[str, ...] = (NEGATED, FAMILY, HISTORICAL)

#: Clause boundaries. A cue does not reach past one of these.
_CLAUSE_BREAK = re.compile(r"[,;.:\n()\[\]]|\bnhưng\b|\bcòn\b|\bmà\b|\btuy nhiên\b")

_NEGATION_CUES: tuple[str, ...] = (
    "không", "chưa", "ko", "k có", "phủ định", "loại trừ", "không có", "chưa có",
    "không thấy", "chưa thấy", "không ghi nhận", "chưa ghi nhận", "âm tính", "không bị",
    "không còn", "hết",
)
#: A cue that only negates when it precedes the entity closely (it is also an ordinary word).
_NEGATION_STRICT: frozenset[str] = frozenset({"hết", "ko", "k có"})

_FAMILY_CUES: tuple[str, ...] = (
    "gia đình", "bố", "cha", "mẹ", "ba", "má", "anh trai", "chị gái", "em trai",
    "em gái", "ông", "bà", "con trai", "con gái", "họ hàng", "người thân", "cậu", "dì",
    "chú", "bác", "cô",
)
#: The patient as the subject of the clause - either narrating in the first person or
#: referred to as the patient. Both mean "this is about the patient", not about a relative.
_SELF_CUES: tuple[str, ...] = (
    "tôi", "em", "cháu", "mình", "bản thân", "bệnh nhân", "bn", "người bệnh",
)

_HISTORY_SECTION_CUES: tuple[str, ...] = (
    "tiền sử", "tiền căn", "bệnh sử", "quá trình bệnh lý", "tiền sử bệnh",
)
_FAMILY_SECTION_CUES: tuple[str, ...] = ("tiền sử gia đình", "gia đình")
_PAST_TIME_CUES: tuple[str, ...] = (
    "đã từng", "từng", "trước đây", "trước đó", "cách đây", "năm ngoái", "trong quá khứ",
    "đã điều trị", "đã mổ", "đã phẫu thuật",
)
#: Chronicity is NOT history. Listed so the intent is explicit rather than accidental.
_CHRONICITY_NOT_HISTORY: tuple[str, ...] = ("mạn tính", "mãn tính", "mạn", "kéo dài")

#: A section heading looks like `Tiền sử:` or an ALL-CAPS line.
_HEADING = re.compile(r"^\s*([^:\n]{2,60}):\s*(.*)$")


def fold(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def _contains_cue(text: str, cues: tuple[str, ...]) -> str:
    """The first cue present as a whole phrase, or empty."""
    return _first_cue(text, cues)[0]


def _first_cue(text: str, cues: tuple[str, ...]) -> tuple[str, int]:
    """The earliest cue in the text and where it starts. `("", -1)` when there is none."""
    folded = fold(text)
    best, position = "", -1
    for cue in cues:
        match = re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", folded)
        if match and (position < 0 or match.start() < position):
            best, position = cue, match.start()
    return best, position


@dataclass(frozen=True, slots=True)
class Section:
    """One heading and the span of document it governs."""

    heading: str
    start: int
    end: int

    @property
    def is_history(self) -> bool:
        return bool(_contains_cue(self.heading, _HISTORY_SECTION_CUES))

    @property
    def is_family(self) -> bool:
        return bool(_contains_cue(self.heading, _FAMILY_SECTION_CUES))


def detect_sections(document: DocumentView) -> list[Section]:
    """Headed regions of the note. A heading governs until the next heading."""
    marks: list[tuple[str, int]] = []
    for line in document.lines:
        match = _HEADING.match(line.text)
        if match and match.group(1).strip():
            marks.append((match.group(1).strip(), line.start))
    sections: list[Section] = []
    for index, (heading, start) in enumerate(marks):
        end = marks[index + 1][1] if index + 1 < len(marks) else len(document.source)
        sections.append(Section(heading=heading, start=start, end=end))
    return sections


def section_for(sections: list[Section], offset: int) -> Section | None:
    for section in sections:
        if section.start <= offset < section.end:
            return section
    return None


def clause_of(source: str, start: int, end: int) -> tuple[int, int]:
    """The entity's own clause: back to the previous clause break, forward to the next."""
    left = 0
    for match in _CLAUSE_BREAK.finditer(source, 0, start):
        left = match.end()
    right_match = _CLAUSE_BREAK.search(source, end)
    right = right_match.start() if right_match else len(source)
    return left, right


@dataclass
class AssertionOutcome:
    assertions: dict[tuple[int, int], tuple[str, ...]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, reason: str, amount: int = 1) -> None:
        self.counters[reason] = self.counters.get(reason, 0) + amount

    def as_dict(self) -> dict[str, Any]:
        return {
            "assertions": {f"{k[0]}:{k[1]}": list(v) for k, v in self.assertions.items()},
            "counters": dict(self.counters),
        }


def assert_entity(
    document: DocumentView,
    entity: Proposal,
    sections: list[Section],
) -> tuple[str, ...]:
    """The assertions for one entity. Empty tuple when the text supports none."""
    if entity.type not in ASSERTION_TYPES:
        return ()

    source = document.source
    left, right = clause_of(source, entity.start, entity.end)
    before = source[left : entity.start]
    clause = source[left:right]
    section = section_for(sections, entity.start)

    out: list[str] = []

    cue = _contains_cue(before, _NEGATION_CUES)
    if cue and (cue not in _NEGATION_STRICT or entity.start - left <= 40):
        out.append(NEGATED)

    # Whoever is named FIRST in the clause is its subject. Vietnamese noun phrases are
    # head-initial, so "mẹ bệnh nhân" is the patient's mother while "bệnh nhân" alone is the
    # patient - and a patient narrating their own history is never family history, however
    # the sentence is phrased.
    _, self_at = _first_cue(clause, _SELF_CUES)
    family_cue, family_at = _first_cue(clause, _FAMILY_CUES)
    subject_is_patient = self_at >= 0 and (family_at < 0 or self_at < family_at)
    if family_cue and not subject_is_patient:
        out.append(FAMILY)
    elif section is not None and section.is_family and not subject_is_patient:
        out.append(FAMILY)

    if section is not None and section.is_history:
        out.append(HISTORICAL)
    elif _contains_cue(clause, _PAST_TIME_CUES):
        out.append(HISTORICAL)
    # Chronicity alone is deliberately not enough; a chronic condition is current.

    return tuple(dict.fromkeys(out))


def run_assertions(
    document: DocumentView, entities: list[Proposal]
) -> AssertionOutcome:
    outcome = AssertionOutcome()
    sections = detect_sections(document)
    for entity in entities:
        if entity.type not in ASSERTION_TYPES:
            continue
        values = assert_entity(document, entity, sections)
        outcome.assertions[(entity.start, entity.end)] = values
        for value in values:
            outcome.count(value)
    return outcome


__all__ = [
    "ASSERTION_KEYS",
    "FAMILY",
    "HISTORICAL",
    "NEGATED",
    "AssertionOutcome",
    "Section",
    "assert_entity",
    "clause_of",
    "detect_sections",
    "run_assertions",
    "section_for",
]
