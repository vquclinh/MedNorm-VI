"""Qwen pass 1: contextual entity proposer (0081).

The model reads the whole document with line ids and says *what* it sees and *where roughly*
- never an offset. Its output is `(line_id, text, occurrence)`; `document.align` turns that
into offsets by finding the text in the source, so a proposal the document does not contain
simply does not survive.

The rubric teaches the distinctions the public-test diagnostics showed were being missed:
complete minimal concepts rather than fragments, diagnosis versus symptom, test name versus
test result, complete medication names with their regimen, and the two failure modes that
produce junk - abbreviations matched inside words, and nearby numbers annotated as results.

Nothing in the prompt names a disease, a drug or a test. It teaches shape, not content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..reasoner.validator import ORGANIZER_TYPES
from .document import DocumentView

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)

PROPOSER_RUBRIC = (
    "You extract clinical entities from a Vietnamese clinical note.\n"
    "\n"
    "You will be shown the note with a stable identifier on every line. For each entity you "
    "find, you return the line identifier, the EXACT substring of that line, its type, and "
    "which occurrence on that line it is (0 for the first).\n"
    "\n"
    "ABSOLUTE RULES\n"
    "* The text you return must be copied character for character from the line. Do not fix "
    "spelling, do not expand an abbreviation, do not translate, do not re-order words.\n"
    "* Never return a character position. Return the line id and the text.\n"
    "* If a mention appears several times, return each occurrence separately with its own "
    "line id and occurrence number. Repeated mentions are not duplicates.\n"
    "\n"
    "THE FIVE TYPES\n"
    "* TRIỆU_CHỨNG - a symptom or sign the patient experiences or the clinician observes.\n"
    "* CHẨN_ĐOÁN - a named disease or condition being diagnosed, suspected or carried.\n"
    "* TÊN_XÉT_NGHIỆM - the NAME of a test, measurement or investigation.\n"
    "* KẾT_QUẢ_XÉT_NGHIỆM - the RESULT of a test: its value, with unit, or its stated "
    "finding. A result belongs to a test that is actually named nearby. A number that is not "
    "the result of a named test is NOT this type.\n"
    "* THUỐC - a medication.\n"
    "\n"
    "DIAGNOSIS VERSUS SYMPTOM\n"
    "A symptom is what the patient feels or shows. A diagnosis is the named condition that "
    "explains it. If the text names a disease entity, it is CHẨN_ĐOÁN even when it appears in "
    "a list of complaints; if it describes a sensation, a finding or a change, it is "
    "TRIỆU_CHỨNG.\n"
    "\n"
    "TEST NAME VERSUS TEST RESULT\n"
    "The name of the measurement and the value it produced are two different entities. "
    "Return the name as TÊN_XÉT_NGHIỆM and the value as KẾT_QUẢ_XÉT_NGHIỆM. Do not merge them "
    "into one span, and do not return a value without its test being present.\n"
    "\n"
    "COMPLETE MINIMAL CONCEPTS\n"
    "Return the complete clinical concept, and nothing beyond it. A fragment of a concept is "
    "wrong; so is a whole clause. If a modifier is part of the concept as written, keep it; "
    "if it is a separate statement, leave it out.\n"
    "\n"
    "MEDICATIONS\n"
    "Return the complete medication as written, including its strength, form, route and "
    "frequency when they are written as part of the same item. Stop at the end of that item: "
    "do not run into the next drug, into a reason for prescribing, or across a list "
    "separator.\n"
    "\n"
    "TWO THINGS THAT ARE ALWAYS WRONG\n"
    "* Never return a fragment that sits inside a larger ordinary word. An abbreviation is "
    "only an entity when it stands as its own word.\n"
    "* Never return an arbitrary nearby number as a test result. Dates, ages, doses, counts "
    "and identifiers are not results.\n"
    "\n"
    "If you are not confident an entity is there, leave it out. A missing proposal costs "
    "less than a wrong one."
)

PROPOSER_SCHEMA = (
    'Return JSON only, in exactly this shape:\n'
    '{"entities": [{"line_id": "L012", "text": "<exact substring>", '
    '"type": "<one of the five types>", "occurrence": 0}]}\n'
    'If you find nothing, return {"entities": []}.'
)


def build_proposal_prompt(document: DocumentView, *, section_hint: str = "") -> str:
    """The whole document, line-addressed. The model sees the note and nothing else."""
    heading = f"NOTE SECTIONS: {section_hint}\n" if section_hint else ""
    return (
        f"{PROPOSER_RUBRIC}\n\n"
        f"{heading}"
        f"DOCUMENT:\n{document.render()}\n\n"
        f"{PROPOSER_SCHEMA}\n"
    )


@dataclass(frozen=True, slots=True)
class RawProposal:
    """Exactly what the model said, before any alignment. Never trusted, only parsed."""

    line_id: str
    text: str
    type: str
    occurrence: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id, "text": self.text, "type": self.type,
            "occurrence": self.occurrence,
        }


PARSE_FAILED = "proposer_parse_failed"
PARSE_BAD_ROW = "proposer_malformed_row"


def parse_proposals(reply: str) -> tuple[list[RawProposal], dict[str, int]]:
    """Strict parse. A malformed reply yields no proposals, never a guess.

    Rows are validated individually so one bad row does not discard a good document; the
    counts come back so a prompt that stops landing shows up as a number.
    """
    counters: dict[str, int] = {}

    def count(reason: str) -> None:
        counters[reason] = counters.get(reason, 0) + 1

    match = _JSON_OBJECT.search(reply or "")
    if not match:
        count(PARSE_FAILED)
        return [], counters
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        count(PARSE_FAILED)
        return [], counters
    if not isinstance(payload, dict):
        count(PARSE_FAILED)
        return [], counters

    out: list[RawProposal] = []
    for row in payload.get("entities") or ():
        if not isinstance(row, dict):
            count(PARSE_BAD_ROW)
            continue
        text = str(row.get("text", ""))
        entity_type = str(row.get("type", ""))
        if not text.strip() or entity_type not in ORGANIZER_TYPES:
            count(PARSE_BAD_ROW)
            continue
        try:
            occurrence = int(row.get("occurrence", 0) or 0)
        except (TypeError, ValueError):
            occurrence = 0
        out.append(
            RawProposal(
                line_id=str(row.get("line_id", "")).strip(),
                text=text, type=entity_type, occurrence=max(occurrence, 0),
            )
        )
    return out, counters


__all__ = [
    "PARSE_BAD_ROW",
    "PARSE_FAILED",
    "PROPOSER_RUBRIC",
    "PROPOSER_SCHEMA",
    "RawProposal",
    "build_proposal_prompt",
    "parse_proposals",
]
