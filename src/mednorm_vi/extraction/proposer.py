"""Qwen contextual proposer: three specialized passes (0081, repaired).

The first real Colab smoke failed every document: `proposer_parse_failed = 3` out of three
documents. The audit found two causes, both here rather than in the model:

1. **Truncation.** The proposer asked one question covering every entity type in a whole
   document, and was answered with the shared `QwenDisambiguator` default of 256 new tokens.
   A document-level entity list does not fit in 256 tokens, so the reply ended mid-object,
   the closing brace never arrived, and the extractor - which requires one - found nothing.
2. **A single fragile contract.** One large JSON list meant any one problem discarded the
   whole document, and it asked one prompt to teach five type distinctions at once.

So the pass is now split into three narrow extractions that parse independently - a failure
in one keeps the other two - and JSON extraction is brace-matched rather than regex-greedy,
tolerant of Markdown fences and of a Qwen3 thinking block, and honest about truncation
instead of silently returning nothing.

Diagnostics record shape, not content: output length, whether generation reached its limit,
whether extraction succeeded, the failure category, and a short sanitized head and tail of
the reply for debugging. Thinking blocks are stripped before anything is recorded, so no
chain-of-thought is stored. Hallucinated text is never repaired - it is aligned against the
source or dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..validation.organizer import ORGANIZER_TYPES

#: Qwen3 emits its reasoning inside these when thinking mode is on. Removed before parsing
#: and before any diagnostic is recorded.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_OPEN_THINK = re.compile(r"<think>.*\Z", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)

PASS_DIAGNOSIS = "diagnosis_symptom"
PASS_MEDICATION = "medication"
PASS_LAB = "laboratory"
PASSES: tuple[str, ...] = (PASS_DIAGNOSIS, PASS_MEDICATION, PASS_LAB)

#: Types each pass is allowed to return. A pass returning a type it was not asked for is a
#: sign the prompt did not land, so the row is dropped and counted rather than accepted.
PASS_TYPES: dict[str, frozenset[str]] = {
    PASS_DIAGNOSIS: frozenset({"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}),
    PASS_MEDICATION: frozenset({"THUỐC"}),
    PASS_LAB: frozenset({"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}),
}

_SHARED_RULES = (
    "You will see the note with a stable identifier on every line. For each entity you find, "
    "return the line identifier, the EXACT substring of that line, its type, and which "
    "occurrence on that line it is (0 for the first).\n"
    "\n"
    "ABSOLUTE RULES\n"
    "* Copy the text character for character from the line. Do not fix spelling, do not "
    "expand an abbreviation, do not translate, do not re-order words.\n"
    "* Never return a character position. Return the line id and the text.\n"
    "* A mention that appears several times is returned once per occurrence, each with its "
    "own line id and occurrence number. Repeated mentions are not duplicates.\n"
    "* Never return a fragment that sits inside a larger ordinary word. A short form is an "
    "entity only when it stands as its own word.\n"
    "* If you are not confident an entity is there, leave it out.\n"
)

RUBRIC_DIAGNOSIS = (
    "You extract DIAGNOSES and SYMPTOMS from a Vietnamese clinical note. Extract nothing "
    "else - no medications, no tests, no test results.\n"
    "\n"
    f"{_SHARED_RULES}"
    "\n"
    "THE TWO TYPES\n"
    "* CHẨN_ĐOÁN - a named disease or condition being diagnosed, suspected or carried.\n"
    "* TRIỆU_CHỨNG - a symptom or sign the patient experiences or the clinician observes.\n"
    "\n"
    "DISEASE VERSUS SYMPTOM\n"
    "A symptom is what the patient feels or shows. A diagnosis is the named condition that "
    "explains it. If the text names a disease entity it is CHẨN_ĐOÁN even inside a list of "
    "complaints; if it describes a sensation, a finding or a change, it is TRIỆU_CHỨNG.\n"
    "\n"
    "COMPLETE MINIMAL CONCEPTS\n"
    "Return the COMPLETE clinical concept and nothing beyond it. Do NOT split one concept "
    "into pieces: if a condition is named by several words together, return those words as "
    "ONE entity, never as two. Equally, do not swallow a whole clause around it."
)

RUBRIC_MEDICATION = (
    "You extract MEDICATIONS from a Vietnamese clinical note. Extract nothing else - no "
    "diagnoses, no symptoms, no tests, no results.\n"
    "\n"
    f"{_SHARED_RULES}"
    "\n"
    "THE TYPE\n"
    "* THUỐC - a medication.\n"
    "\n"
    "WHAT COUNTS AS THE NAME\n"
    "A medication may be written in Vietnamese or English, as a brand name, a generic name "
    "or a transliteration. All of those are medications.\n"
    "\n"
    "BOUNDARIES\n"
    "* NEVER return a substring of a longer medication name. Return the whole name as "
    "written.\n"
    "* Include the strength, form, route and frequency when they are written as part of the "
    "SAME prescribed item.\n"
    "* Stop at the end of that item. Do not run into the next drug, into the reason it was "
    "given, or across a list separator."
)

RUBRIC_LAB = (
    "You extract LABORATORY TESTS and their RESULTS from a Vietnamese clinical note. Extract "
    "nothing else - no diagnoses, no symptoms, no medications.\n"
    "\n"
    f"{_SHARED_RULES}"
    "\n"
    "THE TWO TYPES\n"
    "* TÊN_XÉT_NGHIỆM - the NAME of a test, measurement or investigation.\n"
    "* KẾT_QUẢ_XÉT_NGHIỆM - the RESULT that a named test produced: its value with unit, or "
    "its stated finding.\n"
    "\n"
    "A RESULT BELONGS TO A TEST\n"
    "Only return a result when the test that produced it is identifiable nearby. A number "
    "with no named measurement is NOT a result: dates, ages, doses, counts and identifiers "
    "are never results.\n"
    "\n"
    "The name of the measurement and the value it produced are two separate entities. Do not "
    "merge them into one span."
)

RUBRICS: dict[str, str] = {
    PASS_DIAGNOSIS: RUBRIC_DIAGNOSIS,
    PASS_MEDICATION: RUBRIC_MEDICATION,
    PASS_LAB: RUBRIC_LAB,
}


def schema_for(pass_name: str) -> str:
    allowed = " | ".join(sorted(PASS_TYPES[pass_name]))
    return (
        "Return JSON only, in exactly this shape and nothing else:\n"
        '{"entities": [{"line_id": "L012", "text": "<exact substring>", '
        f'"type": "<{allowed}>", "occurrence": 0}}]}}\n'
        'If you find nothing, return {"entities": []}.'
    )


def build_pass_prompt(pass_name: str, rendered_document: str) -> str:
    """One narrow extraction question over the line-addressed document."""
    if pass_name not in RUBRICS:
        raise KeyError(f"unknown proposer pass {pass_name!r}; expected one of {PASSES}")
    return (
        f"{RUBRICS[pass_name]}\n\n"
        f"DOCUMENT:\n{rendered_document}\n\n"
        f"{schema_for(pass_name)}\n"
    )


# ------------------------------------------------------------------ robust extraction

PARSE_OK = "ok"
PARSE_EMPTY = "empty_reply"
PARSE_NO_JSON = "no_json_object_found"
PARSE_TRUNCATED = "truncated_before_json_closed"
PARSE_INVALID_JSON = "invalid_json"
PARSE_WRONG_SHAPE = "json_not_an_entity_list"
PARSE_BAD_ROW = "malformed_row"
PARSE_WRONG_TYPE_FOR_PASS = "type_not_requested_by_this_pass"

#: How much of a reply is kept for debugging. Bounded on purpose.
DEBUG_EXCERPT = 160


def strip_reasoning(reply: str) -> str:
    """Remove Qwen3 thinking blocks. Applied before parsing AND before any diagnostic."""
    text = _THINK_BLOCK.sub(" ", reply or "")
    return _OPEN_THINK.sub(" ", text).strip()


def extract_json_object(text: str) -> tuple[str, str]:
    """The first balanced top-level JSON object, or `("", reason)`.

    Brace matching rather than a greedy regex: a greedy `\\{.*\\}` spans from the first brace
    to the last one anywhere in the reply, which silently swallows prose between two
    unrelated objects, and finds nothing at all when the reply was cut off mid-object.
    Distinguishing "no object" from "object never closed" is what makes truncation visible.
    """
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start = body.find("{")
    if start < 0:
        return "", PARSE_NO_JSON

    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(body)):
        character = body[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return body[start : position + 1], PARSE_OK
    return "", PARSE_TRUNCATED


@dataclass
class GenerationReport:
    """Shape-only diagnostics for one Qwen call. Never stores reasoning."""

    pass_name: str
    characters: int = 0
    generated_tokens: int = 0
    hit_token_limit: bool = False
    json_extracted: bool = False
    category: str = PARSE_OK
    rows_returned: int = 0
    rows_kept: int = 0
    head: str = ""
    tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_name,
            "characters": self.characters,
            "generated_tokens": self.generated_tokens,
            "hit_token_limit": self.hit_token_limit,
            "json_extracted": self.json_extracted,
            "category": self.category,
            "rows_returned": self.rows_returned,
            "rows_kept": self.rows_kept,
            "excerpt_head": self.head,
            "excerpt_tail": self.tail,
        }


@dataclass(frozen=True, slots=True)
class RawProposal:
    """Exactly what the model said, before any alignment. Never trusted, only parsed."""

    line_id: str
    text: str
    type: str
    occurrence: int
    source_pass: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id, "text": self.text, "type": self.type,
            "occurrence": self.occurrence, "pass": self.source_pass,
        }


@dataclass
class PassResult:
    proposals: list[RawProposal] = field(default_factory=list)
    report: GenerationReport = field(default_factory=lambda: GenerationReport(""))
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, reason: str, amount: int = 1) -> None:
        if amount:
            self.counters[reason] = self.counters.get(reason, 0) + amount


def parse_pass(
    pass_name: str,
    reply: str,
    *,
    generated_tokens: int = 0,
    hit_token_limit: bool = False,
) -> PassResult:
    """Parse one specialized pass. Each pass fails on its own and never affects another."""
    result = PassResult()
    cleaned = strip_reasoning(reply)
    report = GenerationReport(
        pass_name=pass_name,
        characters=len(cleaned),
        generated_tokens=generated_tokens,
        hit_token_limit=hit_token_limit,
        head=cleaned[:DEBUG_EXCERPT],
        tail=cleaned[-DEBUG_EXCERPT:] if len(cleaned) > DEBUG_EXCERPT else "",
    )
    result.report = report

    if not cleaned:
        report.category = PARSE_EMPTY
        result.count(f"proposer_{pass_name}_{PARSE_EMPTY}")
        return result

    payload_text, reason = extract_json_object(cleaned)
    if not payload_text:
        report.category = reason
        result.count(f"proposer_{pass_name}_{reason}")
        return result

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        report.category = PARSE_INVALID_JSON
        result.count(f"proposer_{pass_name}_{PARSE_INVALID_JSON}")
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        report.category = PARSE_WRONG_SHAPE
        result.count(f"proposer_{pass_name}_{PARSE_WRONG_SHAPE}")
        return result

    report.json_extracted = True
    rows = payload["entities"]
    report.rows_returned = len(rows)
    allowed = PASS_TYPES[pass_name]

    for row in rows:
        if not isinstance(row, dict):
            result.count(PARSE_BAD_ROW)
            continue
        text = str(row.get("text", ""))
        entity_type = str(row.get("type", ""))
        if not text.strip() or entity_type not in ORGANIZER_TYPES:
            result.count(PARSE_BAD_ROW)
            continue
        if entity_type not in allowed:
            result.count(PARSE_WRONG_TYPE_FOR_PASS)
            continue
        try:
            occurrence = int(row.get("occurrence", 0) or 0)
        except (TypeError, ValueError):
            occurrence = 0
        result.proposals.append(
            RawProposal(
                line_id=str(row.get("line_id", "")).strip(),
                text=text, type=entity_type, occurrence=max(occurrence, 0),
                source_pass=pass_name,
            )
        )
    report.rows_kept = len(result.proposals)
    return result


__all__ = [
    "DEBUG_EXCERPT",
    "PARSE_BAD_ROW",
    "PARSE_EMPTY",
    "PARSE_INVALID_JSON",
    "PARSE_NO_JSON",
    "PARSE_OK",
    "PARSE_TRUNCATED",
    "PARSE_WRONG_SHAPE",
    "PARSE_WRONG_TYPE_FOR_PASS",
    "PASSES",
    "PASS_DIAGNOSIS",
    "PASS_LAB",
    "PASS_MEDICATION",
    "PASS_TYPES",
    "RUBRICS",
    "GenerationReport",
    "PassResult",
    "RawProposal",
    "build_pass_prompt",
    "extract_json_object",
    "parse_pass",
    "schema_for",
    "strip_reasoning",
]
