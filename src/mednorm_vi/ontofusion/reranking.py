"""Set-wise Qwen reranking over a small governed candidate set (0082).

BeLink's set-wise framing: the model sees every surviving candidate at once with an explicit
NONE, and picks one option letter. Deciding one candidate at a time makes NONE almost
impossible to reach, because there is nothing to compare against.

Two things are deliberately withheld from the prompt: **retrieval scores and ranks**. A
similarity number is not semantic evidence, and showing one anchors the model on whatever
the retrievers happened to like. Retriever metadata still exists - it drives deterministic
gating and the diagnostics - it just never reaches the model.

The model answers with an option letter and nothing else. The parser maps that letter back
to the governed id from the list this code built; there is no path by which a code the model
writes could become an answer. Anything unexpected - an unknown letter, several letters, a
literal code, prose - fails closed to NONE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..graphcent.ontology import IcdFacts, RxNormFacts

NONE_OPTION = "NONE"

#: A, B, C ... Enough letters for any set this pipeline will ever offer.
OPTION_LETTERS = tuple(chr(ord("A") + i) for i in range(20))

SELECT = "SELECT"
NONE = "NONE"

PARSE_FAILED = "reranker_parse_failed"
INVALID_OPTION = "reranker_invalid_option"
MULTIPLE_OPTIONS = "reranker_multiple_options"
LITERAL_CODE = "reranker_returned_a_code"

#: The answer must be a bare option token. Anything else is refused rather than salvaged.
_ANSWER = re.compile(r"^[\s\"'`]*\[?([A-Z]{1,4})\]?[\s\"'`.]*$")
_CODE_SHAPED = re.compile(r"\b([A-TV-Z]\d{2,4}(?:\.\d{1,2})?|\d{4,8})\b")

RERANK_RUBRIC = (
    "You are a conservative clinical ontology disambiguator.\n"
    "\n"
    "Choose the ONE ontology concept whose defining semantics are supported by the mention "
    "and its context, or choose NONE.\n"
    "\n"
    "DO NOT infer anything the text does not state:\n"
    "* not a subtype, not an anatomical site, not laterality;\n"
    "* not pregnancy, not malignancy;\n"
    "* not an acute or chronic state;\n"
    "* not a historical state;\n"
    "* not a medication strength, form or route.\n"
    "\n"
    "Prefer a BROADER supported concept over a more specific child the text does not "
    "support. A concept that adds detail the note does not contain is a different concept, "
    "not a better match.\n"
    "\n"
    "The order of the options carries no meaning. It is not a ranking, and no option is more "
    "likely because of where it appears.\n"
    "\n"
    "Several options that are near-synonyms, or the same concept at different levels of "
    "detail, are NOT several concepts. Pick the one the text supports, or NONE.\n"
    "\n"
    "Choose NONE if every option would require an assumption the text does not support. "
    "NONE is a correct answer and is expected to be common."
)

RERANK_SCHEMA = (
    "Answer with exactly one option token and nothing else - no explanation, no code, no "
    "punctuation. For example:\nA\nor\nNONE"
)


@dataclass(frozen=True, slots=True)
class Option:
    """One offered candidate. `concept_id` stays on this side of the prompt."""

    letter: str
    concept_id: str
    lines: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"letter": self.letter, "concept_id": self.concept_id,
                "lines": list(self.lines)}


def facts_lines(facts: IcdFacts | RxNormFacts | None, fallback: str) -> tuple[str, ...]:
    if facts is None:
        return (fallback,)
    return tuple(facts.as_prompt_lines())


def build_options(
    concept_ids: list[str],
    facts_by_id: dict[str, IcdFacts | RxNormFacts | None],
) -> list[Option]:
    """Letter the candidates in the order given. Ordering itself is decided upstream."""
    return [
        Option(
            letter=OPTION_LETTERS[position],
            concept_id=concept_id,
            lines=facts_lines(facts_by_id.get(concept_id), concept_id),
        )
        for position, concept_id in enumerate(concept_ids[: len(OPTION_LETTERS)])
    ]


def build_rerank_prompt(
    mention: str,
    entity_type: str,
    options: list[Option],
    *,
    section: str = "",
    previous_sentence: str = "",
    sentence: str = "",
    next_sentence: str = "",
) -> str:
    """Entity, context and candidates. No score, no rank, no retriever name."""
    context: list[str] = []
    if section:
        context.append(f"SECTION: {section}")
    for label, text in (
        ("previous", previous_sentence), ("current", sentence), ("next", next_sentence)
    ):
        if text.strip():
            context.append(f"({label}) {' '.join(text.split())}")

    listed: list[str] = []
    for option in options:
        listed.append(f"[{option.letter}]")
        listed.extend(f"  {line}" for line in option.lines)
    listed.append(f"[{NONE_OPTION}]")
    listed.append("  none of these concepts is supported by the mention and its context")

    return (
        f"{RERANK_RUBRIC}\n\n"
        f"MENTION: {mention}\nENTITY TYPE: {entity_type}\n"
        + ("CONTEXT:\n  " + "\n  ".join(context) + "\n" if context else "")
        + "\nCANDIDATES:\n" + "\n".join(listed) + "\n\n"
        + RERANK_SCHEMA + "\n"
    )


@dataclass
class Selection:
    """What the model chose, resolved against the offered options."""

    decision: str = NONE
    concept_id: str = ""
    letter: str = ""
    counters: dict[str, int] = field(default_factory=dict)

    @property
    def selected(self) -> bool:
        return self.decision == SELECT and bool(self.concept_id)

    def count(self, reason: str) -> None:
        self.counters[reason] = self.counters.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision, "concept_id": self.concept_id,
            "letter": self.letter, "counters": dict(self.counters),
        }


def parse_selection(reply: str, options: list[Option]) -> Selection:
    """Map one option token back to a governed id. Everything else fails closed to NONE."""
    selection = Selection()
    text = (reply or "").strip()
    if not text:
        selection.count(PARSE_FAILED)
        return selection

    if _CODE_SHAPED.search(text):
        # The model wrote a code instead of choosing. Refused: ids come from this list only.
        selection.count(LITERAL_CODE)
        return selection

    match = _ANSWER.match(text)
    if not match:
        letters = re.findall(r"\b([A-Z]{1,4})\b", text)
        if len({letter for letter in letters}) > 1:
            selection.count(MULTIPLE_OPTIONS)
        else:
            selection.count(PARSE_FAILED)
        return selection

    token = match.group(1).upper()
    if token == NONE_OPTION:
        selection.decision = NONE
        selection.letter = NONE_OPTION
        return selection

    chosen = next((o for o in options if o.letter == token), None)
    if chosen is None:
        selection.count(INVALID_OPTION)
        return selection

    selection.decision = SELECT
    selection.concept_id = chosen.concept_id
    selection.letter = chosen.letter
    return selection


__all__ = [
    "INVALID_OPTION",
    "LITERAL_CODE",
    "MULTIPLE_OPTIONS",
    "NONE",
    "NONE_OPTION",
    "OPTION_LETTERS",
    "PARSE_FAILED",
    "RERANK_RUBRIC",
    "RERANK_SCHEMA",
    "SELECT",
    "Option",
    "Selection",
    "build_options",
    "build_rerank_prompt",
    "facts_lines",
    "parse_selection",
]
