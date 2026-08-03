"""Qwen pass 2: finite entity verifier (0081).

**One index space, always local to one invocation.** Every call shows a fresh list numbered
from zero, and an index means a position in *that* list and nothing else - there is no global
proposal id anywhere in this protocol, so there is nothing to confuse it with. The worked
example in the output contract is generated from the offered indices for the same reason: a
fixed example showing `[0] [1] [2]` to a caller offering one candidate is an instruction to
produce two invalid indices, which is exactly what the first smoke did.

The model is shown an indexed list of spans that already exist in the document and may do
exactly three things to each: accept it, reject it, or change its type. It cannot write text,
so it cannot introduce a span; the only thing it returns is an index and a verdict. Anything
referring to an index outside the list is discarded and counted - fail closed, because an
out-of-range index means the model was not answering about this list.

A reply that cannot be parsed is not a licence to accept everything. The fallback keeps only
the proposals the trusted mention expert already found, which is the previous milestone's
behaviour and therefore never worse than the control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..reasoner.validator import ORGANIZER_TYPES
from .document import DocumentView
from .lattice import LatticeGroup
from .proposals import SOURCE_E3, Proposal
from .proposer import extract_json_object, strip_reasoning

ACCEPT = "accept"
REJECT = "reject"
RETYPE = "retype"
ACTIONS: tuple[str, ...] = (ACCEPT, REJECT, RETYPE)

PARSE_FAILED = "verifier_parse_failed"
INVALID_INDEX = "verifier_invalid_index"
INVALID_ACTION = "verifier_invalid_action"
INVALID_TYPE = "verifier_invalid_type"

VERIFIER_RUBRIC = (
    "You are verifying candidate clinical entities that were extracted from a Vietnamese "
    "clinical note. Every candidate below is already an exact substring of the note.\n"
    "\n"
    "You may ONLY do three things to a candidate, by its index:\n"
    "  accept  - it is a correct entity of the given type\n"
    "  reject  - it is not an entity, or it is the wrong piece of text\n"
    "  retype  - it is an entity but of a different one of the five types\n"
    "\n"
    "You CANNOT write new text. You CANNOT change where a candidate starts or ends. If the "
    "correct span is not in the list, reject the ones that are wrong and accept nothing "
    "else.\n"
    "\n"
    "HOW TO CHOOSE BETWEEN OVERLAPPING CANDIDATES\n"
    "* Candidates in the same group compete for the same text. Accept the one that is the "
    "COMPLETE minimal clinical concept.\n"
    "* Prefer the complete concept over a fragment of it.\n"
    "* Prefer the concept over a whole clause that merely contains it.\n"
    "* When two candidates are genuinely different entities that happen to touch, you may "
    "accept both.\n"
    "* When you cannot tell which boundary is right, reject them all. Leaving a mention out "
    "costs less than annotating the wrong text.\n"
    "\n"
    "TYPES\n"
    "  TRIỆU_CHỨNG - symptom or sign\n"
    "  CHẨN_ĐOÁN - named disease or condition\n"
    "  TÊN_XÉT_NGHIỆM - the name of a test or measurement\n"
    "  KẾT_QUẢ_XÉT_NGHIỆM - the value or finding a named test produced\n"
    "  THUỐC - a medication\n"
    "\n"
    "Reject a candidate that is a fragment sitting inside an ordinary word. Reject a number "
    "that is not the result of a test named nearby."
)

def schema_for(option_count: int) -> str:
    """The output contract, with an example built from THIS invocation's indices.

    The first smoke returned 675 invalid indices against 357 offered options. The cause was
    here: the schema carried a fixed worked example using indices 0, 1 and 2, and most
    groups offer a single option `[0]`. The model copied the example, so every one-option
    group produced two out-of-range indices - 347 groups, ~676 invalid indices, which is
    what was observed. An example must never show an index the caller did not offer.
    """
    last = max(option_count - 1, 0)
    if option_count <= 1:
        example = '{"verdicts": [{"index": 0, "action": "accept"}]}'
    else:
        example = (
            '{"verdicts": [{"index": 0, "action": "accept"}, '
            f'{{"index": {last}, "action": "reject"}}]}}'
        )
    allowed = "0" if option_count <= 1 else f"0 to {last}"
    return (
        "Return JSON only, for example:\n"
        f"{example}\n"
        f"This list has {option_count} candidate(s), so the ONLY valid index values are "
        f"{allowed}. Return a verdict for each candidate you have an opinion about, and "
        "never an index outside that range. Do not return any other field."
    )


def build_verification_prompt(
    document: DocumentView,
    group: LatticeGroup,
    *,
    context_radius: int = 200,
) -> str:
    """One group of competing candidates, with the surrounding text as context."""
    left = max(0, group.start - context_radius)
    right = min(len(document.source), group.end + context_radius)
    context = " ".join(document.source[left:right].split())
    listed = "\n".join(
        f"[{index}] {option.type}: \"{option.text}\""
        for index, option in enumerate(group.options)
    )
    return (
        f"{VERIFIER_RUBRIC}\n\n"
        f"CONTEXT: {context}\n\n"
        f"CANDIDATES:\n{listed}\n\n"
        f"{schema_for(len(group.options))}\n"
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    index: int
    action: str
    entity_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "action": self.action, "type": self.entity_type}


@dataclass
class VerificationResult:
    accepted: list[Proposal] = field(default_factory=list)
    rejected: list[Proposal] = field(default_factory=list)
    retyped: int = 0
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, reason: str, amount: int = 1) -> None:
        self.counters[reason] = self.counters.get(reason, 0) + amount


def parse_verdicts(reply: str, option_count: int) -> tuple[list[Verdict], dict[str, int]]:
    """Strict parse restricted to the indices actually offered."""
    counters: dict[str, int] = {}

    def count(reason: str) -> None:
        counters[reason] = counters.get(reason, 0) + 1

    payload_text, _ = extract_json_object(strip_reasoning(reply))
    if not payload_text:
        count(PARSE_FAILED)
        return [], counters
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        count(PARSE_FAILED)
        return [], counters
    if not isinstance(payload, dict) or "verdicts" not in payload:
        count(PARSE_FAILED)
        return [], counters

    seen: set[int] = set()
    out: list[Verdict] = []
    for row in payload.get("verdicts") or ():
        if not isinstance(row, dict):
            count(PARSE_FAILED)
            continue
        raw_index = row.get("index")
        # A genuine integer only. `int("0")` and `int(1.5)` both succeed in Python and would
        # silently turn a string or a float into a position, which is the sort of quiet
        # reinterpretation this protocol exists to prevent.
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            count(INVALID_INDEX)
            continue
        index = raw_index
        if not 0 <= index < option_count or index in seen:
            count(INVALID_INDEX)
            continue
        action = str(row.get("action", "")).strip().lower()
        if action not in ACTIONS:
            count(INVALID_ACTION)
            continue
        entity_type = str(row.get("type", "")).strip()
        if action == RETYPE and entity_type not in ORGANIZER_TYPES:
            count(INVALID_TYPE)
            continue
        seen.add(index)
        out.append(Verdict(index=index, action=action, entity_type=entity_type))
    return out, counters


def apply_verdicts(
    group: LatticeGroup, verdicts: list[Verdict], *, parse_failed: bool
) -> VerificationResult:
    """Turn verdicts into accepted/rejected proposals. Unmentioned candidates are rejected.

    Silence is not acceptance: a candidate the model did not vote on has no support, and the
    whole point of this pass is that support is required. When the reply could not be parsed
    at all, the E3 proposals survive and nothing else does.
    """
    result = VerificationResult()
    by_index = {v.index: v for v in verdicts}
    for index, option in enumerate(group.options):
        if parse_failed:
            if SOURCE_E3 in option.sources:
                result.accepted.append(option)
            else:
                result.rejected.append(option)
            continue
        verdict = by_index.get(index)
        if verdict is None or verdict.action == REJECT:
            result.rejected.append(option)
            continue
        if verdict.action == RETYPE and verdict.entity_type != option.type:
            import dataclasses

            result.accepted.append(dataclasses.replace(option, type=verdict.entity_type))
            result.retyped += 1
            continue
        result.accepted.append(option)
    if parse_failed:
        result.count(PARSE_FAILED)
    return result


__all__ = [
    "ACCEPT",
    "ACTIONS",
    "INVALID_ACTION",
    "INVALID_INDEX",
    "INVALID_TYPE",
    "PARSE_FAILED",
    "REJECT",
    "RETYPE",
    "VERIFIER_RUBRIC",
    "VerificationResult",
    "Verdict",
    "apply_verdicts",
    "build_verification_prompt",
    "parse_verdicts",
    "schema_for",
]
