"""Finite, exact-source span alternatives (GraphCENT 0080).

Two generators, both returning literal slices of the note. No text is ever reformatted, and
neither generator can produce a span the source does not contain.

**Diagnosis**: reuses the 0079 safe-bridge rule unchanged - a fragment completion is offered
only when the text between fragments is at most one ordinary word, free of punctuation,
connectors and unexpected capitals. That rule was validated against every merge the 0078 run
produced.

**Medication**: the organizer's own example is `amlodipine 10 mg po daily`, not `amlodipine`.
E3 tends to extract the ingredient alone, so this walks rightward through the regimen tail -
strength, unit, concentration, form, route, frequency, PRN - and stops hard at anything that
starts a new item: a numbered entry, a newline, clause punctuation, or an indication phrase
like `điều trị`, which introduces why the drug was given rather than what was given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..reasoner.safe_bridge import evaluate_merge

SPAN_ORIGINAL = "e3_original"
SPAN_SAFE_BRIDGE = "safe_bridge_fragment_completion"
SPAN_DRUG_REGIMEN = "drug_regimen_right_expansion"

#: Regimen tokens worth absorbing after a medication name.
_UNIT = r"(?:mg|mcg|µg|ug|g|ml|l|iu|ui|đv|viên|ống|gói|lọ|chai|tuýp|%)"
_REGIMEN_TOKEN = re.compile(
    rf"^\s*(?:"
    rf"\d+(?:[.,]\d+)?\s*{_UNIT}"  # 10 mg, 0.5 ml
    rf"|\d+(?:[.,]\d+)?\s*{_UNIT}\s*/\s*\d*\s*{_UNIT}"  # 5 mg/5 ml
    rf"|\d+(?:[.,]\d+)?"  # bare quantity
    rf"|{_UNIT}"
    rf"|po|iv|im|sc|uống|tiêm|truyền|ngậm|bôi|nhỏ|đặt|xịt"  # route
    rf"|daily|bid|tid|qid|qd|hs|prn|khi\s+cần|mỗi\s+ngày|hàng\s+ngày"
    rf"|[x×]\s*\d+|\d+\s*lần(?:/\s*ngày)?|sáng|chiều|tối"  # frequency
    rf"|viên\s+nén|viên\s+nang|dung\s+dịch|hỗn\s+dịch|siro"  # formulation
    rf")\b",
    re.IGNORECASE,
)
#: Anything here ends the medication expression.
_HARD_STOP = re.compile(
    r"^\s*(?:[,;:.\n\r]|\d+\s*[.)]\s|-\s|\+\s|điều\s+trị|chỉ\s+định|do\s|vì\s|để\s|kèm\b)",
    re.IGNORECASE,
)
MAX_REGIMEN_TOKENS = 8


@dataclass(frozen=True, slots=True)
class SpanAlternative:
    """One selectable span. `text` is always `source[start:end]`."""

    text: str
    start: int
    end: int
    entity_type: str
    provenance: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "provenance": self.provenance,
        }


def _original(source: str, entity: dict[str, Any]) -> SpanAlternative:
    start, end = int(entity["position"][0]), int(entity["position"][1])
    return SpanAlternative(source[start:end], start, end, entity["type"], SPAN_ORIGINAL)


def drug_regimen_alternative(source: str, entity: dict[str, Any]) -> SpanAlternative | None:
    """Extend a medication mention rightward through its own regimen tail, or None."""
    start, end = int(entity["position"][0]), int(entity["position"][1])
    cursor = end
    absorbed = 0
    while absorbed < MAX_REGIMEN_TOKENS:
        remainder = source[cursor:]
        if not remainder.strip() or _HARD_STOP.match(remainder):
            break
        match = _REGIMEN_TOKEN.match(remainder)
        if not match:
            break
        cursor += match.end()
        absorbed += 1
    cursor = len(source[:cursor].rstrip())
    if cursor <= end:
        return None
    return SpanAlternative(source[start:cursor], start, cursor, entity["type"], SPAN_DRUG_REGIMEN)


def safe_bridge_alternative(source: str, entities: list[dict[str, Any]]) -> SpanAlternative | None:
    """Fragment completion across same-type neighbours, gated by the 0079 rule."""
    if len(entities) < 2:
        return None
    types = {e["type"] for e in entities}
    if len(types) != 1:
        return None
    offsets = [[int(e["position"][0]), int(e["position"][1])] for e in entities]
    if not evaluate_merge(source, offsets).accepted:
        return None
    start = min(o[0] for o in offsets)
    end = max(o[1] for o in offsets)
    return SpanAlternative(source[start:end], start, end, next(iter(types)), SPAN_SAFE_BRIDGE)


def alternatives_for(
    source: str,
    entity: dict[str, Any],
    *,
    joint_safe_span: bool,
    neighbours: list[dict[str, Any]] | None = None,
) -> list[SpanAlternative]:
    """The finite option set for one mention. `candidate_only` yields exactly the E3 span."""
    options = [_original(source, entity)]
    if not joint_safe_span:
        return options
    if entity["type"] == "THUỐC":
        extended = drug_regimen_alternative(source, entity)
        if extended is not None:
            options.append(extended)
    elif entity["type"] == "CHẨN_ĐOÁN" and neighbours:
        bridged = safe_bridge_alternative(source, [entity, *neighbours])
        if bridged is not None:
            options.append(bridged)
    seen: dict[tuple[int, int], SpanAlternative] = {}
    for option in options:
        assert source[option.start : option.end] == option.text
        seen.setdefault((option.start, option.end), option)
    return list(seen.values())


__all__ = [
    "MAX_REGIMEN_TOKENS",
    "SPAN_DRUG_REGIMEN",
    "SPAN_ORIGINAL",
    "SPAN_SAFE_BRIDGE",
    "SpanAlternative",
    "alternatives_for",
    "drug_regimen_alternative",
    "safe_bridge_alternative",
]
