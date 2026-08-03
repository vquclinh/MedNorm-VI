"""Reversible ICD-10 output serialization transforms (Audit 0060).

`UNRES-ICD-DOTTED` has been an unresolved organizer policy since the policy registry
was written: *"Should codes be emitted dotted (A09.9) or undotted (A099)?"*, confidence
`low`, `internal_default: unknown`, and `test_method: probe both formats for the same
diagnosis`. This module is that probe's instrument.

It is a **serialization** transform, not a KB change. The concept identity stays
`ontology + code + snapshot_id` exactly as `candidate-representation-schema-v1` defines
it; only the character form written into the submission moves. That distinction is the
whole reason this lives here rather than in `kb/`: nothing about the frozen snapshot,
the offered set or the L9 membership gate is touched.

**It fails closed.** A code that is neither a well-formed dotted nor a well-formed
dotless ICD-10 code is never guessed at, never "cleaned up", and never silently passed
through — it is reported and the transform refuses. A probe whose job is to isolate one
variable must not quietly introduce a second.

The transform is exactly reversible: `undot(dot(x)) == x` for every code it accepts,
which is what lets a probe result be attributed to the format and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CODE_FORMAT_VERSION = "icd-code-format-v1"

#: A dotless ICD-10 code: one letter, two digits, then optional subdivision.
_DOTLESS = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]*$")
#: A dotted ICD-10 code: the same, with a single dot after the three-character category.
_DOTTED = re.compile(r"^[A-Z][0-9]{2}\.[0-9A-Z]+$")

#: Length of the ICD-10 category prefix that the dot follows.
CATEGORY_LENGTH = 3


class MalformedIcdCode(ValueError):
    """Raised when a code is neither well-formed dotted nor well-formed dotless.

    Deliberately fatal rather than skip-and-continue: the probe's claim is that the
    *only* difference between two submissions is the dot. A code the transform does not
    understand would break that claim silently.
    """

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"malformed ICD-10 code {code!r}: {reason}")


def classify_icd_code(code: str) -> str:
    """``"dotted"``, ``"dotless"``, or ``"malformed"``."""
    if _DOTTED.match(code):
        return "dotted"
    if _DOTLESS.match(code):
        return "dotless"
    return "malformed"


def to_dotted(code: str) -> str:
    """Dotted form of an ICD-10 code.

    A three-character category has no subdivision to separate, so it is returned
    unchanged — ``N03`` is already its own dotted form. An already-dotted code is
    idempotent. Anything else raises.
    """
    kind = classify_icd_code(code)
    if kind == "dotted":
        return code
    if kind == "malformed":
        raise MalformedIcdCode(code, "not a well-formed dotted or dotless ICD-10 code")
    if len(code) <= CATEGORY_LENGTH:
        return code
    return f"{code[:CATEGORY_LENGTH]}.{code[CATEGORY_LENGTH:]}"


def to_dotless(code: str) -> str:
    """Dotless form of an ICD-10 code. Inverse of :func:`to_dotted`."""
    kind = classify_icd_code(code)
    if kind == "malformed":
        raise MalformedIcdCode(code, "not a well-formed dotted or dotless ICD-10 code")
    return code.replace(".", "")


@dataclass(frozen=True, slots=True)
class TransformReport:
    """What a transform did, in aggregate. Carries codes, never clinical text."""

    transformed: int = 0
    unchanged_category: int = 0
    unchanged_already_dotted: int = 0
    malformed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return (
            self.transformed
            + self.unchanged_category
            + self.unchanged_already_dotted
            + len(self.malformed)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "code_format_version": CODE_FORMAT_VERSION,
            "codes_seen": self.total,
            "transformed": self.transformed,
            "unchanged_three_char_category": self.unchanged_category,
            "unchanged_already_dotted": self.unchanged_already_dotted,
            "malformed": list(self.malformed),
        }


def transform_diagnosis_codes(
    codes: list[str], *, report: dict[str, int] | None = None
) -> tuple[list[str], TransformReport]:
    """Dot every code in one candidate list, preserving order and multiplicity.

    Order is preserved because candidate order is part of the submission contract; a
    reordering here would be a second variable in a one-variable experiment.
    """
    out: list[str] = []
    transformed = category = already = 0
    malformed: list[str] = []
    for code in codes:
        kind = classify_icd_code(code)
        if kind == "malformed":
            malformed.append(code)
            out.append(code)
            continue
        dotted = to_dotted(code)
        out.append(dotted)
        if kind == "dotted":
            already += 1
        elif dotted == code:
            category += 1
        else:
            transformed += 1
    if report is not None:
        report["transformed"] = report.get("transformed", 0) + transformed
    return out, TransformReport(transformed, category, already, tuple(malformed))


__all__ = [
    "CATEGORY_LENGTH",
    "CODE_FORMAT_VERSION",
    "MalformedIcdCode",
    "TransformReport",
    "classify_icd_code",
    "to_dotless",
    "to_dotted",
    "transform_diagnosis_codes",
]
