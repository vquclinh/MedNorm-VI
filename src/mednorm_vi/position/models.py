"""Submission-position abstraction (Phase 1C-A).

The internal invariant ``original_text[start:end] == text`` is ALWAYS enforced on
raw code-point spans (L1 + proposal validation are unchanged). A ``PositionPolicy``
only describes how a raw span is *encoded* into a submission coordinate, and every
encoded position keeps reversible provenance back to the internal raw span.

No policy is asserted to be the organizer's global choice. The default internal
and diagnostic policy is ``raw-codepoint-half-open``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Coordinate spaces an encoder may target.
COORDINATE_SPACES: frozenset[str] = frozenset(
    {"codepoint", "utf8_byte", "utf16_unit", "normalized_codepoint"}
)
INTERVALS: frozenset[str] = frozenset({"half_open", "closed"})
LINE_ENDINGS: frozenset[str] = frozenset(
    {"raw", "lf_canonical", "crlf_canonical", "configurable"}
)


@dataclass(frozen=True, slots=True)
class PositionPolicy:
    """A registered, reversible mapping from a raw code-point span to a submission
    coordinate. Declarative metadata only; the encoder lives in ``encoders.py``.
    """

    policy_id: str
    title: str
    coordinate_space: str  # one of COORDINATE_SPACES
    interval: str  # half_open | closed
    line_ending: str  # raw | lf_canonical | crlf_canonical | configurable
    reversible: str  # "true" | "false" | "conditional"
    description: str = ""

    def __post_init__(self) -> None:
        if self.coordinate_space not in COORDINATE_SPACES:
            raise ValueError(f"unknown coordinate_space {self.coordinate_space!r}")
        if self.interval not in INTERVALS:
            raise ValueError(f"unknown interval {self.interval!r}")
        if self.line_ending not in LINE_ENDINGS:
            raise ValueError(f"unknown line_ending {self.line_ending!r}")


@dataclass(frozen=True, slots=True)
class PositionProvenance:
    """Reversible link from an encoded submission position back to the raw span."""

    policy_id: str
    coordinate_space: str
    raw_start: int  # code-point offset into original_text (authoritative)
    raw_end: int
    reversible: bool
    note: str = ""


@dataclass(frozen=True, slots=True)
class PositionEncodingResult:
    """An encoded submission position with provenance. ``text`` is unchanged — an
    encoder never edits entity text, only re-expresses coordinates.
    """

    policy_id: str
    coordinate_space: str
    interval: str
    start: int  # encoded start in the target coordinate space
    end: int  # encoded end (end-exclusive unless interval == "closed")
    text: str
    provenance: PositionProvenance

    @property
    def encoded_position(self) -> tuple[int, int]:
        return (self.start, self.end)


__all__ = [
    "COORDINATE_SPACES",
    "INTERVALS",
    "LINE_ENDINGS",
    "PositionPolicy",
    "PositionProvenance",
    "PositionEncodingResult",
]
