"""Offset verification and substring resolution for L3 proposals (spec §4, §6).

Spec §4 states the one invariant the whole system rests on:

    assert original_text[start:end] == entity["text"]

and it adds that "an LLM must never calculate offsets freely" (spec §1). Those two
sentences produce the two functions here, both expert-independent:

:func:`verify_span`
    the invariant, applied to any span from any source. A tagger that returns its
    own offsets is checked against the original text and rejected outright when
    they disagree — a span that cannot be recovered from the text it claims to
    come from is not evidence.

:func:`resolve_occurrence`
    deterministic position recovery for a source that returns **strings only**.
    A single occurrence resolves. Several occurrences resolve only if an anchor — a
    longer literal substring — occurs exactly once and contains exactly one copy of
    the mention. Otherwise it fails closed: picking the first of three identical
    surface forms would fabricate which one was meant, and spec §5 case C7 exists
    precisely because repeated surface forms at different offsets are different
    entities.

Rejections are counted by reason code in a :class:`RejectionLedger`. The ledger
never holds clinical text; a reason code and an offset pair carry everything a
diagnosis needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

OFFSET_CONTRACT_VERSION = "mention-offsets-v1"

# Rejection reason codes. Counted and reported; never accompanied by the text.
REJECT_NOT_A_SUBSTRING = "not_a_literal_substring_of_the_segment"
REJECT_AMBIGUOUS_OCCURRENCE = "repeated_substring_without_a_disambiguating_anchor"
REJECT_UNSUPPORTED_TYPE = "unsupported_entity_type"
REJECT_MALFORMED_JSON = "malformed_json"
REJECT_MALFORMED_RECORD = "malformed_proposal_record"
REJECT_OFFSET_MISMATCH = "offsets_do_not_reproduce_the_claimed_text"
REJECT_EMPTY_SPAN = "empty_or_inverted_span"
REJECT_OUT_OF_RANGE = "span_outside_the_document"
REJECT_ANCHOR_NOT_FOUND = "anchor_not_found_in_the_segment"

REJECTION_REASONS: tuple[str, ...] = (
    REJECT_NOT_A_SUBSTRING,
    REJECT_AMBIGUOUS_OCCURRENCE,
    REJECT_UNSUPPORTED_TYPE,
    REJECT_MALFORMED_JSON,
    REJECT_MALFORMED_RECORD,
    REJECT_OFFSET_MISMATCH,
    REJECT_EMPTY_SPAN,
    REJECT_OUT_OF_RANGE,
    REJECT_ANCHOR_NOT_FOUND,
)


class ProposalRejected(ValueError):
    """Raised with a reason code when a proposal cannot be accepted."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}{f': {detail}' if detail else ''}")


@dataclass
class RejectionLedger:
    """Counts by reason code. Never holds clinical text."""

    counts: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str) -> None:
        if reason not in REJECTION_REASONS:
            raise ValueError(f"unknown rejection reason {reason!r}")
        self.counts[reason] = self.counts.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset_contract_version": OFFSET_CONTRACT_VERSION,
            "total_rejected": self.total,
            "by_reason": {
                reason: self.counts.get(reason, 0) for reason in REJECTION_REASONS
            },
            "contains_clinical_text": False,
        }


def verify_span(text: str, start: int, end: int, claimed: str) -> None:
    """Spec §4's invariant, with a reason code for each way it can fail."""
    if end <= start:
        raise ProposalRejected(REJECT_EMPTY_SPAN, f"{start}:{end}")
    if start < 0 or end > len(text):
        raise ProposalRejected(REJECT_OUT_OF_RANGE, f"{start}:{end}")
    if text[start:end] != claimed:
        raise ProposalRejected(REJECT_OFFSET_MISMATCH, f"{start}:{end}")


def resolve_occurrence(
    segment: str, mention: str, *, anchor: str = ""
) -> tuple[int, int]:
    """Deterministic exact-substring resolution, or a refusal.

    Used for any source that returns text without offsets. Fails closed on an
    ambiguous surface form rather than choosing an occurrence.
    """
    if not mention:
        raise ProposalRejected(REJECT_NOT_A_SUBSTRING, "empty mention")
    occurrences = [m.start() for m in re.finditer(re.escape(mention), segment)]
    if not occurrences:
        raise ProposalRejected(REJECT_NOT_A_SUBSTRING)
    if len(occurrences) == 1:
        start = occurrences[0]
        return start, start + len(mention)

    if not anchor:
        raise ProposalRejected(
            REJECT_AMBIGUOUS_OCCURRENCE, f"{len(occurrences)} occurrences")
    anchor_positions = [m.start() for m in re.finditer(re.escape(anchor), segment)]
    if len(anchor_positions) != 1:
        raise ProposalRejected(
            REJECT_ANCHOR_NOT_FOUND, f"{len(anchor_positions)} anchor matches")
    anchor_start = anchor_positions[0]
    inner = [m.start() for m in re.finditer(re.escape(mention), anchor)]
    if len(inner) != 1:
        raise ProposalRejected(
            REJECT_AMBIGUOUS_OCCURRENCE, f"{len(inner)} occurrences inside the anchor")
    start = anchor_start + inner[0]
    return start, start + len(mention)


def resolve_in_document(
    *,
    original_text: str,
    segment_start: int,
    segment_text: str,
    mention: str,
    anchor: str = "",
) -> tuple[int, int]:
    """Resolve inside a segment, then re-verify in **document** coordinates.

    A correct segment-relative offset that lands wrong in the document is still
    wrong, so the invariant is applied against the document text, not the segment.
    """
    local_start, local_end = resolve_occurrence(segment_text, mention, anchor=anchor)
    start = segment_start + local_start
    end = segment_start + local_end
    verify_span(original_text, start, end, mention)
    return start, end


__all__ = [
    "OFFSET_CONTRACT_VERSION",
    "REJECTION_REASONS",
    "REJECT_AMBIGUOUS_OCCURRENCE",
    "REJECT_ANCHOR_NOT_FOUND",
    "REJECT_EMPTY_SPAN",
    "REJECT_MALFORMED_JSON",
    "REJECT_MALFORMED_RECORD",
    "REJECT_NOT_A_SUBSTRING",
    "REJECT_OFFSET_MISMATCH",
    "REJECT_OUT_OF_RANGE",
    "REJECT_UNSUPPORTED_TYPE",
    "ProposalRejected",
    "RejectionLedger",
    "resolve_in_document",
    "resolve_occurrence",
    "verify_span",
]
