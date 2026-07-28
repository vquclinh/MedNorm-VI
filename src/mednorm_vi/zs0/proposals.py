"""ZS0 zero-shot mention proposals (Audit 0048).

Two pretrained proposal sources, held to the same offset invariant the whole
project runs on (spec §4): ``original_text[start:end] == text``, end-exclusive.

**GLiNER** returns offsets. They are verified against the original text and the
proposal is rejected outright if they disagree — a span that cannot be recovered
from the text it claims to come from is not evidence.

**Qwen** returns *strings only*. It is never trusted with offsets, and never
allowed to invent text: every returned mention must be a literal substring of the
segment it was shown, and its position is resolved here by deterministic exact
matching. A mention that occurs more than once is **rejected unless an anchor
identifies exactly one occurrence** — guessing which "sốt" was meant would be
fabricating evidence.

No clinical text is ever logged. Rejection reasons carry codes and offsets.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..lattice.models import EXPERT_GLINER, EXPERT_QWEN_PROPOSER, ExpertSpanProposal
from ..schemas.constants import ENTITY_TYPES

PROPOSAL_CONTRACT_VERSION = "zs0-proposal-v1"

# GLiNER is an open-type tagger; its label space is prompted, so the mapping into
# the five BTC types is ours and is recorded rather than assumed.
GLINER_LABEL_TO_TYPE: Mapping[str, str] = {
    "medication": "MEDICATION",
    "drug": "MEDICATION",
    "medicine": "MEDICATION",
    "disease": "DIAGNOSIS",
    "diagnosis": "DIAGNOSIS",
    "condition": "DIAGNOSIS",
    "symptom": "SYMPTOM",
    "sign": "SYMPTOM",
    "test": "TEST_NAME",
    "laboratory test": "TEST_NAME",
    "test name": "TEST_NAME",
    "test result": "TEST_RESULT",
    "laboratory result": "TEST_RESULT",
    "measurement": "TEST_RESULT",
}

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
    REJECT_NOT_A_SUBSTRING, REJECT_AMBIGUOUS_OCCURRENCE, REJECT_UNSUPPORTED_TYPE,
    REJECT_MALFORMED_JSON, REJECT_MALFORMED_RECORD, REJECT_OFFSET_MISMATCH,
    REJECT_EMPTY_SPAN, REJECT_OUT_OF_RANGE, REJECT_ANCHOR_NOT_FOUND,
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
            "total_rejected": self.total,
            "by_reason": {reason: self.counts.get(reason, 0)
                          for reason in REJECTION_REASONS},
            "contains_clinical_text": False,
        }


def _validate_span(text: str, start: int, end: int, claimed: str) -> None:
    if end <= start:
        raise ProposalRejected(REJECT_EMPTY_SPAN, f"{start}:{end}")
    if start < 0 or end > len(text):
        raise ProposalRejected(REJECT_OUT_OF_RANGE, f"{start}:{end}")
    if text[start:end] != claimed:
        raise ProposalRejected(REJECT_OFFSET_MISMATCH, f"{start}:{end}")


# ---------------------------------------------------------------------------
# GLiNER
# ---------------------------------------------------------------------------


def map_gliner_label(label: str) -> str:
    """Map a prompted GLiNER label into the five BTC types."""
    mapped = GLINER_LABEL_TO_TYPE.get(label.strip().lower())
    if mapped is None:
        raise ProposalRejected(REJECT_UNSUPPORTED_TYPE, label.strip().lower())
    return mapped


def gliner_proposal(
    *,
    document_id: str,
    original_text: str,
    start: int,
    end: int,
    label: str,
    score: float,
    ordinal: int,
    model_revision: str,
    checkpoint_sha256: str,
    route: str = "",
    section: str = "",
) -> ExpertSpanProposal:
    """One verified GLiNER span. Offsets are checked, never trusted."""
    entity_type = map_gliner_label(label)
    claimed = original_text[start:end] if 0 <= start < end <= len(original_text) else ""
    _validate_span(original_text, start, end, claimed)
    return ExpertSpanProposal(
        document_id=document_id, start=start, end=end, text=claimed,
        type_scores={entity_type: float(score)}, local_score=float(score),
        expert_id=EXPERT_GLINER, proposal_id=f"zs0-e6-{document_id}-{ordinal:04d}",
        route=route, section=section, original_start=start, original_end=end,
        features={"gliner_score": float(score)},
        config_version=PROPOSAL_CONTRACT_VERSION,
        model_revision=model_revision, checkpoint_sha256=checkpoint_sha256)


def convert_gliner_spans(
    *,
    document_id: str,
    original_text: str,
    raw_spans: Sequence[Mapping[str, Any]],
    model_revision: str,
    checkpoint_sha256: str,
    ledger: RejectionLedger | None = None,
) -> tuple[tuple[ExpertSpanProposal, ...], RejectionLedger]:
    """Convert a GLiNER batch, rejecting anything that fails the invariant."""
    ledger = ledger or RejectionLedger()
    accepted: list[ExpertSpanProposal] = []
    for ordinal, raw in enumerate(raw_spans, start=1):
        try:
            accepted.append(gliner_proposal(
                document_id=document_id, original_text=original_text,
                start=int(raw["start"]), end=int(raw["end"]),
                label=str(raw["label"]), score=float(raw.get("score", 0.0)),
                ordinal=ordinal, model_revision=model_revision,
                checkpoint_sha256=checkpoint_sha256))
        except ProposalRejected as rejected:
            ledger.record(rejected.reason)
        except (KeyError, TypeError, ValueError):
            ledger.record(REJECT_MALFORMED_RECORD)
    return tuple(accepted), ledger


# ---------------------------------------------------------------------------
# Qwen: strings in, deterministic offsets out
# ---------------------------------------------------------------------------

QWEN_PROPOSAL_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["mentions"],
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "type"],
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(ENTITY_TYPES)},
                    "anchor": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


def resolve_occurrence(
    segment: str, mention: str, *, anchor: str = "",
) -> tuple[int, int]:
    """Deterministic exact-substring resolution, or a refusal.

    A single occurrence resolves. Several occurrences resolve **only** if an
    anchor — a longer literal substring containing the mention — occurs exactly
    once and contains exactly one copy of it. Otherwise this fails closed:
    picking the first "sốt" out of three would be inventing which one the model
    meant.
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


def parse_qwen_proposals(payload: str) -> tuple[Mapping[str, Any], ...]:
    """Parse strict JSON. Anything else is rejected, not repaired."""
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as error:
        raise ProposalRejected(REJECT_MALFORMED_JSON, type(error).__name__) from error
    if not isinstance(document, dict) or "mentions" not in document:
        raise ProposalRejected(REJECT_MALFORMED_JSON, "missing 'mentions'")
    mentions = document["mentions"]
    if not isinstance(mentions, list):
        raise ProposalRejected(REJECT_MALFORMED_JSON, "'mentions' is not a list")
    records: list[Mapping[str, Any]] = []
    for item in mentions:
        if not isinstance(item, dict):
            raise ProposalRejected(REJECT_MALFORMED_RECORD, "not an object")
        records.append(item)
    return tuple(records)


def qwen_proposals(
    *,
    document_id: str,
    original_text: str,
    segment_start: int,
    segment_text: str,
    payload: str,
    model_revision: str,
    checkpoint_sha256: str,
    route: str = "",
    section: str = "",
    ledger: RejectionLedger | None = None,
) -> tuple[tuple[ExpertSpanProposal, ...], RejectionLedger]:
    """Convert one Qwen response into verified proposals.

    ``segment_start`` is the segment's offset inside the full document, so the
    resolved position is translated back into document coordinates and then
    re-verified against the *document* text — not merely against the segment.
    """
    ledger = ledger or RejectionLedger()
    try:
        records = parse_qwen_proposals(payload)
    except ProposalRejected as rejected:
        ledger.record(rejected.reason)
        return (), ledger

    accepted: list[ExpertSpanProposal] = []
    for ordinal, record in enumerate(records, start=1):
        try:
            mention = record.get("text")
            entity_type = record.get("type")
            if not isinstance(mention, str) or not isinstance(entity_type, str):
                raise ProposalRejected(REJECT_MALFORMED_RECORD)
            if entity_type not in ENTITY_TYPES:
                raise ProposalRejected(REJECT_UNSUPPORTED_TYPE, entity_type)
            anchor = record.get("anchor") or ""
            if not isinstance(anchor, str):
                raise ProposalRejected(REJECT_MALFORMED_RECORD, "anchor is not a string")
            local_start, local_end = resolve_occurrence(
                segment_text, mention, anchor=anchor)
            start = segment_start + local_start
            end = segment_start + local_end
            # Re-verify in DOCUMENT coordinates; a correct segment offset that
            # lands wrong in the document is still wrong.
            _validate_span(original_text, start, end, mention)
            accepted.append(ExpertSpanProposal(
                document_id=document_id, start=start, end=end, text=mention,
                type_scores={entity_type: 1.0}, local_score=1.0,
                expert_id=EXPERT_QWEN_PROPOSER,
                proposal_id=f"zs0-e7-{document_id}-{ordinal:04d}",
                route=route, section=section, original_start=start, original_end=end,
                features={"qwen_anchor_used": float(bool(anchor))},
                config_version=PROPOSAL_CONTRACT_VERSION,
                model_revision=model_revision, checkpoint_sha256=checkpoint_sha256))
        except ProposalRejected as rejected:
            ledger.record(rejected.reason)
        except (KeyError, TypeError, ValueError):
            ledger.record(REJECT_MALFORMED_RECORD)
    return tuple(accepted), ledger


__all__ = [
    "GLINER_LABEL_TO_TYPE",
    "PROPOSAL_CONTRACT_VERSION",
    "QWEN_PROPOSAL_SCHEMA",
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
    "convert_gliner_spans",
    "gliner_proposal",
    "map_gliner_label",
    "parse_qwen_proposals",
    "qwen_proposals",
    "resolve_occurrence",
]
