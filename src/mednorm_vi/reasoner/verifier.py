"""E3-constrained Qwen3-8B verifier (variant F, sprint 0076).

The free-form 8B run over-extracted badly - 1,644 entities against E3's 1,147 - and scored
11.0782 with WER 86.0958. The model was allowed to write spans, so it wrote too many.

Here it cannot. The model receives the note plus a finite, numbered list of E3 proposals and
returns decisions keyed **only by proposal id**. It never emits text and never emits offsets:
those are copied from the E3 proposal it is judging. A hallucinated span is therefore not
"filtered out", it is unrepresentable.

Everything else fails closed. An unknown id, a duplicate id, an unknown type or a non-boolean
assertion invalidates that decision, and an invalid decision means the E3 proposal is kept
unchanged - never dropped, never invented.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .validator import ASSERTION_KEYS, ASSERTION_TYPES, CANDIDATE_TYPES, ORGANIZER_TYPES

VERIFIER_VERSION = "e3-constrained-verifier-v1"

#: Why a model decision was discarded. The E3 proposal survives unchanged in every case.
INVALID_UNKNOWN_ID = "unknown_proposal_id"
INVALID_DUPLICATE_ID = "duplicate_proposal_id"
INVALID_UNKNOWN_TYPE = "unknown_type"
INVALID_ASSERTION_VALUE = "non_boolean_assertion"
INVALID_MALFORMED = "malformed_decision"

#: Deterministic fallback when the model's reply cannot be parsed at all. Keeping the E3
#: proposals unchanged reproduces the configuration that currently scores best (14.3749); it
#: is the one behaviour we know the leaderboard rewards, and it invents nothing.
FALLBACK_KEEP_E3 = "keep_e3_unchanged"


@dataclass
class VerifierDiagnostics:
    """Everything needed to interpret the leaderboard result afterwards."""

    documents: int = 0
    proposals_in: int = 0
    kept: int = 0
    dropped: int = 0
    type_changes: int = 0
    assertion_changes: int = 0
    invalid_decisions: Counter[str] = field(default_factory=Counter)
    parse_fallbacks: int = 0
    keep_by_type: Counter[str] = field(default_factory=Counter)
    drop_by_type: Counter[str] = field(default_factory=Counter)
    type_change_pairs: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verifier_version": VERIFIER_VERSION,
            "documents": self.documents,
            "e3_proposals_in": self.proposals_in,
            "kept": self.kept,
            "dropped": self.dropped,
            "type_changes": self.type_changes,
            "assertion_changes": self.assertion_changes,
            "invalid_decisions": dict(sorted(self.invalid_decisions.items())),
            "parse_fallbacks": self.parse_fallbacks,
            "keep_by_type": dict(sorted(self.keep_by_type.items())),
            "drop_by_type": dict(sorted(self.drop_by_type.items())),
            "type_change_pairs": dict(sorted(self.type_change_pairs.items())),
            "candidate_policy": "forced_all_null",
        }


def _assertion_list(flags: Any, entity_type: str) -> tuple[list[str] | None, str | None]:
    """(serialized assertions, invalid reason). Lab types serialize nothing."""
    if entity_type not in ASSERTION_TYPES:
        return None, None
    if flags is None:
        return [], None
    if not isinstance(flags, dict):
        return None, INVALID_ASSERTION_VALUE
    out: list[str] = []
    for key in ASSERTION_KEYS:
        value = flags.get(key, False)
        if not isinstance(value, bool):
            return None, INVALID_ASSERTION_VALUE
        if value:
            out.append(key)
    return out, None


def apply_decisions(
    proposals: list[dict[str, Any]],
    decisions: list[dict[str, Any]] | None,
    diagnostics: VerifierDiagnostics,
) -> list[dict[str, Any]]:
    """Project model decisions onto E3 proposals. Text and position are never model-derived.

    ``proposals`` are organizer-shaped E3 entities in document order; their index IS the
    proposal id the model was shown. ``decisions is None`` means the reply could not be
    parsed, which triggers the documented fallback.
    """
    diagnostics.documents += 1
    diagnostics.proposals_in += len(proposals)

    if decisions is None:
        diagnostics.parse_fallbacks += 1
        decisions = []

    by_id: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    for raw in decisions:
        if not isinstance(raw, dict):
            diagnostics.invalid_decisions[INVALID_MALFORMED] += 1
            continue
        identifier = raw.get("id")
        if not isinstance(identifier, int) or not 0 <= identifier < len(proposals):
            diagnostics.invalid_decisions[INVALID_UNKNOWN_ID] += 1
            continue
        if identifier in seen:
            diagnostics.invalid_decisions[INVALID_DUPLICATE_ID] += 1
            continue
        seen.add(identifier)
        by_id[identifier] = raw

    out: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals):
        original_type = str(proposal.get("type", ""))
        decision = by_id.get(index)

        entity_type = original_type
        keep = True
        flags: Any = None
        if decision is not None:
            proposed_type = decision.get("type", original_type)
            if proposed_type not in ORGANIZER_TYPES:
                diagnostics.invalid_decisions[INVALID_UNKNOWN_TYPE] += 1
            else:
                entity_type = str(proposed_type)
            keep = bool(decision.get("keep", True))
            flags = decision.get("assertions")

        assertions, invalid = _assertion_list(flags, entity_type)
        if invalid:
            diagnostics.invalid_decisions[invalid] += 1
            assertions, _ = _assertion_list(None, entity_type)

        if not keep:
            diagnostics.dropped += 1
            diagnostics.drop_by_type[original_type] += 1
            continue

        # Text and position come from the E3 proposal, always.
        row: dict[str, Any] = {
            "text": proposal["text"],
            "type": entity_type,
            "position": proposal["position"],
        }
        if entity_type in ASSERTION_TYPES:
            before = list(proposal.get("assertions") or [])
            row["assertions"] = assertions if assertions is not None else before
            if sorted(row["assertions"]) != sorted(before):
                diagnostics.assertion_changes += 1
        if entity_type in CANDIDATE_TYPES:
            row["candidates"] = []  # variant F preserves the measured all-null policy

        if entity_type != original_type:
            diagnostics.type_changes += 1
            diagnostics.type_change_pairs[f"{original_type}->{entity_type}"] += 1
        diagnostics.kept += 1
        diagnostics.keep_by_type[entity_type] += 1
        out.append(row)
    return out


__all__ = [
    "FALLBACK_KEEP_E3",
    "INVALID_ASSERTION_VALUE",
    "INVALID_DUPLICATE_ID",
    "INVALID_MALFORMED",
    "INVALID_UNKNOWN_ID",
    "INVALID_UNKNOWN_TYPE",
    "VERIFIER_VERSION",
    "VerifierDiagnostics",
    "apply_decisions",
]
