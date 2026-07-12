"""Unified L3 proposal + relation contracts (Phase 1B).

Specialists (medication grammar, laboratory parser) emit ``SpanProposal`` and
``RelationProposal`` objects only — never final entities. Every proposal carries
exact absolute coordinates into ``original_text`` and rich provenance. This
refines the abstract ``schemas.hypotheses.SpanProposal`` concept with the fields
Phase 1B needs; it reuses ``schemas.spans.Span``.

Deterministic heuristic scores (``local_score``) are NOT calibrated
probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.spans import Span

# Internal relation type (never emitted to organizer output).
HAS_RESULT = "has_result"


@dataclass(frozen=True, slots=True)
class ComponentSpan:
    """A structured sub-component with exact absolute offsets (e.g. STRENGTH)."""

    role: str  # e.g. "name" | "strength_value" | "strength_unit" | "route" | ...
    start: int
    end: int
    text: str
    normalized: str | None = None
    detail: str | None = None

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)


@dataclass(frozen=True, slots=True)
class SpanProposal:
    """A candidate medical mention span from one specialist (proposal, not final).

    ``original_text[start:end] == text`` is a hard invariant (checked by
    ``mention_factory.validation`` / Phase 1B validation).
    """

    proposal_id: str
    document_id: str
    start: int
    end: int
    text: str
    proposed_types: tuple[str, ...]  # e.g. ("MEDICATION",) | ("TEST_NAME",)
    source_specialist: str  # "medication" | "laboratory"
    source_node_id: str
    source_routes: tuple[str, ...]  # e.g. ("C1",)
    local_score: float  # deterministic heuristic, NOT a probability
    matched_rule: str | None = None
    normalized_form: str | None = None
    parse_ref: str | None = None  # id of the MedicationParse / LaboratoryParse
    # Boundary alternatives of the SAME logical mention/value share this id, so a
    # relation on one alternative applies to the whole group (e.g. a lab result's
    # value-only and value+unit spans). ``None`` for standalone proposals.
    boundary_group_id: str | None = None
    source_node_kind: str | None = None  # L1 node kind that produced this (routing granularity)
    parent_line_id: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    components: tuple[ComponentSpan, ...] = field(default_factory=tuple)
    config_version: str = ""
    lexicon_version: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)
    features: dict[str, float] = field(default_factory=dict)

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)

    @property
    def position(self) -> tuple[int, int]:
        return (self.start, self.end)

    def identity(self) -> tuple[str, int, int, tuple[str, ...], str]:
        """Identity key: (document, start, end, proposed_types, specialist).

        Never text-only; repeated identical text at different positions differs.
        """
        return (self.document_id, self.start, self.end, self.proposed_types,
                self.source_specialist)


@dataclass(frozen=True, slots=True)
class RelationProposal:
    """An internal relation between two proposals (e.g. TEST_NAME has_result RESULT)."""

    relation_id: str
    document_id: str
    relation_type: str  # HAS_RESULT
    source_proposal_id: str  # the TEST_NAME proposal
    target_proposal_id: str  # a TEST_RESULT boundary alternative
    score: float
    pairing_cost: float
    # One LOGICAL name→result pairing produces one relation per result boundary
    # alternative, all sharing this ``pair_group_id``; ``is_primary`` marks the
    # value-only alternative. Count logical pairs by distinct pair_group_id.
    pair_group_id: str = ""
    is_primary: bool = True
    target_boundary_group_id: str | None = None
    source_node_id: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    provenance: str | None = None


@dataclass(frozen=True, slots=True)
class SpecialistRunResult:
    """Everything one specialist produced for a document."""

    specialist: str
    proposals: tuple[SpanProposal, ...] = field(default_factory=tuple)
    relations: tuple[RelationProposal, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "HAS_RESULT",
    "ComponentSpan",
    "SpanProposal",
    "RelationProposal",
    "SpecialistRunResult",
]
