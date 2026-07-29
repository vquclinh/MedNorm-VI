"""Deterministic L4 resolver contracts (Phase 1C-A foundation).

The resolver consumes Phase 1B proposals + relations and produces
``EntityHypothesis`` objects: a chosen boundary per logical mention, retained
alternatives, an accepted/rejected/unresolved status, and boundary/type/overlap
evidence. It does NOT link to any ontology and never emits final organizer
entities. Heuristic scores are not calibrated probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# All five organizer-facing types are now resolvable when proposal evidence is
# present. Earlier deterministic experts still emit only medication/lab types.
RESOLVABLE_TYPES: frozenset[str] = frozenset(
    {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
)

# Resolution outcome for one hypothesis.
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class BoundaryAlternative:
    """One retained (non-chosen) boundary option for a hypothesis."""

    proposal_id: str
    start: int
    end: int
    text: str
    kind: str  # e.g. name_only / name_strength / full / value_only / value_unit


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    """Why the chosen boundary was chosen."""

    policy: str  # boundary policy applied
    chosen_kind: str
    chosen_proposal_id: str
    considered_kinds: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True, slots=True)
class TypeEvidence:
    """Why the entity type was assigned (single-type in Phase 1C-A)."""

    entity_type: str
    source_specialist: str
    proposed_types: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True, slots=True)
class OverlapDecision:
    """Record of a same-type overlap resolution touching this hypothesis."""

    outcome: str  # "winner" | "suppressed" | "abstained"
    counterpart_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class EntityHypothesis:
    """A resolved, typed span hypothesis (NOT a final organizer entity)."""

    hypothesis_id: str
    document_id: str
    start: int
    end: int
    text: str
    entity_type: str
    status: str  # accepted | rejected | unresolved
    chosen_proposal_id: str
    source_proposal_ids: tuple[str, ...]
    boundary_evidence: BoundaryEvidence
    type_evidence: TypeEvidence
    retained_alternatives: tuple[BoundaryAlternative, ...] = field(default_factory=tuple)
    overlap_decision: OverlapDecision | None = None
    rejection_reason: str | None = None
    has_result_pair_group_ids: tuple[str, ...] = field(default_factory=tuple)
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    # Which L3 experts proposed these exact coordinates (Audit 0052). Empty on the
    # legacy Phase-1B path; populated by the canonical lattice L4.
    expert_ids: tuple[str, ...] = field(default_factory=tuple)
    # Structured sub-span evidence carried through from E1/E2 (spec §10.1, §6.2).
    # L5 needs the field decomposition — ingredient/strength/unit/dose-form/route —
    # and it used to be destroyed at the lattice boundary. Kept as frozen mappings
    # so this contract does not depend on the specialist's dataclass.
    components: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    @property
    def position(self) -> tuple[int, int]:
        return (self.start, self.end)

    def components_by_role(self) -> dict[str, tuple[Mapping[str, Any], ...]]:
        """Structured sub-components grouped by grammar role (spec §10.1)."""
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for component in self.components:
            grouped.setdefault(str(component.get("role", "")), []).append(component)
        return {role: tuple(items) for role, items in sorted(grouped.items())}


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Everything the resolver produced for one document."""

    document_id: str
    hypotheses: tuple[EntityHypothesis, ...]
    config_version: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def accepted(self) -> tuple[EntityHypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.status == STATUS_ACCEPTED)

    def rejected(self) -> tuple[EntityHypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.status == STATUS_REJECTED)

    def unresolved(self) -> tuple[EntityHypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.status == STATUS_UNRESOLVED)


__all__ = [
    "RESOLVABLE_TYPES",
    "STATUS_ACCEPTED",
    "STATUS_REJECTED",
    "STATUS_UNRESOLVED",
    "BoundaryAlternative",
    "BoundaryEvidence",
    "TypeEvidence",
    "OverlapDecision",
    "EntityHypothesis",
    "ResolutionResult",
]
