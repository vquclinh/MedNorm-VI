"""Unified L3 span-lattice contracts (spec §6).

The Mention Factory produces **one** lattice from every expert. A lattice node is
a *span*, not an entity: it carries competing type scores from every expert that
proposed those exact coordinates, and L4 decides which type — if any — survives.
No expert ever emits a final organizer entity.

Three rules shape the data model, all from spec §4 / §5 (C7) / §7.3:

* ``original_text[start:end] == text`` for every node, always, end-exclusive;
* **never deduplicate by text alone** — identity is coordinates, so the same
  surface form at two offsets is two nodes (spec §5, case C7);
* exact-coordinate duplicates from different experts merge their *evidence* into
  one node while **every source is retained** individually for replay/ablation.

Normalization is stored beside the original, never over it: ``normalized_view``
is derived data and mutating ``text`` is impossible (frozen dataclasses).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schemas.constants import ENTITY_TYPES
from ..schemas.spans import Span

# Every architecture-declared L3 expert has a stable source id. Experts without a
# valid local checkpoint remain disabled by profile, but their proposal contract
# is still wired into the lattice so trained checkpoints can be evaluated without
# changing downstream L4 code.
EXPERT_MEDICATION_GRAMMAR = "E1_medication_grammar"
EXPERT_LABORATORY_PARSER = "E2_laboratory_parser"
EXPERT_VIHEALTHBERT = "E3_vihealthbert_span_type"
EXPERT_XLMR_MRC = "E5_xlmr_mrc_ner"
EXPERT_GLINER = "E6_gliner_open_type"
EXPERT_QWEN_PROPOSER = "E7_qwen3_1_7b_proposer"

AVAILABLE_EXPERTS: tuple[str, ...] = (
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_LABORATORY_PARSER,
    EXPERT_VIHEALTHBERT,
    EXPERT_XLMR_MRC,
    EXPERT_GLINER,
    EXPERT_QWEN_PROPOSER,
)

# E4 PhoBERT-W2NER is RETIRED_FROM_ACTIVE_ARCHITECTURE (Audits 0048, 0051). It is
# absent from AVAILABLE_EXPERTS, so a proposal that names it is refused by the same
# unknown-expert check as any typo — no separate code path, and no way to re-enter
# the lattice. ``governance.e4_retirement`` holds the record; the audits hold the
# evidence.
RETIRED_EXPERTS: tuple[str, ...] = ("E4_phobert_w2ner",)

# Provenance families, so deterministic and neural contributions stay separable in
# every report (Audit 0033 grouping).
FAMILY_DETERMINISTIC = "deterministic"
FAMILY_NEURAL = "neural"
FAMILY_OPEN_TYPE = "open_type"
FAMILY_LLM_INTERFACE = "llm_interface"

EXPERT_FAMILY: dict[str, str] = {
    EXPERT_MEDICATION_GRAMMAR: FAMILY_DETERMINISTIC,
    EXPERT_LABORATORY_PARSER: FAMILY_DETERMINISTIC,
    EXPERT_VIHEALTHBERT: FAMILY_NEURAL,
    EXPERT_XLMR_MRC: FAMILY_NEURAL,
    EXPERT_GLINER: FAMILY_OPEN_TYPE,
    EXPERT_QWEN_PROPOSER: FAMILY_LLM_INTERFACE,
}


class LatticeError(ValueError):
    """Raised when a proposal or lattice violates an invariant. Never repaired."""


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """One expert's complete contribution to a lattice node.

    Every field an expert knows is retained here rather than being averaged into
    the node, so an ablation can always ask "what did E1 alone say about this
    span?" and get an exact answer.
    """

    expert_id: str
    proposal_id: str
    local_score: float
    type_scores: Mapping[str, float]
    route: str = ""
    section: str = ""
    matched_rule: str = ""
    normalized_form: str = ""
    node_id: str = ""
    node_kind: str = ""
    parent_line_id: str = ""
    boundary_group_id: str = ""
    routes: tuple[str, ...] = field(default_factory=tuple)
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    rule_ids: tuple[str, ...] = field(default_factory=tuple)
    features: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    config_version: str = ""
    lexicon_version: str = ""
    model_revision: str = ""
    checkpoint_sha256: str = ""
    config_sha256: str = ""

    @property
    def family(self) -> str:
        return EXPERT_FAMILY.get(self.expert_id, FAMILY_DETERMINISTIC)

    def as_dict(self) -> dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "proposal_id": self.proposal_id,
            "family": self.family,
            "local_score": round(float(self.local_score), 6),
            "type_scores": {k: round(float(v), 6) for k, v in sorted(self.type_scores.items())},
            "route": self.route,
            "section": self.section,
            "matched_rule": self.matched_rule,
            "normalized_form": self.normalized_form,
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "parent_line_id": self.parent_line_id,
            "boundary_group_id": self.boundary_group_id,
            "routes": list(self.routes),
            "evidence_ids": list(self.evidence_ids),
            "rule_ids": list(self.rule_ids),
            "features": {k: round(float(v), 6) for k, v in sorted(self.features.items())},
            "warnings": list(self.warnings),
            "config_version": self.config_version,
            "lexicon_version": self.lexicon_version,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExpertSpanProposal:
    """Trainable L3 expert output before coordinate-identity lattice merge."""

    document_id: str
    start: int
    end: int
    text: str
    type_scores: Mapping[str, float]
    local_score: float
    expert_id: str
    proposal_id: str
    route: str = ""
    section: str = ""
    normalized_view: str = ""
    original_start: int | None = None
    original_end: int | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    features: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    config_version: str = ""
    model_revision: str = ""
    checkpoint_sha256: str = ""
    config_sha256: str = ""

    def __post_init__(self) -> None:
        if self.expert_id not in AVAILABLE_EXPERTS:
            raise LatticeError(f"unknown expert id: {self.expert_id}")
        if self.end <= self.start:
            raise LatticeError(f"invalid expert span offsets: {self.start}:{self.end}")
        if self.original_start is not None and self.original_start != self.start:
            raise LatticeError("expert proposal original_start must equal start")
        if self.original_end is not None and self.original_end != self.end:
            raise LatticeError("expert proposal original_end must equal end")
        if not self.proposal_id:
            raise LatticeError("expert proposal_id is required")
        for entity_type in self.type_scores:
            if entity_type not in ENTITY_TYPES:
                raise LatticeError(f"unsupported proposed type {entity_type!r}")

    def validate_against(self, original_text: str) -> None:
        if original_text[self.start:self.end] != self.text:
            raise LatticeError(
                f"{self.expert_id} proposed [{self.start}, {self.end}) whose text "
                "does not slice out of original_text (spec §4)"
            )

    def as_source_evidence(self) -> SourceEvidence:
        return SourceEvidence(
            expert_id=self.expert_id,
            proposal_id=self.proposal_id,
            local_score=float(self.local_score),
            type_scores=dict(self.type_scores),
            route=self.route,
            section=self.section,
            normalized_form=self.normalized_view,
            routes=(self.route,) if self.route else (),
            evidence_ids=tuple(self.evidence_ids),
            features=dict(self.features),
            warnings=tuple(self.warnings),
            config_version=self.config_version,
            model_revision=self.model_revision,
            checkpoint_sha256=self.checkpoint_sha256,
            config_sha256=self.config_sha256,
        )


@dataclass(frozen=True, slots=True)
class SpanProposal:
    """One node of the unified span lattice: a span plus all evidence for it.

    ``type_scores`` is the *merged* per-type evidence across sources (the maximum
    each expert assigned), which is what L4 scores. The per-expert originals stay
    in :attr:`sources`.
    """

    document_id: str
    start: int
    end: int
    text: str
    type_scores: Mapping[str, float]
    sources: tuple[SourceEvidence, ...]
    routes: tuple[str, ...] = field(default_factory=tuple)
    section: str = ""
    normalized_view: str = ""
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sources:
            raise LatticeError(
                f"lattice node [{self.start}, {self.end}) has no source expert; "
                "a node without provenance is never valid")
        for entity_type in self.type_scores:
            if entity_type not in ENTITY_TYPES:
                raise LatticeError(f"unsupported proposed type {entity_type!r}")

    # -- identity --------------------------------------------------------------
    @property
    def coordinates(self) -> tuple[int, int]:
        """The merge key. Coordinates only — never text (spec §5, case C7)."""
        return (self.start, self.end)

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return tuple(sorted({source.expert_id for source in self.sources}))

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({source.family for source in self.sources}))

    def local_score(self) -> float:
        """The strongest local score any expert gave this span."""
        return max(float(source.local_score) for source in self.sources)

    def score_for(self, entity_type: str) -> float:
        return float(self.type_scores.get(entity_type, 0.0))

    def best_type(self) -> str:
        """Highest-scoring type; deterministic on ties (alphabetical)."""
        if not self.type_scores:
            return ""
        return min(self.type_scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def sources_for(self, expert_id: str) -> tuple[SourceEvidence, ...]:
        return tuple(s for s in self.sources if s.expert_id == expert_id)

    def has_expert(self, expert_id: str) -> bool:
        return any(s.expert_id == expert_id for s in self.sources)

    def overlaps(self, other: SpanProposal) -> bool:
        return self.start < other.end and other.start < self.end

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "type_scores": {k: round(float(v), 6) for k, v in sorted(self.type_scores.items())},
            "routes": list(self.routes),
            "section": self.section,
            "expert_ids": list(self.expert_ids),
            "families": list(self.families),
            "features": {k: round(float(v), 6) for k, v in sorted(self.features.items())},
            "sources": [source.as_dict() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class SpanLattice:
    """Every span proposal for one document, in deterministic order.

    ``original_text`` is carried so any consumer can re-check the §4 invariant
    without trusting the producer.
    """

    document_id: str
    original_text: str
    proposals: tuple[SpanProposal, ...]
    routes_by_node: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    expert_ids: tuple[str, ...] = field(default_factory=tuple)
    merged_coordinate_groups: int = 0
    config_hash: str = ""

    def __len__(self) -> int:
        return len(self.proposals)

    def by_expert(self, expert_id: str) -> tuple[SpanProposal, ...]:
        return tuple(p for p in self.proposals if p.has_expert(expert_id))

    def by_family(self, family: str) -> tuple[SpanProposal, ...]:
        return tuple(p for p in self.proposals if family in p.families)

    def repeated_surface_forms(self) -> tuple[str, ...]:
        """Surface forms proposed at more than one offset — kept distinct, never merged."""
        offsets: dict[str, set[int]] = {}
        for proposal in self.proposals:
            offsets.setdefault(proposal.text, set()).add(proposal.start)
        return tuple(sorted(text for text, starts in offsets.items() if len(starts) > 1))

    def determinism_hash(self) -> str:
        """Stable digest of the whole lattice, for replay verification."""
        payload = json.dumps(
            [proposal.as_dict() for proposal in self.proposals],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        by_expert = {
            expert: len(self.by_expert(expert))
            for expert in self.expert_ids
        }
        return {
            "document_id": self.document_id,
            "proposal_count": len(self.proposals),
            "expert_ids": list(self.expert_ids),
            "proposals_by_expert": dict(sorted(by_expert.items())),
            "merged_coordinate_groups": self.merged_coordinate_groups,
            "repeated_surface_form_count": len(self.repeated_surface_forms()),
            "determinism_hash": self.determinism_hash(),
            "config_hash": self.config_hash,
            "warnings": list(self.warnings),
        }


def order_proposals(proposals: Sequence[SpanProposal]) -> tuple[SpanProposal, ...]:
    """Canonical lattice order: start, then end, then best type, then experts."""
    return tuple(sorted(
        proposals,
        key=lambda p: (p.start, p.end, p.best_type(), p.expert_ids)))


__all__ = [
    "RETIRED_EXPERTS",
    "AVAILABLE_EXPERTS",
    "EXPERT_FAMILY",
    "EXPERT_GLINER",
    "EXPERT_LABORATORY_PARSER",
    "EXPERT_MEDICATION_GRAMMAR",
    "EXPERT_QWEN_PROPOSER",
    "EXPERT_VIHEALTHBERT",
    "EXPERT_XLMR_MRC",
    "ExpertSpanProposal",
    "FAMILY_DETERMINISTIC",
    "FAMILY_LLM_INTERFACE",
    "FAMILY_NEURAL",
    "FAMILY_OPEN_TYPE",
    "LatticeError",
    "SourceEvidence",
    "SpanLattice",
    "SpanProposal",
    "order_proposals",
]
