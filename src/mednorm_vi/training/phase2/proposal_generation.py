"""Frozen L3 proposal dataset generation for learned-L4 v2 training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...lattice import ExpertSpanProposal, SpanLattice, build_span_lattice
from ...lattice.models import (
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_PHOBERT_W2NER,
    EXPERT_VIHEALTHBERT,
    EXPERT_XLMR_MRC,
    SourceEvidence,
)
from .common import (
    Phase2ReadinessError,
    canonical_json_sha256,
    privacy_safe_group_id,
    sha256_file,
    write_json,
    write_jsonl,
)

PROPOSAL_DATASET_SCHEMA_VERSION = "phase2-frozen-proposals-v1"
STATUS_AVAILABLE = "AVAILABLE"
STATUS_UNAVAILABLE_UNTRAINED = "UNAVAILABLE_UNTRAINED"
ALLOWED_PROPOSAL_SPLITS = {"train", "validation"}


class ProposalDatasetError(Phase2ReadinessError):
    """Raised when proposal generation would violate leakage or offset rules."""


@dataclass(frozen=True, slots=True)
class ProposalDocument:
    document_id: str
    original_text: str
    source_group: str
    split: str

    def validate(self) -> None:
        if self.split not in ALLOWED_PROPOSAL_SPLITS:
            raise ProposalDatasetError("proposal datasets may only use train or validation")
        if not self.document_id:
            raise ProposalDatasetError("proposal document_id is required")
        if not self.source_group:
            raise ProposalDatasetError("proposal source_group is required")


@dataclass(frozen=True, slots=True)
class FrozenExpertAvailability:
    expert_id: str
    status: str
    config_sha256: str
    checkpoint_sha256: str = ""
    model_revision: str = ""
    reason: str = ""

    def validate(self) -> None:
        if self.status not in {STATUS_AVAILABLE, STATUS_UNAVAILABLE_UNTRAINED}:
            raise ProposalDatasetError(f"unsupported expert status {self.status!r}")
        if self.status == STATUS_AVAILABLE and not self.checkpoint_sha256:
            raise ProposalDatasetError("available frozen expert missing checkpoint SHA-256")


@dataclass(frozen=True, slots=True)
class FrozenProposalRecord:
    document_id: str
    privacy_safe_group_id: str
    split: str
    start: int
    end: int
    text: str
    type_scores: Mapping[str, float]
    local_score: float
    expert_specific_scores: Mapping[str, float]
    route: str
    section: str
    boundary_alternatives: tuple[tuple[int, int], ...]
    grammar_features: Mapping[str, float]
    laboratory_features: Mapping[str, float]
    expert_agreement: Mapping[str, float]
    provenance: tuple[Mapping[str, Any], ...]
    model_revisions: Mapping[str, str]
    config_hashes: Mapping[str, str]
    checkpoint_hashes: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type_scores"] = dict(self.type_scores)
        payload["expert_specific_scores"] = dict(self.expert_specific_scores)
        payload["boundary_alternatives"] = [list(pair) for pair in self.boundary_alternatives]
        payload["grammar_features"] = dict(self.grammar_features)
        payload["laboratory_features"] = dict(self.laboratory_features)
        payload["expert_agreement"] = dict(self.expert_agreement)
        payload["provenance"] = [dict(item) for item in self.provenance]
        payload["model_revisions"] = dict(self.model_revisions)
        payload["config_hashes"] = dict(self.config_hashes)
        payload["checkpoint_hashes"] = dict(self.checkpoint_hashes)
        return payload


@dataclass(frozen=True, slots=True)
class FrozenProposalDatasetManifest:
    schema_version: str
    split: str
    config_sha256: str
    documents: int
    proposals: int
    lattice_hashes: Mapping[str, str]
    proposal_dataset_sha256: str
    expert_availability: tuple[FrozenExpertAvailability, ...]
    source_group_count: int
    internal_test_accessed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "config_sha256": self.config_sha256,
            "documents": self.documents,
            "proposals": self.proposals,
            "lattice_hashes": dict(self.lattice_hashes),
            "proposal_dataset_sha256": self.proposal_dataset_sha256,
            "expert_availability": [asdict(item) for item in self.expert_availability],
            "source_group_count": self.source_group_count,
            "internal_test_accessed": self.internal_test_accessed,
        }


@dataclass(frozen=True, slots=True)
class FrozenProposalDataset:
    split: str
    records: tuple[FrozenProposalRecord, ...]
    manifest: FrozenProposalDatasetManifest

    def determinism_hash(self) -> str:
        return canonical_json_sha256([record.as_dict() for record in self.records])


def _source_summary(source: SourceEvidence) -> dict[str, Any]:
    return {
        "expert_id": source.expert_id,
        "proposal_id": source.proposal_id,
        "local_score": float(source.local_score),
        "type_scores": dict(source.type_scores),
        "route": source.route,
        "section": source.section,
        "model_revision": source.model_revision,
        "config_sha256": source.config_sha256,
        "checkpoint_sha256": source.checkpoint_sha256,
        "features": dict(source.features),
    }


def _record_from_lattice_node(
    *,
    document: ProposalDocument,
    lattice: SpanLattice,
    node_index: int,
) -> FrozenProposalRecord:
    proposal = lattice.proposals[node_index]
    if lattice.original_text[proposal.start:proposal.end] != proposal.text:
        raise ProposalDatasetError("lattice proposal is not reversible to original offsets")
    competitors = tuple(
        (other.start, other.end)
        for other in lattice.proposals
        if other is not proposal and proposal.overlaps(other)
    )
    expert_scores = {
        source.expert_id: max(
            float(source.local_score),
            float(source.type_scores.get(proposal.best_type(), 0.0)),
        )
        for source in proposal.sources
    }
    source_experts = {source.expert_id for source in proposal.sources}
    agreement = {
        "expert_count": float(len(source_experts)),
        "source_count": float(len(proposal.sources)),
        "has_e1": float(EXPERT_MEDICATION_GRAMMAR in source_experts),
        "has_e2": float(EXPERT_LABORATORY_PARSER in source_experts),
        "has_e3": float(EXPERT_VIHEALTHBERT in source_experts),
        "has_e4": float(EXPERT_PHOBERT_W2NER in source_experts),
        "has_e5": float(EXPERT_XLMR_MRC in source_experts),
    }
    return FrozenProposalRecord(
        document_id=document.document_id,
        privacy_safe_group_id=privacy_safe_group_id(document.source_group),
        split=document.split,
        start=proposal.start,
        end=proposal.end,
        text=proposal.text,
        type_scores=dict(proposal.type_scores),
        local_score=proposal.local_score(),
        expert_specific_scores=expert_scores,
        route=proposal.routes[0] if proposal.routes else "",
        section=proposal.section,
        boundary_alternatives=tuple(sorted(set(competitors))),
        grammar_features={
            "grammar_completeness": max(
                (float(source.features.get("grammar_component_count", 0.0))
                 for source in proposal.sources),
                default=0.0,
            )
        },
        laboratory_features={
            "laboratory_structure": max(
                (float(source.features.get("lab_value_with_unit", 0.0))
                 for source in proposal.sources),
                default=0.0,
            )
        },
        expert_agreement=agreement,
        provenance=tuple(_source_summary(source) for source in proposal.sources),
        model_revisions={
            source.expert_id: source.model_revision
            for source in proposal.sources
            if source.model_revision
        },
        config_hashes={
            source.expert_id: source.config_sha256
            for source in proposal.sources
            if source.config_sha256
        },
        checkpoint_hashes={
            source.expert_id: source.checkpoint_sha256
            for source in proposal.sources
            if source.checkpoint_sha256
        },
    )


def build_frozen_proposal_dataset(
    documents: Sequence[ProposalDocument],
    proposals_by_document: Mapping[str, Sequence[ExpertSpanProposal]],
    *,
    split: str,
    config_sha256: str,
    expert_availability: Sequence[FrozenExpertAvailability],
) -> FrozenProposalDataset:
    if split not in ALLOWED_PROPOSAL_SPLITS:
        raise ProposalDatasetError("proposal generation split must be train or validation")
    for availability in expert_availability:
        availability.validate()
    records: list[FrozenProposalRecord] = []
    lattice_hashes: dict[str, str] = {}
    source_groups: set[str] = set()
    for document in sorted(documents, key=lambda item: item.document_id):
        document.validate()
        if document.split != split:
            raise ProposalDatasetError("all proposal documents must match requested split")
        source_groups.add(document.source_group)
        lattice = build_span_lattice(
            document.document_id,
            document.original_text,
            expert_spans=tuple(proposals_by_document.get(document.document_id, ())),
            config_hash=config_sha256,
        )
        lattice_hashes[document.document_id] = lattice.determinism_hash()
        for node_index in range(len(lattice.proposals)):
            records.append(
                _record_from_lattice_node(
                    document=document,
                    lattice=lattice,
                    node_index=node_index,
                )
            )
    ordered_records = tuple(
        sorted(records, key=lambda row: (row.document_id, row.start, row.end, row.text))
    )
    digest = hashlib.sha256()
    for record in ordered_records:
        digest.update(
            (
                json.dumps(
                    record.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    dataset_sha = digest.hexdigest()
    manifest = FrozenProposalDatasetManifest(
        schema_version=PROPOSAL_DATASET_SCHEMA_VERSION,
        split=split,
        config_sha256=config_sha256,
        documents=len(documents),
        proposals=len(ordered_records),
        lattice_hashes=lattice_hashes,
        proposal_dataset_sha256=dataset_sha,
        expert_availability=tuple(
            sorted(expert_availability, key=lambda item: item.expert_id)
        ),
        source_group_count=len(source_groups),
        internal_test_accessed=False,
    )
    return FrozenProposalDataset(split=split, records=ordered_records, manifest=manifest)


def write_frozen_proposal_dataset(
    root: str | Path,
    dataset: FrozenProposalDataset,
) -> dict[str, str]:
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = output_dir / "proposals.jsonl"
    manifest_path = output_dir / "proposal_manifest.json"
    write_jsonl(proposals_path, (record.as_dict() for record in dataset.records))
    write_json(manifest_path, dataset.manifest.as_dict())
    written_sha = sha256_file(proposals_path)
    if written_sha != dataset.manifest.proposal_dataset_sha256:
        raise ProposalDatasetError("written proposal dataset hash differs from manifest")
    return {
        "proposals_jsonl_sha256": written_sha,
        "proposal_manifest_sha256": sha256_file(manifest_path),
    }


def load_frozen_proposal_manifest(path: str | Path) -> FrozenProposalDatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProposalDatasetError("proposal manifest must be a JSON object")
    availability_raw = payload.get("expert_availability", [])
    if not isinstance(availability_raw, list):
        raise ProposalDatasetError("proposal manifest expert_availability must be a list")
    availability = tuple(
        FrozenExpertAvailability(
            expert_id=str(item["expert_id"]),
            status=str(item["status"]),
            config_sha256=str(item["config_sha256"]),
            checkpoint_sha256=str(item.get("checkpoint_sha256", "")),
            model_revision=str(item.get("model_revision", "")),
            reason=str(item.get("reason", "")),
        )
        for item in availability_raw
        if isinstance(item, Mapping)
    )
    return FrozenProposalDatasetManifest(
        schema_version=str(payload["schema_version"]),
        split=str(payload["split"]),
        config_sha256=str(payload["config_sha256"]),
        documents=int(payload["documents"]),
        proposals=int(payload["proposals"]),
        lattice_hashes={
            str(key): str(value)
            for key, value in dict(payload.get("lattice_hashes", {})).items()
        },
        proposal_dataset_sha256=str(payload["proposal_dataset_sha256"]),
        expert_availability=availability,
        source_group_count=int(payload["source_group_count"]),
        internal_test_accessed=bool(payload.get("internal_test_accessed", False)),
    )


__all__ = [
    "ALLOWED_PROPOSAL_SPLITS",
    "PROPOSAL_DATASET_SCHEMA_VERSION",
    "STATUS_AVAILABLE",
    "STATUS_UNAVAILABLE_UNTRAINED",
    "FrozenExpertAvailability",
    "FrozenProposalDataset",
    "FrozenProposalDatasetManifest",
    "FrozenProposalRecord",
    "ProposalDatasetError",
    "ProposalDocument",
    "build_frozen_proposal_dataset",
    "load_frozen_proposal_manifest",
    "write_frozen_proposal_dataset",
]
