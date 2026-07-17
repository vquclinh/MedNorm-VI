"""Public NER dataset manifests + label-mapping contracts (Phase 1C-A).

A foundation only: no dataset is downloaded and no training pipeline is built.
These contracts let us record — auditably and versioned — a dataset's labels,
their proposed mapping into MedNorm-VI's label space, boundary/assertion/coding
availability, redistribution/privacy constraints, split provenance, and leakage
risk, BEFORE any dataset is used. License terms are never claimed automatically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas.constants import ENTITY_TYPES
from ..validator.results import Severity, ValidationIssue, ValidationResult
from .models import (
    LICENSE_STATUSES,
    REDISTRIBUTION_PERMISSIONS,
    USABLE_STATUSES,
    LicenseRecord,
    RedistributionPolicy,
    SourceRecord,
)

# A source label maps to a MedNorm entity type, or to an explicit action.
MAPPING_ACTIONS: frozenset[str] = frozenset({"DROP", "REVIEW"})
MAPPING_REVIEW_STATES: frozenset[str] = frozenset(
    {
        "exact",
        "partial",
        "ambiguous",
        "auxiliary_only",
        "unmappable",
        "requires_human_review",
    }
)


@dataclass(frozen=True, slots=True)
class LabelMapping:
    """source label -> canonical intermediate -> MedNorm target (or DROP/REVIEW)."""

    source_label: str
    canonical_intermediate: str
    target: str  # a MedNorm ENTITY_TYPES value, or "DROP" / "REVIEW"
    note: str = ""
    mapping_status: str = "requires_human_review"

    @property
    def is_action(self) -> bool:
        return self.target in MAPPING_ACTIONS


@dataclass(frozen=True, slots=True)
class LabelMappingSet:
    """A versioned, auditable set of label mappings for one dataset."""

    dataset_id: str
    version: int
    mappings: tuple[LabelMapping, ...]
    config_hash: str = ""


@dataclass(frozen=True, slots=True)
class NerDatasetManifest:
    """Governance + NER-specific description of an external NER dataset."""

    dataset_id: str
    title: str
    source: SourceRecord
    license: LicenseRecord
    redistribution: RedistributionPolicy
    local_root: str = ""
    source_revision: str = ""
    source_snapshot_id: str = ""
    tree_hash_sha256: str = ""
    file_count: int | None = None
    total_bytes: int | None = None
    splits: tuple[str, ...] = field(default_factory=tuple)
    formats: tuple[str, ...] = field(default_factory=tuple)
    label_inventory_hash: str = ""
    schema_version: str = ""
    original_labels: tuple[str, ...] = field(default_factory=tuple)
    label_mappings: tuple[LabelMapping, ...] = field(default_factory=tuple)
    incompatible_labels: tuple[str, ...] = field(default_factory=tuple)
    boundary_convention: str = ""  # differences vs MedNorm-VI boundary policy
    assertion_availability: str = "none"  # none | partial | full
    normalization_availability: str = "none"  # none | icd | rxnorm | both
    redistribution_constraints: str = ""
    privacy_considerations: str = ""
    split_provenance: str = ""
    leakage_risk: str = "unknown"  # none | low | medium | high | unknown
    permitted_use: tuple[str, ...] = field(default_factory=tuple)
    intended_mednorm_use: tuple[str, ...] = field(default_factory=tuple)
    prohibited_uses: tuple[str, ...] = field(default_factory=tuple)
    attribution_requirements: str = ""
    governance_status: str = "REVIEW_REQUIRED"
    readiness_status: str = "not_ready"
    review_status: str = "pending"
    reviewer: str = ""
    manifest_version: int = 1


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _mapping(d: dict[str, Any]) -> LabelMapping:
    return LabelMapping(
        source_label=str(d["source_label"]),
        canonical_intermediate=str(d.get("canonical_intermediate", "")),
        target=str(d["target"]),
        note=str(d.get("note", "")),
        mapping_status=str(d.get("mapping_status", "requires_human_review")),
    )


def _opt_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ner_manifest_from_mapping(d: dict[str, Any]) -> NerDatasetManifest:
    src = d.get("source", {}) or {}
    lic = d.get("license", {}) or {}
    red = d.get("redistribution", {}) or {}
    return NerDatasetManifest(
        dataset_id=str(d.get("dataset_id", "")), title=str(d.get("title", "")),
        source=SourceRecord(organization=str(src.get("organization", "")),
                            url=str(src.get("url", "")),
                            description=str(src.get("description", ""))),
        license=LicenseRecord(name=str(lic.get("name", "")),
                             status=str(lic.get("status", "LICENSE_UNKNOWN")),
                             text_reference=str(lic.get("text_reference", "")),
                             permits_redistribution=lic.get("permits_redistribution"),
                             notes=str(lic.get("notes", ""))),
        redistribution=RedistributionPolicy(
            permission=str(red.get("permission", "unknown")),
            organizer_submission_implications=str(
                red.get("organizer_submission_implications", "")),
            notes=str(red.get("notes", ""))),
        local_root=str(d.get("local_root", "")),
        source_revision=str(d.get("source_revision", "")),
        source_snapshot_id=str(d.get("source_snapshot_id", "")),
        tree_hash_sha256=str(d.get("tree_hash_sha256", "")),
        file_count=_opt_int(d.get("file_count")),
        total_bytes=_opt_int(d.get("total_bytes")),
        splits=tuple(str(x) for x in d.get("splits", []) or []),
        formats=tuple(str(x) for x in d.get("formats", []) or []),
        label_inventory_hash=str(d.get("label_inventory_hash", "")),
        schema_version=str(d.get("schema_version", "")),
        original_labels=tuple(str(x) for x in d.get("original_labels", []) or []),
        label_mappings=tuple(_mapping(m) for m in d.get("label_mappings", []) or []),
        incompatible_labels=tuple(str(x) for x in d.get("incompatible_labels", []) or []),
        boundary_convention=str(d.get("boundary_convention", "")),
        assertion_availability=str(d.get("assertion_availability", "none")),
        normalization_availability=str(d.get("normalization_availability", "none")),
        redistribution_constraints=str(d.get("redistribution_constraints", "")),
        privacy_considerations=str(d.get("privacy_considerations", "")),
        split_provenance=str(d.get("split_provenance", "")),
        leakage_risk=str(d.get("leakage_risk", "unknown")),
        permitted_use=tuple(str(x) for x in d.get("permitted_use", []) or []),
        intended_mednorm_use=tuple(str(x) for x in d.get("intended_mednorm_use", []) or []),
        prohibited_uses=tuple(str(x) for x in d.get("prohibited_uses", []) or []),
        attribution_requirements=str(d.get("attribution_requirements", "")),
        governance_status=str(d.get("governance_status", "REVIEW_REQUIRED")),
        readiness_status=str(d.get("readiness_status", "not_ready")),
        review_status=str(d.get("review_status", "pending")),
        reviewer=str(d.get("reviewer", "")),
        manifest_version=int(d.get("manifest_version", 1)))


def load_ner_manifest(path: str | Path) -> NerDatasetManifest:
    import yaml  # type: ignore[import-untyped]

    p = Path(path)
    return ner_manifest_from_mapping(yaml.safe_load(p.read_text(encoding="utf-8")) or {})


def label_mapping_set(dataset_id: str, version: int,
                      mappings: list[LabelMapping]) -> LabelMappingSet:
    payload = [(m.source_label, m.canonical_intermediate, m.target) for m in mappings]
    return LabelMappingSet(dataset_id=dataset_id, version=version,
                           mappings=tuple(mappings),
                           config_hash=_hash({"dataset": dataset_id, "v": version,
                                              "m": payload}))


def validate_ner_manifest(m: NerDatasetManifest) -> ValidationResult:
    """Governance completeness for an NER dataset manifest (no legal conclusion)."""
    issues: list[ValidationIssue] = []
    did = m.dataset_id or "<no id>"

    def err(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, f"{did}: {msg}"))

    def warn(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, f"{did}: {msg}", severity=Severity.WARNING))

    if not m.dataset_id:
        err("ner.missing_id", "missing dataset_id")
    if not (m.source.organization or m.source.url):
        err("ner.missing_source", "missing source")
    if m.license.status not in LICENSE_STATUSES:
        err("ner.bad_license_status",
            f"license.status {m.license.status!r} not in {sorted(LICENSE_STATUSES)}")
    if m.redistribution.permission not in REDISTRIBUTION_PERMISSIONS:
        err("ner.bad_redistribution",
            f"redistribution.permission {m.redistribution.permission!r} invalid")
    if m.tree_hash_sha256 and not (
        len(m.tree_hash_sha256) == 64
        and all(c in "0123456789abcdefABCDEF" for c in m.tree_hash_sha256)
    ):
        err("ner.bad_tree_hash", "tree_hash_sha256 is not SHA-256")
    if m.file_count is not None and m.file_count <= 0:
        err("ner.bad_file_count", "file_count must be positive")
    if m.total_bytes is not None and m.total_bytes <= 0:
        err("ner.bad_total_bytes", "total_bytes must be positive")
    if m.label_inventory_hash and not (
        len(m.label_inventory_hash) == 64
        and all(c in "0123456789abcdefABCDEF" for c in m.label_inventory_hash)
    ):
        err("ner.bad_label_inventory_hash", "label_inventory_hash is not SHA-256")
    if not m.original_labels:
        warn("ner.no_labels", "no original_labels declared")
    valid_targets = set(ENTITY_TYPES) | MAPPING_ACTIONS
    for lm in m.label_mappings:
        if lm.target not in valid_targets:
            err("ner.bad_mapping_target",
                f"label {lm.source_label!r} -> {lm.target!r} not in {sorted(valid_targets)}")
        if lm.mapping_status not in MAPPING_REVIEW_STATES:
            err("ner.bad_mapping_status",
                f"label {lm.source_label!r} mapping_status {lm.mapping_status!r} invalid")
    mapped = {lm.source_label for lm in m.label_mappings}
    missing = sorted(set(m.original_labels) - mapped - set(m.incompatible_labels))
    if missing:
        err("ner.unmapped_labels", f"labels missing mapping/rejection: {missing}")
    if m.leakage_risk in ("high",):
        warn("ner.high_leakage", "leakage_risk is high — verify split provenance before use")
    if m.license.status == "REJECTED":
        err("ner.rejected", "license REJECTED — dataset must not be used")
    elif m.license.status not in USABLE_STATUSES:
        warn("ner.not_yet_usable",
             f"license status {m.license.status} — not usable until reviewed")
    if m.governance_status.upper() in {"APPROVED", "TRAINING_READY"} and (
        m.license.status not in USABLE_STATUSES
    ):
        err("ner.governance_exceeds_license",
            "governance_status cannot approve use before license review permits it")
    return ValidationResult.from_issues(issues)


__all__ = [
    "MAPPING_ACTIONS",
    "MAPPING_REVIEW_STATES",
    "LabelMapping",
    "LabelMappingSet",
    "NerDatasetManifest",
    "ner_manifest_from_mapping",
    "load_ner_manifest",
    "label_mapping_set",
    "validate_ner_manifest",
]
