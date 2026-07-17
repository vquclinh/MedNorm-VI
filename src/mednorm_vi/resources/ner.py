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
from .models import LicenseRecord, RedistributionPolicy, SourceRecord

# A source label maps to a MedNorm entity type, or to an explicit action.
MAPPING_ACTIONS: frozenset[str] = frozenset({"DROP", "REVIEW"})


@dataclass(frozen=True, slots=True)
class LabelMapping:
    """source label -> canonical intermediate -> MedNorm target (or DROP/REVIEW)."""

    source_label: str
    canonical_intermediate: str
    target: str  # a MedNorm ENTITY_TYPES value, or "DROP" / "REVIEW"
    note: str = ""

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
        target=str(d["target"]), note=str(d.get("note", "")))


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
    if not m.original_labels:
        warn("ner.no_labels", "no original_labels declared")
    valid_targets = set(ENTITY_TYPES) | MAPPING_ACTIONS
    for lm in m.label_mappings:
        if lm.target not in valid_targets:
            err("ner.bad_mapping_target",
                f"label {lm.source_label!r} -> {lm.target!r} not in {sorted(valid_targets)}")
    if m.leakage_risk in ("high",):
        warn("ner.high_leakage", "leakage_risk is high — verify split provenance before use")
    if m.license.status == "REJECTED":
        err("ner.rejected", "license REJECTED — dataset must not be used")
    return ValidationResult.from_issues(issues)


__all__ = [
    "MAPPING_ACTIONS",
    "LabelMapping",
    "LabelMappingSet",
    "NerDatasetManifest",
    "ner_manifest_from_mapping",
    "load_ner_manifest",
    "label_mapping_set",
    "validate_ner_manifest",
]
