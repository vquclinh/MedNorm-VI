"""Typed contracts for external-resource governance (Phase 1C-A).

Every external dataset or knowledge base (RxNorm, ICD-10-VI, public NER corpora)
must be described by a tracked :class:`ResourceManifest` before it is ingested or
used. Raw/derived data stays git-ignored; only manifests/templates are tracked.

These contracts record provenance, license review, checksums, and intended use.
They make NO automatic legal conclusion — a human reviewer sets the license
status; the validator only enforces that governance fields are present and
internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# License review status — set by a human reviewer, never inferred automatically.
LICENSE_STATUSES: frozenset[str] = frozenset({
    "LICENSE_UNKNOWN",          # license not yet identified
    "REVIEW_REQUIRED",          # identified but not yet reviewed/approved
    "PERMITTED_INTERNAL_USE",   # reviewed; internal use permitted
    "REDISTRIBUTION_RESTRICTED",  # usable internally; may not be redistributed
    "REJECTED",                 # not usable
})
# Statuses under which a resource may actually be USED (still no legal claim).
USABLE_STATUSES: frozenset[str] = frozenset(
    {"PERMITTED_INTERNAL_USE", "REDISTRIBUTION_RESTRICTED"}
)

# Declared intended uses (a resource must declare at least one).
USAGE_PURPOSES: frozenset[str] = frozenset({
    "domain_adaptation", "ner_training", "assertion_training", "linking", "evaluation",
})

REDISTRIBUTION_PERMISSIONS: frozenset[str] = frozenset(
    {"permitted", "restricted", "forbidden", "unknown"}
)
REVIEW_STATES: frozenset[str] = frozenset({"pending", "in_review", "reviewed", "rejected"})


@dataclass(frozen=True, slots=True)
class SourceRecord:
    organization: str = ""
    url: str = ""  # placeholder until manually acquired
    description: str = ""


@dataclass(frozen=True, slots=True)
class LicenseRecord:
    name: str = ""
    status: str = "LICENSE_UNKNOWN"
    text_reference: str = ""  # path/URL to the license text, not the text itself
    permits_redistribution: bool | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class RedistributionPolicy:
    permission: str = "unknown"  # one of REDISTRIBUTION_PERMISSIONS
    organizer_submission_implications: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ChecksumRecord:
    path: str  # relative path of the file inside the resource
    sha256: str
    bytes: int | None = None


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    snapshot_id: str = ""
    version: str = ""
    release_date: str = ""
    acquisition_date: str = ""


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    acquired_by: str = ""
    acquisition_date: str = ""
    method: str = ""  # manual download, institutional access, ...
    notes: str = ""


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    script: str = ""  # path to the (deterministic) transformation script
    description: str = ""
    deterministic: bool = True
    output_paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DerivedArtifactRecord:
    artifact_id: str
    description: str = ""
    path: str = ""
    sha256: str = ""
    source_resource_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ResourceManifest:
    """A tracked description of one external resource. Raw data stays untracked."""

    resource_id: str
    title: str
    source: SourceRecord
    snapshot: SnapshotIdentity
    license: LicenseRecord
    redistribution: RedistributionPolicy
    files: tuple[ChecksumRecord, ...] = field(default_factory=tuple)
    permitted_use: tuple[str, ...] = field(default_factory=tuple)
    intended_use: tuple[str, ...] = field(default_factory=tuple)
    raw_path: str = ""
    processed_path: str = ""
    transformation: TransformationRecord | None = None
    label_schema: tuple[str, ...] = field(default_factory=tuple)
    acquisition: AcquisitionRecord | None = None
    provenance: str = ""
    review_status: str = "pending"
    reviewer: str = ""
    unresolved_legal_questions: tuple[str, ...] = field(default_factory=tuple)
    derived_artifacts: tuple[DerivedArtifactRecord, ...] = field(default_factory=tuple)
    manifest_version: int = 1

    @property
    def is_usable(self) -> bool:
        """True only when license review explicitly permits use. No legal claim."""
        return self.license.status in USABLE_STATUSES


__all__ = [
    "LICENSE_STATUSES",
    "USABLE_STATUSES",
    "USAGE_PURPOSES",
    "REDISTRIBUTION_PERMISSIONS",
    "REVIEW_STATES",
    "SourceRecord",
    "LicenseRecord",
    "RedistributionPolicy",
    "ChecksumRecord",
    "SnapshotIdentity",
    "AcquisitionRecord",
    "TransformationRecord",
    "DerivedArtifactRecord",
    "ResourceManifest",
]
