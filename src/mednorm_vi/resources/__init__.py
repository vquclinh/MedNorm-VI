"""External-resource governance (Phase 1C-A).

Tracked manifests describe every external dataset / KB (provenance, license
review, checksums, intended use). Raw and derived data stay git-ignored. No legal
conclusion is made automatically — validation enforces governance completeness.
"""

from __future__ import annotations

from .manifest import load_manifest, manifest_from_mapping
from .models import (
    AcquisitionRecord,
    ArchiveRecord,
    ChecksumRecord,
    DerivedArtifactRecord,
    ExtractedSnapshotRecord,
    LicenseRecord,
    RedistributionPolicy,
    ResourceManifest,
    SnapshotIdentity,
    SourceRecord,
    TransformationRecord,
)
from .validation import validate_manifest

__all__ = [
    "SourceRecord",
    "LicenseRecord",
    "RedistributionPolicy",
    "ChecksumRecord",
    "SnapshotIdentity",
    "AcquisitionRecord",
    "TransformationRecord",
    "DerivedArtifactRecord",
    "ArchiveRecord",
    "ExtractedSnapshotRecord",
    "ResourceManifest",
    "manifest_from_mapping",
    "load_manifest",
    "validate_manifest",
]
