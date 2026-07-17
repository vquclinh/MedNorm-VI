"""Load an external-resource manifest (YAML/JSON) into typed contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    AcquisitionRecord,
    ChecksumRecord,
    DerivedArtifactRecord,
    LicenseRecord,
    RedistributionPolicy,
    ResourceManifest,
    SnapshotIdentity,
    SourceRecord,
    TransformationRecord,
)


def _load_doc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore[import-untyped]

        doc = yaml.safe_load(text) or {}
    else:
        doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError(f"resource manifest {path} must be a mapping")
    return doc


def _opt_int(value: Any) -> int | None:
    """Coerce to int; tolerate template placeholders (e.g. <BYTES>) as None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def manifest_from_mapping(d: dict[str, Any]) -> ResourceManifest:
    src = d.get("source", {}) or {}
    snap = d.get("snapshot", {}) or {}
    lic = d.get("license", {}) or {}
    red = d.get("redistribution", {}) or {}
    trans = d.get("transformation")
    acq = d.get("acquisition")

    files = tuple(
        ChecksumRecord(path=str(f["path"]), sha256=str(f.get("sha256", "")),
                       bytes=_opt_int(f.get("bytes")))
        for f in d.get("files", []) or []
    )
    derived = tuple(
        DerivedArtifactRecord(
            artifact_id=str(a["artifact_id"]), description=str(a.get("description", "")),
            path=str(a.get("path", "")), sha256=str(a.get("sha256", "")),
            source_resource_ids=tuple(str(x) for x in a.get("source_resource_ids", []) or []))
        for a in d.get("derived_artifacts", []) or []
    )
    return ResourceManifest(
        resource_id=str(d.get("resource_id", "")),
        title=str(d.get("title", "")),
        source=SourceRecord(organization=str(src.get("organization", "")),
                            url=str(src.get("url", "")),
                            description=str(src.get("description", ""))),
        snapshot=SnapshotIdentity(
            snapshot_id=str(snap.get("snapshot_id", "")),
            version=str(snap.get("version", "")),
            release_date=str(snap.get("release_date", "")),
            acquisition_date=str(snap.get("acquisition_date", ""))),
        license=LicenseRecord(
            name=str(lic.get("name", "")), status=str(lic.get("status", "LICENSE_UNKNOWN")),
            text_reference=str(lic.get("text_reference", "")),
            permits_redistribution=lic.get("permits_redistribution"),
            notes=str(lic.get("notes", ""))),
        redistribution=RedistributionPolicy(
            permission=str(red.get("permission", "unknown")),
            organizer_submission_implications=str(red.get("organizer_submission_implications", "")),
            notes=str(red.get("notes", ""))),
        files=files,
        permitted_use=tuple(str(x) for x in d.get("permitted_use", []) or []),
        intended_use=tuple(str(x) for x in d.get("intended_use", []) or []),
        raw_path=str(d.get("raw_path", "")),
        processed_path=str(d.get("processed_path", "")),
        transformation=(TransformationRecord(
            script=str(trans.get("script", "")), description=str(trans.get("description", "")),
            deterministic=bool(trans.get("deterministic", True)),
            output_paths=tuple(str(x) for x in trans.get("output_paths", []) or []))
            if trans else None),
        label_schema=tuple(str(x) for x in d.get("label_schema", []) or []),
        acquisition=(AcquisitionRecord(
            acquired_by=str(acq.get("acquired_by", "")),
            acquisition_date=str(acq.get("acquisition_date", "")),
            method=str(acq.get("method", "")), notes=str(acq.get("notes", "")))
            if acq else None),
        provenance=str(d.get("provenance", "")),
        review_status=str(d.get("review_status", "pending")),
        reviewer=str(d.get("reviewer", "")),
        unresolved_legal_questions=tuple(
            str(x) for x in d.get("unresolved_legal_questions", []) or []),
        derived_artifacts=derived,
        manifest_version=int(d.get("manifest_version", 1)))


def load_manifest(path: str | Path) -> ResourceManifest:
    return manifest_from_mapping(_load_doc(Path(path)))


__all__ = ["manifest_from_mapping", "load_manifest"]
