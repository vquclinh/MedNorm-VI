"""Governance validation for resource manifests (Phase 1C-A).

A resource may not be *used* without: a version, a source, at least one file
checksum, a license review status, and an intended-use declaration. Validation
enforces field COMPLETENESS and internal consistency only — it makes no legal
conclusion (a human sets the license status).
"""

from __future__ import annotations

from ..validator.results import Severity, ValidationIssue, ValidationResult
from .models import (
    LICENSE_STATUSES,
    REDISTRIBUTION_PERMISSIONS,
    REVIEW_STATES,
    USABLE_STATUSES,
    USAGE_PURPOSES,
    ResourceManifest,
)


def validate_manifest(m: ResourceManifest) -> ValidationResult:
    """Validate one manifest's governance completeness + consistency."""
    issues: list[ValidationIssue] = []
    rid = m.resource_id or "<no id>"

    def err(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, f"{rid}: {msg}"))

    def warn(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, f"{rid}: {msg}", severity=Severity.WARNING))

    if not m.resource_id:
        err("resource.missing_id", "missing resource_id")
    if not m.title:
        warn("resource.missing_title", "missing title")

    # --- required-to-use governance fields ---
    if not (m.snapshot.version or m.snapshot.snapshot_id):
        err("resource.missing_version", "missing snapshot.version / snapshot_id")
    if not (m.source.organization or m.source.url):
        err("resource.missing_source", "missing source organization/url")
    if not m.files:
        err("resource.missing_checksums", "no file checksums declared")
    else:
        for f in m.files:
            if not f.sha256:
                err("resource.missing_checksum", f"file {f.path!r} has no sha256")
    if m.license.status not in LICENSE_STATUSES:
        err("resource.bad_license_status",
            f"license.status {m.license.status!r} not in {sorted(LICENSE_STATUSES)}")
    if not m.intended_use:
        err("resource.missing_intended_use", "no intended_use declared")

    # --- enum consistency ---
    for purpose in (*m.intended_use, *m.permitted_use):
        if purpose not in USAGE_PURPOSES:
            err("resource.bad_usage_purpose",
                f"usage purpose {purpose!r} not in {sorted(USAGE_PURPOSES)}")
    if m.redistribution.permission not in REDISTRIBUTION_PERMISSIONS:
        err("resource.bad_redistribution",
            f"redistribution.permission {m.redistribution.permission!r} invalid")
    if m.review_status not in REVIEW_STATES:
        err("resource.bad_review_state", f"review_status {m.review_status!r} invalid")

    # --- blocking governance states (not a legal conclusion) ---
    if m.license.status == "REJECTED":
        err("resource.rejected", "license status REJECTED — resource must not be used")
    elif m.license.status not in USABLE_STATUSES:
        warn("resource.not_yet_usable",
             f"license status {m.license.status} — not usable until reviewed")

    # intended use must be within permitted use when permitted use is declared
    if m.permitted_use:
        extra = sorted(set(m.intended_use) - set(m.permitted_use))
        if extra:
            err("resource.intended_exceeds_permitted",
                f"intended_use {extra} exceeds permitted_use")
    return ValidationResult.from_issues(issues)


__all__ = ["validate_manifest"]
