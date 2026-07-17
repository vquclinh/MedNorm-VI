"""Governance validation for resource manifests (Phase 1C-A).

A resource may not be *used* without: a version, a source, at least one file
checksum, a license review status, and an intended-use declaration. Validation
enforces field COMPLETENESS and internal consistency only — it makes no legal
conclusion (a human sets the license status).
"""

from __future__ import annotations

import re

from ..validator.results import Severity, ValidationIssue, ValidationResult
from .models import (
    ARCHIVE_VERIFICATION_STATES,
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
            elif not re.fullmatch(r"[0-9a-fA-F]{64}", f.sha256):
                err("resource.bad_sha256", f"file {f.path!r} sha256 is not SHA-256")
            if f.md5 and not re.fullmatch(r"[0-9a-fA-F]{32}", f.md5):
                err("resource.bad_md5", f"file {f.path!r} md5 is not MD5")
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

    if m.archive is not None:
        a = m.archive
        if not a.original_filename:
            err("resource.archive_missing_filename", "archive.original_filename is required")
        if not a.package_type:
            err("resource.archive_missing_package_type", "archive.package_type is required")
        if (a.expected_published_md5
                and not re.fullmatch(r"[0-9a-fA-F]{32}", a.expected_published_md5)):
            err("resource.archive_bad_expected_md5", "archive.expected_published_md5 is not MD5")
        if (a.locally_calculated_md5
                and not re.fullmatch(r"[0-9a-fA-F]{32}", a.locally_calculated_md5)):
            err("resource.archive_bad_local_md5", "archive.locally_calculated_md5 is not MD5")
        if a.archive_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", a.archive_sha256):
            err("resource.archive_bad_sha256", "archive.archive_sha256 is not SHA-256")
        if a.locally_verified_archive_md5 not in ARCHIVE_VERIFICATION_STATES:
            err("resource.archive_bad_md5_verification",
                "archive.locally_verified_archive_md5 must be true, false, or unknown")
        if (a.expected_published_md5 and a.locally_calculated_md5
                and a.expected_published_md5.lower() != a.locally_calculated_md5.lower()):
            err("resource.archive_md5_mismatch",
                "archive.locally_calculated_md5 does not match expected_published_md5")
        if a.archive_present is True:
            if not a.locally_calculated_md5:
                err("resource.archive_missing_local_md5",
                    "present archive requires locally_calculated_md5")
            if not a.archive_sha256:
                err("resource.archive_missing_sha256", "present archive requires archive_sha256")
        elif a.archive_present is False and not a.archive_sha256:
            if a.archive_deleted_before_manifest_completion:
                warn(
                    "resource.archive_sha256_unknown",
                    "archive absent; SHA-256 unknown because deletion preceded manifest completion",
                )
            else:
                err("resource.archive_missing_sha256",
                    "absent archive with unknown SHA-256 requires "
                    "archive_deleted_before_manifest_completion")

    if m.extracted_snapshot is not None:
        e = m.extracted_snapshot
        if not e.local_path:
            err("resource.extracted_missing_path", "extracted_snapshot.local_path is required")
        if not e.tree_hash_sha256:
            err("resource.extracted_missing_tree_hash",
                "extracted_snapshot.tree_hash_sha256 is required")
        elif not re.fullmatch(r"[0-9a-fA-F]{64}", e.tree_hash_sha256):
            err("resource.extracted_bad_tree_hash",
                "extracted_snapshot.tree_hash_sha256 is not SHA-256")
        if e.file_count is not None and e.file_count <= 0:
            err("resource.extracted_bad_file_count",
                "extracted_snapshot.file_count must be positive")
        if e.total_bytes is not None and e.total_bytes <= 0:
            err("resource.extracted_bad_total_bytes",
                "extracted_snapshot.total_bytes must be positive")

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
