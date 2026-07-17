"""Integrity validation for a local ICD-10 snapshot (offline)."""

from __future__ import annotations

from ...validator.results import Severity, ValidationIssue, ValidationResult
from . import normalization as norm
from .models import IcdSnapshot


def validate_snapshot(snapshot: IcdSnapshot) -> ValidationResult:
    issues: list[ValidationIssue] = []

    def err(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, msg))

    def warn(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, msg, severity=Severity.WARNING))

    if not snapshot.concepts:
        err("icd.empty", "snapshot has no concepts")
    known = set(snapshot.codes())
    for c in snapshot.concepts:
        if not norm.is_wellformed(c.undotted):
            warn("icd.malformed_code", f"code {c.code_supplied!r} is not well-formed ICD-10")
        if not c.label_vi:
            warn("icd.no_label", f"code {c.undotted} has no Vietnamese label")
        if c.parent and c.parent not in known:
            warn("icd.dangling_parent", f"code {c.undotted} parent {c.parent} not in snapshot")
        # reversible dotted/undotted
        if norm.to_undotted(c.dotted) != c.undotted:
            err("icd.format_irreversible",
                f"code {c.undotted}: dotted/undotted are not reversible")
    return ValidationResult.from_issues(issues)


__all__ = ["validate_snapshot"]
