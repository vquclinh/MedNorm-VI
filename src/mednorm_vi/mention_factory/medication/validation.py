"""Medication-specific proposal validation (offset invariants + components)."""

from __future__ import annotations

from ...validator.results import Severity, ValidationIssue, ValidationResult
from ..models import SpanProposal


def validate_medication_proposals(
    proposals: list[SpanProposal], original_text: str
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    n = len(original_text)
    for p in proposals:
        if not (0 <= p.start <= p.end <= n):
            issues.append(ValidationIssue("medication.out_of_bounds",
                                          f"{p.proposal_id} span out of bounds"))
            continue
        if original_text[p.start : p.end] != p.text:
            issues.append(ValidationIssue("medication.text_mismatch",
                                          f"{p.proposal_id} text/offset mismatch"))
        for c in p.components:
            if not (0 <= c.start <= c.end <= n) or original_text[c.start : c.end] != c.text:
                issues.append(ValidationIssue(
                    "medication.component_offset",
                    f"{p.proposal_id} component {c.role} offset/text mismatch"))
        if "unknown_medication_name" in p.warnings:
            issues.append(ValidationIssue("medication.unknown_name",
                                          f"{p.proposal_id} unknown medication name",
                                          severity=Severity.WARNING))
    return ValidationResult.from_issues(issues)


__all__ = ["validate_medication_proposals"]
