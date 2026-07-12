"""Laboratory-specific proposal/relation validation."""

from __future__ import annotations

from ...validator.results import Severity, ValidationIssue, ValidationResult
from ..models import RelationProposal, SpanProposal


def validate_laboratory(
    proposals: list[SpanProposal], relations: list[RelationProposal], original_text: str
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    n = len(original_text)
    ids = {p.proposal_id for p in proposals}
    for p in proposals:
        if not (0 <= p.start <= p.end <= n) or original_text[p.start : p.end] != p.text:
            issues.append(ValidationIssue("laboratory.text_offset",
                                          f"{p.proposal_id} text/offset mismatch"))
        for c in p.components:
            if not (0 <= c.start <= c.end <= n) or original_text[c.start : c.end] != c.text:
                issues.append(ValidationIssue("laboratory.component_offset",
                                              f"{p.proposal_id} component {c.role} mismatch"))
        if "unknown_test_name" in p.warnings:
            issues.append(ValidationIssue("laboratory.unknown_test",
                                          f"{p.proposal_id} unknown test name",
                                          severity=Severity.WARNING))
    for r in relations:
        if r.source_proposal_id not in ids or r.target_proposal_id not in ids:
            issues.append(ValidationIssue("laboratory.broken_relation",
                                          f"{r.relation_id} endpoint not a known proposal"))
    return ValidationResult.from_issues(issues)


__all__ = ["validate_laboratory"]
