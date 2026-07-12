"""Deterministic validation of routing decisions."""

from __future__ import annotations

from ..validator.results import ValidationIssue, ValidationResult
from .models import NodeRouting

_VALID_CASES = {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}


def validate_routings(routings: list[NodeRouting], original_text: str) -> ValidationResult:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for r in routings:
        if r.decision_id in seen_ids:
            issues.append(ValidationIssue("router.duplicate_decision_id",
                                          f"duplicate decision id {r.decision_id}"))
        seen_ids.add(r.decision_id)
        if not (0 <= r.start <= r.end <= len(original_text)):
            issues.append(ValidationIssue("router.bad_span",
                                          f"{r.decision_id} span out of bounds"))
        elif original_text[r.start : r.end] != r.text:
            issues.append(ValidationIssue("router.text_mismatch",
                                          f"{r.decision_id} routed text/offset mismatch"))
        for c in r.cases:
            if c.case not in _VALID_CASES:
                issues.append(ValidationIssue("router.invalid_case",
                                              f"{r.decision_id}: unknown case {c.case!r}"))
            if not (0.0 <= c.score <= 1.0):
                issues.append(ValidationIssue("router.score_out_of_range",
                                              f"{r.decision_id}/{c.case} score {c.score}"))
    return ValidationResult.from_issues(issues)


__all__ = ["validate_routings"]
