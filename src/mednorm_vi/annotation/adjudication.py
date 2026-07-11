"""Adjudication helpers: build and validate adjudication records.

An adjudication record preserves the disagreement (annotators A/B, reason) and
its resolution (adjudicated entity, adjudicator, guideline version, notes). The
adjudicated result must itself be a valid annotation and become ADJUDICATED.
"""

from __future__ import annotations

import dataclasses

from ..validator.results import ValidationIssue, ValidationResult
from .models import AdjudicationRecord, AnnotationEntity, ReviewStatus


def make_adjudication(
    *,
    document_id: str,
    annotator_a: str,
    annotator_b: str,
    disagreement_reason: str,
    adjudicated_result: AnnotationEntity,
    adjudicator_id: str,
    guideline_version: str,
    notes: str | None = None,
) -> AdjudicationRecord:
    """Create an :class:`AdjudicationRecord`, forcing the result to ADJUDICATED."""
    resolved = dataclasses.replace(adjudicated_result, review_status=ReviewStatus.ADJUDICATED)
    return AdjudicationRecord(
        document_id=document_id,
        annotator_a=annotator_a,
        annotator_b=annotator_b,
        disagreement_reason=disagreement_reason,
        adjudicated_result=resolved,
        adjudicator_id=adjudicator_id,
        guideline_version=guideline_version,
        notes=notes,
    )


def validate_adjudication(record: AdjudicationRecord) -> ValidationResult:
    """Validate an adjudication record's integrity."""
    issues: list[ValidationIssue] = []
    if not record.adjudicator_id:
        issues.append(ValidationIssue("adjudication.missing_adjudicator",
                                      "adjudicator_id is required"))
    if not record.disagreement_reason:
        issues.append(ValidationIssue("adjudication.missing_reason",
                                      "disagreement_reason is required"))
    if not record.guideline_version:
        issues.append(ValidationIssue("adjudication.missing_guideline",
                                      "guideline_version is required"))
    if record.adjudicated_result.review_status is not ReviewStatus.ADJUDICATED:
        issues.append(ValidationIssue(
            "adjudication.result_not_adjudicated",
            "adjudicated_result.review_status must be ADJUDICATED"))
    if record.annotator_a == record.annotator_b:
        issues.append(ValidationIssue("adjudication.same_annotator",
                                      "annotator_a and annotator_b must differ"))
    return ValidationResult.from_issues(issues)


__all__ = ["make_adjudication", "validate_adjudication"]
