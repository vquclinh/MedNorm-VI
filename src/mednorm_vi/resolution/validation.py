"""Validation for resolver output (Phase 1C-A).

Enforces: exact offsets (``original_text[start:end] == text``), resolvable types
only, valid statuses, unique hypothesis ids, and — critically — that NO ontology
candidate was emitted (linking is out of scope for this milestone).
"""

from __future__ import annotations

from ..validator.results import ValidationIssue, ValidationResult
from .models import (
    RESOLVABLE_TYPES,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNRESOLVED,
    ResolutionResult,
)

_VALID_STATUS = {STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNRESOLVED}
# feature keys that would indicate a leaked ontology candidate
_ONTOLOGY_KEYS = {"rxcui", "rxnorm", "icd10", "icd", "candidate", "code"}


def validate_result(result: ResolutionResult, original_text: str) -> ValidationResult:
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    for h in result.hypotheses:
        if h.hypothesis_id in seen:
            issues.append(ValidationIssue("resolution.duplicate_id",
                                          f"duplicate hypothesis id {h.hypothesis_id}"))
        seen.add(h.hypothesis_id)
        if not (0 <= h.start <= h.end <= len(original_text)):
            issues.append(ValidationIssue("resolution.out_of_bounds",
                                          f"{h.hypothesis_id}: span out of bounds"))
        elif original_text[h.start:h.end] != h.text:
            issues.append(ValidationIssue("resolution.text_offset_mismatch",
                                          f"{h.hypothesis_id}: text != original_text[start:end]"))
        if h.status not in _VALID_STATUS:
            issues.append(ValidationIssue("resolution.bad_status",
                                          f"{h.hypothesis_id}: status {h.status!r}"))
        if h.status == STATUS_ACCEPTED and h.entity_type not in RESOLVABLE_TYPES:
            issues.append(ValidationIssue("resolution.unresolvable_accepted",
                                          f"{h.hypothesis_id}: accepted non-resolvable type"))
        # no ontology candidate may leak through features
        leaked = _ONTOLOGY_KEYS & {k.lower() for k in h.features}
        if leaked:
            issues.append(ValidationIssue(
                "resolution.ontology_candidate_leak",
                f"{h.hypothesis_id}: ontology-like feature keys {leaked}"))
    return ValidationResult.from_issues(issues)


__all__ = ["validate_result"]
