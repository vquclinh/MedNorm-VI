"""L9 — Validator: deterministic, fail-fast foundation (spec sections 16, 19).

This package holds the Phase-0 deterministic guards that MUST exist before any
heavy model, so model changes can be judged against the organizer's true metric.

Implemented here:
    1. Offset-invariant validation      (offsets.py)
    2. Entity schema validation         (entities.py)
    3. Allowed entity-type validation   (entities.py)
    4. Allowed assertion-label validation (entities.py)
    5. Duplicate detection (text/type/position) (duplicates.py)
    6. Parameter-budget validation      (budget.py)
    7. Submission-layout validation     (submission.py)
    8. Ontology membership against the frozen snapshots (kb_membership.py)
    9. **The final gate over the serialized payload** (final_gate.py)

``validate_document`` composes the per-entity checks for one document, over the
INTERNAL representation. :mod:`mednorm_vi.validator.final_gate` is what the
canonical packaging path runs, over the ORGANIZER-facing serialized JSON plus the
source text — item 9 exists because items 1-7 were each enforced somewhere upstream
and never together on the bytes that would actually be submitted (Audit 0056a).

(Item 7 was described as a "STUB" here until Audit 0056a. It has not been a stub
since Audit 0002 confirmed the layout.)
"""

from __future__ import annotations

from ._coerce import EntityLike
from .budget import (
    load_budget_config,
    total_adapter_params,
    total_base_params,
    validate_parameter_budget,
)
from .duplicates import deduplicate, find_duplicate_groups, validate_no_duplicates
from .entities import (
    validate_assertion_labels,
    validate_entities,
    validate_entity_schema,
    validate_entity_type,
)
from .kb_membership import (
    KbMembershipViolation,
    LockedSnapshots,
    MembershipReport,
    validate_document_candidates,
    validate_entity_candidates,
)
from .offsets import validate_offset_invariant, validate_offsets
from .organizer import validate_organizer_document, validate_organizer_entity
from .results import Severity, ValidationIssue, ValidationResult
from .submission import (
    expected_filenames,
    validate_output_directory,
    validate_submission_zip,
    write_submission_zip,
)


def validate_document(
    original_text: str,
    entities: list[EntityLike],
    *,
    document_id: str | None = None,
) -> ValidationResult:
    """Run the full deterministic entity-level check suite for one document.

    Composes: schema + type + assertion-label + offset invariant + duplicate
    detection. Returns an aggregated :class:`ValidationResult`; callers may treat
    any ERROR as fail-fast.
    """
    result = validate_entities(entities, document_id=document_id)
    result = result.merged_with(validate_offsets(original_text, entities, document_id=document_id))
    result = result.merged_with(validate_no_duplicates(entities, document_id=document_id))
    return result


__all__ = [
    "KbMembershipViolation",
    "LockedSnapshots",
    "MembershipReport",
    "validate_document_candidates",
    "validate_entity_candidates",
    # results
    "ValidationIssue",
    "ValidationResult",
    "Severity",
    # composed
    "validate_document",
    # offsets
    "validate_offset_invariant",
    "validate_offsets",
    # entities
    "validate_entity_schema",
    "validate_entity_type",
    "validate_assertion_labels",
    "validate_entities",
    # duplicates
    "find_duplicate_groups",
    "validate_no_duplicates",
    "deduplicate",
    # budget
    "validate_parameter_budget",
    "total_base_params",
    "total_adapter_params",
    "load_budget_config",
    # organizer-facing entity validation
    "validate_organizer_entity",
    "validate_organizer_document",
    # submission (confirmed layout: output.zip -> output/1.json..100.json)
    "expected_filenames",
    "validate_output_directory",
    "validate_submission_zip",
    "write_submission_zip",
]
