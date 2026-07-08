"""Organizer-facing entity validation (spec sections 1, 7-8; confirmed labels).

Validates entities in the ORGANIZER submission shape — i.e. after serialization,
where ``type`` is the exact Vietnamese label and only the fields allowed for that
type are present. This is distinct from ``validator.entities`` (which validates
the INTERNAL English-enum representation).

Enforced here:
* ``type`` must be a valid Vietnamese organizer label; English enum values and
  any other string are rejected.
* Exactly the fields allowed for the type are present — no missing, no extra
  (unsupported) field may reach final JSON.
* Candidates: MEDICATION -> numeric RxNorm identifiers; DIAGNOSIS -> ICD-10
  strings; SYMPTOM/TEST_NAME/TEST_RESULT -> candidates field must be absent.
* Assertions: allowed only for MEDICATION/DIAGNOSIS/SYMPTOM; valid labels,
  no duplicates.
* ``position`` is a 2-integer ``[start, end]`` pair (end-exclusive shape).

KB membership of a code is deferred until frozen KB snapshots exist; candidate
syntax is kept deliberately light (numeric RxCUIs; ICD-10 as opaque strings).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schemas.constants import (
    ASSERTION_LABELS,
    CANDIDATE_ONTOLOGY_BY_TYPE,
    ENTITY_TYPES,
    ORGANIZER_FIELDS_BY_TYPE,
    ORGANIZER_LABELS,
    TYPE_BY_ORGANIZER_LABEL,
)
from .results import Severity, ValidationIssue, ValidationResult


def _issue(
    code: str,
    message: str,
    document_id: str | None,
    entity_index: int | None,
    **ctx: object,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=Severity.ERROR,
        document_id=document_id,
        entity_index=entity_index,
        context=ctx,
    )


def _validate_position(
    value: object, document_id: str | None, entity_index: int | None
) -> list[ValidationIssue]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return [
            _issue(
                "organizer.bad_position",
                f"position must be a 2-integer [start, end] pair, got {value!r}",
                document_id,
                entity_index,
            )
        ]
    start, end = value
    if isinstance(start, bool) or isinstance(end, bool):
        return [
            _issue(
                "organizer.bad_position",
                "position values must be integers, not booleans",
                document_id,
                entity_index,
            )
        ]
    if not isinstance(start, int) or not isinstance(end, int):
        return [
            _issue(
                "organizer.bad_position",
                f"position values must be integers, got {value!r}",
                document_id,
                entity_index,
            )
        ]
    issues: list[ValidationIssue] = []
    if start < 0:
        issues.append(
            _issue("organizer.negative_start", f"start must be >= 0, got {start}",
                   document_id, entity_index)
        )
    if end < start:
        issues.append(
            _issue("organizer.end_before_start",
                   f"end {end} must be >= start {start} (end-exclusive)",
                   document_id, entity_index)
        )
    return issues


def _validate_candidates(
    internal_type: str,
    candidates: object,
    document_id: str | None,
    entity_index: int | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(candidates, (list, tuple)):
        return [
            _issue("organizer.candidates_not_list",
                   f"candidates must be a list, got {type(candidates)!r}",
                   document_id, entity_index)
        ]
    ontology = CANDIDATE_ONTOLOGY_BY_TYPE.get(internal_type)
    for c in candidates:
        if not isinstance(c, str) or c == "":
            issues.append(
                _issue("organizer.candidate_not_str",
                       f"every candidate must be a non-empty string, got {c!r}",
                       document_id, entity_index)
            )
            continue
        # Light, known-only syntax checks (no KB membership).
        if ontology == "RXNORM" and not c.isdigit():
            issues.append(
                _issue("organizer.rxnorm_not_numeric",
                       f"RxNorm candidate must be a numeric RxCUI string, got {c!r}",
                       document_id, entity_index, value=c)
            )
        # ICD-10 kept as opaque strings; no pattern constraint imposed yet.
    return issues


def _validate_assertions(
    assertions: object, document_id: str | None, entity_index: int | None
) -> list[ValidationIssue]:
    if not isinstance(assertions, (list, tuple)):
        return [
            _issue("organizer.assertions_not_list",
                   f"assertions must be a list, got {type(assertions)!r}",
                   document_id, entity_index)
        ]
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    for label in assertions:
        if label not in ASSERTION_LABELS:
            issues.append(
                _issue("organizer.invalid_assertion",
                       f"assertion {label!r} not in allowed {sorted(ASSERTION_LABELS)}",
                       document_id, entity_index, value=label)
            )
        elif label in seen:
            issues.append(
                _issue("organizer.duplicate_assertion",
                       f"assertion {label!r} listed more than once",
                       document_id, entity_index, value=label)
            )
        seen.add(label)
    return issues


def validate_organizer_entity(
    entity: Mapping[str, Any],
    *,
    document_id: str | None = None,
    entity_index: int | None = None,
) -> ValidationResult:
    """Validate a single organizer-facing entity dict (post-serialization)."""
    if not isinstance(entity, Mapping):
        return ValidationResult.from_issues(
            [_issue("organizer.not_object",
                    f"entity must be a JSON object, got {type(entity)!r}",
                    document_id, entity_index)]
        )

    issues: list[ValidationIssue] = []
    etype = entity.get("type")

    # 1) type must be a valid Vietnamese organizer label. Reject English enums.
    if etype not in ORGANIZER_LABELS:
        if isinstance(etype, str) and etype in ENTITY_TYPES:
            issues.append(
                _issue("organizer.english_type",
                       f"English entity value {etype!r} is not allowed in submission "
                       f"JSON; use the Vietnamese label",
                       document_id, entity_index, value=etype)
            )
        else:
            issues.append(
                _issue("organizer.invalid_type",
                       f"type {etype!r} not in allowed organizer labels "
                       f"{sorted(ORGANIZER_LABELS)}",
                       document_id, entity_index, value=etype)
            )
        # Cannot apply per-type field policy without a known type.
        return ValidationResult.from_issues(issues)

    internal_type = TYPE_BY_ORGANIZER_LABEL[etype]
    allowed = set(ORGANIZER_FIELDS_BY_TYPE[internal_type])
    present = set(entity.keys())

    # 2) exact field-set match: no missing, no unsupported field.
    for missing in sorted(allowed - present):
        issues.append(
            _issue("organizer.missing_field",
                   f"{etype!r} entity missing required field {missing!r}",
                   document_id, entity_index, field=missing)
        )
    for extra in sorted(present - allowed):
        issues.append(
            _issue("organizer.unsupported_field",
                   f"{etype!r} entity has unsupported field {extra!r} "
                   f"(allowed: {sorted(allowed)})",
                   document_id, entity_index, field=extra)
        )

    # 3) field-level checks (only when the field is allowed & present).
    if "text" in present:
        text = entity.get("text")
        if not isinstance(text, str) or text == "":
            issues.append(
                _issue("organizer.bad_text", "text must be a non-empty string",
                       document_id, entity_index)
            )
    if "position" in present:
        issues.extend(_validate_position(entity.get("position"), document_id, entity_index))
    if "candidates" in allowed and "candidates" in present:
        issues.extend(
            _validate_candidates(internal_type, entity.get("candidates"),
                                 document_id, entity_index)
        )
    if "assertions" in allowed and "assertions" in present:
        issues.extend(_validate_assertions(entity.get("assertions"), document_id, entity_index))

    return ValidationResult.from_issues(issues)


def validate_organizer_document(
    root: object,
    *,
    document_id: str | None = None,
) -> ValidationResult:
    """Validate a whole organizer document: root must be a list of entities."""
    if not isinstance(root, list):
        return ValidationResult.from_issues(
            [ValidationIssue(
                code="organizer.root_not_list",
                message=f"document root must be a JSON list, got {type(root)!r}",
                document_id=document_id,
            )]
        )
    result = ValidationResult()
    for idx, entity in enumerate(root):
        result = result.merged_with(
            validate_organizer_entity(entity, document_id=document_id, entity_index=idx)
        )
    return result


__all__ = ["validate_organizer_entity", "validate_organizer_document"]
