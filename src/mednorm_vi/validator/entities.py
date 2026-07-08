"""Entity schema, type, and assertion-label validation (spec sections 1, 7-8).

Deterministic checks that an entity has the required fields with correct shapes,
a valid entity type, valid assertion labels, and candidates that belong to the
ontology allowed for its type (ICD-10 for DIAGNOSIS, RxNorm for MEDICATION; no
cross-linking). KB membership of a code is NOT checked here (that requires the
frozen KB snapshot) — only the ontology/type consistency the spec mandates.
"""

from __future__ import annotations

from ..schemas.constants import (
    ASSERTION_LABELS,
    CANDIDATE_ONTOLOGY_BY_TYPE,
    ENTITY_TYPES,
)
from ._coerce import EntityLike, as_entity_dict
from .results import Severity, ValidationIssue, ValidationResult

_REQUIRED_FIELDS = ("text", "type", "assertions", "candidates", "position")


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


def validate_entity_type(
    entity: EntityLike,
    *,
    document_id: str | None = None,
    entity_index: int | None = None,
) -> ValidationResult:
    """Validate that ``entity.type`` is one of the five allowed categories."""
    e = as_entity_dict(entity)
    etype = e.get("type")
    if etype not in ENTITY_TYPES:
        return ValidationResult.from_issues(
            [
                _issue(
                    "entity.invalid_type",
                    f"type {etype!r} not in allowed {sorted(ENTITY_TYPES)}",
                    document_id,
                    entity_index,
                    value=etype,
                )
            ]
        )
    return ValidationResult()


def validate_assertion_labels(
    entity: EntityLike,
    *,
    document_id: str | None = None,
    entity_index: int | None = None,
) -> ValidationResult:
    """Validate assertions is a list of allowed, non-duplicated labels."""
    e = as_entity_dict(entity)
    assertions = e.get("assertions")
    issues: list[ValidationIssue] = []
    if not isinstance(assertions, (list, tuple)):
        issues.append(
            _issue(
                "assertion.not_list",
                f"assertions must be a list, got {type(assertions)!r}",
                document_id,
                entity_index,
            )
        )
        return ValidationResult.from_issues(issues)
    seen: set[str] = set()
    for label in assertions:
        if label not in ASSERTION_LABELS:
            issues.append(
                _issue(
                    "assertion.invalid_label",
                    f"assertion {label!r} not in allowed {sorted(ASSERTION_LABELS)}",
                    document_id,
                    entity_index,
                    value=label,
                )
            )
        elif label in seen:
            issues.append(
                _issue(
                    "assertion.duplicate_label",
                    f"assertion {label!r} listed more than once",
                    document_id,
                    entity_index,
                    value=label,
                )
            )
        seen.add(label)
    return ValidationResult.from_issues(issues)


def validate_entity_schema(
    entity: EntityLike,
    *,
    document_id: str | None = None,
    entity_index: int | None = None,
) -> ValidationResult:
    """Validate presence/shape of required fields, type, assertions, candidates.

    Does NOT check the offset invariant (see ``validator.offsets``) or KB
    membership of candidate codes (requires the frozen KB snapshot).
    """
    e = as_entity_dict(entity)
    issues: list[ValidationIssue] = []

    for field_name in _REQUIRED_FIELDS:
        if field_name not in e:
            issues.append(
                _issue(
                    "schema.missing_field",
                    f"missing required field {field_name!r}",
                    document_id,
                    entity_index,
                    field=field_name,
                )
            )
    if issues:
        # Missing fields make deeper checks unreliable; return early.
        return ValidationResult.from_issues(issues)

    if not isinstance(e["text"], str):
        issues.append(
            _issue("schema.text_not_str", "text must be a string", document_id, entity_index)
        )

    candidates = e["candidates"]
    if not isinstance(candidates, (list, tuple)):
        issues.append(
            _issue(
                "schema.candidates_not_list",
                f"candidates must be a list, got {type(candidates)!r}",
                document_id,
                entity_index,
            )
        )
    else:
        if any(not isinstance(c, str) for c in candidates):
            issues.append(
                _issue(
                    "schema.candidate_not_str",
                    "every candidate must be a string code",
                    document_id,
                    entity_index,
                )
            )
        # Ontology/type consistency: a type that carries no candidates must not
        # have any (spec section 7.3 forbids cross-linking).
        etype = e.get("type")
        if etype in CANDIDATE_ONTOLOGY_BY_TYPE:
            ontology = CANDIDATE_ONTOLOGY_BY_TYPE[etype]
            if ontology is None and len(candidates) > 0:
                issues.append(
                    _issue(
                        "schema.candidates_not_allowed_for_type",
                        f"type {etype!r} must not carry candidates (no ontology)",
                        document_id,
                        entity_index,
                        value=list(candidates),
                    )
                )

    # Delegate type + assertion checks.
    result = ValidationResult.from_issues(issues)
    result = result.merged_with(
        validate_entity_type(entity, document_id=document_id, entity_index=entity_index)
    )
    result = result.merged_with(
        validate_assertion_labels(entity, document_id=document_id, entity_index=entity_index)
    )
    return result


def validate_entities(
    entities: list[EntityLike],
    *,
    document_id: str | None = None,
) -> ValidationResult:
    """Run full schema validation over every entity in a document."""
    result = ValidationResult()
    for idx, entity in enumerate(entities):
        result = result.merged_with(
            validate_entity_schema(entity, document_id=document_id, entity_index=idx)
        )
    return result


__all__ = [
    "validate_entity_schema",
    "validate_entity_type",
    "validate_assertion_labels",
    "validate_entities",
]
