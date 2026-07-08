"""Duplicate detection (spec sections 7.3, 11, C7).

The organizer forbids text-only deduplication: the same surface form at two
different positions denotes two distinct entities and BOTH must be kept. A true
duplicate is an entity with the same ``(text, type, start, end)`` key — i.e. the
same text, same type, AND same absolute position.

* ``find_duplicate_groups`` — groups indices sharing an identity key.
* ``validate_no_duplicates`` — flags exact duplicates as errors.
* ``deduplicate`` — deterministically keeps the first occurrence of each key.
"""

from __future__ import annotations

from typing import Any

from ._coerce import EntityLike, as_entity_dict
from .results import Severity, ValidationIssue, ValidationResult

# Identity key: (text, type, start, end). Position is mandatory to the key so
# duplicate text at different positions stays distinct.
IdentityKey = tuple[str, object, int, int]


def _identity_key(entity_dict: dict[str, Any]) -> IdentityKey:
    pos = entity_dict.get("position")
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        start, end = int(pos[0]), int(pos[1])
    else:
        # Fall back to sentinel offsets; malformed positions are caught by
        # validator.offsets. Keep them distinct so we do not merge blindly.
        start, end = -1, -1
    return (
        str(entity_dict.get("text")),
        entity_dict.get("type"),
        start,
        end,
    )


def find_duplicate_groups(entities: list[EntityLike]) -> list[list[int]]:
    """Return groups (lists of indices) of entities that share an identity key.

    Only groups with more than one member are returned. Order is stable: groups
    appear in order of first occurrence, indices within a group are ascending.
    """
    buckets: dict[IdentityKey, list[int]] = {}
    order: list[IdentityKey] = []
    for idx, entity in enumerate(entities):
        key = _identity_key(as_entity_dict(entity))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(idx)
    return [buckets[k] for k in order if len(buckets[k]) > 1]


def validate_no_duplicates(
    entities: list[EntityLike],
    *,
    document_id: str | None = None,
) -> ValidationResult:
    """Flag exact ``(text, type, start, end)`` duplicates as errors."""
    issues: list[ValidationIssue] = []
    for group in find_duplicate_groups(entities):
        first, *rest = group
        for dup_index in rest:
            key = _identity_key(as_entity_dict(entities[dup_index]))
            issues.append(
                ValidationIssue(
                    code="duplicate.exact",
                    message=(
                        f"entity {dup_index} is an exact duplicate of entity {first} "
                        f"(text/type/position identical): {key}"
                    ),
                    severity=Severity.ERROR,
                    document_id=document_id,
                    entity_index=dup_index,
                    context={"first_index": first, "identity_key": key},
                )
            )
    return ValidationResult.from_issues(issues)


def deduplicate(entities: list[EntityLike]) -> tuple[list[EntityLike], ValidationResult]:
    """Deterministically drop exact duplicates, keeping the first occurrence.

    Returns the deduplicated list (original order preserved) and a result whose
    WARNING issues record what was merged away. Distinct entities that merely
    share text are always retained.
    """
    seen: set[IdentityKey] = set()
    kept: list[EntityLike] = []
    issues: list[ValidationIssue] = []
    first_index: dict[IdentityKey, int] = {}
    for idx, entity in enumerate(entities):
        key = _identity_key(as_entity_dict(entity))
        if key in seen:
            issues.append(
                ValidationIssue(
                    code="duplicate.merged",
                    message=(
                        f"dropped exact duplicate at index {idx} "
                        f"(kept index {first_index[key]}): {key}"
                    ),
                    severity=Severity.WARNING,
                    entity_index=idx,
                    context={"kept_index": first_index[key], "identity_key": key},
                )
            )
            continue
        seen.add(key)
        first_index[key] = idx
        kept.append(entity)
    return kept, ValidationResult.from_issues(issues)


__all__ = ["find_duplicate_groups", "validate_no_duplicates", "deduplicate"]
