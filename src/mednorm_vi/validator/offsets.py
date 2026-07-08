"""Offset-invariant validation (spec section 4).

Enforces the mandatory invariant ``original_text[start:end] == entity.text`` and
the end-exclusive ``[start, end)`` position convention. Indexing is by Unicode
code point (Python ``str`` slicing), matching the spec's code-point alignment.

This is the single most important guard in the pipeline: normalization must
never move an emitted span, and no LLM may compute offsets freely.
"""

from __future__ import annotations

from ._coerce import EntityLike, as_entity_dict
from .results import Severity, ValidationIssue, ValidationResult


def _position_pair(position: object) -> tuple[int, int] | None:
    """Return ``(start, end)`` if ``position`` is a valid 2-int pair, else None."""
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        return None
    start, end = position
    # bool is a subclass of int; reject it explicitly to avoid True/False offsets.
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return start, end


def validate_offset_invariant(
    original_text: str,
    entity: EntityLike,
    *,
    document_id: str | None = None,
    entity_index: int | None = None,
) -> ValidationResult:
    """Validate the substring + end-exclusive offset invariants for one entity."""
    e = as_entity_dict(entity)
    issues: list[ValidationIssue] = []

    def add(code: str, message: str, **ctx: object) -> None:
        issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=Severity.ERROR,
                document_id=document_id,
                entity_index=entity_index,
                context=ctx,
            )
        )

    text = e.get("text")
    if not isinstance(text, str):
        add("offset.text_not_str", f"entity text must be a string, got {type(text)!r}")
        return ValidationResult.from_issues(issues)
    if text == "":
        add("offset.empty_text", "entity text must be non-empty")

    pair = _position_pair(e.get("position"))
    if pair is None:
        add(
            "offset.bad_position",
            f"position must be a 2-integer [start, end] pair, got {e.get('position')!r}",
        )
        return ValidationResult.from_issues(issues)

    start, end = pair
    if start < 0:
        add("offset.negative_start", f"start must be >= 0, got {start}")
    if end < start:
        add("offset.end_before_start", f"end {end} must be >= start {start} (end-exclusive)")
    if end > len(original_text):
        add(
            "offset.out_of_bounds",
            f"end {end} exceeds original_text length {len(original_text)}",
        )

    # End-exclusive length check: [start, end) must span exactly len(text) code points.
    if end >= start and (end - start) != len(text):
        add(
            "offset.length_mismatch",
            f"[start, end) length {end - start} != len(text) {len(text)} "
            "(positions must be end-exclusive)",
        )

    # The core invariant. Only meaningful if indices are in range.
    if start >= 0 and end <= len(original_text) and end >= start:
        actual = original_text[start:end]
        if actual != text:
            add(
                "offset.substring_mismatch",
                f"original_text[{start}:{end}] == {actual!r} != entity.text {text!r}",
                actual=actual,
                expected=text,
            )

    return ValidationResult.from_issues(issues)


def validate_offsets(
    original_text: str,
    entities: list[EntityLike],
    *,
    document_id: str | None = None,
) -> ValidationResult:
    """Validate the offset invariant for every entity in a document."""
    result = ValidationResult()
    for idx, entity in enumerate(entities):
        result = result.merged_with(
            validate_offset_invariant(
                original_text, entity, document_id=document_id, entity_index=idx
            )
        )
    return result


__all__ = ["validate_offset_invariant", "validate_offsets"]
