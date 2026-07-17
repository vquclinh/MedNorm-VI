"""Exact-offset and schema validation for canonical annotations."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import CanonicalDocument


@dataclass(frozen=True, slots=True)
class AnnotationValidation:
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def validate_annotations(document: CanonicalDocument) -> AnnotationValidation:
    errors: list[str] = []
    seen: set[str] = set()
    for ann in document.annotations:
        if ann.annotation_id in seen:
            errors.append(f"duplicate_annotation_id:{ann.annotation_id}")
        seen.add(ann.annotation_id)
        if ann.document_id != document.document_id:
            errors.append(f"annotation_document_mismatch:{ann.annotation_id}")
        if ann.span.end > len(document.text):
            errors.append(f"span_out_of_bounds:{ann.annotation_id}")
            continue
        if document.text[ann.span.start : ann.span.end] != ann.text:
            errors.append(f"offset_text_mismatch:{ann.annotation_id}")
        if not ann.is_valid_type():
            errors.append(f"invalid_type:{ann.annotation_id}:{ann.entity_type}")
        if not ann.assertions_valid():
            errors.append(f"invalid_assertion:{ann.annotation_id}")
    return AnnotationValidation(ok=not errors, errors=tuple(errors))


__all__ = ["AnnotationValidation", "validate_annotations"]
