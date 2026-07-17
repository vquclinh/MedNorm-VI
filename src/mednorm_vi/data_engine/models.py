"""Canonical data contracts used by training stages S0-S6."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.constants import ASSERTION_LABELS, ENTITY_TYPES
from ..schemas.spans import Span


@dataclass(frozen=True, slots=True)
class CanonicalAnnotation:
    annotation_id: str
    document_id: str
    span: Span
    text: str
    entity_type: str
    assertions: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)
    source: str = ""
    confidence: float = 1.0

    def is_valid_type(self) -> bool:
        return self.entity_type in ENTITY_TYPES

    def assertions_valid(self) -> bool:
        return all(assertion in ASSERTION_LABELS for assertion in self.assertions)


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    document_id: str
    text: str
    source_id: str = ""
    template_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    annotations: tuple[CanonicalAnnotation, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    source_ids: tuple[str, ...]
    document_count: int
    annotation_count: int
    build_hash: str
    folds: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ReviewItem:
    item_id: str
    document_id: str
    annotation_id: str
    reason: str
    priority: float


@dataclass(frozen=True, slots=True)
class TeacherGenerationContract:
    """Specification for optional offline teacher generation.

    This is intentionally only a contract. Competition inference must not import
    or invoke teacher APIs.
    """

    task_id: str
    prompt_template_id: str
    allowed_output_schema: str
    stores_chain_of_thought: bool = False
    competition_inference_allowed: bool = False


__all__ = [
    "CanonicalAnnotation",
    "CanonicalDocument",
    "DatasetManifest",
    "ReviewItem",
    "TeacherGenerationContract",
]
