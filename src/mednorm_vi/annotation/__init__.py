"""Team-owned annotation contracts, validation, agreement, and adjudication.

Annotations are the only legitimate source of development gold. Only ADJUDICATED
(or explicitly approved GOLD) records count as gold by default. Organizer test
inputs are never labeled.
"""

from __future__ import annotations

from .adjudication import make_adjudication, validate_adjudication
from .agreement import corpus_agreement, document_agreement
from .export import export_gold_dataset
from .models import (
    ACCEPTED_GOLD_STATUSES,
    AdjudicationRecord,
    AgreementSummary,
    AnnotationEntity,
    DatasetManifest,
    IntendedUse,
    Provenance,
    ReviewStatus,
)
from .validation import validate_annotation_entity, validate_dataset_dir, validate_manifest

__all__ = [
    "AnnotationEntity",
    "DatasetManifest",
    "AgreementSummary",
    "AdjudicationRecord",
    "ReviewStatus",
    "IntendedUse",
    "Provenance",
    "ACCEPTED_GOLD_STATUSES",
    "validate_annotation_entity",
    "validate_manifest",
    "validate_dataset_dir",
    "document_agreement",
    "corpus_agreement",
    "make_adjudication",
    "validate_adjudication",
    "export_gold_dataset",
]
