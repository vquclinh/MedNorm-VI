"""MedNorm-VI Data Engine contracts and deterministic builders."""

from .build import DatasetBuild, build_dataset
from .models import CanonicalAnnotation, CanonicalDocument, DatasetManifest
from .public_ner import require_training_allowed
from .validation import validate_annotations

__all__ = [
    "CanonicalAnnotation",
    "CanonicalDocument",
    "DatasetBuild",
    "DatasetManifest",
    "build_dataset",
    "require_training_allowed",
    "validate_annotations",
]
