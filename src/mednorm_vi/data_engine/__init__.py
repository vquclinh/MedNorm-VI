"""MedNorm-VI Data Engine contracts and deterministic builders."""

from .build import DatasetBuild, build_dataset
from .corpus_build import build_governed_corpus, concrete_type_mapping
from .models import CanonicalAnnotation, CanonicalDocument, DatasetManifest
from .public_ner import require_training_allowed
from .validation import validate_annotations

__all__ = [
    "CanonicalAnnotation",
    "CanonicalDocument",
    "DatasetBuild",
    "DatasetManifest",
    "build_dataset",
    "build_governed_corpus",
    "concrete_type_mapping",
    "require_training_allowed",
    "validate_annotations",
]
