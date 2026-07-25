"""MedNorm-VI Data Engine contracts and deterministic builders."""

from .build import DatasetBuild, build_dataset
from .corpus_build import build_governed_corpus, concrete_type_mapping
from .models import CanonicalAnnotation, CanonicalDocument, DatasetManifest
from .public_ner import require_training_allowed
from .validation import validate_annotations
from .vietmed_ner import (
    convert_rows as vietmed_convert_rows,
)
from .vietmed_ner import (
    load_vietmed_artifacts,
    load_vietmed_mapping,
)

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
    "vietmed_convert_rows",
    "load_vietmed_artifacts",
    "load_vietmed_mapping",
]
