"""Deterministic dataset build manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .folds import make_folds
from .leakage import duplicate_text_groups, fold_leakage
from .models import CanonicalDocument, DatasetManifest
from .validation import validate_annotations


@dataclass(frozen=True, slots=True)
class DatasetBuild:
    documents: tuple[CanonicalDocument, ...]
    manifest: DatasetManifest
    errors: tuple[str, ...]


def _build_hash(documents: tuple[CanonicalDocument, ...], folds: dict[str, tuple[str, ...]]) -> str:
    payload = {
        "documents": [
            {
                "document_id": d.document_id,
                "source_id": d.source_id,
                "template_id": d.template_id,
                "text_sha256": hashlib.sha256(d.text.encode("utf-8")).hexdigest(),
                "annotations": [
                    {
                        "id": a.annotation_id,
                        "span": [a.span.start, a.span.end],
                        "type": a.entity_type,
                        "assertions": a.assertions,
                        "candidates": a.candidates,
                    }
                    for a in d.annotations
                ],
            }
            for d in sorted(documents, key=lambda doc: doc.document_id)
        ],
        "folds": folds,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_dataset(documents: tuple[CanonicalDocument, ...], *, folds: int = 5) -> DatasetBuild:
    errors: list[str] = []
    for document in documents:
        validation = validate_annotations(document)
        errors.extend(f"{document.document_id}:{error}" for error in validation.errors)
    fold_map = make_folds(documents, k=folds)
    errors.extend(f"leakage:{leak}" for leak in fold_leakage(fold_map, documents))
    warnings = tuple(
        f"duplicate_group:{','.join(group)}" for group in duplicate_text_groups(documents)
    )
    manifest = DatasetManifest(
        dataset_id=f"dataset-{_build_hash(documents, fold_map)[:16]}",
        source_ids=tuple(sorted({d.source_id for d in documents if d.source_id})),
        document_count=len(documents),
        annotation_count=sum(len(d.annotations) for d in documents),
        build_hash=_build_hash(documents, fold_map),
        folds=fold_map,
        warnings=warnings,
    )
    return DatasetBuild(documents=documents, manifest=manifest, errors=tuple(sorted(errors)))


__all__ = ["DatasetBuild", "build_dataset"]
