"""Deterministic source/template-aware fold assignment."""

from __future__ import annotations

import hashlib

from .models import CanonicalDocument


def group_key(document: CanonicalDocument) -> str:
    if document.template_id:
        return f"template:{document.template_id}"
    if document.source_id:
        return f"source:{document.source_id}"
    return f"doc:{document.document_id}"


def make_folds(
    documents: tuple[CanonicalDocument, ...], *, k: int = 5
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for document in documents:
        grouped.setdefault(group_key(document), []).append(document.document_id)
    folds: dict[str, list[str]] = {f"fold_{i}": [] for i in range(k)}
    for key, ids in sorted(grouped.items()):
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % k
        folds[f"fold_{bucket}"].extend(sorted(ids))
    return {name: tuple(sorted(ids)) for name, ids in sorted(folds.items())}


__all__ = ["group_key", "make_folds"]
