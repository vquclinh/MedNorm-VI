"""Duplicate and leakage detection for train/dev/test construction."""

from __future__ import annotations

import hashlib

from .models import CanonicalDocument


def text_hash(document: CanonicalDocument) -> str:
    return hashlib.sha256(document.text.encode("utf-8")).hexdigest()


def duplicate_text_groups(documents: tuple[CanonicalDocument, ...]) -> tuple[tuple[str, ...], ...]:
    groups: dict[str, list[str]] = {}
    for document in documents:
        groups.setdefault(text_hash(document), []).append(document.document_id)
    dupes = [tuple(sorted(ids)) for ids in groups.values() if len(ids) > 1]
    return tuple(sorted(dupes))


def fold_leakage(
    folds: dict[str, tuple[str, ...]], documents: tuple[CanonicalDocument, ...]
) -> tuple[str, ...]:
    by_id = {document.document_id: document for document in documents}
    owner: dict[str, str] = {}
    leaks: list[str] = []
    for fold, doc_ids in folds.items():
        for doc_id in doc_ids:
            h = text_hash(by_id[doc_id])
            if h in owner and owner[h] != fold:
                leaks.append(f"text_hash:{doc_id}:{owner[h]}->{fold}")
            owner[h] = fold
    return tuple(sorted(leaks))


__all__ = ["duplicate_text_groups", "fold_leakage", "text_hash"]
