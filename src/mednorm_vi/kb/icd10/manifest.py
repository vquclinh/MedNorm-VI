"""Snapshot identity + summary statistics for a local ICD-10 snapshot."""

from __future__ import annotations

from collections import Counter

from ...resources.models import SnapshotIdentity
from .models import IcdSnapshot


def snapshot_identity(snapshot: IcdSnapshot) -> SnapshotIdentity:
    return SnapshotIdentity(snapshot_id=snapshot.snapshot_id, version=snapshot.version,
                            release_date="", acquisition_date="")


def snapshot_stats(snapshot: IcdSnapshot) -> dict[str, int]:
    by_chapter: Counter[str] = Counter(c.chapter for c in snapshot.concepts)
    categories = sum(1 for c in snapshot.concepts if c.is_category)
    return {
        "concepts": len(snapshot.concepts),
        "categories": categories,
        "subcodes": len(snapshot.concepts) - categories,
        "chapters": len(by_chapter),
        "with_aliases": sum(1 for c in snapshot.concepts if c.aliases),
        "with_english": sum(1 for c in snapshot.concepts if c.label_en),
    }


__all__ = ["snapshot_identity", "snapshot_stats"]
