"""Snapshot identity + summary statistics for a local RxNorm snapshot."""

from __future__ import annotations

from collections import Counter

from ...resources.models import SnapshotIdentity
from .models import RxnormSnapshot


def snapshot_identity(snapshot: RxnormSnapshot) -> SnapshotIdentity:
    return SnapshotIdentity(
        snapshot_id=snapshot.snapshot_id, version=snapshot.release_version,
        release_date=snapshot.release_date, acquisition_date="")


def snapshot_stats(snapshot: RxnormSnapshot) -> dict[str, int]:
    """Deterministic counts for the doctor CLI / forensics summaries."""
    by_tty: Counter[str] = Counter(a.tty for a in snapshot.atoms)
    by_sab: Counter[str] = Counter(a.sab for a in snapshot.atoms)
    suppressed = sum(1 for c in snapshot.rxcuis() if snapshot.is_suppressed(c))
    stats = {
        "concepts": len(snapshot.rxcuis()),
        "atoms": len(snapshot.atoms),
        "relations": len(snapshot.relations),
        "attributes": len(snapshot.attributes),
        "semantic_types": len(snapshot.semantic_types),
        "suppressed_concepts": suppressed,
        "distinct_tty": len(by_tty),
        "distinct_sab": len(by_sab),
    }
    return stats


__all__ = ["snapshot_identity", "snapshot_stats"]
