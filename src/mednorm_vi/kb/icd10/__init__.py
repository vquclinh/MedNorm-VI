"""Local-only Vietnamese ICD-10 snapshot interface (Phase 1C-A).

Ingests a locally acquired ICD-10 table (configurable columns) into typed
contracts and compares two local snapshots. No network, no real ICD content in
the repo, no final diagnosis code. Dotted/undotted output policy is UNRESOLVED.
"""

from __future__ import annotations

from . import normalization
from .forensics import IcdSnapshotDiff, diff_snapshots
from .loaders import ColumnMap, RawIcdRow, load_rows
from .manifest import snapshot_identity, snapshot_stats
from .models import IcdConcept, IcdSnapshot
from .snapshot import build_snapshot, build_snapshot_from_rows
from .validation import validate_snapshot

__all__ = [
    "normalization",
    "IcdConcept",
    "IcdSnapshot",
    "ColumnMap",
    "RawIcdRow",
    "load_rows",
    "build_snapshot",
    "build_snapshot_from_rows",
    "diff_snapshots",
    "IcdSnapshotDiff",
    "snapshot_identity",
    "snapshot_stats",
    "validate_snapshot",
]
