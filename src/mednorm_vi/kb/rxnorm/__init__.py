"""Local-only RxNorm snapshot interface (Phase 1C-A).

Ingests a locally acquired RxNorm RRF snapshot into typed contracts, compares two
local snapshots, and resolves legacy↔current RXCUIs. No network, no real RxNorm
content in the repo, no final medication decode. Provisional decoding-policy
hypotheses live in ``configs/organizer/rxnorm_decoding_hypotheses_v1.yaml``.
"""

from __future__ import annotations

from .compatibility import RemapResolution, is_active, resolve_current
from .discovery import RrfFileSet, discover_rrf
from .forensics import SnapshotDiff, diff_snapshots, lookup_mention
from .manifest import snapshot_identity, snapshot_stats
from .models import (
    RxnormAtom,
    RxnormAttribute,
    RxnormRelation,
    RxnormSemanticType,
    RxnormSnapshot,
)
from .snapshot import build_snapshot
from .validation import validate_snapshot

__all__ = [
    "RxnormAtom",
    "RxnormRelation",
    "RxnormAttribute",
    "RxnormSemanticType",
    "RxnormSnapshot",
    "discover_rrf",
    "RrfFileSet",
    "build_snapshot",
    "resolve_current",
    "is_active",
    "RemapResolution",
    "diff_snapshots",
    "SnapshotDiff",
    "lookup_mention",
    "snapshot_identity",
    "snapshot_stats",
    "validate_snapshot",
]
