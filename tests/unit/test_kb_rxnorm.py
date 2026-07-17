"""Local RxNorm snapshot ingestion, comparison, and remap (Phase 1C-A)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.kb.rxnorm import (
    build_snapshot,
    diff_snapshots,
    is_active,
    lookup_mention,
    resolve_current,
    snapshot_stats,
    validate_snapshot,
)

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "kb" / "rxnorm"
A = build_snapshot(FIX / "snapshot_a")
B = build_snapshot(FIX / "snapshot_b")


def test_ingestion_parses_rrf() -> None:
    assert set(A.rxcuis()) == {"100001", "100002", "100003"}
    assert A.preferred_string("100001") == "amlodipine"
    assert A.ttys("100002") == ("SCD",)
    assert validate_snapshot(A).ok


def test_suppressed_and_active() -> None:
    assert A.is_suppressed("100003") and not A.is_suppressed("100001")
    assert is_active(A, "100001") and not is_active(A, "100003")
    assert snapshot_stats(A)["suppressed_concepts"] == 1


def test_mention_lookup() -> None:
    assert lookup_mention(A, "amlodipine") == ("100001",)
    assert lookup_mention(A, "AMLODIPINE") == ("100001",)  # normalized


def test_snapshot_comparison() -> None:
    d = diff_snapshots(A, B)
    assert d.only_left == ("100003",)
    assert d.only_right == ("100004",)
    assert [c for c, _, _ in d.label_changed] == ["100002"]


def test_legacy_remapped_rxcui() -> None:
    # 100003 was retired in B and remapped_to 100002.
    res = resolve_current(B, "100003")
    assert res.was_remapped and res.resolved_rxcui == "100002"
    assert res.resolved_exists
    d = diff_snapshots(A, B)
    assert ("100003", "100002") in d.remapped


def test_snapshot_id_deterministic() -> None:
    assert build_snapshot(FIX / "snapshot_a").snapshot_id == A.snapshot_id
