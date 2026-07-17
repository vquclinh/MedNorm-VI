"""Local Vietnamese ICD-10 snapshot ingestion + comparison (Phase 1C-A)."""

from __future__ import annotations

from pathlib import Path

import yaml

from mednorm_vi.kb.icd10 import (
    ColumnMap,
    build_snapshot,
    diff_snapshots,
    snapshot_stats,
    validate_snapshot,
)
from mednorm_vi.kb.icd10 import normalization as norm

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "kb" / "icd10"
CM = ColumnMap.from_mapping(yaml.safe_load((FIX / "columns.yaml").read_text(encoding="utf-8")))
A = build_snapshot(FIX / "snapshot_a.csv", CM, source="MoH-VI", version="a")
B = build_snapshot(FIX / "snapshot_b.csv", CM, source="MoH-VI", version="b")


def test_dotted_undotted_reversible() -> None:
    assert norm.to_undotted("A09.9") == "A099"
    assert norm.to_dotted("A099") == "A09.9"
    assert norm.to_dotted("A09") == "A09"  # category unchanged
    assert norm.specificity("A09.9") == 1 and norm.is_category("A09")


def test_ingestion_and_hierarchy() -> None:
    assert validate_snapshot(A).ok
    c = A.get("A099")  # undotted lookup resolves dotted concept
    assert c is not None and c.dotted == "A09.9" and c.parent == "A09"
    assert A.children_of("A09") == ("A099",)
    assert snapshot_stats(A)["categories"] == 3


def test_dotted_undotted_alias_lookup() -> None:
    # snapshot B supplies A09.9 in UNDOTTED form; lookups still resolve either way.
    assert B.get("A09.9") is not None
    assert B.get("A099") is not None
    assert B.get("A09.9").undotted == "A099"


def test_snapshot_comparison() -> None:
    d = diff_snapshots(A, B)
    assert d.only_right == ("E1165",)
    assert [c for c, _, _ in d.label_changed] == ["A099"]
    assert [c for c, _, _ in d.format_changed] == ["A099"]  # dotted vs undotted supplied
    assert [c for c, _, _ in d.alias_changed] == ["I10"]


def test_snapshot_id_deterministic() -> None:
    assert build_snapshot(FIX / "snapshot_a.csv", CM, source="MoH-VI", version="a").snapshot_id \
        == A.snapshot_id
