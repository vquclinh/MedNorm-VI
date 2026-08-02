"""Organizer-schema projection (sprint 0075 packaging hotfix).

`assertions` is legal only for TRIỆU_CHỨNG / CHẨN_ĐOÁN / THUỐC and must be ABSENT - not empty -
for the two laboratory types. Emitting it on a lab entity is an unsupported field, which
blocked packaging of the full-8B run on 385 entities (230 KẾT_QUẢ_XÉT_NGHIỆM + 155
TÊN_XÉT_NGHIỆM).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mednorm_vi.reasoner.validator import ORGANIZER_TYPES, validate

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/sanitize_organizer_schema_0075.py"

spec = importlib.util.spec_from_file_location("sanitize_0075", SCRIPT)
assert spec and spec.loader
sanitize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanitize)

NOTE = "sốt cao, glucose 7.2, xét nghiệm máu, viêm phổi, paracetamol"

#: The shapes the scored baseline emits, per type. This is the contract.
EXPECTED_FIELDS = {
    "CHẨN_ĐOÁN": {"text", "type", "position", "assertions", "candidates"},
    "THUỐC": {"text", "type", "position", "assertions", "candidates"},
    "TRIỆU_CHỨNG": {"text", "type", "position", "assertions"},
    "TÊN_XÉT_NGHIỆM": {"text", "type", "position"},
    "KẾT_QUẢ_XÉT_NGHIỆM": {"text", "type", "position"},
}


@pytest.mark.parametrize("entity_type", ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"])
def test_assertions_retained_for_symptom_diagnosis_drug(entity_type: str) -> None:
    entities, _ = validate(
        NOTE, [{"text": "sốt", "type": entity_type, "assertions": {"isNegated": True}}]
    )
    row = entities[0].as_organizer_json()
    assert row["assertions"] == ["isNegated"]


@pytest.mark.parametrize("entity_type", ["TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"])
def test_assertions_absent_from_both_lab_types(entity_type: str) -> None:
    """Absent, not empty - an empty list is still an unsupported field."""
    entities, _ = validate(
        NOTE, [{"text": "glucose", "type": entity_type, "assertions": {"isNegated": True}}]
    )
    row = entities[0].as_organizer_json()
    assert "assertions" not in row
    assert set(row) == EXPECTED_FIELDS[entity_type]


@pytest.mark.parametrize("entity_type", sorted(ORGANIZER_TYPES))
def test_every_type_emits_exactly_its_allowed_fields(entity_type: str) -> None:
    entities, _ = validate(
        NOTE,
        [{"text": "sốt", "type": entity_type, "assertions": {"isNegated": True},
          "pool_key": "0", "candidates": ["X"]}],
        {"0": {"X"}},
    )
    assert set(entities[0].as_organizer_json()) == EXPECTED_FIELDS[entity_type]


@pytest.mark.parametrize("entity_type", ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"])
def test_candidates_only_on_diagnosis_and_drug(entity_type: str) -> None:
    entities, _ = validate(NOTE, [{"text": "sốt", "type": entity_type}])
    assert "candidates" not in entities[0].as_organizer_json()


# ------------------------------------------------------------------ migration projection


def test_projection_removes_only_unsupported_fields_and_keeps_values() -> None:
    entity = {"text": "glucose", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [3, 10],
              "assertions": ["isNegated"]}
    projected = sanitize.project(entity)
    assert projected == {"text": "glucose", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [3, 10]}


def test_projection_preserves_values_and_order_for_allowed_fields() -> None:
    entity = {"text": "viêm phổi", "type": "CHẨN_ĐOÁN", "position": [1, 9],
              "assertions": ["isHistorical"], "candidates": ["J18.9", "J18.0"]}
    projected = sanitize.project(entity)
    assert projected == entity
    assert list(projected)[:3] == ["text", "type", "position"]
    assert projected["candidates"] == ["J18.9", "J18.0"]  # order untouched


def test_migration_preserves_entity_count_and_order(tmp_path: Path) -> None:
    rows = [
        {"text": "a", "type": "TRIỆU_CHỨNG", "position": [0, 1], "assertions": []},
        {"text": "b", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [2, 3], "assertions": []},
        {"text": "c", "type": "CHẨN_ĐOÁN", "position": [4, 5], "assertions": [],
         "candidates": ["J18.9"]},
    ]
    source = tmp_path / "src"
    source.mkdir()
    (source / "1.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    assert sanitize.main(
        ["--source-dir", str(source), "--output-dir", str(out), "--expected-documents", "1"]
    ) == 0
    after = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert [e["text"] for e in after] == ["a", "b", "c"]
    assert "assertions" not in after[1]
    assert after[2]["candidates"] == ["J18.9"]


def test_migration_fails_closed_on_wrong_document_count(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    assert sanitize.main(
        ["--source-dir", str(source), "--output-dir", str(tmp_path / "o"),
         "--expected-documents", "100"]
    ) == 2


def test_strict_organizer_validator_is_not_weakened() -> None:
    """The fix is upstream serialization; L9/derivation validation is untouched."""
    from mednorm_vi.inference import derive_submission

    source = (ROOT / "src/mednorm_vi/inference/derive_submission.py").read_text(
        encoding="utf-8"
    )
    assert "DerivationRefused" in source
    assert derive_submission.DERIVE_ICD_DOTTED == "icd_dotted"
    # No suppression of unsupported-field errors anywhere in the hotfix.
    hotfix = SCRIPT.read_text(encoding="utf-8")
    for banned in ("unsupported_field", "ignore_errors", "skip_validation", "--force"):
        assert banned not in hotfix
