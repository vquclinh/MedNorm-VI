"""Integration smoke test: build a valid output.zip, validate, corrupt, fail.

Exercises the full confirmed submission path end-to-end:
serialize organizer entities -> write 100 files -> package output.zip ->
validate OK -> corrupt one condition -> validate fails.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mednorm_vi.schemas import EntityPrediction, Span, SpanCoordinates
from mednorm_vi.validator import (
    expected_filenames,
    validate_submission_zip,
    write_submission_zip,
)


def _doc_payload() -> str:
    entities = [
        EntityPrediction(
            "paracetamol", "MEDICATION", SpanCoordinates(absolute=Span(0, 11)),
            candidates=("1049640",), assertions=("isHistorical",),
        ).to_submission_dict(),
        EntityPrediction(
            "viêm phổi", "DIAGNOSIS", SpanCoordinates(absolute=Span(12, 21)),
            candidates=("J18.9",),
        ).to_submission_dict(),
        EntityPrediction(
            "sốt", "SYMPTOM", SpanCoordinates(absolute=Span(22, 25)),
        ).to_submission_dict(),
    ]
    return json.dumps(entities, ensure_ascii=False)


def test_valid_zip_then_corruption_fails(tmp_path: Path) -> None:
    src = tmp_path / "output"
    src.mkdir()
    payload = _doc_payload()
    for name in expected_filenames():
        (src / name).write_text(payload, encoding="utf-8")

    zip_path = tmp_path / "output.zip"
    write_submission_zip(src, zip_path)

    # 1) The freshly built package validates cleanly.
    ok_result = validate_submission_zip(zip_path)
    assert ok_result.ok, [i.message for i in ok_result.errors]

    # 2) Corrupt exactly one condition: drop a required file, re-package, re-check.
    (src / "50.json").unlink()
    corrupt_zip = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(corrupt_zip, "w") as zf:
        for name in expected_filenames():
            fpath = src / name
            if fpath.exists():
                zf.writestr(f"output/{name}", fpath.read_text(encoding="utf-8"))
    bad_result = validate_submission_zip(corrupt_zip)
    assert not bad_result.ok
    assert any(i.code == "submission.file_missing" for i in bad_result.errors)


def test_utf8_preserved_through_packaging(tmp_path: Path) -> None:
    src = tmp_path / "output"
    src.mkdir()
    for name in expected_filenames(3):
        (src / name).write_text(_doc_payload(), encoding="utf-8")
    zip_path = tmp_path / "output.zip"
    write_submission_zip(src, zip_path, n=3)

    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read("output/1.json").decode("utf-8")
    assert "CHẨN_ĐOÁN" in raw and "\\u" not in raw
    assert validate_submission_zip(zip_path, n=3).ok
