"""Submission layout tests — confirmed layout: output.zip -> output/1..100.json."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from mednorm_vi.schemas import EntityPrediction, Span, SpanCoordinates
from mednorm_vi.validator import (
    expected_filenames,
    validate_output_directory,
    validate_submission_zip,
    write_submission_zip,
)


def _valid_doc_json() -> str:
    entities = [
        EntityPrediction("ho", "SYMPTOM", SpanCoordinates(absolute=Span(0, 2)),
                         assertions=("isHistorical",)).to_submission_dict(),
        EntityPrediction("viêm phổi", "DIAGNOSIS", SpanCoordinates(absolute=Span(3, 12)),
                         candidates=("J18.9",)).to_submission_dict(),
    ]
    return json.dumps(entities, ensure_ascii=False)


def _populate(dir_path: Path, n: int = 100, content: str | None = None) -> None:
    payload = content if content is not None else _valid_doc_json()
    for name in expected_filenames(n):
        (dir_path / name).write_text(payload, encoding="utf-8")


# --- expected filenames ------------------------------------------------------

def test_expected_filenames() -> None:
    names = expected_filenames()
    assert len(names) == 100
    assert names[0] == "1.json" and names[-1] == "100.json"


# --- directory validation ----------------------------------------------------

def test_complete_directory_passes(tmp_path: Path) -> None:
    _populate(tmp_path)
    assert validate_output_directory(tmp_path).ok


def test_missing_file_fails(tmp_path: Path) -> None:
    _populate(tmp_path)
    (tmp_path / "100.json").unlink()
    result = validate_output_directory(tmp_path)
    assert not result.ok
    assert any(i.code == "submission.file_missing" for i in result.errors)


def test_extra_prediction_json_fails(tmp_path: Path) -> None:
    _populate(tmp_path)
    (tmp_path / "101.json").write_text(_valid_doc_json(), encoding="utf-8")
    result = validate_output_directory(tmp_path)
    assert not result.ok
    assert any(i.code == "submission.extra_prediction_file" for i in result.errors)


def test_invalid_json_root_type_fails(tmp_path: Path) -> None:
    _populate(tmp_path)
    (tmp_path / "1.json").write_text('{"not": "a list"}', encoding="utf-8")
    result = validate_output_directory(tmp_path)
    assert not result.ok
    assert any(i.code == "submission.root_not_list" for i in result.errors)


def test_english_type_in_file_fails(tmp_path: Path) -> None:
    _populate(tmp_path)
    bad = json.dumps([{"text": "ho", "type": "SYMPTOM", "assertions": [], "position": [0, 2]}])
    (tmp_path / "2.json").write_text(bad, encoding="utf-8")
    result = validate_output_directory(tmp_path)
    assert not result.ok
    assert any(i.code == "organizer.english_type" for i in result.errors)


def test_metadata_file_is_warning_not_error(tmp_path: Path) -> None:
    _populate(tmp_path)
    (tmp_path / "README.txt").write_text("notes", encoding="utf-8")
    result = validate_output_directory(tmp_path)
    assert result.ok
    assert any(i.code == "submission.metadata_file" for i in result.warnings)


# --- ZIP validation ----------------------------------------------------------

def test_valid_zip_passes(tmp_path: Path) -> None:
    src = tmp_path / "output"
    src.mkdir()
    _populate(src)
    zip_path = tmp_path / "output.zip"
    write_submission_zip(src, zip_path)
    assert validate_submission_zip(zip_path).ok


def test_zip_wrong_top_level_dir_fails(tmp_path: Path) -> None:
    zip_path = tmp_path / "output.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in expected_filenames():
            zf.writestr(f"predictions/{name}", _valid_doc_json())
    result = validate_submission_zip(zip_path)
    assert not result.ok
    codes = {i.code for i in result.errors}
    assert "submission.wrong_top_level_dir" in codes or "submission.missing_output_dir" in codes


def test_zip_missing_file_fails(tmp_path: Path) -> None:
    zip_path = tmp_path / "output.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in expected_filenames()[:-1]:  # omit 100.json
            zf.writestr(f"output/{name}", _valid_doc_json())
    result = validate_submission_zip(zip_path)
    assert not result.ok
    assert any(i.code == "submission.file_missing" for i in result.errors)


def test_zip_extra_prediction_json_fails(tmp_path: Path) -> None:
    src = tmp_path / "output"
    src.mkdir()
    _populate(src)
    zip_path = tmp_path / "output.zip"
    write_submission_zip(src, zip_path)
    # Append an extra prediction file.
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr("output/101.json", _valid_doc_json())
    result = validate_submission_zip(zip_path)
    assert not result.ok
    assert any(i.code == "submission.extra_prediction_file" for i in result.errors)


def test_zip_unsafe_path_flagged(tmp_path: Path) -> None:
    zip_path = tmp_path / "output.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in expected_filenames():
            zf.writestr(f"output/{name}", _valid_doc_json())
        zf.writestr("../evil.json", "[]")
    result = validate_submission_zip(zip_path)
    assert not result.ok
    assert any(i.code == "submission.unsafe_path" for i in result.errors)


def test_zip_metadata_is_warning(tmp_path: Path) -> None:
    src = tmp_path / "output"
    src.mkdir()
    _populate(src)
    zip_path = tmp_path / "output.zip"
    write_submission_zip(src, zip_path)
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr("README.txt", "metadata")
    result = validate_submission_zip(zip_path)
    assert result.ok  # metadata allowed, only reported
    assert any(i.code == "submission.metadata_file" for i in result.warnings)


def test_write_submission_zip_is_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "output"
    src.mkdir()
    _populate(src, n=5)
    z1 = write_submission_zip(src, tmp_path / "a.zip", n=5)
    z2 = write_submission_zip(src, tmp_path / "b.zip", n=5)
    assert z1.read_bytes() == z2.read_bytes()  # byte-reproducible packaging
