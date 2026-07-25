"""Organizer input-update comparison, classification, and safety (Audit 0015).

Synthetic fixtures only — no real BTC clinical text is used.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml

from mednorm_vi.round2 import (
    classify_input_change,
    fingerprint,
    fingerprint_archive,
    temporary_input_extraction,
    unsafe_zip_members,
)
from mednorm_vi.round2.input_update import (
    CONTENT_CHANGED,
    EXACTLY_IDENTICAL,
    INVALID_OR_INCOMPLETE,
    REPACKAGED_ONLY,
    SCHEMA_OR_LAYOUT_CHANGED,
)

REPO = Path(__file__).resolve().parents[2]
ACTIVE_INPUT_CFG = REPO / "configs" / "organizer" / "active_input.yaml"


def _write(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    return root


# --- fingerprint --------------------------------------------------------------

def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", {"1.txt": b"hello\n", "2.txt": b"world\n"})
    b = _write(tmp_path / "b", {"1.txt": b"hello\n", "2.txt": b"world\n"})
    fa, fb = fingerprint(a), fingerprint(b)
    assert fa.tree_hash == fb.tree_hash
    assert fa.file_count == 2 and fa.total_bytes == 12


# --- classification -----------------------------------------------------------

def test_exactly_identical(tmp_path: Path) -> None:
    old = _write(tmp_path / "o", {"1.txt": b"same\n"})
    new = _write(tmp_path / "n", {"1.txt": b"same\n"})
    assert classify_input_change(old, new).classification == EXACTLY_IDENTICAL


def test_repackaged_only_newline_and_bom(tmp_path: Path) -> None:
    old = _write(tmp_path / "o", {"1.txt": b"a\nb\n"})
    new = _write(tmp_path / "n", {"1.txt": b"\xef\xbb\xbfa\r\nb\r\n"})  # BOM + CRLF
    r = classify_input_change(old, new)
    assert r.classification == REPACKAGED_ONLY
    assert r.newline_or_bom_only == 1 and r.genuinely_changed == 0


def test_content_changed(tmp_path: Path) -> None:
    old = _write(tmp_path / "o", {"1.txt": b"paracetamol 500mg\n", "2.txt": b"x\n"})
    new = _write(tmp_path / "n", {"1.txt": b"amoxicillin 250mg\n", "2.txt": b"x\n"})
    r = classify_input_change(old, new)
    assert r.classification == CONTENT_CHANGED
    assert r.genuinely_changed == 1 and r.byte_identical == 1
    assert r.modified == ("1.txt",)


def test_schema_or_layout_changed(tmp_path: Path) -> None:
    old = _write(tmp_path / "o", {"1.txt": b"a\n"})
    new = _write(tmp_path / "n", {"1.txt": b"a\n", "2.txt": b"b\n"})
    r = classify_input_change(old, new)
    assert r.classification == SCHEMA_OR_LAYOUT_CHANGED
    assert r.added == ("2.txt",) and r.removed == ()


def test_invalid_when_new_empty(tmp_path: Path) -> None:
    old = _write(tmp_path / "o", {"1.txt": b"a\n"})
    new = (tmp_path / "n")
    new.mkdir()
    assert classify_input_change(old, new).classification == INVALID_OR_INCOMPLETE


# --- zip safety ---------------------------------------------------------------

def test_safe_zip_has_no_unsafe_members(tmp_path: Path) -> None:
    z = tmp_path / "safe.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("input/1.txt", "ok")
    assert unsafe_zip_members(z) == ()


def test_zip_traversal_is_detected(tmp_path: Path) -> None:
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("input/1.txt", "ok")
        zf.writestr("../evil.txt", "pwn")
    unsafe = unsafe_zip_members(z)
    assert "../evil.txt" in unsafe


def test_temporary_extraction_yields_input_root_and_cleans_up(tmp_path: Path) -> None:
    z = tmp_path / "release.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("input/1.txt", "alpha\n")
        zf.writestr("input/2.txt", "beta\n")
    seen: Path | None = None
    with temporary_input_extraction(z) as root:
        seen = root
        assert root.name == "input"
        assert {p.name for p in root.iterdir()} == {"1.txt", "2.txt"}
    assert seen is not None and not seen.exists()   # cleaned up on exit


def test_temporary_extraction_refuses_unsafe_archive(tmp_path: Path) -> None:
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "pwn")
    with pytest.raises(ValueError):
        with temporary_input_extraction(z):
            pass


def test_fingerprint_archive_matches_extracted(tmp_path: Path) -> None:
    src = _write(tmp_path / "input", {"1.txt": b"a\n", "2.txt": b"bb\n"})
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in sorted(src.iterdir()):
            zf.write(p, arcname=f"input/{p.name}")
    assert fingerprint_archive(z).tree_hash == fingerprint(src).tree_hash


# --- active-input descriptor --------------------------------------------------

def test_active_input_descriptor_is_consistent() -> None:
    cfg = yaml.safe_load(ACTIVE_INPUT_CFG.read_text(encoding="utf-8"))
    active = cfg["active_input"]
    assert active["id"] == "turn2_vong1"
    assert active["path"] == "data/organizer_test/input"
    assert active["archive_path"] == \
        "data/organizer_test/archives/turn2_vong1/input_turn2_vong1.zip"
    assert len(active["payload_tree_hash"]) == 64
    assert active["document_count"] == 100
    prev = cfg["previous_input"]
    assert prev["id"] == "initial_release"
    assert prev["extracted_path"] is None                 # archive-only, not extracted
    assert prev["archive_path"] == \
        "data/organizer_test/archives/initial_release/input.zip"
    assert prev["recoverable_from_archive"] is True
    assert cfg["update"]["classification"] == "CONTENT_CHANGED"
    assert cfg["update"]["activated"] is True
    assert cfg["raw_data_tracking"] == "prohibited"


def test_active_descriptor_paths_are_not_hash_or_snapshot_based() -> None:
    # The descriptor's PATH values must be human-readable — no snapshots/ dir and
    # no hash-named path. (Hashes are fine as metadata VALUES, not as paths.)
    cfg = yaml.safe_load(ACTIVE_INPUT_CFG.read_text(encoding="utf-8"))
    paths = [
        cfg["active_input"]["path"],
        cfg["active_input"]["archive_path"],
        cfg["previous_input"]["archive_path"],
    ]
    for p in paths:
        assert "snapshots/" not in p
        assert "btc-input-prior" not in p and "btc-input-turn2-vong1" not in p
        assert "56244aa3e772dd67" not in p and "d7ace1838a3433ba" not in p


def test_active_descriptor_has_no_raw_text() -> None:
    # heuristic: descriptor should be small metadata, not a corpus dump
    assert ACTIVE_INPUT_CFG.stat().st_size < 8192
