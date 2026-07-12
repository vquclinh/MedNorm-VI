"""Phase 1B stress tests: noisy, multilingual, and edge inputs stay valid."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from mednorm_vi.deterministic_baseline import Phase1BConfig, determinism_hash, run_phase1b
from mednorm_vi.document_intelligence import analyze_text

REPO = Path(__file__).resolve().parents[2]
CONFIG = Phase1BConfig.load(
    REPO / "configs" / "case_router" / "base.yaml",
    REPO / "configs" / "medication" / "grammar_v1.yaml",
    REPO / "configs" / "laboratory" / "parser_v1.yaml",
)


def _ok(text: str):
    g = analyze_text(text)
    res = run_phase1b(g, CONFIG)
    errors = [i.message for i in res.issues if i.severity.value == "error"]
    assert res.l1_valid and res.proposals_valid, errors
    for p in res.proposals:  # exact offsets everywhere
        assert g.original_text[p.start : p.end] == p.text
    assert determinism_hash(res) == determinism_hash(run_phase1b(g, CONFIG))
    return res


def test_crlf() -> None:
    _ok("Thuốc tại nhà:\r\n1. amlodipine 10 mg po daily\r\n\r\nXét nghiệm:\r\nWBC: 14,43\r\n")


def test_nfc_nfd() -> None:
    base = "Xét nghiệm:\nGlucose: 7,8 mmol/L\n"
    _ok(unicodedata.normalize("NFC", base))
    _ok(unicodedata.normalize("NFD", base))


def test_no_diacritics() -> None:
    _ok("Thuoc tai nha:\n1. amlodipine 10 mg po daily\nXet nghiem:\nWBC: 14.43\n")


def test_mixed_english_vietnamese() -> None:
    _ok("Home meds:\n1. paracetamol 500mg khi cần\nLabs:\nWBC: 14.43 (H)\n")


def test_repeated_spaces_and_tabs() -> None:
    _ok("Xét nghiệm:\nWBC:\t\t14,43\t\tH\nGlucose:    7.8   mmol/L\n")


def test_long_medication_line() -> None:
    _ok("Thuốc tại nhà:\n1. " + "amlodipine 10 mg po daily và " * 30 + "metformin 500 mg bid\n")


def test_semicolon_heavy_lab() -> None:
    _ok("Xét nghiệm:\n" + "; ".join(f"T{i}: {i}.{i}" for i in range(1, 15)) + "\n")


def test_duplicate_names_overlap() -> None:
    res = _ok("Thuốc tại nhà:\n1. aspirin 81 mg\n2. aspirin 81 mg\n")
    assert res.merge_diagnostics.repeated_surface


def test_unicode_dashes_slashes_micro() -> None:
    _ok("Thuốc tại nhà:\n1. paracetamol 325–650 mg\nXét nghiệm:\nCreatinine: 88 µmol/L\n")


def test_noisy_abbreviations() -> None:
    _ok("THA, DTD type 2.\nBN dùng amlodipine 10 mg po.\n")


def test_empty_document() -> None:
    res = _ok("")
    assert res.proposals == ()


def test_document_with_no_medical_structures() -> None:
    res = _ok("Hôm nay trời đẹp. Bệnh nhân cảm thấy khỏe và vui vẻ hơn.")
    assert res.medication_proposals() == () and res.laboratory_proposals() == ()
