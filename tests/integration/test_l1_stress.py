"""L1 stress tests: noisy, mixed, and realistic synthetic clinical inputs."""

from __future__ import annotations

import unicodedata

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.document_intelligence.models import NodeKind
from mednorm_vi.document_intelligence.serialization import determinism_hash
from mednorm_vi.document_intelligence.validation import validate_graph


def _ok(text: str) -> None:
    g = analyze_text(text)
    result = validate_graph(g)
    assert result.ok, [i.message for i in result.errors]
    # offset invariant for every node
    for n in g.nodes:
        assert g.original_text[n.start : n.end] == n.text(g.original_text)
    # deterministic rebuild
    assert determinism_hash(g) == determinism_hash(analyze_text(text))


def test_nfc_and_nfd_vietnamese() -> None:
    _ok(unicodedata.normalize("NFC", "Chẩn đoán: viêm phổi\nsốt cao"))
    _ok(unicodedata.normalize("NFD", "Chẩn đoán: viêm phổi\nsốt cao"))


def test_mixed_vietnamese_english() -> None:
    _ok("Diagnosis: viêm phổi (pneumonia)\nPMH: hypertension, đái tháo đường")


def test_repeated_spaces_and_tabs() -> None:
    _ok("WBC:    14,43\t\tH\nRBC:\t4.5")


def test_crlf_document() -> None:
    _ok("Chẩn đoán:\r\nviêm phổi.\r\n\r\nXét nghiệm:\r\nWBC: 14\r\n")


def test_punctuation_variants() -> None:
    _ok("“Chẩn đoán” — viêm phổi… (nặng); mạch 96–100 l/ph")


def test_long_line() -> None:
    _ok("Bệnh nhân " + "sốt ho khan mệt mỏi " * 200 + "và khó thở.")


def test_duplicated_substrings() -> None:
    _ok("táo bón " * 50)


def test_noisy_headings() -> None:
    _ok("chan doan :\nviem phoi\n\nTIEN SU:\ntang huyet ap")


def test_medication_like_abbreviations() -> None:
    _ok("1. paracetamol 500 mg po bid\n2. amlodipine 10 mg q.d.\n3. insulin 10 UI sc")


def test_laboratory_like_records() -> None:
    _ok("WBC: 14,43 (H)\nRBC: 4.5\nGlucose: 7.8 mmol/L\nNa: 138; K: 4.0; Cl: 101")


def test_realistic_multisection_document() -> None:
    text = (
        "BỆNH ÁN\n\n"
        "Lý do vào viện: sốt cao, ho.\n\n"
        "Tiền sử:\n- Tăng huyết áp\n- Đái tháo đường\n\n"
        "Thuốc tại nhà:\n1. paracetamol 500mg\n2. amlodipine 10 mg\n\n"
        "Xét nghiệm:\nWBC: 14,43\nRBC: 4.5\n\n"
        "Chẩn đoán: viêm phổi.\n\n"
        "Điều trị:\n- Kháng sinh\n"
    )
    g = analyze_text(text)
    assert validate_graph(g).ok
    cats = {s.category for s in g.nodes_of_kind(NodeKind.SECTION)}
    assert {"medical_history", "laboratory", "diagnosis"} <= cats


def test_only_whitespace_and_empty() -> None:
    _ok("")
    _ok("   \n\t\n   ")
