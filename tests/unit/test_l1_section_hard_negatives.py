"""Section detection: fuzzy hard negatives + exact-alias positives.

Ordinary sentences that resemble headers must not be promoted to sections
without structural evidence. Exact aliases (with header structure) still work,
including unaccented and bilingual headings.
"""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.document_intelligence.models import NodeKind

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "document_intelligence" / "section_hard_negatives_v1.txt"
)


def _detected_categories(text: str) -> set[str]:
    g = analyze_text(text)
    return {
        s.category for s in g.nodes_of_kind(NodeKind.SECTION) if s.category != "unknown"
    }


def _load_negatives() -> list[str]:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln.strip() and not ln.startswith("#")]


def test_hard_negatives_are_not_headers() -> None:
    for sentence in _load_negatives():
        cats = _detected_categories(sentence)
        assert cats == set(), f"{sentence!r} wrongly classified as section {cats}"


def test_hard_negatives_in_context_are_not_headers() -> None:
    # Even surrounded by blank lines, a full ordinary sentence is not a header
    # (it exceeds max_header_chars / is not colon-key-matched to an alias).
    for sentence in _load_negatives():
        cats = _detected_categories(f"\n{sentence}\n")
        assert cats == set(), f"{sentence!r} wrongly classified in context as {cats}"


# --- positives: exact/structural headers must still be detected ---

def test_exact_colon_header_detected() -> None:
    assert "diagnosis" in _detected_categories("Chẩn đoán:\nviêm phổi")


def test_unaccented_header_still_detected() -> None:
    assert "medical_history" in _detected_categories("Tien su:\ntang huyet ap")


def test_bilingual_english_header_still_detected() -> None:
    assert "laboratory" in _detected_categories("Laboratory:\nWBC 14")


def test_uppercase_isolated_header_detected() -> None:
    assert "laboratory" in _detected_categories("mở đầu\n\nXÉT NGHIỆM\n\nWBC: 14")


def test_exact_alias_vs_fuzzy_are_separate() -> None:
    # exact alias key "Chẩn đoán" → detected; a near-miss non-header sentence → not.
    assert "diagnosis" in _detected_categories("Chẩn đoán: viêm phổi.")
    assert "diagnosis" not in _detected_categories("Cần chẩn đoán phân biệt với lao phổi.")
