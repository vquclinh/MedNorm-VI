"""L1 section-header detection tests."""

from __future__ import annotations

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.document_intelligence.models import NodeKind


def _sections(text: str) -> list[tuple[str, str]]:
    g = analyze_text(text)
    out = []
    for s in g.nodes_of_kind(NodeKind.SECTION):
        header = "" if s.header_start is None else g.original_text[s.header_start : s.header_end]
        out.append((s.category, header))
    return out


def _categories(text: str) -> set[str]:
    return {c for c, _ in _sections(text)}


def test_explicit_vietnamese_heading() -> None:
    assert "diagnosis" in _categories("Chẩn đoán:\nviêm phổi")


def test_unaccented_heading() -> None:
    assert "diagnosis" in _categories("Chan doan:\nviem phoi")


def test_english_heading() -> None:
    assert "diagnosis" in _categories("Diagnosis:\npneumonia")


def test_abbreviation_heading() -> None:
    assert "medical_history" in _categories("PMH:\nhypertension")


def test_heading_with_colon_inline_value() -> None:
    cats = _categories("Chẩn đoán: viêm phổi.")
    assert "diagnosis" in cats


def test_uppercase_heading_isolated() -> None:
    assert "laboratory" in _categories("Trước đó\n\nXÉT NGHIỆM\n\nWBC: 14")


def test_ordinary_sentence_is_not_header() -> None:
    # A long sentence merely mentioning "chẩn đoán" must not become a header.
    text = "Cần chẩn đoán phân biệt với viêm phổi và lao phổi ở bệnh nhân này."
    cats = _categories(text)
    assert "diagnosis" not in cats
    assert cats == {"unknown"} or "unknown" in cats


def test_history_false_positive_inline_mention() -> None:
    text = "Bệnh nhân có tiền sử hút thuốc lá nhiều năm nay."
    assert "medical_history" not in _categories(text)


def test_repeated_section_names_distinct_nodes() -> None:
    g = analyze_text("Xét nghiệm:\nWBC: 1\n\nXét nghiệm:\nRBC: 2")
    labs = [s for s in g.nodes_of_kind(NodeKind.SECTION) if s.category == "laboratory"]
    assert len(labs) == 2
    assert labs[0].node_id != labs[1].node_id
    assert labs[0].start != labs[1].start


def test_nested_subsection_by_indent() -> None:
    text = "Tiền sử:\n  Gia đình:\n  bố tăng huyết áp\n"
    g = analyze_text(text)
    sections = {s.category: s for s in g.nodes_of_kind(NodeKind.SECTION)}
    assert "medical_history" in sections and "family_history" in sections
    fam = sections["family_history"]
    hist = sections["medical_history"]
    # the indented family header nests under the history section
    assert fam.parent_id == hist.node_id


def test_unknown_preamble_section() -> None:
    g = analyze_text("Bệnh án nội trú\n\nChẩn đoán:\nviêm phổi")
    cats = [s.category for s in g.nodes_of_kind(NodeKind.SECTION)]
    assert "unknown" in cats


def test_section_prior_recorded_but_not_asserted() -> None:
    g = analyze_text("Tiền sử:\ntăng huyết áp")
    hist = [s for s in g.nodes_of_kind(NodeKind.SECTION) if s.category == "medical_history"][0]
    assert hist.prior_label == "isHistorical"
    assert hist.prior_strength > 0
    # L1 stores the prior as evidence only; no entity/assertion nodes exist.
    assert not any(n.kind.value in {"entity"} for n in g.nodes)
