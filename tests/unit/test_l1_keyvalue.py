"""Dual key-value + narrative structure for colon-delimited lines (non-exclusive)."""

from __future__ import annotations

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.document_intelligence.models import NodeKind
from mednorm_vi.document_intelligence.serialization import determinism_hash
from mednorm_vi.document_intelligence.validation import validate_graph

SAMPLE = (
    "Lý do vào viện: đau ngực, khó thở ba ngày.\n"
    "Chẩn đoán: Viêm phổi cộng đồng.\n"
    "Tiền sử: Tăng huyết áp 10 năm.\n"
)


def _cells(g, kind: str) -> list[str]:
    return [
        g.original_text[c.start : c.end]
        for c in g.nodes_of_kind(NodeKind.TABLE_CELL)
        if c.attributes.get("cell_kind") == kind
    ]


def test_key_and_value_offsets_are_exact() -> None:
    g = analyze_text(SAMPLE)
    keys = _cells(g, "key")
    values = _cells(g, "value")
    assert "Lý do vào viện" in keys and "Chẩn đoán" in keys and "Tiền sử" in keys
    assert "đau ngực, khó thở ba ngày." in values
    assert "Viêm phổi cộng đồng." in values
    for c in g.nodes_of_kind(NodeKind.TABLE_CELL):
        assert g.original_text[c.start : c.end] == c.text(g.original_text)


def test_row_and_value_sentence_coexist() -> None:
    g = analyze_text(SAMPLE)
    rows = g.nodes_of_kind(NodeKind.TABLE_ROW)
    assert all(r.row_kind == "key_value_like" for r in rows)
    # the narrative value is ALSO available as a sentence-like segment
    sentences = [g.original_text[s.start : s.end] for s in g.nodes_of_kind(NodeKind.SENTENCE)]
    assert "đau ngực, khó thở ba ngày." in sentences
    assert "Viêm phổi cộng đồng." in sentences


def test_value_sentence_nested_in_value_cell() -> None:
    g = analyze_text(SAMPLE)
    index = g.by_id()
    for s in g.nodes_of_kind(NodeKind.SENTENCE):
        parent = index[s.parent_id] if s.parent_id else None
        assert parent is not None and parent.kind is NodeKind.TABLE_CELL
        assert parent.start <= s.start and s.end <= parent.end  # containment


def test_delimiters_preserved_and_full_line_covered() -> None:
    g = analyze_text(SAMPLE)
    # the colon between key and value is neither key nor value text, but the line
    # is fully covered by lines + newline delimiters (validation checks this).
    assert validate_graph(g).ok
    newlines = [n for n in g.nodes_of_kind(NodeKind.DELIMITER)
                if n.attributes.get("delimiter") == "newline"]
    assert len(newlines) == 3


def test_containment_and_validation_hold() -> None:
    g = analyze_text(SAMPLE)
    result = validate_graph(g)
    assert result.ok, [i.message for i in result.errors]


def test_deterministic_serialization_unchanged() -> None:
    a = analyze_text(SAMPLE)
    b = analyze_text(SAMPLE)
    assert determinism_hash(a) == determinism_hash(b)


def test_tokens_map_to_value_sentence_not_row() -> None:
    g = analyze_text("Chẩn đoán: viêm phổi nặng.")
    # a token inside the value should reference the (smaller) sentence segment
    value_toks = [
        t for t in g.nodes_of_kind(NodeKind.TOKEN)
        if g.original_text[t.start : t.end] == "phổi"
    ]
    assert value_toks
    seg = g.by_id().get(value_toks[0].segment_id or "")
    assert seg is not None and seg.kind is NodeKind.SENTENCE
