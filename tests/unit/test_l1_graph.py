"""L1 DocumentGraph construction, validation, and serialization tests."""

from __future__ import annotations

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.document_intelligence.models import NodeKind
from mednorm_vi.document_intelligence.serialization import (
    determinism_hash,
    from_debug_dict,
    to_debug_dict,
    to_json,
)
from mednorm_vi.document_intelligence.validation import validate_graph

SAMPLE = (
    "Chẩn đoán: viêm phổi.\n\nTiền sử:\n- Tăng huyết áp\n\nXét nghiệm:\nWBC: 14,43\n"
)


def test_every_node_span_matches_exact_substring() -> None:
    g = analyze_text(SAMPLE)
    for n in g.nodes:
        assert g.original_text[n.start : n.end] == n.text(g.original_text)


def test_graph_validates() -> None:
    g = analyze_text(SAMPLE)
    result = validate_graph(g)
    assert result.ok, [i.message for i in result.errors]


def test_deterministic_ids_and_content() -> None:
    a = analyze_text(SAMPLE)
    b = analyze_text(SAMPLE)
    assert [n.node_id for n in a.nodes] == [n.node_id for n in b.nodes]
    assert determinism_hash(a) == determinism_hash(b)


def test_unique_node_ids() -> None:
    g = analyze_text(SAMPLE)
    ids = [n.node_id for n in g.nodes]
    assert len(ids) == len(set(ids))


def test_parent_child_containment() -> None:
    g = analyze_text(SAMPLE)
    index = g.by_id()
    for n in g.nodes:
        if n.parent_id is not None:
            p = index[n.parent_id]
            assert p.start <= n.start and n.end <= p.end


def test_original_text_not_mutated() -> None:
    g = analyze_text(SAMPLE)
    assert g.original_text == SAMPLE


def test_roundtrip_serialization() -> None:
    g = analyze_text(SAMPLE)
    d = to_debug_dict(g)
    g2 = from_debug_dict(d)
    assert to_debug_dict(g2) == d
    assert determinism_hash(g2) == determinism_hash(g)


def test_json_is_utf8_and_not_escaped() -> None:
    g = analyze_text(SAMPLE)
    text = to_json(g)
    assert "viêm phổi" in text
    assert "\\u" not in text


def test_normalized_views_present_and_not_output_coordinates() -> None:
    g = analyze_text(SAMPLE)
    assert "match" in g.normalized and "accent" in g.normalized
    # A normalized view maps back to a COVERING original span, never the reverse.
    view = g.normalized["match"]
    os, oe = view.alignment.normalized_span_to_original(0, len(view.text))
    assert (os, oe) == (0, len(SAMPLE))
    # output coordinates come only from node absolute spans over original_text
    for n in g.nodes:
        assert 0 <= n.start <= n.end <= len(g.original_text)


def test_repeated_surface_forms_distinct_nodes() -> None:
    g = analyze_text("Tiền sử:\nWBC: 1\n\nTiền sử:\nWBC: 2")
    headers = [s for s in g.nodes_of_kind(NodeKind.SECTION) if s.category == "medical_history"]
    assert len(headers) == 2
    assert headers[0].node_id != headers[1].node_id


def test_full_line_coverage_holds() -> None:
    g = analyze_text("a\n\nb\r\nc")
    assert validate_graph(g).ok
