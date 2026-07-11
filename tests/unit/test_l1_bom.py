"""L1 BOM policy tests: default preserves, strip is explicit + audited, error fails."""

from __future__ import annotations

import pytest

from mednorm_vi.document_intelligence import analyze_text, build_document_graph
from mednorm_vi.document_intelligence.builder import default_config
from mednorm_vi.document_intelligence.io import DocumentLoadError, from_text, load_text

BOM = "﻿"


def test_default_policy_preserves_bom_and_text() -> None:
    doc = load_text((BOM + "Chẩn đoán").encode())  # default bom_policy
    assert doc.source.bom_policy == "preserve"
    assert doc.original_text == BOM + "Chẩn đoán"
    assert doc.source.has_bom is True
    assert doc.source.bom_stripped is False
    assert doc.warnings == ()


def test_bom_presence_causes_no_invisible_offset_shift() -> None:
    # With preserve, the BOM occupies index 0; the first real char is at index 1.
    text = BOM + "abc"
    g = analyze_text(text)
    assert g.original_text == text
    # every node still slices back exactly (no hidden shift)
    for node in g.nodes:
        assert g.original_text[node.start : node.end] == node.text(g.original_text)


def test_strip_mode_is_explicit_and_audited() -> None:
    doc = load_text((BOM + "hello").encode(), bom_policy="strip")
    assert doc.original_text == "hello"
    assert doc.source.bom_stripped is True
    assert doc.source.has_bom is False
    assert any(w.startswith("bom.stripped") for w in doc.warnings)


def test_strip_warning_propagates_to_graph() -> None:
    config, lexicon = default_config()
    doc = load_text((BOM + "Chẩn đoán:\nviêm phổi").encode(), bom_policy="strip")
    graph = build_document_graph(doc, config, lexicon)
    assert any(w.code == "bom.stripped" for w in graph.warnings)


def test_error_mode_fails_deterministically() -> None:
    with pytest.raises(DocumentLoadError):
        load_text((BOM + "x").encode(), bom_policy="error")
    # deterministic: same input, same failure
    with pytest.raises(DocumentLoadError):
        load_text((BOM + "x").encode(), bom_policy="error")


def test_no_bom_documents_are_unaffected() -> None:
    doc = from_text("no bom here")
    assert doc.source.has_bom is False
    assert doc.source.bom_stripped is False
    assert doc.warnings == ()
