"""L1 line / sentence / list / table-row / token segmentation tests."""

from __future__ import annotations

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.document_intelligence.builder import default_config
from mednorm_vi.document_intelligence.lines import paragraphs, split_lines
from mednorm_vi.document_intelligence.models import NodeKind
from mednorm_vi.document_intelligence.sentences import segment_sentences
from mednorm_vi.document_intelligence.tokens import tokenize

CONFIG, LEXICON = default_config()


def _texts(graph, kind: NodeKind) -> list[str]:
    return [graph.original_text[n.start : n.end] for n in graph.nodes_of_kind(kind)]


# --- lines / paragraphs ---

def test_lines_tile_document_with_delimiters() -> None:
    text = "a\n\nb\r\nc"
    pieces = split_lines(text)
    covered = 0
    intervals = []
    for p in pieces:
        if p.content_end > p.content_start:
            intervals.append((p.content_start, p.content_end))
        if p.has_newline:
            intervals.append((p.newline_start, p.newline_end))
    intervals.sort()
    for s, e in intervals:
        assert s == covered
        covered = e
    assert covered == len(text)


def test_empty_document() -> None:
    g = analyze_text("")
    assert len(g.original_text) == 0
    assert g.nodes_of_kind(NodeKind.DOCUMENT)


def test_single_line_document() -> None:
    g = analyze_text("chỉ một dòng")
    assert len(g.nodes_of_kind(NodeKind.LINE)) == 1


def test_whitespace_only_document() -> None:
    g = analyze_text("   \n\t \n")
    # no crash; lines exist, no sentences/tokens
    assert g.nodes_of_kind(NodeKind.LINE)
    assert not g.nodes_of_kind(NodeKind.TOKEN)


def test_paragraph_grouping() -> None:
    text = "l1\nl2\n\np2a\n"
    paras = paragraphs(text, split_lines(text))
    assert len(paras) == 2


# --- sentences ---

def test_period_not_always_boundary_abbreviation() -> None:
    spans = segment_sentences("Cho 500 mg. po mỗi ngày", 0, (".", "!", "?", "…", ";"))
    # "mg." should not split (known abbreviation); one sentence-like span
    assert len(spans) == 1


def test_decimal_not_a_boundary() -> None:
    spans = segment_sentences("Nhiệt độ 38.5 độ C", 0, (".", "!", "?", "…", ";"))
    assert len(spans) == 1


def test_semicolon_splits() -> None:
    spans = segment_sentences("mạch 96; huyết áp 120", 0, (".", "!", "?", "…", ";"))
    assert len(spans) == 2


def test_narrative_multiple_sentences() -> None:
    text = "Bệnh nhân sốt cao. Ho khan nhiều. Không đau ngực."
    spans = segment_sentences(text, 0, (".", "!", "?", "…", ";"))
    assert len(spans) == 3
    # spans keep their terminating period
    assert text[spans[0].start : spans[0].end].endswith(".")


# --- lists ---

def test_numbered_and_bulleted_lists() -> None:
    g = analyze_text("1. paracetamol 500mg\n2) amlodipine\n- aspirin\n• metformin")
    items = _texts(g, NodeKind.LIST_ITEM)
    assert len(items) == 4
    markers = _texts(g, NodeKind.LIST_MARKER)
    assert markers[0] == "1."
    assert "•" in markers[-1]


def test_list_marker_not_in_content() -> None:
    g = analyze_text("1. paracetamol")
    item = g.nodes_of_kind(NodeKind.LIST_ITEM)[0]
    content = g.original_text[item.content_start : item.content_end]
    assert content == "paracetamol"  # number excluded from content span


# --- table-like rows ---

def test_key_value_row() -> None:
    g = analyze_text("WBC: 14,43")
    rows = g.nodes_of_kind(NodeKind.TABLE_ROW)
    assert rows and rows[0].row_kind == "key_value_like"
    cells = _texts(g, NodeKind.TABLE_CELL)
    assert "WBC" in cells and "14,43" in cells


def test_tab_row_is_table_like() -> None:
    g = analyze_text("WBC\t14.43\tH")
    rows = g.nodes_of_kind(NodeKind.TABLE_ROW)
    assert rows and rows[0].row_kind == "table_like"


# --- tokens ---

def test_tokens_keep_decimals_and_units() -> None:
    toks = tokenize("glucose 7.8 mmol/L", 0, len("glucose 7.8 mmol/L"))
    texts = [t.text for t in toks]
    assert "7.8" in texts
    assert "mmol/L" in texts


def test_tokens_keep_decimal_comma_and_ranges() -> None:
    toks = tokenize("14,43 325-650", 0, len("14,43 325-650"))
    texts = [t.text for t in toks]
    assert "14,43" in texts
    assert "325-650" in texts


def test_repeated_text_distinct_nodes() -> None:
    g = analyze_text("táo bón rồi táo bón")
    toks = [t for t in g.nodes_of_kind(NodeKind.TOKEN) if g.original_text[t.start:t.end] == "táo"]
    assert len(toks) == 2
    assert toks[0].start != toks[1].start
    assert toks[0].node_id != toks[1].node_id
