"""Deterministic DocumentGraph builder (L1 orchestrator).

Assembles the structural node tree (document → sections → paragraphs → lines →
sentences/list-items/table-rows → tokens, plus newline delimiters) with
deterministic ids and absolute offsets, over the immutable ``original_text``.
Normalized views are attached for downstream matching but never provide output
coordinates.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .io import LoadedDocument
from .lines import is_blank, paragraphs, split_lines
from .lists import detect_list_item
from .models import (
    DocumentGraph,
    DocumentWarning,
    L1Config,
    NodeKind,
    StructuralNode,
)
from .normalization import build_views
from .sections import SectionHit, SectionLexicon, detect_sections, load_lexicon
from .sentences import segment_sentences
from .table_rows import detect_row
from .tokens import tokenize

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "document_intelligence"


class _Ids:
    def __init__(self, pad: int) -> None:
        self.pad = pad
        self.counters: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        return f"{prefix}-{self.counters[prefix]:0{self.pad}d}"


@dataclasses.dataclass
class _SectionSpan:
    node_id: str
    start: int
    end: int
    depth: int


def _build_section_spans(
    hits: list[SectionHit], pieces_start: dict[int, int], text_len: int, ids: _Ids,
) -> tuple[list[StructuralNode], list[_SectionSpan]]:
    """Create SECTION nodes (with nesting by indent) and their spans."""
    nodes: list[StructuralNode] = []
    spans: list[_SectionSpan] = []

    first_start = pieces_start[hits[0].line_index] if hits else text_len
    # Preamble (unknown/other) before the first header.
    if first_start > 0:
        pid = ids.next("sec")
        nodes.append(StructuralNode(
            node_id=pid, kind=NodeKind.SECTION, start=0, end=first_start,
            category="unknown", confidence=0.0, matched_rule="preamble",
            provenance="preamble"))
        spans.append(_SectionSpan(pid, 0, first_start, 0))

    stack: list[tuple[int, str]] = []  # (indent, section_id)
    for i, hit in enumerate(hits):
        start = pieces_start[hit.line_index]
        end = text_len
        for j in range(i + 1, len(hits)):
            if hits[j].indent <= hit.indent:
                end = pieces_start[hits[j].line_index]
                break
        while stack and stack[-1][0] >= hit.indent:
            stack.pop()
        parent_id = stack[-1][1] if stack else None
        depth = len(stack)
        sid = ids.next("sec")
        nodes.append(StructuralNode(
            node_id=sid, kind=NodeKind.SECTION, start=start, end=end,
            parent_id=parent_id, category=hit.category, confidence=hit.confidence,
            matched_rule=hit.matched_rule, header_start=hit.header_start,
            header_end=hit.header_end, prior_label=hit.prior_label,
            prior_strength=hit.prior_strength, provenance="section_lexicon",
            warnings=hit.warnings))
        spans.append(_SectionSpan(sid, start, end, depth))
        stack.append((hit.indent, sid))
    return nodes, spans


def _section_for(spans: list[_SectionSpan], pos: int) -> str | None:
    best: _SectionSpan | None = None
    for s in spans:
        if s.start <= pos < s.end and (best is None or s.start > best.start):
            best = s
    return best.node_id if best is not None else None


def build_document_graph(
    document: LoadedDocument, config: L1Config, lexicon: SectionLexicon,
) -> DocumentGraph:
    """Build the DocumentGraph for a loaded document."""
    text = document.original_text
    n = len(text)
    ids = _Ids(config.id_zero_pad)
    warnings: list[DocumentWarning] = []
    nodes: list[StructuralNode] = []

    doc_id = ids.next("doc")
    nodes.append(StructuralNode(node_id=doc_id, kind=NodeKind.DOCUMENT, start=0, end=n))

    # Propagate load-time warnings (e.g. a stripped BOM) as structural warnings.
    for msg in document.warnings:
        warnings.append(DocumentWarning(msg.split(":", 1)[0], msg, doc_id))

    pieces = split_lines(text)
    pieces_start = {p.index: p.content_start for p in pieces}

    hits = detect_sections(text, pieces, config, lexicon)
    # Paragraphs must not cross a section start (header line).
    header_lines = frozenset(hit.line_index for hit in hits)
    paras = paragraphs(text, pieces, break_before=header_lines)
    section_nodes, section_spans = _build_section_spans(hits, pieces_start, n, ids)
    for sn in section_nodes:
        parent = sn.parent_id or doc_id
        nodes.append(dataclasses.replace(sn, parent_id=parent))

    # Paragraph nodes.
    para_id_by_line: dict[int, str] = {}
    para_ids: dict[int, str] = {}
    for pi, para in enumerate(paras):
        sect = _section_for(section_spans, para.start) or doc_id
        pid = ids.next("par")
        para_ids[pi] = pid
        nodes.append(StructuralNode(
            node_id=pid, kind=NodeKind.PARAGRAPH, start=para.start, end=para.end,
            parent_id=sect))
        for li in para.line_indices:
            para_id_by_line[li] = pid

    # Line-level nodes (in document order).
    for piece in pieces:
        blank = is_blank(text, piece)
        line_parent = para_id_by_line.get(piece.index) or _section_for(
            section_spans, piece.content_start) or doc_id
        line_id = ids.next("line")
        section_id = _section_for(section_spans, piece.content_start) or doc_id
        nodes.append(StructuralNode(
            node_id=line_id, kind=NodeKind.LINE, start=piece.content_start,
            end=piece.content_end, parent_id=line_parent, section_id=section_id,
            attributes={"blank": "true" if blank else "false"}))

        content = text[piece.content_start : piece.content_end]
        segment_ids: list[tuple[int, int, str]] = []  # (start,end,id) for token assignment

        if content.strip():
            lm = detect_list_item(content, piece.content_start, config)
            row = detect_row(content, piece.content_start, config)
            if lm is not None:
                item_id = ids.next("li")
                nodes.append(StructuralNode(
                    node_id=item_id, kind=NodeKind.LIST_ITEM, start=piece.content_start,
                    end=piece.content_end, parent_id=line_id,
                    marker_start=lm.marker_start, marker_end=lm.marker_end,
                    content_start=lm.content_start, content_end=lm.content_end,
                    indent=lm.indent, section_id=section_id))
                nodes.append(StructuralNode(
                    node_id=ids.next("mk"), kind=NodeKind.LIST_MARKER,
                    start=lm.marker_start, end=lm.marker_end, parent_id=item_id))
                segment_ids.append((piece.content_start, piece.content_end, item_id))
            elif row is not None:
                row_id = ids.next("row")
                nodes.append(StructuralNode(
                    node_id=row_id, kind=NodeKind.TABLE_ROW, start=piece.content_start,
                    end=piece.content_end, parent_id=line_id, row_kind=row.row_kind,
                    section_id=section_id))
                value_cell_id: str | None = None
                value_span: tuple[int, int] | None = None
                for cell in row.cells:
                    cid = ids.next("cell")
                    nodes.append(StructuralNode(
                        node_id=cid, kind=NodeKind.TABLE_CELL,
                        start=cell.start, end=cell.end, parent_id=row_id,
                        attributes={"cell_kind": cell.kind}))
                    if cell.kind == "value":
                        value_cell_id = cid
                        value_span = (cell.start, cell.end)
                segment_ids.append((piece.content_start, piece.content_end, row_id))
                # Dual representation: a key-value row's narrative value is ALSO
                # segmented into sentence-like span(s) for downstream use. Still no
                # medical entity/assertion is emitted here.
                if row.row_kind == "key_value_like" and value_cell_id and value_span:
                    vtext = text[value_span[0] : value_span[1]]
                    for span in segment_sentences(
                        vtext, value_span[0], config.sentence_terminators
                    ):
                        sent_id = ids.next("sent")
                        nodes.append(StructuralNode(
                            node_id=sent_id, kind=NodeKind.SENTENCE, start=span.start,
                            end=span.end, parent_id=value_cell_id, section_id=section_id))
                        segment_ids.append((span.start, span.end, sent_id))
            else:
                for span in segment_sentences(
                    content, piece.content_start, config.sentence_terminators
                ):
                    sent_id = ids.next("sent")
                    nodes.append(StructuralNode(
                        node_id=sent_id, kind=NodeKind.SENTENCE, start=span.start,
                        end=span.end, parent_id=line_id, section_id=section_id))
                    segment_ids.append((span.start, span.end, sent_id))

            # Tokens (children of the line; segment_id references the SMALLEST
            # (most specific) covering sub-unit — e.g. a value sentence over its row).
            for tok in tokenize(text, piece.content_start, piece.content_end):
                covering = [
                    (e - s, sid) for (s, e, sid) in segment_ids
                    if s <= tok.start and tok.end <= e
                ]
                seg_id = min(covering)[1] if covering else None
                nodes.append(StructuralNode(
                    node_id=ids.next("tok"), kind=NodeKind.TOKEN, start=tok.start,
                    end=tok.end, parent_id=line_id, section_id=section_id,
                    segment_id=seg_id, line_id=line_id, normalized_text=tok.normalized,
                    token_category=tok.category, is_punctuation=tok.is_punctuation,
                    whitespace_before=tok.whitespace_before))

        # Newline delimiter (preserves the exact terminator).
        if piece.has_newline:
            nodes.append(StructuralNode(
                node_id=ids.next("nl"), kind=NodeKind.DELIMITER,
                start=piece.newline_start, end=piece.newline_end,
                parent_id=_section_for(section_spans, piece.newline_start) or doc_id,
                attributes={"delimiter": "newline"}))

    # Wire children from parent links (deterministic order by start,end,id).
    children: dict[str, list[str]] = {}
    for node in nodes:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node.node_id)
    by_start = {node.node_id: (node.start, node.end, node.node_id) for node in nodes}
    final_nodes = tuple(
        dataclasses.replace(
            node,
            children=tuple(sorted(children.get(node.node_id, []), key=lambda cid: by_start[cid])),
        )
        for node in nodes
    )

    normalized = build_views(text, config.normalization_views)
    for hit in hits:
        for w in hit.warnings:
            warnings.append(DocumentWarning(w, f"section header at {hit.header_start}",
                                            None))

    return DocumentGraph(
        document_id=document.source.sha256[:16],
        source=document.source,
        original_text=text,
        normalized=normalized,
        nodes=final_nodes,
        warnings=tuple(warnings),
        builder_version=config.builder_version,
        config_hash=config.config_hash,
    )


def load_config(path: str | Path) -> tuple[L1Config, SectionLexicon]:
    """Load the L1 config + its section lexicon from a base.yaml path."""
    import yaml

    base = Path(path)
    data: dict[str, Any] = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    config = L1Config.from_mapping(data)
    lex_path = base.parent / config.section_lexicon_file
    lexicon = load_lexicon(lex_path)
    return config, lexicon


def default_config() -> tuple[L1Config, SectionLexicon]:
    return load_config(_CONFIG_DIR / "base.yaml")


__all__ = ["build_document_graph", "load_config", "default_config"]
