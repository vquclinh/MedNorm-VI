"""Per-document descriptive analysis over the existing L1 graph (Phase 2A).

Consumes the L1 ``DocumentGraph`` and the L2 canonical routable units EXACTLY as
they are produced today — Phase 2A changes no preprocessing behavior. A route
case is a PROCESSING CASE, not an entity type and not a prediction; the mention
specialists are deliberately NOT run, so derived cases C6/C7 are out of scope.
"""

from __future__ import annotations

import re

from ..case_router.router import CaseRouter
from ..case_router.signals import RouterConfig
from ..document_intelligence.models import DocumentGraph, NodeKind
from .config import AnalysisConfig
from .layouts import analyze_layouts
from .loader import CorpusDocument
from .models import (
    DocumentAnalysis,
    LengthStats,
    RoutingStats,
    SectionStats,
    StructureStats,
)

_NUMBERED_MARKER = re.compile(r"^\s*\d+\s*[.)]\s*$")
# Structural cases only: C6/C7 are derived from specialist proposals (not run).
_STRUCTURAL_CASES = ("C1", "C2", "C3", "C4", "C5")


def _length_stats(graph: DocumentGraph) -> LengthStats:
    text = graph.original_text
    lines = text.splitlines()
    tokens = graph.nodes_of_kind(NodeKind.TOKEN)
    return LengthStats(
        characters=len(text),
        bytes_utf8=len(text.encode("utf-8")),
        lines=len(lines),
        non_blank_lines=sum(1 for line in lines if line.strip()),
        blank_lines=sum(1 for line in lines if not line.strip()),
        max_line_length=max((len(line) for line in lines), default=0),
        tokens=len(tokens),
        word_tokens=sum(1 for t in tokens if t.token_category == "word"),
        number_tokens=sum(1 for t in tokens if t.token_category == "number"),
        sentences=len(graph.nodes_of_kind(NodeKind.SENTENCE)))


def _section_stats(graph: DocumentGraph) -> SectionStats:
    sections = graph.nodes_of_kind(NodeKind.SECTION)
    headings: list[str] = []
    for s in sections:
        if s.header_start is not None and s.header_end is not None:
            heading = graph.original_text[s.header_start:s.header_end].strip()
            if heading:
                headings.append(heading)
    categorized = sum(1 for s in sections if s.category)
    return SectionStats(
        sections=len(sections), categorized=categorized,
        uncategorized=len(sections) - categorized,
        categories=tuple(sorted({s.category for s in sections if s.category})),
        headings=tuple(headings))


def _structure_stats(graph: DocumentGraph) -> StructureStats:
    items = graph.nodes_of_kind(NodeKind.LIST_ITEM)
    numbered = 0
    for it in items:
        if it.marker_start is not None and it.marker_end is not None:
            marker = graph.original_text[it.marker_start:it.marker_end]
            if _NUMBERED_MARKER.match(marker):
                numbered += 1
    rows = graph.nodes_of_kind(NodeKind.TABLE_ROW)
    return StructureStats(
        list_items=len(items), bullet_items=len(items) - numbered, numbered_items=numbered,
        max_indent=max((it.indent or 0 for it in items), default=0),
        max_depth=max((it.depth or 0 for it in items), default=0),
        table_rows=len(rows),
        key_value_rows=sum(1 for r in rows if r.row_kind == "key_value_like"),
        table_like_rows=sum(1 for r in rows if r.row_kind == "table_like"),
        paragraphs=len(graph.nodes_of_kind(NodeKind.PARAGRAPH)))


def _routing_stats(graph: DocumentGraph, router_config: RouterConfig) -> RoutingStats:
    routings = CaseRouter(router_config).route_graph(graph)
    by_kind: dict[str, int] = {}
    cases: dict[str, int] = dict.fromkeys(_STRUCTURAL_CASES, 0)
    multi = unrouted = 0
    for r in routings:
        by_kind[r.node_kind] = by_kind.get(r.node_kind, 0) + 1
        tags = [t for t in r.route_tags if t in _STRUCTURAL_CASES]
        for t in tags:
            cases[t] += 1
        if len(tags) >= 2:
            multi += 1
        if not tags:
            unrouted += 1
    return RoutingStats(units_by_kind=dict(sorted(by_kind.items())), cases=cases,
                        multi_case_units=multi, unrouted_units=unrouted)


def _profile(structure: StructureStats, length: LengthStats,
             config: AnalysisConfig) -> str:
    if length.non_blank_lines == 0:
        return "empty"
    share = structure.list_items / length.non_blank_lines
    if share >= config.list_dominant_min_share:
        return "list_dominant"
    if share <= config.narrative_dominant_max_share:
        return "narrative_dominant"
    return "mixed"


def analyze_document(
    doc: CorpusDocument, graph: DocumentGraph, *, router_config: RouterConfig,
    config: AnalysisConfig, l1_valid: bool = True,
) -> DocumentAnalysis:
    """Measure one public input document. Descriptive statistics only."""
    length = _length_stats(graph)
    sections = _section_stats(graph)
    structure = _structure_stats(graph)
    routing = _routing_stats(graph, router_config)
    layout = analyze_layouts(graph.original_text.splitlines(), config)
    return DocumentAnalysis(
        document_id=graph.document_id, source_name=doc.source_name,
        content_sha256=doc.content_sha256, length=length, sections=sections,
        structure=structure, routing=routing, layout=layout,
        profile=_profile(structure, length, config), l1_valid=l1_valid,
        warnings=tuple(sorted({w.code for w in graph.warnings})))


__all__ = ["analyze_document"]
