"""Test helpers: build NodeRouting lists over canonical routable units.

Specialists consume ``list[NodeRouting]`` (one per canonical L1 unit — list-item
content, table/key-value row, sentence, or line fallback). These helpers force a
fixed case set onto every canonical unit so a specialist can be exercised in
isolation, matching the router's granularity without invoking the full scorer.
"""

from __future__ import annotations

from mednorm_vi.case_router.models import CaseScore, NodeRouting
from mednorm_vi.case_router.router import build_route_contexts
from mednorm_vi.document_intelligence.models import DocumentGraph


def forced_routings(
    graph: DocumentGraph, cases: tuple[str, ...] = ("C1",),
    *, section: str | None = None,
) -> list[NodeRouting]:
    """One NodeRouting per canonical routable unit, all carrying ``cases``.

    Narrative-only units (sentences nested in a row) drop structured cases C1/C2,
    mirroring the production router so duplicate specialist runs stay suppressed.
    """
    out: list[NodeRouting] = []
    for i, ctx in enumerate(build_route_contexts(graph)):
        unit_cases = tuple(
            c for c in cases if not (ctx.narrative_only and c in ("C1", "C2")))
        scores = tuple(CaseScore(c, 1.0, (), ()) for c in unit_cases)
        out.append(NodeRouting(
            decision_id=f"route-{i + 1:04d}", document_id=ctx.document_id,
            node_id=ctx.node_id, start=ctx.start, end=ctx.end, text=ctx.text,
            cases=scores, node_kind=ctx.node_kind, parent_line_id=ctx.parent_line_id,
            section_category=section if section is not None else ctx.section_category))
    return out
