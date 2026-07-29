"""Deterministic multi-label Case Router over the L1 DocumentGraph.

Routes each non-blank canonical routable unit (list-item content, table/key-value
row, sentence-like span, or a whole-line fallback) with zero or more case tags
(C1-C7). Never forces exclusivity. C1-C5 come from structural/lexical signals;
C6/C7 are derived from specialist proposals via :func:`augment_derived`. A route
tag only activates processing — it is not an entity type or a final prediction.
"""

from __future__ import annotations

from ..document_intelligence.models import DocumentGraph, NodeKind, StructuralNode
from ..mention_factory.models import ComponentSpan, SpanProposal
from .models import CaseScore, NodeRouting, RouteSignal
from .rules import score_line
from .signals import LineContext, RouterConfig


def _section_categories(graph: DocumentGraph) -> dict[str, str | None]:
    return {n.node_id: n.category for n in graph.nodes_of_kind(NodeKind.SECTION)}


def by_id_attr(node: StructuralNode, key: str) -> bool:
    return node.attributes.get(key) == "true"


def _ctx(
    graph: DocumentGraph, node_id: str, node_kind: str, start: int, end: int,
    *, parent_line_id: str | None, section_cat: str | None,
    is_list_item: bool, row_kind: str | None, all_tokens: list[StructuralNode],
    narrative_only: bool = False,
) -> LineContext:
    toks = [t for t in all_tokens if start <= t.start and t.end <= end]
    return LineContext(
        document_id=graph.document_id, node_id=node_id, start=start, end=end,
        text=graph.original_text[start:end], section_category=section_cat,
        is_list_item=is_list_item, row_kind=row_kind,
        numeric_present=any(t.token_category == "number" for t in toks),
        word_count=sum(1 for t in toks if t.token_category == "word"),
        tokens=tuple(graph.original_text[t.start : t.end] for t in toks),
        node_kind=node_kind, parent_line_id=parent_line_id, narrative_only=narrative_only)


def build_route_contexts(graph: DocumentGraph) -> list[LineContext]:
    """Build one context per CANONICAL routable unit (deterministic specificity).

    Policy: route specific child structures first — list-item content, table/
    key-value rows, and sentence-like spans (incl. narrative values nested in
    cells) — and fall back to the whole line only when no more specific unit
    exists. A parent line is never routed through a specialist that an eligible
    child already represents (prevents duplicate specialist execution).
    """
    sections = _section_categories(graph)
    all_tokens = list(graph.nodes_of_kind(NodeKind.TOKEN))
    all_sentences = list(graph.nodes_of_kind(NodeKind.SENTENCE))
    contexts: list[LineContext] = []
    for line in graph.nodes_of_kind(NodeKind.LINE):
        if by_id_attr(line, "blank") or not graph.original_text[line.start:line.end].strip():
            continue
        section_cat = sections.get(line.section_id or "", None)
        children = graph.children_of(line.node_id)
        list_items = [c for c in children if c.kind is NodeKind.LIST_ITEM]
        table_rows = [c for c in children if c.kind is NodeKind.TABLE_ROW]
        # sentences within this line (including value sentences nested in cells)
        line_sentences = [s for s in all_sentences if line.start <= s.start and s.end <= line.end]

        emitted = False
        for li in list_items:  # medication-list items → route the item CONTENT
            cs = li.content_start if li.content_start is not None else li.start
            ce = li.content_end if li.content_end is not None else li.end
            contexts.append(_ctx(graph, li.node_id, "list_item", cs, ce,
                                 parent_line_id=line.node_id, section_cat=section_cat,
                                 is_list_item=True, row_kind=None, all_tokens=all_tokens))
            emitted = True
        for tr in table_rows:  # lab rows → route the row
            contexts.append(_ctx(graph, tr.node_id, "table_row", tr.start, tr.end,
                                 parent_line_id=line.node_id, section_cat=section_cat,
                                 is_list_item=False, row_kind=tr.row_kind, all_tokens=all_tokens))
            emitted = True
        for s in line_sentences:  # narrative (incl. key-value value narrative)
            # A sentence inside a table/key-value row is narrative-only (the row
            # already handles structured C1/C2), preventing duplicate specialist runs.
            nested_in_row = any(tr.start <= s.start and s.end <= tr.end for tr in table_rows)
            contexts.append(_ctx(graph, s.node_id, "sentence", s.start, s.end,
                                 parent_line_id=line.node_id, section_cat=section_cat,
                                 is_list_item=False, row_kind=None, all_tokens=all_tokens,
                                 narrative_only=nested_in_row))
            emitted = True
        if not emitted:  # line fallback
            contexts.append(_ctx(graph, line.node_id, "line", line.start, line.end,
                                 parent_line_id=None, section_cat=section_cat,
                                 is_list_item=False, row_kind=None, all_tokens=all_tokens))
    return contexts


# Backwards-compatible alias.
build_line_contexts = build_route_contexts


class CaseRouter:
    """Deterministic multi-label router."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config

    def route_graph(self, graph: DocumentGraph) -> list[NodeRouting]:
        cfg = self.config
        contexts = build_route_contexts(graph)
        contexts.sort(key=lambda c: (c.start, c.end, c.node_kind, c.node_id))
        c3_spec = cfg.case_spec("C3")
        routings: list[NodeRouting] = []
        for idx, ctx in enumerate(contexts):
            scored = score_line(ctx, cfg)
            # Two independent conditions, both required (Audit 0053):
            #   1. the weighted score reaches the activation threshold;
            #   2. the case has at least one of its declared POSITIVE-evidence
            #      signals, when it declares any.
            #
            # Condition 2 is what stops a route firing on punctuation and a numeral
            # alone. The suppression is recorded, never silent: a case that scored
            # high but lacked evidence appears in `route_gate_reasons`, so an
            # operator can see the route was considered and why it was withheld.
            gate_reasons: list[str] = []
            active = []
            for candidate in scored:
                spec = cfg.case_spec(candidate.case)
                fired_names = frozenset(s.name for s in candidate.fired_signals)
                reaches = candidate.score >= cfg.activate
                has_evidence = spec is None or spec.evidence_satisfied(fired_names)
                if reaches and has_evidence:
                    active.append(candidate)
                    continue
                if reaches and not has_evidence:
                    required = ",".join(spec.required_evidence) if spec else ""
                    gate_reasons.append(
                        f"{candidate.case}:suppressed_no_required_evidence:"
                        f"score={candidate.score:.2f}:requires={required}")
                elif candidate.fired_signals:
                    gate_reasons.append(
                        f"{candidate.case}:below_activate:score={candidate.score:.2f}")
            active.sort(key=lambda c: c.case)
            if (cfg.narrative_fallback and c3_spec is not None
                    and not any(c.case in ("C1", "C2") for c in active)
                    and not any(c.case == "C3" for c in active)
                    and ctx.word_count >= 3):
                active.append(CaseScore(
                    case="C3", score=cfg.activate,
                    fired_signals=(RouteSignal("C3", "narrative_fallback", "structural",
                                               cfg.activate),),
                    activated_specialists=c3_spec.activated_specialists))
            warnings: list[str] = []
            if active:
                top = max(c.score for c in active)
                if 0.0 < (top - cfg.activate) <= cfg.uncertainty_margin:
                    warnings.append("ambiguous_routing")
            routings.append(NodeRouting(
                decision_id=f"route-{idx + 1:04d}",
                document_id=ctx.document_id,
                node_id=ctx.node_id,
                start=ctx.start,
                end=ctx.end,
                text=ctx.text,
                cases=tuple(active),
                node_kind=ctx.node_kind,
                parent_line_id=ctx.parent_line_id,
                section_category=ctx.section_category,
                section_priors=dict(cfg.section_priors.get(ctx.section_category or "", {})),
                warnings=tuple(warnings),
                router_version=cfg.router_version,
                signals_version=cfg.signals_version,
                gate_reasons=tuple(gate_reasons),
            ))
        return routings

    def augment_derived(
        self, routings: list[NodeRouting], proposals: list[SpanProposal]
    ) -> list[NodeRouting]:
        weights = self._derived_weights()
        specialists = {
            "C6": self._specialists("C6"),
            "C7": self._specialists("C7"),
        }
        # global surface-form → distinct positions (for repeated_surface)
        positions: dict[str, set[int]] = {}
        for p in proposals:
            positions.setdefault(p.text, set()).add(p.start)
        by_node: dict[str, list[SpanProposal]] = {}
        for p in proposals:
            by_node.setdefault(p.source_node_id, []).append(p)

        out: list[NodeRouting] = []
        for r in routings:
            node_props = by_node.get(r.node_id, [])
            extra: list[CaseScore] = []
            c6 = self._c6(node_props, weights, specialists["C6"])
            if c6 is not None:
                extra.append(c6)
            c7 = self._c7(node_props, positions, weights, specialists["C7"])
            if c7 is not None:
                extra.append(c7)
            if extra:
                have = {c.case for c in r.cases}
                merged = list(r.cases) + [c for c in extra if c.case not in have]
                merged.sort(key=lambda c: c.case)
                out.append(NodeRouting(
                    decision_id=r.decision_id, document_id=r.document_id, node_id=r.node_id,
                    start=r.start, end=r.end, text=r.text, cases=tuple(merged),
                    node_kind=r.node_kind, parent_line_id=r.parent_line_id,
                    section_category=r.section_category, section_priors=r.section_priors,
                    warnings=r.warnings, router_version=r.router_version,
                    signals_version=r.signals_version))
            else:
                out.append(r)
        return out

    # --- derived helpers ---

    def _derived_weights(self) -> dict[tuple[str, str], float]:
        w: dict[tuple[str, str], float] = {}
        for case in ("C6", "C7"):
            spec = self.config.case_spec(case)
            if spec is not None:
                for s in spec.signals:
                    w[(case, s.name)] = s.weight
        return w

    def _specialists(self, case: str) -> tuple[str, ...]:
        spec = self.config.case_spec(case)
        return spec.activated_specialists if spec is not None else ()

    @staticmethod
    def _c6(
        props: list[SpanProposal], weights: dict[tuple[str, str], float],
        specialists: tuple[str, ...],
    ) -> CaseScore | None:
        """Meaningful medication linking ambiguity — NOT routine boundary candidates.

        Works at the PARSE level (grouped by ``parse_ref``), so a complete
        medication with several progressive boundary spans never triggers C6.
        """
        meds = [p for p in props if p.source_specialist == "medication" and p.parse_ref]
        parses: dict[str, list[SpanProposal]] = {}
        for m in meds:
            assert m.parse_ref is not None
            parses.setdefault(m.parse_ref, []).append(m)

        fired: list[RouteSignal] = []
        total = 0.0

        def _has_strength(comps: list[ComponentSpan]) -> bool:
            return any(c.role in ("strength_value", "concentration") for c in comps)

        # incomplete identity: a KNOWN medication named without a strength.
        incomplete = any(
            (not _has_strength(list(group[0].components)))
            and "unknown_medication_name" not in group[0].warnings
            for group in parses.values())
        if incomplete:
            w = weights.get(("C6", "incomplete_identity"), 0.50)
            fired.append(RouteSignal("C6", "incomplete_identity", "derived", w))
            total += w
        # unknown medication name (ambiguous linking target)
        if any("unknown_medication_name" in group[0].warnings for group in parses.values()):
            w = weights.get(("C6", "unknown_medication"), 0.50)
            fired.append(RouteSignal("C6", "unknown_medication", "derived", w))
            total += w
        # conflicting strength: same ingredient name, different strength values.
        by_name: dict[str, set[str]] = {}
        for group in parses.values():
            rep = group[0]
            strengths = {c.text for c in rep.components if c.role == "strength_value"}
            if rep.normalized_form and strengths:
                by_name.setdefault(rep.normalized_form, set()).update(strengths)
        if any(len(s) > 1 for s in by_name.values()):
            w = weights.get(("C6", "conflicting_strength"), 0.50)
            fired.append(RouteSignal("C6", "conflicting_strength", "derived", w))
            total += w
        if not fired:
            return None
        return CaseScore("C6", min(1.0, total), tuple(fired), specialists)

    @staticmethod
    def _c7(
        props: list[SpanProposal], positions: dict[str, set[int]],
        weights: dict[tuple[str, str], float], specialists: tuple[str, ...],
    ) -> CaseScore | None:
        fired: list[RouteSignal] = []
        total = 0.0
        # overlap BETWEEN different parses/specialists (not same-parse boundaries)
        cross_overlap = False
        for i, a in enumerate(props):
            for b in props[i + 1:]:
                same_parse = a.parse_ref is not None and a.parse_ref == b.parse_ref
                if same_parse:
                    continue
                if a.start < b.end and b.start < a.end and (a.start, a.end) != (b.start, b.end):
                    cross_overlap = True
                    break
            if cross_overlap:
                break
        if cross_overlap:
            w = weights.get(("C7", "cross_parse_overlap"), 0.45)
            fired.append(RouteSignal("C7", "cross_parse_overlap", "derived", w))
            total += w
        if any(len(positions.get(p.text, set())) > 1 for p in props):
            w = weights.get(("C7", "repeated_surface"), 0.45)
            fired.append(RouteSignal("C7", "repeated_surface", "derived", w))
            total += w
        if not fired:
            return None
        return CaseScore("C7", min(1.0, total), tuple(fired), specialists)


__all__ = ["CaseRouter", "build_line_contexts"]
