"""Deterministic laboratory / test-result parser (proposal source).

Consumes C2-routed canonical routable units and emits TEST_NAME / TEST_RESULT
proposals plus internal ``has_result`` relations (a globally minimum-cost
assignment). It never fabricates a numeric result from vague narrative and never
emits final organizer entities. Exact offsets always.
"""

from __future__ import annotations

from ...case_router.models import NodeRouting
from ...document_intelligence.models import DocumentGraph
from ...schemas.constants import ORGANIZER_LABEL_BY_TYPE
from ..models import (
    HAS_RESULT,
    ComponentSpan,
    RelationProposal,
    SpanProposal,
    SpecialistRunResult,
)
from .lexicon import LabLexicon
from .pairing import ResultGroup, pair_names_to_groups
from .patterns import find_flag, find_reference, find_unit, find_value, hard_negative
from .scoring import score_test_name, score_test_result

NAME_LABEL = ORGANIZER_LABEL_BY_TYPE["TEST_NAME"]
RESULT_LABEL = ORGANIZER_LABEL_BY_TYPE["TEST_RESULT"]


def _split_offsets(seg: str, sep: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = 0
    for chunk in seg.split(sep):
        out.append((pos, pos + len(chunk)))
        pos += len(chunk) + len(sep)
    return out


def _trim(seg: str, s: int, e: int) -> tuple[int, int]:
    while s < e and seg[s].isspace():
        s += 1
    while e > s and seg[e - 1].isspace():
        e -= 1
    return s, e


class _Groups:
    """A row's (name_span, value_span, group_idx) candidates."""

    def __init__(self) -> None:
        self.items: list[tuple[tuple[int, int] | None, tuple[int, int], int]] = []

    def add(self, name: tuple[int, int] | None, value: tuple[int, int], idx: int) -> None:
        self.items.append((name, value, idx))


def _extract_groups(seg: str, source_kind: str) -> _Groups:
    groups = _Groups()
    chunks = _split_offsets(seg, ";") if ";" in seg else [(0, len(seg))]
    for idx, (cs, ce) in enumerate(chunks):
        chunk = seg[cs:ce]
        colon = chunk.find(":")
        if colon != -1:
            ns, ne = _trim(seg, cs, cs + colon)
            vs, ve = _trim(seg, cs + colon + 1, ce)
            name = (ns, ne) if ne > ns else None
            if ve > vs:
                groups.add(name, (vs, ve), idx)
        elif "\t" in chunk:
            cells = _split_offsets(chunk, "\t")
            name_span: tuple[int, int] | None = None
            for j, (a, b) in enumerate(cells):
                s, e = _trim(seg, cs + a, cs + b)
                if e <= s:
                    continue
                if j == 0:
                    name_span = (s, e)
                else:
                    groups.add(name_span, (s, e), idx)
        else:
            s, e = _trim(seg, cs, ce)
            if e > s:
                groups.add(None, (s, e), idx)
    return groups


def _source_kind(seg: str) -> str:
    if ";" in seg:
        return "semicolon_row"
    if "\t" in seg:
        return "table_like"
    if ":" in seg:
        return "key_value"
    return "narrative"


def parse_graph(
    graph: DocumentGraph, routings: list[NodeRouting],
    lex: LabLexicon, config_version: str,
) -> SpecialistRunResult:
    """Run the lab parser over C2-activated canonical routed units."""
    proposals: list[SpanProposal] = []
    relations: list[RelationProposal] = []
    warnings: list[str] = []
    p_counter = 0
    r_counter = 0
    parse_counter = 0
    pg_counter = 0

    for routing in routings:
        if "C2" not in set(routing.route_tags):
            continue
        seg = routing.text
        cstart = routing.start
        if not seg.strip():
            continue
        node_id = routing.node_id
        skind = _source_kind(seg)
        key_value = skind == "key_value"
        structured = skind in ("key_value", "table_like", "semicolon_row")
        parse_counter += 1
        parse_id = f"labparse-{graph.document_id}-{parse_counter:04d}"

        groups = _extract_groups(seg, skind)
        line_names: list[SpanProposal] = []
        result_groups: list[ResultGroup] = []
        group_of: dict[str, int] = {}

        for name_span, value_span, gidx in groups.items:
            value_local = seg[value_span[0]:value_span[1]]
            # Hard negatives may appear in the key (e.g. "Tuổi:") or the value
            # (e.g. a date / bare med dose) — check the whole name+value region.
            check_start = name_span[0] if name_span is not None else value_span[0]
            if hard_negative(seg[check_start:value_span[1]], lex) is not None:
                warnings.append(f"lab_hard_negative_skipped:{node_id}")
                continue
            ref = find_reference(value_local, lex)
            exclude = (ref.start, ref.end) if ref else None
            value = find_value(value_local, lex, exclude=exclude)
            if value is None:  # no numeric/qualitative value → never fabricate one
                continue
            base = cstart + value_span[0]
            unit = find_unit(value_local, lex, value.end)
            flag = find_flag(value_local, lex, value.end)

            if name_span is None:  # recover a lexicon test name before the value
                before = value_span[0] + value.start
                nm = lex.test_re.search(seg, 0, before)
                if nm is not None and nm.end() <= before:
                    name_span = (nm.start(), nm.end())

            name_prop: SpanProposal | None = None
            if name_span is not None:
                nseg = seg[name_span[0]:name_span[1]]
                known = nseg.lower() in lex.test_names
                if not (skind == "narrative" and not known):
                    p_counter += 1
                    if not known:
                        warnings.append(f"unknown_test_name:{node_id}")
                    name_comp = ComponentSpan("test_name", cstart + name_span[0],
                                              cstart + name_span[1], nseg, nseg.lower())
                    name_prop = SpanProposal(
                        proposal_id=f"labname-{graph.document_id}-{p_counter:04d}",
                        document_id=graph.document_id, start=cstart + name_span[0],
                        end=cstart + name_span[1], text=nseg, proposed_types=(NAME_LABEL,),
                        source_specialist="laboratory", source_node_id=node_id,
                        source_routes=("C2",), source_node_kind=routing.node_kind,
                        parent_line_id=routing.parent_line_id, boundary_group_id=None,
                        local_score=score_test_name(lex, known=known, structured_row=structured,
                                                    key_value=key_value),
                        matched_rule=f"lab:test_name:{skind}", normalized_form=nseg.lower(),
                        parse_ref=parse_id, components=(name_comp,),
                        config_version=config_version, lexicon_version=lex.lexicon_version,
                        warnings=() if known else ("unknown_test_name",),
                        features={"source_" + skind: 1.0})
                    proposals.append(name_prop)
                    line_names.append(name_prop)
                    group_of[name_prop.proposal_id] = gidx

            # TEST_RESULT boundary alternatives sharing a result boundary group id.
            v_comp = ComponentSpan(value.role, base + value.start, base + value.end,
                                   value.text, value.normalized)
            rg_id = f"labrg-{graph.document_id}-{parse_counter:04d}-{gidx}-{v_comp.start}"
            extra: list[ComponentSpan] = []
            if unit is not None:
                extra.append(ComponentSpan("unit", base + unit.start, base + unit.end, unit.text))
            if flag is not None:
                extra.append(ComponentSpan("flag", base + flag.start, base + flag.end, flag.text))
            if ref is not None:
                extra.append(ComponentSpan("reference_range", cstart + value_span[0] + ref.start,
                                           cstart + value_span[0] + ref.end, ref.text))
            variants: list[tuple[str, int, int]] = [("value_only", v_comp.start, v_comp.end)]
            if unit is not None and unit.start >= value.end:
                variants.append(("value_unit", v_comp.start, base + unit.end))

            members: list[SpanProposal] = []
            for kind, s_abs, e_abs in variants:
                p_counter += 1
                rp = SpanProposal(
                    proposal_id=f"labres-{graph.document_id}-{p_counter:04d}",
                    document_id=graph.document_id, start=s_abs, end=e_abs,
                    text=graph.original_text[s_abs:e_abs], proposed_types=(RESULT_LABEL,),
                    source_specialist="laboratory", source_node_id=node_id,
                    source_routes=("C2",), source_node_kind=routing.node_kind,
                    parent_line_id=routing.parent_line_id, boundary_group_id=rg_id,
                    local_score=score_test_result(lex, structured_row=structured,
                                                  key_value=key_value, has_unit=unit is not None,
                                                  has_reference=ref is not None),
                    matched_rule=f"lab:test_result:{kind}:{skind}",
                    normalized_form=value.normalized, parse_ref=parse_id,
                    components=(v_comp, *extra),
                    config_version=config_version, lexicon_version=lex.lexicon_version,
                    features={"boundary_" + kind: 1.0})
                proposals.append(rp)
                members.append(rp)
            result_groups.append(ResultGroup(rg_id, members[0], tuple(members), gidx,
                                             has_unit=unit is not None))

        # Global min-cost pairing of names → result GROUPS; relations propagate to
        # every boundary alternative in the matched group (one logical pair).
        for lp in pair_names_to_groups(line_names, result_groups, group_of,
                                       weights=lex.pairing_weights, max_cost=lex.pairing_max_cost,
                                       line_len=len(seg) or 1):
            pg_counter += 1
            pair_group_id = f"labpg-{graph.document_id}-{pg_counter:04d}"
            score = round(max(0.0, 1.0 - lp.cost), 6)
            for member in lp.members:
                r_counter += 1
                is_primary = bool(member.matched_rule and ":value_only:" in member.matched_rule)
                relations.append(RelationProposal(
                    relation_id=f"labrel-{graph.document_id}-{r_counter:04d}",
                    document_id=graph.document_id, relation_type=HAS_RESULT,
                    source_proposal_id=lp.name_id, target_proposal_id=member.proposal_id,
                    score=score, pairing_cost=lp.cost, pair_group_id=pair_group_id,
                    is_primary=is_primary, target_boundary_group_id=lp.result_group_id,
                    source_node_id=node_id, provenance=f"pairing:{skind}",
                    warnings=() if lp.same_group else ("cross_group_pairing",)))

    return SpecialistRunResult(specialist="laboratory", proposals=tuple(proposals),
                               relations=tuple(relations), warnings=tuple(warnings))


__all__ = ["parse_graph"]
