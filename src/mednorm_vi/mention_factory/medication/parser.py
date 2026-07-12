"""Deterministic medication grammar parser (high-recall proposal source).

Consumes canonical routable units routed to a medication case (structured C1/C5,
or C3 narrative with strong deterministic evidence) and emits a MedicationParse
plus multiple boundary-candidate SpanProposals per ingredient. It never selects a
single final boundary (L4 does). Exact original offsets are always preserved.
"""

from __future__ import annotations

import re

from ...case_router.models import NodeRouting
from ...document_intelligence.models import DocumentGraph
from ...schemas.constants import ORGANIZER_LABEL_BY_TYPE
from ..models import ComponentSpan, SpanProposal, SpecialistRunResult
from .lexicon import MedicationLexicon
from .models import MedicationParse
from .patterns import hard_negative, scan_duration, scan_first_role, scan_strength
from .scoring import score_medication

_WORD_BEFORE = re.compile(r"([^\W\d_][\w\-]*)\s*$", re.UNICODE)
# Administration predicates that (with a strength) license medication in narrative.
_ADMIN_PREDICATE = re.compile(
    r"\b(dùng|uống|kê|chỉ định|điều trị bằng|cho|tiêm|administer(?:ed)?|prescribed|taking|"
    r"start(?:ed)?|takes?)\b", re.IGNORECASE | re.UNICODE)
MED_LABEL = ORGANIZER_LABEL_BY_TYPE["MEDICATION"]  # organizer label, evidence only


def _strong_narrative_medication(seg: str, lex: MedicationLexicon, in_med_section: bool) -> bool:
    """Strong deterministic evidence that a C3 narrative segment names a medication.

    True when: a known ingredient + (strength | route | frequency); OR an
    administration predicate + a strength; OR a medication-section context + a
    known ingredient. Otherwise medication grammar does NOT run on narrative.
    """
    known = lex.ingredient_re.search(seg) is not None
    has_strength = lex.strength_re.search(seg) is not None or lex.concentration_re.search(seg)
    has_route = lex.route_re.search(seg) is not None
    has_freq = lex.frequency_re.search(seg) is not None
    if known and (has_strength or has_route or has_freq):
        return True
    if _ADMIN_PREDICATE.search(seg) is not None and has_strength:
        return True
    return bool(in_med_section and known)


def _abs_component(role: str, cstart: int, lm_start: int, lm_end: int, text: str,
                   normalized: str | None = None) -> ComponentSpan:
    return ComponentSpan(role=role, start=cstart + lm_start, end=cstart + lm_end,
                         text=text, normalized=normalized)


def _parse_ingredient(
    document_id: str, node_id: str, cstart: int, seg: str,
    name_local_start: int, name_local_end: int, scope_end: int, name_known: bool,
    lex: MedicationLexicon, parse_id: str,
) -> MedicationParse:
    name_text = seg[name_local_start:name_local_end]
    components: list[ComponentSpan] = [
        _abs_component("name", cstart, name_local_start, name_local_end, name_text,
                       name_text.lower())
    ]
    after = name_local_end
    # strength / concentration (split into value + unit)
    for lm in scan_strength(seg, lex, after):
        if lm.start >= scope_end:
            break
        components.append(_abs_component(lm.role, cstart, lm.start, lm.end, lm.text, lm.normalized))
    for role, pat in (
        ("release", lex.release_re), ("dose_form", lex.dose_form_re),
        ("route", lex.route_re), ("frequency", lex.frequency_re), ("prn", lex.prn_re),
        ("salt", lex.salt_re),
    ):
        fm = scan_first_role(role, pat, seg, after)
        if fm is not None and fm.start < scope_end:
            components.append(_abs_component(role, cstart, fm.start, fm.end, fm.text))
    dur = scan_duration(seg, lex, after)
    if dur is not None and dur.start < scope_end:
        components.append(_abs_component("duration", cstart, dur.start, dur.end, dur.text))

    warnings: list[str] = []
    if not name_known:
        warnings.append("unknown_medication_name")
    if not ({c.role for c in components} & {"strength_value", "concentration"}):
        warnings.append("incomplete_parse_no_strength")
    components.sort(key=lambda c: (c.start, c.end, c.role))
    return MedicationParse(
        parse_id=parse_id, document_id=document_id, source_node_id=node_id,
        name_start=cstart + name_local_start, name_end=cstart + name_local_end,
        name_text=name_text, name_known=name_known, components=tuple(components),
        warnings=tuple(warnings))


def _boundary_candidates(parse: MedicationParse) -> list[tuple[str, int, int]]:
    """Deterministic progressive right-extension boundaries (name → full)."""
    by_role: dict[str, ComponentSpan] = {}
    for c in parse.components:
        by_role.setdefault(c.role, c)  # first occurrence per role
    name = (parse.name_start, parse.name_end)
    cands: list[tuple[str, int, int]] = [("name_only", name[0], name[1])]
    strength_end = max((c.end for c in parse.components
                        if c.role in ("strength_value", "strength_unit", "concentration")),
                       default=None)
    if strength_end is not None and strength_end > name[1]:
        cands.append(("name_strength", name[0], strength_end))
    if "dose_form" in by_role and by_role["dose_form"].end > name[1]:
        cands.append(("name_strength_form", name[0], by_role["dose_form"].end))
    if "route" in by_role and by_role["route"].end > name[1]:
        cands.append(("name_strength_route", name[0], by_role["route"].end))
    full_end = max((c.end for c in parse.components), default=name[1])
    if full_end > name[1]:
        cands.append(("full", name[0], full_end))
    # de-duplicate identical (start,end), keeping the earliest kind
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, int, int]] = []
    for kind, s, e in cands:
        if (s, e) not in seen:
            seen.add((s, e))
            out.append((kind, s, e))
    return out


_MED_SECTIONS = frozenset({"home_medications", "pre_admission_medications",
                           "current_medications"})


def parse_graph(
    graph: DocumentGraph, routings: list[NodeRouting],
    lex: MedicationLexicon, config_version: str,
) -> SpecialistRunResult:
    """Run the medication grammar over medication-activated canonical routed units.

    Activation: C1/C5 units (full grammar incl. unknown-name heuristic), or C3
    narrative units with STRONG deterministic medication evidence (known names
    only). The router guarantees one unit per content region (no duplicate runs).
    """
    proposals: list[SpanProposal] = []
    warnings: list[str] = []
    p_counter = 0
    parse_counter = 0

    for routing in routings:
        cases = set(routing.route_tags)
        seg = routing.text
        if not seg.strip():
            continue
        in_med_section = routing.section_category in _MED_SECTIONS
        structured = bool({"C1", "C5"} & cases)
        narrative = (not structured and "C3" in cases
                     and _strong_narrative_medication(seg, lex, in_med_section))
        if not (structured or narrative):
            continue
        restrict_to_known = narrative  # narrative context → known ingredients only
        cstart = routing.start
        in_list = routing.node_kind == "list_item"

        names = [(m.start(), m.end(), True) for m in lex.ingredient_re.finditer(seg)]
        if not names and not restrict_to_known and lex.unknown_requires_structure:
            # An unknown name needs real medication structure — a STRENGTH /
            # concentration anchor. A bare route/dose cue (e.g. the verb "uống")
            # is not enough, so ordinary prose never fabricates a drug name.
            anchor = None
            for pat in (lex.strength_re, lex.concentration_re):
                m = pat.search(seg)
                if m is not None and (anchor is None or m.start() < anchor):
                    anchor = m.start()
            if anchor is not None:
                wm = _WORD_BEFORE.search(seg[:anchor])
                if wm is not None and len(wm.group(1)) >= 3:
                    names = [(wm.start(1), wm.end(1), False)]
        if not names:
            continue
        hn = hard_negative(seg, lex)
        routes = tuple(sorted({"C1", "C5", "C3"} & cases))
        starts = [n[0] for n in names]
        for ns, ne, known in names:
            if not known and hn is not None:
                warnings.append(f"suppressed_unknown_medication_hard_negative:{routing.node_id}")
                continue
            scope_end = min([s for s in starts if s > ns], default=len(seg))
            parse_counter += 1
            parse = _parse_ingredient(
                graph.document_id, routing.node_id, cstart, seg, ns, ne, scope_end, known, lex,
                f"medparse-{graph.document_id}-{parse_counter:04d}")
            for kind, s_abs, e_abs in _boundary_candidates(parse):
                p_counter += 1
                score = score_medication(lex, parse, kind, s_abs, e_abs,
                                         in_med_section=in_med_section, in_list=in_list)
                proposals.append(SpanProposal(
                    proposal_id=f"medprop-{graph.document_id}-{p_counter:04d}",
                    document_id=graph.document_id, start=s_abs, end=e_abs,
                    text=graph.original_text[s_abs:e_abs], proposed_types=(MED_LABEL,),
                    source_specialist="medication", source_node_id=routing.node_id,
                    source_routes=routes, local_score=score,
                    matched_rule=f"med_grammar:{kind}", normalized_form=parse.name_text.lower(),
                    parse_ref=parse.parse_id, boundary_group_id=parse.parse_id,
                    source_node_kind=routing.node_kind, parent_line_id=routing.parent_line_id,
                    components=parse.components,
                    config_version=config_version, lexicon_version=lex.lexicon_version,
                    warnings=parse.warnings,
                    features={"boundary_kind_full": 1.0 if kind == "full" else 0.0,
                              "narrative_activation": 1.0 if narrative else 0.0}))
    return SpecialistRunResult(specialist="medication", proposals=tuple(proposals),
                               warnings=tuple(warnings))


__all__ = ["parse_graph"]
