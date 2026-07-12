"""Phase 1B pipeline: L1 → router → specialists → relations → merge → validate.

Deterministic and offset-preserving. Produces PROPOSALS only (no final entities,
assertions, ontology codes, or organizer JSON).
"""

from __future__ import annotations

from ..case_router.router import CaseRouter
from ..case_router.validation import validate_routings
from ..document_intelligence.models import DocumentGraph
from ..document_intelligence.validation import validate_graph
from ..mention_factory.laboratory import parse_graph as lab_parse
from ..mention_factory.medication import parse_graph as med_parse
from ..mention_factory.merge import collect
from ..mention_factory.validation import validate_proposals
from .models import Phase1BConfig, Phase1BResult


def run_phase1b(graph: DocumentGraph, config: Phase1BConfig) -> Phase1BResult:
    """Run the deterministic Phase 1B pipeline over an L1 DocumentGraph."""
    warnings: list[str] = []

    # 1. Validate the L1 graph (fail-fast signal, surfaced in the result).
    l1_result = validate_graph(graph)

    # 2-3. Multi-label routing (structural C1-C5) over CANONICAL routable units.
    router = CaseRouter(config.router)
    routings = router.route_graph(graph)

    # 4-5. Activate specialists by route and run them over the routed units.
    med = med_parse(graph, routings, config.medication,
                    config.medication_config_version)
    lab = lab_parse(graph, routings, config.laboratory,
                    config.laboratory_config_version)
    warnings.extend(med.warnings)
    warnings.extend(lab.warnings)

    all_proposals = list(med.proposals) + list(lab.proposals)
    all_relations = list(med.relations) + list(lab.relations)

    # 6. Derive C6/C7 route tags from proposals (evidence only).
    routings = router.augment_derived(routings, all_proposals)

    # 7. Deterministic collection + overlap/duplicate diagnostics (never by text).
    ordered, diagnostics = collect(all_proposals)

    # 8-9. Validate proposals/relations + routings against original_text.
    prop_result = validate_proposals(ordered, all_relations, graph.original_text)
    route_result = validate_routings(routings, graph.original_text)
    issues = prop_result.issues + route_result.issues

    return Phase1BResult(
        document_id=graph.document_id,
        routings=tuple(routings),
        proposals=tuple(ordered),
        relations=tuple(all_relations),
        merge_diagnostics=diagnostics,
        warnings=tuple(warnings),
        l1_valid=l1_result.ok,
        proposals_valid=prop_result.ok and route_result.ok,
        issues=tuple(issues),
    )


__all__ = ["run_phase1b"]
