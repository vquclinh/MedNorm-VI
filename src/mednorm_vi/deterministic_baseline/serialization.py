"""Deterministic debug serialization for Phase 1B (NOT organizer output)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..case_router.serialization import routings_to_list
from ..mention_factory.models import RelationProposal, SpanProposal
from .models import Phase1BResult

SCHEMA_VERSION = "phase1b-1"


def _component(c: Any) -> dict[str, Any]:
    return {"role": c.role, "start": c.start, "end": c.end, "text": c.text,
            "normalized": c.normalized, "detail": c.detail}


def proposal_to_dict(p: SpanProposal) -> dict[str, Any]:
    return {
        "proposal_id": p.proposal_id,
        "document_id": p.document_id,
        "start": p.start,
        "end": p.end,
        "text": p.text,
        "proposed_types": list(p.proposed_types),
        "source_specialist": p.source_specialist,
        "source_node_id": p.source_node_id,
        "source_routes": list(p.source_routes),
        "local_score": p.local_score,
        "matched_rule": p.matched_rule,
        "normalized_form": p.normalized_form,
        "parse_ref": p.parse_ref,
        "boundary_group_id": p.boundary_group_id,
        "source_node_kind": p.source_node_kind,
        "parent_line_id": p.parent_line_id,
        "evidence_ids": list(p.evidence_ids),
        "components": [_component(c) for c in p.components],
        "config_version": p.config_version,
        "lexicon_version": p.lexicon_version,
        "warnings": list(p.warnings),
        "features": {k: p.features[k] for k in sorted(p.features)},
    }


def relation_to_dict(r: RelationProposal) -> dict[str, Any]:
    return {
        "relation_id": r.relation_id,
        "document_id": r.document_id,
        "relation_type": r.relation_type,
        "source_proposal_id": r.source_proposal_id,
        "target_proposal_id": r.target_proposal_id,
        "score": r.score,
        "pairing_cost": r.pairing_cost,
        "pair_group_id": r.pair_group_id,
        "is_primary": r.is_primary,
        "target_boundary_group_id": r.target_boundary_group_id,
        "source_node_id": r.source_node_id,
        "evidence_ids": list(r.evidence_ids),
        "warnings": list(r.warnings),
        "provenance": r.provenance,
    }


def count_summary(result: Phase1BResult) -> dict[str, int]:
    """Deterministic Phase-1B count summary (canonical units, cases, specialist
    execution, medication/laboratory parses, and the logical-pair vs concrete
    relation-alternative distinction). Pure function of ``result``.
    """
    cases = {f"C{i}": 0 for i in range(1, 8)}
    units_by_kind: dict[str, int] = {}
    multi = 0
    for r in result.routings:
        units_by_kind[r.node_kind] = units_by_kind.get(r.node_kind, 0) + 1
        if len(r.cases) >= 2:
            multi += 1
        for c in r.cases:
            cases[c.case] = cases.get(c.case, 0) + 1
    med = result.medication_proposals()
    lab = result.laboratory_proposals()
    # A parent line decomposed into child units is never re-routed as a whole line,
    # preventing a second specialist run over the same content.
    prevented = len({r.parent_line_id for r in result.routings
                     if r.node_kind != "line" and r.parent_line_id is not None})
    rels = result.relations
    logical_pairs = len({r.pair_group_id for r in rels if r.pair_group_id}) + \
        sum(1 for r in rels if not r.pair_group_id)
    out: dict[str, int] = {
        "routable_nodes": len(result.routings),
        **{f"unit_{k}": v for k, v in units_by_kind.items()},
        **cases,
        "multi_route": multi,
        "exec_medication": len({p.source_node_id for p in med}),
        "exec_laboratory": len({p.source_node_id for p in lab}),
        "duplicate_specialist_prevention": prevented,
        "medication_parses": len({p.parse_ref for p in med if p.parse_ref}),
        "medication_proposals": len(med),
        "laboratory_parses": len({p.parse_ref for p in lab if p.parse_ref}),
        "test_name_proposals": sum(1 for p in lab if "TÊN_XÉT_NGHIỆM" in p.proposed_types),
        "test_result_proposals": sum(1 for p in lab if "KẾT_QUẢ_XÉT_NGHIỆM" in p.proposed_types),
        "logical_test_result_pairs": logical_pairs,
        "relation_alternatives": len(rels),
        "has_result_relations": len(rels),
        "duplicate_identity_groups": len(result.merge_diagnostics.duplicate_identity_groups),
        "overlap_pairs": len(result.merge_diagnostics.overlap_pairs),
    }
    return out


def to_debug_dict(result: Phase1BResult) -> dict[str, Any]:
    diag = result.merge_diagnostics
    summary = count_summary(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": result.document_id,
        "l1_valid": result.l1_valid,
        "proposals_valid": result.proposals_valid,
        "counts": {k: summary[k] for k in sorted(summary)},
        "routings": routings_to_list(list(result.routings)),
        "proposals": [proposal_to_dict(p) for p in result.proposals],
        "relations": [relation_to_dict(r) for r in result.relations],
        "merge_diagnostics": {
            "duplicate_identity_groups": [list(g) for g in diag.duplicate_identity_groups],
            "overlap_pairs": [list(p) for p in diag.overlap_pairs],
            "repeated_surface": list(diag.repeated_surface),
        },
        "warnings": sorted(set(result.warnings)),
        "issues": [{"code": i.code, "severity": i.severity.value, "message": i.message}
                   for i in result.issues],
    }


def to_json(result: Phase1BResult) -> str:
    return json.dumps(to_debug_dict(result), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def determinism_hash(result: Phase1BResult) -> str:
    canonical = json.dumps(to_debug_dict(result), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_VERSION", "count_summary", "proposal_to_dict", "relation_to_dict",
           "to_debug_dict", "to_json", "determinism_hash"]
