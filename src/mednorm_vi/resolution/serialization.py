"""Deterministic debug serialization for resolver output (Phase 1C-A)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import EntityHypothesis, ResolutionResult

SCHEMA_VERSION = "resolution-1c-a"


def hypothesis_to_dict(h: EntityHypothesis) -> dict[str, Any]:
    return {
        "hypothesis_id": h.hypothesis_id,
        "document_id": h.document_id,
        "start": h.start,
        "end": h.end,
        "text": h.text,
        "entity_type": h.entity_type,
        "status": h.status,
        "chosen_proposal_id": h.chosen_proposal_id,
        "source_proposal_ids": list(h.source_proposal_ids),
        "boundary_evidence": {
            "policy": h.boundary_evidence.policy,
            "chosen_kind": h.boundary_evidence.chosen_kind,
            "considered_kinds": list(h.boundary_evidence.considered_kinds),
            "note": h.boundary_evidence.note,
        },
        "type_evidence": {
            "entity_type": h.type_evidence.entity_type,
            "source_specialist": h.type_evidence.source_specialist,
            "proposed_types": list(h.type_evidence.proposed_types),
            "note": h.type_evidence.note,
        },
        "retained_alternatives": [
            {"proposal_id": a.proposal_id, "start": a.start, "end": a.end,
             "text": a.text, "kind": a.kind}
            for a in h.retained_alternatives
        ],
        "overlap_decision": (
            {"outcome": h.overlap_decision.outcome,
             "counterpart_id": h.overlap_decision.counterpart_id,
             "reason": h.overlap_decision.reason}
            if h.overlap_decision else None),
        "rejection_reason": h.rejection_reason,
        "has_result_pair_group_ids": list(h.has_result_pair_group_ids),
        "score": h.score,
        "features": {k: h.features[k] for k in sorted(h.features)},
    }


def to_debug_dict(result: ResolutionResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": result.document_id,
        "config_version": result.config_version,
        "counts": {
            "hypotheses": len(result.hypotheses),
            "accepted": len(result.accepted()),
            "rejected": len(result.rejected()),
            "unresolved": len(result.unresolved()),
        },
        "hypotheses": [hypothesis_to_dict(h) for h in result.hypotheses],
        "warnings": sorted(set(result.warnings)),
    }


def to_json(result: ResolutionResult) -> str:
    return json.dumps(to_debug_dict(result), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def determinism_hash(result: ResolutionResult) -> str:
    canonical = json.dumps(to_debug_dict(result), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_VERSION", "hypothesis_to_dict", "to_debug_dict", "to_json",
           "determinism_hash"]
