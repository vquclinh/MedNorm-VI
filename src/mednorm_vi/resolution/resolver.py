"""Deterministic L4 Boundary & Type Resolver foundation (Phase 1C-A).

Consumes Phase 1B proposals + relations and produces ``EntityHypothesis`` objects
with a chosen boundary, retained alternatives, and accepted/rejected/unresolved
status. It performs NO ontology linking, preserves exact offsets, never
deduplicates by text, keeps repeated occurrences distinct, and retains
``has_result`` evidence without requiring pairing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..mention_factory.models import RelationProposal, SpanProposal
from .boundary import choose_boundary
from .features import boundary_kind, group_key
from .models import (
    STATUS_ACCEPTED,
    STATUS_UNRESOLVED,
    BoundaryAlternative,
    BoundaryEvidence,
    EntityHypothesis,
    ResolutionResult,
)
from .overlap import resolve_overlaps
from .scoring import score_hypothesis
from .typing import assign_type


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    config_version: str = "resolver-v1"
    medication_boundary: str = "full"  # confirmed golden example favours the full span
    test_result_boundary: str = "value_only"  # UNRESOLVED policy; conservative default
    abstain_on_conflict: bool = False
    config_hash: str = ""

    @staticmethod
    def load(path: str | Path) -> ResolverConfig:
        import yaml  # type: ignore[import-untyped]

        doc: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        h = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        return ResolverConfig(
            config_version=str(doc.get("config_version", "resolver-v1")),
            medication_boundary=str(doc.get("medication_boundary", "full")),
            test_result_boundary=str(doc.get("test_result_boundary", "value_only")),
            abstain_on_conflict=bool(doc.get("abstain_on_conflict", False)),
            config_hash=h)


def _pair_groups_by_proposal(relations: list[RelationProposal]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in relations:
        if r.pair_group_id:
            out.setdefault(r.source_proposal_id, set()).add(r.pair_group_id)
            out.setdefault(r.target_proposal_id, set()).add(r.pair_group_id)
    return out


def resolve(
    document_id: str, original_text: str, proposals: list[SpanProposal],
    relations: list[RelationProposal], config: ResolverConfig,
) -> ResolutionResult:
    """Resolve Phase 1B proposals into deterministic entity hypotheses."""
    # 1. group proposals into logical mentions
    groups: dict[str, list[SpanProposal]] = {}
    for p in proposals:
        groups.setdefault(group_key(p), []).append(p)

    pair_groups = _pair_groups_by_proposal(relations)
    warnings: list[str] = []

    # 2. one hypothesis per group (order groups deterministically for stable ids)
    ordered_keys = sorted(
        groups, key=lambda k: (min(p.start for p in groups[k]),
                               min(p.end for p in groups[k]), k))
    provisional: list[EntityHypothesis] = []
    for idx, key in enumerate(ordered_keys):
        group = sorted(groups[key], key=lambda p: (p.start, p.end, p.proposal_id))
        entity_type, type_ev = assign_type(group)
        chosen, considered, note = choose_boundary(
            entity_type, group, medication_policy=config.medication_boundary,
            test_result_policy=config.test_result_boundary)
        alternatives = tuple(
            BoundaryAlternative(p.proposal_id, p.start, p.end, p.text, boundary_kind(p))
            for p in group if p.proposal_id != chosen.proposal_id)
        pgids = set()
        for p in group:
            pgids |= pair_groups.get(p.proposal_id, set())
        status = STATUS_ACCEPTED
        if entity_type not in ("THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
            status = STATUS_UNRESOLVED
            warnings.append(f"unresolvable_type:{entity_type}:{key}")
        provisional.append(EntityHypothesis(
            hypothesis_id=f"ent-{document_id}-{idx + 1:04d}",
            document_id=document_id, start=chosen.start, end=chosen.end, text=chosen.text,
            entity_type=entity_type, status=status, chosen_proposal_id=chosen.proposal_id,
            source_proposal_ids=tuple(p.proposal_id for p in group),
            boundary_evidence=BoundaryEvidence(
                policy=(config.medication_boundary if entity_type == "THUỐC"
                        else config.test_result_boundary if entity_type == "KẾT_QUẢ_XÉT_NGHIỆM"
                        else "single"),
                chosen_kind=boundary_kind(chosen), chosen_proposal_id=chosen.proposal_id,
                considered_kinds=considered, note=note),
            type_evidence=type_ev, retained_alternatives=alternatives,
            has_result_pair_group_ids=tuple(sorted(pgids)),
            score=score_hypothesis(chosen, group)))

    # 3. deterministic same-type overlap resolution
    resolved = resolve_overlaps(provisional, abstain_on_conflict=config.abstain_on_conflict)
    resolved.sort(key=lambda h: (h.start, h.end, h.entity_type, h.hypothesis_id))
    return ResolutionResult(document_id=document_id, hypotheses=tuple(resolved),
                            config_version=config.config_version, warnings=tuple(warnings))


__all__ = ["ResolverConfig", "resolve"]
