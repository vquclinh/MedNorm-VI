"""Central proposal/relation validation (offset + provenance + relation integrity).

Fails fast on: out-of-bounds spans, text/offset mismatch, missing provenance,
missing route references, broken/cross-document relation endpoints, duplicate
proposal/relation ids, unsupported ontology codes, and any accidental final
organizer entity emission. Warnings for low-confidence / ambiguity.
"""

from __future__ import annotations

from ..schemas.constants import ORGANIZER_LABELS
from ..validator.results import Severity, ValidationIssue, ValidationResult
from .models import RelationProposal, SpanProposal

_ALLOWED_PROPOSED = {"THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}  # med/lab proposals


def validate_proposals(
    proposals: list[SpanProposal], relations: list[RelationProposal], original_text: str
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    n = len(original_text)
    seen: set[str] = set()
    for p in proposals:
        if p.proposal_id in seen:
            issues.append(ValidationIssue("proposal.duplicate_id",
                                          f"duplicate proposal id {p.proposal_id}"))
        seen.add(p.proposal_id)
        if not (0 <= p.start <= p.end <= n):
            issues.append(ValidationIssue("proposal.out_of_bounds",
                                          f"{p.proposal_id} span out of bounds"))
        elif original_text[p.start : p.end] != p.text:
            issues.append(ValidationIssue("proposal.text_offset_mismatch",
                                          f"{p.proposal_id} text/offset mismatch"))
        if not p.source_specialist or not p.source_node_id:
            issues.append(ValidationIssue("proposal.missing_provenance",
                                          f"{p.proposal_id} missing provenance"))
        if not p.source_routes:
            issues.append(ValidationIssue("proposal.missing_route",
                                          f"{p.proposal_id} has no source route"))
        for t in p.proposed_types:
            if t not in _ALLOWED_PROPOSED:
                issues.append(ValidationIssue("proposal.unsupported_type",
                                              f"{p.proposal_id} unexpected proposed type {t!r}"))
        # Phase 1B must not emit final candidate codes or organizer-final entities.
        if any(k in p.features for k in ("icd10", "rxnorm", "candidate")):
            issues.append(ValidationIssue("proposal.final_code",
                                          f"{p.proposal_id} carries a final ontology code"))
        # ambiguity warnings
        if "unknown_medication_name" in p.warnings or "unknown_test_name" in p.warnings:
            issues.append(ValidationIssue("proposal.unknown_name",
                                          f"{p.proposal_id} unknown name",
                                          severity=Severity.WARNING))

    prop_ids = {p.proposal_id for p in proposals}
    prop_doc = {p.proposal_id: p.document_id for p in proposals}
    seen_rel: set[str] = set()
    for r in relations:
        if r.relation_id in seen_rel:
            issues.append(ValidationIssue("relation.duplicate_id",
                                          f"duplicate relation id {r.relation_id}"))
        seen_rel.add(r.relation_id)
        if r.source_proposal_id not in prop_ids or r.target_proposal_id not in prop_ids:
            issues.append(ValidationIssue("relation.broken_endpoint",
                                          f"{r.relation_id} endpoint not a known proposal"))
            continue
        if prop_doc[r.source_proposal_id] != prop_doc[r.target_proposal_id]:
            issues.append(ValidationIssue("relation.cross_document",
                                          f"{r.relation_id} crosses documents"))
        if r.document_id != prop_doc[r.source_proposal_id]:
            issues.append(ValidationIssue("relation.document_mismatch",
                                          f"{r.relation_id} document mismatch"))
    return ValidationResult.from_issues(issues)


def contains_organizer_final_entity(features: dict[str, float]) -> bool:
    """A guard used by tests: proposal features must not contain organizer labels."""
    return any(k in ORGANIZER_LABELS for k in features)


__all__ = ["validate_proposals", "contains_organizer_final_entity"]
