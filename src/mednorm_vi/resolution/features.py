"""Deterministic feature extraction from Phase 1B proposals (Phase 1C-A)."""

from __future__ import annotations

from ..mention_factory.models import SpanProposal


def boundary_kind(proposal: SpanProposal) -> str:
    """Extract the boundary kind from a proposal's ``matched_rule``.

    Medication: ``med_grammar:<kind>`` -> ``<kind>``.
    Laboratory result: ``lab:test_result:<value_only|value_unit>:...`` -> that token.
    Laboratory name / anything else: a stable fallback token.
    """
    rule = proposal.matched_rule or ""
    if rule.startswith("med_grammar:"):
        return rule.split(":", 1)[1]
    if ":value_only:" in rule or rule.endswith(":value_only"):
        return "value_only"
    if ":value_unit:" in rule or rule.endswith(":value_unit"):
        return "value_unit"
    if "test_name" in rule:
        return "test_name"
    return "single"


def group_key(proposal: SpanProposal) -> str:
    """Logical-mention key: shared ``boundary_group_id`` else the proposal id.

    Medication boundary alternatives share the parse's boundary_group_id; a lab
    result's value-only/value+unit alternatives share the result-group id; a lab
    test name has no group and is its own singleton.
    """
    return proposal.boundary_group_id or proposal.proposal_id


def width(proposal: SpanProposal) -> int:
    return proposal.end - proposal.start


__all__ = ["boundary_kind", "group_key", "width"]
