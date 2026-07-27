"""Type assignment for the L4 resolver.

Two layers live here:

* :func:`assign_type` — the Phase 1C-A direct read used by the deterministic
  foundation resolver. Unchanged: Phase 1B proposals carry a single organizer
  label per specialist, so typing is a direct read and the resolver does not
  invent CHẨN_ĐOÁN / TRIỆU_CHỨNG types that have no proposal source.
* :func:`type_utilities` / :func:`decide_type` — the v1 evidence combination
  (spec §7.2). Each evidence family is bounded to ``[0, 1]`` before it is
  weighted, so no family can dominate by scale alone, and **abstention** is a
  first-class outcome: a wrong type is double-penalised, so a weak winner or a
  near-tie is dropped rather than guessed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..lattice.models import (
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_VIHEALTHBERT,
    FAMILY_DETERMINISTIC,
    FAMILY_NEURAL,
)
from ..lattice.models import SpanProposal as LatticeProposal
from ..mention_factory.models import SpanProposal
from .config_v1 import ResolverV1Config
from .models import RESOLVABLE_TYPES, TypeEvidence

# Ontology evidence is carried on a source as a feature named ``ontology_<NAME>``.
ONTOLOGY_FEATURE_PREFIX = "ontology_"

# Spec §7.2: "Colon + numeric  WBC:14.43  ⇒ TEST_NAME / TEST_RESULT".
_NUMERIC_VALUE = re.compile(r"^[<>]?\s*\d+(?:[.,]\d+)?")
_COLON_THEN_NUMBER = re.compile(r"^\s*:\s*[<>]?\s*\d")

# The medication grammar is complete when it contributed this many components
# (name, strength value, strength unit, form/route/frequency…). Spec §6.1.
_FULL_GRAMMAR_COMPONENTS = 4.0


def assign_type(group: list[SpanProposal]) -> tuple[str, TypeEvidence]:
    """Return (entity_type, evidence) for a group of boundary alternatives."""
    rep = group[0]
    types = tuple(sorted({t for p in group for t in p.proposed_types}))
    entity_type = rep.proposed_types[0] if rep.proposed_types else ""
    note = ""
    if len(types) > 1:
        note = f"multiple proposed types across group: {types}"
    if entity_type not in RESOLVABLE_TYPES:
        note = f"type {entity_type!r} not resolvable in Phase 1C-A"
    return entity_type, TypeEvidence(
        entity_type=entity_type, source_specialist=rep.source_specialist,
        proposed_types=types, note=note)


# ------------------------------------------------------------------------------
# v1 evidence combination (spec §7.2)
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypeDecision:
    """The chosen type for one lattice node, or an abstention, with its reasons."""

    entity_type: str
    utility: float
    margin: float
    runner_up: str
    abstained: bool
    reason: str
    utilities: Mapping[str, float] = field(default_factory=dict)
    contributions: Mapping[str, float] = field(default_factory=dict)


def grammar_completeness(proposal: LatticeProposal) -> float:
    """How complete the E1 medication grammar parse was, in ``[0, 1]``."""
    best = 0.0
    for source in proposal.sources_for(EXPERT_MEDICATION_GRAMMAR):
        components = float(source.features.get("grammar_component_count", 0.0))
        best = max(best, min(1.0, components / _FULL_GRAMMAR_COMPONENTS))
    return best


def laboratory_evidence(proposal: LatticeProposal, entity_type: str) -> float:
    """Strength of E2's structural evidence for a laboratory type, in ``[0, 1]``."""
    if entity_type not in ("TEST_NAME", "TEST_RESULT"):
        return 0.0
    best = 0.0
    for source in proposal.sources_for(EXPERT_LABORATORY_PARSER):
        if entity_type in source.type_scores:
            best = max(best, min(1.0, float(source.local_score)))
    return best


def neural_score(proposal: LatticeProposal, entity_type: str) -> float:
    """The strongest E3 probability for this exact span and type."""
    best = 0.0
    for source in proposal.sources_for(EXPERT_VIHEALTHBERT):
        best = max(best, float(source.type_scores.get(entity_type, 0.0)))
    return best


def expert_agreement(proposal: LatticeProposal, entity_type: str) -> float:
    """1.0 when a deterministic AND the neural expert propose this exact span+type."""
    families = {
        source.family for source in proposal.sources
        if entity_type in source.type_scores
    }
    return 1.0 if {FAMILY_DETERMINISTIC, FAMILY_NEURAL} <= families else 0.0


def expert_disagreement(proposal: LatticeProposal, entity_type: str) -> float:
    """1.0 when another expert proposed this span with a *different* type only."""
    proposing = {
        source.expert_id for source in proposal.sources
        if entity_type in source.type_scores
    }
    dissenting = {
        source.expert_id for source in proposal.sources
        if source.expert_id not in proposing and source.type_scores
    }
    return 1.0 if proposing and dissenting else 0.0


def flagged_evidence_only(proposal: LatticeProposal, entity_type: str) -> bool:
    """True when every expert supporting this type flagged its own parse as weak.

    E1 and E2 record ``unknown_test_name`` / ``unknown_medication_name`` /
    ``incomplete_parse_no_strength`` when they matched structure without a lexicon
    or grammar anchor. If that is the *only* support a type has, the evidence is
    self-reported as weak — which is exactly the case spec §7.2 says to drop.
    """
    supporting = [
        source for source in proposal.sources if entity_type in source.type_scores
    ]
    if not supporting:
        return False
    return all(
        source.family == FAMILY_DETERMINISTIC and bool(source.warnings)
        for source in supporting)


def ontology_evidence(
    proposal: LatticeProposal, entity_type: str, config: ResolverV1Config,
) -> tuple[float, bool]:
    """``(allowed evidence strength, conflict)`` for one type.

    Spec §7.3 forbids ICD candidates on MEDICATION and RxNorm candidates on
    DIAGNOSIS. That is enforced here as an *evidence* rule: forbidden ontology
    evidence contributes nothing and instead raises a conflict flag, so it can
    never reinforce the type it is forbidden for.
    """
    strength = 0.0
    conflict = False
    for source in proposal.sources:
        for name, value in source.features.items():
            if not name.startswith(ONTOLOGY_FEATURE_PREFIX):
                continue
            ontology = name[len(ONTOLOGY_FEATURE_PREFIX):]
            if config.ontology_evidence_allowed(entity_type, ontology):
                strength = max(strength, min(1.0, float(value)))
            elif (float(value) > 0.0
                    and config.forbidden_ontology_evidence.get(entity_type) == ontology):
                conflict = True
    return strength, conflict


def delimiter_shape(
    proposal: LatticeProposal, entity_type: str, original_text: str,
) -> float:
    """Spec §7.2: a colon followed by a number marks TEST_NAME / TEST_RESULT."""
    if entity_type == "TEST_NAME":
        tail = original_text[proposal.end:proposal.end + 8]
        return 1.0 if _COLON_THEN_NUMBER.match(tail) else 0.0
    if entity_type == "TEST_RESULT":
        return 1.0 if _NUMERIC_VALUE.match(proposal.text) else 0.0
    return 0.0


def type_utilities(
    proposal: LatticeProposal, original_text: str, config: ResolverV1Config,
) -> tuple[dict[str, float], dict[str, bool], dict[str, dict[str, float]]]:
    """Weighted utility per candidate type, plus conflicts and the breakdown.

    Only types some expert actually proposed are candidates. The resolver never
    invents a type no expert put forward — a section prior alone is evidence, not
    a proposal (spec §4.2).
    """
    weights = config.type_weights
    utilities: dict[str, float] = {}
    conflicts: dict[str, bool] = {}
    breakdown: dict[str, dict[str, float]] = {}

    for entity_type in sorted(proposal.type_scores):
        ontology_strength, conflict = ontology_evidence(proposal, entity_type, config)
        parts = {
            "neural_span_type": weights.neural_span_type * neural_score(
                proposal, entity_type),
            "grammar_completeness": weights.grammar_completeness * (
                grammar_completeness(proposal) if entity_type == "MEDICATION" else 0.0),
            "laboratory_evidence": weights.laboratory_evidence * laboratory_evidence(
                proposal, entity_type),
            "section_prior": weights.section_prior * config.section_prior(
                proposal.section, entity_type),
            "route_prior": weights.route_prior * config.route_prior(
                proposal.routes, entity_type),
            "expert_agreement": weights.expert_agreement * expert_agreement(
                proposal, entity_type),
            "ontology_evidence": weights.ontology_evidence * ontology_strength,
            "delimiter_shape": weights.delimiter_shape * delimiter_shape(
                proposal, entity_type, original_text),
            "expert_disagreement": -config.expert_disagreement_penalty * expert_disagreement(
                proposal, entity_type),
        }
        utilities[entity_type] = round(sum(parts.values()), 6)
        conflicts[entity_type] = conflict
        breakdown[entity_type] = {k: round(v, 6) for k, v in parts.items()}
    return utilities, conflicts, breakdown


def decide_type(
    proposal: LatticeProposal, original_text: str, config: ResolverV1Config,
) -> TypeDecision:
    """Choose one type, or abstain because the wrong-type risk is too high."""
    utilities, conflicts, breakdown = type_utilities(proposal, original_text, config)
    if not utilities:
        return TypeDecision("", 0.0, 0.0, "", True, "no_candidate_type", {}, {})

    ranked = sorted(utilities.items(), key=lambda kv: (-kv[1], kv[0]))
    best_type, best_utility = ranked[0]
    runner_up, runner_utility = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = round(best_utility - runner_utility, 6)

    if conflicts.get(best_type) and config.abstention.abstain_on_ontology_conflict:
        return TypeDecision(best_type, best_utility, margin, runner_up, True,
                            "type_ontology_conflict", utilities, breakdown[best_type])
    if (config.abstention.abstain_on_flagged_deterministic_evidence
            and flagged_evidence_only(proposal, best_type)):
        return TypeDecision(best_type, best_utility, margin, runner_up, True,
                            "flagged_deterministic_evidence_only",
                            utilities, breakdown[best_type])
    if best_utility < config.abstention.min_type_utility:
        return TypeDecision(best_type, best_utility, margin, runner_up, True,
                            "utility_below_threshold", utilities, breakdown[best_type])
    if runner_up and margin < config.abstention.min_type_margin:
        return TypeDecision(best_type, best_utility, margin, runner_up, True,
                            "type_margin_below_threshold", utilities, breakdown[best_type])
    return TypeDecision(best_type, best_utility, margin, runner_up, False,
                        "accepted", utilities, breakdown[best_type])


__all__ = [
    "ONTOLOGY_FEATURE_PREFIX",
    "TypeDecision",
    "assign_type",
    "decide_type",
    "delimiter_shape",
    "expert_agreement",
    "expert_disagreement",
    "flagged_evidence_only",
    "grammar_completeness",
    "laboratory_evidence",
    "neural_score",
    "ontology_evidence",
    "type_utilities",
]
