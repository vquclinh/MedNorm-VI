"""L4 Boundary & Type Resolver v1 (spec §7).

Consumes the unified L3 span lattice and emits ``TypedHypothesis`` objects — the
contract in ``schemas.hypotheses`` — never organizer JSON, never final entities
with candidates or assertions.

The pipeline for one document::

    lattice nodes
      -> boundary shaping   (bounded trim; expand only onto a competing boundary)
      -> type decision      (weighted evidence; abstain on wrong-type risk)
      -> overlap resolution (near-complete same-type competition; pairs protected)
      -> TypedHypothesis

Everything is driven by the single tracked config
``configs/resolution/boundary_type_resolver_v1.yaml``, whose SHA-256 travels with
every report.

**No learned boundary-offset head exists.** Spec §7.1 describes one; it was not
built and not trained in this milestone, and no claim is made that it was.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..lattice.models import SpanLattice
from ..lattice.models import SpanProposal as LatticeProposal
from ..mention_factory.models import HAS_RESULT, RelationProposal
from ..schemas.hypotheses import TypedHypothesis
from ..schemas.spans import Span, SpanCoordinates
from .boundary import expand_to_competitor, trim_span
from .config_v1 import ResolverV1Config
from .features import boundary_kind_of_rule
from .overlap import OverlapCandidate, resolve_near_complete_overlaps
from .typing import TypeDecision, decide_type, grammar_completeness

RESOLVER_VERSION = "l4-boundary-type-resolver-v1"

# Recorded on every report so a reader can never assume a trained head produced
# these boundaries.
LEARNED_BOUNDARY_OFFSET_HEAD_TRAINED = False


@dataclass(frozen=True, slots=True)
class ResolutionDecision:
    """The complete, replayable record of what the resolver did to one node."""

    hypothesis_id: str
    original_start: int
    original_end: int
    start: int
    end: int
    entity_type: str
    utility: float
    margin: float
    runner_up: str
    status: str  # accepted | abstained | suppressed
    reason: str
    boundary_actions: tuple[str, ...] = field(default_factory=tuple)
    expert_ids: tuple[str, ...] = field(default_factory=tuple)
    routes: tuple[str, ...] = field(default_factory=tuple)
    section: str = ""
    suppressed_by: str = ""
    overlap_iou: float = 0.0
    utilities: Mapping[str, float] = field(default_factory=dict)
    contributions: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "original_span": [self.original_start, self.original_end],
            "span": [self.start, self.end],
            "entity_type": self.entity_type,
            "utility": round(self.utility, 6),
            "margin": round(self.margin, 6),
            "runner_up": self.runner_up,
            "status": self.status,
            "reason": self.reason,
            "boundary_actions": list(self.boundary_actions),
            "expert_ids": list(self.expert_ids),
            "routes": list(self.routes),
            "section": self.section,
            "suppressed_by": self.suppressed_by,
            "overlap_iou": round(self.overlap_iou, 6),
            "utilities": {k: round(v, 6) for k, v in sorted(self.utilities.items())},
            "contributions": {k: round(v, 6) for k, v in sorted(self.contributions.items())},
        }


@dataclass(frozen=True, slots=True)
class ResolverV1Result:
    """Everything the L4 resolver v1 produced for one document."""

    document_id: str
    hypotheses: tuple[TypedHypothesis, ...]
    decisions: tuple[ResolutionDecision, ...]
    config_sha256: str = ""
    config_version: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def accepted(self) -> tuple[TypedHypothesis, ...]:
        return tuple(h for h in self.hypotheses if not h.abstained)

    def determinism_hash(self) -> str:
        payload = json.dumps(
            [decision.as_dict() for decision in self.decisions],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def counts(self) -> dict[str, int]:
        out = {"accepted": 0, "abstained": 0, "suppressed": 0, "boundary_changed": 0}
        for decision in self.decisions:
            out[decision.status] = out.get(decision.status, 0) + 1
            if decision.boundary_actions and decision.status == "accepted":
                out["boundary_changed"] += 1
        return out


def _competitors(
    proposal: LatticeProposal, lattice: SpanLattice,
) -> tuple[tuple[int, int, float], ...]:
    """Competing boundaries this node could legally expand onto.

    Only boundaries that some expert already proposed, and only when the expert
    that proposed them had enough grammar completeness to justify a wider span.
    """
    out: list[tuple[int, int, float]] = []
    for other in lattice.proposals:
        if other.coordinates == proposal.coordinates:
            continue
        if other.start <= proposal.start and other.end >= proposal.end:
            out.append((other.start, other.end, grammar_completeness(other)))
    return tuple(out)


def _boundary_groups(proposal: LatticeProposal) -> tuple[tuple[str, str], ...]:
    """``(boundary_group_id, boundary_kind)`` pairs this node belongs to.

    Boundary ALTERNATIVES of one logical mention share a group id: E1's
    name_only→full ladder, E2's value_only/value+unit pair. They are competing
    boundaries for the same mention, not separate mentions.
    """
    return tuple(sorted({
        (source.boundary_group_id, boundary_kind_of_rule(source.matched_rule))
        for source in proposal.sources if source.boundary_group_id
    }))


def _preferred_kind(entity_type: str, config: ResolverV1Config) -> str:
    if entity_type == "MEDICATION":
        return config.boundary.group_preference.get("medication", "")
    if entity_type == "TEST_RESULT":
        return config.boundary.group_preference.get("test_result", "")
    return ""


def _select_within_boundary_groups(
    staged: Sequence[tuple[str, LatticeProposal, TypeDecision, int, int, tuple[str, ...]]],
    config: ResolverV1Config,
) -> dict[str, str]:
    """Pick one alternative per boundary group. Returns ``{loser id: winner id}``.

    Preference order: the configured boundary kind for that type, then the higher
    type utility, then the narrower span, then the id — fully deterministic.
    """
    members: dict[str, list[tuple[str, str, float, int]]] = {}
    for hypothesis_id, proposal, decision, start, end, _actions in staged:
        if decision.abstained:
            continue
        for group_id, kind in _boundary_groups(proposal):
            members.setdefault(group_id, []).append(
                (hypothesis_id, kind, decision.utility, end - start))

    losers: dict[str, str] = {}
    for group in members.values():
        if len(group) < 2:
            continue
        entity_types = {
            decision.entity_type
            for hypothesis_id, _p, decision, _s, _e, _a in staged
            if hypothesis_id in {m[0] for m in group}
        }
        preferred = _preferred_kind(next(iter(entity_types)) if entity_types else "", config)
        ranked = sorted(
            group,
            key=lambda m: (0 if m[1] == preferred else 1, -m[2], m[3], m[0]))
        winner = ranked[0][0]
        for member in ranked[1:]:
            losers[member[0]] = winner
    return losers


def _protected_partners(
    relations: Sequence[RelationProposal],
    proposal_ids_by_coordinates: Mapping[str, str],
    strong_score: float,
) -> dict[str, set[str]]:
    """Map each hypothesis id to the partners a strong has_result edge protects."""
    partners: dict[str, set[str]] = {}
    for relation in relations:
        if relation.relation_type != HAS_RESULT or relation.score < strong_score:
            continue
        left = proposal_ids_by_coordinates.get(relation.source_proposal_id)
        right = proposal_ids_by_coordinates.get(relation.target_proposal_id)
        if left is None or right is None:
            continue
        partners.setdefault(left, set()).add(right)
        partners.setdefault(right, set()).add(left)
    return partners


def resolve_lattice(
    lattice: SpanLattice,
    config: ResolverV1Config,
    *,
    relations: Sequence[RelationProposal] = (),
) -> ResolverV1Result:
    """Resolve one document's span lattice into typed hypotheses."""
    text = lattice.original_text
    warnings: list[str] = []
    decisions: list[ResolutionDecision] = []
    staged: list[tuple[str, LatticeProposal, TypeDecision, int, int, tuple[str, ...]]] = []
    hypothesis_by_source: dict[str, str] = {}

    for index, proposal in enumerate(lattice.proposals, start=1):
        hypothesis_id = f"hyp-{lattice.document_id}-{index:04d}"
        for source in proposal.sources:
            hypothesis_by_source[source.proposal_id] = hypothesis_id

        start, end = proposal.start, proposal.end
        actions: list[str] = []
        if config.boundary.enable_trim:
            shaped = trim_span(
                text, start, end,
                trim_characters=config.boundary.trim_characters,
                max_trim_chars=config.boundary.max_trim_chars,
                trim_leading_list_markers=config.boundary.trim_leading_list_markers)
            start, end = shaped.start, shaped.end
            actions.extend(shaped.actions)
        if config.boundary.enable_expand:
            expanded = expand_to_competitor(
                start, end, _competitors(proposal, lattice),
                min_grammar_completeness=config.boundary.expand_requires_grammar_completeness,
                max_expand_chars=config.boundary.max_expand_chars)
            start, end = expanded.start, expanded.end
            actions.extend(expanded.actions)
        # Shaping may only move edges inside the text; the §4 invariant is
        # re-checked here because a resolver that quietly emits a bad span is
        # worse than one that stops.
        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"boundary shaping produced an invalid span [{start}, {end}) "
                f"for a text of length {len(text)}")

        decision = decide_type(proposal, text, config)
        staged.append((hypothesis_id, proposal, decision, start, end, tuple(actions)))

    partners = _protected_partners(
        relations, hypothesis_by_source, config.overlap.strong_has_result_score)

    # One logical mention contributes one hypothesis: competing boundaries of the
    # same mention are resolved before the global overlap competition runs.
    group_losers = _select_within_boundary_groups(staged, config)

    candidates = [
        OverlapCandidate(
            candidate_id=hypothesis_id, start=start, end=end,
            entity_type=decision.entity_type, utility=decision.utility,
            protected_partners=frozenset(partners.get(hypothesis_id, set())))
        for hypothesis_id, _proposal, decision, start, end, _actions in staged
        if not decision.abstained and hypothesis_id not in group_losers
    ]
    outcome = resolve_near_complete_overlaps(
        candidates,
        near_complete_iou=config.overlap.near_complete_iou,
        competition_penalty=config.overlap.competition_penalty,
        suppress_cross_type=config.overlap.suppress_cross_type,
        protect_pairs=config.overlap.protect_has_result_pairs)
    suppressed = {loser: (winner, iou) for loser, winner, iou in outcome.suppressed}

    hypotheses: list[TypedHypothesis] = []
    for hypothesis_id, proposal, decision, start, end, shaping in staged:
        status = "accepted"
        reason = decision.reason
        suppressed_by, overlap_iou = "", 0.0
        if decision.abstained:
            status = "abstained"
        elif hypothesis_id in group_losers:
            status = "suppressed"
            suppressed_by = group_losers[hypothesis_id]
            reason = "boundary_alternative_not_selected"
        elif hypothesis_id in suppressed:
            status = "suppressed"
            suppressed_by, overlap_iou = suppressed[hypothesis_id]
            reason = "near_complete_overlap"
        decisions.append(ResolutionDecision(
            hypothesis_id=hypothesis_id,
            original_start=proposal.start, original_end=proposal.end,
            start=start, end=end, entity_type=decision.entity_type,
            utility=decision.utility, margin=decision.margin,
            runner_up=decision.runner_up, status=status, reason=reason,
            boundary_actions=shaping, expert_ids=proposal.expert_ids,
            routes=proposal.routes, section=proposal.section,
            suppressed_by=suppressed_by, overlap_iou=overlap_iou,
            utilities=dict(decision.utilities),
            contributions=dict(decision.contributions)))
        if status != "accepted":
            continue
        hypotheses.append(TypedHypothesis(
            hypothesis_id=hypothesis_id,
            text=text[start:end],
            coords=SpanCoordinates(absolute=Span(start, end)),
            type_distribution=dict(decision.utilities),
            calibrated_score=decision.utility,
            abstained=False,
            source_proposal_ids=tuple(source.proposal_id for source in proposal.sources),
            evidence_ids=tuple(
                eid for source in proposal.sources for eid in source.evidence_ids),
            features={
                "utility": decision.utility,
                "margin": decision.margin,
                "expert_count": float(len(proposal.expert_ids)),
                "boundary_actions": float(len(shaping)),
            }))

    # Two accepted hypotheses may not occupy identical coordinates with identical
    # types: that would be a text-level duplicate, which the organizer rejects.
    seen: set[tuple[int, int, str]] = set()
    for record in decisions:
        if record.status != "accepted":
            continue
        key = (record.start, record.end, record.entity_type)
        if key in seen:
            warnings.append(f"duplicate_accepted_span:{key[0]}:{key[1]}:{key[2]}")
        seen.add(key)

    return ResolverV1Result(
        document_id=lattice.document_id, hypotheses=tuple(hypotheses),
        decisions=tuple(decisions), config_sha256=config.config_sha256,
        config_version=config.config_version, warnings=tuple(warnings))


__all__ = [
    "LEARNED_BOUNDARY_OFFSET_HEAD_TRAINED",
    "RESOLVER_VERSION",
    "ResolutionDecision",
    "ResolverV1Result",
    "resolve_lattice",
]
