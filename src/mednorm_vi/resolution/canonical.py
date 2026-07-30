"""The canonical L4 entry point: one SpanLattice in, one ResolutionResult out.

Before Audit 0052 this repository had **two live L4 implementations** on disjoint
contracts, and the canonical runner used the weaker one:

* ``resolution/resolver.py`` — Phase-1C-A. Consumed ``mention_factory`` proposals
  directly, so the runner could never see a lattice and no neural expert could
  reach L4 at all. Emitted ``EntityHypothesis``, which L5-L9 consume.
* ``resolution/resolver_v1.py`` — the real §7 resolver over a ``SpanLattice``:
  boundary trim/expand, evidence-weighted type utilities with abstention,
  boundary-group selection, near-complete-overlap competition, and ``has_result``
  pair protection. Emitted ``TypedHypothesis``, which L5-L9 do **not** consume — so
  its behaviour was reachable only from the ablation harness.

This module resolves that: it runs ``resolver_v1``'s decision logic — every part of
it, unchanged — and adapts the outcome to the ``ResolutionResult`` /
``EntityHypothesis`` contract the rest of the pipeline already speaks. One entry
point, the better behaviour, the contract that was already wired.

It also carries through what the old boundary destroyed: ``expert_ids`` (which
experts proposed a span) and ``components`` (E1's structured field decomposition,
spec §10.1), so L5 can consume them in Milestone 2 without another migration.

Audit 0055 finished the job: the retired ``resolver.py`` had one unique behaviour —
a configurable per-type boundary policy — and that policy now lives in
``boundary.preference_rank``, which **both** paths used before the old module was
deleted. ``resolution/resolver.py`` no longer exists, and this is the only public L4
entry point in the repository.

Audit 0056a made it a **dispatcher** without adding a second entry point. The
learned slot (``learned_v2``) was reachable only from the trainer and the ablation
harness, so ``enable_l4_learned_v2`` was inert — it changed nothing and raised
nothing. ``resolve_lattice_to_hypotheses`` now takes an optional
``learned_backend``: absent, it runs the deterministic path below exactly as before;
present, it delegates to the learned backend and returns the **same**
``ResolutionResult`` contract. There is no third possibility, and in particular no
path on which an explicitly enabled learned route silently degrades to the
deterministic one — the runner obtains its backend from
``learned_dispatch.build_learned_l4_backend``, which raises rather than returning
``None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..lattice.models import SpanLattice
from ..mention_factory.models import RelationProposal
from ..schemas.constants import ORGANIZER_LABEL_BY_TYPE
from .config_v1 import ResolverV1Config
from .learned_dispatch import LearnedL4Backend
from .models import (
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNRESOLVED,
    BoundaryAlternative,
    BoundaryEvidence,
    EntityHypothesis,
    OverlapDecision,
    ResolutionResult,
    TypeEvidence,
)
from .resolver_v1 import RESOLVER_VERSION, ResolutionDecision, resolve_lattice

CANONICAL_L4_VERSION = "l4-canonical-lattice-v1"

# How resolver_v1's per-node status maps onto the pipeline's three-state contract.
# `abstained` and `suppressed` are both "not emitted", but they are different
# reasons and the distinction is preserved rather than flattened: abstention is a
# type-confidence decision (spec §7.2), suppression is an overlap outcome (§7.3).
_STATUS_MAP: Mapping[str, str] = {
    "accepted": STATUS_ACCEPTED,
    "abstained": STATUS_UNRESOLVED,
    "suppressed": STATUS_REJECTED,
}


def _organizer_label(entity_type: str) -> str:
    """L5-L9 key off the organizer label, so convert once, here.

    ``resolver_v1`` decides in internal English enums; ``EntityHypothesis`` has
    always carried the Vietnamese organizer label (``RESOLVABLE_TYPES``). Doing the
    conversion at this single boundary is what lets both halves keep their own
    vocabulary without either guessing.
    """
    return ORGANIZER_LABEL_BY_TYPE.get(entity_type, entity_type)


def _boundary_alternatives(
    lattice: SpanLattice,
    decision: ResolutionDecision,
) -> tuple[BoundaryAlternative, ...]:
    """Competing coordinates for the same mention, retained for replay (§7.1)."""
    alternatives: list[BoundaryAlternative] = []
    for proposal in lattice.proposals:
        if proposal.start == decision.start and proposal.end == decision.end:
            continue
        if not (proposal.start < decision.end and decision.start < proposal.end):
            continue
        alternatives.append(
            BoundaryAlternative(
                proposal_id=proposal.sources[0].proposal_id if proposal.sources else "",
                start=proposal.start,
                end=proposal.end,
                text=proposal.text,
                kind="lattice_overlap",
            )
        )
    return tuple(sorted(alternatives, key=lambda a: (a.start, a.end, a.proposal_id)))


def resolve_lattice_to_hypotheses(
    lattice: SpanLattice,
    config: ResolverV1Config,
    *,
    relations: Sequence[RelationProposal] = (),
    learned_backend: LearnedL4Backend | None = None,
) -> ResolutionResult:
    """**The canonical L4.** Resolve one lattice into typed entity hypotheses.

    With ``learned_backend`` absent — the shipped configuration — every deterministic
    behaviour of ``resolver_v1`` applies: boundary shaping, type utilities and
    abstention, boundary-group selection, overlap competition and ``has_result``
    protection. Nothing is re-decided here; this function adapts the result and
    preserves provenance.

    With ``learned_backend`` present, the profile explicitly selected the learned
    route and this function delegates to it. It does **not** fall back: a backend
    that fails raises out of here, because an operator who asked for the learned
    resolver and silently received the deterministic one would have no way to tell.
    """
    if learned_backend is not None:
        return learned_backend.resolve(lattice, relations=relations)

    inner = resolve_lattice(lattice, config, relations=relations)

    # Index the lattice by the coordinates each decision started from, so a
    # decision can be traced back to the node whose evidence produced it.
    by_original: dict[tuple[int, int], Any] = {
        (proposal.start, proposal.end): proposal for proposal in lattice.proposals
    }

    hypotheses: list[EntityHypothesis] = []
    for decision in inner.decisions:
        proposal = by_original.get((decision.original_start, decision.original_end))
        sources = tuple(proposal.sources) if proposal is not None else ()
        components = proposal.components() if proposal is not None else ()
        relation_refs = proposal.relation_refs() if proposal is not None else ()

        chosen = sources[0].proposal_id if sources else ""
        source_specialist = ""
        for source in sources:
            # Report the deterministic specialist when one contributed, because
            # that is what downstream type evidence has always meant here.
            if source.expert_id.startswith(("E1", "E2")):
                source_specialist = source.expert_id
                break

        overlap: OverlapDecision | None = None
        if decision.status == "suppressed":
            overlap = OverlapDecision(
                outcome="suppressed", counterpart_id=decision.suppressed_by, reason=decision.reason
            )
        elif decision.suppressed_by:
            overlap = OverlapDecision(
                outcome="winner", counterpart_id=decision.suppressed_by, reason=decision.reason
            )

        features = {
            "utility": float(decision.utility),
            "margin": float(decision.margin),
            "expert_count": float(len(decision.expert_ids)),
            "boundary_action_count": float(len(decision.boundary_actions)),
            "overlap_iou": float(decision.overlap_iou),
            "component_count": float(len(components)),
            "source_count": float(len(sources)),
        }
        for name, value in decision.contributions.items():
            features[f"contribution_{name}"] = float(value)

        hypotheses.append(
            EntityHypothesis(
                hypothesis_id=decision.hypothesis_id,
                document_id=lattice.document_id,
                start=decision.start,
                end=decision.end,
                text=lattice.original_text[decision.start : decision.end],
                entity_type=_organizer_label(decision.entity_type),
                status=_STATUS_MAP.get(decision.status, STATUS_UNRESOLVED),
                chosen_proposal_id=chosen,
                source_proposal_ids=tuple(source.proposal_id for source in sources),
                boundary_evidence=BoundaryEvidence(
                    # The migrated boundary policy (Audit 0055) is the policy field when
                    # one applied, so a hypothesis records WHICH ladder rung selected it
                    # rather than only that the canonical L4 ran.
                    policy=decision.boundary_policy or CANONICAL_L4_VERSION,
                    chosen_kind="|".join(decision.boundary_actions) or "unshaped",
                    chosen_proposal_id=chosen,
                    considered_kinds=tuple(decision.boundary_actions),
                    note=(
                        f"original=[{decision.original_start},{decision.original_end})"
                        f" l4={CANONICAL_L4_VERSION}"
                    ),
                ),
                type_evidence=TypeEvidence(
                    entity_type=_organizer_label(decision.entity_type),
                    source_specialist=source_specialist,
                    proposed_types=tuple(
                        _organizer_label(name) for name in sorted(decision.utilities)
                    ),
                    note=decision.reason,
                ),
                retained_alternatives=_boundary_alternatives(lattice, decision),
                overlap_decision=overlap,
                rejection_reason=(decision.reason if decision.status != "accepted" else None),
                has_result_pair_group_ids=relation_refs,
                score=float(decision.utility),
                features=features,
                expert_ids=tuple(decision.expert_ids),
                components=components,
            )
        )

    return ResolutionResult(
        document_id=lattice.document_id,
        hypotheses=tuple(hypotheses),
        config_version=f"{CANONICAL_L4_VERSION}+{RESOLVER_VERSION}",
        warnings=tuple(inner.warnings) + tuple(lattice.warnings),
    )


def resolution_counts(result: ResolutionResult) -> dict[str, int]:
    """Accepted / rejected / unresolved counts, for the run manifest."""
    return {
        "accepted": len(result.accepted()),
        "rejected": len(result.rejected()),
        "unresolved": len(result.unresolved()),
        "total": len(result.hypotheses),
    }


__all__ = [
    "CANONICAL_L4_VERSION",
    "resolution_counts",
    "resolve_lattice_to_hypotheses",
]
