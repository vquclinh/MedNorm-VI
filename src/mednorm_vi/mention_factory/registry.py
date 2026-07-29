"""L3 expert registry: how an expert reaches the canonical runner (spec §6).

Before Audit 0052 the canonical runner hard-coded E1 and E2 and had no path for any
other expert, so E3 — which has a trained, validated checkpoint — could not run in
production at all, and adding E4/E5/E6/E7 would have meant editing
``inference/pipeline.py`` once per expert. This registry is the seam that removes
that: the runner asks the registry which experts are enabled and ready, and every
expert answers through one contract.

Three rules the registry enforces, all from spec §6:

* **no expert emits a final entity.** Every expert returns
  :class:`~mednorm_vi.lattice.models.ExpertSpanProposal` — proposals that must pass
  through the lattice and L4. The protocol has no way to return an
  ``EntityHypothesis`` or organizer JSON;
* **a disabled expert loads nothing.** ``prepare()`` is called only for enabled
  experts, and model construction belongs inside it, so a profile that turns an
  expert off costs no memory and touches no checkpoint;
* **an enabled expert that is not ready fails closed, by name.** It never degrades
  to "no proposals", because an expert silently contributing nothing is
  indistinguishable from an expert that genuinely found nothing.

Registration is by ``expert_id`` from ``lattice.models.AVAILABLE_EXPERTS``, so a
retired expert (E4) cannot be registered even by mistake: its id is not in that
tuple and :func:`register` refuses it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..case_router.models import NodeRouting
from ..document_intelligence.models import DocumentGraph
from ..lattice.models import AVAILABLE_EXPERTS, RETIRED_EXPERTS, ExpertSpanProposal

REGISTRY_CONTRACT_VERSION = "l3-expert-registry-v1"


class ExpertRegistryError(RuntimeError):
    """Raised when an expert cannot be registered, prepared or run."""


class ExpertNotReady(ExpertRegistryError):
    """Raised when an ENABLED expert cannot run.

    Deliberately loud and specific: it names the expert, the role and the exact
    missing path. Returning zero proposals instead would report a broken
    environment as a measurement.
    """

    def __init__(self, expert_id: str, role: str, reason: str, path: str = "") -> None:
        self.expert_id = expert_id
        self.role = role
        self.reason = reason
        self.path = path
        detail = f" (path: {path})" if path else ""
        super().__init__(
            f"L3 expert {expert_id} [{role}] is enabled but not ready: {reason}{detail}")


@runtime_checkable
class L3Expert(Protocol):
    """What every L3 expert must provide.

    ``propose`` returns proposals only. There is no method on this protocol that
    can emit a resolved entity, which is how spec §6's "no individual module is
    allowed to emit a final entity directly" is enforced structurally rather than
    by convention.
    """

    expert_id: str
    role: str

    def readiness(self) -> tuple[bool, str, str]:
        """``(ready, reason, path)``. Must not load a model."""
        ...

    def prepare(self) -> None:
        """Lazily construct whatever inference needs. Called only when enabled."""
        ...

    def propose(
        self,
        graph: DocumentGraph,
        routings: Sequence[NodeRouting],
    ) -> Sequence[ExpertSpanProposal]:
        """Proposals for one document, in absolute end-exclusive coordinates."""
        ...


@dataclass(frozen=True, slots=True)
class ExpertRegistration:
    """One registered expert: its id, the flag that enables it, and its factory."""

    expert_id: str
    feature_flag: str
    role: str
    factory: Callable[[Mapping[str, Any]], L3Expert]
    checkpoint_key: str = ""

    def __post_init__(self) -> None:
        if self.expert_id in RETIRED_EXPERTS:
            raise ExpertRegistryError(
                f"{self.expert_id} is retired from the active architecture and must "
                "not be registered")
        if self.expert_id not in AVAILABLE_EXPERTS:
            raise ExpertRegistryError(
                f"unknown expert id {self.expert_id!r}; expected one of "
                f"{list(AVAILABLE_EXPERTS)}")
        if not self.feature_flag:
            raise ExpertRegistryError(f"{self.expert_id}: a feature flag is required")


@dataclass
class ExpertRegistry:
    """The experts the runner may consult, in deterministic order."""

    registrations: dict[str, ExpertRegistration] = field(default_factory=dict)

    def register(self, registration: ExpertRegistration) -> ExpertRegistration:
        if registration.expert_id in self.registrations:
            raise ExpertRegistryError(
                f"{registration.expert_id} is already registered")
        self.registrations[registration.expert_id] = registration
        return registration

    def unregister(self, expert_id: str) -> None:
        self.registrations.pop(expert_id, None)

    def __contains__(self, expert_id: object) -> bool:
        return expert_id in self.registrations

    def ordered(self) -> tuple[ExpertRegistration, ...]:
        """Registrations in architecture order, so a lattice is reproducible."""
        order = {expert: index for index, expert in enumerate(AVAILABLE_EXPERTS)}
        return tuple(sorted(
            self.registrations.values(),
            key=lambda r: (order.get(r.expert_id, len(order)), r.expert_id)))

    def enabled(
        self, feature_flags: Mapping[str, Any]
    ) -> tuple[ExpertRegistration, ...]:
        return tuple(
            registration for registration in self.ordered()
            if bool(feature_flags.get(registration.feature_flag, False)))


# The process-wide default registry. Experts register into it at import time of
# their own module, so adding an expert never touches the runner.
DEFAULT_REGISTRY = ExpertRegistry()


def register(registration: ExpertRegistration) -> ExpertRegistration:
    """Register an expert in the default registry."""
    return DEFAULT_REGISTRY.register(registration)


@dataclass(frozen=True, slots=True)
class ExpertRunRecord:
    """What one expert actually did on one document. Never a guess."""

    expert_id: str
    role: str
    enabled: bool
    ready: bool
    executed: bool
    proposal_count: int
    reason: str = ""
    path: str = ""
    model_revision: str = ""
    checkpoint_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "role": self.role,
            "enabled": self.enabled,
            "ready": self.ready,
            "executed": self.executed,
            "proposal_count": self.proposal_count,
            "reason": self.reason,
            "path": self.path,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def run_registered_experts(
    graph: DocumentGraph,
    routings: Sequence[NodeRouting],
    *,
    feature_flags: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    registry: ExpertRegistry | None = None,
    prepared: dict[str, L3Expert] | None = None,
) -> tuple[tuple[ExpertSpanProposal, ...], tuple[ExpertRunRecord, ...]]:
    """Run every ENABLED registered expert. Returns proposals and a run record.

    Raises :class:`ExpertNotReady` for an enabled-but-unready expert rather than
    skipping it, so a profile can never quietly become a weaker profile.

    ``prepared`` is an optional cache of already-constructed experts, keyed by
    ``expert_id``; the runner passes one so a multi-document run loads each model
    once instead of once per document.
    """
    active = registry if registry is not None else DEFAULT_REGISTRY
    cache = prepared if prepared is not None else {}
    proposals: list[ExpertSpanProposal] = []
    records: list[ExpertRunRecord] = []

    for registration in active.ordered():
        enabled = bool(feature_flags.get(registration.feature_flag, False))
        if not enabled:
            # A disabled expert is never constructed, so no model is loaded.
            records.append(ExpertRunRecord(
                expert_id=registration.expert_id, role=registration.role,
                enabled=False, ready=False, executed=False, proposal_count=0,
                reason="disabled_by_profile"))
            continue

        expert = cache.get(registration.expert_id)
        if expert is None:
            expert = registration.factory(dict(settings or {}))
            ready, reason, path = expert.readiness()
            if not ready:
                raise ExpertNotReady(
                    registration.expert_id, registration.role, reason, path)
            expert.prepare()
            cache[registration.expert_id] = expert
        else:
            ready, reason, path = expert.readiness()
            if not ready:
                raise ExpertNotReady(
                    registration.expert_id, registration.role, reason, path)

        produced = tuple(expert.propose(graph, routings))
        for proposal in produced:
            if proposal.expert_id != registration.expert_id:
                raise ExpertRegistryError(
                    f"{registration.expert_id} emitted a proposal labelled "
                    f"{proposal.expert_id!r}; an expert may not impersonate another")
            # Spec §4: verified here as well as in the lattice, because a wrong
            # offset must be attributable to the expert that produced it.
            proposal.validate_against(graph.original_text)
        proposals.extend(produced)
        records.append(ExpertRunRecord(
            expert_id=registration.expert_id, role=registration.role,
            enabled=True, ready=True, executed=True, proposal_count=len(produced),
            model_revision=getattr(expert, "model_revision", "") or "",
            checkpoint_sha256=getattr(expert, "checkpoint_sha256", "") or ""))

    return tuple(proposals), tuple(records)


__all__ = [
    "DEFAULT_REGISTRY",
    "REGISTRY_CONTRACT_VERSION",
    "ExpertNotReady",
    "ExpertRegistration",
    "ExpertRegistry",
    "ExpertRegistryError",
    "ExpertRunRecord",
    "L3Expert",
    "register",
    "run_registered_experts",
]
