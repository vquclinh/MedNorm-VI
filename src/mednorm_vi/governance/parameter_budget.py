"""Candidate vs deployment parameter accounting and the 9B gate (Audit 0042).

Spec §17 fixes one hard constraint: the total parameters across base models in the
**submitted** system must not exceed 9B, and it says explicitly that counts "are
approximate and must be verified by an automated parameter-count script before
submission". This module is that script's engine.

Two inventories are kept strictly apart, because conflating them would either
block legitimate research or under-count a submission:

**candidate inventory**
    every model trained or evaluated during research. Its total *may* exceed 9B —
    training seven alternatives is allowed as long as they are not all deployed
    together. The gate is never applied here.

**deployment inventory**
    only the models a single final inference pipeline loads together. The 9B gate
    applies here, and only here.

Two accounting conventions are reported side by side, and the difference is not
hidden:

``base_only_parameters``
    spec §17's convention — "Heads/adapters are outside the budget according to
    organizer confirmation".
``total_loaded_parameters``
    the **conservative** convention this project gates on: everything the process
    actually loads, adapters included. A LoRA deployment counts the base model
    *plus* its adapters.

The conservative total is always >= the base-only total, so passing the
conservative gate necessarily satisfies §17. The gate uses the conservative
number; the §17 number is reported for traceability.

Nothing here downloads or instantiates a model on import. Counting from a real
configuration is done only when a caller supplies a factory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PARAMETER_REGISTRY_VERSION = "candidate-model-registry-v1"
DEPLOYMENT_BUDGET_VERSION = "deployment-budget-v1"

# Spec §17: "the total number of parameters across all base models in the system
# must not exceed 9B".
MAX_DEPLOYMENT_PARAMETERS = 9_000_000_000

# Lifecycle of a registry entry.
STATUS_PLANNED = "PLANNED"            # architecture-declared, not implemented yet
STATUS_IMPLEMENTED = "IMPLEMENTED"    # code exists, no trained checkpoint
STATUS_TRAINED = "TRAINED"            # a validated checkpoint exists
STATUS_EXCLUDED_BY_ABLATION = "EXCLUDED_BY_ABLATION"  # researched, not deployed
# Audit 0048: researched to completion, then withdrawn from the active stack by
# an owner decision. Distinct from EXCLUDED_BY_ABLATION, which is a measurement
# outcome — retirement is a decision recorded on top of one.
STATUS_RETIRED_FROM_ACTIVE_STACK = "RETIRED_FROM_ACTIVE_STACK"

REGISTRY_STATUSES: tuple[str, ...] = (
    STATUS_PLANNED, STATUS_IMPLEMENTED, STATUS_TRAINED,
    STATUS_EXCLUDED_BY_ABLATION, STATUS_RETIRED_FROM_ACTIVE_STACK)

# Statuses a deployment manifest may never select.
NON_DEPLOYABLE_STATUSES: tuple[str, ...] = (
    STATUS_PLANNED, STATUS_EXCLUDED_BY_ABLATION, STATUS_RETIRED_FROM_ACTIVE_STACK)

# How a parameter count was obtained. Only COUNTED_FROM_CONFIG is trustworthy for
# a deployment decision.
METHOD_COUNTED_FROM_CONFIG = "counted_from_config"
METHOD_COUNTED_FROM_CHECKPOINT = "counted_from_checkpoint"
METHOD_PUBLISHED_ESTIMATE = "published_estimate"
METHOD_UNKNOWN = "unknown"

VERIFIED_METHODS: frozenset[str] = frozenset(
    {METHOD_COUNTED_FROM_CONFIG, METHOD_COUNTED_FROM_CHECKPOINT})


class ParameterBudgetError(ValueError):
    """Raised when a registry entry or deployment manifest is not usable."""


class DeploymentBudgetExceeded(ParameterBudgetError):
    """Raised when a selected deployment exceeds the 9B limit."""


class UnverifiedDeploymentComponent(ParameterBudgetError):
    """Raised when a selected component has no verified parameter count."""


@dataclass(frozen=True, slots=True)
class CandidateModel:
    """One researched component. ``PLANNED`` entries carry no invented counts."""

    component_id: str
    architecture_layer: str
    training_stage: str
    status: str
    model_id: str = ""
    pinned_revision: str = ""
    checkpoint_role: str = ""
    total_parameters: int | None = None
    trainable_parameters: int | None = None
    adapter_parameters: int = 0
    loaded_at_inference: bool = False
    shares_weights_with: str = ""
    parameter_count_method: str = METHOD_UNKNOWN
    parameter_count_verified: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ParameterBudgetError("component_id is required")
        if self.status not in REGISTRY_STATUSES:
            raise ParameterBudgetError(
                f"{self.component_id}: unknown status {self.status!r}")
        if self.total_parameters is not None and self.total_parameters < 0:
            raise ParameterBudgetError(f"{self.component_id}: negative total_parameters")
        if self.adapter_parameters < 0:
            raise ParameterBudgetError(f"{self.component_id}: negative adapter_parameters")
        if self.parameter_count_verified and self.total_parameters is None:
            raise ParameterBudgetError(
                f"{self.component_id}: cannot be verified without a total_parameters count")
        if self.parameter_count_verified and self.parameter_count_method not in VERIFIED_METHODS:
            raise ParameterBudgetError(
                f"{self.component_id}: method {self.parameter_count_method!r} cannot be "
                "marked verified")
        if self.status == STATUS_PLANNED and self.total_parameters is not None:
            raise ParameterBudgetError(
                f"{self.component_id}: a PLANNED entry must not carry a guessed count")

    @property
    def has_verified_count(self) -> bool:
        return bool(self.parameter_count_verified and self.total_parameters is not None)

    def deployment_parameters(self) -> int:
        """Parameters this component contributes when loaded, adapters included.

        A LoRA deployment is base + adapter: claiming only the trainable adapter
        would under-count what the process actually loads.
        """
        if self.total_parameters is None:
            raise UnverifiedDeploymentComponent(
                f"{self.component_id}: no parameter count available")
        return int(self.total_parameters) + int(self.adapter_parameters)

    def base_only_parameters(self) -> int:
        """Spec §17's convention: base weights, adapters excluded."""
        if self.total_parameters is None:
            raise UnverifiedDeploymentComponent(
                f"{self.component_id}: no parameter count available")
        return int(self.total_parameters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "architecture_layer": self.architecture_layer,
            "training_stage": self.training_stage,
            "model_id": self.model_id,
            "pinned_revision": self.pinned_revision,
            "checkpoint_role": self.checkpoint_role,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "adapter_parameters": self.adapter_parameters,
            "loaded_at_inference": self.loaded_at_inference,
            "shares_weights_with": self.shares_weights_with,
            "parameter_count_method": self.parameter_count_method,
            "parameter_count_verified": self.parameter_count_verified,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CandidateRegistry:
    """The complete research inventory. The 9B gate is NEVER applied to it."""

    components: tuple[CandidateModel, ...]
    version: str = PARAMETER_REGISTRY_VERSION

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for component in self.components:
            if component.component_id in seen:
                raise ParameterBudgetError(
                    f"duplicate component_id {component.component_id!r}")
            seen.add(component.component_id)

    def by_id(self, component_id: str) -> CandidateModel:
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise ParameterBudgetError(f"unknown component_id {component_id!r}")

    def with_status(self, status: str) -> tuple[CandidateModel, ...]:
        return tuple(c for c in self.components if c.status == status)

    def candidate_total_parameters(self) -> int:
        """Sum of every KNOWN candidate count. May legitimately exceed 9B."""
        return sum(
            int(c.total_parameters) + int(c.adapter_parameters)
            for c in self.components if c.total_parameters is not None)

    def summary(self) -> dict[str, Any]:
        known = [c for c in self.components if c.total_parameters is not None]
        return {
            "registry_version": self.version,
            "component_count": len(self.components),
            "components_with_counts": len(known),
            "verified_components": sum(1 for c in self.components if c.has_verified_count),
            "planned_components": len(self.with_status(STATUS_PLANNED)),
            "candidate_total_parameters": self.candidate_total_parameters(),
            "candidate_total_exceeds_9b": (
                self.candidate_total_parameters() > MAX_DEPLOYMENT_PARAMETERS),
            "gate_applies_to_candidates": False,
            "max_deployment_parameters": MAX_DEPLOYMENT_PARAMETERS,
        }


@dataclass(frozen=True, slots=True)
class DeploymentComponent:
    """One entry in a deployment budget report."""

    component_id: str
    counted_parameters: int
    base_only_parameters: int
    adapter_parameters: int
    shared_with: str
    counted_once: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "counted_parameters": self.counted_parameters,
            "base_only_parameters": self.base_only_parameters,
            "adapter_parameters": self.adapter_parameters,
            "shared_with": self.shared_with,
            "counted_once": self.counted_once,
        }


@dataclass(frozen=True, slots=True)
class DeploymentBudgetReport:
    manifest_name: str
    components: tuple[DeploymentComponent, ...]
    total_loaded_parameters: int
    base_only_parameters: int
    adapter_parameters: int
    within_budget: bool
    remaining_margin: int
    version: str = DEPLOYMENT_BUDGET_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "deployment_budget_version": self.version,
            "manifest_name": self.manifest_name,
            "max_deployment_parameters": MAX_DEPLOYMENT_PARAMETERS,
            "total_loaded_parameters": self.total_loaded_parameters,
            "base_only_parameters_spec_section_17": self.base_only_parameters,
            "adapter_parameters": self.adapter_parameters,
            "within_budget": self.within_budget,
            "remaining_margin": self.remaining_margin,
            "components": [component.as_dict() for component in self.components],
            "accounting_policy": (
                "conservative: total parameters loaded by the final inference "
                "system, adapters included; a shared base loaded once is counted "
                "once"),
        }


def compute_deployment_budget(
    registry: CandidateRegistry,
    selected: Sequence[str],
    *,
    manifest_name: str = "deployment",
    enforce: bool = True,
) -> DeploymentBudgetReport:
    """Total what a deployment actually loads, and gate it at 9B.

    Fails **closed**: a selected component without a verified count raises rather
    than being optimistically treated as zero or as its published estimate.

    Shared weights are counted once. Two components that declare
    ``shares_weights_with`` pointing at the same base contribute that base a single
    time; anything loaded independently is counted in full.
    """
    if not selected:
        raise ParameterBudgetError("a deployment manifest must select at least one component")

    entries: list[DeploymentComponent] = []
    counted_bases: set[str] = set()
    total = 0
    base_total = 0
    adapter_total = 0

    for component_id in selected:
        component = registry.by_id(component_id)
        if not component.has_verified_count:
            raise UnverifiedDeploymentComponent(
                f"{component_id}: selected for deployment without a verified parameter "
                f"count (method={component.parameter_count_method!r}, "
                f"status={component.status}); the budget fails closed rather than "
                "guessing")

        # Shared-weight alias: the base is loaded once no matter how many heads
        # or adapters sit on top of it.
        share_key = component.shares_weights_with
        counted_once = True
        if share_key:
            if share_key in counted_bases:
                # The base is already counted; only this component's adapter adds.
                contribution = int(component.adapter_parameters)
                base_contribution = 0
                counted_once = False
            else:
                counted_bases.add(share_key)
                contribution = component.deployment_parameters()
                base_contribution = component.base_only_parameters()
        else:
            contribution = component.deployment_parameters()
            base_contribution = component.base_only_parameters()

        total += contribution
        base_total += base_contribution
        adapter_total += int(component.adapter_parameters)
        entries.append(DeploymentComponent(
            component_id=component_id, counted_parameters=contribution,
            base_only_parameters=base_contribution,
            adapter_parameters=int(component.adapter_parameters),
            shared_with=share_key, counted_once=counted_once))

    within = total <= MAX_DEPLOYMENT_PARAMETERS
    report = DeploymentBudgetReport(
        manifest_name=manifest_name, components=tuple(entries),
        total_loaded_parameters=total, base_only_parameters=base_total,
        adapter_parameters=adapter_total, within_budget=within,
        remaining_margin=MAX_DEPLOYMENT_PARAMETERS - total)
    if enforce and not within:
        raise DeploymentBudgetExceeded(
            f"{manifest_name}: deployment loads {total:,} parameters, which exceeds the "
            f"{MAX_DEPLOYMENT_PARAMETERS:,} limit by "
            f"{total - MAX_DEPLOYMENT_PARAMETERS:,}")
    return report


# ---------------------------------------------------------------------------
# Programmatic counting
# ---------------------------------------------------------------------------


def count_parameters(model: Any) -> tuple[int, int]:
    """``(total, trainable)`` for a Torch module. Never an estimate."""
    parameters = list(model.parameters())
    total = sum(int(p.numel()) for p in parameters)
    trainable = sum(int(p.numel()) for p in parameters if bool(p.requires_grad))
    return total, trainable


def count_adapter_parameters(model: Any, *, adapter_name_markers: Sequence[str] = (
        "lora_", "adapter", "lora.")) -> int:
    """Parameters belonging to adapter modules, counted separately from the base."""
    total = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(marker in lowered for marker in adapter_name_markers):
            total += int(parameter.numel())
    return total


def count_from_config(
    model_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Instantiate from configuration and count for real.

    Spec §17 requires an automated count rather than a published figure. The
    caller supplies the factory, so this module never downloads anything itself.
    """
    model = model_factory()
    total, trainable = count_parameters(model)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "adapter_parameters": count_adapter_parameters(model),
        "parameter_count_method": METHOD_COUNTED_FROM_CONFIG,
        "parameter_count_verified": True,
    }


def verify_component_count(
    component: CandidateModel, model_factory: Callable[[], Any],
) -> CandidateModel:
    """Return a copy of ``component`` with real, verified counts."""
    from dataclasses import replace

    counted = count_from_config(model_factory)
    return replace(
        component,
        total_parameters=int(counted["total_parameters"]),
        trainable_parameters=int(counted["trainable_parameters"]),
        adapter_parameters=int(counted["adapter_parameters"]),
        parameter_count_method=METHOD_COUNTED_FROM_CONFIG,
        parameter_count_verified=True)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _component_from_mapping(payload: Mapping[str, Any]) -> CandidateModel:
    total = payload.get("total_parameters")
    trainable = payload.get("trainable_parameters")
    return CandidateModel(
        component_id=str(payload["component_id"]),
        architecture_layer=str(payload.get("architecture_layer", "")),
        training_stage=str(payload.get("training_stage", "")),
        status=str(payload.get("status", STATUS_PLANNED)),
        model_id=str(payload.get("model_id", "")),
        pinned_revision=str(payload.get("pinned_revision", "")),
        checkpoint_role=str(payload.get("checkpoint_role", "")),
        total_parameters=None if total is None else int(total),
        trainable_parameters=None if trainable is None else int(trainable),
        adapter_parameters=int(payload.get("adapter_parameters", 0) or 0),
        loaded_at_inference=bool(payload.get("loaded_at_inference", False)),
        shares_weights_with=str(payload.get("shares_weights_with", "") or ""),
        parameter_count_method=str(
            payload.get("parameter_count_method", METHOD_UNKNOWN)),
        parameter_count_verified=bool(payload.get("parameter_count_verified", False)),
        notes=str(payload.get("notes", "")))


def load_candidate_registry(path: str | Path) -> CandidateRegistry:
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    components = tuple(
        _component_from_mapping(entry) for entry in document.get("components", []))
    return CandidateRegistry(
        components=components,
        version=str(document.get("version", PARAMETER_REGISTRY_VERSION)))


def load_deployment_selection(path: str | Path) -> tuple[str, tuple[str, ...]]:
    """``(manifest name, selected component ids)`` from a deployment manifest."""
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return (
        str(document.get("manifest_name", Path(path).stem)),
        tuple(str(value) for value in document.get("selected_components", [])))


def render_budget_report(report: DeploymentBudgetReport) -> str:
    """Human-readable per-component contribution table."""
    lines = [
        f"deployment: {report.manifest_name}",
        f"limit:                 {MAX_DEPLOYMENT_PARAMETERS:>15,}",
        f"total loaded (gated):  {report.total_loaded_parameters:>15,}",
        f"base only (spec §17):  {report.base_only_parameters:>15,}",
        f"adapters:              {report.adapter_parameters:>15,}",
        f"remaining margin:      {report.remaining_margin:>15,}",
        f"within budget:         {report.within_budget}",
        "",
        f"{'component':<34}{'counted':>15}  shared",
    ]
    for component in report.components:
        marker = component.shared_with or "-"
        suffix = "" if component.counted_once else "  (base already counted)"
        lines.append(
            f"{component.component_id:<34}{component.counted_parameters:>15,}  "
            f"{marker}{suffix}")
    return "\n".join(lines)


def iter_unverified(registry: CandidateRegistry) -> Iterable[CandidateModel]:
    """Components that may not be deployed until counted programmatically."""
    return (c for c in registry.components if not c.has_verified_count)


__all__ = [
    "DEPLOYMENT_BUDGET_VERSION",
    "MAX_DEPLOYMENT_PARAMETERS",
    "METHOD_COUNTED_FROM_CHECKPOINT",
    "METHOD_COUNTED_FROM_CONFIG",
    "METHOD_PUBLISHED_ESTIMATE",
    "METHOD_UNKNOWN",
    "PARAMETER_REGISTRY_VERSION",
    "REGISTRY_STATUSES",
    "STATUS_EXCLUDED_BY_ABLATION",
    "STATUS_IMPLEMENTED",
    "NON_DEPLOYABLE_STATUSES",
    "STATUS_PLANNED",
    "STATUS_RETIRED_FROM_ACTIVE_STACK",
    "STATUS_TRAINED",
    "VERIFIED_METHODS",
    "CandidateModel",
    "CandidateRegistry",
    "DeploymentBudgetExceeded",
    "DeploymentBudgetReport",
    "DeploymentComponent",
    "ParameterBudgetError",
    "UnverifiedDeploymentComponent",
    "compute_deployment_budget",
    "count_adapter_parameters",
    "count_from_config",
    "count_parameters",
    "iter_unverified",
    "load_candidate_registry",
    "load_deployment_selection",
    "render_budget_report",
    "verify_component_count",
]
