"""Phase-2 validation-profile budget and metadata checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...model_registry.registry import ModelRole, load_registry
from .common import is_immutable_revision

PHASE2_BASE_PARAMETER_LIMIT = 9_000_000_000


@dataclass(frozen=True, slots=True)
class Phase2BudgetReport:
    profile_name: str
    model_ids: tuple[str, ...]
    base_parameters: int
    adapter_parameters: int
    within_budget: bool
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.within_budget and not self.failures

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_name": self.profile_name,
            "model_ids": list(self.model_ids),
            "base_parameters": self.base_parameters,
            "adapter_parameters": self.adapter_parameters,
            "within_budget": self.within_budget,
            "failures": list(self.failures),
        }


def _index_roles(roles: Sequence[ModelRole]) -> dict[str, ModelRole]:
    return {role.model_id: role for role in roles}


def validate_phase2_validation_profile_budget(
    roles: Sequence[ModelRole],
    *,
    profile_name: str,
    model_ids: Sequence[str],
    require_local_paths: bool = True,
) -> Phase2BudgetReport:
    by_id = _index_roles(roles)
    failures: list[str] = []
    seen_backbones: set[str] = set()
    base = 0
    adapters = 0
    for model_id in model_ids:
        role = by_id.get(model_id)
        if role is None:
            failures.append(f"model_missing_from_registry:{model_id}")
            continue
        backbone = role.shared_backbone_id or role.model_id
        if backbone not in seen_backbones:
            base += role.base_parameter_count
            seen_backbones.add(backbone)
        adapters += role.adapter_parameter_count
        if not is_immutable_revision(role.model_revision):
            failures.append(f"model_revision_not_immutable:{model_id}")
        if not role.checkpoint_hash or role.checkpoint_hash.startswith("UNAVAILABLE"):
            failures.append(f"checkpoint_hash_missing:{model_id}")
        if require_local_paths:
            if not role.local_path:
                failures.append(f"local_path_missing:{model_id}")
            elif not Path(role.local_path).exists():
                failures.append(f"local_path_not_found:{model_id}")
        if role.local_path.startswith(("http://", "https://")):
            failures.append(f"silent_network_acquisition_path:{model_id}")
    within_budget = base <= PHASE2_BASE_PARAMETER_LIMIT
    if not within_budget:
        failures.append("base_parameter_budget_exceeded")
    return Phase2BudgetReport(
        profile_name=profile_name,
        model_ids=tuple(model_ids),
        base_parameters=base,
        adapter_parameters=adapters,
        within_budget=within_budget,
        failures=tuple(sorted(failures)),
    )


def validate_phase2_profile_budget_from_registry(
    registry_path: str | Path,
    *,
    profile_name: str,
    model_ids: Sequence[str],
    require_local_paths: bool = True,
) -> Phase2BudgetReport:
    return validate_phase2_validation_profile_budget(
        load_registry(registry_path),
        profile_name=profile_name,
        model_ids=model_ids,
        require_local_paths=require_local_paths,
    )


__all__ = [
    "PHASE2_BASE_PARAMETER_LIMIT",
    "Phase2BudgetReport",
    "validate_phase2_profile_budget_from_registry",
    "validate_phase2_validation_profile_budget",
]
