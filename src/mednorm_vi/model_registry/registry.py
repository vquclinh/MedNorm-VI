"""Model registry with shared-backbone accounting and profile budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelRole:
    model_id: str
    role: str
    base_parameter_count: int
    adapter_parameter_count: int = 0
    shared_backbone_id: str = ""
    quantization: str = ""
    checkpoint_hash: str = ""
    tokenizer_hash: str = ""
    license: str = ""
    local_path: str = ""
    vram_gb: float = 0.0
    load_order: int = 0
    enabled_profiles: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ProfileBudget:
    profile: str
    base_parameters: int
    adapter_parameters: int
    total_parameters: int
    within_9b: bool
    missing_checkpoints: tuple[str, ...] = field(default_factory=tuple)


def load_registry(path: str | Path) -> tuple[ModelRole, ...]:
    import yaml  # type: ignore[import-untyped]

    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    roles: list[ModelRole] = []
    for row in doc.get("models", []):
        data: dict[str, Any] = dict(row)
        roles.append(
            ModelRole(
                model_id=str(data["model_id"]),
                role=str(data["role"]),
                base_parameter_count=int(data.get("base_parameter_count", 0)),
                adapter_parameter_count=int(data.get("adapter_parameter_count", 0)),
                shared_backbone_id=str(data.get("shared_backbone_id", "")),
                quantization=str(data.get("quantization", "")),
                checkpoint_hash=str(data.get("checkpoint_hash", "")),
                tokenizer_hash=str(data.get("tokenizer_hash", "")),
                license=str(data.get("license", "")),
                local_path=str(data.get("local_path", "")),
                vram_gb=float(data.get("vram_gb", 0.0)),
                load_order=int(data.get("load_order", 0)),
                enabled_profiles=tuple(str(v) for v in data.get("enabled_profiles", [])),
            )
        )
    return tuple(sorted(roles, key=lambda r: (r.load_order, r.model_id)))


def validate_profile_budget(
    roles: tuple[ModelRole, ...],
    *,
    profile: str,
    limit_parameters: int = 9_000_000_000,
    require_local_paths: bool = False,
) -> ProfileBudget:
    enabled = [role for role in roles if profile in role.enabled_profiles]
    seen_backbones: set[str] = set()
    base = 0
    adapters = 0
    missing: list[str] = []
    for role in enabled:
        backbone = role.shared_backbone_id or role.model_id
        if backbone not in seen_backbones:
            base += role.base_parameter_count
            seen_backbones.add(backbone)
        adapters += role.adapter_parameter_count
        if require_local_paths and role.local_path and not Path(role.local_path).exists():
            missing.append(role.local_path)
    total = base + adapters
    return ProfileBudget(
        profile=profile,
        base_parameters=base,
        adapter_parameters=adapters,
        total_parameters=total,
        within_9b=total <= limit_parameters,
        missing_checkpoints=tuple(sorted(missing)),
    )


__all__ = ["ModelRole", "ProfileBudget", "load_registry", "validate_profile_budget"]
