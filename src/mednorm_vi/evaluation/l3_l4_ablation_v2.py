"""Phase-2 L3/L4 ablation plan and availability reporting (spec §18.2).

This entry point defines the required Phase-2 arms without fabricating scores
for untrained experts. Arms whose checkpoints are not present are reported as
``UNAVAILABLE_UNTRAINED`` and must be excluded from metric tables until a frozen
validation-selected checkpoint exists.

Audit 0051 removed every arm that contained E4 PhoBERT-W2NER. An arm whose
membership can never be satisfied is not a deferred measurement — it is a
permanently `UNAVAILABLE_UNTRAINED` row that makes the plan look larger than the
set of questions this project can still answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..lattice.models import (
    EXPERT_GLINER,
    EXPERT_VIHEALTHBERT,
    EXPERT_XLMR_MRC,
)

STATUS_IMPLEMENTED = "IMPLEMENTED"
STATUS_EVALUABLE = "EVALUABLE"
STATUS_UNAVAILABLE_UNTRAINED = "UNAVAILABLE_UNTRAINED"
STATUS_DISABLED = "DISABLED_BY_PROFILE"

ARM_E3_ONLY = "E3_only"
ARM_E3_E5 = "E3_plus_E5_xlmr_mrc"
ARM_E3_E6 = "E3_plus_E6_gliner"
ARM_E3_E5_E6 = "E3_plus_E5_plus_E6"
ARM_ALL_L3 = "all_available_l3_experts"
ARM_L4_V1 = "deterministic_l4_v1"
ARM_L4_V2 = "learned_l4_v2"


@dataclass(frozen=True, slots=True)
class AblationArmSpec:
    arm: str
    required_experts: tuple[str, ...]
    required_flags: tuple[str, ...]
    checkpoint_keys: tuple[str, ...]
    resolver: str = ""


@dataclass(frozen=True, slots=True)
class AblationArmStatus:
    arm: str
    status: str
    reason: str
    required_experts: tuple[str, ...]
    missing_checkpoints: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "status": self.status,
            "reason": self.reason,
            "required_experts": list(self.required_experts),
            "missing_checkpoints": list(self.missing_checkpoints),
        }


PHASE2_ABLATION_ARMS: tuple[AblationArmSpec, ...] = (
    AblationArmSpec(
        ARM_E3_ONLY,
        (EXPERT_VIHEALTHBERT,),
        ("enable_e3_vihealthbert",),
        ("e3_vihealthbert",),
    ),
    AblationArmSpec(
        ARM_E3_E5,
        (EXPERT_VIHEALTHBERT, EXPERT_XLMR_MRC),
        ("enable_e3_vihealthbert", "enable_e5_xlmr_mrc"),
        ("e3_vihealthbert", "e5_xlmr_mrc"),
    ),
    AblationArmSpec(
        ARM_E3_E6,
        (EXPERT_VIHEALTHBERT, EXPERT_GLINER),
        ("enable_e3_vihealthbert", "enable_e6_gliner"),
        ("e3_vihealthbert", "e6_gliner"),
    ),
    AblationArmSpec(
        ARM_E3_E5_E6,
        (EXPERT_VIHEALTHBERT, EXPERT_XLMR_MRC, EXPERT_GLINER),
        ("enable_e3_vihealthbert", "enable_e5_xlmr_mrc", "enable_e6_gliner"),
        ("e3_vihealthbert", "e5_xlmr_mrc", "e6_gliner"),
    ),
    AblationArmSpec(
        ARM_ALL_L3,
        (EXPERT_VIHEALTHBERT, EXPERT_XLMR_MRC, EXPERT_GLINER),
        (
            "enable_e3_vihealthbert",
            "enable_e5_xlmr_mrc",
            "enable_e6_gliner",
        ),
        ("e3_vihealthbert", "e5_xlmr_mrc", "e6_gliner"),
    ),
    AblationArmSpec(
        ARM_L4_V1,
        (EXPERT_VIHEALTHBERT,),
        ("enable_e3_vihealthbert", "enable_l4_deterministic_v1"),
        ("e3_vihealthbert",),
        resolver="deterministic_l4_v1",
    ),
    AblationArmSpec(
        ARM_L4_V2,
        (EXPERT_VIHEALTHBERT,),
        ("enable_e3_vihealthbert", "enable_l4_learned_v2"),
        ("e3_vihealthbert", "l4_learned_v2"),
        resolver="learned_l4_v2",
    ),
)


def _checkpoint_missing(path: str) -> bool:
    return not path or not Path(path).exists()


def plan_phase2_ablation(
    feature_flags: Mapping[str, bool],
    checkpoint_paths: Mapping[str, str],
) -> tuple[AblationArmStatus, ...]:
    statuses: list[AblationArmStatus] = []
    for arm in PHASE2_ABLATION_ARMS:
        disabled_flags = tuple(
            flag for flag in arm.required_flags if not feature_flags.get(flag, False)
        )
        missing = tuple(
            key for key in arm.checkpoint_keys if _checkpoint_missing(checkpoint_paths.get(key, ""))
        )
        if disabled_flags:
            statuses.append(
                AblationArmStatus(
                    arm.arm,
                    STATUS_DISABLED,
                    "required feature flag disabled: " + ",".join(disabled_flags),
                    arm.required_experts,
                    missing,
                )
            )
            continue
        if missing:
            statuses.append(
                AblationArmStatus(
                    arm.arm,
                    STATUS_UNAVAILABLE_UNTRAINED,
                    "one or more required local checkpoints are unavailable",
                    arm.required_experts,
                    missing,
                )
            )
            continue
        statuses.append(
            AblationArmStatus(
                arm.arm,
                STATUS_EVALUABLE,
                "all required flags and local checkpoints are present",
                arm.required_experts,
                (),
            )
        )
    return tuple(statuses)


__all__ = [
    "ARM_ALL_L3",
    "ARM_E3_E5",
    "ARM_E3_E5_E6",
    "ARM_E3_E6",
    "ARM_E3_ONLY",
    "ARM_L4_V1",
    "ARM_L4_V2",
    "AblationArmSpec",
    "AblationArmStatus",
    "PHASE2_ABLATION_ARMS",
    "STATUS_DISABLED",
    "STATUS_EVALUABLE",
    "STATUS_IMPLEMENTED",
    "STATUS_UNAVAILABLE_UNTRAINED",
    "plan_phase2_ablation",
]
