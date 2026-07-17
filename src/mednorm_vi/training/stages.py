"""Executable training-stage contracts for S0-S6.

The functions here create deterministic stage plans and readiness manifests.
They do not launch GPU training or download models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TrainingStage:
    stage_id: str
    role: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    requires_gpu: bool = False
    requires_checkpoint: bool = False
    status: str = "planned"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    plan_id: str
    stages: tuple[TrainingStage, ...]
    seed: int = 20260717
    mixed_precision: str = "bf16"
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_training_plan(*, artifact_root: str = "models/checkpoints/full_v1") -> TrainingPlan:
    stages = (
        TrainingStage("S0", "data_engine_and_folds", ("data/manifests",), ("data/derived/folds",)),
        TrainingStage(
            "S1", "mention_span_type_models", ("folds",), (f"{artifact_root}/mention",), True
        ),
        TrainingStage("S2", "assertion_models", ("folds",), (f"{artifact_root}/assertion",), True),
        TrainingStage(
            "S3", "icd_rxnorm_retrieval", ("indices",), (f"{artifact_root}/retrieval",), True
        ),
        TrainingStage(
            "S4", "reranker_resolver", ("candidate_pairs",), (f"{artifact_root}/reranker",), True
        ),
        TrainingStage(
            "S5",
            "qwen_lora_critic_adjudicator",
            ("reviewed_gold",),
            (f"{artifact_root}/qwen",),
            True,
            True,
        ),
        TrainingStage(
            "S6",
            "calibration_and_packaging",
            ("validation_predictions",),
            (f"{artifact_root}/calibration",),
        ),
    )
    payload = [(s.stage_id, s.role, s.inputs, s.outputs) for s in stages]
    plan_id = "train-plan-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return TrainingPlan(plan_id=plan_id, stages=stages)


__all__ = ["TrainingPlan", "TrainingStage", "build_training_plan"]
