"""Trainable learned-L4 v2 model contract and artifact helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...resolution.learned_v2 import (
    BOUNDARY_DROP,
    BOUNDARY_KEEP,
    BOUNDARY_SELECT_EXISTING,
    RESOLVER_V2_VERSION,
    SUPPORTED_BOUNDARY_ACTIONS,
    TYPE_DROP,
    TYPE_ORDER,
    ResolverV2Config,
    ResolverV2TrainingExample,
)
from .artifacts import (
    MODE_SMOKE,
    Phase2TrainingManifest,
    checkpoint_payload,
    write_checkpoint_payload,
)
from .common import canonical_json_sha256, sha256_file

L4_STAGE_ID = "phase2-learned-l4-resolver-v2"
L4_MODEL_ID = "mednorm-learned-l4-mlp-v2"
L4_MODEL_REVISION = "mednorm-learned-l4-mlp-v2"
L4_FULL_AUTHORIZATION = "I_AUTHORIZE_L4_V2_FULL_TRAINING"
TYPE_ACTION_ORDER: tuple[str, ...] = TYPE_ORDER + (TYPE_DROP,)

DEFAULT_FEATURE_ORDER: tuple[str, ...] = (
    "bias",
    "candidate_length",
    "local_score_max",
    "expert_count",
    "source_count",
    "type_support_count",
    "max_type_score",
    "diagnosis_score",
    "symptom_score",
    "diagnosis_minus_symptom",
    "has_route",
    "section_laboratory",
    "section_medication",
    "overlap_count",
    "max_overlap_iou",
    "grammar_completeness",
    "laboratory_structure",
    "left_is_space",
    "right_is_space",
    "left_is_punct",
    "right_is_punct",
    "ontology_placeholder_med_or_diag",
)


class L4TrainingContractError(ValueError):
    """Raised when learned-L4 v2 training contracts are invalid."""


@dataclass(frozen=True, slots=True)
class L4LossWeights:
    boundary_action: float = 1.0
    type_action: float = 1.0
    wrong_type: float = 0.5
    iou_auxiliary: float = 0.2
    wer_boundary: float = 0.1
    class_imbalance: float = 1.0


@dataclass(frozen=True, slots=True)
class L4ModelConfig:
    architecture_id: str = L4_MODEL_ID
    model_revision: str = L4_MODEL_REVISION
    feature_order: tuple[str, ...] = DEFAULT_FEATURE_ORDER
    hidden_size: int = 64
    max_boundary_delta: int = 12
    loss_weights: L4LossWeights = L4LossWeights()

    def parameter_count(self) -> int:
        input_dim = len(self.feature_order)
        boundary_dim = len(SUPPORTED_BOUNDARY_ACTIONS)
        type_dim = len(TYPE_ACTION_ORDER)
        # two-layer compact MLP plus boundary/type/wrong-type/offset/IoU heads
        return (
            input_dim * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * (boundary_dim + type_dim + 5)
            + boundary_dim
            + type_dim
            + 5
        )


@dataclass(frozen=True, slots=True)
class L4TargetIndices:
    boundary_action_index: int
    type_action_index: int
    wrong_type_cost: float
    token_iou: float
    wer_boundary_surrogate: float
    start_delta: int
    end_delta: int


@dataclass(frozen=True, slots=True)
class L4ModelOutputs:
    boundary_logits: tuple[float, ...]
    type_logits: tuple[float, ...]
    wrong_type_logit: float
    iou_score: float
    start_delta: int
    end_delta: int

    def validate_shapes(self) -> None:
        if len(self.boundary_logits) != len(SUPPORTED_BOUNDARY_ACTIONS):
            raise L4TrainingContractError("boundary logits have the wrong shape")
        if len(self.type_logits) != len(TYPE_ACTION_ORDER):
            raise L4TrainingContractError("type logits have the wrong shape")


def feature_vector(
    example: ResolverV2TrainingExample,
    feature_order: Sequence[str] = DEFAULT_FEATURE_ORDER,
) -> tuple[float, ...]:
    return tuple(float(example.features.get(name, 0.0)) for name in feature_order)


def target_indices(example: ResolverV2TrainingExample) -> L4TargetIndices:
    return L4TargetIndices(
        boundary_action_index=SUPPORTED_BOUNDARY_ACTIONS.index(example.boundary_target.action),
        type_action_index=TYPE_ACTION_ORDER.index(example.type_target.entity_type),
        wrong_type_cost=example.type_target.wrong_type_cost,
        token_iou=example.boundary_target.token_iou,
        wer_boundary_surrogate=example.boundary_target.wer_boundary_surrogate,
        start_delta=example.boundary_target.start_delta,
        end_delta=example.boundary_target.end_delta,
    )


def _cross_entropy(logits: Sequence[float], target_index: int) -> float:
    if target_index < 0 or target_index >= len(logits):
        raise L4TrainingContractError("target index outside logits")
    maximum = max(float(value) for value in logits)
    total = sum(math.exp(float(value) - maximum) for value in logits)
    return -float(logits[target_index]) + maximum + math.log(total)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_cross_entropy_with_logit(logit: float, target: float) -> float:
    if logit >= 0:
        return (1.0 - target) * logit + math.log1p(math.exp(-logit))
    return -target * logit + math.log1p(math.exp(logit))


def l4_loss_terms(
    outputs: L4ModelOutputs,
    targets: L4TargetIndices,
    *,
    weights: L4LossWeights | None = None,
) -> dict[str, float]:
    resolved_weights = weights or L4LossWeights()
    outputs.validate_shapes()
    wrong_type_target = min(1.0, max(0.0, targets.wrong_type_cost / 2.0))
    iou_delta = _sigmoid(outputs.iou_score) - targets.token_iou
    start_delta_error = abs(outputs.start_delta - targets.start_delta)
    end_delta_error = abs(outputs.end_delta - targets.end_delta)
    terms = {
        "boundary_action_loss": _cross_entropy(
            outputs.boundary_logits,
            targets.boundary_action_index,
        ),
        "type_loss": _cross_entropy(outputs.type_logits, targets.type_action_index),
        "wrong_type_loss": _binary_cross_entropy_with_logit(
            outputs.wrong_type_logit,
            wrong_type_target,
        ),
        "token_iou_auxiliary_loss": iou_delta * iou_delta,
        "wer_boundary_surrogate_loss": (
            targets.wer_boundary_surrogate * float(start_delta_error + end_delta_error)
        ),
    }
    terms["total_loss"] = (
        resolved_weights.boundary_action * terms["boundary_action_loss"]
        + resolved_weights.type_action * terms["type_loss"]
        + resolved_weights.wrong_type * terms["wrong_type_loss"]
        + resolved_weights.iou_auxiliary * terms["token_iou_auxiliary_loss"]
        + resolved_weights.wer_boundary * terms["wer_boundary_surrogate_loss"]
    )
    return terms


def validate_boundary_action_output(
    *,
    proposal_start: int,
    proposal_end: int,
    original_text: str,
    action: str,
    start_delta: int,
    end_delta: int,
    config: ResolverV2Config,
) -> tuple[int, int]:
    if action not in SUPPORTED_BOUNDARY_ACTIONS:
        raise L4TrainingContractError(f"unsupported L4 boundary action {action!r}")
    if action in {BOUNDARY_KEEP, BOUNDARY_SELECT_EXISTING, BOUNDARY_DROP}:
        start_delta = 0
        end_delta = 0
    if abs(start_delta) > config.max_boundary_delta or abs(end_delta) > config.max_boundary_delta:
        raise L4TrainingContractError("L4 OFFSET action exceeded configured boundary delta")
    start = proposal_start + start_delta
    end = proposal_end + end_delta
    if action == BOUNDARY_DROP:
        return proposal_start, proposal_start
    if start < 0 or end <= start or end > len(original_text):
        raise L4TrainingContractError("L4 boundary action produced invalid offsets")
    if original_text[start:end] == "":
        raise L4TrainingContractError("L4 boundary action produced empty text")
    return start, end


def build_l4_mlp(input_dim: int, hidden_size: int = 64) -> object:
    """Construct the compact Colab-training MLP lazily."""
    if input_dim <= 0 or hidden_size <= 0:
        raise L4TrainingContractError("input_dim and hidden_size must be positive")
    from torch import nn

    class LearnedL4MLP(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.boundary_head = nn.Linear(hidden_size, len(SUPPORTED_BOUNDARY_ACTIONS))
            self.type_head = nn.Linear(hidden_size, len(TYPE_ACTION_ORDER))
            self.wrong_type_head = nn.Linear(hidden_size, 1)
            self.iou_head = nn.Linear(hidden_size, 1)
            self.offset_head = nn.Linear(hidden_size, 2)

        def forward(self, features: Any) -> dict[str, Any]:
            hidden = self.encoder(features)
            return {
                "boundary_logits": self.boundary_head(hidden),
                "type_logits": self.type_head(hidden),
                "wrong_type_logit": self.wrong_type_head(hidden).squeeze(-1),
                "iou_logit": self.iou_head(hidden).squeeze(-1),
                "offset_logits": self.offset_head(hidden),
            }

    return LearnedL4MLP()


def assert_full_not_initialized_from_smoke(
    *,
    run_full_training: bool,
    resume_from_smoke_checkpoint: bool,
) -> None:
    if run_full_training and resume_from_smoke_checkpoint:
        raise L4TrainingContractError("learned L4 full training may not resume from smoke")


def build_l4_resolved_config(
    *,
    mode: str,
    seed: int,
    effective_batch_size: int,
    model_config: L4ModelConfig | None = None,
) -> dict[str, Any]:
    resolved_model_config = model_config or L4ModelConfig()
    return {
        "stage_id": L4_STAGE_ID,
        "expert_id": RESOLVER_V2_VERSION,
        "mode": mode,
        "model_id": resolved_model_config.architecture_id,
        "model_revision": resolved_model_config.model_revision,
        "seed": seed,
        "feature_order": list(resolved_model_config.feature_order),
        "hidden_size": resolved_model_config.hidden_size,
        "max_boundary_delta": resolved_model_config.max_boundary_delta,
        "effective_batch_size": effective_batch_size,
        "boundary_action_space": list(SUPPORTED_BOUNDARY_ACTIONS),
        "type_action_space": list(TYPE_ACTION_ORDER),
        "loss_weights": {
            "boundary_action": resolved_model_config.loss_weights.boundary_action,
            "type_action": resolved_model_config.loss_weights.type_action,
            "wrong_type": resolved_model_config.loss_weights.wrong_type,
            "iou_auxiliary": resolved_model_config.loss_weights.iou_auxiliary,
            "wer_boundary": resolved_model_config.loss_weights.wer_boundary,
            "class_imbalance": resolved_model_config.loss_weights.class_imbalance,
        },
        "internal_test_accessed": False,
    }


def write_l4_checkpoint_stub(
    path: str | Path,
    *,
    mode: str,
    config_sha256: str,
    model_config: L4ModelConfig,
) -> str:
    payload = checkpoint_payload(
        expert_id=RESOLVER_V2_VERSION,
        mode=mode,
        config_sha256=config_sha256,
        model_revision=model_config.model_revision,
        parameter_count=model_config.parameter_count(),
        label_space=TYPE_ACTION_ORDER,
        boundary_action_space=SUPPORTED_BOUNDARY_ACTIONS,
        type_action_space=TYPE_ACTION_ORDER,
    )
    payload["feature_order"] = list(model_config.feature_order)
    write_checkpoint_payload(path, payload)
    return sha256_file(path)


def build_l4_manifest(
    *,
    mode: str,
    status: str,
    run_completed: bool,
    repository_commit: str,
    corpus_hashes: Mapping[str, str],
    data_hashes: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    seed: int,
    completed_epochs: int,
    optimizer_steps: int,
    effective_batch_size: int,
    checkpoint_hashes: Mapping[str, str],
    best_metric: float,
    train_split_id: str,
    validation_split_id: str,
    safe_to_resume: bool,
    initialization_source: str,
    model_config: L4ModelConfig | None = None,
) -> Phase2TrainingManifest:
    resolved_model_config = model_config or L4ModelConfig()
    return Phase2TrainingManifest(
        stage_id=L4_STAGE_ID,
        expert_id=RESOLVER_V2_VERSION,
        mode=mode,
        status=status,
        run_completed=run_completed,
        interrupted_reason="",
        safe_to_resume=safe_to_resume,
        repository_commit=repository_commit,
        corpus_hashes=corpus_hashes,
        data_hashes=data_hashes,
        config_sha256=canonical_json_sha256(dict(resolved_config)),
        model_id=resolved_model_config.architecture_id,
        model_revision=resolved_model_config.model_revision,
        tokenizer_revision="",
        query_revision="",
        query_hash="",
        seed=seed,
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        effective_batch_size=effective_batch_size,
        parameter_count=resolved_model_config.parameter_count(),
        checkpoint_hashes=checkpoint_hashes,
        best_metric=best_metric,
        best_metric_name="validation_exact_f1",
        best_criterion="max_validation_exact_f1_governed_validation_only",
        train_split_id=train_split_id,
        validation_split_id=validation_split_id,
        internal_test_accessed=False,
        initialization_source=initialization_source,
        label_space=TYPE_ACTION_ORDER,
        boundary_action_space=SUPPORTED_BOUNDARY_ACTIONS,
        type_action_space=TYPE_ACTION_ORDER,
        threshold_config={
            "keep_threshold": 0.5,
            "wrong_type_risk_threshold": 0.35,
            "abstain_margin": 0.05,
        },
        best_latest_identical_allowed=mode == MODE_SMOKE,
        best_latest_identical_reason=(
            "bounded smoke may save identical best/latest after one validation point"
            if mode == MODE_SMOKE
            else ""
        ),
    )


__all__ = [
    "DEFAULT_FEATURE_ORDER",
    "L4_FULL_AUTHORIZATION",
    "L4_MODEL_ID",
    "L4_MODEL_REVISION",
    "L4_STAGE_ID",
    "L4LossWeights",
    "L4ModelConfig",
    "L4ModelOutputs",
    "L4TargetIndices",
    "L4TrainingContractError",
    "TYPE_ACTION_ORDER",
    "assert_full_not_initialized_from_smoke",
    "build_l4_manifest",
    "build_l4_mlp",
    "build_l4_resolved_config",
    "feature_vector",
    "l4_loss_terms",
    "target_indices",
    "validate_boundary_action_output",
    "write_l4_checkpoint_stub",
]
