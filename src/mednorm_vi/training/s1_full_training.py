"""Full S1 mention-training contracts (Audit 0025).

The smoke notebook proved the S1 pipeline runs end-to-end. This module holds the
**pure, testable** contracts the full run needs on top of that: the tracked
hyperparameter configuration, the schedule arithmetic, the validation metrics,
the checkpoint/resume rules, and the training-manifest schema.

Two rules drive most of the design:

* the smoke checkpoint is *execution evidence*, never an initialization — full
  training starts from the approved pretrained ViHealthBERT revision;
* that revision must be an **immutable commit hash**. ``main`` moves, and a
  moving reference makes a leaderboard run unreproducible (spec §15.2: "every
  leaderboard decision records config, seed, Git commit, and artifact hash").

Nothing here imports Torch or Transformers; the notebook performs the training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .s1_artifact_validation import is_immutable_revision
from .s1_mention_smoke import ENTITY_TYPE_ORDER

# Full-training artifact layout. Every path is relative to the output directory,
# which must be distinct from the smoke artifact directory.
LATEST_CHECKPOINT_NAME = "checkpoints/latest.pt"
BEST_CHECKPOINT_NAME = "checkpoints/best.pt"
TRAINING_HISTORY_NAME = "logs/training_history.jsonl"
RESOLVED_CONFIG_NAME = "resolved_config.json"
TRAINING_MANIFEST_NAME = "training_manifest.json"
VALIDATION_METRICS_NAME = "validation_metrics.json"

# Marks a checkpoint as a genuine full-training checkpoint. The smoke checkpoint
# carries "SMOKE_ONLY" and is therefore rejected as a resume/initialization source.
FULL_TRAINING_MODE = "FULL_TRAINING"
SMOKE_MODE = "SMOKE_ONLY"
CHECKPOINT_REQUIRED_KEYS = (
    "mode", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "epoch", "global_step", "best_metric", "entity_type_order", "seed",
    "pinned_model_revision", "config_sha256",
)

# Best-checkpoint criterion. Span-level micro F1 is the closest available proxy
# for the organizer's entity-level scoring; per-type token metrics are recorded
# alongside it for the error analysis required by spec §18.1.
BEST_METRIC_KEY = "validation_span_micro_f1"
BEST_METRIC_MODE = "max"

SUPPORTED_LOSSES = ("bce", "focal")


class FullTrainingConfigError(ValueError):
    """Raised when the full-training configuration violates a contract."""


@dataclass(frozen=True, slots=True)
class FullTrainingConfig:
    """Tracked, validated hyperparameters for the full S1 mention run."""

    stage_id: str
    seed: int
    hf_model_id: str
    registry_model_id: str
    requested_revision: str
    pinned_revision: str
    output_dir: str
    smoke_artifact_dir: str
    initialize_from: str
    num_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    head_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    max_sequence_length: int
    loss_type: str
    focal_gamma: float
    focal_alpha: float
    decision_threshold: float
    mixed_precision: str
    filter_unsupervised_examples: bool
    corpus: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def config_sha256(self) -> str:
        """Deterministic hash of the resolved configuration."""
        payload = json.dumps(self.resolved(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolved(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id, "seed": self.seed,
            "hf_model_id": self.hf_model_id,
            "registry_model_id": self.registry_model_id,
            "requested_revision": self.requested_revision,
            "pinned_revision": self.pinned_revision,
            "output_dir": self.output_dir,
            "initialize_from": self.initialize_from,
            "num_epochs": self.num_epochs,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "learning_rate": self.learning_rate,
            "head_learning_rate": self.head_learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "max_sequence_length": self.max_sequence_length,
            "loss_type": self.loss_type,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "decision_threshold": self.decision_threshold,
            "mixed_precision": self.mixed_precision,
            "filter_unsupervised_examples": self.filter_unsupervised_examples,
            "entity_type_order": list(ENTITY_TYPE_ORDER),
            "best_metric_key": BEST_METRIC_KEY,
            "best_metric_mode": BEST_METRIC_MODE,
        }

    def validate(self) -> None:
        """Reject every configuration that cannot produce a reproducible run."""
        if self.stage_id != "S1":
            raise FullTrainingConfigError(f"stage_id must be S1, got {self.stage_id!r}")
        if not is_immutable_revision(self.pinned_revision):
            raise FullTrainingConfigError(
                f"pinned_revision must be an immutable 40-hex commit hash, got "
                f"{self.pinned_revision!r}. Read it from the validated smoke manifest "
                "(model.resolved_model_revision); never invent one and never ship 'main'.")
        if self.initialize_from != "pretrained_base":
            raise FullTrainingConfigError(
                "full S1 training must initialize from the approved pretrained base "
                f"revision, not {self.initialize_from!r}. The one-step smoke checkpoint is "
                "execution evidence only.")
        if _same_path(self.output_dir, self.smoke_artifact_dir):
            raise FullTrainingConfigError(
                "full-training output_dir must differ from the smoke artifact directory; "
                "the validated smoke artifact must never be overwritten")
        if self.loss_type not in SUPPORTED_LOSSES:
            raise FullTrainingConfigError(
                f"loss_type must be one of {SUPPORTED_LOSSES}, got {self.loss_type!r}")
        if self.num_epochs < 1:
            raise FullTrainingConfigError("num_epochs must be >= 1")
        if self.per_device_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise FullTrainingConfigError("batch size and accumulation must be >= 1")
        if not 0.0 < self.learning_rate < 1e-2:
            raise FullTrainingConfigError(f"implausible learning_rate {self.learning_rate}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise FullTrainingConfigError("warmup_ratio must be in [0, 1)")
        if not 0.0 < self.decision_threshold < 1.0:
            raise FullTrainingConfigError("decision_threshold must be in (0, 1)")
        # ViHealthBERT-Word inherits PhoBERT's 258 position embeddings.
        if self.max_sequence_length < 32 or self.max_sequence_length > 256:
            raise FullTrainingConfigError(
                "max_sequence_length must be between 32 and 256 (PhoBERT position limit)")
        if self.mixed_precision not in ("auto", "fp16", "bf16", "none"):
            raise FullTrainingConfigError(f"unknown mixed_precision {self.mixed_precision!r}")


def _same_path(left: str, right: str) -> bool:
    return Path(str(left)).resolve() == Path(str(right)).resolve()


def load_full_training_config(
    path: str | Path, *, pinned_revision: str | None = None,
    smoke_artifact_dir: str | Path | None = None,
) -> FullTrainingConfig:
    """Load and validate the tracked full-training configuration.

    ``pinned_revision`` and ``smoke_artifact_dir`` override the file, which is how
    the Colab notebook feeds the revision **and the directory** of the smoke
    artifact it actually validated at runtime. Full training therefore consumes
    the artifact that passed, never a hardcoded or historical one.
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise FullTrainingConfigError("full-training config must be a mapping")
    model = doc.get("model") or {}
    optimization = doc.get("optimization") or {}
    loss = doc.get("loss") or {}
    data = doc.get("data") or {}
    output = doc.get("output") or {}
    config = FullTrainingConfig(
        stage_id=str(doc.get("stage_id", "")),
        seed=int(doc.get("seed", 0)),
        hf_model_id=str(model.get("hf_model_id", "")),
        registry_model_id=str(model.get("registry_model_id", "")),
        requested_revision=str(model.get("requested_revision", "")),
        pinned_revision=str(
            pinned_revision if pinned_revision is not None
            else model.get("pinned_revision", "")),
        output_dir=str(output.get("output_dir", "")),
        smoke_artifact_dir=str(
            smoke_artifact_dir if smoke_artifact_dir is not None
            else output.get("smoke_artifact_dir", "")),
        initialize_from=str(model.get("initialize_from", "")),
        num_epochs=int(optimization.get("num_epochs", 0)),
        per_device_batch_size=int(optimization.get("per_device_batch_size", 0)),
        gradient_accumulation_steps=int(optimization.get("gradient_accumulation_steps", 0)),
        learning_rate=float(optimization.get("learning_rate", 0.0)),
        head_learning_rate=float(optimization.get("head_learning_rate", 0.0)),
        weight_decay=float(optimization.get("weight_decay", 0.0)),
        warmup_ratio=float(optimization.get("warmup_ratio", 0.0)),
        max_grad_norm=float(optimization.get("max_grad_norm", 0.0)),
        max_sequence_length=int(data.get("max_sequence_length", 0)),
        loss_type=str(loss.get("type", "")),
        focal_gamma=float(loss.get("focal_gamma", 0.0)),
        focal_alpha=float(loss.get("focal_alpha", 0.0)),
        decision_threshold=float(loss.get("decision_threshold", 0.0)),
        mixed_precision=str(optimization.get("mixed_precision", "")),
        filter_unsupervised_examples=bool(data.get("filter_unsupervised_examples", False)),
        corpus=dict(doc.get("corpus") or {}),
        raw=dict(doc),
    )
    config.validate()
    return config


# --- schedule -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TrainingSchedule:
    supervised_examples: int
    steps_per_epoch: int
    total_optimizer_steps: int
    warmup_steps: int

    def as_dict(self) -> dict[str, int]:
        return {
            "supervised_examples": self.supervised_examples,
            "steps_per_epoch": self.steps_per_epoch,
            "total_optimizer_steps": self.total_optimizer_steps,
            "warmup_steps": self.warmup_steps,
        }


def derive_schedule(config: FullTrainingConfig, supervised_examples: int) -> TrainingSchedule:
    """Optimizer-step accounting from the real number of supervised examples."""
    if supervised_examples < 1:
        raise FullTrainingConfigError("no supervised training examples available")
    micro_batches = -(-supervised_examples // config.per_device_batch_size)  # ceil
    steps_per_epoch = max(1, micro_batches // config.gradient_accumulation_steps)
    total = steps_per_epoch * config.num_epochs
    return TrainingSchedule(
        supervised_examples=supervised_examples,
        steps_per_epoch=steps_per_epoch,
        total_optimizer_steps=total,
        warmup_steps=int(total * config.warmup_ratio),
    )


def is_supervised_example(loss_mask: Mapping[str, Any]) -> bool:
    """True when an example actually contributes mention supervision.

    ``phoner_covid19`` declares ``boundary: false`` / ``entity_type: false`` in the
    governed annotation-coverage manifest, so its ``label_mask`` is entirely zero.
    Such examples produce no gradient; training on them only burns Colab GPU time.
    The governed corpus itself is never modified — only the sampling is scoped.
    """
    return bool(loss_mask.get("span")) and bool(loss_mask.get("entity_type"))


# --- validation metrics -------------------------------------------------------

def _spans_from_row(row: Sequence[Sequence[int]], mask: Sequence[int]) -> set[tuple[int, int, int]]:
    """Contiguous positive runs per entity type as ``(type_id, start, end)``.

    Token-index space, half-open — the same convention as the character offsets
    in the spec. Character-exact scoring belongs to the L4 resolver and the
    mandatory local evaluator; here it is the training-time selection signal.
    """
    spans: set[tuple[int, int, int]] = set()
    type_count = len(ENTITY_TYPE_ORDER)
    for type_id in range(type_count):
        start: int | None = None
        for position, token in enumerate(row):
            active = bool(mask[position]) and bool(token[type_id])
            if active and start is None:
                start = position
            elif not active and start is not None:
                spans.add((type_id, start, position))
                start = None
        if start is not None:
            spans.add((type_id, start, len(row)))
    return spans


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if (
        true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (
        true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative}


class MentionMetrics:
    """Streaming P/R/F1 for S1 mention detection and typing.

    Accumulates token-level counts per entity type plus span-level counts, so
    validation never materializes the whole epoch in memory. Padding and
    unsupervised positions are excluded through ``label_mask``.
    """

    def __init__(self) -> None:
        size = len(ENTITY_TYPE_ORDER)
        self.token_tp = [0] * size
        self.token_fp = [0] * size
        self.token_fn = [0] * size
        self.span_tp = 0
        self.span_fp = 0
        self.span_fn = 0
        self.supervised_tokens = 0

    def update(
        self,
        predictions: Sequence[Sequence[Sequence[int]]],
        labels: Sequence[Sequence[Sequence[int]]],
        label_mask: Sequence[Sequence[int]],
    ) -> None:
        for pred_row, gold_row, mask_row in zip(predictions, labels, label_mask, strict=True):
            for position, keep in enumerate(mask_row):
                if not keep:
                    continue
                self.supervised_tokens += 1
                for type_id in range(len(ENTITY_TYPE_ORDER)):
                    predicted = bool(pred_row[position][type_id])
                    gold = bool(gold_row[position][type_id])
                    if predicted and gold:
                        self.token_tp[type_id] += 1
                    elif predicted:
                        self.token_fp[type_id] += 1
                    elif gold:
                        self.token_fn[type_id] += 1
            predicted_spans = _spans_from_row(pred_row, mask_row)
            gold_spans = _spans_from_row(gold_row, mask_row)
            self.span_tp += len(predicted_spans & gold_spans)
            self.span_fp += len(predicted_spans - gold_spans)
            self.span_fn += len(gold_spans - predicted_spans)

    def compute(self) -> dict[str, Any]:
        per_type = {
            name: _prf(self.token_tp[i], self.token_fp[i], self.token_fn[i])
            for i, name in enumerate(ENTITY_TYPE_ORDER)
        }
        micro = _prf(sum(self.token_tp), sum(self.token_fp), sum(self.token_fn))
        # Macro over types that actually occur; a type with no gold and no
        # prediction would otherwise drag the average toward zero.
        observed = [
            name for i, name in enumerate(ENTITY_TYPE_ORDER)
            if self.token_tp[i] + self.token_fn[i] + self.token_fp[i] > 0
        ]
        macro_f1 = (
            sum(per_type[name]["f1"] for name in observed) / len(observed) if observed else 0.0
        )
        span = _prf(self.span_tp, self.span_fp, self.span_fn)
        return {
            "supervised_tokens": self.supervised_tokens,
            "token_micro_precision": micro["precision"],
            "token_micro_recall": micro["recall"],
            "token_micro_f1": micro["f1"],
            "token_macro_f1": macro_f1,
            "observed_entity_types": observed,
            "per_type": per_type,
            "validation_span_precision": span["precision"],
            "validation_span_recall": span["recall"],
            BEST_METRIC_KEY: span["f1"],
        }


def is_better_metric(candidate: float, incumbent: float | None) -> bool:
    """Best-checkpoint criterion: strictly greater span micro F1."""
    if incumbent is None:
        return True
    return float(candidate) > float(incumbent)


# --- checkpoint and resume contracts ------------------------------------------

def build_checkpoint_payload(
    config: FullTrainingConfig, *, epoch: int, global_step: int, best_metric: float,
    model_state_dict: Any, optimizer_state_dict: Any, scheduler_state_dict: Any,
) -> dict[str, Any]:
    """Assemble a full-training checkpoint that satisfies the resume contract."""
    return {
        "mode": FULL_TRAINING_MODE,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "entity_type_order": list(ENTITY_TYPE_ORDER),
        "seed": int(config.seed),
        "pinned_model_revision": config.pinned_revision,
        "config_sha256": config.config_sha256,
    }


def validate_resume_checkpoint(
    payload: Mapping[str, Any], config: FullTrainingConfig,
) -> list[str]:
    """Reasons this checkpoint may not be resumed; empty means it is safe.

    The one-step smoke checkpoint is rejected here by design: it carries
    ``mode == "SMOKE_ONLY"`` and no scheduler/step state, so resuming from it
    would silently restart a full run from a one-batch artifact.
    """
    problems: list[str] = []
    mode = str(payload.get("mode", ""))
    if mode == SMOKE_MODE:
        problems.append(
            "refusing to resume from the SMOKE_ONLY checkpoint: it is execution "
            "evidence, not a full-training state")
    elif mode != FULL_TRAINING_MODE:
        problems.append(f"unknown checkpoint mode {mode!r}")
    for key in CHECKPOINT_REQUIRED_KEYS:
        if key not in payload:
            problems.append(f"missing checkpoint field: {key}")
    if payload.get("pinned_model_revision") not in (None, config.pinned_revision):
        problems.append(
            f"checkpoint was trained on revision {payload.get('pinned_model_revision')!r}, "
            f"config pins {config.pinned_revision!r}")
    if list(payload.get("entity_type_order") or []) not in ([], list(ENTITY_TYPE_ORDER)):
        problems.append("checkpoint entity_type_order does not match this repository")
    return problems


def full_training_output_paths(output_dir: str | Path) -> dict[str, str]:
    """Canonical artifact paths; the base-model cache never lives here."""
    base = Path(output_dir)
    return {
        "latest_checkpoint": str(base / LATEST_CHECKPOINT_NAME),
        "best_checkpoint": str(base / BEST_CHECKPOINT_NAME),
        "training_history": str(base / TRAINING_HISTORY_NAME),
        "resolved_config": str(base / RESOLVED_CONFIG_NAME),
        "training_manifest": str(base / TRAINING_MANIFEST_NAME),
        "validation_metrics": str(base / VALIDATION_METRICS_NAME),
    }


def build_full_training_manifest(
    config: FullTrainingConfig, *,
    schedule: TrainingSchedule,
    repository: Mapping[str, Any],
    corpus: Mapping[str, Any],
    environment: Mapping[str, Any],
    segmentation: Mapping[str, Any],
    alignment: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    completed_epochs: int,
    completed_optimizer_steps: int,
    validation_metrics: Mapping[str, Any],
    checkpoint_hashes: Mapping[str, str],
    run_completed: bool,
    interrupted_reason: str = "",
) -> dict[str, Any]:
    """The full-training manifest: everything needed to judge and rebuild the run."""
    return {
        "manifest_version": 1,
        "stage_id": "S1",
        "role": "mention/vihealthbert",
        "status": FULL_TRAINING_MODE,
        "smoke_only_not_full_training": False,
        "architecture_spec_version": "1.1",
        "repository": dict(repository),
        "corpus": dict(corpus),
        "model": {
            "registry_model_id": config.registry_model_id,
            "hf_model_id": config.hf_model_id,
            "requested_revision": config.requested_revision,
            "pinned_model_revision": config.pinned_revision,
            "tokenizer_revision": config.pinned_revision,
            "initialize_from": config.initialize_from,
            "initialized_from_smoke_checkpoint": False,
        },
        "tokenizer": dict(tokenizer),
        "word_segmentation": dict(segmentation),
        "alignment": dict(alignment),
        "environment": dict(environment),
        "hyperparameters": config.resolved(),
        "effective_batch_size": config.effective_batch_size,
        "schedule": schedule.as_dict(),
        "completed_epochs": int(completed_epochs),
        "completed_optimizer_steps": int(completed_optimizer_steps),
        "validation_metrics": dict(validation_metrics),
        "best_checkpoint_criterion": {"key": BEST_METRIC_KEY, "mode": BEST_METRIC_MODE},
        "artifacts": {
            **full_training_output_paths(config.output_dir),
            "checkpoint_sha256": dict(checkpoint_hashes),
            "smoke_artifact_dir": config.smoke_artifact_dir,
        },
        "config_sha256": config.config_sha256,
        "run_completed": bool(run_completed),
        "interrupted_reason": str(interrupted_reason),
        # A run is resumable whenever a valid latest checkpoint exists, whether it
        # finished or was cut short by a Colab disconnect.
        "safe_to_resume": bool(checkpoint_hashes.get("latest_checkpoint", "")),
    }


__all__ = [
    "BEST_CHECKPOINT_NAME",
    "BEST_METRIC_KEY",
    "BEST_METRIC_MODE",
    "CHECKPOINT_REQUIRED_KEYS",
    "FULL_TRAINING_MODE",
    "LATEST_CHECKPOINT_NAME",
    "SMOKE_MODE",
    "SUPPORTED_LOSSES",
    "FullTrainingConfig",
    "FullTrainingConfigError",
    "MentionMetrics",
    "TrainingSchedule",
    "build_checkpoint_payload",
    "build_full_training_manifest",
    "derive_schedule",
    "full_training_output_paths",
    "is_better_metric",
    "is_supervised_example",
    "load_full_training_config",
    "validate_resume_checkpoint",
]
