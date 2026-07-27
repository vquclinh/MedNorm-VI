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
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .s1_artifact_validation import SHA256_PATTERN as _SHA256_PATTERN
from .s1_artifact_validation import is_immutable_revision
from .s1_mention_smoke import ENTITY_TYPE_ORDER
from .s1_mention_smoke import sha256_file as _sha256_file

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


# --- checkpoint location contract (Audit 0032) ---------------------------------
#
# ONE resolver shared by the CLI and the notebooks, so local and Colab runs can
# never drift apart. No digest or revision is hardcoded here: the caller supplies
# what it is accepting, exactly as the full-artifact validator does.

S1_BEST_CHECKPOINT_ENV = "MEDNORM_S1_BEST_CHECKPOINT"
COLAB_BEST_CHECKPOINT = Path(
    "/content/drive/MyDrive/MedNorm-VI/artifacts/s1_mention_full_training_v1"
    "/checkpoints/best.pt")
LOCAL_BEST_CHECKPOINT_RELATIVE = Path("checkpoint/s1_mention_full_training_v1/best.pt")

ENVIRONMENT_ENV_OVERRIDE = "env_override"
ENVIRONMENT_COLAB = "colab"
ENVIRONMENT_LOCAL = "local"


@dataclass(frozen=True, slots=True)
class CheckpointLocation:
    """Where the S1 best checkpoint lives, and how that was decided."""

    path: Path
    environment: str
    exists: bool

    def as_dict(self) -> dict[str, Any]:
        return {"checkpoint_path": str(self.path), "environment": self.environment,
                "exists": self.exists}

    def require(self) -> Path:
        """The path, or a clear error naming exactly what is missing."""
        if self.exists:
            return self.path
        hint = {
            ENVIRONMENT_ENV_OVERRIDE: f"{S1_BEST_CHECKPOINT_ENV} points at a missing file",
            ENVIRONMENT_COLAB: "mount Drive and confirm the full-training artifact exists",
            ENVIRONMENT_LOCAL: (
                f"place the checkpoint at {LOCAL_BEST_CHECKPOINT_RELATIVE} "
                f"or set {S1_BEST_CHECKPOINT_ENV}"),
        }[self.environment]
        raise FileNotFoundError(
            f"S1 best checkpoint not found at {self.path} "
            f"(environment: {self.environment}). {hint}.")


def resolve_s1_best_checkpoint(
    repository_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    in_colab: bool | None = None,
) -> CheckpointLocation:
    """Resolve the S1 best checkpoint: env override, then Colab, then local.

    ``in_colab`` defaults to whether ``google.colab`` has been imported, which is
    how every S1 notebook already detects the runtime.
    """
    variables = os.environ if environ is None else environ
    override = str(variables.get(S1_BEST_CHECKPOINT_ENV, "") or "").strip()
    if override:
        path = Path(override).expanduser()
        return CheckpointLocation(path, ENVIRONMENT_ENV_OVERRIDE, path.is_file())
    colab = ("google.colab" in sys.modules) if in_colab is None else bool(in_colab)
    if colab:
        return CheckpointLocation(
            COLAB_BEST_CHECKPOINT, ENVIRONMENT_COLAB, COLAB_BEST_CHECKPOINT.is_file())
    root = Path(repository_root) if repository_root is not None else Path.cwd()
    path = root / LOCAL_BEST_CHECKPOINT_RELATIVE
    return CheckpointLocation(path, ENVIRONMENT_LOCAL, path.is_file())


# --- best-checkpoint-only validation (Audit 0032) ------------------------------
#
# Deliberately SEPARATE from validate_full_training_artifact(). It checks only
# what a lone best.pt can prove and always reports full_artifact_validated=False,
# so a local check can never be mistaken for the full-artifact result.

@dataclass(frozen=True, slots=True)
class BestCheckpointValidationOutcome:
    passed: bool
    failures: tuple[str, ...]
    checkpoint_sha256: str
    epoch: int | None
    global_step: int | None
    pinned_model_revision: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def best_checkpoint_validated(self) -> bool:
        return self.passed and not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            # Stated explicitly and unconditionally: a lone best.pt can never
            # establish that the complete training artifact is valid.
            "full_artifact_validated": False,
            "best_checkpoint_validated": self.best_checkpoint_validated,
            "failed_conditions": list(self.failures),
            "failed_condition_count": len(self.failures),
            "checkpoint_sha256": self.checkpoint_sha256,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "pinned_model_revision": self.pinned_model_revision,
            "diagnostics": dict(self.diagnostics),
        }


def validate_best_checkpoint_only(
    checkpoint_path: str | Path, *,
    expected_sha256: str,
    expected_pinned_revision: str,
    expected_epoch: int,
    expected_global_step: int,
    payload: Mapping[str, Any] | None = None,
    hasher: Callable[[str | Path], str] | None = None,
) -> BestCheckpointValidationOutcome:
    """Validate a lone best checkpoint, read-only.

    ``payload`` is the already-loaded checkpoint mapping (the caller owns Torch).
    The file is never rewritten and no optimizer state is ever constructed.
    """
    path = Path(checkpoint_path)
    digest = hasher if hasher is not None else _sha256_file
    failures: list[str] = []
    if not path.is_file():
        return BestCheckpointValidationOutcome(
            passed=False, failures=(f"checkpoint file does not exist: {path}",),
            checkpoint_sha256="", epoch=None, global_step=None, pinned_model_revision="",
            diagnostics={"checkpoint_path": str(path), "payload_inspected": False})

    computed = digest(path)
    wanted = str(expected_sha256 or "").strip().lower()
    if not _SHA256_PATTERN.match(wanted):
        failures.append(
            "expected checkpoint SHA-256 was not supplied (or is malformed): the "
            f"recomputed digest is {computed!r} (observed: {expected_sha256!r})")
    elif computed != wanted:
        failures.append(
            f"checkpoint SHA-256 does not match the operator-supplied hash "
            f"(recomputed: {computed!r}, expected: {wanted!r})")

    epoch: int | None = None
    global_step: int | None = None
    pinned = ""
    if payload is None:
        failures.append("checkpoint payload was not supplied; nothing inside it was checked")
    else:
        if not isinstance(payload, Mapping):
            failures.append(f"checkpoint payload is not a mapping (got {type(payload).__name__})")
        mode = str(payload.get("mode", ""))
        if mode == SMOKE_MODE:
            failures.append("checkpoint carries the SMOKE_ONLY mode")
        elif mode != FULL_TRAINING_MODE:
            failures.append(f"checkpoint mode is not {FULL_TRAINING_MODE} (observed: {mode!r})")
        for key in CHECKPOINT_REQUIRED_KEYS:
            if key not in payload:
                failures.append(f"checkpoint is missing the required field {key!r}")
        epoch = payload.get("epoch") if isinstance(payload.get("epoch"), int) else None
        global_step = (payload.get("global_step")
                       if isinstance(payload.get("global_step"), int) else None)
        if epoch != int(expected_epoch):
            failures.append(
                f"epoch does not match (observed: {payload.get('epoch')!r}, "
                f"expected: {expected_epoch!r})")
        if global_step != int(expected_global_step):
            failures.append(
                f"global_step does not match (observed: {payload.get('global_step')!r}, "
                f"expected: {expected_global_step!r})")
        pinned = str(payload.get("pinned_model_revision", ""))
        if not is_immutable_revision(pinned):
            failures.append(f"pinned model revision is not immutable (observed: {pinned!r})")
        elif pinned != str(expected_pinned_revision):
            failures.append(
                f"pinned model revision does not match (observed: {pinned!r}, "
                f"expected: {expected_pinned_revision!r})")
        if list(payload.get("entity_type_order") or []) != list(ENTITY_TYPE_ORDER):
            failures.append(
                "checkpoint label space does not match this repository "
                f"(observed: {payload.get('entity_type_order')!r})")
        if not payload.get("model_state_dict"):
            failures.append("checkpoint has no model_state_dict")

    return BestCheckpointValidationOutcome(
        passed=not failures, failures=tuple(failures), checkpoint_sha256=computed,
        epoch=epoch, global_step=global_step, pinned_model_revision=pinned,
        diagnostics={
            "checkpoint_path": str(path),
            "payload_inspected": payload is not None,
            "checkpoint_bytes": path.stat().st_size,
            "scope": "best_checkpoint_only",
            "full_artifact_files_checked": [],
        },
    )


# --- read-only validation of a completed full-training artifact (Audit 0031) ---

@dataclass(frozen=True, slots=True)
class FullTrainingValidationOutcome:
    """Result of validating a completed full-training artifact directory."""

    passed: bool
    failures: tuple[str, ...]
    checkpoint_sha256: dict[str, str]
    best_metric: float | None
    completed_epochs: int | None
    completed_optimizer_steps: int | None
    pinned_model_revision: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def validated(self) -> bool:
        return self.passed and not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "full_training_artifact_validated": self.validated,
            "failed_conditions": list(self.failures),
            "failed_condition_count": len(self.failures),
            "checkpoint_sha256": dict(self.checkpoint_sha256),
            "best_metric": self.best_metric,
            "best_metric_key": BEST_METRIC_KEY,
            "completed_epochs": self.completed_epochs,
            "completed_optimizer_steps": self.completed_optimizer_steps,
            "pinned_model_revision": self.pinned_model_revision,
            "diagnostics": dict(self.diagnostics),
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_full_training_artifact(
    artifact_dir: str | Path, *,
    expected_checkpoint_sha256: Mapping[str, str],
    expected_pinned_revision: str,
    checkpoint_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    hasher: Callable[[str | Path], str] | None = None,
) -> FullTrainingValidationOutcome:
    """Validate a completed run **read-only**; nothing is written or regenerated.

    ``expected_checkpoint_sha256`` maps ``"best_checkpoint"`` / ``"latest_checkpoint"``
    to the digests the operator is accepting, so every hash must agree three ways:
    recomputed from bytes, recorded in the manifest, and supplied here. No digest
    is ever baked into source.

    ``checkpoint_payloads`` carries already-loaded ``.pt`` contents (the caller owns
    Torch, so this module stays free of it). When omitted, file-level checks still
    run and the schema checks are reported as not performed.
    """
    base = Path(artifact_dir)
    digest = hasher if hasher is not None else _sha256_file
    failures: list[str] = []
    paths = full_training_output_paths(base)

    required = {
        "best_checkpoint": Path(paths["best_checkpoint"]),
        "latest_checkpoint": Path(paths["latest_checkpoint"]),
        "training_history": Path(paths["training_history"]),
        "resolved_config": Path(paths["resolved_config"]),
        "validation_metrics": Path(paths["validation_metrics"]),
        "training_manifest": Path(paths["training_manifest"]),
    }
    present = {name: path.is_file() for name, path in required.items()}
    for name, exists in sorted(present.items()):
        if not exists:
            failures.append(f"required artifact missing: {name} ({required[name]})")
    if not present["training_manifest"]:
        return FullTrainingValidationOutcome(
            passed=False, failures=tuple(failures), checkpoint_sha256={},
            best_metric=None, completed_epochs=None, completed_optimizer_steps=None,
            pinned_model_revision="", diagnostics={"files_present": present})

    manifest = _read_json(required["training_manifest"])
    if not isinstance(manifest, dict):
        failures.append("training_manifest.json is not a JSON object")
        manifest = {}

    def field_of(dotted: str, default: Any = None) -> Any:
        node: Any = manifest
        for key in dotted.split("."):
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node

    # --- run identity and completion -----------------------------------------
    for label, holds, observed in (
        ("status is FULL_TRAINING", manifest.get("status") == FULL_TRAINING_MODE,
         manifest.get("status")),
        ("smoke_only_not_full_training is false",
         manifest.get("smoke_only_not_full_training") is False,
         manifest.get("smoke_only_not_full_training")),
        ("run_completed is true", manifest.get("run_completed") is True,
         manifest.get("run_completed")),
        ("interrupted_reason is empty", not str(manifest.get("interrupted_reason", "")),
         manifest.get("interrupted_reason")),
        ("safe_to_resume is true", manifest.get("safe_to_resume") is True,
         manifest.get("safe_to_resume")),
        ("stage is S1", manifest.get("stage_id") == "S1", manifest.get("stage_id")),
    ):
        if not holds:
            failures.append(f"{label} (observed: {observed!r})")

    # --- epoch / step accounting ---------------------------------------------
    completed_epochs = field_of("completed_epochs")
    completed_steps = field_of("completed_optimizer_steps")
    planned_epochs = field_of("hyperparameters.num_epochs")
    planned_steps = field_of("schedule.total_optimizer_steps")
    if completed_epochs != planned_epochs:
        failures.append(
            f"completed_epochs does not match the plan (observed: {completed_epochs!r}, "
            f"planned: {planned_epochs!r})")
    if completed_steps != planned_steps:
        failures.append(
            f"completed_optimizer_steps does not match the plan "
            f"(observed: {completed_steps!r}, planned: {planned_steps!r})")

    # --- pinned revision and initialization ----------------------------------
    pinned = str(field_of("model.pinned_model_revision", ""))
    if not is_immutable_revision(pinned):
        failures.append(f"pinned model revision is not immutable (observed: {pinned!r})")
    if pinned != str(expected_pinned_revision):
        failures.append(
            f"pinned model revision does not match the expected revision "
            f"(observed: {pinned!r}, expected: {expected_pinned_revision!r})")
    if field_of("model.tokenizer_revision") != pinned:
        failures.append("tokenizer revision does not match the pinned model revision")
    if field_of("model.initialize_from") != "pretrained_base":
        failures.append(
            "full training did not initialize from the pretrained base "
            f"(observed: {field_of('model.initialize_from')!r})")
    if field_of("model.initialized_from_smoke_checkpoint") is not False:
        failures.append(
            "manifest does not assert that the smoke checkpoint was NOT the initializer "
            f"(observed: {field_of('model.initialized_from_smoke_checkpoint')!r})")

    # --- best-checkpoint criterion and metric ---------------------------------
    if field_of("best_checkpoint_criterion.key") != BEST_METRIC_KEY:
        failures.append(
            f"best-checkpoint criterion key is not {BEST_METRIC_KEY} "
            f"(observed: {field_of('best_checkpoint_criterion.key')!r})")
    if field_of("best_checkpoint_criterion.mode") != BEST_METRIC_MODE:
        failures.append("best-checkpoint criterion mode is not 'max'")
    best_metric = field_of(f"validation_metrics.{BEST_METRIC_KEY}")
    if not isinstance(best_metric, (int, float)) or not 0.0 <= float(best_metric) <= 1.0:
        failures.append(f"best metric is not a value in [0, 1] (observed: {best_metric!r})")
        best_metric = None
    else:
        best_metric = float(best_metric)

    # --- config hash agreement ------------------------------------------------
    manifest_config_sha = str(manifest.get("config_sha256", ""))
    if present["resolved_config"]:
        resolved = _read_json(required["resolved_config"])
        recomputed = hashlib.sha256(
            json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if recomputed != manifest_config_sha:
            failures.append(
                "resolved_config.json does not hash to the manifest config_sha256 "
                f"(recomputed: {recomputed!r}, manifest: {manifest_config_sha!r})")
        if resolved.get("pinned_revision") != pinned:
            failures.append("resolved_config.json pins a different model revision")

    # --- validation_metrics.json agrees with the manifest ---------------------
    if present["validation_metrics"]:
        metrics = _read_json(required["validation_metrics"])
        if best_metric is not None and metrics.get(BEST_METRIC_KEY) != best_metric:
            failures.append(
                "validation_metrics.json disagrees with the manifest on the best metric "
                f"(file: {metrics.get(BEST_METRIC_KEY)!r}, manifest: {best_metric!r})")

    # --- training history -----------------------------------------------------
    history_epochs = 0
    if present["training_history"]:
        records = [
            json.loads(line) for line in
            required["training_history"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validations = [r for r in records if r.get("event") == "validation"]
        history_epochs = len(validations)
        if not validations:
            failures.append("training_history.jsonl contains no validation records")
        elif isinstance(completed_epochs, int) and history_epochs != completed_epochs:
            failures.append(
                f"training_history.jsonl has {history_epochs} validation record(s) but the "
                f"manifest reports {completed_epochs} completed epoch(s)")

    # --- three-way checkpoint hash agreement ----------------------------------
    recorded = field_of("artifacts.checkpoint_sha256", {}) or {}
    computed: dict[str, str] = {}
    for name in ("best_checkpoint", "latest_checkpoint"):
        if not present[name]:
            continue
        actual = digest(required[name])
        computed[name] = actual
        if actual != str(recorded.get(name, "")):
            failures.append(
                f"{name} SHA-256 does not match the manifest (recomputed: {actual!r}, "
                f"manifest: {recorded.get(name)!r})")
        wanted = str(expected_checkpoint_sha256.get(name, "") or "").strip().lower()
        if not _SHA256_PATTERN.match(wanted):
            failures.append(
                f"expected {name} SHA-256 was not supplied (or is malformed): the "
                f"recomputed digest is {actual!r} (observed: "
                f"{expected_checkpoint_sha256.get(name)!r})")
        elif actual != wanted:
            failures.append(
                f"{name} SHA-256 does not match the operator-supplied hash "
                f"(recomputed: {actual!r}, expected: {wanted!r})")
    if computed.get("best_checkpoint") and computed.get("best_checkpoint") == computed.get(
            "latest_checkpoint"):
        failures.append("best and latest checkpoints are byte-identical")

    # --- checkpoint payload schema (caller supplies the loaded contents) -------
    schema_checked = False
    if checkpoint_payloads:
        schema_checked = True
        for name, payload in sorted(checkpoint_payloads.items()):
            if str(payload.get("mode", "")) == SMOKE_MODE:
                failures.append(f"{name} carries the SMOKE_ONLY mode")
            elif str(payload.get("mode", "")) != FULL_TRAINING_MODE:
                failures.append(f"{name} has an unknown mode {payload.get('mode')!r}")
            for key in CHECKPOINT_REQUIRED_KEYS:
                if key not in payload:
                    failures.append(f"{name} is missing the resume field {key!r}")
            if str(payload.get("pinned_model_revision", "")) != pinned:
                failures.append(f"{name} was trained on a different pinned revision")
            if list(payload.get("entity_type_order") or []) != list(ENTITY_TYPE_ORDER):
                failures.append(f"{name} has a different label space")
            if str(payload.get("config_sha256", "")) != manifest_config_sha:
                failures.append(f"{name} config_sha256 does not match the manifest")

    cache_files = sorted(
        str(path.relative_to(base)) for path in base.rglob("*")
        if path.is_file() and path.suffix in (".safetensors", ".h5", ".onnx", ".msgpack")
    )
    if cache_files:
        failures.append(f"base-model cache files inside the artifact: {cache_files!r}")

    return FullTrainingValidationOutcome(
        passed=not failures,
        failures=tuple(failures),
        checkpoint_sha256=computed,
        best_metric=best_metric,
        completed_epochs=completed_epochs if isinstance(completed_epochs, int) else None,
        completed_optimizer_steps=completed_steps if isinstance(completed_steps, int) else None,
        pinned_model_revision=pinned,
        diagnostics={
            "files_present": present,
            "artifact_dir": str(base),
            "checkpoint_schema_checked": schema_checked,
            "history_validation_records": history_epochs,
            "effective_batch_size": manifest.get("effective_batch_size"),
            "hf_model_id": field_of("model.hf_model_id", ""),
            "corpus_manifest_sha256": field_of("corpus.corpus_manifest_sha256", ""),
            "repository_commit": field_of("repository.resolved_commit", ""),
            "smoke_artifact_dir": field_of("artifacts.smoke_artifact_dir", ""),
            "validation_metrics": manifest.get("validation_metrics", {}),
        },
    )


__all__ = [
    "BEST_CHECKPOINT_NAME",
    "BEST_METRIC_KEY",
    "BEST_METRIC_MODE",
    "CHECKPOINT_REQUIRED_KEYS",
    "FULL_TRAINING_MODE",
    "LATEST_CHECKPOINT_NAME",
    "SMOKE_MODE",
    "SUPPORTED_LOSSES",
    "BestCheckpointValidationOutcome",
    "CheckpointLocation",
    "COLAB_BEST_CHECKPOINT",
    "LOCAL_BEST_CHECKPOINT_RELATIVE",
    "S1_BEST_CHECKPOINT_ENV",
    "FullTrainingConfig",
    "resolve_s1_best_checkpoint",
    "validate_best_checkpoint_only",
    "FullTrainingValidationOutcome",
    "validate_full_training_artifact",
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
