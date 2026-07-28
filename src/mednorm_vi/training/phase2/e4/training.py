"""E4 training-loop contracts and collapse detection (Audit 0045).

Accumulation accounting, T4 precision policy, checkpoint custody, and the guard
that would have stopped the collapsed run at epoch 5 instead of epoch 12.

Nothing here executes a training step. It defines the arithmetic and the
invariants; the Colab notebook applies them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..training_contracts import (
    DEVICE_CPU,
    DEVICE_CUDA,
    PRECISION_BF16,
    PRECISION_FP16,
    PRECISION_FP32,
    SUPPORTED_PRECISION_MODES,
    AccumulationPlan,
    MixedPrecisionPolicy,
    assert_step_accounting,
    assert_training_device,
    resolve_mixed_precision_policy,
)
from ..training_contracts import (
    plan_gradient_accumulation as _plan_generic,
)
from .contracts import (
    E4_CHECKPOINT_SCHEMA_VERSION,
    E4_INPUT_CONTRACT_VERSION,
    E4ContractError,
    reject_superseded_checkpoint,
)
from .recipes import OptimizerGroups, ScheduleConfig

TRAINING_CONTRACT_VERSION = "e4-training-v1"

INITIALIZATION_PINNED_BASE = "pinned_pretrained_base_fresh_head"
INITIALIZATION_SAME_RUN_RESUME = "same_run_interrupted_resume"

# Everything an exact same-run resume needs.
REQUIRED_RESUME_KEYS: tuple[str, ...] = (
    "model_state", "optimizer_state", "scaler_state", "scheduler_state",
    "epoch", "optimizer_steps", "best_metric", "recipe", "run_id",
)
# Fields that must match exactly before a resume is accepted.
RESUME_COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "e4_input_contract_version",
    "e4_checkpoint_schema_version",
    "atomic_projection_version",
    "config_sha256",
    "model_revision",
    "tokenizer_revision",
    "recipe",
    "run_id",
    "optimizer_signature",
    "accumulation_signature",
)


class E4TrainingError(E4ContractError):
    """Raised when a training contract or accounting invariant is violated."""


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


def plan_gradient_accumulation(
    example_count: int, *, micro_batch_size: int = 1,
    accumulation_steps: int, epochs: int,
) -> AccumulationPlan:
    """E4 plans with microbatch 1: relation grids are variable-sized per document."""
    if micro_batch_size != 1:
        raise E4TrainingError(
            "E4 grids are variable-sized per document; micro_batch_size is 1 and the "
            "effective batch is reached by accumulation")
    return _plan_generic(
        example_count, micro_batch_size=1,
        accumulation_steps=accumulation_steps, epochs=epochs)


# ---------------------------------------------------------------------------
# Collapse guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationSnapshot:
    """One validation pass, in the terms the guard actually needs."""

    epoch: int
    predicted_mentions: int
    gold_mentions: int
    true_positives: int
    thw_predictions: int
    nnw_predictions: int
    gold_positive_background_rate: float
    train_loss: float

    @property
    def recall(self) -> float:
        return self.true_positives / self.gold_mentions if self.gold_mentions else 0.0

    @property
    def is_collapsed(self) -> bool:
        """Every symptom the audited run showed, at once.

        All four are required. Any one alone is survivable early in training —
        an epoch-1 model legitimately predicts nothing — so a guard keyed on a
        single symptom would abort healthy runs.
        """
        return (self.predicted_mentions == 0
                and self.thw_predictions == 0
                and self.recall == 0.0
                and self.gold_positive_background_rate >= COLLAPSE_BACKGROUND_RATE)


# The audited run reached 0.99910 at its best epoch and 1.0 at its last.
COLLAPSE_BACKGROUND_RATE = 0.999
# Consecutive collapsed validations tolerated after warmup before stopping.
DEFAULT_COLLAPSE_PATIENCE = 2


@dataclass(frozen=True, slots=True)
class CollapseVerdict:
    collapsed: bool
    reason: str
    consecutive_collapsed_epochs: int
    first_collapsed_epoch: int
    loss_decreasing_while_collapsed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "collapsed": self.collapsed,
            "reason": self.reason,
            "consecutive_collapsed_epochs": self.consecutive_collapsed_epochs,
            "first_collapsed_epoch": self.first_collapsed_epoch,
            "loss_decreasing_while_collapsed": self.loss_decreasing_while_collapsed,
            "status_if_collapsed": "COLLAPSED_NOT_TRAINED",
        }


def evaluate_collapse_guard(
    history: Sequence[ValidationSnapshot],
    *,
    warmup_epochs: int = 1,
    patience: int = DEFAULT_COLLAPSE_PATIENCE,
) -> CollapseVerdict:
    """Stop a run that is optimizing loss without producing any mention.

    Applied only after ``warmup_epochs`` so an untrained epoch-1 model is not
    mistaken for a collapse. Against the audited history this fires at epoch 6 —
    the third consecutive collapsed validation, epochs 4/5/6 — instead of letting
    the run burn eight more epochs and 200,000 backward passes.

    "Loss decreasing while collapsed" is reported but not required: the audited
    run's loss actually *rose* into the attractor, so requiring a falling loss
    would have missed it entirely.
    """
    if patience < 1:
        raise E4TrainingError("collapse patience must be at least 1")
    eligible = [s for s in history if s.epoch > warmup_epochs]
    streak = 0
    first = -1
    for snapshot in eligible:
        if snapshot.is_collapsed:
            if streak == 0:
                first = snapshot.epoch
            streak += 1
        else:
            streak = 0
            first = -1
    collapsed_tail = [s for s in eligible if s.is_collapsed][-streak:] if streak else []
    loss_falling = (
        len(collapsed_tail) >= 2
        and collapsed_tail[-1].train_loss < collapsed_tail[0].train_loss)
    if streak >= patience:
        return CollapseVerdict(
            collapsed=True,
            reason=(
                f"{streak} consecutive post-warmup validations predicted zero "
                f"mentions, zero THW relations and zero recall while at least "
                f"{COLLAPSE_BACKGROUND_RATE:.1%} of gold-positive cells were "
                f"predicted background"),
            consecutive_collapsed_epochs=streak,
            first_collapsed_epoch=first,
            loss_decreasing_while_collapsed=loss_falling)
    return CollapseVerdict(
        collapsed=False, reason="", consecutive_collapsed_epochs=streak,
        first_collapsed_epoch=first, loss_decreasing_while_collapsed=loss_falling)


def assert_not_collapsed_when_marking_trained(
    verdict: CollapseVerdict, status: str,
) -> None:
    """A collapsed run must never be recorded as successfully trained."""
    if verdict.collapsed and status not in ("COLLAPSED_NOT_TRAINED", "FAILED"):
        raise E4TrainingError(
            f"refusing to record status {status!r} for a collapsed run; "
            f"{verdict.reason}")


# ---------------------------------------------------------------------------
# Early stopping and checkpoint selection
# ---------------------------------------------------------------------------


@dataclass
class BestCheckpointSelector:
    """Selection by exact span-and-type F1 on governed validation only."""

    patience: int = 3
    best_metric: float = -1.0
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def observe(self, *, epoch: int, exact_f1: float) -> bool:
        """Return True when this epoch is a new best."""
        if exact_f1 > self.best_metric:
            self.best_metric = float(exact_f1)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "best_metric_name": "validation_exact_f1",
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "early_stopping_patience": self.patience,
            "selection_split": "governed_validation_only",
        }


# ---------------------------------------------------------------------------
# Resume custody
# ---------------------------------------------------------------------------


def assert_resume_custody(payload: Mapping[str, Any]) -> None:
    missing = tuple(key for key in REQUIRED_RESUME_KEYS if key not in payload)
    if missing:
        raise E4TrainingError(
            "E4 checkpoint is missing resume state: " + ", ".join(sorted(missing)))
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise E4TrainingError("E4 checkpoint model_state must be a mapping")
    for key in ("base_model", "w2ner_head"):
        if key not in model_state:
            raise E4TrainingError(f"E4 checkpoint model_state lacks {key!r}")


def assert_same_run_resume(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any],
) -> None:
    """Accept a resume only for an interruption of *this* run.

    ``run_id`` is in the compatibility set on purpose. Resuming is for a Colab
    session that died mid-run; it is not a way to continue a different run, and
    it is certainly not a way to warm-start from the collapsed implementation —
    :func:`reject_superseded_checkpoint` refuses that by schema version first.
    """
    reject_superseded_checkpoint(payload)
    assert_resume_custody(payload)
    differences = [
        f"{field}: checkpoint={payload.get(field)!r} expected={expected.get(field)!r}"
        for field in RESUME_COMPATIBILITY_FIELDS
        if str(payload.get(field, "")) != str(expected.get(field, ""))
    ]
    if differences:
        raise E4TrainingError(
            "E4 resume is incompatible: " + "; ".join(differences))


def resolve_initialization_source(
    *, resume_from_same_run: bool, checkpoint: Mapping[str, Any] | None = None,
) -> str:
    """Fresh pretrained weights unless this is a same-run interruption resume."""
    if not resume_from_same_run:
        return INITIALIZATION_PINNED_BASE
    if checkpoint is None:
        raise E4TrainingError("a resume requires the checkpoint payload to validate")
    reject_superseded_checkpoint(checkpoint)
    return INITIALIZATION_SAME_RUN_RESUME


def build_training_accounting(
    *,
    plan: AccumulationPlan,
    precision: MixedPrecisionPolicy,
    optimizer: OptimizerGroups,
    schedule: ScheduleConfig,
    observed_optimizer_steps: int,
    observed_backward_passes: int,
    observed_examples: int,
    recipe_name: str,
) -> dict[str, Any]:
    assert_step_accounting(plan, observed_optimizer_steps)
    if observed_backward_passes != plan.expected_backward_passes:
        raise E4TrainingError(
            f"backward-pass accounting mismatch: expected "
            f"{plan.expected_backward_passes}, observed {observed_backward_passes}")
    return {
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        **plan.as_dict(),
        **precision.as_dict(),
        "optimizer": optimizer.as_dict(),
        "schedule": schedule.as_dict(),
        "recipe": recipe_name,
        "observed_optimizer_steps": int(observed_optimizer_steps),
        "observed_backward_passes": int(observed_backward_passes),
        "examples_processed": int(observed_examples),
        "e4_input_contract_version": E4_INPUT_CONTRACT_VERSION,
        "e4_checkpoint_schema_version": E4_CHECKPOINT_SCHEMA_VERSION,
        "internal_test_accessed": False,
    }


__all__ = [
    "COLLAPSE_BACKGROUND_RATE",
    "DEFAULT_COLLAPSE_PATIENCE",
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "INITIALIZATION_PINNED_BASE",
    "INITIALIZATION_SAME_RUN_RESUME",
    "PRECISION_BF16",
    "PRECISION_FP16",
    "PRECISION_FP32",
    "REQUIRED_RESUME_KEYS",
    "RESUME_COMPATIBILITY_FIELDS",
    "SUPPORTED_PRECISION_MODES",
    "TRAINING_CONTRACT_VERSION",
    "AccumulationPlan",
    "BestCheckpointSelector",
    "CollapseVerdict",
    "E4TrainingError",
    "MixedPrecisionPolicy",
    "ValidationSnapshot",
    "assert_not_collapsed_when_marking_trained",
    "assert_resume_custody",
    "assert_same_run_resume",
    "assert_step_accounting",
    "assert_training_device",
    "build_training_accounting",
    "evaluate_collapse_guard",
    "plan_gradient_accumulation",
    "resolve_initialization_source",
    "resolve_mixed_precision_policy",
]
