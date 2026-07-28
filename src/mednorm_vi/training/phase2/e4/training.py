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


# ---------------------------------------------------------------------------
# Tiny-overfit stopping policy (Audit 0046)
# ---------------------------------------------------------------------------
#
# Stage 2 originally reused BestCheckpointSelector, the FULL-TRAINING stopper.
# On a 12-example set with accumulation 4 that gives 3 optimizer steps an epoch,
# so a 200-epoch bound plans 600 steps and a 10% warmup needs 60. Validation F1
# is legitimately 0.0 while the head is still warming up, so patience-3 counted
# three "no improvement" epochs and stopped at epoch 4 — 12 optimizer steps, 20%
# of warmup, with the backbone at 1e-6 and the head at 2e-4 instead of the
# configured 5e-6 and 1e-3.
#
# That is not evidence about the recipes. It is a stopper answering a question
# nobody asked: memorization is the goal here, and "F1 has not improved yet" is
# the expected state during warmup, not a reason to abandon the run.

TINY_HEARTBEAT_EVERY_N_EPOCHS = 5

TINY_STOP_GATE_MET = "tiny_gate_met"
TINY_STOP_EPOCH_BOUND = "reached_epoch_bound_without_meeting_the_gate"
TINY_STOP_NUMERIC_FAILURE = "numeric_failure"
TINY_STOP_PROVEN_NOT_LEARNING = "proven_not_learning_after_full_warmup"
TINY_CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class TinyEpochSignal:
    """What one tiny-overfit epoch produced, in the terms the policy needs."""

    epoch: int
    optimizer_steps: int
    exact_f1: float
    predicted_mentions: int
    positive_cell_accuracy: float
    types_predicted: tuple[str, ...]
    loss_total: float
    loss_is_finite: bool = True

    def gate_met(self, *, required_types: Sequence[str], target_f1: float) -> bool:
        """Every tiny pass condition except save/reload, which runs after."""
        return (self.exact_f1 >= target_f1
                and self.predicted_mentions > 0
                and self.positive_cell_accuracy > 0.0
                and all(t in self.types_predicted for t in required_types))

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "optimizer_steps": self.optimizer_steps,
            "exact_f1": self.exact_f1,
            "predicted_mentions": self.predicted_mentions,
            "positive_cell_accuracy": self.positive_cell_accuracy,
            "types_predicted": list(self.types_predicted),
            "loss_total": self.loss_total,
            "loss_is_finite": self.loss_is_finite,
        }


@dataclass(frozen=True, slots=True)
class TinyOverfitStopPolicy:
    """Stopping contract for Stage 2. Deliberately not validation patience.

    A tiny run ends when it has *succeeded*, when it has exhausted its epoch
    bound, or when something is genuinely broken. "F1 is still zero" is none of
    those while the scheduler has not even reached its peak learning rate.

    ``allow_fail_fast`` is opt-in and can only fire **after the full configured
    warmup**, so a fail-fast can never be blamed on an under-warmed schedule.
    """

    epoch_bound: int
    warmup_steps: int
    target_exact_f1: float = 0.95
    required_types: tuple[str, ...] = ()
    heartbeat_every_n_epochs: int = TINY_HEARTBEAT_EVERY_N_EPOCHS
    allow_fail_fast: bool = False
    # Epochs of a completely dead positive signal, after warmup, before the
    # optional fail-fast concludes the run is not learning.
    fail_fast_patience_after_warmup: int = 25

    def __post_init__(self) -> None:
        if self.epoch_bound < 1:
            raise E4TrainingError("epoch_bound must be at least 1")
        if self.warmup_steps < 0:
            raise E4TrainingError("warmup_steps must be non-negative")
        if self.heartbeat_every_n_epochs < 1:
            raise E4TrainingError("heartbeat_every_n_epochs must be at least 1")
        if self.fail_fast_patience_after_warmup < 1:
            raise E4TrainingError("fail_fast patience must be at least 1")

    def should_heartbeat(self, epoch: int) -> bool:
        return (epoch == 1
                or epoch % self.heartbeat_every_n_epochs == 0
                or epoch >= self.epoch_bound)

    def decide(self, history: Sequence[TinyEpochSignal]) -> tuple[bool, str]:
        """``(stop, reason)`` for the run so far."""
        if not history:
            return False, TINY_CONTINUE
        latest = history[-1]

        if not latest.loss_is_finite:
            return True, TINY_STOP_NUMERIC_FAILURE
        if latest.gate_met(required_types=self.required_types,
                           target_f1=self.target_exact_f1):
            return True, TINY_STOP_GATE_MET
        if latest.epoch >= self.epoch_bound:
            return True, TINY_STOP_EPOCH_BOUND

        if self.allow_fail_fast:
            # Only ever after the configured warmup is fully served.
            after_warmup = [s for s in history if s.optimizer_steps >= self.warmup_steps]
            if len(after_warmup) >= self.fail_fast_patience_after_warmup:
                window = after_warmup[-self.fail_fast_patience_after_warmup:]
                if all(s.predicted_mentions == 0 and s.positive_cell_accuracy == 0.0
                       for s in window):
                    return True, TINY_STOP_PROVEN_NOT_LEARNING
        return False, TINY_CONTINUE

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": "tiny_overfit_v1",
            "epoch_bound": self.epoch_bound,
            "warmup_steps": self.warmup_steps,
            "target_exact_f1": self.target_exact_f1,
            "required_types": list(self.required_types),
            "heartbeat_every_n_epochs": self.heartbeat_every_n_epochs,
            "validation_patience_early_stopping_used": False,
            "collapse_guard_enabled": False,
            "allow_fail_fast": self.allow_fail_fast,
            "fail_fast_patience_after_warmup": self.fail_fast_patience_after_warmup,
            "fail_fast_permitted_before_full_warmup": False,
        }


# ---------------------------------------------------------------------------
# Scheduler accounting (Audit 0046)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SchedulePlan:
    """Planned and realized optimizer-step accounting for one run.

    The premature stop was invisible because nothing compared the steps a run
    actually took against the steps its schedule was built for. Both are now
    recorded, and :attr:`peak_learning_rate_reached` says plainly whether the
    configured rates were ever in force.
    """

    examples: int
    accumulation_steps: int
    epoch_bound: int
    optimizer_steps_per_epoch: int
    planned_total_optimizer_steps: int
    warmup_steps: int
    realized_optimizer_steps: int = 0
    realized_epochs: int = 0

    @property
    def warmup_completed(self) -> bool:
        return self.realized_optimizer_steps >= self.warmup_steps

    @property
    def peak_learning_rate_reached(self) -> bool:
        """True once the schedule has served its whole warmup."""
        return self.warmup_completed

    @property
    def warmup_fraction_served(self) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, self.realized_optimizer_steps / self.warmup_steps)

    def realized(self, *, optimizer_steps: int, epochs: int) -> SchedulePlan:
        return SchedulePlan(
            examples=self.examples,
            accumulation_steps=self.accumulation_steps,
            epoch_bound=self.epoch_bound,
            optimizer_steps_per_epoch=self.optimizer_steps_per_epoch,
            planned_total_optimizer_steps=self.planned_total_optimizer_steps,
            warmup_steps=self.warmup_steps,
            realized_optimizer_steps=optimizer_steps,
            realized_epochs=epochs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "accumulation_steps": self.accumulation_steps,
            "epoch_bound": self.epoch_bound,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "planned_total_optimizer_steps": self.planned_total_optimizer_steps,
            "warmup_steps": self.warmup_steps,
            "realized_optimizer_steps": self.realized_optimizer_steps,
            "realized_epochs": self.realized_epochs,
            "warmup_completed": self.warmup_completed,
            "warmup_fraction_served": self.warmup_fraction_served,
            "peak_learning_rate_reached": self.peak_learning_rate_reached,
        }


def plan_schedule(
    *, examples: int, accumulation_steps: int, epoch_bound: int, warmup_ratio: float,
) -> SchedulePlan:
    """Derive the step budget from examples, accumulation and the FULL bound.

    The epoch bound used here must be the one actually requested. Planning a
    schedule over 200 epochs and then stopping at 4 is what produced learning
    rates five times below target.
    """
    plan = plan_gradient_accumulation(
        examples, accumulation_steps=accumulation_steps, epochs=epoch_bound)
    total = plan.expected_optimizer_steps
    return SchedulePlan(
        examples=examples,
        accumulation_steps=accumulation_steps,
        epoch_bound=epoch_bound,
        optimizer_steps_per_epoch=plan.optimizer_steps_per_epoch,
        planned_total_optimizer_steps=total,
        warmup_steps=int(total * warmup_ratio))


__all__ = [  # noqa: F822 - extends the module surface defined above
    *__all__,
    "SchedulePlan",
    "TINY_CONTINUE",
    "TINY_HEARTBEAT_EVERY_N_EPOCHS",
    "TINY_STOP_EPOCH_BOUND",
    "TINY_STOP_GATE_MET",
    "TINY_STOP_NUMERIC_FAILURE",
    "TINY_STOP_PROVEN_NOT_LEARNING",
    "TinyEpochSignal",
    "TinyOverfitStopPolicy",
    "plan_schedule",
]
