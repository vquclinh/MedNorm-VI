"""E4 training recipes (Audit 0045).

Exactly three candidates. This is not a hyperparameter search — it is a
controlled ablation over the one thing Audit 0044 proved was wrong.

What the collapsed run optimized
--------------------------------

    loss = cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))   # per example
    epoch_loss = mean over examples of that per-example mean

Two independent defects, both measured:

* **Per-example normalization.** A 5-word example contributes 25 cells and a
  162-word example contributes 26,244, yet each counted once. The gradient was
  dominated by short documents.
* **No compensation for 577:1 background.** 0.173% of train cells are positive
  and 81.34% of train examples contain no positive cell at all, so an
  input-independent predictor emitting the class marginal is a stationary point.
  The run converged to 92.14% of that predictor's loss — a 7.9% improvement over
  ignoring the input entirely, after 405,912 backward passes.

What every recipe here fixes
----------------------------

**Batch-global valid-cell reduction.** One effective batch accumulates a
numerator over every valid cell in every microbatch and divides **once** by the
total valid-cell count. A cell is worth the same wherever it occurs, and
gradient accumulation becomes mathematically identical to a single large batch
rather than an approximation of it (:class:`BatchGlobalAccumulator`).

The three recipes then differ in exactly one dimension each:

===================== =============================== ========================
recipe                objective                       data order
===================== =============================== ========================
reference_ce          batch-global CE                 shuffled + interleaved
reference_ce_resampled batch-global CE                positive-aware sampling
balanced_focal        batch-global focal              shuffled + interleaved
===================== =============================== ========================

Nothing here imports torch at module scope, and nothing trains. The reductions
are expressed over plain numeric inputs so they are testable on CPU without a
model, and the Colab loop applies the identical arithmetic to tensors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import E4ContractError

RECIPE_CONTRACT_VERSION = "e4-recipe-v1"

REFERENCE_CE = "reference_ce"
REFERENCE_CE_RESAMPLED = "reference_ce_resampled"
BALANCED_FOCAL = "balanced_focal"

# Deliberately closed. A fourth recipe is a new milestone, not a config value.
RECIPE_NAMES: tuple[str, ...] = (REFERENCE_CE, REFERENCE_CE_RESAMPLED, BALANCED_FOCAL)

# Complexity order, used to break an exact tie in favour of the simpler recipe.
RECIPE_COMPLEXITY: Mapping[str, int] = {
    REFERENCE_CE: 0,
    REFERENCE_CE_RESAMPLED: 1,
    BALANCED_FOCAL: 2,
}

# ---------------------------------------------------------------------------
# Optimizer and schedule defaults
# ---------------------------------------------------------------------------
#
# The backbone is 24 pretrained layers; the relation head is randomly
# initialized. The collapsed run gave both 2e-5, which is far too slow for a
# fresh head and fast enough to drag the backbone into the head's early garbage.
DEFAULT_BACKBONE_LR = 5e-6
DEFAULT_HEAD_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.10
DEFAULT_MAX_GRAD_NORM = 5.0

# Focal defaults. Conservative on purpose: gamma 2.0 is the standard focusing
# exponent, and alpha 0.25 weights the positive classes without approaching the
# raw 577:1 inverse-frequency weight, which is explicitly refused.
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_FOCAL_ALPHA = 0.25
MAX_PERMITTED_FOCAL_ALPHA = 0.9
# A guard, not a tuning knob: any positive-class weight at or above this is the
# inverse-frequency regime that turns background into noise.
REFUSED_CLASS_WEIGHT_RATIO = 100.0


class RecipeError(E4ContractError):
    """Raised when a recipe is configured outside its supported contract."""


# ---------------------------------------------------------------------------
# Batch-global valid-cell reduction
# ---------------------------------------------------------------------------


@dataclass
class BatchGlobalAccumulator:
    """Accumulate a loss numerator and a valid-cell count across microbatches.

    The contract, stated as arithmetic:

        loss(effective batch) = (sum over microbatches of per-cell loss sums)
                              / (total valid cells in the effective batch)

    Not ``mean(per-example means)``. The difference is the whole point: under
    the mean-of-means, one 5-word document outweighs a 162-word document by
    three orders of magnitude per cell.

    Because the divisor is known only after the last microbatch, a training loop
    scales each microbatch's backward pass by ``1 / expected_valid_cells`` when
    it must step immediately. :meth:`microbatch_scale` returns that factor, and
    :meth:`reduced` returns the exact value for reporting.
    """

    loss_numerator: float = 0.0
    valid_cells: int = 0
    positive_numerator: float = 0.0
    positive_cells: int = 0
    background_numerator: float = 0.0
    background_cells: int = 0
    microbatches: int = 0

    def observe_microbatch(
        self,
        *,
        loss_sum: float,
        cells: int,
        positive_sum: float = 0.0,
        positive_cells: int = 0,
    ) -> None:
        if cells <= 0:
            raise RecipeError("a microbatch must contribute at least one valid cell")
        if positive_cells > cells:
            raise RecipeError("positive cells cannot exceed valid cells")
        self.loss_numerator += float(loss_sum)
        self.valid_cells += int(cells)
        self.positive_numerator += float(positive_sum)
        self.positive_cells += int(positive_cells)
        self.background_numerator += float(loss_sum) - float(positive_sum)
        self.background_cells += int(cells) - int(positive_cells)
        self.microbatches += 1

    def reduced(self) -> float:
        """The effective batch's loss: one division, at the very end."""
        if self.valid_cells == 0:
            raise RecipeError("cannot reduce an effective batch with no valid cells")
        return self.loss_numerator / self.valid_cells

    def loss_breakdown(self) -> dict[str, float]:
        """Total, positive and background loss, reported separately.

        A single total is what let the collapse hide: it fell steadily while the
        positive term — the only one that can produce a mention — never moved.
        """
        return {
            "total": self.reduced(),
            "positive": (self.positive_numerator / self.positive_cells
                         if self.positive_cells else 0.0),
            "background": (self.background_numerator / self.background_cells
                           if self.background_cells else 0.0),
            "positive_cells": float(self.positive_cells),
            "background_cells": float(self.background_cells),
            "valid_cells": float(self.valid_cells),
        }

    def reset(self) -> None:
        self.loss_numerator = 0.0
        self.valid_cells = 0
        self.positive_numerator = 0.0
        self.positive_cells = 0
        self.background_numerator = 0.0
        self.background_cells = 0
        self.microbatches = 0


def microbatch_scale(expected_valid_cells: int) -> float:
    """Backward-pass scale so accumulated gradients equal one batch-global mean.

    Each microbatch backward-propagates ``cell_loss_sum / expected_valid_cells``.
    Summing those over the effective batch gives exactly
    ``total_loss_sum / expected_valid_cells`` — identical to reducing the whole
    batch at once, which a test asserts numerically.
    """
    if expected_valid_cells <= 0:
        raise RecipeError("expected_valid_cells must be positive")
    return 1.0 / float(expected_valid_cells)


# ---------------------------------------------------------------------------
# Objectives, expressed over plain numbers so they are testable without torch
# ---------------------------------------------------------------------------


def _log_softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(float(v) for v in logits)
    exps = [math.exp(float(v) - maximum) for v in logits]
    total = math.log(sum(exps)) + maximum
    return [float(v) - total for v in logits]


def cross_entropy_cell(logits: Sequence[float], label_id: int) -> float:
    if label_id < 0 or label_id >= len(logits):
        raise RecipeError("label id outside the relation-logit vector")
    return -_log_softmax(logits)[label_id]


def focal_cell(
    logits: Sequence[float],
    label_id: int,
    *,
    gamma: float,
    alpha: float,
    background_id: int = 0,
) -> float:
    """Focal loss for one cell: ``-w * (1 - p_t)**gamma * log p_t``.

    ``w`` is ``alpha`` for a positive class and ``1 - alpha`` for background.
    An easy background cell (``p_t`` near 1) is suppressed by the ``(1 - p_t)``
    factor while every positive cell keeps a weight of at least ``alpha``, so
    positives always participate — which is the property that distinguishes this
    from simply reweighting the classes.
    """
    log_probs = _log_softmax(logits)
    if label_id < 0 or label_id >= len(log_probs):
        raise RecipeError("label id outside the relation-logit vector")
    log_pt = log_probs[label_id]
    pt = math.exp(log_pt)
    weight = (1.0 - alpha) if label_id == background_id else alpha
    return float(-weight * ((1.0 - pt) ** gamma) * log_pt)


@dataclass(frozen=True, slots=True)
class FocalConfig:
    gamma: float = DEFAULT_FOCAL_GAMMA
    alpha: float = DEFAULT_FOCAL_ALPHA
    background_id: int = 0

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise RecipeError("focal gamma must be non-negative")
        if not 0.0 < self.alpha < 1.0:
            raise RecipeError("focal alpha must lie strictly between 0 and 1")
        if self.alpha > MAX_PERMITTED_FOCAL_ALPHA:
            raise RecipeError(
                f"focal alpha {self.alpha} exceeds {MAX_PERMITTED_FOCAL_ALPHA}; that "
                "is the inverse-frequency regime this milestone refuses")
        # alpha/(1-alpha) is the effective positive:background weight ratio.
        ratio = self.alpha / (1.0 - self.alpha)
        if ratio >= REFUSED_CLASS_WEIGHT_RATIO:
            raise RecipeError(
                f"effective class-weight ratio {ratio:.1f} is at or beyond the raw "
                f"577:1 inverse-frequency regime, which is explicitly refused")

    def as_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "background_id": self.background_id,
            "effective_positive_to_background_weight_ratio": (
                self.alpha / (1.0 - self.alpha)),
        }


def reduce_grid(
    logits: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[Sequence[int]],
    pair_mask: Sequence[Sequence[bool]],
    *,
    objective: str,
    focal: FocalConfig | None = None,
    background_id: int = 0,
) -> tuple[float, int, float, int]:
    """One grid's ``(loss_sum, cells, positive_loss_sum, positive_cells)``.

    Returns **sums**, never a mean. The caller feeds them to
    :class:`BatchGlobalAccumulator`, which performs the single division. Any
    function here that returned a mean would silently reintroduce the defect
    this module exists to remove.
    """
    if objective not in ("cross_entropy", "focal"):
        raise RecipeError(f"unknown objective {objective!r}")
    if objective == "focal" and focal is None:
        focal = FocalConfig()
    if len(logits) != len(labels) or len(labels) != len(pair_mask):
        raise RecipeError("logits, labels and pair mask must have the same rows")

    loss_sum = 0.0
    positive_sum = 0.0
    cells = 0
    positive_cells = 0
    for logit_row, label_row, mask_row in zip(logits, labels, pair_mask, strict=True):
        if len(logit_row) != len(label_row) or len(label_row) != len(mask_row):
            raise RecipeError("a grid row has inconsistent width")
        for scores, label_id, valid in zip(logit_row, label_row, mask_row, strict=True):
            if not valid:
                continue
            label = int(label_id)
            value = (
                cross_entropy_cell(scores, label) if objective == "cross_entropy"
                else focal_cell(scores, label, gamma=focal.gamma,  # type: ignore[union-attr]
                                alpha=focal.alpha,                 # type: ignore[union-attr]
                                background_id=focal.background_id)  # type: ignore[union-attr]
            )
            loss_sum += value
            cells += 1
            if label != background_id:
                positive_sum += value
                positive_cells += 1
    if cells == 0:
        raise RecipeError("grid reduction saw no valid cell")
    return loss_sum, cells, positive_sum, positive_cells


# ---------------------------------------------------------------------------
# Recipe definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptimizerGroups:
    """Differential learning rates. The head must not inherit the backbone's."""

    backbone_lr: float = DEFAULT_BACKBONE_LR
    head_lr: float = DEFAULT_HEAD_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    name: str = "AdamW"

    def __post_init__(self) -> None:
        if self.backbone_lr <= 0.0 or self.head_lr <= 0.0:
            raise RecipeError("both learning rates must be positive")
        if self.backbone_lr == self.head_lr:
            raise RecipeError(
                "the pretrained backbone and the freshly initialized relation head "
                "must not share one learning rate; that is what the collapsed run did")
        if self.head_lr < self.backbone_lr:
            raise RecipeError(
                "the randomly initialized head must learn faster than the "
                "pretrained backbone")

    @property
    def signature(self) -> str:
        return (f"{self.name}-backbone{self.backbone_lr:g}"
                f"-head{self.head_lr:g}-wd{self.weight_decay:g}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backbone_lr": self.backbone_lr,
            "head_lr": self.head_lr,
            "weight_decay": self.weight_decay,
            "signature": self.signature,
            "parameter_groups": ["backbone", "relation_head"],
        }


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    """Linear warmup then linear decay. The collapsed run had neither."""

    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    kind: str = "linear_warmup_linear_decay"

    def __post_init__(self) -> None:
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise RecipeError("warmup_ratio must lie in [0, 1)")
        if self.max_grad_norm <= 0.0:
            raise RecipeError("max_grad_norm must be positive")

    def warmup_steps(self, total_steps: int) -> int:
        if total_steps <= 0:
            raise RecipeError("total_steps must be positive")
        return int(total_steps * self.warmup_ratio)

    def multiplier_at(self, step: int, total_steps: int) -> float:
        """LR multiplier at a 0-based optimizer step."""
        if step < 0:
            raise RecipeError("step must be non-negative")
        if total_steps <= 0:
            raise RecipeError("total_steps must be positive")
        warmup = self.warmup_steps(total_steps)
        if step >= total_steps:
            return 0.0
        if warmup and step < warmup:
            return (step + 1) / warmup
        remaining = total_steps - max(warmup, 0)
        if remaining <= 0:
            return 1.0
        return max(0.0, (total_steps - step) / remaining)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass(frozen=True, slots=True)
class Recipe:
    """One complete, self-describing training contract."""

    name: str
    objective: str
    reduction: str
    data_order: str
    optimizer: OptimizerGroups = field(default_factory=OptimizerGroups)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    focal: FocalConfig | None = None
    contract_version: str = RECIPE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.name not in RECIPE_NAMES:
            raise RecipeError(
                f"unknown recipe {self.name!r}; expected one of {RECIPE_NAMES}")
        if self.reduction != "batch_global_valid_cell_mean":
            raise RecipeError(
                "every recipe must use the batch-global valid-cell reduction; the "
                "per-example mean is the defect this milestone removes")
        if self.objective == "focal" and self.focal is None:
            raise RecipeError("the focal objective requires an explicit FocalConfig")
        if self.objective == "cross_entropy" and self.focal is not None:
            raise RecipeError("a cross-entropy recipe must not carry a focal config")

    @property
    def uses_positive_aware_sampling(self) -> bool:
        return self.data_order == "positive_aware_resampled"

    @property
    def complexity(self) -> int:
        return RECIPE_COMPLEXITY[self.name]

    def reduce(
        self,
        logits: Sequence[Sequence[Sequence[float]]],
        labels: Sequence[Sequence[int]],
        pair_mask: Sequence[Sequence[bool]],
    ) -> tuple[float, int, float, int]:
        return reduce_grid(
            logits, labels, pair_mask,
            objective=self.objective, focal=self.focal)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": self.contract_version,
            "name": self.name,
            "objective": self.objective,
            "reduction": self.reduction,
            "data_order": self.data_order,
            "positive_aware_sampling": self.uses_positive_aware_sampling,
            "optimizer": self.optimizer.as_dict(),
            "schedule": self.schedule.as_dict(),
            "per_example_mean_used": False,
            "class_weights_used": self.objective == "focal",
        }
        if self.focal is not None:
            payload["focal"] = self.focal.as_dict()
        return payload


def build_recipe(name: str, **overrides: Any) -> Recipe:
    """Construct one of the three candidates by name."""
    if name not in RECIPE_NAMES:
        raise RecipeError(f"unknown recipe {name!r}; expected one of {RECIPE_NAMES}")
    optimizer = overrides.pop("optimizer", None) or OptimizerGroups()
    schedule = overrides.pop("schedule", None) or ScheduleConfig()
    if overrides:
        raise RecipeError(f"unsupported recipe overrides: {sorted(overrides)}")
    if name == REFERENCE_CE:
        return Recipe(
            name=name, objective="cross_entropy",
            reduction="batch_global_valid_cell_mean",
            data_order="shuffled_source_interleaved",
            optimizer=optimizer, schedule=schedule)
    if name == REFERENCE_CE_RESAMPLED:
        return Recipe(
            name=name, objective="cross_entropy",
            reduction="batch_global_valid_cell_mean",
            data_order="positive_aware_resampled",
            optimizer=optimizer, schedule=schedule)
    return Recipe(
        name=name, objective="focal",
        reduction="batch_global_valid_cell_mean",
        data_order="shuffled_source_interleaved",
        optimizer=optimizer, schedule=schedule, focal=FocalConfig())


def all_recipes() -> tuple[Recipe, ...]:
    """The complete candidate set, in ablation order."""
    return tuple(build_recipe(name) for name in RECIPE_NAMES)


__all__ = [
    "BALANCED_FOCAL",
    "DEFAULT_BACKBONE_LR",
    "DEFAULT_FOCAL_ALPHA",
    "DEFAULT_FOCAL_GAMMA",
    "DEFAULT_HEAD_LR",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_WARMUP_RATIO",
    "MAX_PERMITTED_FOCAL_ALPHA",
    "RECIPE_COMPLEXITY",
    "RECIPE_CONTRACT_VERSION",
    "RECIPE_NAMES",
    "REFERENCE_CE",
    "REFERENCE_CE_RESAMPLED",
    "REFUSED_CLASS_WEIGHT_RATIO",
    "BatchGlobalAccumulator",
    "FocalConfig",
    "OptimizerGroups",
    "Recipe",
    "RecipeError",
    "ScheduleConfig",
    "all_recipes",
    "build_recipe",
    "cross_entropy_cell",
    "focal_cell",
    "microbatch_scale",
    "reduce_grid",
]
