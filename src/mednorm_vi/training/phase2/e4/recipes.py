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

**Batch-global reduction.** One effective batch accumulates numerators and
counts across every microbatch and divides **once** at the end, so gradient
accumulation is mathematically identical to a single large batch rather than an
approximation of it (:class:`BatchGlobalAccumulator`). All three reductions are
batch-global; they differ in *which* cells the denominator counts.

**Why a second and third objective exist.** The completed Stage-2 ablation
(Audit 0047) showed natural-frequency cell CE is still dominated by background
even with the batch-global fix: positive loss sat near 1.79 while background loss
was near 0.0055, and the total near 0.016 followed the background. The two new
objectives attack exactly that ratio — one by averaging the groups separately,
one by discarding easy negatives — without inverse-frequency class weights.

Stage 2 compares exactly three objectives (Audit 0047):

====================== ==================================== =================
recipe                 reduction                            Stage-2 candidate
====================== ==================================== =================
reference_ce           batch-global valid-cell mean         yes (baseline)
group_balanced_ce      0.5*positive_mean + 0.5*background   yes
hard_negative_ce       positives + top-K hardest negatives  yes
reference_ce_resampled batch-global valid-cell mean         no - data order
                                                            only; kept for
                                                            Stage 3 and full
====================== ==================================== =================

`balanced_focal` was retired: a completed 200-epoch run with full warmup
produced zero predicted mentions.

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
GROUP_BALANCED_CE = "group_balanced_ce"
HARD_NEGATIVE_CE = "hard_negative_ce"

# Every defined recipe. Deliberately closed; a fifth is a new milestone.
RECIPE_NAMES: tuple[str, ...] = (
    REFERENCE_CE, REFERENCE_CE_RESAMPLED, GROUP_BALANCED_CE, HARD_NEGATIVE_CE)

# ---------------------------------------------------------------------------
# What Stage 2 actually compares (Audit 0047)
# ---------------------------------------------------------------------------
#
# The completed 200-epoch ablation changed this set on evidence:
#
# * `balanced_focal` is REMOVED. It ran the full bound, served all 60 warmup
#   steps, reached peak learning rates, and finished with zero predicted
#   mentions and zero positive-cell accuracy. Its background loss fell to
#   0.000521 while its positive loss sat at 0.468997 — it optimized the easy
#   majority and never produced a relation. A completed, valid failure, so the
#   candidate is retired rather than carried along.
#
# * `reference_ce_resampled` is NOT a Stage-2 objective. It differs from
#   `reference_ce` only in DATA ORDER, and the tiny set is entirely positive
#   examples, so zero-entity resampling has nothing to reorder. Comparing it
#   here would compare a recipe against itself. It stays defined and available
#   for Stage 3 and full training, where the ordering is real.
STAGE2_RECIPE_NAMES: tuple[str, ...] = (
    REFERENCE_CE, GROUP_BALANCED_CE, HARD_NEGATIVE_CE)

# Retired candidate, named so a stale gate or config referencing it is caught.
RETIRED_RECIPE_NAMES: tuple[str, ...] = ("balanced_focal",)

# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
#
# All three are batch-global: numerators and counts accumulate across every
# microbatch of an effective batch and are divided ONCE at the end. None is the
# per-example mean Audit 0044 measured.
BATCH_GLOBAL_VALID_CELL_MEAN = "batch_global_valid_cell_mean"
BATCH_GLOBAL_GROUP_BALANCED_MEAN = "batch_global_group_balanced_mean"
BATCH_GLOBAL_SELECTED_CELL_MEAN = "batch_global_selected_cell_mean"
BATCH_GLOBAL_REDUCTIONS: tuple[str, ...] = (
    BATCH_GLOBAL_VALID_CELL_MEAN,
    BATCH_GLOBAL_GROUP_BALANCED_MEAN,
    BATCH_GLOBAL_SELECTED_CELL_MEAN,
)

# group_balanced_ce defaults. Equal group weight, NOT inverse frequency: the two
# group *means* are combined, so a 577:1 imbalance changes neither weight.
DEFAULT_POSITIVE_GROUP_WEIGHT = 0.5
DEFAULT_BACKGROUND_GROUP_WEIGHT = 0.5

# hard_negative_ce defaults.
DEFAULT_NEGATIVE_TO_POSITIVE_RATIO = 3
# When an effective batch has no positive cell at all, keep this many
# highest-loss background cells — not all of them, and not none.
DEFAULT_NO_POSITIVE_BACKGROUND_CAP = 32

# Complexity order, used to break an exact tie in favour of the simpler recipe.
RECIPE_COMPLEXITY: Mapping[str, int] = {
    REFERENCE_CE: 0,
    REFERENCE_CE_RESAMPLED: 1,
    GROUP_BALANCED_CE: 2,
    HARD_NEGATIVE_CE: 3,
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

# A guard, not a tuning knob: any positive:background weight ratio at or above
# this is the inverse-frequency regime that turns background into noise. It
# applies to every objective that weights the two groups.
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

    def positive_mean(self) -> float:
        """Mean CE over every valid non-NONE gold cell in the effective batch."""
        if self.positive_cells == 0:
            raise RecipeError("the effective batch has no positive cell")
        return self.positive_numerator / self.positive_cells

    def background_mean(self) -> float:
        """Mean CE over every valid NONE gold cell in the effective batch."""
        if self.background_cells == 0:
            raise RecipeError("the effective batch has no background cell")
        return self.background_numerator / self.background_cells

    def group_balanced(
        self, *,
        positive_weight: float = DEFAULT_POSITIVE_GROUP_WEIGHT,
        background_weight: float = DEFAULT_BACKGROUND_GROUP_WEIGHT,
    ) -> float:
        """``w_p * positive_mean + w_b * background_mean``, one division each.

        Averaging the two groups *before* weighting is what gives the 44,340
        positive cells the same influence as the 25.6 million background cells
        without any inverse-frequency weight appearing anywhere.

        An effective batch with no positive cell falls back to background only —
        explicitly, because there is no positive mean to combine, and silently
        returning 0.0 would hand the optimizer a free batch.
        """
        if self.valid_cells == 0:
            raise RecipeError("cannot reduce an effective batch with no valid cells")
        if self.positive_cells == 0:
            return self.background_mean()
        if self.background_cells == 0:
            return self.positive_mean()
        return (positive_weight * self.positive_mean()
                + background_weight * self.background_mean())

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


@dataclass(frozen=True, slots=True)
class GroupWeights:
    """Weights for the positive and background *group means*.

    Combining two means is not the same as weighting two classes by inverse
    frequency: a 577:1 imbalance changes neither weight here, because each group
    is averaged before it is weighted.
    """

    positive: float = DEFAULT_POSITIVE_GROUP_WEIGHT
    background: float = DEFAULT_BACKGROUND_GROUP_WEIGHT

    def __post_init__(self) -> None:
        if self.positive <= 0.0 or self.background <= 0.0:
            raise RecipeError("both group weights must be positive")
        ratio = self.positive / self.background
        if ratio >= REFUSED_CLASS_WEIGHT_RATIO:
            raise RecipeError(
                f"positive:background group weight ratio {ratio:.1f} is at or "
                "beyond the raw inverse-frequency regime, which is refused")

    def as_dict(self) -> dict[str, Any]:
        return {
            "positive": self.positive,
            "background": self.background,
            "positive_to_background_ratio": self.positive / self.background,
            "is_inverse_frequency_weighting": False,
        }


@dataclass(frozen=True, slots=True)
class HardNegativeConfig:
    """Bounded hard-negative mining. Every positive cell always participates."""

    negative_to_positive_ratio: int = DEFAULT_NEGATIVE_TO_POSITIVE_RATIO
    no_positive_background_cap: int = DEFAULT_NO_POSITIVE_BACKGROUND_CAP

    def __post_init__(self) -> None:
        if self.negative_to_positive_ratio < 1:
            raise RecipeError("negative_to_positive_ratio must be at least 1")
        if self.negative_to_positive_ratio > 50:
            raise RecipeError(
                "an unbounded negative ratio is natural-frequency CE by another "
                "name; keep it small enough to matter")
        if self.no_positive_background_cap < 1:
            raise RecipeError("no_positive_background_cap must be at least 1")

    def keep_count(self, *, positive_cells: int, background_cells: int) -> int:
        """How many background cells to retain. Explicit in both branches."""
        if positive_cells > 0:
            wanted = positive_cells * self.negative_to_positive_ratio
        else:
            # No positive cell anywhere in this batch. Falling back to *all*
            # background would restore natural-frequency CE for exactly the
            # batches where it does most damage; falling back to none would drop
            # the batch silently. A bounded slice of the hardest negatives is
            # the explicit middle.
            wanted = self.no_positive_background_cap
        return min(wanted, background_cells)

    def as_dict(self) -> dict[str, Any]:
        return {
            "negative_to_positive_ratio": self.negative_to_positive_ratio,
            "no_positive_background_cap": self.no_positive_background_cap,
            "every_positive_cell_participates": True,
            "falls_back_to_all_cell_ce": False,
        }


def select_hard_negatives(
    background_cells: Sequence[tuple[float, int, int]], *, keep: int,
) -> tuple[tuple[float, int, int], ...]:
    """The ``keep`` highest-loss background cells, deterministically.

    Ordered by ``(-loss, row, column)``. The positional tiebreak makes the
    selection reproducible without consulting any RNG at all, so it is stable
    across processes and does not merely depend on the recorded seed.
    """
    if keep < 0:
        raise RecipeError("keep must be non-negative")
    ordered = sorted(background_cells, key=lambda cell: (-cell[0], cell[1], cell[2]))
    return tuple(ordered[:keep])


def reduce_grid(
    logits: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[Sequence[int]],
    pair_mask: Sequence[Sequence[bool]],
    *,
    objective: str,
    hard_negative: HardNegativeConfig | None = None,
    background_id: int = 0,
) -> tuple[float, int, float, int]:
    """One grid's ``(loss_sum, cells, positive_loss_sum, positive_cells)``.

    Returns **sums**, never a mean. :class:`BatchGlobalAccumulator` performs the
    single division per effective batch. Any function here returning a mean
    would reintroduce the per-example normalization Audit 0044 measured.

    For ``hard_negative_ce`` the returned ``cells`` counts the positives plus the
    *selected* negatives, so the accumulator's mean is over exactly the cells the
    objective kept.
    """
    if objective not in ("cross_entropy", "hard_negative"):
        raise RecipeError(f"unknown objective {objective!r}")
    if len(logits) != len(labels) or len(labels) != len(pair_mask):
        raise RecipeError("logits, labels and pair mask must have the same rows")

    positive_sum = 0.0
    positive_cells = 0
    background: list[tuple[float, int, int]] = []
    for row_index, (logit_row, label_row, mask_row) in enumerate(
            zip(logits, labels, pair_mask, strict=True)):
        if len(logit_row) != len(label_row) or len(label_row) != len(mask_row):
            raise RecipeError("a grid row has inconsistent width")
        for column_index, (scores, label_id, valid) in enumerate(
                zip(logit_row, label_row, mask_row, strict=True)):
            if not valid:
                continue
            label = int(label_id)
            value = cross_entropy_cell(scores, label)
            if label != background_id:
                positive_sum += value
                positive_cells += 1
            else:
                background.append((value, row_index, column_index))

    if positive_cells == 0 and not background:
        raise RecipeError("grid reduction saw no valid cell")

    if objective == "cross_entropy":
        loss_sum = positive_sum + sum(cell[0] for cell in background)
        return loss_sum, positive_cells + len(background), positive_sum, positive_cells

    config = hard_negative or HardNegativeConfig()
    keep = config.keep_count(
        positive_cells=positive_cells, background_cells=len(background))
    selected = select_hard_negatives(background, keep=keep)
    loss_sum = positive_sum + sum(cell[0] for cell in selected)
    return loss_sum, positive_cells + len(selected), positive_sum, positive_cells


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
    group_weights: GroupWeights | None = None
    hard_negative: HardNegativeConfig | None = None
    contract_version: str = RECIPE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.name not in RECIPE_NAMES:
            raise RecipeError(
                f"unknown recipe {self.name!r}; expected one of {RECIPE_NAMES}")
        if self.reduction not in BATCH_GLOBAL_REDUCTIONS:
            raise RecipeError(
                f"reduction {self.reduction!r} is not batch-global; the "
                "per-example mean is the defect this milestone removed")
        if (self.reduction == BATCH_GLOBAL_GROUP_BALANCED_MEAN
                and self.group_weights is None):
            raise RecipeError("a group-balanced recipe requires explicit GroupWeights")
        if self.objective == "hard_negative" and self.hard_negative is None:
            raise RecipeError(
                "a hard-negative recipe requires an explicit HardNegativeConfig")
        if self.objective != "hard_negative" and self.hard_negative is not None:
            raise RecipeError("only a hard-negative recipe carries a HardNegativeConfig")

    @property
    def uses_positive_aware_sampling(self) -> bool:
        return self.data_order == "positive_aware_resampled"

    @property
    def is_stage2_objective(self) -> bool:
        return self.name in STAGE2_RECIPE_NAMES

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
            objective=self.objective, hard_negative=self.hard_negative)

    def reduce_batch(self, accumulator: BatchGlobalAccumulator) -> float:
        """Reduce one effective batch. One division per group, at the end."""
        if self.reduction == BATCH_GLOBAL_GROUP_BALANCED_MEAN:
            weights = self.group_weights or GroupWeights()
            return accumulator.group_balanced(
                positive_weight=weights.positive,
                background_weight=weights.background)
        return accumulator.reduced()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": self.contract_version,
            "name": self.name,
            "objective": self.objective,
            "reduction": self.reduction,
            "data_order": self.data_order,
            "positive_aware_sampling": self.uses_positive_aware_sampling,
            "stage2_objective": self.is_stage2_objective,
            "optimizer": self.optimizer.as_dict(),
            "schedule": self.schedule.as_dict(),
            "per_example_mean_used": False,
            "every_positive_cell_participates": True,
        }
        if self.group_weights is not None:
            payload["group_weights"] = self.group_weights.as_dict()
        if self.hard_negative is not None:
            payload["hard_negative"] = self.hard_negative.as_dict()
        return payload


def build_recipe(name: str, **overrides: Any) -> Recipe:
    """Construct a recipe by name."""
    if name in RETIRED_RECIPE_NAMES:
        raise RecipeError(
            f"recipe {name!r} was retired by Audit 0047 after a completed "
            "200-epoch run produced zero predicted mentions")
    if name not in RECIPE_NAMES:
        raise RecipeError(f"unknown recipe {name!r}; expected one of {RECIPE_NAMES}")
    optimizer = overrides.pop("optimizer", None) or OptimizerGroups()
    schedule = overrides.pop("schedule", None) or ScheduleConfig()
    group_weights = overrides.pop("group_weights", None)
    hard_negative = overrides.pop("hard_negative", None)
    if overrides:
        raise RecipeError(f"unsupported recipe overrides: {sorted(overrides)}")

    if name == REFERENCE_CE:
        return Recipe(
            name=name, objective="cross_entropy",
            reduction=BATCH_GLOBAL_VALID_CELL_MEAN,
            data_order="shuffled_source_interleaved",
            optimizer=optimizer, schedule=schedule)
    if name == REFERENCE_CE_RESAMPLED:
        return Recipe(
            name=name, objective="cross_entropy",
            reduction=BATCH_GLOBAL_VALID_CELL_MEAN,
            data_order="positive_aware_resampled",
            optimizer=optimizer, schedule=schedule)
    if name == GROUP_BALANCED_CE:
        return Recipe(
            name=name, objective="cross_entropy",
            reduction=BATCH_GLOBAL_GROUP_BALANCED_MEAN,
            data_order="shuffled_source_interleaved",
            optimizer=optimizer, schedule=schedule,
            group_weights=group_weights or GroupWeights())
    return Recipe(
        name=name, objective="hard_negative",
        reduction=BATCH_GLOBAL_SELECTED_CELL_MEAN,
        data_order="shuffled_source_interleaved",
        optimizer=optimizer, schedule=schedule,
        hard_negative=hard_negative or HardNegativeConfig())


def stage2_recipes() -> tuple[Recipe, ...]:
    """Exactly the three objectives Stage 2 compares, in ablation order."""
    return tuple(build_recipe(name) for name in STAGE2_RECIPE_NAMES)


def all_recipes() -> tuple[Recipe, ...]:
    """Every defined recipe, Stage-2 and otherwise."""
    return tuple(build_recipe(name) for name in RECIPE_NAMES)


__all__ = [
    "BATCH_GLOBAL_GROUP_BALANCED_MEAN",
    "BATCH_GLOBAL_REDUCTIONS",
    "BATCH_GLOBAL_SELECTED_CELL_MEAN",
    "BATCH_GLOBAL_VALID_CELL_MEAN",
    "DEFAULT_BACKBONE_LR",
    "DEFAULT_BACKGROUND_GROUP_WEIGHT",
    "DEFAULT_HEAD_LR",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_NEGATIVE_TO_POSITIVE_RATIO",
    "DEFAULT_NO_POSITIVE_BACKGROUND_CAP",
    "DEFAULT_POSITIVE_GROUP_WEIGHT",
    "DEFAULT_WARMUP_RATIO",
    "GROUP_BALANCED_CE",
    "HARD_NEGATIVE_CE",
    "RECIPE_COMPLEXITY",
    "RECIPE_CONTRACT_VERSION",
    "RECIPE_NAMES",
    "REFERENCE_CE",
    "REFERENCE_CE_RESAMPLED",
    "REFUSED_CLASS_WEIGHT_RATIO",
    "RETIRED_RECIPE_NAMES",
    "STAGE2_RECIPE_NAMES",
    "BatchGlobalAccumulator",
    "GroupWeights",
    "HardNegativeConfig",
    "OptimizerGroups",
    "Recipe",
    "RecipeError",
    "ScheduleConfig",
    "all_recipes",
    "build_recipe",
    "cross_entropy_cell",
    "microbatch_scale",
    "reduce_grid",
    "select_hard_negatives",
    "stage2_recipes",
]
