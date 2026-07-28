"""E4 tiny-overfit diagnostic contracts (Audit 0043).

A single decisive experiment: take a handful of governed training examples that
really do contain entities, train on **only** those, and evaluate on the very
same examples. A correctly wired W2NER pipeline must be able to memorize them —
exact mention F1 close to 1.0 is the expected outcome, not a good one.

The point is to separate two failures that look identical from the outside:

* training loss falls to ~0 **and** exact mention F1 stays 0 — the target, the
  loss and the decoder disagree about what the model is being asked to produce,
  and no amount of data or class weighting will fix it;
* training loss falls **and** exact mention F1 rises to ~1.0 — the wiring is
  correct, and the full-run collapse is an optimization/imbalance problem on the
  real corpus rather than a broken contract.

Because grid accuracy is ~99.8% for a model that predicts background everywhere,
this module refuses to report a single accuracy number: :class:`TinyOverfitScore`
reports background accuracy, **positive-cell accuracy** and decoded exact
mention F1 as three separate quantities.

Nothing here trains. It builds the deterministic example selection, the scoring,
the stop rule and the artifact-safety guards; the Colab notebook executes them,
and only after an explicit authorization string is set by an operator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...mention_factory.w2ner import (
    EntitySpan,
    W2NERLabelVocab,
    build_w2ner_grid,
    decode_w2ner_grid,
    tokenize_atomic_words,
)
from .e4_collapse_diagnosis import (
    E4_FULL_ARTIFACT_NAME,
    GovernedExample,
    assert_split_allowed,
)

TINY_OVERFIT_VERSION = "e4-tiny-overfit-diagnostic-v1"
TINY_OVERFIT_STAGE_ID = "phase2-e4-tiny-overfit-diagnostic-v1"
TINY_OVERFIT_EXPERT_ID = "E4_phobert_w2ner"
TINY_OVERFIT_MODE = "diagnostic"

# An operator must paste this before the diagnostic trains anything. The notebook
# is committed with the flag off and the string empty.
TINY_OVERFIT_AUTHORIZATION = "I_AUTHORIZE_E4_TINY_OVERFIT_DIAGNOSTIC"

# A separate artifact directory. The completed full run's artifact is immutable
# evidence and must never be a write target for a diagnostic.
TINY_OVERFIT_ARTIFACT_NAME = "e4_tiny_overfit_diagnostic_v1"
PROTECTED_ARTIFACT_NAMES: frozenset[str] = frozenset({
    E4_FULL_ARTIFACT_NAME,
    "e4_phobert_w2ner_smoke_v1",
    "e4_phobert_w2ner_smoke_v2",
})

# Bounds. The experiment is meant to be cheap and to terminate.
TINY_OVERFIT_MIN_EXAMPLES = 8
TINY_OVERFIT_MAX_EXAMPLES = 16
TINY_OVERFIT_DEFAULT_EXAMPLES = 12
# Keep the O(n^2) grids small so the whole diagnostic fits on any CUDA runtime.
TINY_OVERFIT_MAX_ATOMIC_WORDS = 64
# Stop as soon as the pipeline has proved it can memorize the set...
TINY_OVERFIT_TARGET_EXACT_F1 = 0.95
# ...and stop regardless after this many passes, so a broken pipeline cannot burn
# a Colab session proving the same negative over and over.
TINY_OVERFIT_MAX_EPOCHS = 200
TINY_OVERFIT_EVALUATE_EVERY_N_EPOCHS = 5

# The three types the governed corpus actually supervises. TEST_NAME and
# TEST_RESULT have zero instances anywhere in train or validation, so requiring
# them would make the selection unsatisfiable; that absence is a corpus finding
# recorded in Audit 0043, not something the diagnostic works around silently.
TINY_OVERFIT_REQUIRED_TYPES: tuple[str, ...] = ("DIAGNOSIS", "SYMPTOM", "MEDICATION")


class TinyOverfitError(ValueError):
    """Raised when the tiny-overfit diagnostic is configured unsafely."""


def assert_artifact_dir_is_not_protected(artifact_dir: str | Path) -> Path:
    """Refuse any write target that is a completed E4 run's artifact directory."""
    path = Path(artifact_dir)
    for part in path.parts:
        if part in PROTECTED_ARTIFACT_NAMES:
            raise TinyOverfitError(
                f"refusing to write a diagnostic into the immutable artifact "
                f"directory {part!r}; use a separate directory such as "
                f"{TINY_OVERFIT_ARTIFACT_NAME}")
    return path


def assert_tiny_overfit_authorized(confirmation: str, *, enabled: bool) -> None:
    """The diagnostic trains only when both the flag and the string are set."""
    if not enabled:
        raise TinyOverfitError(
            "the E4 tiny-overfit diagnostic is disabled; it is committed in the "
            "not-authorized state and an operator must enable it explicitly")
    if confirmation != TINY_OVERFIT_AUTHORIZATION:
        raise TinyOverfitError(
            "the E4 tiny-overfit diagnostic requires the exact authorization "
            f"string {TINY_OVERFIT_AUTHORIZATION!r}")


@dataclass(frozen=True, slots=True)
class TinyOverfitSelection:
    """The deterministic example set, described without any clinical text."""

    split: str
    row_indices: tuple[int, ...]
    document_ids: tuple[str, ...]
    entity_count: int
    entities_by_type: Mapping[str, int]
    covered_required_types: tuple[str, ...]
    missing_required_types: tuple[str, ...]
    atomic_words_by_example: tuple[int, ...]
    selection_version: str = TINY_OVERFIT_VERSION

    @property
    def example_count(self) -> int:
        return len(self.row_indices)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selection_version": self.selection_version,
            "split": self.split,
            "example_count": self.example_count,
            "row_indices": list(self.row_indices),
            "document_ids": list(self.document_ids),
            "entity_count": self.entity_count,
            "entities_by_type": dict(self.entities_by_type),
            "covered_required_types": list(self.covered_required_types),
            "missing_required_types": list(self.missing_required_types),
            "atomic_words_by_example": list(self.atomic_words_by_example),
            "internal_test_accessed": False,
        }


def select_tiny_overfit_examples(
    examples: Iterable[GovernedExample],
    *,
    split: str = "train",
    target_size: int = TINY_OVERFIT_DEFAULT_EXAMPLES,
    required_types: Sequence[str] = TINY_OVERFIT_REQUIRED_TYPES,
    max_atomic_words: int = TINY_OVERFIT_MAX_ATOMIC_WORDS,
) -> TinyOverfitSelection:
    """Pick a small, deterministic, entity-bearing training subset.

    Selection is a single pass in governed file order with no randomness and no
    seed, so the same corpus always yields the same set:

    1. reserve a per-type quota and take the earliest eligible example that
       covers a required type still under quota — this is what guarantees all
       three supervised types appear, since no single governed example carries
       DIAGNOSIS, SYMPTOM and MEDICATION together;
    2. fill any remaining slots with the earliest eligible examples not already
       taken.

    Eligible means: at least one entity, every entity representable on the atomic
    word surface, and at most ``max_atomic_words`` atomic words. An example whose
    entity does not align is skipped rather than repaired — repairing it here
    would smuggle the very failure mode this diagnostic is meant to detect.
    """
    assert_split_allowed(split)
    if not (TINY_OVERFIT_MIN_EXAMPLES <= target_size <= TINY_OVERFIT_MAX_EXAMPLES):
        raise TinyOverfitError(
            f"target_size must be between {TINY_OVERFIT_MIN_EXAMPLES} and "
            f"{TINY_OVERFIT_MAX_EXAMPLES}, got {target_size}")
    if not required_types:
        raise TinyOverfitError("at least one required entity type is needed")

    eligible: list[tuple[GovernedExample, int, frozenset[str]]] = []
    for example in examples:
        if not example.entities:
            continue
        words = tokenize_atomic_words(example.text)
        if not words or len(words) > max_atomic_words:
            continue
        starts = {word.start for word in words}
        ends = {word.end for word in words}
        if any(entity.start not in starts or entity.end not in ends
               for entity in example.entities):
            continue
        eligible.append((
            example, len(words),
            frozenset(entity.entity_type for entity in example.entities)))

    quota = max(1, target_size // len(required_types))
    taken: dict[int, tuple[GovernedExample, int]] = {}
    per_type: dict[str, int] = dict.fromkeys(required_types, 0)

    for entity_type in required_types:
        for example, word_count, types in eligible:
            if len(taken) >= target_size or per_type[entity_type] >= quota:
                break
            if example.row_index in taken or entity_type not in types:
                continue
            taken[example.row_index] = (example, word_count)
            for present in types:
                if present in per_type:
                    per_type[present] += 1

    for example, word_count, types in eligible:
        if len(taken) >= target_size:
            break
        if example.row_index in taken:
            continue
        taken[example.row_index] = (example, word_count)
        for present in types:
            if present in per_type:
                per_type[present] += 1

    if len(taken) < TINY_OVERFIT_MIN_EXAMPLES:
        raise TinyOverfitError(
            f"only {len(taken)} eligible governed examples were found; the "
            f"diagnostic needs at least {TINY_OVERFIT_MIN_EXAMPLES}")

    ordered = [taken[row] for row in sorted(taken)]
    entities_by_type: dict[str, int] = {}
    entity_total = 0
    for example, _word_count in ordered:
        for entity in example.entities:
            entities_by_type[entity.entity_type] = (
                entities_by_type.get(entity.entity_type, 0) + 1)
            entity_total += 1
    covered = tuple(t for t in required_types if entities_by_type.get(t, 0) > 0)
    missing = tuple(t for t in required_types if entities_by_type.get(t, 0) == 0)
    return TinyOverfitSelection(
        split=split,
        row_indices=tuple(example.row_index for example, _ in ordered),
        document_ids=tuple(example.document_id for example, _ in ordered),
        entity_count=entity_total,
        entities_by_type=dict(sorted(entities_by_type.items())),
        covered_required_types=covered,
        missing_required_types=missing,
        atomic_words_by_example=tuple(word_count for _, word_count in ordered),
    )


@dataclass(frozen=True, slots=True)
class TinyOverfitScore:
    """Three separate signals. A single "accuracy" number would hide the failure."""

    epoch: int
    mean_training_loss: float
    grid_cells: int
    grid_cells_correct: int
    positive_cells: int
    positive_cells_correct: int
    background_cells: int
    background_cells_correct: int
    gold_mentions: int
    predicted_mentions: int
    true_positives: int

    @property
    def grid_cell_accuracy(self) -> float:
        return self.grid_cells_correct / self.grid_cells if self.grid_cells else 0.0

    @property
    def positive_cell_accuracy(self) -> float:
        """The number that actually moves. Near 0 while grid accuracy is ~0.998
        is the exact signature of an all-background solution."""
        return (self.positive_cells_correct / self.positive_cells
                if self.positive_cells else 0.0)

    @property
    def background_cell_accuracy(self) -> float:
        return (self.background_cells_correct / self.background_cells
                if self.background_cells else 0.0)

    @property
    def exact_precision(self) -> float:
        return (self.true_positives / self.predicted_mentions
                if self.predicted_mentions else 0.0)

    @property
    def exact_recall(self) -> float:
        return self.true_positives / self.gold_mentions if self.gold_mentions else 0.0

    @property
    def exact_f1(self) -> float:
        precision, recall = self.exact_precision, self.exact_recall
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "tiny_overfit_evaluation",
            "epoch": self.epoch,
            "mean_training_loss": self.mean_training_loss,
            "grid_cells": self.grid_cells,
            "grid_cell_accuracy": self.grid_cell_accuracy,
            "positive_cells": self.positive_cells,
            "positive_cell_accuracy": self.positive_cell_accuracy,
            "background_cells": self.background_cells,
            "background_cell_accuracy": self.background_cell_accuracy,
            "gold_mentions": self.gold_mentions,
            "predicted_mentions": self.predicted_mentions,
            "true_positives": self.true_positives,
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "exact_f1": self.exact_f1,
            "internal_test_accessed": False,
        }


def score_predicted_grid(
    *,
    epoch: int,
    mean_training_loss: float,
    target_grids: Sequence[Sequence[Sequence[int]]],
    predicted_grids: Sequence[Sequence[Sequence[int]]],
    gold_mention_sets: Sequence[Iterable[tuple[int, int, str]]],
    predicted_mention_sets: Sequence[Iterable[tuple[int, int, str]]],
    background_label_id: int = 0,
) -> TinyOverfitScore:
    """Score argmax label grids against targets, splitting positive from background."""
    if len(target_grids) != len(predicted_grids):
        raise TinyOverfitError("target and predicted grid counts differ")
    if len(gold_mention_sets) != len(predicted_mention_sets):
        raise TinyOverfitError("gold and predicted mention-set counts differ")

    cells = correct = 0
    positive = positive_correct = 0
    background = background_correct = 0
    for target_grid, predicted_grid in zip(target_grids, predicted_grids, strict=True):
        if len(target_grid) != len(predicted_grid):
            raise TinyOverfitError("target and predicted grids differ in size")
        for target_row, predicted_row in zip(target_grid, predicted_grid, strict=True):
            if len(target_row) != len(predicted_row):
                raise TinyOverfitError("target and predicted grid rows differ in width")
            for target, predicted in zip(target_row, predicted_row, strict=True):
                cells += 1
                matched = int(target) == int(predicted)
                correct += int(matched)
                if int(target) == background_label_id:
                    background += 1
                    background_correct += int(matched)
                else:
                    positive += 1
                    positive_correct += int(matched)

    gold_total = predicted_total = true_positives = 0
    for gold_mentions, predicted_mentions in zip(
            gold_mention_sets, predicted_mention_sets, strict=True):
        gold_set = set(gold_mentions)
        predicted_set = set(predicted_mentions)
        gold_total += len(gold_set)
        predicted_total += len(predicted_set)
        true_positives += len(gold_set & predicted_set)

    return TinyOverfitScore(
        epoch=epoch,
        mean_training_loss=mean_training_loss,
        grid_cells=cells,
        grid_cells_correct=correct,
        positive_cells=positive,
        positive_cells_correct=positive_correct,
        background_cells=background,
        background_cells_correct=background_correct,
        gold_mentions=gold_total,
        predicted_mentions=predicted_total,
        true_positives=true_positives,
    )


def build_tiny_overfit_targets(
    example: GovernedExample, *, vocab: W2NERLabelVocab | None = None,
) -> tuple[tuple[tuple[int, ...], ...], frozenset[tuple[int, int, str]]]:
    """Target grid plus the gold mention set the decoder must reproduce."""
    label_vocab = vocab or W2NERLabelVocab()
    words = tokenize_atomic_words(example.text)
    grid = build_w2ner_grid(
        example.document_id, example.text, example.entities,
        words=words, vocab=label_vocab)
    gold = frozenset(
        (span.start, span.end, span.entity_type) for span in decode_w2ner_grid(grid))
    return grid.labels, gold


def should_stop_tiny_overfit(
    *,
    epoch: int,
    exact_f1: float,
    target_exact_f1: float = TINY_OVERFIT_TARGET_EXACT_F1,
    max_epochs: int = TINY_OVERFIT_MAX_EPOCHS,
) -> tuple[bool, str]:
    """Bounded stop rule: success, exhaustion, or keep going."""
    if epoch < 1:
        raise TinyOverfitError("tiny-overfit epochs are 1-based")
    if exact_f1 >= target_exact_f1:
        return True, "reached_target_exact_f1"
    if epoch >= max_epochs:
        return True, "reached_max_epochs_without_target_exact_f1"
    return False, "continue"


def build_tiny_overfit_resolved_config(
    *,
    selection: TinyOverfitSelection,
    model_revision: str,
    tokenizer_revision: str,
    seed: int,
    learning_rate: float,
    max_epochs: int = TINY_OVERFIT_MAX_EPOCHS,
    target_exact_f1: float = TINY_OVERFIT_TARGET_EXACT_F1,
    evaluate_every_n_epochs: int = TINY_OVERFIT_EVALUATE_EVERY_N_EPOCHS,
    precision_mode: str = "fp32",
    device_type: str = "cpu",
) -> dict[str, Any]:
    """Resolved config for the diagnostic. Records what it is NOT, too."""
    if selection.example_count < TINY_OVERFIT_MIN_EXAMPLES:
        raise TinyOverfitError("tiny-overfit selection is below the minimum size")
    if evaluate_every_n_epochs <= 0:
        raise TinyOverfitError("evaluate_every_n_epochs must be positive")
    return {
        "tiny_overfit_version": TINY_OVERFIT_VERSION,
        "stage_id": TINY_OVERFIT_STAGE_ID,
        "expert_id": TINY_OVERFIT_EXPERT_ID,
        "mode": TINY_OVERFIT_MODE,
        "model_id": "vinai/phobert-large",
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "seed": seed,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "target_exact_f1": target_exact_f1,
        "evaluate_every_n_epochs": evaluate_every_n_epochs,
        "precision_mode": precision_mode,
        "precision_device_type": device_type,
        "selection": selection.as_dict(),
        "train_split_only": True,
        "evaluates_on_the_training_subset_by_design": True,
        "is_a_quality_result": False,
        "produces_a_deployable_checkpoint": False,
        "may_initialize_a_full_run": False,
        "internal_test_accessed": False,
        "organizer_inference_performed": False,
    }


@dataclass(frozen=True, slots=True)
class TinyOverfitOutcome:
    """Verdict of one completed diagnostic run."""

    stopped_reason: str
    epochs_run: int
    final_exact_f1: float
    final_positive_cell_accuracy: float
    final_grid_cell_accuracy: float
    final_mean_training_loss: float

    @property
    def pipeline_can_memorize(self) -> bool:
        return self.final_exact_f1 >= TINY_OVERFIT_TARGET_EXACT_F1

    @property
    def interpretation(self) -> str:
        if self.pipeline_can_memorize:
            return (
                "the target/loss/decoder pipeline is coherent: it memorized the "
                "subset, so the full-run collapse is not a wiring defect")
        if self.final_mean_training_loss <= 0.001 and self.final_exact_f1 == 0.0:
            return (
                "training loss reached ~0 while exact mention F1 stayed 0: the "
                "loss is being minimized without producing decodable mentions, "
                "which is a target/loss/decoder inconsistency")
        return (
            "the subset was not memorized within the epoch bound; the result is "
            "inconclusive on its own and must not be reported as a root cause")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "tiny_overfit_outcome",
            "stopped_reason": self.stopped_reason,
            "epochs_run": self.epochs_run,
            "final_exact_f1": self.final_exact_f1,
            "final_positive_cell_accuracy": self.final_positive_cell_accuracy,
            "final_grid_cell_accuracy": self.final_grid_cell_accuracy,
            "final_mean_training_loss": self.final_mean_training_loss,
            "pipeline_can_memorize": self.pipeline_can_memorize,
            "interpretation": self.interpretation,
            "internal_test_accessed": False,
        }


def summarize_tiny_overfit(
    scores: Sequence[TinyOverfitScore], *, stopped_reason: str,
) -> TinyOverfitOutcome:
    if not scores:
        raise TinyOverfitError("no tiny-overfit evaluations were recorded")
    final = scores[-1]
    return TinyOverfitOutcome(
        stopped_reason=stopped_reason,
        epochs_run=final.epoch,
        final_exact_f1=final.exact_f1,
        final_positive_cell_accuracy=final.positive_cell_accuracy,
        final_grid_cell_accuracy=final.grid_cell_accuracy,
        final_mean_training_loss=final.mean_training_loss,
    )


def entity_spans(example: GovernedExample) -> tuple[EntitySpan, ...]:
    """Explicit accessor so notebook code never reaches into the dataclass."""
    return example.entities


__all__ = [
    "PROTECTED_ARTIFACT_NAMES",
    "TINY_OVERFIT_ARTIFACT_NAME",
    "TINY_OVERFIT_AUTHORIZATION",
    "TINY_OVERFIT_DEFAULT_EXAMPLES",
    "TINY_OVERFIT_EVALUATE_EVERY_N_EPOCHS",
    "TINY_OVERFIT_MAX_ATOMIC_WORDS",
    "TINY_OVERFIT_MAX_EPOCHS",
    "TINY_OVERFIT_MAX_EXAMPLES",
    "TINY_OVERFIT_MIN_EXAMPLES",
    "TINY_OVERFIT_MODE",
    "TINY_OVERFIT_REQUIRED_TYPES",
    "TINY_OVERFIT_STAGE_ID",
    "TINY_OVERFIT_TARGET_EXACT_F1",
    "TINY_OVERFIT_VERSION",
    "TinyOverfitError",
    "TinyOverfitOutcome",
    "TinyOverfitScore",
    "TinyOverfitSelection",
    "assert_artifact_dir_is_not_protected",
    "assert_tiny_overfit_authorized",
    "build_tiny_overfit_resolved_config",
    "build_tiny_overfit_targets",
    "entity_spans",
    "score_predicted_grid",
    "select_tiny_overfit_examples",
    "should_stop_tiny_overfit",
    "summarize_tiny_overfit",
]
