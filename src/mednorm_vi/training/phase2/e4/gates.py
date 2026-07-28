"""E4 staged authorization gates (Audit 0045).

Four sequential stages, each with its own authorization string and a
default-False run flag:

    1  preflight        dependencies and runtime
    2  tiny ablation    all three recipes on 12 governed examples
    3  subset smoke     the selected recipe on a representative subset
    4  full training    the validated recipe on the governed corpus

The chain **fails closed**. Flipping the full-training flag does not bypass
Stages 2 and 3: :func:`assert_full_training_allowed` requires their saved gate
artifacts, requires each to have *passed*, and requires the code/config/corpus
hashes recorded in them to match the current ones. A gate artifact from a
different configuration is refused rather than reused.

The audited run had no such chain. It went straight to 12 epochs and 405,912
backward passes on a recipe nothing had ever validated end to end.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    E4_FULL_AUTHORIZATION,
    E4_SUBSET_AUTHORIZATION,
    E4_SUPERVISED_TYPES,
    E4_TINY_AUTHORIZATION,
    E4ContractError,
)
from .recipes import RECIPE_COMPLEXITY, RECIPE_NAMES

GATE_CONTRACT_VERSION = "e4-stage-gate-v1"

STAGE_PREFLIGHT = "preflight"
STAGE_TINY = "tiny_recipe_ablation"
STAGE_SUBSET = "subset_smoke"
STAGE_FULL = "full_training"
STAGES: tuple[str, ...] = (STAGE_PREFLIGHT, STAGE_TINY, STAGE_SUBSET, STAGE_FULL)

STAGE_AUTHORIZATIONS: Mapping[str, str] = {
    STAGE_TINY: E4_TINY_AUTHORIZATION,
    STAGE_SUBSET: E4_SUBSET_AUTHORIZATION,
    STAGE_FULL: E4_FULL_AUTHORIZATION,
}

TINY_GATE_FILENAME = "stage2_tiny_ablation.json"
SUBSET_GATE_FILENAME = "stage3_subset_smoke.json"

# Stage-2 pass thresholds.
TINY_TARGET_EXACT_F1 = 0.95
# Stage-3 pass thresholds. Deliberately weak: a subset smoke proves the pipeline
# learns *something* real, not that it is good.
SUBSET_MAX_GOLD_POSITIVE_BACKGROUND_RATE = 0.98


class GateError(E4ContractError):
    """Raised when a stage gate is not satisfied."""


def assert_stage_authorized(stage: str, confirmation: str, *, enabled: bool) -> None:
    """Both the flag and the exact string, or the stage does not run."""
    if stage not in STAGE_AUTHORIZATIONS:
        raise GateError(f"stage {stage!r} has no authorization contract")
    if not enabled:
        raise GateError(
            f"stage {stage!r} is disabled; the notebook is committed with every "
            "run flag False and an operator must enable it explicitly")
    expected = STAGE_AUTHORIZATIONS[stage]
    if confirmation != expected:
        raise GateError(
            f"stage {stage!r} requires the exact authorization string {expected!r}")


# ---------------------------------------------------------------------------
# Stage 2: tiny recipe ablation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecipeResult:
    """One recipe's tiny-overfit outcome. Grid accuracy is never a criterion."""

    recipe: str
    exact_precision: float
    exact_recall: float
    exact_f1: float
    predicted_mentions: int
    gold_mentions: int
    false_positives: int
    positive_cell_accuracy: float
    gold_positive_background_rate: float
    nnw_predictions: int
    thw_predictions_by_type: Mapping[str, int]
    loss_total: float
    loss_positive: float
    loss_background: float
    seconds: float
    peak_vram_gib: float
    # A real fresh-model re-evaluation, or None. There is deliberately no bare
    # boolean: Audit 0046 removed a "reproduction check" that compared two
    # state-dict key names and would have passed for a checkpoint of zeros.
    reproduction: ReproductionCheck | None
    grid_cell_accuracy: float = 0.0
    # Planned vs realized schedule accounting, so a run that stopped inside
    # warmup can never be mistaken for evidence about the recipe.
    schedule: Mapping[str, Any] = field(default_factory=dict)
    stopped_reason: str = ""

    def types_predicted(self) -> tuple[str, ...]:
        return tuple(sorted(
            name for name, count in self.thw_predictions_by_type.items() if count > 0))

    @property
    def save_reload_reproduced(self) -> bool:
        return self.reproduction is not None and self.reproduction.reproduced

    @property
    def stopped_inside_warmup(self) -> bool:
        """A run that never reached its configured learning rates.

        Its numbers describe the schedule it was denied, not the recipe.
        """
        return bool(self.schedule) and not self.schedule.get(
            "peak_learning_rate_reached", True)

    def passes(self, *, required_types: Sequence[str]) -> tuple[bool, tuple[str, ...]]:
        """Pass/fail with every unmet condition named."""
        failures: list[str] = []
        if self.exact_f1 < TINY_TARGET_EXACT_F1:
            failures.append(
                f"exact F1 {self.exact_f1:.4f} < {TINY_TARGET_EXACT_F1}")
        if self.predicted_mentions <= 0:
            failures.append("predicted no mentions")
        if self.positive_cell_accuracy <= 0.0:
            failures.append("positive-cell accuracy is zero")
        missing = [t for t in required_types if t not in self.types_predicted()]
        if missing:
            failures.append(f"never predicted {sorted(missing)}")
        if not self.save_reload_reproduced:
            detail = ("no fresh-model re-evaluation was performed"
                      if self.reproduction is None
                      else "; ".join(self.reproduction.failures()))
            failures.append(f"save/reload did not reproduce the metrics ({detail})")
        return (not failures), tuple(failures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "exact_f1": self.exact_f1,
            "predicted_mentions": self.predicted_mentions,
            "gold_mentions": self.gold_mentions,
            "false_positives": self.false_positives,
            "positive_cell_accuracy": self.positive_cell_accuracy,
            "gold_positive_background_rate": self.gold_positive_background_rate,
            "grid_cell_accuracy": self.grid_cell_accuracy,
            "grid_cell_accuracy_is_a_pass_criterion": False,
            "nnw_predictions": self.nnw_predictions,
            "thw_predictions_by_type": dict(self.thw_predictions_by_type),
            "types_predicted": list(self.types_predicted()),
            "loss_total": self.loss_total,
            "loss_positive": self.loss_positive,
            "loss_background": self.loss_background,
            "seconds": self.seconds,
            "peak_vram_gib": self.peak_vram_gib,
            "save_reload_reproduced": self.save_reload_reproduced,
            "reproduction": (
                self.reproduction.as_dict() if self.reproduction is not None else None),
            "schedule": dict(self.schedule),
            "stopped_reason": self.stopped_reason,
            "stopped_inside_warmup": self.stopped_inside_warmup,
        }


def select_recipe(
    results: Sequence[RecipeResult], *, required_types: Sequence[str],
) -> tuple[RecipeResult | None, dict[str, Any]]:
    """Apply the selection order to the passing recipes.

    1. highest exact F1;
    2. highest exact recall;
    3. fewest false positives;
    4. lower runtime, then lower peak VRAM;
    5. on an exact tie, the simpler recipe.

    ``reference_ce`` is the simplest and therefore wins any exact tie by
    construction — no special case is needed for it.
    """
    if not results:
        raise GateError("the tiny ablation produced no results")
    evaluated = []
    for result in results:
        ok, failures = result.passes(required_types=required_types)
        evaluated.append({
            "recipe": result.recipe, "passed": ok, "failures": list(failures)})
    passing = [
        r for r in results if r.passes(required_types=required_types)[0]]
    report: dict[str, Any] = {
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "stage": STAGE_TINY,
        "recipes_evaluated": [r.as_dict() for r in results],
        "pass_summary": evaluated,
        "required_types": list(required_types),
        "target_exact_f1": TINY_TARGET_EXACT_F1,
        "selection_order": [
            "exact_f1", "exact_recall", "fewest_false_positives",
            "lower_runtime", "lower_peak_vram", "simpler_recipe"],
    }
    if not passing:
        report["selected_recipe"] = None
        report["passed"] = False
        denied = [r.recipe for r in results if r.stopped_inside_warmup]
        report["recipes_stopped_inside_warmup"] = denied
        report["stop_reason"] = (
            "no candidate recipe met the tiny-overfit criteria; subset and full "
            "training must not run")
        if denied:
            # Audit 0046: this is exactly how the first Stage-2 run produced a
            # three-way "failure" that said nothing about the recipes.
            report["result_is_invalid_for_recipe_comparison"] = True
            report["stop_reason"] = (
                f"{denied} stopped before the scheduler reached its configured "
                "learning rates, so this run is NOT evidence about those recipes; "
                "re-run Stage 2 with the corrected stopping policy")
        return None, report
    selected = min(
        passing,
        key=lambda r: (
            -round(r.exact_f1, 9), -round(r.exact_recall, 9), r.false_positives,
            round(r.seconds, 3), round(r.peak_vram_gib, 4),
            RECIPE_COMPLEXITY[r.recipe], r.recipe))
    report["selected_recipe"] = selected.recipe
    report["passed"] = True
    report["passing_recipes"] = [r.recipe for r in passing]
    return selected, report


def render_recipe_table(results: Sequence[RecipeResult]) -> str:
    """A compact Markdown comparison for the audit."""
    header = (
        "| recipe | exact P | exact R | exact F1 | pred | FP | pos-cell acc | "
        "gold-pos bg | NNW | THW | loss+ | loss bg | s | VRAM GiB | reload |")
    rule = "| " + " | ".join(["---"] * 15) + " |"
    rows = [
        f"| {r.recipe} | {r.exact_precision:.4f} | {r.exact_recall:.4f} | "
        f"{r.exact_f1:.4f} | {r.predicted_mentions} | {r.false_positives} | "
        f"{r.positive_cell_accuracy:.4f} | {r.gold_positive_background_rate:.4f} | "
        f"{r.nnw_predictions} | {sum(r.thw_predictions_by_type.values())} | "
        f"{r.loss_positive:.6f} | {r.loss_background:.6f} | {r.seconds:.1f} | "
        f"{r.peak_vram_gib:.2f} | {r.save_reload_reproduced} |"
        for r in results
    ]
    return "\n".join([header, rule, *rows])


# ---------------------------------------------------------------------------
# Stage 3: subset smoke
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubsetResult:
    """The selected recipe's behaviour on a representative governed subset."""

    recipe: str
    validation_predicted_mentions: int
    validation_gold_mentions: int
    validation_recall: float
    validation_exact_f1: float
    nnw_predictions: int
    thw_predictions_by_type: Mapping[str, int]
    gold_positive_background_rate: float
    collapse_guard_fired: bool
    save_reload_reproduced: bool
    artifact_validator_ok: bool

    def types_predicted(self) -> tuple[str, ...]:
        return tuple(sorted(
            name for name, count in self.thw_predictions_by_type.items() if count > 0))

    def passes(self, *, required_types: Sequence[str]) -> tuple[bool, tuple[str, ...]]:
        failures: list[str] = []
        if self.validation_predicted_mentions <= 0:
            failures.append("validation predicted no mentions")
        if self.validation_recall <= 0.0:
            failures.append("validation recall is zero")
        if self.nnw_predictions <= 0:
            failures.append("no NNW relation predicted")
        if sum(self.thw_predictions_by_type.values()) <= 0:
            failures.append("no THW relation predicted")
        missing = [t for t in required_types if t not in self.types_predicted()]
        if missing:
            failures.append(f"never predicted {sorted(missing)}")
        if self.gold_positive_background_rate >= (
                SUBSET_MAX_GOLD_POSITIVE_BACKGROUND_RATE):
            failures.append(
                f"gold-positive background rate "
                f"{self.gold_positive_background_rate:.4f} is not materially below 1")
        if self.collapse_guard_fired:
            failures.append("the collapse guard fired")
        if not self.save_reload_reproduced:
            failures.append("best checkpoint save/reload did not reproduce metrics")
        if not self.artifact_validator_ok:
            failures.append("the artifact validator failed")
        return (not failures), tuple(failures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "validation_predicted_mentions": self.validation_predicted_mentions,
            "validation_gold_mentions": self.validation_gold_mentions,
            "validation_recall": self.validation_recall,
            "validation_exact_f1": self.validation_exact_f1,
            "nnw_predictions": self.nnw_predictions,
            "thw_predictions_by_type": dict(self.thw_predictions_by_type),
            "types_predicted": list(self.types_predicted()),
            "gold_positive_background_rate": self.gold_positive_background_rate,
            "max_gold_positive_background_rate": (
                SUBSET_MAX_GOLD_POSITIVE_BACKGROUND_RATE),
            "collapse_guard_fired": self.collapse_guard_fired,
            "save_reload_reproduced": self.save_reload_reproduced,
            "artifact_validator_ok": self.artifact_validator_ok,
        }


# ---------------------------------------------------------------------------
# Save/reload reproduction (Audit 0046)
# ---------------------------------------------------------------------------
#
# The first Stage-2 implementation "verified" reproduction with
#
#     set(reloaded["model_state"]) == {"base_model", "w2ner_head"}
#
# which proves only that two dictionary keys exist. It would pass for a
# checkpoint full of zeros, for the wrong recipe's weights, or for a payload
# whose tensors never loaded. Reproduction now means: restore into a FRESH
# architecture and re-evaluate the same examples.

# Metrics that must reproduce, and the tolerance they must reproduce within.
REPRODUCTION_METRICS: tuple[str, ...] = (
    "exact_precision", "exact_recall", "exact_f1",
    "predicted_mentions", "positive_cell_accuracy",
)
DEFAULT_REPRODUCTION_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class ReproductionCheck:
    """Evidence that a saved checkpoint reproduces its pre-save evaluation.

    Constructing one requires the metrics measured *before* saving and the
    metrics measured *after* restoring into a freshly built model. There is no
    constructor path that accepts "the keys looked right".
    """

    checkpoint_sha256: str
    metrics_before: Mapping[str, float]
    metrics_after: Mapping[str, float]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    fresh_model_evaluated: bool
    examples_evaluated: int
    tolerance: float = DEFAULT_REPRODUCTION_TOLERANCE

    def __post_init__(self) -> None:
        for name, payload in (("metrics_before", self.metrics_before),
                              ("metrics_after", self.metrics_after)):
            missing = [m for m in REPRODUCTION_METRICS if m not in payload]
            if missing:
                raise GateError(
                    f"{name} is missing {sorted(missing)}; a reproduction check "
                    "needs the real evaluated metrics, not a key-name comparison")
        if self.tolerance < 0.0:
            raise GateError("reproduction tolerance must be non-negative")

    def differences(self) -> dict[str, float]:
        return {
            metric: abs(float(self.metrics_after[metric])
                        - float(self.metrics_before[metric]))
            for metric in REPRODUCTION_METRICS
        }

    @property
    def reproduced(self) -> bool:
        if not self.fresh_model_evaluated or self.examples_evaluated <= 0:
            return False
        if self.missing_keys or self.unexpected_keys:
            return False
        return all(delta <= self.tolerance for delta in self.differences().values())

    def failures(self) -> tuple[str, ...]:
        problems: list[str] = []
        if not self.fresh_model_evaluated:
            problems.append("the checkpoint was never restored into a fresh model")
        if self.examples_evaluated <= 0:
            problems.append("no example was re-evaluated after restoration")
        if self.missing_keys:
            problems.append(f"missing state-dict keys: {sorted(self.missing_keys)}")
        if self.unexpected_keys:
            problems.append(
                f"unexpected state-dict keys: {sorted(self.unexpected_keys)}")
        problems.extend(
            f"{metric} drifted by {delta:g} (tolerance {self.tolerance:g})"
            for metric, delta in sorted(self.differences().items())
            if delta > self.tolerance)
        return tuple(problems)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "metrics_before": dict(self.metrics_before),
            "metrics_after": dict(self.metrics_after),
            "differences": self.differences(),
            "tolerance": self.tolerance,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "fresh_model_evaluated": self.fresh_model_evaluated,
            "examples_evaluated": self.examples_evaluated,
            "reproduced": self.reproduced,
            "failures": list(self.failures()),
            "state_dict_key_names_alone_are_sufficient": False,
        }


def assert_real_reproduction(check: ReproductionCheck | None) -> bool:
    """A recipe may only claim reproduction from a real re-evaluation."""
    if check is None:
        raise GateError(
            "save/reload reproduction requires a ReproductionCheck built from a "
            "fresh-model re-evaluation; comparing state-dict key names is not a "
            "reproduction check")
    return check.reproduced


# ---------------------------------------------------------------------------
# Gate artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateArtifact:
    """A saved, hash-bound record that one stage passed."""

    stage: str
    passed: bool
    recipe: str
    config_sha256: str
    code_sha256: str
    corpus_sha256: Mapping[str, str]
    detail: Mapping[str, Any]
    gate_contract_version: str = GATE_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_contract_version": self.gate_contract_version,
            "stage": self.stage,
            "passed": self.passed,
            "recipe": self.recipe,
            "config_sha256": self.config_sha256,
            "code_sha256": self.code_sha256,
            "corpus_sha256": dict(self.corpus_sha256),
            "detail": dict(self.detail),
            "internal_test_accessed": False,
        }

    def write(self, path: str | Path) -> Path:
        """Replace the gate atomically.

        A rerun must never leave a half-written gate, and must never leave the
        previous run's verdict alongside the new one. The temporary file lands
        on the same filesystem so ``os.replace`` is atomic.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".tmp")
        staged.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(staged, target)
        return target


def read_gate_artifact(path: str | Path) -> GateArtifact:
    resolved = Path(path)
    if not resolved.is_file():
        raise GateError(
            f"gate artifact {resolved} does not exist; the stage that writes it "
            "has not run")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("gate_contract_version") != GATE_CONTRACT_VERSION:
        raise GateError(
            f"gate artifact {resolved} was written by contract "
            f"{payload.get('gate_contract_version')!r}, expected "
            f"{GATE_CONTRACT_VERSION!r}")
    return GateArtifact(
        stage=str(payload["stage"]),
        passed=bool(payload["passed"]),
        recipe=str(payload.get("recipe", "")),
        config_sha256=str(payload.get("config_sha256", "")),
        code_sha256=str(payload.get("code_sha256", "")),
        corpus_sha256=dict(payload.get("corpus_sha256", {})),
        detail=dict(payload.get("detail", {})),
    )


def assert_full_training_allowed(
    *,
    gate_dir: str | Path,
    config_sha256: str,
    code_sha256: str,
    corpus_sha256: Mapping[str, str],
    confirmation: str,
    enabled: bool,
) -> str:
    """Every condition for Stage 4, or a named refusal. Returns the recipe.

    Checked in order, so the first unmet condition is the one reported:
    authorization, both gate artifacts exist, both passed, both agree on the
    recipe, and all three hashes match the current run. Flipping the run flag
    satisfies exactly one of those.
    """
    assert_stage_authorized(STAGE_FULL, confirmation, enabled=enabled)
    root = Path(gate_dir)
    tiny = read_gate_artifact(root / TINY_GATE_FILENAME)
    subset = read_gate_artifact(root / SUBSET_GATE_FILENAME)

    for artifact, expected_stage in ((tiny, STAGE_TINY), (subset, STAGE_SUBSET)):
        if artifact.stage != expected_stage:
            raise GateError(
                f"gate artifact declares stage {artifact.stage!r}, expected "
                f"{expected_stage!r}")
        if not artifact.passed:
            raise GateError(
                f"stage {artifact.stage!r} did not pass; full training is refused")
        if artifact.recipe not in RECIPE_NAMES:
            raise GateError(
                f"stage {artifact.stage!r} recorded unknown recipe {artifact.recipe!r}")

    if tiny.recipe != subset.recipe:
        raise GateError(
            f"the tiny ablation selected {tiny.recipe!r} but the subset smoke "
            f"validated {subset.recipe!r}; full training is refused")

    for artifact in (tiny, subset):
        if artifact.config_sha256 != config_sha256:
            raise GateError(
                f"stage {artifact.stage!r} passed under config "
                f"{artifact.config_sha256[:12]}… but the current config is "
                f"{config_sha256[:12]}…; re-run the gated stages")
        if artifact.code_sha256 != code_sha256:
            raise GateError(
                f"stage {artifact.stage!r} passed under code "
                f"{artifact.code_sha256[:12]}… but the current code is "
                f"{code_sha256[:12]}…; re-run the gated stages")
        if dict(artifact.corpus_sha256) != dict(corpus_sha256):
            raise GateError(
                f"stage {artifact.stage!r} passed against a different governed "
                "corpus; re-run the gated stages")
    return tiny.recipe


def required_supervised_types(present_in_data: Sequence[str]) -> tuple[str, ...]:
    """Types a stage must predict: supervised *and* actually present.

    A stage is never failed for not predicting TEST_NAME or TEST_RESULT, which
    the governed corpus does not contain — and never passed by predicting a type
    that was in its data but absent from its predictions.
    """
    present = set(present_in_data)
    return tuple(t for t in E4_SUPERVISED_TYPES if t in present)


__all__ = [
    "GATE_CONTRACT_VERSION",
    "STAGES",
    "STAGE_AUTHORIZATIONS",
    "STAGE_FULL",
    "STAGE_PREFLIGHT",
    "STAGE_SUBSET",
    "STAGE_TINY",
    "SUBSET_GATE_FILENAME",
    "SUBSET_MAX_GOLD_POSITIVE_BACKGROUND_RATE",
    "TINY_GATE_FILENAME",
    "TINY_TARGET_EXACT_F1",
    "DEFAULT_REPRODUCTION_TOLERANCE",
    "REPRODUCTION_METRICS",
    "GateArtifact",
    "GateError",
    "ReproductionCheck",
    "assert_real_reproduction",
    "RecipeResult",
    "SubsetResult",
    "assert_full_training_allowed",
    "assert_stage_authorized",
    "read_gate_artifact",
    "render_recipe_table",
    "required_supervised_types",
    "select_recipe",
]
