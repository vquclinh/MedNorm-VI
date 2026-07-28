"""The current E4 implementation and its gated pipeline (Audit 0045).

The removed implementation collapsed to an input-independent all-background
predictor. These tests lock down the four corrections — batch-global valid-cell
reduction, imbalance-aware recipes, deterministic order, and a real schedule —
plus the fail-closed stage chain that stops a bad recipe on 12 examples instead
of 405,912 backward passes.

Nothing here trains, constructs an optimizer, or calls backward.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import tokenize
from pathlib import Path

import pytest
import yaml

from mednorm_vi.mention_factory.w2ner import W2NERLabelVocab
from mednorm_vi.training.phase2.e4 import (
    RECIPE_NAMES,
    BatchGlobalAccumulator,
    ExampleIndex,
    GateArtifact,
    RecipeResult,
    ReproductionCheck,
    SchedulePlan,
    SubsetResult,
    TinyEpochSignal,
    TinyOverfitStopPolicy,
    ValidationSnapshot,
    all_recipes,
    assert_full_training_allowed,
    assert_stage_authorized,
    build_epoch_order,
    build_recipe,
    evaluate_collapse_guard,
    measure_order,
    plan_gradient_accumulation,
    plan_schedule,
    reject_superseded_checkpoint,
    select_recipe,
)
from mednorm_vi.training.phase2.e4.contracts import (
    E4_CHECKPOINT_SCHEMA_VERSION,
    E4_REJECTED_CHECKPOINT_SCHEMA_VERSIONS,
    E4_SUPERVISED_TYPES,
    E4_UNSUPERVISED_TYPES,
    E4ContractError,
    e4_checkpoint_payload,
)
from mednorm_vi.training.phase2.e4.gates import (
    SUBSET_GATE_FILENAME,
    TINY_GATE_FILENAME,
    TINY_TARGET_EXACT_F1,
    GateError,
    assert_real_reproduction,
    required_supervised_types,
)
from mednorm_vi.training.phase2.e4.recipes import (
    BALANCED_FOCAL,
    REFERENCE_CE,
    REFERENCE_CE_RESAMPLED,
    FocalConfig,
    OptimizerGroups,
    RecipeError,
    ScheduleConfig,
    cross_entropy_cell,
    focal_cell,
    microbatch_scale,
    reduce_grid,
)
from mednorm_vi.training.phase2.e4.sampling import (
    POSITIVE_AWARE_RESAMPLED,
    SHUFFLED_SOURCE_INTERLEAVED,
    SamplingError,
    assert_order_preserves_corpus,
)
from mednorm_vi.training.phase2.e4.training import (
    TINY_STOP_EPOCH_BOUND,
    TINY_STOP_GATE_MET,
    TINY_STOP_NUMERIC_FAILURE,
    TINY_STOP_PROVEN_NOT_LEARNING,
    BestCheckpointSelector,
    E4TrainingError,
    assert_not_collapsed_when_marking_trained,
)

REPO = Path(__file__).resolve().parents[2]
E4_PACKAGE = REPO / "src" / "mednorm_vi" / "training" / "phase2" / "e4"
NOTEBOOK = REPO / "notebooks" / "MedNorm_E4_Clean_Training.ipynb"
CONFIG = REPO / "configs" / "training" / "phase2_e4.yaml"

# Audits are append-only. Their bytes are the record of why the removal happened.
IMMUTABLE_SHA256: dict[str, str] = {
    "docs/audits/0043-e4-post-training-collapse-diagnosis.md":
        "186fd370c784da7772e77b4f9d14acb7452b63946f4521063a9900cb5640e66a",
    "docs/audits/0044-e4-checkpoint-probe-and-root-cause-verdict.md":
        "b8163e2c0d64a9a2a2db0872b3b663fbfc5815fbe232bbfde618ee03553e8b92",
    "docs/MedNorm-VI_Architecture.pdf":
        "0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook_code() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"] if cell.get("cell_type") == "code")


def _executable_tokens(path: Path) -> str:
    """Space-joined code tokens; comments and string literals removed.

    A docstring promising "no optimizer is constructed" would satisfy a plain
    substring search, and ``backward_passes`` contains ``backward``. Joining
    tokens makes ``" backward ( "`` match a call and not a field.
    """
    kept: list[str] = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(io.BytesIO(handle.read()).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if token.string.strip():
                kept.append(token.string)
    return " " + " ".join(kept) + " "


# ---------------------------------------------------------------------------
# The removed implementation is actually gone
# ---------------------------------------------------------------------------

OBSOLETE_PATHS = (
    "src/mednorm_vi/training/phase2/e4_w2ner_training.py",
    "src/mednorm_vi/training/phase2/e4_collapse_diagnosis.py",
    "src/mednorm_vi/training/phase2/e4_checkpoint_probe.py",
    "src/mednorm_vi/training/phase2/e4_tiny_overfit.py",
    "src/mednorm_vi/training/phase2/e4_runtime_io.py",
    "src/mednorm_vi/training/phase2/e4_progress.py",
    "src/mednorm_vi/training/phase2/e4_alignment_diagnostic.py",
    "scripts/diagnose_e4_collapse.py",
    "notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb",
    "notebooks/MedNorm_E4_TinyOverfit_Diagnostic.ipynb",
    "configs/training/phase2_e4_phobert_w2ner_colab.yaml",
    "configs/training/phase2_e4_tiny_overfit_diagnostic.yaml",
)

OBSOLETE_MODULES = (
    "mednorm_vi.training.phase2.e4_w2ner_training",
    "mednorm_vi.training.phase2.e4_collapse_diagnosis",
    "mednorm_vi.training.phase2.e4_checkpoint_probe",
    "mednorm_vi.training.phase2.e4_tiny_overfit",
)


@pytest.mark.parametrize("relative_path", OBSOLETE_PATHS)
def test_obsolete_e4_path_is_deleted(relative_path: str) -> None:
    assert not (REPO / relative_path).exists(), relative_path


@pytest.mark.parametrize("relative_path", OBSOLETE_PATHS)
def test_obsolete_e4_path_is_untracked(relative_path: str) -> None:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", relative_path],
        check=True, capture_output=True, text=True).stdout.strip()
    assert tracked == "", f"{relative_path} is still tracked"


@pytest.mark.parametrize("module", OBSOLETE_MODULES)
def test_obsolete_e4_module_cannot_be_imported(module: str) -> None:
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_nothing_still_imports_the_removed_modules() -> None:
    """Match import paths, not bare words.

    ``tiny_overfit`` is now a legitimate concept name (``TinyOverfitStopPolicy``,
    the Stage-2 artifact directory), so a basename search would flag healthy
    code. What must be absent is the flat ``phase2.e4_*`` module path.
    """
    patterns = "|".join(
        [r"phase2\.e4_[a-z_]+", r"from \.e4_[a-z_]+ import",
         r"phase2/e4_[a-z_]+\.py"])
    hits = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-l", "-E", patterns,
         "--", "src", "tests", "scripts", "notebooks", "configs",
         # This file is the guard; its OBSOLETE_* lists necessarily name them.
         f":!{Path(__file__).relative_to(REPO)}"],
        capture_output=True, text=True).stdout.split()
    assert hits == [], f"stale module references remain in {hits}"


def test_there_is_exactly_one_current_e4_implementation_path() -> None:
    package_modules = sorted(p.name for p in E4_PACKAGE.glob("*.py"))
    assert package_modules == [
        "__init__.py", "alignment.py", "alignment_diagnostic.py", "contracts.py",
        "gates.py", "progress.py", "recipes.py", "runtime_io.py", "sampling.py",
        "training.py"]
    siblings = sorted(
        p.name for p in E4_PACKAGE.parent.glob("e4_*.py"))
    assert siblings == [], f"parallel E4 modules survive: {siblings}"
    assert len(list(REPO.glob("notebooks/MedNorm_E4_*.ipynb"))) == 1
    assert len(list(REPO.glob("configs/training/phase2_e4*.yaml"))) == 1


def test_the_audits_that_justify_the_removal_are_unchanged() -> None:
    pdf = REPO / "docs" / "MedNorm-VI_Architecture.pdf"
    assert _sha256(pdf) == IMMUTABLE_SHA256["docs/MedNorm-VI_Architecture.pdf"]
    for name in ("0043-e4-post-training-collapse-diagnosis.md",
                 "0044-e4-checkpoint-probe-and-root-cause-verdict.md"):
        path = REPO / "docs" / "audits" / name
        assert path.is_file(), name
        # Committed and untouched by this milestone.
        status = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", f"docs/audits/{name}"],
            check=True, capture_output=True, text=True).stdout.strip()
        assert status == "", f"{name} was modified; audits are append-only"


# ---------------------------------------------------------------------------
# Batch-global valid-cell reduction
# ---------------------------------------------------------------------------


def test_batch_global_reduction_divides_once_over_all_valid_cells() -> None:
    accumulator = BatchGlobalAccumulator()
    accumulator.observe_microbatch(loss_sum=10.0, cells=25)      # a 5-word doc
    accumulator.observe_microbatch(loss_sum=90.0, cells=26_244)  # a 162-word doc
    assert accumulator.valid_cells == 26_269
    assert accumulator.reduced() == pytest.approx(100.0 / 26_269)


def test_batch_global_reduction_is_not_the_mean_of_per_example_means() -> None:
    """The measured defect: a 5-word doc outweighed a 162-word doc per cell."""
    small = (10.0, 25)
    large = (90.0, 26_244)
    accumulator = BatchGlobalAccumulator()
    for loss_sum, cells in (small, large):
        accumulator.observe_microbatch(loss_sum=loss_sum, cells=cells)
    batch_global = accumulator.reduced()
    mean_of_means = ((small[0] / small[1]) + (large[0] / large[1])) / 2
    assert batch_global == pytest.approx(0.003806, abs=1e-6)
    assert mean_of_means == pytest.approx(0.201715, abs=1e-6)
    # Two orders of magnitude apart; the old reduction was dominated by the
    # short document, which contributed 0.1% of the cells.
    assert mean_of_means > batch_global * 50


def test_gradient_accumulation_equals_one_effective_batch_reduction() -> None:
    """Accumulated microbatch scaling must be exact, not approximate."""
    microbatches = [(12.5, 40), (7.25, 31), (30.0, 129), (1.5, 8)]
    total_cells = sum(cells for _sum, cells in microbatches)
    scale = microbatch_scale(total_cells)
    accumulated = sum(loss_sum * scale for loss_sum, _cells in microbatches)
    single_batch = sum(loss_sum for loss_sum, _cells in microbatches) / total_cells
    assert accumulated == pytest.approx(single_batch, rel=1e-12)


def test_microbatch_scale_refuses_a_nonpositive_divisor() -> None:
    with pytest.raises(RecipeError, match="must be positive"):
        microbatch_scale(0)


def test_accumulator_reports_positive_and_background_loss_separately() -> None:
    """A single total is what let the collapse hide."""
    accumulator = BatchGlobalAccumulator()
    accumulator.observe_microbatch(
        loss_sum=100.0, cells=1000, positive_sum=95.0, positive_cells=5)
    breakdown = accumulator.loss_breakdown()
    assert breakdown["total"] == pytest.approx(0.1)
    assert breakdown["positive"] == pytest.approx(19.0)
    assert breakdown["background"] == pytest.approx(5.0 / 995)
    # The total looks small while the positive term is 19 nats per cell.
    assert breakdown["positive"] > breakdown["total"] * 100


def test_accumulator_refuses_an_empty_or_inconsistent_microbatch() -> None:
    accumulator = BatchGlobalAccumulator()
    with pytest.raises(RecipeError, match="at least one valid cell"):
        accumulator.observe_microbatch(loss_sum=1.0, cells=0)
    with pytest.raises(RecipeError, match="cannot exceed"):
        accumulator.observe_microbatch(loss_sum=1.0, cells=4, positive_cells=5)
    with pytest.raises(RecipeError, match="no valid cells"):
        BatchGlobalAccumulator().reduced()


def test_reduce_grid_returns_sums_never_a_mean() -> None:
    logits = [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
              [[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]]]
    labels = [[0, 1], [2, 0]]
    mask = [[True, True], [True, True]]
    loss_sum, cells, positive_sum, positive_cells = reduce_grid(
        logits, labels, mask, objective="cross_entropy")
    assert cells == 4
    assert positive_cells == 2
    # A sum, not a mean: it must exceed the largest single-cell value.
    assert loss_sum > max(cross_entropy_cell(logits[0][0], 0), 0.0)
    assert loss_sum == pytest.approx(4 * cross_entropy_cell(logits[0][0], 0))
    assert positive_sum == pytest.approx(2 * cross_entropy_cell(logits[0][0], 0))


def test_reduce_grid_skips_masked_cells() -> None:
    logits = [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
    labels = [[0, 0], [0, 0]]
    _sum, cells, _p, _pc = reduce_grid(
        logits, labels, [[True, False], [False, True]], objective="cross_entropy")
    assert cells == 2


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


def test_exactly_three_candidate_recipes_exist() -> None:
    assert RECIPE_NAMES == (REFERENCE_CE, REFERENCE_CE_RESAMPLED, BALANCED_FOCAL)
    assert len(all_recipes()) == 3
    with pytest.raises(RecipeError, match="unknown recipe"):
        build_recipe("reference_ce_v2")


def test_every_recipe_uses_the_batch_global_reduction() -> None:
    for recipe in all_recipes():
        assert recipe.reduction == "batch_global_valid_cell_mean"
        assert recipe.as_dict()["per_example_mean_used"] is False


def test_a_recipe_cannot_declare_the_failed_per_example_reduction() -> None:
    from mednorm_vi.training.phase2.e4.recipes import Recipe
    with pytest.raises(RecipeError, match="per-example mean is the defect"):
        Recipe(name=REFERENCE_CE, objective="cross_entropy",
               reduction="per_example_mean", data_order="shuffled_source_interleaved")


def test_backbone_and_head_use_distinct_learning_rates() -> None:
    groups = OptimizerGroups()
    assert groups.backbone_lr == 5e-6
    assert groups.head_lr == 1e-3
    assert groups.head_lr > groups.backbone_lr
    assert groups.as_dict()["parameter_groups"] == ["backbone", "relation_head"]
    for recipe in all_recipes():
        assert recipe.optimizer.backbone_lr != recipe.optimizer.head_lr


def test_a_shared_learning_rate_is_refused() -> None:
    """Exactly what the collapsed run did: 2e-5 for both."""
    with pytest.raises(RecipeError, match="must not share one learning rate"):
        OptimizerGroups(backbone_lr=2e-5, head_lr=2e-5)
    with pytest.raises(RecipeError, match="must learn faster"):
        OptimizerGroups(backbone_lr=1e-3, head_lr=5e-6)


def test_the_schedule_warms_up_then_decays() -> None:
    schedule = ScheduleConfig(warmup_ratio=0.10)
    assert schedule.warmup_steps(1000) == 100
    assert schedule.multiplier_at(0, 1000) == pytest.approx(0.01)
    # Warmup ends at full rate, then decay begins from 1.0.
    assert schedule.multiplier_at(99, 1000) == pytest.approx(1.0)
    assert schedule.multiplier_at(100, 1000) == pytest.approx(1.0)
    assert schedule.multiplier_at(101, 1000) < 1.0
    assert schedule.multiplier_at(999, 1000) < schedule.multiplier_at(500, 1000)
    assert schedule.multiplier_at(1000, 1000) == 0.0
    # The collapsed run held one constant rate for all 50,748 steps.
    assert schedule.multiplier_at(0, 1000) != schedule.multiplier_at(500, 1000)
    assert schedule.max_grad_norm == 5.0


def test_focal_downweights_easy_background_but_keeps_every_positive() -> None:
    config = FocalConfig()
    easy_background = [12.0, 0.0, 0.0]     # confident NONE
    hard_positive = [4.0, 0.0, 0.0]        # confident NONE where gold is positive
    easy_ce = cross_entropy_cell(easy_background, 0)
    easy_focal = focal_cell(easy_background, 0, gamma=config.gamma,
                            alpha=config.alpha)
    positive_ce = cross_entropy_cell(hard_positive, 2)
    positive_focal = focal_cell(hard_positive, 2, gamma=config.gamma,
                                alpha=config.alpha)
    # An easy background cell is suppressed by orders of magnitude...
    assert easy_focal < easy_ce / 1000
    # ...while a hard positive keeps essentially the full alpha weight: the
    # focal factor is exactly (1 - p_t)**gamma, and p_t is tiny where the model
    # is confidently wrong, so almost nothing is removed.
    import math
    p_t = math.exp(-positive_ce)
    assert positive_focal == pytest.approx(
        config.alpha * ((1.0 - p_t) ** config.gamma) * positive_ce)
    assert positive_focal >= config.alpha * positive_ce * 0.95
    assert positive_focal > easy_focal * 1000


def test_focal_refuses_the_raw_inverse_frequency_regime() -> None:
    with pytest.raises(RecipeError, match="inverse-frequency regime"):
        FocalConfig(alpha=0.995)
    with pytest.raises(RecipeError, match="between 0 and 1"):
        FocalConfig(alpha=1.0)
    with pytest.raises(RecipeError, match="non-negative"):
        FocalConfig(gamma=-1.0)
    # 577:1 is the measured background:positive ratio; nothing near it is allowed.
    assert FocalConfig().as_dict()[
        "effective_positive_to_background_weight_ratio"] < 1.0


def test_only_the_focal_recipe_carries_a_focal_config() -> None:
    assert build_recipe(BALANCED_FOCAL).focal is not None
    assert build_recipe(REFERENCE_CE).focal is None
    assert build_recipe(REFERENCE_CE_RESAMPLED).focal is None


def test_only_the_resampled_recipe_uses_positive_aware_sampling() -> None:
    assert build_recipe(REFERENCE_CE_RESAMPLED).uses_positive_aware_sampling
    assert not build_recipe(REFERENCE_CE).uses_positive_aware_sampling
    assert not build_recipe(BALANCED_FOCAL).uses_positive_aware_sampling


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


def _governed_layout() -> list[ExampleIndex]:
    """The real governed train composition measured in Audit 0043 §7."""
    examples: list[ExampleIndex] = []
    row = 0
    for source, count, positives in (("phoner_covid19", 10_027, 0),
                                     ("vimedner", 5_796, 4_965),
                                     ("vimq", 8_736, 619),
                                     ("vietmed_ner", 9_267, 2_114)):
        for offset in range(count):
            examples.append(
                ExampleIndex(row, source, 1 if offset < positives else 0))
            row += 1
    return examples


def test_the_epoch_order_is_deterministic_and_varies_by_epoch() -> None:
    examples = _governed_layout()[:2000]
    first = build_epoch_order(
        examples, data_order=SHUFFLED_SOURCE_INTERLEAVED, seed=7, epoch=1)
    again = build_epoch_order(
        examples, data_order=SHUFFLED_SOURCE_INTERLEAVED, seed=7, epoch=1)
    later = build_epoch_order(
        examples, data_order=SHUFFLED_SOURCE_INTERLEAVED, seed=7, epoch=2)
    assert [e.row_index for e in first] == [e.row_index for e in again]
    assert [e.row_index for e in first] != [e.row_index for e in later]
    other_seed = build_epoch_order(
        examples, data_order=SHUFFLED_SOURCE_INTERLEAVED, seed=8, epoch=1)
    assert [e.row_index for e in first] != [e.row_index for e in other_seed]


def test_source_interleaving_breaks_the_ten_thousand_example_block() -> None:
    examples = _governed_layout()
    order = build_epoch_order(
        examples, data_order=SHUFFLED_SOURCE_INTERLEAVED, seed=7, epoch=1)
    composition = measure_order(
        order, epoch=1, data_order=SHUFFLED_SOURCE_INTERLEAVED)
    # File order gave 10,027 for both of these.
    assert composition.longest_same_source_streak <= 4
    assert composition.longest_zero_entity_streak < 100
    assert composition.first_positive_position < 100


def test_resampled_mode_bounds_the_zero_entity_streak() -> None:
    examples = _governed_layout()
    order = build_epoch_order(
        examples, data_order=POSITIVE_AWARE_RESAMPLED, seed=7, epoch=1)
    composition = measure_order(order, epoch=1, data_order=POSITIVE_AWARE_RESAMPLED)
    assert composition.longest_zero_entity_streak <= 4
    assert composition.first_positive_position == 0


def test_every_order_is_a_permutation_that_keeps_zero_entity_examples() -> None:
    examples = _governed_layout()
    for data_order in (SHUFFLED_SOURCE_INTERLEAVED, POSITIVE_AWARE_RESAMPLED):
        order = build_epoch_order(examples, data_order=data_order, seed=3, epoch=1)
        assert_order_preserves_corpus(examples, order)
        composition = measure_order(order, epoch=1, data_order=data_order)
        assert composition.examples == len(examples)
        assert composition.zero_entity_examples == sum(
            1 for e in examples if not e.has_entities)
        assert composition.as_dict()["examples_dropped"] == 0
        assert composition.as_dict()["examples_duplicated"] == 0


def test_an_unsatisfiable_zero_entity_cap_is_refused_not_approximated() -> None:
    """The corpus is 77.2% zero-entity; a 50% cap cannot be met honestly."""
    with pytest.raises(SamplingError, match="cannot be met without dropping"):
        build_epoch_order(
            _governed_layout(), data_order=POSITIVE_AWARE_RESAMPLED, seed=1,
            epoch=1, max_zero_entity_fraction=0.5)


# ---------------------------------------------------------------------------
# Supervision scope
# ---------------------------------------------------------------------------


def test_absent_laboratory_types_never_create_fake_supervision() -> None:
    assert E4_SUPERVISED_TYPES == ("DIAGNOSIS", "MEDICATION", "SYMPTOM")
    assert E4_UNSUPERVISED_TYPES == ("TEST_NAME", "TEST_RESULT")
    # The global five-type schema and all seven grid labels are preserved.
    assert set(E4_SUPERVISED_TYPES) | set(E4_UNSUPERVISED_TYPES) == set(
        W2NERLabelVocab().type_order)
    assert len(W2NERLabelVocab().labels) == 7


def test_a_stage_is_never_failed_for_an_absent_type() -> None:
    present = ["DIAGNOSIS", "SYMPTOM"]
    assert required_supervised_types(present) == ("DIAGNOSIS", "SYMPTOM")
    assert "TEST_NAME" not in required_supervised_types(
        ["DIAGNOSIS", "MEDICATION", "SYMPTOM"])


def test_the_config_records_the_absence_rather_than_hiding_it() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["data"]["supervised_types"] == ["DIAGNOSIS", "MEDICATION", "SYMPTOM"]
    assert config["data"]["unsupervised_types"] == ["TEST_NAME", "TEST_RESULT"]
    assert config["data"]["unsupervised_type_policy"]
    assert config["data"]["internal_test_allowed"] is False


# ---------------------------------------------------------------------------
# No superseded checkpoint can initialize the new path
# ---------------------------------------------------------------------------


def test_a_current_checkpoint_payload_is_accepted() -> None:
    payload = e4_checkpoint_payload(
        mode="full", config_sha256="0" * 64, model_revision="a" * 40,
        tokenizer_revision="a" * 40, parameter_count=1, recipe_name=REFERENCE_CE)
    reject_superseded_checkpoint(payload)
    assert payload["e4_checkpoint_schema_version"] == E4_CHECKPOINT_SCHEMA_VERSION


@pytest.mark.parametrize("schema", E4_REJECTED_CHECKPOINT_SCHEMA_VERSIONS)
def test_a_superseded_checkpoint_is_refused(schema: str) -> None:
    """The collapsed checkpoints restore *perfectly*, which is why this is
    mechanical rather than a matter of discipline."""
    with pytest.raises(E4ContractError, match="superseded|unknown"):
        reject_superseded_checkpoint({
            "e4_checkpoint_schema_version": schema,
            "e4_input_contract_version": "e4-atomic-grid-word-v1",
            "model_state": {"base_model": {}, "w2ner_head": {}},
        })


def test_the_collapsed_run_checkpoint_schema_is_explicitly_named() -> None:
    assert "phase2-e4-checkpoint-v2" in E4_REJECTED_CHECKPOINT_SCHEMA_VERSIONS
    assert E4_CHECKPOINT_SCHEMA_VERSION == "phase2-e4-checkpoint-v3"


# ---------------------------------------------------------------------------
# Collapse guard
# ---------------------------------------------------------------------------


def _snapshot(epoch: int, *, predicted: int = 0, thw: int = 0,
              true_positives: int = 0, rate: float = 1.0,
              loss: float = 0.0136) -> ValidationSnapshot:
    return ValidationSnapshot(
        epoch=epoch, predicted_mentions=predicted, gold_mentions=1991,
        true_positives=true_positives, thw_predictions=thw,
        nnw_predictions=0, gold_positive_background_rate=rate, train_loss=loss)


def test_the_guard_stops_an_all_none_run() -> None:
    history = [_snapshot(e) for e in range(1, 5)]
    verdict = evaluate_collapse_guard(history)
    assert verdict.collapsed
    assert verdict.consecutive_collapsed_epochs >= 2
    assert "zero mentions" in verdict.reason


def test_the_guard_fires_on_the_real_audited_history_at_epoch_five() -> None:
    """Replaying Audit 0043 §4: 7 epochs and 236,782 backward passes saved."""
    audited = [
        (1, 0, 0, 0, 0.0, 0.013644), (2, 13, 127, 2, 0.99910, 0.006473),
        (3, 4, 40, 1, 0.99910, 0.005934), (4, 0, 0, 0, 1.0, 0.009257),
        (5, 0, 0, 0, 1.0, 0.013688),
    ]
    history: list[ValidationSnapshot] = []
    fired_at = None
    for epoch, predicted, thw, true_positives, rate, loss in audited:
        history.append(_snapshot(epoch, predicted=predicted, thw=thw,
                                 true_positives=true_positives, rate=rate, loss=loss))
        if evaluate_collapse_guard(history).collapsed:
            fired_at = epoch
            break
    assert fired_at == 5


def test_the_guard_tolerates_an_untrained_first_epoch() -> None:
    """Epoch 1 legitimately predicts nothing; aborting there is wrong."""
    assert not evaluate_collapse_guard([_snapshot(1)]).collapsed
    assert not evaluate_collapse_guard([_snapshot(1), _snapshot(2)]).collapsed


def test_the_guard_does_not_fire_on_a_healthy_run() -> None:
    healthy = [
        _snapshot(1), _snapshot(2, predicted=40, thw=60, true_positives=12, rate=0.7),
        _snapshot(3, predicted=90, thw=140, true_positives=45, rate=0.4),
    ]
    assert not evaluate_collapse_guard(healthy).collapsed


def test_the_guard_requires_all_four_symptoms() -> None:
    """A run still emitting THW relations is not collapsed."""
    partial = [_snapshot(e, thw=5) for e in range(1, 6)]
    assert not evaluate_collapse_guard(partial).collapsed


def test_a_collapsed_run_can_never_be_marked_fully_trained() -> None:
    verdict = evaluate_collapse_guard([_snapshot(e) for e in range(1, 5)])
    with pytest.raises(E4TrainingError, match="refusing to record status"):
        assert_not_collapsed_when_marking_trained(verdict, "FULLY_TRAINED")
    assert_not_collapsed_when_marking_trained(verdict, "COLLAPSED_NOT_TRAINED")
    assert verdict.as_dict()["status_if_collapsed"] == "COLLAPSED_NOT_TRAINED"


# ---------------------------------------------------------------------------
# Stage gates
# ---------------------------------------------------------------------------


def _reproduction(*, reproduced: bool = True, **overrides) -> ReproductionCheck:
    before = {"exact_precision": 1.0, "exact_recall": 1.0, "exact_f1": 1.0,
              "predicted_mentions": 22.0, "positive_cell_accuracy": 0.98}
    after = dict(before) if reproduced else {**before, "exact_f1": 0.4}
    fields = {
        "checkpoint_sha256": "a" * 64, "metrics_before": before,
        "metrics_after": after, "missing_keys": (), "unexpected_keys": (),
        "fresh_model_evaluated": True, "examples_evaluated": 12,
    }
    fields.update(overrides)
    return ReproductionCheck(**fields)


def _schedule(*, peak: bool = True) -> dict:
    return SchedulePlan(
        examples=12, accumulation_steps=4, epoch_bound=200,
        optimizer_steps_per_epoch=3, planned_total_optimizer_steps=600,
        warmup_steps=60,
        realized_optimizer_steps=600 if peak else 12,
        realized_epochs=200 if peak else 4).as_dict()


def _result(name: str, **overrides) -> RecipeResult:
    fields = {
        "recipe": name, "exact_precision": 1.0, "exact_recall": 1.0,
        "exact_f1": 1.0, "predicted_mentions": 22, "gold_mentions": 22,
        "false_positives": 0, "positive_cell_accuracy": 0.98,
        "gold_positive_background_rate": 0.01, "nnw_predictions": 30,
        "thw_predictions_by_type": {"DIAGNOSIS": 10, "SYMPTOM": 8, "MEDICATION": 4},
        "loss_total": 0.001, "loss_positive": 0.01, "loss_background": 0.0001,
        "seconds": 60.0, "peak_vram_gib": 9.0, "reproduction": _reproduction(),
        "grid_cell_accuracy": 0.9985, "schedule": _schedule(),
        "stopped_reason": "tiny_gate_met",
    }
    fields.update(overrides)
    return RecipeResult(**fields)


REQUIRED = ("DIAGNOSIS", "MEDICATION", "SYMPTOM")


def test_the_tiny_stage_compares_all_three_recipes() -> None:
    results = [_result(name) for name in RECIPE_NAMES]
    selected, report = select_recipe(results, required_types=REQUIRED)
    assert {r["recipe"] for r in report["recipes_evaluated"]} == set(RECIPE_NAMES)
    assert len(report["pass_summary"]) == 3
    assert selected is not None


def test_an_exact_tie_selects_the_simplest_recipe() -> None:
    results = [_result(name) for name in RECIPE_NAMES]
    selected, _report = select_recipe(results, required_types=REQUIRED)
    assert selected.recipe == REFERENCE_CE


def test_selection_prefers_f1_then_recall_then_fewer_false_positives() -> None:
    results = [
        _result(REFERENCE_CE, exact_f1=0.96, exact_recall=0.96),
        _result(BALANCED_FOCAL, exact_f1=0.99, exact_recall=0.99),
    ]
    selected, _ = select_recipe(results, required_types=REQUIRED)
    assert selected.recipe == BALANCED_FOCAL
    tie = [
        _result(REFERENCE_CE, exact_f1=0.99, exact_recall=0.99, false_positives=5),
        _result(BALANCED_FOCAL, exact_f1=0.99, exact_recall=0.99, false_positives=1),
    ]
    selected, _ = select_recipe(tie, required_types=REQUIRED)
    assert selected.recipe == BALANCED_FOCAL


def test_grid_cell_accuracy_is_never_a_pass_criterion() -> None:
    """An all-background model already scores ~0.998 on it."""
    collapsed = _result(
        REFERENCE_CE, exact_f1=0.0, exact_recall=0.0, predicted_mentions=0,
        positive_cell_accuracy=0.0, gold_positive_background_rate=1.0,
        thw_predictions_by_type={}, grid_cell_accuracy=0.9985)
    passed, failures = collapsed.passes(required_types=REQUIRED)
    assert not passed
    assert any("exact F1" in f for f in failures)
    assert any("predicted no mentions" in f for f in failures)
    assert collapsed.as_dict()["grid_cell_accuracy_is_a_pass_criterion"] is False


def test_a_recipe_missing_a_supervised_type_fails() -> None:
    partial = _result(REFERENCE_CE,
                      thw_predictions_by_type={"DIAGNOSIS": 10, "SYMPTOM": 8})
    passed, failures = partial.passes(required_types=REQUIRED)
    assert not passed
    assert any("MEDICATION" in f for f in failures)


def test_when_no_recipe_passes_the_chain_stops() -> None:
    failing = [_result(name, exact_f1=0.1) for name in RECIPE_NAMES]
    selected, report = select_recipe(failing, required_types=REQUIRED)
    assert selected is None
    assert report["passed"] is False
    assert "must not run" in report["stop_reason"]


def test_the_subset_gate_requires_real_predictions() -> None:
    good = SubsetResult(
        recipe=REFERENCE_CE, validation_predicted_mentions=120,
        validation_gold_mentions=400, validation_recall=0.22,
        validation_exact_f1=0.25, nnw_predictions=300,
        thw_predictions_by_type={"DIAGNOSIS": 60, "SYMPTOM": 40, "MEDICATION": 20},
        gold_positive_background_rate=0.6, collapse_guard_fired=False,
        save_reload_reproduced=True, artifact_validator_ok=True)
    assert good.passes(required_types=REQUIRED)[0]
    collapsed = SubsetResult(
        recipe=REFERENCE_CE, validation_predicted_mentions=0,
        validation_gold_mentions=400, validation_recall=0.0,
        validation_exact_f1=0.0, nnw_predictions=0, thw_predictions_by_type={},
        gold_positive_background_rate=1.0, collapse_guard_fired=True,
        save_reload_reproduced=True, artifact_validator_ok=True)
    passed, failures = collapsed.passes(required_types=REQUIRED)
    assert not passed
    assert len(failures) >= 5


def test_stage_authorization_needs_both_the_flag_and_the_string() -> None:
    with pytest.raises(GateError, match="disabled"):
        assert_stage_authorized(
            "tiny_recipe_ablation", "I_AUTHORIZE_E4_TINY_RECIPE_ABLATION",
            enabled=False)
    with pytest.raises(GateError, match="exact authorization string"):
        assert_stage_authorized("tiny_recipe_ablation", "yes", enabled=True)
    assert_stage_authorized(
        "tiny_recipe_ablation", "I_AUTHORIZE_E4_TINY_RECIPE_ABLATION", enabled=True)


def _write_gates(root: Path, *, tiny_passed: bool = True, subset_passed: bool = True,
                 tiny_recipe: str = REFERENCE_CE, subset_recipe: str = REFERENCE_CE,
                 config: str = "cfg", code: str = "code",
                 corpus: dict | None = None) -> None:
    corpus = corpus or {"train": "t", "validation": "v"}
    GateArtifact(stage="tiny_recipe_ablation", passed=tiny_passed,
                 recipe=tiny_recipe, config_sha256=config, code_sha256=code,
                 corpus_sha256=corpus, detail={}).write(root / TINY_GATE_FILENAME)
    GateArtifact(stage="subset_smoke", passed=subset_passed, recipe=subset_recipe,
                 config_sha256=config, code_sha256=code, corpus_sha256=corpus,
                 detail={}).write(root / SUBSET_GATE_FILENAME)


FULL_OK = {"config_sha256": "cfg", "code_sha256": "code",
           "corpus_sha256": {"train": "t", "validation": "v"},
           "confirmation": "I_AUTHORIZE_E4_FULL_TRAINING", "enabled": True}


def test_full_training_runs_only_when_both_gates_passed(tmp_path: Path) -> None:
    _write_gates(tmp_path)
    assert assert_full_training_allowed(gate_dir=tmp_path, **FULL_OK) == REFERENCE_CE


def test_flipping_the_full_flag_cannot_bypass_the_tiny_gate(tmp_path: Path) -> None:
    """The whole point of the chain."""
    _write_gates(tmp_path, tiny_passed=False)
    with pytest.raises(GateError, match="did not pass"):
        assert_full_training_allowed(gate_dir=tmp_path, **FULL_OK)


def test_flipping_the_full_flag_cannot_bypass_the_subset_gate(tmp_path: Path) -> None:
    _write_gates(tmp_path, subset_passed=False)
    with pytest.raises(GateError, match="did not pass"):
        assert_full_training_allowed(gate_dir=tmp_path, **FULL_OK)


def test_full_training_is_refused_when_no_gate_artifact_exists(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="has not run"):
        assert_full_training_allowed(gate_dir=tmp_path, **FULL_OK)


def test_full_training_is_refused_when_the_gates_disagree(tmp_path: Path) -> None:
    _write_gates(tmp_path, tiny_recipe=REFERENCE_CE, subset_recipe=BALANCED_FOCAL)
    with pytest.raises(GateError, match="but the subset smoke validated"):
        assert_full_training_allowed(gate_dir=tmp_path, **FULL_OK)


@pytest.mark.parametrize("changed", ["config_sha256", "code_sha256", "corpus_sha256"])
def test_full_training_is_refused_when_the_hashes_moved(
    tmp_path: Path, changed: str,
) -> None:
    _write_gates(tmp_path)
    arguments = dict(FULL_OK)
    arguments[changed] = ({"train": "other", "validation": "v"}
                          if changed == "corpus_sha256" else "moved")
    with pytest.raises(GateError, match="re-run the gated stages"):
        assert_full_training_allowed(gate_dir=tmp_path, **arguments)


def test_full_training_still_needs_its_own_authorization(tmp_path: Path) -> None:
    _write_gates(tmp_path)
    arguments = dict(FULL_OK, enabled=False)
    with pytest.raises(GateError, match="disabled"):
        assert_full_training_allowed(gate_dir=tmp_path, **arguments)


# ---------------------------------------------------------------------------
# Accumulation accounting
# ---------------------------------------------------------------------------


def test_accumulation_accounting_is_derived_not_relabelled() -> None:
    plan = plan_gradient_accumulation(33_826, accumulation_steps=8, epochs=12)
    assert plan.micro_batches_per_epoch == 33_826
    assert plan.optimizer_steps_per_epoch == 4_229
    assert plan.expected_optimizer_steps == 50_748
    assert plan.expected_backward_passes == 405_912
    assert plan.effective_batch_size == 8
    assert plan.as_dict()["loss_reduction"] == "batch_global_valid_cell_mean"


def test_e4_refuses_a_microbatch_larger_than_one() -> None:
    with pytest.raises(E4TrainingError, match="variable-sized per document"):
        plan_gradient_accumulation(
            100, micro_batch_size=4, accumulation_steps=2, epochs=1)


# ---------------------------------------------------------------------------
# Notebook: dependency install, preflight, gating
# ---------------------------------------------------------------------------


def test_the_notebook_installs_py_vncorenlp_before_the_first_import() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [("".join(c.get("source", [])), c.get("cell_type"))
             for c in payload["cells"]]
    install_index = next(
        i for i, (source, kind) in enumerate(cells)
        if kind == "code" and "%pip install -q py_vncorenlp==0.1.4" in source)
    first_import = next(
        i for i, (source, kind) in enumerate(cells)
        if kind == "code" and "import py_vncorenlp" in source)
    assert install_index < first_import


def test_the_preflight_verifies_every_required_dependency() -> None:
    code = _notebook_code()
    for probe in ("import py_vncorenlp", "py_vncorenlp_version",
                  "py_vncorenlp_location", "download_model", "import torch",
                  "import transformers", "torch.cuda.is_available()",
                  "get_device_name", "get_device_capability",
                  "t4_compatible_cuda_runtime", "pinned_phobert_revision",
                  "internal_test_prohibited"):
        assert probe in code, probe
    assert "raise SystemExit" in code


def test_the_notebook_ships_every_stage_disabled() -> None:
    code = _notebook_code()
    for flag in ("RUN_STAGE2_TINY_ABLATION = False",
                 "RUN_STAGE3_SUBSET_SMOKE = False",
                 "RUN_STAGE4_FULL_TRAINING = False",
                 'CONFIRM_STAGE2 = ""', 'CONFIRM_STAGE3 = ""',
                 'CONFIRM_STAGE4 = ""'):
        assert flag in code, flag


def test_the_notebook_gates_every_stage_and_fails_closed() -> None:
    code = _notebook_code()
    assert 'assert_stage_authorized("tiny_recipe_ablation"' in code
    assert 'assert_stage_authorized("subset_smoke"' in code
    assert "assert_full_training_allowed(" in code
    # Stage 4 must go through the hash-bound chain, not a bare authorization.
    assert 'assert_stage_authorized("full_training"' not in code


def test_the_notebook_never_touches_internal_test_or_output_zip() -> None:
    code = _notebook_code()
    assert "internal_test.jsonl" not in code
    assert 'split="internal_test"' not in code
    assert "output.zip" not in code
    assert '"internal_test_accessed": False' in code


def test_the_notebook_trains_from_fresh_weights_and_refuses_old_checkpoints() -> None:
    code = _notebook_code()
    assert "reject_superseded_checkpoint(" in code
    assert "pinned_pretrained_base_fresh_head" in code
    assert "build_relation_grid_head(" in code   # a NEW head every run


def test_every_notebook_code_cell_parses_once_magics_are_stripped() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "\n".join(
            "" if line.lstrip().startswith(("%", "!")) else line
            for line in "".join(cell.get("source", [])).splitlines())
        ast.parse(source, filename=f"e4_clean_cell_{index}")


# ---------------------------------------------------------------------------
# The implementation never trains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", sorted(p.name for p in E4_PACKAGE.glob("*.py")))
def test_no_e4_module_constructs_an_optimizer_or_calls_backward(module: str) -> None:
    tokens = _executable_tokens(E4_PACKAGE / module)
    for forbidden in (" backward ( ", " AdamW ", " optim ", " zero_grad ( "):
        assert forbidden not in tokens, f"{module}: {forbidden}"


def test_the_e4_package_never_names_the_frozen_split_as_a_data_path() -> None:
    for path in E4_PACKAGE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "internal_test.jsonl" not in source, path.name


def test_this_test_module_never_trains() -> None:
    tokens = _executable_tokens(Path(__file__))
    for forbidden in (" backward ( ", " AdamW ", " optim ", " zero_grad ( "):
        assert forbidden not in tokens, forbidden


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------


def test_no_generated_artifact_is_tracked() -> None:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    for path in tracked:
        assert not path.endswith(
            (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip", ".jar")), path
        assert not path.startswith(
            ("local-artifacts/", "checkpoint/", "artifacts/", "weights/",
             "caches/", "model_cache/", ".claude/")), path
        assert Path(path).name not in {"CLAUDE.md", "AGENTS.md"}


@pytest.mark.parametrize("pattern", [
    "local-artifacts/", "checkpoint/", "reports/", "data/derived/",
    "artifacts/", "weights/", "caches/",
])
def test_gitignore_covers_every_artifact_location(pattern: str) -> None:
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    stem = pattern.rstrip("/")
    assert stem in ignore, pattern


def test_the_config_is_valid_and_committed_unauthorized() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "IMPLEMENTED_NOT_AUTHORIZED"
    assert config["enabled_by_default"] is False
    assert config["authorization"]["committed_run_flags"] is False
    assert config["authorization"]["fail_closed"] is True
    assert config["permitted"]["local_training"] is False
    assert config["permitted"]["internal_test_access"] is False
    assert config["permitted"]["output_zip_creation"] is False
    assert config["permitted"]["tracked_checkpoints"] is False
    assert set(config["recipes"]) == {"shared", *RECIPE_NAMES}
    assert config["recipes"]["shared"]["loss_reduction"] == (
        "batch_global_valid_cell_mean")
    assert config["recipes"]["shared"]["per_example_mean"] is False
    assert config["model"]["resume_from_superseded_checkpoint"] is False


# ---------------------------------------------------------------------------
# Tiny-overfit stopping policy (Audit 0046)
# ---------------------------------------------------------------------------
#
# Colab evidence from the first Stage-2 run: every recipe stopped after 4 epochs
# and 12 optimizer steps, against a planned 600 steps with a 60-step warmup, so
# the backbone was at 1e-6 and the head at 2e-4 instead of 5e-6 and 1e-3. That is
# a stopper artefact, not a fact about the recipes.

STAGE2_EXAMPLES = 12
STAGE2_ACCUMULATION = 4
STAGE2_EPOCH_BOUND = 200


def test_the_generic_stopper_reproduces_the_four_epoch_stop() -> None:
    """Regression: exactly what shipped, reproduced before the repair.

    BestCheckpointSelector still exists and is still correct — for full
    training. Applied to a tiny run whose F1 is legitimately 0.0 during warmup,
    patience 3 ends the run at epoch 4.
    """
    plan = plan_gradient_accumulation(
        STAGE2_EXAMPLES, accumulation_steps=STAGE2_ACCUMULATION,
        epochs=STAGE2_EPOCH_BOUND)
    selector = BestCheckpointSelector(patience=3)
    stopped_at = None
    steps = 0
    for epoch in range(1, STAGE2_EPOCH_BOUND + 1):
        steps += plan.optimizer_steps_per_epoch
        selector.observe(epoch=epoch, exact_f1=0.0)
        if selector.should_stop:
            stopped_at = epoch
            break
    assert stopped_at == 4
    assert steps == 12
    assert selector.best_epoch == 1
    assert selector.best_metric == 0.0


def test_the_reproduced_stop_lands_inside_warmup_at_a_fifth_of_target_lr() -> None:
    """The four learning-rate numbers the Colab run reported, derived."""
    schedule = plan_schedule(
        examples=STAGE2_EXAMPLES, accumulation_steps=STAGE2_ACCUMULATION,
        epoch_bound=STAGE2_EPOCH_BOUND, warmup_ratio=0.10)
    assert schedule.planned_total_optimizer_steps == 600
    assert schedule.warmup_steps == 60
    realized = schedule.realized(optimizer_steps=12, epochs=4)
    assert not realized.warmup_completed
    assert not realized.peak_learning_rate_reached
    assert realized.warmup_fraction_served == pytest.approx(0.2)

    recipe = build_recipe(REFERENCE_CE)
    multiplier = recipe.schedule.multiplier_at(11, 600)
    assert multiplier == pytest.approx(0.2)
    assert recipe.optimizer.backbone_lr * multiplier == pytest.approx(1e-6)
    assert recipe.optimizer.head_lr * multiplier == pytest.approx(2e-4)


def _tiny_policy(**overrides) -> TinyOverfitStopPolicy:
    fields = {
        "epoch_bound": STAGE2_EPOCH_BOUND, "warmup_steps": 60,
        "target_exact_f1": TINY_TARGET_EXACT_F1,
        "required_types": ("DIAGNOSIS", "MEDICATION", "SYMPTOM"),
    }
    fields.update(overrides)
    return TinyOverfitStopPolicy(**fields)


def _signal(epoch: int, *, f1: float = 0.0, predicted: int = 0,
            positive_accuracy: float = 0.0, types: tuple[str, ...] = (),
            steps: int | None = None, finite: bool = True) -> TinyEpochSignal:
    return TinyEpochSignal(
        epoch=epoch, optimizer_steps=steps if steps is not None else epoch * 3,
        exact_f1=f1, predicted_mentions=predicted,
        positive_cell_accuracy=positive_accuracy, types_predicted=types,
        loss_total=0.01, loss_is_finite=finite)


def test_the_tiny_policy_does_not_stop_at_epoch_four_on_zero_f1() -> None:
    """The repair, stated as the exact scenario that failed."""
    policy = _tiny_policy()
    history = [_signal(epoch) for epoch in range(1, 5)]
    stop, reason = policy.decide(history)
    assert not stop
    assert reason == "continue"


def test_the_tiny_policy_never_stops_early_on_a_flat_zero_f1_run() -> None:
    policy = _tiny_policy()
    history: list[TinyEpochSignal] = []
    for epoch in range(1, STAGE2_EPOCH_BOUND):
        history.append(_signal(epoch))
        stop, _reason = policy.decide(history)
        assert not stop, f"stopped at epoch {epoch}"


def test_unsuccessful_tiny_training_reaches_the_epoch_bound() -> None:
    policy = _tiny_policy()
    history = [_signal(epoch) for epoch in range(1, STAGE2_EPOCH_BOUND + 1)]
    stop, reason = policy.decide(history)
    assert stop
    assert reason == TINY_STOP_EPOCH_BOUND


def test_successful_tiny_memorization_may_stop_before_the_bound() -> None:
    policy = _tiny_policy()
    history = [_signal(epoch) for epoch in range(1, 40)]
    history.append(_signal(40, f1=0.97, predicted=22, positive_accuracy=0.95,
                           types=("DIAGNOSIS", "MEDICATION", "SYMPTOM")))
    stop, reason = policy.decide(history)
    assert stop
    assert reason == TINY_STOP_GATE_MET


def test_the_gate_needs_every_present_supervised_type() -> None:
    policy = _tiny_policy()
    incomplete = _signal(50, f1=0.99, predicted=20, positive_accuracy=0.9,
                         types=("DIAGNOSIS", "SYMPTOM"))
    assert not incomplete.gate_met(
        required_types=policy.required_types, target_f1=policy.target_exact_f1)
    stop, reason = policy.decide([incomplete])
    assert not stop and reason == "continue"


def test_the_exact_f1_gate_is_not_weakened() -> None:
    policy = _tiny_policy()
    just_under = _signal(50, f1=0.94, predicted=22, positive_accuracy=0.9,
                         types=("DIAGNOSIS", "MEDICATION", "SYMPTOM"))
    assert not just_under.gate_met(
        required_types=policy.required_types, target_f1=policy.target_exact_f1)
    assert policy.target_exact_f1 == 0.95
    assert TINY_TARGET_EXACT_F1 == 0.95


def test_a_numeric_failure_stops_immediately() -> None:
    policy = _tiny_policy()
    stop, reason = policy.decide([_signal(2, finite=False)])
    assert stop
    assert reason == TINY_STOP_NUMERIC_FAILURE


def test_fail_fast_is_opt_in_and_only_after_the_full_warmup() -> None:
    policy = _tiny_policy(allow_fail_fast=True, fail_fast_patience_after_warmup=5)
    # Twenty dead epochs, all still inside warmup: no fail-fast.
    inside = [_signal(e, steps=3 * e) for e in range(1, 20)]
    assert all(s.optimizer_steps < policy.warmup_steps for s in inside)
    stop, _reason = policy.decide(inside)
    assert not stop
    # Past the warmup and still dead: the opt-in fail-fast may fire.
    beyond = inside + [_signal(e, steps=3 * e) for e in range(20, 40)]
    stop, reason = policy.decide(beyond)
    assert stop
    assert reason == TINY_STOP_PROVEN_NOT_LEARNING


def test_fail_fast_is_off_by_default() -> None:
    policy = _tiny_policy()
    assert policy.allow_fail_fast is False
    dead = [_signal(e, steps=3 * e) for e in range(1, 100)]
    stop, _reason = policy.decide(dead)
    assert not stop


def test_the_tiny_policy_records_that_it_is_not_validation_patience() -> None:
    payload = _tiny_policy().as_dict()
    assert payload["validation_patience_early_stopping_used"] is False
    assert payload["collapse_guard_enabled"] is False
    assert payload["fail_fast_permitted_before_full_warmup"] is False
    assert payload["heartbeat_every_n_epochs"] == 5


def test_the_policy_heartbeats_at_least_every_five_epochs() -> None:
    policy = _tiny_policy()
    beats = [e for e in range(1, 51) if policy.should_heartbeat(e)]
    assert 1 in beats
    gaps = [b - a for a, b in zip(beats, beats[1:], strict=False)]
    assert max(gaps) <= 5


def test_full_training_early_stopping_is_unchanged() -> None:
    """The generic stopper keeps its behaviour for Stage 4."""
    selector = BestCheckpointSelector(patience=3)
    assert selector.observe(epoch=1, exact_f1=0.10)
    assert not selector.should_stop
    assert selector.observe(epoch=2, exact_f1=0.20)
    assert selector.best_epoch == 2
    for epoch in (3, 4, 5):
        assert not selector.observe(epoch=epoch, exact_f1=0.15)
    assert selector.should_stop
    assert selector.as_dict()["early_stopping_patience"] == 3
    assert selector.as_dict()["selection_split"] == "governed_validation_only"


# ---------------------------------------------------------------------------
# Scheduler accounting
# ---------------------------------------------------------------------------


def test_the_schedule_is_planned_from_examples_accumulation_and_full_bound() -> None:
    schedule = plan_schedule(
        examples=12, accumulation_steps=4, epoch_bound=200, warmup_ratio=0.10)
    assert schedule.optimizer_steps_per_epoch == 3
    assert schedule.planned_total_optimizer_steps == 3 * 200
    assert schedule.warmup_steps == 60
    # Halving the bound halves the plan; the bound is a real input.
    halved = plan_schedule(
        examples=12, accumulation_steps=4, epoch_bound=100, warmup_ratio=0.10)
    assert halved.planned_total_optimizer_steps == 300
    assert halved.warmup_steps == 30


def test_planned_and_realized_step_counts_are_both_recorded() -> None:
    schedule = plan_schedule(
        examples=12, accumulation_steps=4, epoch_bound=200, warmup_ratio=0.10)
    payload = schedule.realized(optimizer_steps=600, epochs=200).as_dict()
    for key in ("planned_total_optimizer_steps", "realized_optimizer_steps",
                "warmup_steps", "realized_epochs", "warmup_completed",
                "warmup_fraction_served", "peak_learning_rate_reached"):
        assert key in payload, key
    assert payload["planned_total_optimizer_steps"] == 600
    assert payload["realized_optimizer_steps"] == 600


def test_the_scheduler_reaches_peak_learning_rate_after_warmup() -> None:
    schedule = plan_schedule(
        examples=12, accumulation_steps=4, epoch_bound=200, warmup_ratio=0.10)
    recipe = build_recipe(REFERENCE_CE)
    total = schedule.planned_total_optimizer_steps
    # Immediately after the final warmup step the multiplier is exactly 1.0.
    assert recipe.schedule.multiplier_at(
        schedule.warmup_steps - 1, total) == pytest.approx(1.0)
    peak_backbone = recipe.optimizer.backbone_lr * recipe.schedule.multiplier_at(
        schedule.warmup_steps - 1, total)
    peak_head = recipe.optimizer.head_lr * recipe.schedule.multiplier_at(
        schedule.warmup_steps - 1, total)
    assert peak_backbone == pytest.approx(5e-6)
    assert peak_head == pytest.approx(1e-3)
    assert schedule.realized(
        optimizer_steps=schedule.warmup_steps, epochs=20).peak_learning_rate_reached


def test_a_run_that_stopped_inside_warmup_is_flagged_on_the_result() -> None:
    denied = _result(REFERENCE_CE, schedule=_schedule(peak=False))
    assert denied.stopped_inside_warmup
    assert not _result(REFERENCE_CE).stopped_inside_warmup


def test_a_failed_ablation_says_when_it_is_not_evidence_about_the_recipes() -> None:
    """Exactly the first Stage-2 run: three 'failures' that proved nothing."""
    denied = [
        _result(name, exact_f1=0.0, exact_recall=0.0, predicted_mentions=0,
                positive_cell_accuracy=0.0, thw_predictions_by_type={},
                schedule=_schedule(peak=False),
                stopped_reason="early_stopping_patience")
        for name in RECIPE_NAMES
    ]
    selected, report = select_recipe(denied, required_types=REQUIRED)
    assert selected is None
    assert report["passed"] is False
    assert report["result_is_invalid_for_recipe_comparison"] is True
    assert set(report["recipes_stopped_inside_warmup"]) == set(RECIPE_NAMES)
    assert "NOT evidence about those recipes" in report["stop_reason"]


def test_a_genuine_failure_after_full_warmup_is_not_flagged_as_invalid() -> None:
    honest = [
        _result(name, exact_f1=0.0, exact_recall=0.0, predicted_mentions=0,
                positive_cell_accuracy=0.0, thw_predictions_by_type={},
                schedule=_schedule(peak=True),
                stopped_reason=TINY_STOP_EPOCH_BOUND)
        for name in RECIPE_NAMES
    ]
    _selected, report = select_recipe(honest, required_types=REQUIRED)
    assert report["recipes_stopped_inside_warmup"] == []
    assert "result_is_invalid_for_recipe_comparison" not in report


# ---------------------------------------------------------------------------
# Real save/reload reproduction
# ---------------------------------------------------------------------------


def test_reproduction_requires_a_fresh_model_evaluation() -> None:
    assert _reproduction().reproduced
    assert not _reproduction(fresh_model_evaluated=False).reproduced
    assert not _reproduction(examples_evaluated=0).reproduced


def test_state_dict_key_names_alone_cannot_pass_reproduction() -> None:
    """The removed check was ``set(reloaded["model_state"]) == {...}``.

    There is no constructor path that accepts a key-name comparison: the metrics
    are mandatory, and omitting them raises rather than defaulting to success.
    """
    with pytest.raises(GateError, match="not a key-name comparison"):
        ReproductionCheck(
            checkpoint_sha256="a" * 64,
            metrics_before={"model_state_keys": 2.0},
            metrics_after={"model_state_keys": 2.0},
            missing_keys=(), unexpected_keys=(),
            fresh_model_evaluated=True, examples_evaluated=12)
    with pytest.raises(GateError, match="reproduction check"):
        assert_real_reproduction(None)


def test_reproduction_fails_when_a_metric_drifts_beyond_tolerance() -> None:
    drifted = _reproduction(reproduced=False)
    assert not drifted.reproduced
    assert any("exact_f1 drifted" in f for f in drifted.failures())
    assert drifted.differences()["exact_f1"] == pytest.approx(0.6)


def test_reproduction_fails_on_missing_or_unexpected_keys() -> None:
    assert not _reproduction(missing_keys=("encoder.weight",)).reproduced
    assert not _reproduction(unexpected_keys=("lm_head.bias",)).reproduced
    report = _reproduction(missing_keys=("encoder.weight",))
    assert any("missing state-dict keys" in f for f in report.failures())


def test_reproduction_tolerance_is_explicit_and_recorded() -> None:
    payload = _reproduction().as_dict()
    assert payload["tolerance"] == pytest.approx(1e-6)
    assert payload["reproduced"] is True
    assert payload["fresh_model_evaluated"] is True
    assert payload["examples_evaluated"] == 12
    assert payload["state_dict_key_names_alone_are_sufficient"] is False
    assert set(payload["differences"]) == {
        "exact_precision", "exact_recall", "exact_f1", "predicted_mentions",
        "positive_cell_accuracy"}


def test_a_recipe_without_a_reproduction_check_cannot_pass() -> None:
    unproven = _result(REFERENCE_CE, reproduction=None)
    passed, failures = unproven.passes(required_types=REQUIRED)
    assert not passed
    assert any("no fresh-model re-evaluation" in f for f in failures)


def test_a_recipe_with_a_failing_reproduction_cannot_pass() -> None:
    drifted = _result(REFERENCE_CE, reproduction=_reproduction(reproduced=False))
    passed, failures = drifted.passes(required_types=REQUIRED)
    assert not passed
    assert any("save/reload did not reproduce" in f for f in failures)


# ---------------------------------------------------------------------------
# Stale gate handling
# ---------------------------------------------------------------------------


def test_the_stale_failed_gate_cannot_authorize_stage_four(tmp_path: Path) -> None:
    """The previous Stage-2 gate was written by the premature-stop code."""
    _write_gates(tmp_path, tiny_passed=False, code="stale-premature-stop-code")
    with pytest.raises(GateError, match="did not pass"):
        assert_full_training_allowed(gate_dir=tmp_path, **FULL_OK)


def test_even_a_passing_gate_from_the_old_code_hash_is_refused(tmp_path: Path) -> None:
    _write_gates(tmp_path, code="stale-premature-stop-code")
    with pytest.raises(GateError, match="re-run the gated stages"):
        assert_full_training_allowed(gate_dir=tmp_path, **FULL_OK)


def test_a_gate_is_replaced_atomically(tmp_path: Path) -> None:
    """A rerun must not leave the old verdict beside the new one."""
    target = tmp_path / TINY_GATE_FILENAME
    GateArtifact(stage="tiny_recipe_ablation", passed=False, recipe="",
                 config_sha256="cfg", code_sha256="old", corpus_sha256={},
                 detail={"run": "premature-stop"}).write(target)
    assert json.loads(target.read_text(encoding="utf-8"))["passed"] is False

    GateArtifact(stage="tiny_recipe_ablation", passed=True, recipe=REFERENCE_CE,
                 config_sha256="cfg", code_sha256="new", corpus_sha256={},
                 detail={"run": "repaired"}).write(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["code_sha256"] == "new"
    assert payload["detail"]["run"] == "repaired"
    # No temporary file survives, and no second gate file was created.
    assert sorted(p.name for p in tmp_path.iterdir()) == [TINY_GATE_FILENAME]


# ---------------------------------------------------------------------------
# The notebook uses the repaired contracts
# ---------------------------------------------------------------------------


def test_the_notebook_stage_two_uses_the_tiny_policy_not_validation_patience() -> None:
    code = _notebook_code()
    assert "TinyOverfitStopPolicy(" in code
    assert "tiny_policy=policy" in code
    assert "tiny_policy.decide(" in code
    assert "collapse_guard=False" in code


def test_the_notebook_plans_the_schedule_from_the_full_epoch_bound() -> None:
    code = _notebook_code()
    assert "plan_schedule(" in code
    assert "epoch_bound=TINY_EPOCH_BOUND" in code
    assert "warmup_ratio=recipe.schedule.warmup_ratio" in code


def test_the_notebook_performs_a_real_fresh_model_save_reload() -> None:
    code = _notebook_code()
    assert "def evaluate_state(" in code
    assert "ReproductionCheck(" in code
    assert "fresh_model_evaluated=True" in code
    # The removed key-name check must not come back.
    assert 'set(reloaded["model_state"]) == {"base_model", "w2ner_head"}' not in code
    assert "reproduction=reproduction" in code


def test_the_notebook_still_deletes_candidate_checkpoints() -> None:
    code = _notebook_code()
    assert "tmp.unlink(missing_ok=True)" in code


def test_the_notebook_emits_tiny_heartbeats() -> None:
    code = _notebook_code()
    assert "tiny_heartbeat" in code
    assert "should_heartbeat(" in code
