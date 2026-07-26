"""Full S1 mention-training contract tests (Audit 0025).

Pure logic only: no Torch, no downloads, no GPU, no training. The tracked config
is loaded from disk; everything else is synthesized.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mednorm_vi.training.s1_full_training import (
    BEST_CHECKPOINT_NAME,
    BEST_METRIC_KEY,
    BEST_METRIC_MODE,
    CHECKPOINT_REQUIRED_KEYS,
    FULL_TRAINING_MODE,
    LATEST_CHECKPOINT_NAME,
    FullTrainingConfig,
    FullTrainingConfigError,
    MentionMetrics,
    build_checkpoint_payload,
    build_full_training_manifest,
    derive_schedule,
    full_training_output_paths,
    is_better_metric,
    is_supervised_example,
    load_full_training_config,
    validate_resume_checkpoint,
)
from mednorm_vi.training.s1_mention_smoke import ENTITY_TYPE_ORDER

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "training" / "s1_mention_full_training.yaml"
SMOKE_CONFIG_PATH = REPO / "configs" / "training" / "s1_mention_first_run_smoke.yaml"
PINNED = "b" * 40
SMOKE_ARTIFACT_DIR = "/content/drive/MyDrive/MedNorm-VI/artifacts/s1_mention_first_run_smoke"


@pytest.fixture(scope="module")
def config() -> FullTrainingConfig:
    return load_full_training_config(CONFIG_PATH, pinned_revision=PINNED)


def _config_with(tmp_path: Path, **overrides) -> FullTrainingConfig:
    """Load the tracked config with targeted YAML overrides."""
    doc = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    doc["model"]["pinned_revision"] = PINNED
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        doc[section][key] = value
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return load_full_training_config(path)


# --- pinned immutable revision ------------------------------------------------

def test_tracked_config_refuses_to_load_without_a_pinned_revision() -> None:
    """The shipped config deliberately has no revision: it is a BLOCKER until pinned."""
    with pytest.raises(FullTrainingConfigError, match="immutable 40-hex commit hash"):
        load_full_training_config(CONFIG_PATH)


@pytest.mark.parametrize("revision", ["main", "master", "", "v1.0", "b" * 39])
def test_mutable_or_malformed_revisions_are_rejected(tmp_path: Path, revision) -> None:
    with pytest.raises(FullTrainingConfigError, match="immutable"):
        load_full_training_config(CONFIG_PATH, pinned_revision=revision)


def test_pinned_revision_is_used_for_model_and_tokenizer(config) -> None:
    manifest = _manifest(config)
    assert config.pinned_revision == PINNED
    assert manifest["model"]["pinned_model_revision"] == PINNED
    assert manifest["model"]["tokenizer_revision"] == PINNED
    assert manifest["model"]["requested_revision"] == "main"      # honestly recorded


def test_tracked_config_ships_an_empty_pin_not_a_guess() -> None:
    doc = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert doc["model"]["pinned_revision"] == ""
    assert doc["model"]["requested_revision"] == "main"


# --- smoke checkpoint is never an initialization ------------------------------

def test_full_training_initializes_from_the_pretrained_base(config) -> None:
    assert config.initialize_from == "pretrained_base"
    assert _manifest(config)["model"]["initialized_from_smoke_checkpoint"] is False


def test_smoke_checkpoint_as_initialization_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FullTrainingConfigError, match="execution evidence only"):
        _config_with(tmp_path, **{"model.initialize_from": "smoke_checkpoint"})


def test_smoke_checkpoint_is_rejected_as_a_resume_source(config) -> None:
    problems = validate_resume_checkpoint({"mode": "SMOKE_ONLY"}, config)
    assert any("refusing to resume from the SMOKE_ONLY checkpoint" in p for p in problems)


# --- output separation --------------------------------------------------------

def test_output_dir_differs_from_the_smoke_artifact(config) -> None:
    assert config.output_dir != config.smoke_artifact_dir
    assert config.smoke_artifact_dir == SMOKE_ARTIFACT_DIR
    assert config.output_dir.endswith("s1_mention_full_training_v1")


def test_output_dir_equal_to_the_smoke_artifact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FullTrainingConfigError, match="must never be overwritten"):
        _config_with(tmp_path, **{"output.output_dir": SMOKE_ARTIFACT_DIR})


def test_output_paths_separate_latest_best_logs_config_and_manifest(config) -> None:
    paths = full_training_output_paths(config.output_dir)
    assert paths["latest_checkpoint"].endswith(LATEST_CHECKPOINT_NAME)
    assert paths["best_checkpoint"].endswith(BEST_CHECKPOINT_NAME)
    assert paths["latest_checkpoint"] != paths["best_checkpoint"]
    assert len(set(paths.values())) == len(paths)
    for path in paths.values():
        assert path.startswith(config.output_dir)
        assert "s1_mention_first_run_smoke" not in path


# --- checkpoint and resume contracts ------------------------------------------

def test_checkpoint_payload_satisfies_the_resume_contract(config) -> None:
    payload = build_checkpoint_payload(
        config, epoch=2, global_step=1488, best_metric=0.71,
        model_state_dict={"w": 1}, optimizer_state_dict={"o": 1},
        scheduler_state_dict={"s": 1})
    assert payload["mode"] == FULL_TRAINING_MODE
    for key in CHECKPOINT_REQUIRED_KEYS:
        assert key in payload
    assert validate_resume_checkpoint(payload, config) == []


@pytest.mark.parametrize("missing", CHECKPOINT_REQUIRED_KEYS)
def test_incomplete_checkpoint_cannot_be_resumed(config, missing) -> None:
    payload = build_checkpoint_payload(
        config, epoch=1, global_step=10, best_metric=0.1,
        model_state_dict={}, optimizer_state_dict={}, scheduler_state_dict={})
    payload.pop(missing)
    assert validate_resume_checkpoint(payload, config)


def test_checkpoint_from_another_revision_cannot_be_resumed(config) -> None:
    payload = build_checkpoint_payload(
        config, epoch=1, global_step=10, best_metric=0.1,
        model_state_dict={}, optimizer_state_dict={}, scheduler_state_dict={})
    payload["pinned_model_revision"] = "f" * 40
    assert any("was trained on revision" in p for p in validate_resume_checkpoint(payload, config))


def test_checkpoint_with_a_different_label_space_cannot_be_resumed(config) -> None:
    payload = build_checkpoint_payload(
        config, epoch=1, global_step=10, best_metric=0.1,
        model_state_dict={}, optimizer_state_dict={}, scheduler_state_dict={})
    payload["entity_type_order"] = ["DIAGNOSIS"]
    assert any("entity_type_order" in p for p in validate_resume_checkpoint(payload, config))


# --- schedule and batching ----------------------------------------------------

def test_effective_batch_size_and_schedule_match_the_governed_corpus(config) -> None:
    assert config.per_device_batch_size == 16
    assert config.gradient_accumulation_steps == 2
    assert config.effective_batch_size == 32
    # 23,799 supervised train examples in the governed corpus.
    schedule = derive_schedule(config, 23799)
    assert schedule.steps_per_epoch == 744
    assert schedule.total_optimizer_steps == 744 * config.num_epochs
    assert schedule.warmup_steps == int(schedule.total_optimizer_steps * config.warmup_ratio)
    assert 0 < schedule.warmup_steps < schedule.total_optimizer_steps


def test_schedule_requires_supervised_examples(config) -> None:
    with pytest.raises(FullTrainingConfigError, match="no supervised training examples"):
        derive_schedule(config, 0)


def test_unsupervised_examples_are_excluded_from_training() -> None:
    """phoner_covid19 declares boundary=false: its label_mask is entirely zero."""
    assert is_supervised_example({"span": True, "entity_type": True}) is True
    assert is_supervised_example({"span": False, "entity_type": False}) is False
    assert is_supervised_example({"span": True, "entity_type": False}) is False


def test_config_enables_the_unsupervised_filter(config) -> None:
    assert config.filter_unsupervised_examples is True


# --- hyperparameter sanity ----------------------------------------------------

def test_hyperparameters_stay_inside_defensible_ranges(config) -> None:
    assert config.seed == 20260723                      # repository convention
    assert 2e-5 <= config.learning_rate <= 5e-5         # BERT-base fine-tuning band
    assert config.head_learning_rate > config.learning_rate
    assert 3 <= config.num_epochs <= 5
    assert config.max_sequence_length == 256            # PhoBERT position limit
    assert config.loss_type == "focal"                  # spec §15 sanctions BCE/focal
    assert config.mixed_precision == "auto"
    assert config.max_grad_norm == 1.0


@pytest.mark.parametrize("overrides,match", [
    ({"data.max_sequence_length": 512}, "PhoBERT position limit"),
    ({"optimization.learning_rate": 0.5}, "implausible learning_rate"),
    ({"optimization.num_epochs": 0}, "num_epochs"),
    ({"optimization.warmup_ratio": 1.5}, "warmup_ratio"),
    ({"loss.type": "hinge"}, "loss_type"),
    ({"loss.decision_threshold": 0.0}, "decision_threshold"),
    ({"optimization.mixed_precision": "int4"}, "mixed_precision"),
])
def test_invalid_hyperparameters_are_rejected(tmp_path: Path, overrides, match) -> None:
    with pytest.raises(FullTrainingConfigError, match=match):
        _config_with(tmp_path, **overrides)


def test_config_hash_is_deterministic_and_sensitive(tmp_path: Path, config) -> None:
    assert config.config_sha256 == load_full_training_config(
        CONFIG_PATH, pinned_revision=PINNED).config_sha256
    changed = _config_with(tmp_path, **{"optimization.num_epochs": 5})
    assert changed.config_sha256 != config.config_sha256


def test_full_training_corpus_gate_matches_the_smoke_gate() -> None:
    """Full training must consume exactly the corpus the smoke validated."""
    full = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["corpus"]
    smoke = yaml.safe_load(SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))["corpus"]
    assert full == smoke


# --- validation metrics -------------------------------------------------------

def _row(*type_ids_per_token):
    return [[1 if i in ids else 0 for i in range(len(ENTITY_TYPE_ORDER))]
            for ids in type_ids_per_token]


def test_perfect_predictions_score_one() -> None:
    gold = _row((), (0,), (0,), ())
    metrics = MentionMetrics()
    metrics.update([gold], [gold], [[1, 1, 1, 1]])
    result = metrics.compute()
    assert result["token_micro_f1"] == 1.0
    assert result[BEST_METRIC_KEY] == 1.0
    assert result["supervised_tokens"] == 4


def test_wrong_type_is_both_a_false_positive_and_a_false_negative() -> None:
    """Spec §1: a wrong type is double-penalized."""
    gold = _row((0,))
    predicted = _row((1,))
    metrics = MentionMetrics()
    metrics.update([predicted], [gold], [[1]])
    result = metrics.compute()
    assert result["token_micro_precision"] == 0.0
    assert result["token_micro_recall"] == 0.0
    assert result["per_type"][ENTITY_TYPE_ORDER[0]]["false_negative"] == 1
    assert result["per_type"][ENTITY_TYPE_ORDER[1]]["false_positive"] == 1


def test_span_metric_requires_exact_boundaries() -> None:
    gold = _row((), (0,), (0,), ())
    too_long = _row((), (0,), (0,), (0,))
    metrics = MentionMetrics()
    metrics.update([too_long], [gold], [[1, 1, 1, 1]])
    result = metrics.compute()
    assert result[BEST_METRIC_KEY] == 0.0                 # boundary miss, no span credit
    assert result["token_micro_recall"] == 1.0            # but every gold token was found


def test_masked_positions_are_excluded_from_every_metric() -> None:
    gold = _row((0,), (0,))
    predicted = _row((0,), ())
    metrics = MentionMetrics()
    metrics.update([predicted], [gold], [[1, 0]])          # second token is padding
    result = metrics.compute()
    assert result["supervised_tokens"] == 1
    assert result["token_micro_f1"] == 1.0


def test_macro_f1_ignores_entity_types_absent_from_the_data() -> None:
    gold = _row((0,))
    metrics = MentionMetrics()
    metrics.update([gold], [gold], [[1]])
    result = metrics.compute()
    assert result["observed_entity_types"] == [ENTITY_TYPE_ORDER[0]]
    assert result["token_macro_f1"] == 1.0


def test_metrics_accumulate_across_batches() -> None:
    metrics = MentionMetrics()
    metrics.update([_row((0,))], [_row((0,))], [[1]])
    metrics.update([_row(())], [_row((0,))], [[1]])
    result = metrics.compute()
    assert result["token_micro_recall"] == 0.5
    assert result["token_micro_precision"] == 1.0


def test_best_checkpoint_criterion_is_strictly_increasing_span_f1() -> None:
    assert BEST_METRIC_KEY == "validation_span_micro_f1"
    assert BEST_METRIC_MODE == "max"
    assert is_better_metric(0.4, None) is True
    assert is_better_metric(0.5, 0.4) is True
    assert is_better_metric(0.4, 0.4) is False
    assert is_better_metric(0.3, 0.4) is False


# --- manifest -----------------------------------------------------------------

def _manifest(config: FullTrainingConfig, **overrides):
    payload = {
        "schedule": derive_schedule(config, 23799),
        "repository": {"resolved_commit": "c" * 40},
        "corpus": {"corpus_manifest_sha256": "a" * 64},
        "environment": {"dependency_contract_version": "s1-colab-deps-v2"},
        "segmentation": {"word_segmenter": "VnCoreNLP RDRSegmenter"},
        "alignment": {"alignment_backend": "character_offset_reconstruction"},
        "tokenizer": {"tokenizer_class": "PhobertTokenizer", "tokenizer_is_fast": False},
        "completed_epochs": 4,
        "completed_optimizer_steps": 2976,
        "validation_metrics": {BEST_METRIC_KEY: 0.68},
        "checkpoint_hashes": {"latest_checkpoint": "1" * 64, "best_checkpoint": "2" * 64},
        "run_completed": True,
    }
    payload.update(overrides)
    return build_full_training_manifest(config, **payload)


def test_manifest_identifies_everything_needed_to_rebuild_the_run(config) -> None:
    manifest = _manifest(config)
    assert manifest["status"] == FULL_TRAINING_MODE
    assert manifest["smoke_only_not_full_training"] is False
    assert manifest["architecture_spec_version"] == "1.1"
    assert manifest["repository"]["resolved_commit"]
    assert manifest["corpus"]["corpus_manifest_sha256"]
    assert manifest["environment"]["dependency_contract_version"]
    assert manifest["hyperparameters"]["learning_rate"] == config.learning_rate
    assert manifest["effective_batch_size"] == 32
    assert manifest["schedule"]["total_optimizer_steps"] == 2976
    assert manifest["completed_epochs"] == 4
    assert manifest["best_checkpoint_criterion"] == {
        "key": BEST_METRIC_KEY, "mode": BEST_METRIC_MODE}
    assert manifest["artifacts"]["checkpoint_sha256"]["best_checkpoint"] == "2" * 64
    assert manifest["config_sha256"] == config.config_sha256
    assert manifest["word_segmentation"]["word_segmenter"] == "VnCoreNLP RDRSegmenter"
    assert manifest["tokenizer"]["tokenizer_is_fast"] is False


def test_manifest_records_completion_and_resumability(config) -> None:
    completed = _manifest(config)
    assert completed["run_completed"] is True
    assert completed["safe_to_resume"] is True
    assert completed["interrupted_reason"] == ""

    interrupted = _manifest(
        config, run_completed=False, interrupted_reason="CUDA out of memory",
        completed_epochs=2, completed_optimizer_steps=1488)
    assert interrupted["run_completed"] is False
    assert interrupted["interrupted_reason"] == "CUDA out of memory"
    assert interrupted["safe_to_resume"] is True           # latest checkpoint exists

    no_checkpoint = _manifest(config, run_completed=False, checkpoint_hashes={})
    assert no_checkpoint["safe_to_resume"] is False


def test_manifest_never_points_training_output_at_the_smoke_artifact(config) -> None:
    manifest = _manifest(config)
    assert manifest["artifacts"]["smoke_artifact_dir"] == SMOKE_ARTIFACT_DIR
    for key in ("latest_checkpoint", "best_checkpoint", "training_manifest"):
        assert SMOKE_ARTIFACT_DIR not in manifest["artifacts"][key]
