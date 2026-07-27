"""E4 training progress observability (Audit 0041).

The anchor observation is the real T4 High-RAM run: PhoBERT loaded, the first
forward and backward completed, host RSS stayed low, GPU memory was stable — and
then roughly an hour passed with no output because nothing logged between the
first micro-batch and the end-of-epoch summary.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mednorm_vi.training.phase2.e4_progress import (
    DEFAULT_LOG_EVERY_N_TRAIN_SAMPLES,
    DEFAULT_LOG_EVERY_N_VALIDATION_SAMPLES,
    DEFAULT_LOG_FIRST_N_SAMPLES,
    DEFAULT_ROLLING_LOSS_WINDOW,
    PERSISTENCE_PHASES,
    STAGE_EPOCH_CHECKPOINT_PERSISTED,
    STAGE_EPOCH_START,
    STAGE_EPOCH_TRAINING_COMPLETE,
    STAGE_EPOCH_VALIDATION_COMPLETE,
    STAGE_FULL_TRAINING_COMPLETE,
    STAGE_TRAIN_PROGRESS,
    STAGE_TRAINING_FAILED,
    STAGE_VALIDATION_PROGRESS,
    NullProgressBar,
    ProgressConfig,
    ProgressConfigError,
    ProgressLog,
    RollingLoss,
    emit,
    epoch_checkpoint_persisted_record,
    epoch_start_record,
    epoch_training_complete_record,
    epoch_validation_complete_record,
    eta_seconds,
    full_training_complete_record,
    gpu_memory_snapshot,
    make_progress_bar,
    percent_complete,
    persistence_phase_record,
    progress_bar_postfix,
    rate_per_second,
    should_log_train_sample,
    should_log_validation_sample,
    training_failed_record,
    training_heartbeat,
    validation_heartbeat,
)
from mednorm_vi.training.phase2.e4_runtime_io import (
    GovernedW2NERContractSource,
    assert_not_materialized,
)

REPO = Path(__file__).resolve().parents[2]
E4_NOTEBOOK = REPO / "notebooks" / "MedNorm_E4_PhoBERT_W2NER_Training.ipynb"

FULL_TRAIN_EXAMPLES = 33826
FULL_VALIDATION_EXAMPLES = 1045
FULL_EPOCHS = 12
EXPECTED_OPTIMIZER_STEPS = 50748
EXPECTED_BACKWARD_PASSES = 405912


def _notebook_code() -> list[str]:
    payload = json.loads(E4_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in payload["cells"]
            if cell["cell_type"] == "code"]


# ---------------------------------------------------------------------------
# A. Configuration
# ---------------------------------------------------------------------------


def test_committed_full_run_interval_is_100_not_5() -> None:
    config = ProgressConfig()
    assert config.log_every_n_train_samples == 100
    assert DEFAULT_LOG_EVERY_N_TRAIN_SAMPLES == 100
    assert config.log_every_n_train_samples != 5
    assert config.log_first_n_samples == DEFAULT_LOG_FIRST_N_SAMPLES == 10
    assert config.log_every_n_validation_samples == DEFAULT_LOG_EVERY_N_VALIDATION_SAMPLES == 50
    assert config.enabled is True
    assert config.progress_bar_enabled is True


def _notebook_executable_lines() -> str:
    """Notebook code with comment-only lines removed.

    The narrative comment deliberately documents the five-sample debug option, so
    the "not the committed default" assertion must read executable code.
    """
    return "\n".join(
        line for source in _notebook_code() for line in source.splitlines()
        if not line.strip().startswith("#"))


def test_notebook_commits_the_conservative_defaults() -> None:
    executable = _notebook_executable_lines()
    assert "PROGRESS_ENABLED = True" in executable
    assert "PROGRESS_LOG_FIRST_N_SAMPLES = 10" in executable
    assert "PROGRESS_LOG_EVERY_N_TRAIN_SAMPLES = 100" in executable
    assert "PROGRESS_LOG_EVERY_N_VALIDATION_SAMPLES = 50" in executable
    assert "PROGRESS_BAR_ENABLED = True" in executable
    assert "PROGRESS_LOG_EVERY_N_TRAIN_SAMPLES = 5\n" not in executable + "\n"
    # The five-sample debug option is documented in prose for the operator.
    assert "PROGRESS_LOG_EVERY_N_TRAIN_SAMPLES = 5" in "\n".join(_notebook_code())


def test_an_operator_may_set_a_five_sample_debug_interval() -> None:
    config = ProgressConfig(log_every_n_train_samples=5)
    assert config.log_every_n_train_samples == 5
    assert should_log_train_sample(config, 15, 100) is True


def test_first_n_may_be_zero_but_never_negative() -> None:
    assert ProgressConfig(log_first_n_samples=0).log_first_n_samples == 0
    with pytest.raises(ProgressConfigError, match="log_first_n_samples"):
        ProgressConfig(log_first_n_samples=-1)


def test_intervals_must_be_positive_integers() -> None:
    for kwargs in ({"log_every_n_train_samples": 0},
                   {"log_every_n_validation_samples": -5},
                   {"rolling_loss_window": 0},
                   {"log_every_n_train_samples": True}):
        with pytest.raises(ProgressConfigError):
            ProgressConfig(**kwargs)  # type: ignore[arg-type]


def test_progress_configuration_is_recorded_for_config_and_manifest() -> None:
    payload = ProgressConfig().as_dict()
    for key in ("progress_enabled", "progress_log_first_n_samples",
                "progress_log_every_n_train_samples",
                "progress_log_every_n_validation_samples",
                "progress_bar_enabled", "progress_config_version"):
        assert key in payload
    joined = "\n".join(_notebook_code())
    assert "progress=PROGRESS.as_dict()" in joined          # resolved_config.json
    assert '"progress": {**PROGRESS.as_dict()' in joined     # manifest accounting


# ---------------------------------------------------------------------------
# B. Training heartbeat cadence
# ---------------------------------------------------------------------------


def test_first_ten_samples_each_emit_a_heartbeat() -> None:
    config = ProgressConfig()
    for sample in range(1, 11):
        assert should_log_train_sample(config, sample, FULL_TRAIN_EXAMPLES) is True


def test_samples_between_eleven_and_ninetynine_do_not_emit() -> None:
    config = ProgressConfig()
    for sample in range(11, 100):
        assert should_log_train_sample(config, sample, FULL_TRAIN_EXAMPLES) is False


def test_every_hundredth_sample_emits() -> None:
    config = ProgressConfig()
    for sample in (100, 200, 300, 1000, 33800):
        assert should_log_train_sample(config, sample, FULL_TRAIN_EXAMPLES) is True


def test_the_final_epoch_sample_always_emits() -> None:
    config = ProgressConfig()
    assert FULL_TRAIN_EXAMPLES % config.log_every_n_train_samples != 0
    assert should_log_train_sample(config, FULL_TRAIN_EXAMPLES, FULL_TRAIN_EXAMPLES) is True


def test_disabled_progress_emits_nothing() -> None:
    config = ProgressConfig(enabled=False)
    for sample in (1, 5, 100, FULL_TRAIN_EXAMPLES):
        assert should_log_train_sample(config, sample, FULL_TRAIN_EXAMPLES) is False
        assert should_log_validation_sample(config, sample, FULL_VALIDATION_EXAMPLES) is False


def test_heartbeat_volume_is_bounded_per_epoch() -> None:
    config = ProgressConfig()
    hits = sum(1 for sample in range(1, FULL_TRAIN_EXAMPLES + 1)
               if should_log_train_sample(config, sample, FULL_TRAIN_EXAMPLES))
    at_five = sum(1 for sample in range(1, FULL_TRAIN_EXAMPLES + 1)
                  if should_log_train_sample(
                      ProgressConfig(log_every_n_train_samples=5),
                      sample, FULL_TRAIN_EXAMPLES))
    assert hits == 349
    assert at_five > 6000  # why 5 is not the committed default
    assert hits * FULL_EPOCHS < 5000


def test_heartbeat_contains_every_required_field() -> None:
    record = training_heartbeat(
        run_mode="full", epoch=1, total_epochs=12, sample=100,
        total_samples=FULL_TRAIN_EXAMPLES, accumulation_slot=4,
        gradient_accumulation_steps=8, epoch_backward_passes=100,
        global_backward_passes=100, epoch_optimizer_steps=12,
        global_optimizer_steps=12, current_loss=1.5, rolling_mean_loss=1.6,
        epoch_elapsed_seconds=50.0, run_elapsed_seconds=60.0,
        learning_rate=2e-5, precision_mode="bf16", gpu_name="Tesla T4")
    required = {
        "stage", "run_mode", "epoch", "total_epochs", "sample", "total_samples",
        "epoch_percent", "accumulation_slot", "gradient_accumulation_steps",
        "epoch_backward_passes", "global_backward_passes", "epoch_optimizer_steps",
        "global_optimizer_steps", "current_loss", "rolling_mean_loss",
        "samples_per_second", "optimizer_steps_per_second", "epoch_elapsed_seconds",
        "run_elapsed_seconds", "epoch_eta_seconds", "run_eta_seconds",
        "learning_rate", "precision_mode", "gpu_name", "gpu_allocated_gib",
        "gpu_reserved_gib", "gpu_peak_allocated_gib", "process_rss_gib",
        "system_available_gib"}
    assert required <= set(record)
    assert record["stage"] == STAGE_TRAIN_PROGRESS


def test_sample_and_epoch_numbers_are_one_based() -> None:
    record = training_heartbeat(
        run_mode="full", epoch=1, total_epochs=12, sample=1, total_samples=10,
        accumulation_slot=1, gradient_accumulation_steps=8, epoch_backward_passes=1,
        global_backward_passes=1, epoch_optimizer_steps=0, global_optimizer_steps=0,
        current_loss=1.0, rolling_mean_loss=1.0, epoch_elapsed_seconds=1.0,
        run_elapsed_seconds=1.0, learning_rate=2e-5, precision_mode="bf16",
        gpu_name="")
    assert record["epoch"] == 1
    assert record["sample"] == 1
    assert record["accumulation_slot"] >= 1


# ---------------------------------------------------------------------------
# Bounded arithmetic
# ---------------------------------------------------------------------------


def test_zero_elapsed_time_never_divides_by_zero() -> None:
    assert rate_per_second(10, 0.0) == 0.0
    assert rate_per_second(0, 0.0) == 0.0
    assert eta_seconds(5, 100, 0.0) == 0.0


def test_eta_is_non_negative_and_zero_when_complete() -> None:
    assert eta_seconds(50, 100, 10.0) == pytest.approx(10.0)
    assert eta_seconds(100, 100, 10.0) == 0.0
    assert eta_seconds(150, 100, 10.0) == 0.0
    assert eta_seconds(-1, 100, 10.0) == 0.0


def test_percentages_are_bounded_between_zero_and_one_hundred() -> None:
    assert percent_complete(0, 100) == 0.0
    assert percent_complete(50, 100) == 50.0
    assert percent_complete(150, 100) == 100.0
    assert percent_complete(5, 0) == 0.0


def test_heartbeat_eta_fields_are_non_negative() -> None:
    record = training_heartbeat(
        run_mode="full", epoch=12, total_epochs=12, sample=FULL_TRAIN_EXAMPLES,
        total_samples=FULL_TRAIN_EXAMPLES, accumulation_slot=2,
        gradient_accumulation_steps=8, epoch_backward_passes=FULL_TRAIN_EXAMPLES,
        global_backward_passes=EXPECTED_BACKWARD_PASSES, epoch_optimizer_steps=4229,
        global_optimizer_steps=EXPECTED_OPTIMIZER_STEPS, current_loss=0.2,
        rolling_mean_loss=0.25, epoch_elapsed_seconds=3600.0,
        run_elapsed_seconds=43200.0, learning_rate=2e-5, precision_mode="bf16",
        gpu_name="NVIDIA A100")
    assert record["epoch_eta_seconds"] >= 0.0
    assert record["run_eta_seconds"] >= 0.0
    assert record["epoch_percent"] == 100.0


def test_rolling_loss_is_bounded_in_memory() -> None:
    rolling = RollingLoss(window=100)
    for index in range(10_000):
        rolling.observe(float(index))
    assert len(rolling) == 100
    assert rolling.mean == pytest.approx(sum(range(9900, 10000)) / 100)
    assert RollingLoss(window=DEFAULT_ROLLING_LOSS_WINDOW).mean == 0.0


# ---------------------------------------------------------------------------
# D. Validation heartbeat
# ---------------------------------------------------------------------------


def test_validation_emits_at_sample_one_every_fifty_and_the_last_sample() -> None:
    config = ProgressConfig()
    assert should_log_validation_sample(config, 1, FULL_VALIDATION_EXAMPLES) is True
    for sample in (50, 100, 1000):
        assert should_log_validation_sample(config, sample, FULL_VALIDATION_EXAMPLES) is True
    assert should_log_validation_sample(
        config, FULL_VALIDATION_EXAMPLES, FULL_VALIDATION_EXAMPLES) is True
    for sample in (2, 25, 49, 51):
        assert should_log_validation_sample(config, sample, FULL_VALIDATION_EXAMPLES) is False


def test_validation_heartbeat_reports_counts_only() -> None:
    record = validation_heartbeat(
        epoch=3, total_epochs=12, sample=50, total_samples=FULL_VALIDATION_EXAMPLES,
        elapsed_seconds=25.0, predicted_mentions_so_far=120,
        gold_mentions_so_far=140)
    required = {"stage", "epoch", "total_epochs", "sample", "total_samples",
                "validation_percent", "elapsed_seconds", "samples_per_second",
                "eta_seconds", "predicted_mentions_so_far", "gold_mentions_so_far",
                "gpu_allocated_gib", "gpu_reserved_gib", "process_rss_gib",
                "system_available_gib"}
    assert required <= set(record)
    assert record["stage"] == STAGE_VALIDATION_PROGRESS
    assert record["internal_test_accessed"] is False
    # Counts only: no predictions, spans or text.
    assert all(not isinstance(value, (list, dict)) for value in record.values())


# ---------------------------------------------------------------------------
# E/F. Lifecycle records
# ---------------------------------------------------------------------------


def test_epoch_lifecycle_records_are_complete() -> None:
    start = epoch_start_record(
        epoch=1, total_epochs=12, train_examples=FULL_TRAIN_EXAMPLES,
        expected_optimizer_steps_this_epoch=4229,
        expected_backward_passes_this_epoch=FULL_TRAIN_EXAMPLES,
        resume_start_epoch=1, precision_mode="bf16", gpu_name="NVIDIA A100")
    assert start["stage"] == STAGE_EPOCH_START
    assert {"train_examples", "expected_optimizer_steps_this_epoch",
            "expected_backward_passes_this_epoch", "resume_start_epoch",
            "timestamp_utc"} <= set(start)

    trained = epoch_training_complete_record(
        epoch=1, backward_passes=FULL_TRAIN_EXAMPLES, optimizer_steps=4229,
        mean_training_loss=0.5, elapsed_seconds=100.0, samples=FULL_TRAIN_EXAMPLES)
    assert trained["stage"] == STAGE_EPOCH_TRAINING_COMPLETE
    assert trained["samples_per_second"] > 0

    validated = epoch_validation_complete_record(
        epoch=1, exact_precision=0.4, exact_recall=0.3, exact_f1=0.34,
        best_f1_before_epoch=0.2)
    assert validated["stage"] == STAGE_EPOCH_VALIDATION_COMPLETE
    assert validated["is_new_best"] is True
    assert epoch_validation_complete_record(
        epoch=2, exact_precision=0.1, exact_recall=0.1, exact_f1=0.1,
        best_f1_before_epoch=0.34)["is_new_best"] is False

    persisted = epoch_checkpoint_persisted_record(
        epoch=1, latest_checkpoint_path="/drive/latest.pt",
        latest_checkpoint_sha256="a" * 64, best_checkpoint_updated=True,
        best_checkpoint_path="/drive/best.pt", persistent_custody_verified=True,
        elapsed_seconds=12.0)
    assert persisted["stage"] == STAGE_EPOCH_CHECKPOINT_PERSISTED
    assert persisted["persistent_custody_verified"] is True


def test_every_persistence_phase_message_exists() -> None:
    assert PERSISTENCE_PHASES == (
        "saving checkpoint to local staging", "validating checkpoint reload",
        "checking Drive health", "syncing checkpoint to Drive",
        "verifying persistent SHA-256", "persistence complete")
    for phase in PERSISTENCE_PHASES:
        record = persistence_phase_record(phase, epoch=1)
        assert record["phase"] == phase
    with pytest.raises(ProgressConfigError):
        persistence_phase_record("unknown phase", epoch=1)


def test_final_run_summary_is_complete() -> None:
    record = full_training_complete_record(
        epochs_completed=12, global_backward_passes=EXPECTED_BACKWARD_PASSES,
        global_optimizer_steps=EXPECTED_OPTIMIZER_STEPS,
        best_validation_exact_f1=0.42, best_epoch=9, total_elapsed_seconds=40000.0,
        total_samples_processed=EXPECTED_BACKWARD_PASSES,
        best_checkpoint_sha256="a" * 64, latest_checkpoint_sha256="b" * 64,
        artifact_validator_ok=True)
    assert record["stage"] == STAGE_FULL_TRAINING_COMPLETE
    assert record["global_optimizer_steps"] == EXPECTED_OPTIMIZER_STEPS
    assert record["global_backward_passes"] == EXPECTED_BACKWARD_PASSES
    assert record["average_samples_per_second"] > 0
    assert record["internal_test_accessed"] is False


# ---------------------------------------------------------------------------
# G. Local progress log
# ---------------------------------------------------------------------------


def test_progress_log_writes_valid_jsonl_locally(tmp_path: Path) -> None:
    log = ProgressLog(tmp_path / "logs" / "training_progress.jsonl")
    for sample in (1, 2, 3):
        log.append({"stage": STAGE_TRAIN_PROGRESS, "sample": sample})
    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert json.loads(line)["stage"] == STAGE_TRAIN_PROGRESS
    assert log.records_written == 3
    assert log.write_failures == 0


def test_a_progress_log_failure_is_reported_but_never_fatal(tmp_path: Path) -> None:
    log = ProgressLog(tmp_path / "progress.jsonl")
    log.path.unlink()
    log.path.mkdir()  # writing to a directory fails
    assert log.append({"stage": "x"}) is False
    assert log.write_failures == 1
    assert log.last_error
    assert "progress_write_failures" in log.as_dict()


def test_disabled_progress_log_writes_nothing(tmp_path: Path) -> None:
    log = ProgressLog(tmp_path / "progress.jsonl", enabled=False)
    assert log.append({"stage": "x"}) is False
    assert not log.path.exists()


def test_emit_prints_flushed_and_appends(tmp_path: Path, capsys) -> None:
    log = ProgressLog(tmp_path / "progress.jsonl")
    emit({"stage": STAGE_TRAIN_PROGRESS, "sample": 1}, log=log)
    captured = capsys.readouterr().out.strip()
    assert json.loads(captured)["sample"] == 1
    assert log.records_written == 1


def test_progress_log_is_local_during_an_epoch_and_synced_at_boundaries() -> None:
    joined = "\n".join(_notebook_code())
    assert "RUNTIME_LOGS_DIR / DEFAULT_PROGRESS_LOG_NAME" in joined
    assert "/content/mednorm_vi_runtime" in joined
    # Synced through the existing verified persistence, only at boundaries.
    assert joined.count('"training_progress.jsonl", PROGRESS_LOG_PATH') == 3
    assert 'OUTPUT_DIR / "logs" / "training_progress.jsonl"' in joined


# ---------------------------------------------------------------------------
# H. Controlled exception reporting
# ---------------------------------------------------------------------------


def test_failure_record_is_complete_and_does_not_claim_resumability() -> None:
    record = training_failed_record(
        epoch=3, sample=1200, global_backward_passes=50_000,
        global_optimizer_steps=6_250, exception=RuntimeError("CUDA out of memory"),
        local_progress_log_path="/content/mednorm_vi_runtime/logs/training_progress.jsonl",
        latest_persistent_checkpoint_available=False)
    required = {"stage", "epoch", "sample", "global_backward_passes",
                "global_optimizer_steps", "exception_type", "exception_message",
                "gpu_allocated_gib", "gpu_reserved_gib", "gpu_peak_allocated_gib",
                "process_rss_gib", "system_available_gib", "local_progress_log_path",
                "latest_persistent_checkpoint_available"}
    assert required <= set(record)
    assert record["stage"] == STAGE_TRAINING_FAILED
    assert record["exception_type"] == "RuntimeError"
    assert record["latest_persistent_checkpoint_available"] is False


def test_notebook_reports_then_reraises_without_suppressing() -> None:
    joined = "\n".join(_notebook_code())
    assert "training_failed_record(" in joined
    assert "except BaseException as training_error:" in joined
    # The bare `raise` re-raises the original exception; nothing swallows it.
    body = joined[joined.index("except BaseException as training_error:"):]
    assert "\n        raise" in body
    assert "pass  # noqa" not in body


# ---------------------------------------------------------------------------
# C. Progress bar
# ---------------------------------------------------------------------------


def test_progress_bar_can_be_disabled() -> None:
    bar = make_progress_bar(100, "train", enabled=False)
    assert isinstance(bar, NullProgressBar)
    bar.update(1)
    bar.set_postfix(loss="0.1")
    bar.close()
    assert bar.count == 1 and bar.closed is True


def test_progress_bar_total_is_the_streamed_example_count() -> None:
    bar = make_progress_bar(FULL_TRAIN_EXAMPLES, "train epoch 1/12", enabled=False)
    assert bar.total == FULL_TRAIN_EXAMPLES


def test_progress_bar_postfix_shows_the_required_fields() -> None:
    postfix = progress_bar_postfix(
        rolling_mean_loss=0.1234, optimizer_steps=42,
        samples_per_second=3.5, eta_seconds_value=120.0)
    assert set(postfix) == {"loss", "opt_steps", "samples/s", "eta_s"}
    assert progress_bar_postfix(
        rolling_mean_loss=0.0, optimizer_steps=0, samples_per_second=0.0,
        eta_seconds_value=-5.0)["eta_s"] == "0"


def test_notebook_uses_one_bar_per_epoch_and_per_validation_pass() -> None:
    joined = "\n".join(_notebook_code())
    assert joined.count("make_progress_bar(") == 2
    assert 'f"train epoch {epoch}/{epochs}"' in joined
    assert 'f"validation epoch {epoch}/{total_epochs}"' in joined
    assert "bar.close()" in joined
    assert "PROGRESS.progress_bar_enabled" in joined


# ---------------------------------------------------------------------------
# I. Training semantics are untouched
# ---------------------------------------------------------------------------


def test_progress_code_never_materializes_a_streamed_source(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"example_id": "a", "text": "x"}) + "\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = GovernedW2NERContractSource(
        split="train", path=str(path), expected_sha256=digest,
        expected_example_count=1, tokenizer=object(),
        build_contract=lambda split, row: (object(), {"atomic_word_count": 1}),
        max_words=256, max_model_tokens=512, input_contract_version="v1")
    assert_not_materialized(source, label="train_source")
    joined = "\n".join(_notebook_code())
    assert "list(train_source" not in joined
    assert "list(validation_source" not in joined
    assert "assert_not_materialized(" in joined


def test_a_new_iterator_is_still_created_per_epoch_and_validation_pass() -> None:
    joined = "\n".join(_notebook_code())
    assert "train_source.iter_contracts()" in joined
    assert "validation_source.iter_contracts()" in joined


def test_optimizer_and_backward_accounting_is_unchanged() -> None:
    from mednorm_vi.training.phase2.e4_w2ner_training import plan_gradient_accumulation

    plan = plan_gradient_accumulation(
        FULL_TRAIN_EXAMPLES, micro_batch_size=1, accumulation_steps=8,
        epochs=FULL_EPOCHS)
    assert plan.expected_optimizer_steps == EXPECTED_OPTIMIZER_STEPS
    assert plan.expected_backward_passes == EXPECTED_BACKWARD_PASSES
    assert plan.final_partial_group_size == 2
    assert plan.effective_batch_size == 8


def test_progress_logging_does_not_alter_step_counters() -> None:
    """A heartbeat is a pure read: counters are inputs, never mutated."""
    counters = {"backward": 100, "optimizer": 12}
    record = training_heartbeat(
        run_mode="full", epoch=1, total_epochs=12, sample=100,
        total_samples=FULL_TRAIN_EXAMPLES, accumulation_slot=4,
        gradient_accumulation_steps=8,
        epoch_backward_passes=counters["backward"],
        global_backward_passes=counters["backward"],
        epoch_optimizer_steps=counters["optimizer"],
        global_optimizer_steps=counters["optimizer"], current_loss=1.0,
        rolling_mean_loss=1.0, epoch_elapsed_seconds=1.0, run_elapsed_seconds=1.0,
        learning_rate=2e-5, precision_mode="bf16", gpu_name="")
    assert counters == {"backward": 100, "optimizer": 12}
    assert record["global_optimizer_steps"] == 12


def test_notebook_preserves_the_training_contract() -> None:
    joined = "\n".join(_notebook_code())
    for marker in ("MICRO_BATCH_SIZE = 1", "GRADIENT_ACCUMULATION_STEPS = 8",
                   "FULL_EPOCHS = 12", "ACCUMULATION.loss_scale_for(micro_batch_index)",
                   "ACCUMULATION.is_optimizer_step_boundary(micro_batch_index)",
                   "clip_grad_norm_(trainable, MAX_GRAD_NORM)",
                   "use_safetensors=WEIGHT_FORMAT.use_safetensors",
                   "assert_optimizer_step_accounting("):
        assert marker in joined
    # No per-sample garbage collection or cache clearing was added.
    assert "gc.collect()" not in joined
    assert "empty_cache()" not in joined


def test_precision_resolution_is_unchanged() -> None:
    from mednorm_vi.training.phase2.e4_w2ner_training import (
        DEVICE_CUDA,
        PRECISION_BF16,
        PRECISION_FP16,
        resolve_mixed_precision_policy,
    )

    t4 = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=False)
    assert t4.mode == PRECISION_FP16 and t4.use_grad_scaler is True
    a100 = resolve_mixed_precision_policy(
        PRECISION_BF16, device_type=DEVICE_CUDA, bf16_supported=True)
    assert a100.mode == PRECISION_BF16 and a100.use_grad_scaler is False


def test_gpu_snapshot_degrades_without_cuda() -> None:
    snapshot = gpu_memory_snapshot(None, "cpu")
    assert snapshot == {"gpu_allocated_gib": -1.0, "gpu_reserved_gib": -1.0,
                        "gpu_peak_allocated_gib": -1.0}


# ---------------------------------------------------------------------------
# Privacy and hygiene
# ---------------------------------------------------------------------------


def test_no_corpus_text_is_logged() -> None:
    source = (REPO / "src" / "mednorm_vi" / "training" / "phase2" / "e4_progress.py"
              ).read_text(encoding="utf-8")
    for forbidden in ('row["text"]', "original_text", "entity.text", "gold_spans",
                      "contract.grid", "predicted_spans"):
        assert forbidden not in source
    record = training_heartbeat(
        run_mode="full", epoch=1, total_epochs=12, sample=1, total_samples=10,
        accumulation_slot=1, gradient_accumulation_steps=8, epoch_backward_passes=1,
        global_backward_passes=1, epoch_optimizer_steps=0, global_optimizer_steps=0,
        current_loss=1.0, rolling_mean_loss=1.0, epoch_elapsed_seconds=1.0,
        run_elapsed_seconds=1.0, learning_rate=2e-5, precision_mode="bf16",
        gpu_name="Tesla T4")
    for value in record.values():
        assert not isinstance(value, (list, dict, tuple))


def test_no_internal_test_access() -> None:
    source = (REPO / "src" / "mednorm_vi" / "training" / "phase2" / "e4_progress.py"
              ).read_text(encoding="utf-8")
    assert "internal_test.jsonl" not in source
    joined = "\n".join(_notebook_code())
    assert "internal_test.jsonl" not in joined


def test_every_notebook_code_cell_parses() -> None:
    for index, source in enumerate(_notebook_code()):
        compile(source, f"e4_cell_{index}", "exec")


def test_no_checkpoint_model_cache_or_archive_is_tracked_in_git() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in tracked.splitlines():
        assert not line.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip"))
        assert not line.startswith(
            ("artifacts/", "weights/", "caches/", "checkpoint/", ".claude/"))
        assert Path(line).name not in {"CLAUDE.md", "AGENTS.md"}
