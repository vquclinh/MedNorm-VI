"""E4 training progress observability (Audit 0041).

A real T4 High-RAM full-training attempt loaded PhoBERT-large, completed its first
forward and backward pass, printed ``memory_after_first_micro_batch`` with low host
RSS and stable GPU memory — and then produced **no output for roughly an hour**.
``checkpoints/`` and ``logs/`` were empty because epoch 1 had not completed, so
there was no way to tell progress from a hang. The operator stopped the runtime.

Nothing was wrong with the loop. It simply had no logging between the first
micro-batch and the end-of-epoch summary. This module supplies that missing
telemetry and **nothing else**: no training semantics, accounting, data order,
precision or checkpoint behaviour is touched.

Everything here is pure and dependency-light so it is fully unit-testable:
Torch is injected where GPU statistics are wanted, ``tqdm`` is optional with a
lightweight fallback, and no function ever sees corpus text.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_CONFIG_VERSION = "e4-progress-v1"

# Committed full-run defaults.
#
# The first ten samples are logged individually so an operator sees motion within
# seconds of the epoch starting. After that the interval is **100**, not 5:
# a five-sample interval over 405,912 backward passes would emit ~81,000 notebook
# lines per run and force a GPU read far more often than necessary. An operator
# may temporarily set the interval to 5 for smoke/debug without touching training
# semantics.
DEFAULT_PROGRESS_ENABLED = True
DEFAULT_LOG_FIRST_N_SAMPLES = 10
DEFAULT_LOG_EVERY_N_TRAIN_SAMPLES = 100
DEFAULT_LOG_EVERY_N_VALIDATION_SAMPLES = 50
DEFAULT_PROGRESS_BAR_ENABLED = True

# Bounded rolling-loss window. Individual losses are never retained beyond it.
DEFAULT_ROLLING_LOSS_WINDOW = 100

# Local (non-Drive) structured progress log. It is synced to Drive only at
# governed lifecycle boundaries, never per sample.
DEFAULT_PROGRESS_LOG_NAME = "training_progress.jsonl"

STAGE_TRAIN_PROGRESS = "train_progress"
STAGE_VALIDATION_PROGRESS = "validation_progress"
STAGE_EPOCH_START = "epoch_start"
STAGE_EPOCH_TRAINING_COMPLETE = "epoch_training_complete"
STAGE_EPOCH_VALIDATION_COMPLETE = "epoch_validation_complete"
STAGE_EPOCH_CHECKPOINT_PERSISTED = "epoch_checkpoint_persisted"
STAGE_FULL_TRAINING_COMPLETE = "full_training_complete"
STAGE_TRAINING_FAILED = "training_failed"
STAGE_PERSISTENCE_PHASE = "persistence_phase"

# Named phases printed before each potentially slow persistence step, so
# checkpoint serialization or a Drive sync never looks like a training hang.
PERSISTENCE_PHASES: tuple[str, ...] = (
    "saving checkpoint to local staging",
    "validating checkpoint reload",
    "checking Drive health",
    "syncing checkpoint to Drive",
    "verifying persistent SHA-256",
    "persistence complete",
)


class ProgressConfigError(ValueError):
    """Raised when a progress configuration is not usable."""


@dataclass(frozen=True, slots=True)
class ProgressConfig:
    """Governed progress settings. Recorded in resolved config and manifest."""

    enabled: bool = DEFAULT_PROGRESS_ENABLED
    log_first_n_samples: int = DEFAULT_LOG_FIRST_N_SAMPLES
    log_every_n_train_samples: int = DEFAULT_LOG_EVERY_N_TRAIN_SAMPLES
    log_every_n_validation_samples: int = DEFAULT_LOG_EVERY_N_VALIDATION_SAMPLES
    progress_bar_enabled: bool = DEFAULT_PROGRESS_BAR_ENABLED
    rolling_loss_window: int = DEFAULT_ROLLING_LOSS_WINDOW
    version: str = PROGRESS_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.log_first_n_samples < 0:
            raise ProgressConfigError("log_first_n_samples must be >= 0")
        for name in ("log_every_n_train_samples", "log_every_n_validation_samples",
                     "rolling_loss_window"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ProgressConfigError(f"{name} must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "progress_enabled": self.enabled,
            "progress_log_first_n_samples": self.log_first_n_samples,
            "progress_log_every_n_train_samples": self.log_every_n_train_samples,
            "progress_log_every_n_validation_samples": self.log_every_n_validation_samples,
            "progress_bar_enabled": self.progress_bar_enabled,
            "progress_rolling_loss_window": self.rolling_loss_window,
            "progress_config_version": self.version,
        }


def should_log_sample(
    sample: int, total: int, *, enabled: bool, first_n: int, interval: int,
) -> bool:
    """Heartbeat rule for a **1-based** sample index.

    Logs each of the first ``first_n`` samples, then every ``interval`` samples,
    and always the final sample of the pass.
    """
    if not enabled or sample < 1:
        return False
    if sample <= first_n:
        return True
    if total > 0 and sample >= total:
        return True
    return sample % interval == 0


def should_log_train_sample(config: ProgressConfig, sample: int, total: int) -> bool:
    return should_log_sample(
        sample, total, enabled=config.enabled,
        first_n=config.log_first_n_samples,
        interval=config.log_every_n_train_samples)


def should_log_validation_sample(config: ProgressConfig, sample: int, total: int) -> bool:
    """Validation logs sample 1, every configured interval, and the last sample."""
    return should_log_sample(
        sample, total, enabled=config.enabled, first_n=1,
        interval=config.log_every_n_validation_samples)


@dataclass
class RollingLoss:
    """Bounded mean over the most recent losses. Never retains the full history."""

    window: int = DEFAULT_ROLLING_LOSS_WINDOW
    _values: deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ProgressConfigError("rolling loss window must be positive")
        self._values = deque(maxlen=self.window)

    def observe(self, value: float) -> None:
        self._values.append(float(value))

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def __len__(self) -> int:
        return len(self._values)


# ---------------------------------------------------------------------------
# Rate / ETA arithmetic — always bounded and non-negative
# ---------------------------------------------------------------------------


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def rate_per_second(done: int, elapsed_seconds: float) -> float:
    """Items per second; ``0.0`` when no time has elapsed (never divides by zero)."""
    if done <= 0 or elapsed_seconds <= 0:
        return 0.0
    return _finite(done / elapsed_seconds)


def eta_seconds(done: int, total: int, elapsed_seconds: float) -> float:
    """Remaining seconds, clamped to ``>= 0``; ``0.0`` when it cannot be estimated."""
    if done <= 0 or total <= 0 or elapsed_seconds <= 0 or done >= total:
        return 0.0
    remaining = total - done
    return _finite(max(0.0, remaining * (elapsed_seconds / done)))


def percent_complete(done: int, total: int) -> float:
    """Completion percentage bounded to ``[0, 100]``."""
    if total <= 0 or done <= 0:
        return 0.0
    return round(min(100.0, max(0.0, 100.0 * done / total)), 4)


def utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Memory probes
# ---------------------------------------------------------------------------


def gpu_memory_snapshot(torch_module: Any = None, device_type: str = "cpu") -> dict[str, float]:
    """Allocated / reserved / peak GiB. ``-1.0`` when CUDA is unavailable.

    Reading these counters does not force a CPU/GPU synchronization, and the
    caller only invokes it at heartbeat points anyway.
    """
    unavailable = {"gpu_allocated_gib": -1.0, "gpu_reserved_gib": -1.0,
                   "gpu_peak_allocated_gib": -1.0}
    if torch_module is None or device_type != "cuda":
        return unavailable
    try:
        if not torch_module.cuda.is_available():
            return unavailable
        gib = 1 << 30
        return {
            "gpu_allocated_gib": round(torch_module.cuda.memory_allocated() / gib, 3),
            "gpu_reserved_gib": round(torch_module.cuda.memory_reserved() / gib, 3),
            "gpu_peak_allocated_gib": round(
                torch_module.cuda.max_memory_allocated() / gib, 3),
        }
    except Exception:  # noqa: BLE001 - telemetry must never break training
        return unavailable


def _host_memory(memory_snapshot: Any) -> dict[str, float]:
    if memory_snapshot is None:
        return {"process_rss_gib": -1.0, "system_available_gib": -1.0}
    snapshot = memory_snapshot("progress")
    return {
        "process_rss_gib": float(snapshot.get("rss_gib", -1.0)),
        "system_available_gib": float(snapshot.get("available_gib", -1.0)),
    }


# ---------------------------------------------------------------------------
# Structured records. None of these ever receives corpus text.
# ---------------------------------------------------------------------------


def training_heartbeat(
    *,
    run_mode: str,
    epoch: int,
    total_epochs: int,
    sample: int,
    total_samples: int,
    accumulation_slot: int,
    gradient_accumulation_steps: int,
    epoch_backward_passes: int,
    global_backward_passes: int,
    epoch_optimizer_steps: int,
    global_optimizer_steps: int,
    current_loss: float,
    rolling_mean_loss: float,
    epoch_elapsed_seconds: float,
    run_elapsed_seconds: float,
    learning_rate: float,
    precision_mode: str,
    gpu_name: str,
    gpu_memory: Mapping[str, float] | None = None,
    memory_snapshot: Any = None,
) -> dict[str, Any]:
    """One ``train_progress`` record. Sample and epoch numbers are 1-based."""
    epoch_eta = eta_seconds(sample, total_samples, epoch_elapsed_seconds)
    remaining_epochs = max(0, total_epochs - epoch)
    run_eta = epoch_eta + remaining_epochs * (
        epoch_elapsed_seconds / sample * total_samples if sample > 0
        and epoch_elapsed_seconds > 0 else 0.0)
    return {
        "stage": STAGE_TRAIN_PROGRESS,
        "run_mode": run_mode,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "sample": sample,
        "total_samples": total_samples,
        "epoch_percent": percent_complete(sample, total_samples),
        "accumulation_slot": accumulation_slot,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epoch_backward_passes": epoch_backward_passes,
        "global_backward_passes": global_backward_passes,
        "epoch_optimizer_steps": epoch_optimizer_steps,
        "global_optimizer_steps": global_optimizer_steps,
        "current_loss": round(float(current_loss), 6),
        "rolling_mean_loss": round(float(rolling_mean_loss), 6),
        "samples_per_second": round(rate_per_second(sample, epoch_elapsed_seconds), 4),
        "optimizer_steps_per_second": round(
            rate_per_second(epoch_optimizer_steps, epoch_elapsed_seconds), 4),
        "epoch_elapsed_seconds": round(max(0.0, epoch_elapsed_seconds), 3),
        "run_elapsed_seconds": round(max(0.0, run_elapsed_seconds), 3),
        "epoch_eta_seconds": round(epoch_eta, 3),
        "run_eta_seconds": round(max(0.0, _finite(run_eta)), 3),
        "learning_rate": float(learning_rate),
        "precision_mode": precision_mode,
        "gpu_name": gpu_name,
        **dict(gpu_memory or gpu_memory_snapshot()),
        **_host_memory(memory_snapshot),
        "internal_test_accessed": False,
    }


def validation_heartbeat(
    *,
    epoch: int,
    total_epochs: int,
    sample: int,
    total_samples: int,
    elapsed_seconds: float,
    predicted_mentions_so_far: int,
    gold_mentions_so_far: int,
    gpu_memory: Mapping[str, float] | None = None,
    memory_snapshot: Any = None,
) -> dict[str, Any]:
    """One ``validation_progress`` record. Counts only — never predictions."""
    gpu = dict(gpu_memory or gpu_memory_snapshot())
    return {
        "stage": STAGE_VALIDATION_PROGRESS,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "sample": sample,
        "total_samples": total_samples,
        "validation_percent": percent_complete(sample, total_samples),
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
        "samples_per_second": round(rate_per_second(sample, elapsed_seconds), 4),
        "eta_seconds": round(eta_seconds(sample, total_samples, elapsed_seconds), 3),
        "predicted_mentions_so_far": int(predicted_mentions_so_far),
        "gold_mentions_so_far": int(gold_mentions_so_far),
        "gpu_allocated_gib": gpu.get("gpu_allocated_gib", -1.0),
        "gpu_reserved_gib": gpu.get("gpu_reserved_gib", -1.0),
        **_host_memory(memory_snapshot),
        "internal_test_accessed": False,
    }


def epoch_start_record(
    *, epoch: int, total_epochs: int, train_examples: int,
    expected_optimizer_steps_this_epoch: int, expected_backward_passes_this_epoch: int,
    resume_start_epoch: int, precision_mode: str, gpu_name: str,
) -> dict[str, Any]:
    return {
        "stage": STAGE_EPOCH_START,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "train_examples": train_examples,
        "expected_optimizer_steps_this_epoch": expected_optimizer_steps_this_epoch,
        "expected_backward_passes_this_epoch": expected_backward_passes_this_epoch,
        "resume_start_epoch": resume_start_epoch,
        "precision_mode": precision_mode,
        "gpu_name": gpu_name,
        "timestamp_utc": utc_timestamp(),
        "internal_test_accessed": False,
    }


def epoch_training_complete_record(
    *, epoch: int, backward_passes: int, optimizer_steps: int,
    mean_training_loss: float, elapsed_seconds: float, samples: int,
) -> dict[str, Any]:
    return {
        "stage": STAGE_EPOCH_TRAINING_COMPLETE,
        "epoch": epoch,
        "backward_passes": backward_passes,
        "optimizer_steps": optimizer_steps,
        "mean_training_loss": round(float(mean_training_loss), 6),
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
        "samples_per_second": round(rate_per_second(samples, elapsed_seconds), 4),
        "internal_test_accessed": False,
    }


def epoch_validation_complete_record(
    *, epoch: int, exact_precision: float, exact_recall: float, exact_f1: float,
    best_f1_before_epoch: float,
) -> dict[str, Any]:
    return {
        "stage": STAGE_EPOCH_VALIDATION_COMPLETE,
        "epoch": epoch,
        "exact_precision": round(float(exact_precision), 6),
        "exact_recall": round(float(exact_recall), 6),
        "exact_f1": round(float(exact_f1), 6),
        "best_f1_before_epoch": round(float(best_f1_before_epoch), 6),
        "is_new_best": bool(float(exact_f1) >= float(best_f1_before_epoch)),
        "internal_test_accessed": False,
    }


def epoch_checkpoint_persisted_record(
    *, epoch: int, latest_checkpoint_path: str, latest_checkpoint_sha256: str,
    best_checkpoint_updated: bool, best_checkpoint_path: str,
    persistent_custody_verified: bool, elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "stage": STAGE_EPOCH_CHECKPOINT_PERSISTED,
        "epoch": epoch,
        "latest_checkpoint_path": latest_checkpoint_path,
        "latest_checkpoint_sha256": latest_checkpoint_sha256,
        "best_checkpoint_updated": bool(best_checkpoint_updated),
        "best_checkpoint_path": best_checkpoint_path,
        "persistent_custody_verified": bool(persistent_custody_verified),
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
        "internal_test_accessed": False,
    }


def persistence_phase_record(phase: str, *, epoch: int) -> dict[str, Any]:
    if phase not in PERSISTENCE_PHASES:
        raise ProgressConfigError(f"unknown persistence phase {phase!r}")
    return {
        "stage": STAGE_PERSISTENCE_PHASE,
        "epoch": epoch,
        "phase": phase,
        "timestamp_utc": utc_timestamp(),
    }


def full_training_complete_record(
    *, epochs_completed: int, global_backward_passes: int, global_optimizer_steps: int,
    best_validation_exact_f1: float, best_epoch: int, total_elapsed_seconds: float,
    total_samples_processed: int, best_checkpoint_sha256: str,
    latest_checkpoint_sha256: str, artifact_validator_ok: bool,
) -> dict[str, Any]:
    return {
        "stage": STAGE_FULL_TRAINING_COMPLETE,
        "epochs_completed": epochs_completed,
        "global_backward_passes": global_backward_passes,
        "global_optimizer_steps": global_optimizer_steps,
        "best_validation_exact_f1": round(float(best_validation_exact_f1), 6),
        "best_epoch": best_epoch,
        "total_elapsed_seconds": round(max(0.0, total_elapsed_seconds), 3),
        "average_samples_per_second": round(
            rate_per_second(total_samples_processed, total_elapsed_seconds), 4),
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "latest_checkpoint_sha256": latest_checkpoint_sha256,
        "artifact_validator_ok": bool(artifact_validator_ok),
        "internal_test_accessed": False,
    }


def training_failed_record(
    *, epoch: int, sample: int, global_backward_passes: int,
    global_optimizer_steps: int, exception: BaseException,
    local_progress_log_path: str, latest_persistent_checkpoint_available: bool,
    gpu_memory: Mapping[str, float] | None = None, memory_snapshot: Any = None,
) -> dict[str, Any]:
    """A concise failure record. The caller prints it and then **re-raises**."""
    return {
        "stage": STAGE_TRAINING_FAILED,
        "epoch": epoch,
        "sample": sample,
        "global_backward_passes": global_backward_passes,
        "global_optimizer_steps": global_optimizer_steps,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        **dict(gpu_memory or gpu_memory_snapshot()),
        **_host_memory(memory_snapshot),
        "local_progress_log_path": local_progress_log_path,
        # Never claim resumability without a verified persistent latest.pt.
        "latest_persistent_checkpoint_available": bool(
            latest_persistent_checkpoint_available),
        "internal_test_accessed": False,
    }


# ---------------------------------------------------------------------------
# Local structured progress log
# ---------------------------------------------------------------------------


class ProgressLog:
    """Append-only local JSONL. Flushed per record; never written to Drive per sample.

    A progress-log failure is reported but never allowed to corrupt model or
    optimizer state, so writes are best-effort and record their own errors.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.records_written = 0
        self.write_failures = 0
        self.last_error = ""
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            self.records_written += 1
            return True
        except Exception as error:  # noqa: BLE001 - telemetry must never kill training
            self.write_failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "progress_log_path": str(self.path),
            "progress_log_enabled": self.enabled,
            "progress_records_written": self.records_written,
            "progress_write_failures": self.write_failures,
            "progress_last_error": self.last_error,
        }


def emit(record: Mapping[str, Any], *, log: ProgressLog | None = None,
         printer: Any = None) -> None:
    """Print a record immediately-flushed and append it to the local JSONL."""
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if printer is None:
        print(payload, flush=True)
    else:
        printer(payload)
    if log is not None:
        log.append(record)


# ---------------------------------------------------------------------------
# Notebook-friendly progress bar
# ---------------------------------------------------------------------------


class NullProgressBar:
    """Interface-compatible no-op used when bars are disabled or tqdm is absent."""

    def __init__(self, total: int = 0, description: str = "") -> None:
        self.total = total
        self.description = description
        self.count = 0
        self.closed = False
        self.postfix: dict[str, Any] = {}

    def update(self, amount: int = 1) -> None:
        self.count += amount

    def set_postfix(self, **values: Any) -> None:
        self.postfix.update(values)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> NullProgressBar:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def make_progress_bar(total: int, description: str, *, enabled: bool = True) -> Any:
    """``tqdm.auto`` when available, otherwise a no-op bar.

    tqdm is already present transitively (Transformers depends on it); it is not
    added as a new hard requirement, and its absence silently degrades to
    :class:`NullProgressBar` rather than failing a training run.
    """
    if not enabled:
        return NullProgressBar(total, description)
    try:
        from tqdm.auto import tqdm  # noqa: PLC0415 - optional, resolved at call time
    except Exception:  # noqa: BLE001 - never a training requirement
        return NullProgressBar(total, description)
    return tqdm(total=total, desc=description, leave=True, dynamic_ncols=True)


def progress_bar_postfix(
    *, rolling_mean_loss: float, optimizer_steps: int, samples_per_second: float,
    eta_seconds_value: float,
) -> dict[str, str]:
    """Compact one-line bar suffix: loss, optimizer steps, rate and ETA."""
    return {
        "loss": f"{rolling_mean_loss:.4f}",
        "opt_steps": str(optimizer_steps),
        "samples/s": f"{samples_per_second:.2f}",
        "eta_s": f"{max(0.0, eta_seconds_value):.0f}",
    }


def progress_lifecycle_sync_points() -> Sequence[str]:
    """Where the local progress log is synced to Drive — never per sample."""
    return ("epoch_complete", "training_complete", "controlled_exception")


__all__ = [
    "DEFAULT_LOG_EVERY_N_TRAIN_SAMPLES",
    "DEFAULT_LOG_EVERY_N_VALIDATION_SAMPLES",
    "DEFAULT_LOG_FIRST_N_SAMPLES",
    "DEFAULT_PROGRESS_BAR_ENABLED",
    "DEFAULT_PROGRESS_ENABLED",
    "DEFAULT_PROGRESS_LOG_NAME",
    "DEFAULT_ROLLING_LOSS_WINDOW",
    "PERSISTENCE_PHASES",
    "PROGRESS_CONFIG_VERSION",
    "STAGE_EPOCH_CHECKPOINT_PERSISTED",
    "STAGE_EPOCH_START",
    "STAGE_EPOCH_TRAINING_COMPLETE",
    "STAGE_EPOCH_VALIDATION_COMPLETE",
    "STAGE_FULL_TRAINING_COMPLETE",
    "STAGE_PERSISTENCE_PHASE",
    "STAGE_TRAINING_FAILED",
    "STAGE_TRAIN_PROGRESS",
    "STAGE_VALIDATION_PROGRESS",
    "NullProgressBar",
    "ProgressConfig",
    "ProgressConfigError",
    "ProgressLog",
    "RollingLoss",
    "emit",
    "epoch_checkpoint_persisted_record",
    "epoch_start_record",
    "epoch_training_complete_record",
    "epoch_validation_complete_record",
    "eta_seconds",
    "full_training_complete_record",
    "gpu_memory_snapshot",
    "make_progress_bar",
    "percent_complete",
    "persistence_phase_record",
    "progress_bar_postfix",
    "progress_lifecycle_sync_points",
    "rate_per_second",
    "should_log_sample",
    "should_log_train_sample",
    "should_log_validation_sample",
    "training_failed_record",
    "training_heartbeat",
    "utc_timestamp",
    "validation_heartbeat",
]
