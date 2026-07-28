"""E4 post-training collapse diagnosis (Audit 0043).

The real E4 full run completed 12 epochs, 405,912 backward passes and 50,748
optimizer steps, wrote a schema-valid artifact, and never touched internal_test.
It then failed its quality gate outright: the final validation pass predicted
**zero** mentions against 1,991 gold mentions, and the best epoch reached exact
F1 0.00199.

This module answers *why* with measurements rather than with a guess. It is
strictly read-only:

* it never trains, never loads a pretrained model, never opens internal_test;
* it never mutates, renames or regenerates anything under the immutable local
  artifact directory;
* it reports **counts, offsets and label ids only** — no clinical text ever
  leaves a diagnostic record (:func:`assert_no_clinical_text`).

The five diagnostics are deliberately ordered so that a failure upstream
invalidates everything downstream, and the code says so:

1. :func:`verify_artifact_integrity` — is the artifact the one we think it is?
2. :func:`reconstruct_epoch_history` — when exactly did predictions vanish?
3. :func:`gold_grid_round_trip` — can the target builder and decoder reproduce
   the gold mentions *with no model in the loop at all*? If this fails, no
   statement about model quality is meaningful.
4. :func:`grid_class_distribution` — what does the target actually look like?
5. :func:`probe_checkpoint` — what does the trained classifier emit? This is the
   only diagnostic that needs the checkpoints, and it **fails closed** when they
   are absent rather than substituting an assumption.

:func:`resolve_verdict` will not return ``ALL_BACKGROUND_LOSS_COLLAPSE`` from
prediction counts alone. That verdict requires grid logits and gold-positive-cell
predictions from a real checkpoint probe, so an artifact without checkpoints can
only ever reach ``ROOT_CAUSE_NOT_YET_PROVEN``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...mention_factory.w2ner import (
    W2NER_NEXT_NEIGHBORING_WORD,
    W2NER_NONE,
    W2NER_TAIL_HEAD_WORD_PREFIX,
    EntitySpan,
    W2NERLabelVocab,
    build_w2ner_grid,
    decode_w2ner_grid,
    tokenize_atomic_words,
)
from .common import sha256_file

DIAGNOSIS_VERSION = "e4-collapse-diagnosis-v1"

# internal_test is frozen. Nothing in this module may open it, for any reason.
FORBIDDEN_SPLITS = frozenset({"internal_test"})
DIAGNOSABLE_SPLITS: tuple[str, ...] = ("train", "validation")

# ---------------------------------------------------------------------------
# The immutable local artifact
# ---------------------------------------------------------------------------
#
# Recorded here so a diagnosis can prove it examined the run it claims to have
# examined. These digests come from the completed Colab run's own
# ``validation_metrics.json`` / ``training_manifest.json``; they are expectations
# to check against, never values this module produces.
E4_FULL_ARTIFACT_NAME = "e4_phobert_w2ner_full_v1"
E4_FULL_BEST_CHECKPOINT_SHA256 = (
    "4cc934eb5d072bcf827e46745bbcc308beda3552b4156c4c4504f571aa0bd16f")
E4_FULL_LATEST_CHECKPOINT_SHA256 = (
    "22b9017da3e5a56b7086c7f03dda1aed7e78b5e7b9f844d6171ee9333355bf07")

E4_ARTIFACT_JSON_FILES: tuple[str, ...] = (
    "training_manifest.json",
    "resolved_config.json",
    "validation_metrics.json",
    "grid_target_statistics.json",
    "e4_alignment_diagnostic.json",
)
E4_ARTIFACT_LOG_FILES: tuple[str, ...] = (
    "logs/training_history.jsonl",
    "logs/training_progress.jsonl",
)
E4_ARTIFACT_CHECKPOINT_FILES: tuple[str, ...] = (
    "checkpoints/best.pt",
    "checkpoints/latest.pt",
)

# What the completed run reported. Any disagreement is an integrity failure and
# stops the diagnosis before conclusions are drawn from it.
E4_FULL_EXPECTED: Mapping[str, Any] = {
    "expert_id": "E4_phobert_w2ner",
    "stage_id": "phase2-e4-phobert-w2ner-v2",
    "mode": "full",
    "model_id": "vinai/phobert-large",
    "model_revision": "1c7880f20db59c0054c6de5afd71b012369f6ee4",
    "completed_epochs": 12,
    "best_epoch": 2,
    "optimizer_steps": 50748,
    "backward_passes": 405912,
    "parameter_count": 371289161,
    "internal_test_accessed": False,
}

# Keys a full E4 checkpoint must carry for the probe to mean anything.
CHECKPOINT_REQUIRED_KEYS: tuple[str, ...] = (
    "checkpoint_schema_version",
    "e4_checkpoint_schema_version",
    "e4_input_contract_version",
    "atomic_projection_version",
    "expert_id",
    "mode",
    "config_sha256",
    "model_revision",
    "tokenizer_revision",
    "epoch",
    "optimizer_steps",
    "best_metric",
    "model_state",
    "optimizer_state",
)
CHECKPOINT_MODEL_STATE_KEYS: tuple[str, ...] = ("base_model", "w2ner_head")

# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
TARGET_DECODER_MISMATCH = "TARGET_DECODER_MISMATCH"
LABEL_MAPPING_MISMATCH = "LABEL_MAPPING_MISMATCH"
ALL_BACKGROUND_LOSS_COLLAPSE = "ALL_BACKGROUND_LOSS_COLLAPSE"
CHECKPOINT_RESTORE_FAILURE = "CHECKPOINT_RESTORE_FAILURE"
DECODER_THRESHOLD_FAILURE = "DECODER_THRESHOLD_FAILURE"
MULTIPLE_CONFIRMED_FAILURES = "MULTIPLE_CONFIRMED_FAILURES"
ROOT_CAUSE_NOT_YET_PROVEN = "ROOT_CAUSE_NOT_YET_PROVEN"

VERDICTS: tuple[str, ...] = (
    TARGET_DECODER_MISMATCH,
    LABEL_MAPPING_MISMATCH,
    ALL_BACKGROUND_LOSS_COLLAPSE,
    CHECKPOINT_RESTORE_FAILURE,
    DECODER_THRESHOLD_FAILURE,
    MULTIPLE_CONFIRMED_FAILURES,
    ROOT_CAUSE_NOT_YET_PROVEN,
)

# A value the diagnostic could not read. Never silently replaced with 0 or 0.0:
# "no prediction count was recorded" and "the model predicted nothing" are
# completely different findings and must not print the same way.
UNAVAILABLE = "UNAVAILABLE"


class E4DiagnosisError(RuntimeError):
    """Raised when the diagnosis cannot proceed on the evidence available."""


class CheckpointEvidenceUnavailable(E4DiagnosisError):
    """Raised when a probe is requested but the checkpoint is not present.

    Deliberately its own type: "the classifier emits background" and "we could
    not look at the classifier" must never collapse into one outcome.
    """


# ---------------------------------------------------------------------------
# Text safety
# ---------------------------------------------------------------------------


def assert_no_clinical_text(payload: Any, *, corpus_texts: Iterable[str]) -> None:
    """Fail if any governed document text appears inside a diagnostic payload.

    Diagnostic output is written to disk and pasted into audits, so it is checked
    against the real corpus rather than against a regex guess about what clinical
    text looks like.
    """
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    for text in corpus_texts:
        stripped = text.strip()
        # Short fragments ("ho", "sốt") occur as ordinary substrings of field
        # names and ids; only a substantial verbatim slice is evidence of a leak.
        if len(stripped) >= 24 and stripped in serialized:
            raise E4DiagnosisError(
                "diagnostic payload contains verbatim governed document text")


def assert_split_allowed(split: str) -> str:
    if split in FORBIDDEN_SPLITS:
        raise E4DiagnosisError(f"refusing to open the frozen split {split!r}")
    if split not in DIAGNOSABLE_SPLITS:
        raise E4DiagnosisError(
            f"unknown diagnosable split {split!r}; expected one of {DIAGNOSABLE_SPLITS}")
    return split


# ---------------------------------------------------------------------------
# A. Artifact integrity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactIntegrityReport:
    """What the immutable artifact directory actually contains."""

    artifact_dir: str
    present_files: Mapping[str, str]
    missing_files: tuple[str, ...]
    checkpoints_present: bool
    checkpoint_hash_matches: Mapping[str, bool]
    declared_checkpoint_hashes: Mapping[str, str]
    manifest: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    grid_target_statistics: Mapping[str, Any]
    inconsistencies: tuple[str, ...]
    diagnosis_version: str = DIAGNOSIS_VERSION

    @property
    def ok(self) -> bool:
        """True when nothing disagrees. Absent checkpoints are NOT an
        inconsistency — they are a bounded evidence gap, reported separately."""
        return not self.inconsistencies

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnosis_version": self.diagnosis_version,
            "artifact_dir": self.artifact_dir,
            "ok": self.ok,
            "present_files": dict(self.present_files),
            "missing_files": list(self.missing_files),
            "checkpoints_present": self.checkpoints_present,
            "checkpoint_hash_matches": dict(self.checkpoint_hash_matches),
            "declared_checkpoint_hashes": dict(self.declared_checkpoint_hashes),
            "inconsistencies": list(self.inconsistencies),
            "internal_test_accessed": False,
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E4DiagnosisError(f"{path.name} is not a JSON object")
    return payload


def verify_artifact_integrity(artifact_dir: str | Path) -> ArtifactIntegrityReport:
    """Hash and cross-check the completed E4 artifact. Never writes to it."""
    root = Path(artifact_dir)
    if not root.is_dir():
        raise E4DiagnosisError(f"E4 artifact directory does not exist: {root}")

    present: dict[str, str] = {}
    missing: list[str] = []
    for name in (*E4_ARTIFACT_JSON_FILES, *E4_ARTIFACT_LOG_FILES,
                 *E4_ARTIFACT_CHECKPOINT_FILES):
        path = root / name
        if path.is_file():
            present[name] = sha256_file(path)
        else:
            missing.append(name)

    inconsistencies: list[str] = []
    for name in (*E4_ARTIFACT_JSON_FILES, *E4_ARTIFACT_LOG_FILES):
        if name in missing:
            inconsistencies.append(f"missing required artifact file: {name}")

    manifest = _read_json(root / "training_manifest.json") if (
        root / "training_manifest.json").is_file() else {}
    resolved = _read_json(root / "resolved_config.json") if (
        root / "resolved_config.json").is_file() else {}
    metrics = _read_json(root / "validation_metrics.json") if (
        root / "validation_metrics.json").is_file() else {}
    grid_stats = _read_json(root / "grid_target_statistics.json") if (
        root / "grid_target_statistics.json").is_file() else {}

    declared = dict(manifest.get("checkpoint_hashes", {}))
    declared_metrics = dict(metrics.get("checkpoint_hashes", {}))
    if declared and declared_metrics and declared != declared_metrics:
        inconsistencies.append(
            "training_manifest.json and validation_metrics.json declare different "
            "checkpoint hashes")
    for role, expected in (("best", E4_FULL_BEST_CHECKPOINT_SHA256),
                           ("latest", E4_FULL_LATEST_CHECKPOINT_SHA256)):
        actual = str(declared.get(role, declared_metrics.get(role, "")))
        if actual and actual != expected:
            inconsistencies.append(
                f"declared {role} checkpoint hash {actual} does not match the "
                f"recorded run hash {expected}")

    checkpoints_present = all(
        name in present for name in E4_ARTIFACT_CHECKPOINT_FILES)
    hash_matches: dict[str, bool] = {}
    if checkpoints_present:
        hash_matches["best"] = (
            present["checkpoints/best.pt"] == E4_FULL_BEST_CHECKPOINT_SHA256)
        hash_matches["latest"] = (
            present["checkpoints/latest.pt"] == E4_FULL_LATEST_CHECKPOINT_SHA256)
        for role, matched in hash_matches.items():
            if not matched:
                inconsistencies.append(
                    f"{role}.pt on disk does not match its recorded SHA-256")

    inconsistencies.extend(_manifest_inconsistencies(manifest, resolved, metrics))
    inconsistencies.extend(_label_space_inconsistencies(manifest, resolved, grid_stats))

    return ArtifactIntegrityReport(
        artifact_dir=str(root),
        present_files=present,
        missing_files=tuple(missing),
        checkpoints_present=checkpoints_present,
        checkpoint_hash_matches=hash_matches,
        declared_checkpoint_hashes=declared or declared_metrics,
        manifest=manifest,
        resolved_config=resolved,
        validation_metrics=metrics,
        grid_target_statistics=grid_stats,
        inconsistencies=tuple(inconsistencies),
    )


def _manifest_inconsistencies(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    merged: dict[str, Any] = {
        "expert_id": manifest.get("expert_id"),
        "stage_id": manifest.get("stage_id"),
        "mode": manifest.get("mode"),
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
        "completed_epochs": manifest.get("completed_epochs"),
        "best_epoch": metrics.get("best_epoch"),
        "optimizer_steps": manifest.get("optimizer_steps"),
        "backward_passes": metrics.get("backward_passes"),
        "parameter_count": manifest.get("parameter_count"),
        "internal_test_accessed": manifest.get("internal_test_accessed"),
    }
    for key, expected in E4_FULL_EXPECTED.items():
        actual = merged.get(key)
        if actual is None:
            problems.append(f"artifact does not record {key}")
        elif actual != expected:
            problems.append(f"{key}: artifact={actual!r} recorded_run={expected!r}")
    if manifest.get("internal_test_accessed") or metrics.get("internal_test_accessed"):
        problems.append("artifact claims internal_test was accessed")
    for key in ("stage_id", "expert_id", "mode", "model_revision"):
        if key in resolved and manifest.get(key) != resolved.get(key):
            problems.append(
                f"{key} differs between training_manifest.json and resolved_config.json")
    return problems


def _label_space_inconsistencies(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    grid_stats: Mapping[str, Any],
) -> list[str]:
    """The label space is the contract the head, loss and decoder all share."""
    problems: list[str] = []
    vocab = W2NERLabelVocab()
    expected_types = list(vocab.type_order)
    for source, payload in (("training_manifest.json", manifest),
                            ("resolved_config.json", resolved)):
        recorded = list(payload.get("label_space", []))
        if recorded and recorded != expected_types:
            problems.append(
                f"{source} label_space {recorded} != repository type order {expected_types}")
    recorded_count = grid_stats.get("label_count")
    if recorded_count is not None and int(recorded_count) != len(vocab.labels):
        problems.append(
            f"grid_target_statistics.json label_count {recorded_count} != "
            f"repository relation-label count {len(vocab.labels)}")
    return problems


# ---------------------------------------------------------------------------
# B. Epoch history reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpochRow:
    """One reconstructed training epoch. Unread fields stay :data:`UNAVAILABLE`."""

    epoch: int
    mean_training_loss: float | str
    predicted_mentions: int | str
    gold_mentions: int | str
    true_positives: int | str
    false_positives: int | str
    false_negatives: int | str
    exact_precision: float | str
    exact_recall: float | str
    exact_f1: float | str
    is_new_best: bool | str
    checkpoint_hash: str
    optimizer_steps: int | str
    backward_passes: int | str

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "mean_training_loss": self.mean_training_loss,
            "predicted_mentions": self.predicted_mentions,
            "gold_mentions": self.gold_mentions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "exact_f1": self.exact_f1,
            "is_new_best": self.is_new_best,
            "checkpoint_hash": self.checkpoint_hash,
            "optimizer_steps": self.optimizer_steps,
            "backward_passes": self.backward_passes,
        }


@dataclass(frozen=True, slots=True)
class EpochHistoryReport:
    rows: tuple[EpochRow, ...]
    first_epoch_with_zero_predictions: int | str
    last_epoch_with_any_prediction: int | str
    peak_predicted_mentions: int | str
    peak_prediction_epoch: int | str
    final_predicted_mentions: int | str
    unavailable_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "first_epoch_with_zero_predictions": self.first_epoch_with_zero_predictions,
            "last_epoch_with_any_prediction": self.last_epoch_with_any_prediction,
            "peak_predicted_mentions": self.peak_predicted_mentions,
            "peak_prediction_epoch": self.peak_prediction_epoch,
            "final_predicted_mentions": self.final_predicted_mentions,
            "unavailable_fields": list(self.unavailable_fields),
            "internal_test_accessed": False,
        }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, dict):
                yield record


def _validation_totals_by_epoch(progress_path: Path) -> dict[int, dict[str, int]]:
    """Last validation heartbeat per epoch carries that pass's running totals.

    ``training_history.jsonl`` records precision/recall/F1 but not the raw
    predicted/gold counts, so the counts are recovered from the progress log.
    A validation pass that emitted no heartbeat is simply absent here, and the
    caller marks the field UNAVAILABLE rather than inventing a zero.
    """
    totals: dict[int, dict[str, int]] = {}
    for record in _iter_jsonl(progress_path):
        if record.get("stage") != "validation_progress":
            continue
        epoch = record.get("epoch")
        predicted = record.get("predicted_mentions_so_far")
        gold = record.get("gold_mentions_so_far")
        if epoch is None or predicted is None or gold is None:
            continue
        totals[int(epoch)] = {
            "predicted": int(predicted),
            "gold": int(gold),
            "sample": int(record.get("sample", 0)),
        }
    return totals


def _best_flags_by_epoch(progress_path: Path) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    for record in _iter_jsonl(progress_path):
        if record.get("stage") != "epoch_validation_complete":
            continue
        epoch = record.get("epoch")
        if epoch is None or "is_new_best" not in record:
            continue
        flags[int(epoch)] = bool(record["is_new_best"])
    return flags


def _checkpoint_hashes_by_epoch(progress_path: Path) -> dict[int, str]:
    hashes: dict[int, str] = {}
    for record in _iter_jsonl(progress_path):
        if record.get("stage") != "epoch_checkpoint_persisted":
            continue
        epoch = record.get("epoch")
        if epoch is None:
            continue
        digest = record.get("latest_checkpoint_sha256") or record.get("checkpoint_sha256")
        if digest:
            hashes[int(epoch)] = str(digest)
    return hashes


def reconstruct_epoch_history(artifact_dir: str | Path) -> EpochHistoryReport:
    """Join training history, progress heartbeats and final metrics per epoch."""
    root = Path(artifact_dir)
    history_path = root / "logs" / "training_history.jsonl"
    progress_path = root / "logs" / "training_progress.jsonl"
    if not history_path.is_file():
        raise E4DiagnosisError(f"missing epoch history: {history_path}")

    totals = _validation_totals_by_epoch(progress_path)
    best_flags = _best_flags_by_epoch(progress_path)
    checkpoint_hashes = _checkpoint_hashes_by_epoch(progress_path)

    rows: list[EpochRow] = []
    unavailable: set[str] = set()
    for record in _iter_jsonl(history_path):
        epoch = int(record["epoch"])
        counts = totals.get(epoch)
        precision = record.get("validation_exact_precision", UNAVAILABLE)
        recall = record.get("validation_exact_recall", UNAVAILABLE)
        if counts is None:
            unavailable.update({"predicted_mentions", "gold_mentions", "true_positives",
                                "false_positives", "false_negatives"})
            predicted: int | str = UNAVAILABLE
            gold: int | str = UNAVAILABLE
            true_positive: int | str = UNAVAILABLE
            false_positive: int | str = UNAVAILABLE
            false_negative: int | str = UNAVAILABLE
        else:
            predicted = counts["predicted"]
            gold = counts["gold"]
            # TP is recovered from precision x predicted, which is exact here
            # because both are integer-derived; it is rounded, never estimated.
            true_positive = (
                round(float(precision) * predicted)
                if isinstance(precision, (int, float)) else UNAVAILABLE)
            false_positive = (
                predicted - true_positive if isinstance(true_positive, int) else UNAVAILABLE)
            false_negative = (
                gold - true_positive if isinstance(true_positive, int) else UNAVAILABLE)
        if epoch not in best_flags:
            unavailable.add("is_new_best")
        if epoch not in checkpoint_hashes:
            unavailable.add("checkpoint_hash")
        rows.append(EpochRow(
            epoch=epoch,
            mean_training_loss=record.get("train_loss", UNAVAILABLE),
            predicted_mentions=predicted,
            gold_mentions=gold,
            true_positives=true_positive,
            false_positives=false_positive,
            false_negatives=false_negative,
            exact_precision=precision,
            exact_recall=recall,
            exact_f1=record.get("validation_exact_f1", UNAVAILABLE),
            is_new_best=best_flags.get(epoch, UNAVAILABLE),
            checkpoint_hash=checkpoint_hashes.get(epoch, UNAVAILABLE),
            optimizer_steps=record.get("optimizer_steps", UNAVAILABLE),
            backward_passes=record.get("backward_passes", UNAVAILABLE),
        ))

    rows.sort(key=lambda row: row.epoch)
    counted = [row for row in rows if isinstance(row.predicted_mentions, int)]
    if counted:
        with_predictions = [row for row in counted if row.predicted_mentions]
        peak = max(counted, key=lambda row: (int(row.predicted_mentions), -row.epoch))
        zero_after_peak = [
            row.epoch for row in counted
            if row.epoch > peak.epoch and row.predicted_mentions == 0]
        first_zero: int | str = zero_after_peak[0] if zero_after_peak else (
            next((row.epoch for row in counted if row.predicted_mentions == 0), UNAVAILABLE))
        last_any: int | str = (
            with_predictions[-1].epoch if with_predictions else UNAVAILABLE)
        report = EpochHistoryReport(
            rows=tuple(rows),
            first_epoch_with_zero_predictions=first_zero,
            last_epoch_with_any_prediction=last_any,
            peak_predicted_mentions=int(peak.predicted_mentions),
            peak_prediction_epoch=peak.epoch,
            final_predicted_mentions=int(counted[-1].predicted_mentions),
            unavailable_fields=tuple(sorted(unavailable)),
        )
    else:
        report = EpochHistoryReport(
            rows=tuple(rows),
            first_epoch_with_zero_predictions=UNAVAILABLE,
            last_epoch_with_any_prediction=UNAVAILABLE,
            peak_predicted_mentions=UNAVAILABLE,
            peak_prediction_epoch=UNAVAILABLE,
            final_predicted_mentions=UNAVAILABLE,
            unavailable_fields=tuple(sorted(unavailable | {"predicted_mentions"})),
        )
    return report


# ---------------------------------------------------------------------------
# Governed corpus access (shared by C and D)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GovernedExample:
    """One governed row reduced to exactly what the grid needs."""

    row_index: int
    document_id: str
    source_dataset: str
    text: str
    entities: tuple[EntitySpan, ...]


def load_governed_examples(
    path: str | Path, *, split: str, limit: int | None = None,
) -> Iterator[GovernedExample]:
    """Stream a governed split. One example is materialized at a time."""
    assert_split_allowed(split)
    resolved = Path(path)
    if not resolved.is_file():
        raise E4DiagnosisError(f"governed {split} split not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if limit is not None and row_index >= limit:
                return
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            yield GovernedExample(
                row_index=row_index,
                document_id=str(row.get("document_id", "")),
                source_dataset=str(row.get("source_dataset", "")),
                text=str(row["text"]),
                entities=tuple(
                    EntitySpan(
                        int(entity["start"]),
                        int(entity["end"]),
                        str(entity.get("target_type")
                            or entity.get("type")
                            or entity.get("label")),
                        str(entity["text"]),
                    )
                    for entity in row.get("entities", [])
                ),
            )


# ---------------------------------------------------------------------------
# C. Gold grid round-trip
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundTripFailure:
    """Metadata for one round-trip failure. Carries offsets, never text."""

    split: str
    row_index: int
    document_id: str
    entity_type: str
    start: int
    end: int
    direction: str  # "missing_from_reconstruction" | "spurious_reconstruction"

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "row_index": self.row_index,
            "document_id": self.document_id,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class RoundTripReport:
    """gold spans -> atomic words -> target grid -> decoder -> gold spans."""

    split: str
    examples_checked: int
    gold_mentions: int
    reconstructed_mentions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    exact_precision: float
    exact_recall: float
    exact_f1: float
    failures_by_entity_type: Mapping[str, int]
    representative_failures: tuple[RoundTripFailure, ...]

    @property
    def vacuous(self) -> bool:
        """No gold mention was in scope, so the round-trip proved nothing.

        A bounded scan can land entirely inside the corpus's zero-entity prefix.
        That is an absence of evidence, and must never be read as a failed
        round-trip — the two lead to opposite verdicts.
        """
        return self.gold_mentions == 0 and self.reconstructed_mentions == 0

    @property
    def passes(self) -> bool:
        return (self.exact_precision == 1.0
                and self.exact_recall == 1.0
                and self.exact_f1 == 1.0
                and self.gold_mentions > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "examples_checked": self.examples_checked,
            "gold_mentions": self.gold_mentions,
            "reconstructed_mentions": self.reconstructed_mentions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "exact_f1": self.exact_f1,
            "passes": self.passes,
            "vacuous": self.vacuous,
            "failures_by_entity_type": dict(self.failures_by_entity_type),
            "representative_failures": [f.as_dict() for f in self.representative_failures],
            "model_predictions_used": False,
            "internal_test_accessed": False,
        }


def gold_grid_round_trip(
    examples: Iterable[GovernedExample],
    *,
    split: str,
    max_representative_failures: int = 20,
    vocab: W2NERLabelVocab | None = None,
) -> RoundTripReport:
    """Prove the target builder and decoder are mutually inverse on gold data.

    No model is involved. If this does not reach exact P = R = F1 = 1.0 then the
    supervision signal itself is lossy and every downstream statement about model
    quality is meaningless — which is why :func:`resolve_verdict` treats a failure
    here as terminal.
    """
    assert_split_allowed(split)
    label_vocab = vocab or W2NERLabelVocab()
    examples_checked = 0
    gold_total = 0
    reconstructed_total = 0
    true_positives = 0
    failures_by_type: dict[str, int] = {}
    representative: list[RoundTripFailure] = []

    for example in examples:
        examples_checked += 1
        words = tokenize_atomic_words(example.text)
        grid = build_w2ner_grid(
            example.document_id, example.text, example.entities,
            words=words, vocab=label_vocab)
        gold = {(entity.start, entity.end, entity.entity_type)
                for entity in example.entities}
        reconstructed = {(span.start, span.end, span.entity_type)
                         for span in decode_w2ner_grid(grid)}
        gold_total += len(gold)
        reconstructed_total += len(reconstructed)
        true_positives += len(gold & reconstructed)
        for start, end, entity_type in sorted(gold - reconstructed):
            failures_by_type[entity_type] = failures_by_type.get(entity_type, 0) + 1
            if len(representative) < max_representative_failures:
                representative.append(RoundTripFailure(
                    split=split, row_index=example.row_index,
                    document_id=example.document_id, entity_type=entity_type,
                    start=start, end=end,
                    direction="missing_from_reconstruction"))
        for start, end, entity_type in sorted(reconstructed - gold):
            failures_by_type[entity_type] = failures_by_type.get(entity_type, 0) + 1
            if len(representative) < max_representative_failures:
                representative.append(RoundTripFailure(
                    split=split, row_index=example.row_index,
                    document_id=example.document_id, entity_type=entity_type,
                    start=start, end=end,
                    direction="spurious_reconstruction"))

    precision = true_positives / reconstructed_total if reconstructed_total else 0.0
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return RoundTripReport(
        split=split,
        examples_checked=examples_checked,
        gold_mentions=gold_total,
        reconstructed_mentions=reconstructed_total,
        true_positives=true_positives,
        false_positives=reconstructed_total - true_positives,
        false_negatives=gold_total - true_positives,
        exact_precision=precision,
        exact_recall=recall,
        exact_f1=f1,
        failures_by_entity_type=dict(sorted(failures_by_type.items())),
        representative_failures=tuple(representative),
    )


# ---------------------------------------------------------------------------
# D. Grid class distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassDistributionReport:
    """What the W2NER target actually asks the classifier to predict."""

    split: str
    examples: int
    max_words_observed: int
    mean_words: float
    valid_grid_cells: int
    ignored_or_padded_cells: int
    background_cells: int
    positive_cells: int
    cells_by_label: Mapping[str, int]
    positive_to_background_ratio: float
    examples_with_zero_positive_cells: int
    mentions_by_relation_pattern: Mapping[str, int]
    entities_represented: int
    labels_with_no_training_signal: tuple[str, ...]
    mean_per_example_positive_fraction: float
    constant_predictor_loss: float
    constant_predictor_distribution: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "examples": self.examples,
            "max_words_observed": self.max_words_observed,
            "mean_words": self.mean_words,
            "valid_grid_cells": self.valid_grid_cells,
            "ignored_or_padded_cells": self.ignored_or_padded_cells,
            "background_cells": self.background_cells,
            "positive_cells": self.positive_cells,
            "cells_by_label": dict(self.cells_by_label),
            "positive_to_background_ratio": self.positive_to_background_ratio,
            "examples_with_zero_positive_cells": self.examples_with_zero_positive_cells,
            "mentions_by_relation_pattern": dict(self.mentions_by_relation_pattern),
            "entities_represented": self.entities_represented,
            "labels_with_no_training_signal": list(self.labels_with_no_training_signal),
            "mean_per_example_positive_fraction": self.mean_per_example_positive_fraction,
            "constant_predictor_loss": self.constant_predictor_loss,
            "constant_predictor_distribution": dict(self.constant_predictor_distribution),
            "internal_test_accessed": False,
        }


def _relation_pattern(word_span: int) -> str:
    """How many grid cells one mention of ``word_span`` atomic words creates."""
    if word_span == 1:
        return "single_word_thw_only"
    if word_span == 2:
        return "two_word_nnw1_thw1"
    return f"multi_word_nnw{word_span - 1}_thw1"


def grid_class_distribution(
    examples: Iterable[GovernedExample],
    *,
    split: str,
    max_words: int | None = None,
    vocab: W2NERLabelVocab | None = None,
) -> ClassDistributionReport:
    """Count every target cell the training loss actually sees.

    ``max_words`` reproduces the padded ``max_words x max_words`` tensor so the
    padded region can be counted separately. The real run masked padding out of
    the head, and the loss consumed the **unpadded** ``n x n`` grid, so the padded
    count is reported for completeness and is not folded into the valid cells.
    """
    assert_split_allowed(split)
    label_vocab = vocab or W2NERLabelVocab()
    labels = label_vocab.labels

    examples_seen = 0
    word_total = 0
    max_observed = 0
    valid_cells = 0
    padded_cells = 0
    positive_cells = 0
    zero_positive_examples = 0
    entities_represented = 0
    cells_by_label = dict.fromkeys(labels, 0)
    patterns: dict[str, int] = {}
    positive_fractions: list[float] = []
    # Weighted marginal for the constant-predictor bound: the training loss is a
    # per-example mean, so every example contributes its own class fractions with
    # equal weight regardless of how many cells it has.
    weighted_marginal = [0.0] * len(labels)

    for example in examples:
        examples_seen += 1
        words = tokenize_atomic_words(example.text)
        word_count = len(words)
        word_total += word_count
        max_observed = max(max_observed, word_count)
        grid = build_w2ner_grid(
            example.document_id, example.text, example.entities,
            words=words, vocab=label_vocab)
        cells = word_count * word_count
        valid_cells += cells
        if max_words is not None:
            padded_cells += max_words * max_words - cells

        per_example = [0] * len(labels)
        for row in grid.labels:
            for label_id in row:
                per_example[label_id] += 1
        example_positive = cells - per_example[label_vocab.none_id]
        positive_cells += example_positive
        for index, count in enumerate(per_example):
            cells_by_label[labels[index]] += count
            weighted_marginal[index] += count / cells
        positive_fractions.append(example_positive / cells)
        if example_positive == 0:
            zero_positive_examples += 1

        for entity in example.entities:
            start_word = next(
                (word.index for word in words if word.start == entity.start), None)
            end_word = next(
                (word.index for word in words if word.end == entity.end), None)
            if start_word is None or end_word is None:
                continue
            entities_represented += 1
            pattern = _relation_pattern(end_word - start_word + 1)
            patterns[pattern] = patterns.get(pattern, 0) + 1

    if examples_seen == 0:
        raise E4DiagnosisError(f"{split}: no governed examples were read")

    background = cells_by_label[W2NER_NONE]
    marginal = [value / examples_seen for value in weighted_marginal]
    constant_loss = -sum(value * math.log(value) for value in marginal if value > 0.0)
    no_signal = tuple(
        label for label in labels
        if label != W2NER_NONE and cells_by_label[label] == 0)

    return ClassDistributionReport(
        split=split,
        examples=examples_seen,
        max_words_observed=max_observed,
        mean_words=word_total / examples_seen,
        valid_grid_cells=valid_cells,
        ignored_or_padded_cells=padded_cells,
        background_cells=background,
        positive_cells=positive_cells,
        cells_by_label=dict(cells_by_label),
        positive_to_background_ratio=(positive_cells / background) if background else 0.0,
        examples_with_zero_positive_cells=zero_positive_examples,
        mentions_by_relation_pattern=dict(sorted(patterns.items())),
        entities_represented=entities_represented,
        labels_with_no_training_signal=no_signal,
        mean_per_example_positive_fraction=sum(positive_fractions) / examples_seen,
        constant_predictor_loss=constant_loss,
        constant_predictor_distribution={
            label: marginal[index] for index, label in enumerate(labels)},
    )


def constant_predictor_gap(
    observed_mean_loss: float, distribution: ClassDistributionReport,
) -> dict[str, Any]:
    """How much better than "ignore the input entirely" the run actually got.

    The best achievable **input-independent** predictor emits the weighted class
    marginal at every cell; its loss is that marginal's entropy. Comparing the
    observed converged loss against it turns "the loss looks small" into a
    quantity: a ratio near 1.0 means the network learned essentially nothing
    beyond the class prior, however small the absolute loss looks.
    """
    baseline = distribution.constant_predictor_loss
    if baseline <= 0.0:
        raise E4DiagnosisError("constant-predictor baseline is not positive")
    ratio = observed_mean_loss / baseline
    return {
        "observed_mean_training_loss": observed_mean_loss,
        "constant_predictor_loss": baseline,
        "observed_over_constant_ratio": ratio,
        "improvement_over_constant_predictor": 1.0 - ratio,
        "mean_per_example_positive_fraction": (
            distribution.mean_per_example_positive_fraction),
        "implied_mean_cross_entropy_per_positive_cell": (
            observed_mean_loss / distribution.mean_per_example_positive_fraction
            if distribution.mean_per_example_positive_fraction else float("nan")),
    }


# ---------------------------------------------------------------------------
# E. Checkpoint probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointProbeReport:
    """Read-only probe of one trained checkpoint on governed validation."""

    role: str
    checkpoint_sha256: str
    epoch: int
    predicted_mention_total: int
    gold_mention_total: int
    true_positives: int
    exact_precision: float
    exact_recall: float
    exact_f1: float
    predictions_by_entity_type: Mapping[str, int]
    predicted_labels_by_class: Mapping[str, int]
    background_label_count: int
    non_background_label_count: int
    background_logit_quantiles: Mapping[str, float]
    strongest_non_background_logit_quantiles: Mapping[str, float]
    gold_positive_cell_predicted_labels: Mapping[str, int]
    gold_positive_cell_background_rate: float
    decoder_input_positive_relations: int
    decoder_output_mention_count: int
    w2ner_head_keys_restored: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "checkpoint_sha256": self.checkpoint_sha256,
            "epoch": self.epoch,
            "predicted_mention_total": self.predicted_mention_total,
            "gold_mention_total": self.gold_mention_total,
            "true_positives": self.true_positives,
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "exact_f1": self.exact_f1,
            "predictions_by_entity_type": dict(self.predictions_by_entity_type),
            "predicted_labels_by_class": dict(self.predicted_labels_by_class),
            "background_label_count": self.background_label_count,
            "non_background_label_count": self.non_background_label_count,
            "background_logit_quantiles": dict(self.background_logit_quantiles),
            "strongest_non_background_logit_quantiles": dict(
                self.strongest_non_background_logit_quantiles),
            "gold_positive_cell_predicted_labels": dict(
                self.gold_positive_cell_predicted_labels),
            "gold_positive_cell_background_rate": self.gold_positive_cell_background_rate,
            "decoder_input_positive_relations": self.decoder_input_positive_relations,
            "decoder_output_mention_count": self.decoder_output_mention_count,
            "w2ner_head_keys_restored": list(self.w2ner_head_keys_restored),
            "internal_test_accessed": False,
        }


def inspect_checkpoint_payload(payload: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    """Schema-check a loaded checkpoint without touching its tensors.

    Separated from the inference probe on purpose: restoration can be verified
    from the payload alone, so "the head was never saved" is distinguishable from
    "the head was saved and predicts background".
    """
    missing = tuple(key for key in CHECKPOINT_REQUIRED_KEYS if key not in payload)
    model_state = payload.get("model_state")
    state_missing: tuple[str, ...] = ()
    head_keys: tuple[str, ...] = ()
    if isinstance(model_state, Mapping):
        state_missing = tuple(
            key for key in CHECKPOINT_MODEL_STATE_KEYS if key not in model_state)
        head = model_state.get("w2ner_head")
        if isinstance(head, Mapping):
            head_keys = tuple(sorted(str(key) for key in head))
    else:
        state_missing = CHECKPOINT_MODEL_STATE_KEYS
    return {
        "role": role,
        "missing_keys": list(missing),
        "model_state_missing_keys": list(state_missing),
        "w2ner_head_keys": list(head_keys),
        "w2ner_head_restored": bool(head_keys) and not state_missing,
        "epoch": payload.get("epoch", UNAVAILABLE),
        "mode": payload.get("mode", UNAVAILABLE),
        "best_metric": payload.get("best_metric", UNAVAILABLE),
        "e4_input_contract_version": payload.get(
            "e4_input_contract_version", UNAVAILABLE),
        "schema_ok": not missing and not state_missing,
    }


def require_checkpoint(artifact_dir: str | Path, role: str) -> Path:
    """Return the checkpoint path, or fail closed with what is needed to get it."""
    if role not in ("best", "latest"):
        raise E4DiagnosisError(f"unknown checkpoint role {role!r}")
    path = Path(artifact_dir) / "checkpoints" / f"{role}.pt"
    if not path.is_file():
        expected = (E4_FULL_BEST_CHECKPOINT_SHA256 if role == "best"
                    else E4_FULL_LATEST_CHECKPOINT_SHA256)
        raise CheckpointEvidenceUnavailable(
            f"{role}.pt is not present at {path}. The grid-logit and "
            f"gold-positive-cell evidence required to prove or refute "
            f"{ALL_BACKGROUND_LOSS_COLLAPSE} cannot be produced without it. "
            f"Download the completed Colab checkpoint (expected SHA-256 "
            f"{expected}) into the artifact directory and re-run the probe.")
    return path


# ---------------------------------------------------------------------------
# F. Label and loss contract audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelContractTrace:
    """One target cell traced from builder to decoder, symbol for symbol."""

    label_order: tuple[str, ...]
    background_label: str
    background_label_id: int
    nnw_label_id: int
    thw_label_ids: Mapping[str, int]
    classifier_output_size: int
    decoder_thw_interpretation: Mapping[int, str]
    traced_cells: tuple[Mapping[str, Any], ...]
    consistent: bool
    differences: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label_order": list(self.label_order),
            "background_label": self.background_label,
            "background_label_id": self.background_label_id,
            "nnw_label_id": self.nnw_label_id,
            "thw_label_ids": dict(self.thw_label_ids),
            "classifier_output_size": self.classifier_output_size,
            "decoder_thw_interpretation": {
                str(key): value for key, value in self.decoder_thw_interpretation.items()},
            "traced_cells": [dict(cell) for cell in self.traced_cells],
            "consistent": self.consistent,
            "differences": list(self.differences),
        }


def trace_label_contract(vocab: W2NERLabelVocab | None = None) -> LabelContractTrace:
    """Verify one label ordering is shared by builder, head, loss and decoder.

    The head is built with ``relation_count = grid_target_statistics.label_count``
    and the loss is ``cross_entropy(logits.reshape(-1, item.label_count), ...)``,
    where ``label_count = len(grid.vocab.labels)``. Both therefore index the same
    tuple the builder wrote and the decoder reads, so consistency is checked by
    round-tripping every id rather than by comparing two hand-written lists.
    """
    label_vocab = vocab or W2NERLabelVocab()
    labels = label_vocab.labels
    differences: list[str] = []

    if labels[label_vocab.none_id] != W2NER_NONE:
        differences.append(
            f"background id {label_vocab.none_id} is {labels[label_vocab.none_id]!r}")
    if labels[label_vocab.nnw_id] != W2NER_NEXT_NEIGHBORING_WORD:
        differences.append(
            f"NNW id {label_vocab.nnw_id} is {labels[label_vocab.nnw_id]!r}")

    thw_ids: dict[str, int] = {}
    decoder_interpretation: dict[int, str] = {}
    for entity_type in label_vocab.type_order:
        label_id = label_vocab.thw_id(entity_type)
        thw_ids[entity_type] = label_id
        if labels[label_id] != f"{W2NER_TAIL_HEAD_WORD_PREFIX}{entity_type}":
            differences.append(f"THW id {label_id} is {labels[label_id]!r}")
        decoded_type = label_vocab.thw_type(label_id)
        decoder_interpretation[label_id] = decoded_type or ""
        if decoded_type != entity_type:
            differences.append(
                f"decoder reads id {label_id} as {decoded_type!r}, builder wrote "
                f"{entity_type!r}")
    for label_id in (label_vocab.none_id, label_vocab.nnw_id):
        if label_vocab.thw_type(label_id) is not None:
            differences.append(f"decoder treats id {label_id} as a THW relation")

    # Trace real cells through a two-word entity: one NNW cell and one THW cell.
    text = "sốt cao"
    entity = EntitySpan(0, len(text), "SYMPTOM", text)
    grid = build_w2ner_grid("trace", text, (entity,), vocab=label_vocab)
    traced: list[dict[str, Any]] = []
    for row_index, row in enumerate(grid.labels):
        for column_index, label_id in enumerate(row):
            if label_id == label_vocab.none_id:
                continue
            traced.append({
                "grid_row": row_index,
                "grid_column": column_index,
                "target_label_id": label_id,
                "target_label": labels[label_id],
                "classifier_output_index": label_id,
                "cross_entropy_target_index": label_id,
                "argmax_recovers_label": labels[label_id],
                "decoder_relation": label_vocab.thw_type(label_id) or "NNW_path_edge",
            })
    decoded = {(span.start, span.end, span.entity_type)
               for span in decode_w2ner_grid(grid)}
    if (entity.start, entity.end, entity.entity_type) not in decoded:
        differences.append("traced two-word entity did not survive the decoder")

    return LabelContractTrace(
        label_order=labels,
        background_label=W2NER_NONE,
        background_label_id=label_vocab.none_id,
        nnw_label_id=label_vocab.nnw_id,
        thw_label_ids=thw_ids,
        classifier_output_size=len(labels),
        decoder_thw_interpretation=decoder_interpretation,
        traced_cells=tuple(traced),
        consistent=not differences,
        differences=tuple(differences),
    )


@dataclass(frozen=True, slots=True)
class LossContractReport:
    """Recorded facts about the loss the completed run actually optimized.

    Every field is a statement about code that exists at the audited commit, not
    a proposal. This milestone diagnoses; it does not change the loss.
    """

    loss_function: str
    ignore_index_used: bool
    class_weights_used: bool
    padded_cells_in_loss: bool
    triangular_masking_applied: bool
    normalization: str
    background_class_id: int
    positive_relation_ids: tuple[int, ...]
    all_background_solution_is_reachable: bool
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss_function": self.loss_function,
            "ignore_index_used": self.ignore_index_used,
            "class_weights_used": self.class_weights_used,
            "padded_cells_in_loss": self.padded_cells_in_loss,
            "triangular_masking_applied": self.triangular_masking_applied,
            "normalization": self.normalization,
            "background_class_id": self.background_class_id,
            "positive_relation_ids": list(self.positive_relation_ids),
            "all_background_solution_is_reachable": (
                self.all_background_solution_is_reachable),
            "notes": list(self.notes),
        }


def audit_loss_contract(vocab: W2NERLabelVocab | None = None) -> LossContractReport:
    """Record how the executed E4 loss treats background, padding and masking."""
    label_vocab = vocab or W2NERLabelVocab()
    positive_ids = tuple(
        index for index in range(len(label_vocab.labels))
        if index != label_vocab.none_id)
    return LossContractReport(
        loss_function=(
            "torch.nn.functional.cross_entropy(logits.reshape(-1, label_count), "
            "labels.reshape(-1))"),
        ignore_index_used=False,
        class_weights_used=False,
        # The executed loop passes ``item.grid.labels`` and ``item.grid.pair_mask``
        # — the UNPADDED n x n grid — so the max_words padding never reaches the
        # loss. ``item.padded_labels`` exists but is not what was optimized.
        padded_cells_in_loss=False,
        # ``build_w2ner_grid`` sets pair_mask entirely True over the n x n square,
        # so both triangles are scored: NNW lives at [i][i+1] (upper) and THW at
        # [tail][head] (lower/diagonal). Nothing is masked out.
        triangular_masking_applied=False,
        normalization="mean over all n*n valid cells, then mean over examples",
        background_class_id=label_vocab.none_id,
        positive_relation_ids=positive_ids,
        all_background_solution_is_reachable=True,
        notes=(
            "Every one of the n*n cells contributes to the mean, and no class "
            "weighting offsets the background majority.",
            "Per-example normalization gives a 5-word example the same weight as "
            "a 162-word one, independent of how many cells each contains.",
            "An input-independent predictor that emits the class marginal is a "
            "valid stationary point; its loss is the marginal's entropy and is "
            "computed by grid_class_distribution().",
            "No learning-rate warmup or schedule was configured: the recorded run "
            "held learning_rate 2e-05 constant for all 50,748 optimizer steps.",
            "The decoder applies no score threshold: decode_argmax_relation_grid "
            "takes a plain argmax and decode_w2ner_grid is called with scores=None "
            "and threshold=0.0, so score 1.0 always passes.",
        ),
    )


# ---------------------------------------------------------------------------
# H. Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    verdict: str
    ruled_out: tuple[str, ...]
    supported_hypothesis: str
    supporting_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ruled_out": list(self.ruled_out),
            "supported_hypothesis": self.supported_hypothesis,
            "supporting_evidence": list(self.supporting_evidence),
            "missing_evidence": list(self.missing_evidence),
        }


def resolve_verdict(
    *,
    round_trips: Sequence[RoundTripReport],
    label_trace: LabelContractTrace,
    distributions: Sequence[ClassDistributionReport],
    history: EpochHistoryReport,
    loss_contract: LossContractReport,
    probes: Sequence[CheckpointProbeReport] = (),
    checkpoint_inspections: Sequence[Mapping[str, Any]] = (),
) -> Verdict:
    """Choose a verdict from evidence only.

    ``ALL_BACKGROUND_LOSS_COLLAPSE`` is gated behind three independent
    requirements — a passing gold-grid round-trip, grid logits, and
    gold-positive-cell predictions — precisely so that a zero prediction count
    cannot on its own be promoted into a root cause.
    """
    confirmed: list[str] = []
    ruled_out: list[str] = []
    supporting: list[str] = []
    missing: list[str] = []

    # A round-trip over a subset containing no gold mention proves nothing in
    # either direction; counting it as a failure would invert the verdict.
    conclusive = [report for report in round_trips if not report.vacuous]
    vacuous = [report for report in round_trips if report.vacuous]
    for report in vacuous:
        missing.append(
            f"the {report.split} round-trip covered no gold mention and is "
            "inconclusive")
    round_trip_passes = bool(conclusive) and all(report.passes for report in conclusive)
    if round_trip_passes:
        ruled_out.append(TARGET_DECODER_MISMATCH)
        supporting.append(
            "gold-grid round-trip reached exact P = R = F1 = 1.0 on every split "
            "checked, with no model in the loop")
    elif conclusive:
        confirmed.append(TARGET_DECODER_MISMATCH)
        supporting.append(
            "gold-grid round-trip did not reproduce the gold mentions; the "
            "supervision signal itself is lossy")
    elif not round_trips:
        missing.append("no gold-grid round-trip was run")

    if label_trace.consistent and round_trip_passes:
        ruled_out.append(LABEL_MAPPING_MISMATCH)
        supporting.append(
            "one label ordering is shared by target builder, classifier output "
            "index, cross-entropy target and decoder; every id round-trips")
    elif not label_trace.consistent:
        confirmed.append(LABEL_MAPPING_MISMATCH)
        supporting.extend(label_trace.differences)

    # The decoder applies no threshold at all, so no threshold policy can be
    # suppressing predictions. That is a static fact, provable without weights.
    threshold_note = next(
        (note for note in loss_contract.notes if "threshold" in note), "")
    if threshold_note:
        ruled_out.append(DECODER_THRESHOLD_FAILURE)
        supporting.append(threshold_note)

    restored = [dict(item) for item in checkpoint_inspections]
    if restored:
        broken = [item for item in restored if not item.get("w2ner_head_restored")]
        if broken:
            confirmed.append(CHECKPOINT_RESTORE_FAILURE)
            supporting.append(
                "at least one checkpoint does not restore a w2ner_head state dict")
        else:
            ruled_out.append(CHECKPOINT_RESTORE_FAILURE)
            supporting.append("every inspected checkpoint restores its w2ner_head")
    else:
        missing.append(
            "no checkpoint was inspected, so CHECKPOINT_RESTORE_FAILURE is "
            "neither confirmed nor ruled out")

    if distributions:
        for distribution in distributions:
            supporting.append(
                f"{distribution.split}: {distribution.positive_cells} positive "
                f"cells in {distribution.valid_grid_cells} valid cells "
                f"({distribution.positive_cells / distribution.valid_grid_cells:.6%}); "
                f"{distribution.examples_with_zero_positive_cells} of "
                f"{distribution.examples} examples carry no positive cell at all")
        if not loss_contract.class_weights_used and not loss_contract.ignore_index_used:
            supporting.append(
                "the executed loss applies neither class weighting nor an ignore "
                "index, so background dominates the mean directly")

    if isinstance(history.final_predicted_mentions, int):
        supporting.append(
            f"the final validation pass predicted "
            f"{history.final_predicted_mentions} mentions; the peak across all "
            f"recorded epochs was {history.peak_predicted_mentions} at epoch "
            f"{history.peak_prediction_epoch}")

    # The three requirements for the collapse verdict, checked explicitly.
    has_logits = any(probe.background_label_count or probe.non_background_label_count
                     for probe in probes)
    has_gold_cells = any(probe.gold_positive_cell_predicted_labels for probe in probes)
    if not probes:
        missing.append(
            "grid logit distributions from a checkpoint probe")
        missing.append(
            "predicted labels at gold-positive cells from a checkpoint probe")
    else:
        if not has_logits:
            missing.append("grid logit distributions from a checkpoint probe")
        if not has_gold_cells:
            missing.append("predicted labels at gold-positive cells")

    collapse_provable = round_trip_passes and has_logits and has_gold_cells
    if collapse_provable:
        background_dominant = all(
            probe.gold_positive_cell_background_rate >= 0.99 for probe in probes)
        if background_dominant:
            confirmed.append(ALL_BACKGROUND_LOSS_COLLAPSE)
            supporting.append(
                "the probed classifier emits background at essentially every "
                "gold-positive cell")

    if len(confirmed) > 1:
        return Verdict(
            verdict=MULTIPLE_CONFIRMED_FAILURES,
            ruled_out=tuple(ruled_out),
            supported_hypothesis=", ".join(sorted(confirmed)),
            supporting_evidence=tuple(supporting),
            missing_evidence=tuple(missing),
        )
    if len(confirmed) == 1:
        return Verdict(
            verdict=confirmed[0],
            ruled_out=tuple(ruled_out),
            supported_hypothesis=confirmed[0],
            supporting_evidence=tuple(supporting),
            missing_evidence=tuple(missing),
        )
    return Verdict(
        verdict=ROOT_CAUSE_NOT_YET_PROVEN,
        ruled_out=tuple(ruled_out),
        supported_hypothesis=ALL_BACKGROUND_LOSS_COLLAPSE if round_trip_passes else "",
        supporting_evidence=tuple(supporting),
        missing_evidence=tuple(missing),
    )


# ---------------------------------------------------------------------------
# Corpus composition (evidence that belongs with the distribution, not the model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusCompositionReport:
    """Per-source composition and file ordering of a governed split.

    The E4 loop streams ``iter_contracts()`` in deterministic file order and does
    not shuffle, so how the corpus is ordered on disk is part of what the
    optimizer actually experienced and is measured here rather than assumed.
    """

    split: str
    examples: int
    entities: int
    examples_by_source: Mapping[str, int]
    entities_by_source: Mapping[str, int]
    entity_types_by_source: Mapping[str, Mapping[str, int]]
    first_row_index_by_source: Mapping[str, int]
    longest_zero_entity_run: int
    longest_zero_entity_run_start: int
    sources_in_file_order: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "examples": self.examples,
            "entities": self.entities,
            "examples_by_source": dict(self.examples_by_source),
            "entities_by_source": dict(self.entities_by_source),
            "entity_types_by_source": {
                source: dict(counts)
                for source, counts in self.entity_types_by_source.items()},
            "first_row_index_by_source": dict(self.first_row_index_by_source),
            "longest_zero_entity_run": self.longest_zero_entity_run,
            "longest_zero_entity_run_start": self.longest_zero_entity_run_start,
            "sources_in_file_order": list(self.sources_in_file_order),
            "internal_test_accessed": False,
        }


def measure_corpus_composition(
    examples: Iterable[GovernedExample], *, split: str,
) -> CorpusCompositionReport:
    assert_split_allowed(split)
    examples_by_source: dict[str, int] = {}
    entities_by_source: dict[str, int] = {}
    types_by_source: dict[str, dict[str, int]] = {}
    first_row: dict[str, int] = {}
    order: list[str] = []
    total_examples = 0
    total_entities = 0
    run = 0
    run_start = -1
    longest_run = 0
    longest_start = -1

    for example in examples:
        total_examples += 1
        source = example.source_dataset or "unknown"
        examples_by_source[source] = examples_by_source.get(source, 0) + 1
        entities_by_source[source] = entities_by_source.get(source, 0) + len(example.entities)
        counts = types_by_source.setdefault(source, {})
        for entity in example.entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        total_entities += len(example.entities)
        if source not in first_row:
            first_row[source] = example.row_index
            order.append(source)
        if example.entities:
            run = 0
            run_start = -1
        else:
            if run == 0:
                run_start = example.row_index
            run += 1
            if run > longest_run:
                longest_run = run
                longest_start = run_start

    if total_examples == 0:
        raise E4DiagnosisError(f"{split}: no governed examples were read")
    return CorpusCompositionReport(
        split=split,
        examples=total_examples,
        entities=total_entities,
        examples_by_source=dict(sorted(examples_by_source.items())),
        entities_by_source=dict(sorted(entities_by_source.items())),
        entity_types_by_source={
            source: dict(sorted(counts.items()))
            for source, counts in sorted(types_by_source.items())},
        first_row_index_by_source=dict(sorted(first_row.items())),
        longest_zero_entity_run=longest_run,
        longest_zero_entity_run_start=longest_start,
        sources_in_file_order=tuple(order),
    )


# ---------------------------------------------------------------------------
# Consolidated diagnosis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CollapseDiagnosis:
    integrity: ArtifactIntegrityReport
    history: EpochHistoryReport
    round_trips: tuple[RoundTripReport, ...]
    distributions: tuple[ClassDistributionReport, ...]
    compositions: tuple[CorpusCompositionReport, ...]
    label_trace: LabelContractTrace
    loss_contract: LossContractReport
    constant_predictor_comparison: Mapping[str, Any]
    probes: tuple[CheckpointProbeReport, ...]
    checkpoint_inspections: tuple[Mapping[str, Any], ...]
    probe_blocked_reason: str
    verdict: Verdict
    probe_blocked_detail: Mapping[str, Any] = field(default_factory=dict)
    diagnosis_version: str = DIAGNOSIS_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnosis_version": self.diagnosis_version,
            "artifact_integrity": self.integrity.as_dict(),
            "epoch_history": self.history.as_dict(),
            "gold_grid_round_trip": [report.as_dict() for report in self.round_trips],
            "grid_class_distribution": [
                report.as_dict() for report in self.distributions],
            "corpus_composition": [report.as_dict() for report in self.compositions],
            "label_contract": self.label_trace.as_dict(),
            "loss_contract": self.loss_contract.as_dict(),
            "constant_predictor_comparison": dict(self.constant_predictor_comparison),
            "checkpoint_probes": [probe.as_dict() for probe in self.probes],
            "checkpoint_inspections": [dict(item) for item in self.checkpoint_inspections],
            "probe_blocked_reason": self.probe_blocked_reason,
            "probe_blocked_detail": dict(self.probe_blocked_detail),
            "verdict": self.verdict.as_dict(),
            "local_training_performed": False,
            "internal_test_accessed": False,
            "organizer_inference_performed": False,
            "output_zip_created": False,
        }


ProbeRunner = Callable[
    [Any, Mapping[str, Any]],
    tuple[tuple[CheckpointProbeReport, ...], tuple[Mapping[str, Any], ...]],
]


def _default_probe_runner(
    artifact_dir: Any, split_paths: Mapping[str, Any],
) -> tuple[tuple[CheckpointProbeReport, ...], tuple[Mapping[str, Any], ...]]:
    """Load the real probe lazily so this module imports without torch."""
    from .e4_checkpoint_probe import run_default_checkpoint_probe

    return run_default_checkpoint_probe(artifact_dir, split_paths)


def run_collapse_diagnosis(
    *,
    artifact_dir: str | Path,
    split_paths: Mapping[str, str | Path],
    max_words: int | None = None,
    limit: int | None = None,
    probe_runner: ProbeRunner | None = None,
) -> CollapseDiagnosis:
    """Run every diagnostic the available evidence supports, in order.

    When the checkpoints are present the probe **runs**. When it cannot run, the
    reason is recorded with its exception type and remedy. The verdict then
    reflects exactly the evidence obtained.
    """
    integrity = verify_artifact_integrity(artifact_dir)
    history = reconstruct_epoch_history(artifact_dir)

    round_trips: list[RoundTripReport] = []
    distributions: list[ClassDistributionReport] = []
    compositions: list[CorpusCompositionReport] = []
    for split in DIAGNOSABLE_SPLITS:
        path = split_paths.get(split)
        if path is None:
            continue
        round_trips.append(gold_grid_round_trip(
            load_governed_examples(path, split=split, limit=limit), split=split))
        distributions.append(grid_class_distribution(
            load_governed_examples(path, split=split, limit=limit),
            split=split, max_words=max_words))
        compositions.append(measure_corpus_composition(
            load_governed_examples(path, split=split, limit=limit), split=split))

    label_trace = trace_label_contract()
    loss_contract = audit_loss_contract()

    comparison: dict[str, Any] = {}
    train_distribution = next(
        (report for report in distributions if report.split == "train"), None)
    final_loss = next(
        (row.mean_training_loss for row in reversed(history.rows)
         if isinstance(row.mean_training_loss, (int, float))), None)
    if train_distribution is not None and final_loss is not None:
        if train_distribution.constant_predictor_loss > 0.0:
            comparison = constant_predictor_gap(float(final_loss), train_distribution)
        else:
            # A bounded scan can land entirely inside the corpus's zero-entity
            # prefix, where the class marginal is degenerate and its entropy is 0.
            # Say so; do not report a ratio against a baseline that does not exist.
            comparison = {
                "unavailable_reason": (
                    "the scanned subset contains no positive target cells, so an "
                    "input-independent baseline cannot be formed"),
                "examples_scanned": train_distribution.examples,
                "positive_cells": train_distribution.positive_cells,
            }

    # The checkpoint probe. Audit 0043 shipped this block with `probes` bound to
    # an empty tuple and `require_checkpoint` called only for its raise-on-absence
    # side effect, so arriving weights never unblocked anything and the reason
    # string stayed empty. It now actually runs the probe, and every way it can
    # fail produces a *named* reason rather than a bare "BLOCKED" (Audit 0044).
    probes: tuple[CheckpointProbeReport, ...] = ()
    inspections: tuple[Mapping[str, Any], ...] = ()
    blocked_reason = ""
    blocked_detail: dict[str, Any] = {}
    if probe_runner is None:
        probe_runner = _default_probe_runner
    try:
        require_checkpoint(artifact_dir, "best")
        require_checkpoint(artifact_dir, "latest")
    except CheckpointEvidenceUnavailable as error:
        blocked_reason = str(error)
        blocked_detail = {
            "exception_type": type(error).__name__,
            "next_action": (
                "download the completed Colab checkpoints into "
                f"{Path(artifact_dir) / 'checkpoints'}"),
        }
    else:
        try:
            probes, inspections = probe_runner(artifact_dir, split_paths)
        except E4DiagnosisError as error:
            # A dependency or restoration failure is real evidence about why the
            # probe cannot run. It is surfaced with its type and remedy, never
            # flattened into "BLOCKED".
            blocked_reason = str(error)
            detail = getattr(error, "as_dict", None)
            blocked_detail = (
                dict(detail()) if callable(detail)
                else {"exception_type": type(error).__name__})

    verdict = resolve_verdict(
        round_trips=round_trips,
        label_trace=label_trace,
        distributions=distributions,
        history=history,
        loss_contract=loss_contract,
        probes=probes,
        checkpoint_inspections=inspections,
    )
    return CollapseDiagnosis(
        integrity=integrity,
        history=history,
        round_trips=tuple(round_trips),
        distributions=tuple(distributions),
        compositions=tuple(compositions),
        label_trace=label_trace,
        loss_contract=loss_contract,
        constant_predictor_comparison=comparison,
        probes=probes,
        checkpoint_inspections=inspections,
        probe_blocked_reason=blocked_reason,
        probe_blocked_detail=blocked_detail,
        verdict=verdict,
    )


__all__ = [
    "ALL_BACKGROUND_LOSS_COLLAPSE",
    "CHECKPOINT_RESTORE_FAILURE",
    "DECODER_THRESHOLD_FAILURE",
    "DIAGNOSABLE_SPLITS",
    "DIAGNOSIS_VERSION",
    "E4_FULL_ARTIFACT_NAME",
    "E4_FULL_BEST_CHECKPOINT_SHA256",
    "E4_FULL_EXPECTED",
    "E4_FULL_LATEST_CHECKPOINT_SHA256",
    "FORBIDDEN_SPLITS",
    "LABEL_MAPPING_MISMATCH",
    "MULTIPLE_CONFIRMED_FAILURES",
    "ROOT_CAUSE_NOT_YET_PROVEN",
    "TARGET_DECODER_MISMATCH",
    "UNAVAILABLE",
    "VERDICTS",
    "ArtifactIntegrityReport",
    "ProbeRunner",
    "CheckpointEvidenceUnavailable",
    "CheckpointProbeReport",
    "ClassDistributionReport",
    "CollapseDiagnosis",
    "CorpusCompositionReport",
    "E4DiagnosisError",
    "EpochHistoryReport",
    "EpochRow",
    "GovernedExample",
    "LabelContractTrace",
    "LossContractReport",
    "RoundTripFailure",
    "RoundTripReport",
    "Verdict",
    "assert_no_clinical_text",
    "assert_split_allowed",
    "audit_loss_contract",
    "constant_predictor_gap",
    "gold_grid_round_trip",
    "grid_class_distribution",
    "inspect_checkpoint_payload",
    "load_governed_examples",
    "measure_corpus_composition",
    "reconstruct_epoch_history",
    "require_checkpoint",
    "resolve_verdict",
    "run_collapse_diagnosis",
    "trace_label_contract",
    "verify_artifact_integrity",
]
