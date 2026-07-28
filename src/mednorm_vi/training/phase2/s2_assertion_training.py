"""S2 assertion training contracts (Audit 0042).

Spec §8 defines the Assertion Hydra and §1 scores assertions by Jaccard at 30%
weight. Spec §8.1 is explicit that "an entity may carry multiple assertions;
decoding is multi-label, not exclusive softmax".

**The governed corpus currently contains no assertion supervision at all.** Every
source in ``manifests/annotation_coverage.json`` declares ``assertions: false``,
and no governed entity carries an assertion field. This module is therefore built
so that the *absence* of supervision is a first-class, loudly reported state:

* labels are **tri-state** — ``SUPERVISED_TRUE``, ``SUPERVISED_FALSE`` and
  ``UNKNOWN``. Missing supervision is never silently coerced to false, which
  would manufacture thousands of confident negatives out of nothing;
* only entity types the competition scores for assertions are eligible
  (MEDICATION, DIAGNOSIS, SYMPTOM);
* trusted and weak/synthetic supervision live in **separate provenance channels**
  and are counted, split and evaluated separately. Spec §14.1 permits weak
  labeling functions, but §6 of that section requires them to be controlled and
  filtered — so weak labels are opt-in, never trusted by default, and never
  reported as gold;
* document-level grouping prevents leakage between train and validation
  (spec §15.2 "group-split by ... document").

Nothing here trains, downloads a model, or reads internal_test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...schemas.constants import ASSERTION_LABELS
from .common import canonical_json_sha256

S2_STAGE_ID = "phase2-s2-assertion-v1"
S2_EXPERT_ID = "S2_assertion"
S2_MODEL_ID = "demdecuong/vihealthbert-base-word"
S2_INPUT_CONTRACT_VERSION = "s2-assertion-example-v1"
S2_CHECKPOINT_SCHEMA_VERSION = "phase2-s2-checkpoint-v1"
S2_ARTIFACT_SCHEMA_VERSION = "phase2-s2-artifact-v1"
S2_FULL_AUTHORIZATION = "I_AUTHORIZE_S2_FULL_TRAINING"
S2_DEFAULT_HIDDEN_SIZE = 768

S2_REQUIRED_ARTIFACT_FILES: tuple[str, ...] = (
    "checkpoints/best.pt",
    "checkpoints/latest.pt",
    "logs/training_history.jsonl",
    "resolved_config.json",
    "validation_metrics.json",
    "training_manifest.json",
)

S2_REQUIRED_CHECKPOINT_KEYS: tuple[str, ...] = (
    "checkpoint_schema_version",
    "stage_id",
    "expert_id",
    "mode",
    "config_sha256",
    "model_revision",
    "tokenizer_revision",
    "label_order",
    "parameter_count",
    "internal_test_accessed",
    "model_state",
)

# Deterministic label order. Multi-label, never a softmax over these.
ASSERTION_LABEL_ORDER: tuple[str, ...] = ("isNegated", "isHistorical", "isFamily")

# Spec §1 / organizer field policy: assertions apply to these three types only.
# TEST_NAME and TEST_RESULT emit no assertions, so they are excluded by a
# recorded policy rather than silently dropped.
ASSERTION_ELIGIBLE_TYPES: tuple[str, ...] = ("MEDICATION", "DIAGNOSIS", "SYMPTOM")
ASSERTION_INELIGIBLE_TYPES: tuple[str, ...] = ("TEST_NAME", "TEST_RESULT")
INELIGIBLE_TYPE_POLICY = (
    "organizer_field_policy: TEST_NAME and TEST_RESULT carry no assertions field")

# Tri-state supervision. UNKNOWN is a real value, not a missing key.
SUPERVISED_TRUE = "supervised_true"
SUPERVISED_FALSE = "supervised_false"
UNKNOWN = "unknown"
SUPERVISION_STATES: tuple[str, ...] = (SUPERVISED_TRUE, SUPERVISED_FALSE, UNKNOWN)

# Provenance channels, kept strictly apart.
PROVENANCE_TRUSTED = "trusted"
PROVENANCE_WEAK = "weak"
PROVENANCE_CHANNELS: tuple[str, ...] = (PROVENANCE_TRUSTED, PROVENANCE_WEAK)


class S2AssertionContractError(ValueError):
    """Raised when an S2 example or dataset violates its contract."""


@dataclass(frozen=True, slots=True)
class S2ArtifactValidationReport:
    """Read-only validation report for a future S2 training artifact."""

    artifact_dir: str
    expected_mode: str
    ok: bool
    failures: tuple[str, ...]
    checkpoint_hashes: Mapping[str, str]
    artifact_schema_version: str = S2_ARTIFACT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_dir": self.artifact_dir,
            "expected_expert_id": S2_EXPERT_ID,
            "expected_mode": self.expected_mode,
            "ok": self.ok,
            "failures": list(self.failures),
            "checkpoint_hashes": dict(self.checkpoint_hashes),
            "internal_test_accessed": False,
        }


@dataclass(frozen=True, slots=True)
class AssertionLabels:
    """Tri-state labels for one mention. Never defaults a missing label to false."""

    states: Mapping[str, str]
    provenance: str = PROVENANCE_TRUSTED

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_CHANNELS:
            raise S2AssertionContractError(f"unknown provenance {self.provenance!r}")
        unknown_labels = set(self.states) - set(ASSERTION_LABEL_ORDER)
        if unknown_labels:
            raise S2AssertionContractError(
                f"unsupported assertion labels: {sorted(unknown_labels)}")
        for label, state in self.states.items():
            if state not in SUPERVISION_STATES:
                raise S2AssertionContractError(
                    f"{label}: invalid supervision state {state!r}")

    @classmethod
    def all_unknown(cls, provenance: str = PROVENANCE_TRUSTED) -> AssertionLabels:
        return cls({label: UNKNOWN for label in ASSERTION_LABEL_ORDER}, provenance)

    def state_for(self, label: str) -> str:
        """UNKNOWN for an absent label — absence is never evidence of falsity."""
        return self.states.get(label, UNKNOWN)

    @property
    def has_any_supervision(self) -> bool:
        return any(self.state_for(label) != UNKNOWN for label in ASSERTION_LABEL_ORDER)

    @property
    def supervised_label_count(self) -> int:
        return sum(1 for label in ASSERTION_LABEL_ORDER
                   if self.state_for(label) != UNKNOWN)

    def target_vector(self) -> tuple[float | None, ...]:
        """Deterministic per-label targets; ``None`` where supervision is absent.

        A ``None`` must be masked out of the loss. It is deliberately not 0.0.
        """
        mapping = {SUPERVISED_TRUE: 1.0, SUPERVISED_FALSE: 0.0, UNKNOWN: None}
        return tuple(mapping[self.state_for(label)] for label in ASSERTION_LABEL_ORDER)

    def loss_mask(self) -> tuple[int, ...]:
        return tuple(0 if value is None else 1 for value in self.target_vector())

    def as_dict(self) -> dict[str, Any]:
        return {
            "states": {label: self.state_for(label) for label in ASSERTION_LABEL_ORDER},
            "provenance": self.provenance,
            "supervised_label_count": self.supervised_label_count,
        }


@dataclass(frozen=True, slots=True)
class AssertionExample:
    """One mention-conditioned assertion example with exact original offsets."""

    document_id: str
    example_id: str
    source_dataset: str
    entity_type: str
    start: int
    end: int
    mention_text: str
    labels: AssertionLabels
    group_id: str = ""

    def __post_init__(self) -> None:
        if self.entity_type not in ASSERTION_ELIGIBLE_TYPES:
            raise S2AssertionContractError(
                f"{self.entity_type} is not assertion-eligible "
                f"({INELIGIBLE_TYPE_POLICY})")
        if self.end <= self.start:
            raise S2AssertionContractError(
                f"invalid mention offsets {self.start}:{self.end}")

    def validate_against(self, original_text: str) -> None:
        """Spec §4: the mention must still slice exactly out of the source text."""
        if original_text[self.start:self.end] != self.mention_text:
            raise S2AssertionContractError(
                f"{self.example_id}: mention [{self.start}, {self.end}) does not "
                "slice out of original_text")

    @property
    def leakage_group(self) -> str:
        """Grouping key for the split guard: the document, not the mention."""
        return self.group_id or self.document_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "example_id": self.example_id,
            "source_dataset": self.source_dataset,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "labels": self.labels.as_dict(),
            "leakage_group": self.leakage_group,
        }


@dataclass(frozen=True, slots=True)
class AssertionDatasetReport:
    """What supervision actually exists. Designed to make emptiness obvious."""

    split: str
    rows_scanned: int
    entities_scanned: int
    eligible_mentions: int
    ineligible_mentions: int
    trusted_examples: int
    weak_examples: int
    examples_with_any_supervision: int
    supervision_counts: Mapping[str, Mapping[str, int]]
    by_entity_type: Mapping[str, int]
    by_source: Mapping[str, int]
    coverage_manifest_declares_assertions: Mapping[str, bool] = field(
        default_factory=dict)
    ineligible_type_policy: str = INELIGIBLE_TYPE_POLICY
    input_contract_version: str = S2_INPUT_CONTRACT_VERSION

    @property
    def trainable(self) -> bool:
        """S2 may only train when trusted supervision actually exists."""
        return self.trusted_examples > 0 and self.examples_with_any_supervision > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "rows_scanned": self.rows_scanned,
            "entities_scanned": self.entities_scanned,
            "eligible_mentions": self.eligible_mentions,
            "ineligible_mentions": self.ineligible_mentions,
            "trusted_examples": self.trusted_examples,
            "weak_examples": self.weak_examples,
            "examples_with_any_supervision": self.examples_with_any_supervision,
            "supervision_counts": {
                label: dict(counts) for label, counts in self.supervision_counts.items()},
            "by_entity_type": dict(self.by_entity_type),
            "by_source": dict(self.by_source),
            "coverage_manifest_declares_assertions": dict(
                self.coverage_manifest_declares_assertions),
            "ineligible_type_policy": self.ineligible_type_policy,
            "input_contract_version": self.input_contract_version,
            "trainable": self.trainable,
            "internal_test_accessed": False,
        }


def extract_assertion_labels(entity: Mapping[str, Any]) -> AssertionLabels:
    """Read tri-state labels from a governed entity.

    An entity with no ``assertions`` field yields **all UNKNOWN**, not all-false.
    A present mapping supplies ``supervised_true`` / ``supervised_false`` only for
    the labels it actually names; anything it omits stays UNKNOWN.
    """
    payload = entity.get("assertions")
    if payload is None:
        return AssertionLabels.all_unknown()
    if not isinstance(payload, Mapping):
        raise S2AssertionContractError(
            "governed entity 'assertions' must be a mapping of label -> bool")
    states: dict[str, str] = {}
    for label in ASSERTION_LABEL_ORDER:
        if label not in payload:
            states[label] = UNKNOWN
            continue
        value = payload[label]
        if value is None:
            states[label] = UNKNOWN
        elif isinstance(value, bool):
            states[label] = SUPERVISED_TRUE if value else SUPERVISED_FALSE
        else:
            raise S2AssertionContractError(
                f"{label}: assertion supervision must be boolean or null, got "
                f"{type(value).__name__}")
    return AssertionLabels(states, PROVENANCE_TRUSTED)


def build_assertion_examples(
    row: Mapping[str, Any],
    *,
    require_offset_invariant: bool = True,
) -> tuple[tuple[AssertionExample, ...], int]:
    """``(eligible examples, ineligible mention count)`` for one governed row."""
    text = str(row.get("text", ""))
    document_id = str(row.get("document_id", row.get("example_id", "")))
    example_id = str(row.get("example_id", document_id))
    source = str(row.get("source_dataset", ""))
    group_ids = row.get("group_ids") or []
    group_id = str(group_ids[0]) if group_ids else document_id

    examples: list[AssertionExample] = []
    ineligible = 0
    for index, entity in enumerate(row.get("entities") or []):
        entity_type = str(entity.get("target_type", ""))
        if entity_type in ASSERTION_INELIGIBLE_TYPES:
            ineligible += 1
            continue
        if entity_type not in ASSERTION_ELIGIBLE_TYPES:
            ineligible += 1
            continue
        example = AssertionExample(
            document_id=document_id,
            example_id=f"{example_id}#assert-{index:04d}",
            source_dataset=source,
            entity_type=entity_type,
            start=int(entity["start"]),
            end=int(entity["end"]),
            mention_text=str(entity["text"]),
            labels=extract_assertion_labels(entity),
            group_id=group_id)
        if require_offset_invariant:
            example.validate_against(text)
        examples.append(example)
    return tuple(examples), ineligible


def build_s2_assertion_head(hidden_size: int = S2_DEFAULT_HIDDEN_SIZE) -> object:
    """Build the compact S2 multi-task head lazily, with no torch import on import.

    The ViHealthBERT backbone is counted separately and may be shared with E3.
    This head accounts for the trainable assertion-specific parameters: cue,
    scope, entity-cue relation, and final mention assertion logits.
    """
    if hidden_size <= 0:
        raise S2AssertionContractError("hidden_size must be positive")
    from torch import nn

    class S2AssertionHead(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            label_count = len(ASSERTION_LABEL_ORDER)
            self.cue_head = nn.Linear(hidden_size, label_count)
            self.scope_head = nn.Linear(hidden_size, label_count)
            self.relation_head = nn.Linear(hidden_size, label_count)
            self.assertion_head = nn.Linear(hidden_size, label_count)

        def forward(self, mention_representation: Any) -> dict[str, Any]:
            return {
                "cue_logits": self.cue_head(mention_representation),
                "scope_logits": self.scope_head(mention_representation),
                "relation_logits": self.relation_head(mention_representation),
                "assertion_logits": self.assertion_head(mention_representation),
            }

    return S2AssertionHead()


def s2_head_parameter_count(hidden_size: int = S2_DEFAULT_HIDDEN_SIZE) -> int:
    """Closed-form count for the S2 head; tests also instantiate and count it."""
    label_count = len(ASSERTION_LABEL_ORDER)
    return 4 * (hidden_size * label_count + label_count)


def scan_assertion_supervision(
    split: str,
    path: str | Path,
    *,
    coverage_manifest: Mapping[str, Mapping[str, Any]] | None = None,
    max_rows: int | None = None,
) -> tuple[AssertionDatasetReport, tuple[AssertionExample, ...]]:
    """Scan a governed split and report exactly what supervision exists.

    internal_test is refused by name, and the report is designed so that "no
    supervision" is impossible to mistake for "all negative".
    """
    if split == "internal_test":
        raise S2AssertionContractError("S2 must never read internal_test")

    supervision_counts: dict[str, dict[str, int]] = {
        label: dict.fromkeys(SUPERVISION_STATES, 0) for label in ASSERTION_LABEL_ORDER}
    by_type: dict[str, int] = dict.fromkeys(ASSERTION_ELIGIBLE_TYPES, 0)
    by_source: dict[str, int] = {}
    collected: list[AssertionExample] = []
    rows = entities = eligible = ineligible = 0
    trusted = weak = supervised_any = 0

    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            entities += len(row.get("entities") or [])
            examples, skipped = build_assertion_examples(row)
            ineligible += skipped
            for example in examples:
                eligible += 1
                collected.append(example)
                by_type[example.entity_type] = by_type.get(example.entity_type, 0) + 1
                by_source[example.source_dataset] = by_source.get(
                    example.source_dataset, 0) + 1
                if example.labels.provenance == PROVENANCE_TRUSTED:
                    trusted += 1
                else:
                    weak += 1
                if example.labels.has_any_supervision:
                    supervised_any += 1
                for label in ASSERTION_LABEL_ORDER:
                    supervision_counts[label][example.labels.state_for(label)] += 1

    declared: dict[str, bool] = {}
    if coverage_manifest:
        declared = {
            source: bool(entry.get("assertions", False))
            for source, entry in coverage_manifest.items()}

    report = AssertionDatasetReport(
        split=split, rows_scanned=rows, entities_scanned=entities,
        eligible_mentions=eligible, ineligible_mentions=ineligible,
        trusted_examples=trusted if supervised_any else 0,
        weak_examples=weak,
        examples_with_any_supervision=supervised_any,
        supervision_counts=supervision_counts, by_entity_type=by_type,
        by_source=by_source, coverage_manifest_declares_assertions=declared)
    return report, tuple(collected)


def assert_no_document_leakage(
    train: Sequence[AssertionExample], validation: Sequence[AssertionExample],
) -> None:
    """Spec §15.2: no document may appear on both sides of the split."""
    train_groups = {example.leakage_group for example in train}
    validation_groups = {example.leakage_group for example in validation}
    overlap = train_groups & validation_groups
    if overlap:
        raise S2AssertionContractError(
            f"document leakage between train and validation: "
            f"{len(overlap)} shared group(s), e.g. {sorted(overlap)[:3]}")


def assert_trainable(report: AssertionDatasetReport) -> None:
    """Refuse to train S2 on fabricated labels.

    With zero trusted supervision the only way to produce a training signal would
    be to invent one. That is exactly what this raises to prevent.
    """
    if not report.trainable:
        raise S2AssertionContractError(
            f"S2 has no trusted assertion supervision in split {report.split!r}: "
            f"{report.eligible_mentions} eligible mentions but "
            f"{report.examples_with_any_supervision} with any supervised label. "
            "Training here would require fabricating labels, which is refused. "
            "Acquire an assertion-annotated corpus first.")


def multi_label_metrics(
    predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[float | None]],
) -> dict[str, Any]:
    """Exact per-label and micro multi-label metrics over SUPERVISED labels only.

    Positions whose target is ``None`` (unknown supervision) are excluded from
    every count rather than being scored as negatives.
    """
    if len(predictions) != len(targets):
        raise S2AssertionContractError("prediction/target row count mismatch")
    per_label: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0, "supervised": 0}
        for label in ASSERTION_LABEL_ORDER}

    for predicted_row, target_row in zip(predictions, targets, strict=True):
        if len(predicted_row) != len(ASSERTION_LABEL_ORDER) or len(target_row) != len(
                ASSERTION_LABEL_ORDER):
            raise S2AssertionContractError(
                "each row must have one value per assertion label")
        for index, label in enumerate(ASSERTION_LABEL_ORDER):
            target = target_row[index]
            if target is None:
                continue
            bucket = per_label[label]
            bucket["supervised"] += 1
            predicted = int(predicted_row[index])
            if predicted == 1 and target == 1.0:
                bucket["tp"] += 1
            elif predicted == 1 and target == 0.0:
                bucket["fp"] += 1
            elif predicted == 0 and target == 1.0:
                bucket["fn"] += 1

    def score(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    micro_tp = sum(bucket["tp"] for bucket in per_label.values())
    micro_fp = sum(bucket["fp"] for bucket in per_label.values())
    micro_fn = sum(bucket["fn"] for bucket in per_label.values())
    return {
        "label_order": list(ASSERTION_LABEL_ORDER),
        "per_label": {
            label: {**bucket, **score(bucket["tp"], bucket["fp"], bucket["fn"])}
            for label, bucket in per_label.items()},
        "micro": {"tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
                  **score(micro_tp, micro_fp, micro_fn)},
        "supervised_positions": sum(
            bucket["supervised"] for bucket in per_label.values()),
        "internal_test_accessed": False,
    }


def build_s2_resolved_config(
    *,
    mode: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    seed: int,
    max_length: int,
    micro_batch_size: int,
    accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    requested_precision: str = "bf16",
    allow_weak_supervision: bool = False,
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "stage_id": S2_STAGE_ID,
        "expert_id": S2_EXPERT_ID,
        "mode": mode,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "seed": seed,
        "max_length": max_length,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": micro_batch_size * accumulation_steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "requested_precision": requested_precision,
        "precision_policy": "bf16_on_capable_cuda_else_fp16_with_grad_scaler",
        "full_training_requires_cuda": True,
        "best_criterion": "max_validation_macro_f1_governed_validation_only",
        "label_order": list(ASSERTION_LABEL_ORDER),
        "assertion_eligible_types": list(ASSERTION_ELIGIBLE_TYPES),
        "ineligible_type_policy": INELIGIBLE_TYPE_POLICY,
        "supervision_states": list(SUPERVISION_STATES),
        "unknown_supervision_is_masked_not_false": True,
        "allow_weak_supervision": allow_weak_supervision,
        "weak_supervision_evaluated_separately": True,
        "s2_input_contract_version": S2_INPUT_CONTRACT_VERSION,
        "s2_checkpoint_schema_version": S2_CHECKPOINT_SCHEMA_VERSION,
        "head_parameter_count": s2_head_parameter_count(),
        "internal_test_accessed": False,
    }
    if progress is not None:
        config["progress"] = dict(progress)
    return config


def s2_config_sha256(resolved_config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(resolved_config))


def build_s2_checkpoint_payload(
    *,
    mode: str,
    config_sha256: str,
    model_revision: str,
    tokenizer_revision: str,
    parameter_count: int,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": S2_CHECKPOINT_SCHEMA_VERSION,
        "stage_id": S2_STAGE_ID,
        "expert_id": S2_EXPERT_ID,
        "mode": mode,
        "config_sha256": config_sha256,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "label_order": list(ASSERTION_LABEL_ORDER),
        "parameter_count": parameter_count,
        "internal_test_accessed": False,
        "model_state": {},
        "optimizer_state": {},
        "scaler_state": {},
    }


def validate_s2_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_mode: str,
    expected_config_sha256: str,
) -> None:
    missing = tuple(key for key in S2_REQUIRED_CHECKPOINT_KEYS if key not in payload)
    if missing:
        raise S2AssertionContractError(
            "S2 checkpoint missing fields: " + ", ".join(sorted(missing)))
    expected = {
        "checkpoint_schema_version": S2_CHECKPOINT_SCHEMA_VERSION,
        "stage_id": S2_STAGE_ID,
        "expert_id": S2_EXPERT_ID,
        "mode": expected_mode,
        "config_sha256": expected_config_sha256,
        "label_order": list(ASSERTION_LABEL_ORDER),
        "internal_test_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise S2AssertionContractError(f"S2 checkpoint field mismatch: {key}")
    if int(payload.get("parameter_count", -1)) < 0:
        raise S2AssertionContractError("S2 checkpoint parameter_count must be non-negative")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"json_unreadable:{path.name}:{exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path.name}")
        return {}
    return payload


def validate_s2_artifact(artifact_dir: str | Path, *, mode: str = "full"
                         ) -> S2ArtifactValidationReport:
    """Validate a future S2 artifact without loading model weights."""
    root = Path(artifact_dir)
    failures: list[str] = []
    for relative in S2_REQUIRED_ARTIFACT_FILES:
        if not (root / relative).is_file():
            failures.append(f"missing:{relative}")

    resolved = _read_json(root / "resolved_config.json", failures)
    metrics = _read_json(root / "validation_metrics.json", failures)
    manifest = _read_json(root / "training_manifest.json", failures)

    for name, payload in (("resolved_config", resolved), ("training_manifest", manifest)):
        if payload:
            if payload.get("stage_id") != S2_STAGE_ID:
                failures.append(f"{name}_stage_id_mismatch")
            if payload.get("expert_id") != S2_EXPERT_ID:
                failures.append(f"{name}_expert_id_mismatch")
            if payload.get("mode") != mode:
                failures.append(f"{name}_mode_mismatch")
            if bool(payload.get("internal_test_accessed", False)):
                failures.append(f"{name}_internal_test_accessed")
            if tuple(payload.get("label_order", ())) != ASSERTION_LABEL_ORDER:
                failures.append(f"{name}_label_order_mismatch")
    if metrics and bool(metrics.get("internal_test_accessed", False)):
        failures.append("validation_metrics_internal_test_accessed")

    checkpoint_hashes: dict[str, str] = {}
    expected_config = str(manifest.get("config_sha256", resolved.get("config_sha256", "")))
    for checkpoint_name in ("best.pt", "latest.pt"):
        path = root / "checkpoints" / checkpoint_name
        if not path.is_file():
            continue
        checkpoint_hashes[checkpoint_name] = _sha256_file(path)
        payload = _read_json(path, failures)
        if payload:
            try:
                validate_s2_checkpoint_payload(
                    payload,
                    expected_mode=mode,
                    expected_config_sha256=expected_config,
                )
            except S2AssertionContractError as exc:
                failures.append(f"checkpoint_invalid:{checkpoint_name}:{exc}")

    return S2ArtifactValidationReport(
        artifact_dir=str(root), expected_mode=mode, ok=not failures,
        failures=tuple(failures), checkpoint_hashes=checkpoint_hashes)


def iter_supervised(examples: Iterable[AssertionExample]) -> Iterable[AssertionExample]:
    return (example for example in examples if example.labels.has_any_supervision)


__all__ = [
    "ASSERTION_ELIGIBLE_TYPES",
    "ASSERTION_INELIGIBLE_TYPES",
    "ASSERTION_LABELS",
    "ASSERTION_LABEL_ORDER",
    "INELIGIBLE_TYPE_POLICY",
    "PROVENANCE_CHANNELS",
    "PROVENANCE_TRUSTED",
    "PROVENANCE_WEAK",
    "S2_FULL_AUTHORIZATION",
    "S2_ARTIFACT_SCHEMA_VERSION",
    "S2_CHECKPOINT_SCHEMA_VERSION",
    "S2_DEFAULT_HIDDEN_SIZE",
    "S2_EXPERT_ID",
    "S2_INPUT_CONTRACT_VERSION",
    "S2_MODEL_ID",
    "S2_REQUIRED_ARTIFACT_FILES",
    "S2_REQUIRED_CHECKPOINT_KEYS",
    "S2_STAGE_ID",
    "SUPERVISED_FALSE",
    "SUPERVISED_TRUE",
    "SUPERVISION_STATES",
    "UNKNOWN",
    "AssertionDatasetReport",
    "AssertionExample",
    "AssertionLabels",
    "S2ArtifactValidationReport",
    "S2AssertionContractError",
    "assert_no_document_leakage",
    "assert_trainable",
    "build_assertion_examples",
    "build_s2_assertion_head",
    "build_s2_checkpoint_payload",
    "build_s2_resolved_config",
    "extract_assertion_labels",
    "iter_supervised",
    "multi_label_metrics",
    "s2_config_sha256",
    "s2_head_parameter_count",
    "scan_assertion_supervision",
    "validate_s2_artifact",
    "validate_s2_checkpoint_payload",
]
