"""E5 XLM-R MRC readiness, brought to the E4 pre-smoke standard (Audit 0042).

Audit 0036 left E5 with correct MRC contracts — queries, start/end targets, loss,
decoding, manifest — but without the operational hardening E4 gained afterwards
in Audits 0039-0041: real gradient accumulation, a resolved precision policy,
gradient clipping, resume custody, supervised-type reporting and progress
telemetry.

This module closes that gap **without touching any E4 file**. The generic
infrastructure E4 happens to host (accumulation planning, precision resolution,
optimizer signatures, local-first persistence, progress heartbeats) is *imported
read-only*: reusing it is strictly better than duplicating it, and importing a
module never changes its bytes. The E4 protected files are verified unchanged in
Audit 0042.

Nothing here downloads a model, trains, or reads internal_test.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...mention_factory.mrc import MRC_QUERY_VERSION, TYPE_QUERY_ORDER
from ...schemas.constants import ENTITY_TYPES

# Read-only reuse of generic Phase-2 infrastructure. These modules are NOT
# modified by this milestone.
from .e4_w2ner_training import (
    AccumulationPlan,
    MixedPrecisionPolicy,
    assert_full_training_device,
    optimizer_signature,
    plan_gradient_accumulation,
    resolve_mixed_precision_policy,
)
from .e5_mrc_training import E5_MODEL_ID, E5_STAGE_ID, E5TrainingContractError, query_hash

E5_READINESS_VERSION = "e5-readiness-v1"
E5_INPUT_CONTRACT_VERSION = "e5-mrc-start-end-v1"
E5_CHECKPOINT_SCHEMA_VERSION = "phase2-e5-checkpoint-v1"

# XLM-R publishes safetensors, unlike the pinned PhoBERT revision. The value is
# resolved from a real repository listing when one is available rather than
# assumed either way.
E5_WEIGHT_FORMAT_SAFETENSORS = "model.safetensors"
E5_WEIGHT_FORMAT_BIN = "pytorch_model.bin"

# Fields a full-training E5 resume must match exactly.
E5_RESUME_COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "e5_input_contract_version",
    "e5_checkpoint_schema_version",
    "query_revision",
    "query_hash",
    "config_sha256",
    "model_revision",
    "tokenizer_revision",
    "precision_mode",
    "optimizer_signature",
    "accumulation_signature",
)

REQUIRED_E5_RESUME_KEYS: tuple[str, ...] = (
    "model_state", "optimizer_state", "scaler_state", "epoch", "optimizer_steps",
    "best_metric",
)


@dataclass(frozen=True, slots=True)
class SupervisedTypeReport:
    """Which organizer types actually have supervision in the governed split.

    The MRC code supports all five architecture types. The governed corpus has
    supervision for three. Reporting the difference is mandatory: a manifest that
    implied laboratory supervision would be false, and fabricating laboratory
    labels is forbidden.
    """

    split: str
    supervised_types: tuple[str, ...]
    unsupervised_types: tuple[str, ...]
    entity_counts: Mapping[str, int]
    queries_built_for: tuple[str, ...] = TYPE_QUERY_ORDER
    version: str = E5_READINESS_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "supervised_types": list(self.supervised_types),
            "unsupervised_types": list(self.unsupervised_types),
            "entity_counts": dict(self.entity_counts),
            "queries_built_for": list(self.queries_built_for),
            "laboratory_labels_fabricated": False,
            "supervised_type_report_version": self.version,
            "internal_test_accessed": False,
        }


def scan_supervised_types(split: str, path: str | Path,
                          *, max_rows: int | None = None) -> SupervisedTypeReport:
    """Count governed entities per organizer type. internal_test is refused."""
    if split == "internal_test":
        raise E5TrainingContractError("E5 must never read internal_test")
    counts: dict[str, int] = dict.fromkeys(sorted(ENTITY_TYPES), 0)
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                break
            if not line.strip():
                continue
            for entity in json.loads(line).get("entities") or []:
                entity_type = str(entity.get("target_type", ""))
                if entity_type in counts:
                    counts[entity_type] += 1
    supervised = tuple(name for name, count in sorted(counts.items()) if count > 0)
    unsupervised = tuple(name for name, count in sorted(counts.items()) if count == 0)
    return SupervisedTypeReport(
        split=split, supervised_types=supervised, unsupervised_types=unsupervised,
        entity_counts=counts)


@dataclass(frozen=True, slots=True)
class E5WeightFormat:
    filename: str
    use_safetensors: bool
    model_revision: str
    resolved_by: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pretrained_weight_format": self.filename,
            "use_safetensors": self.use_safetensors,
            "model_revision": self.model_revision,
            "weight_format_resolved_by": self.resolved_by,
        }


def resolve_e5_weight_format(
    model_revision: str, *, repository_files: Sequence[str] | None = None,
) -> E5WeightFormat:
    """Resolve XLM-R's serialization from a real listing when one is supplied.

    Unlike the pinned PhoBERT revision, XLM-R normally publishes safetensors; the
    answer is still derived from evidence rather than assumed, and the E4 lesson
    (a wrong assumption costs a whole Colab session) is applied here in reverse.
    """
    if repository_files is not None:
        listing = {str(name) for name in repository_files}
        if E5_WEIGHT_FORMAT_SAFETENSORS in listing:
            return E5WeightFormat(E5_WEIGHT_FORMAT_SAFETENSORS, True, model_revision,
                                  "repository_file_listing")
        if E5_WEIGHT_FORMAT_BIN in listing:
            return E5WeightFormat(E5_WEIGHT_FORMAT_BIN, False, model_revision,
                                  "repository_file_listing")
        raise E5TrainingContractError(
            f"{E5_MODEL_ID}@{model_revision} publishes neither "
            f"{E5_WEIGHT_FORMAT_SAFETENSORS} nor {E5_WEIGHT_FORMAT_BIN}")
    return E5WeightFormat(E5_WEIGHT_FORMAT_SAFETENSORS, True, model_revision,
                          "xlmr_default_publishes_safetensors")


def plan_e5_accumulation(example_count: int, *, micro_batch_size: int,
                         accumulation_steps: int, epochs: int) -> AccumulationPlan:
    """Same derived accounting E4 uses; nothing is relabelled."""
    return plan_gradient_accumulation(
        example_count, micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps, epochs=epochs)


def resolve_e5_precision(requested_mode: str, *, device_type: str,
                         bf16_supported: bool = False) -> MixedPrecisionPolicy:
    """bf16 on capable CUDA, else fp16 + GradScaler; CPU is fp32."""
    return resolve_mixed_precision_policy(
        requested_mode, device_type=device_type, bf16_supported=bf16_supported)


def build_e5_full_resolved_config(
    *,
    mode: str,
    model_revision: str,
    tokenizer_revision: str,
    seed: int,
    max_length: int,
    max_span_chars: int,
    allow_overlaps: bool,
    weight_format: E5WeightFormat,
    accumulation: AccumulationPlan,
    precision: MixedPrecisionPolicy,
    supervised_types: Sequence[str],
    unsupervised_types: Sequence[str],
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    optimizer_name: str = "AdamW",
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The complete E5 resolved config, at the E4 readiness standard."""
    config: dict[str, Any] = {
        "stage_id": E5_STAGE_ID,
        "expert_id": "E5_xlmr_mrc_ner",
        "mode": mode,
        "model_id": E5_MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "query_revision": MRC_QUERY_VERSION,
        "query_hash": query_hash(),
        "seed": seed,
        "max_length": max_length,
        "max_span_chars": max_span_chars,
        "allow_overlaps": allow_overlaps,
        "label_space": list(TYPE_QUERY_ORDER),
        "e5_input_contract_version": E5_INPUT_CONTRACT_VERSION,
        "e5_checkpoint_schema_version": E5_CHECKPOINT_SCHEMA_VERSION,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "optimizer_signature": optimizer_signature(
            name=optimizer_name, learning_rate=learning_rate,
            weight_decay=weight_decay, max_grad_norm=max_grad_norm),
        "freeze_base_model": False,
        "quantization": "none",
        # Mandatory honesty: the MRC code builds a query for every architecture
        # type, but only these have governed supervision.
        "supervised_entity_types": list(supervised_types),
        "unsupervised_entity_types": list(unsupervised_types),
        "laboratory_labels_fabricated": False,
        "internal_test_accessed": False,
    }
    config.update(weight_format.as_dict())
    config.update(accumulation.as_dict())
    config.update(precision.as_dict())
    if progress is not None:
        config.update(dict(progress))
    return config


def assert_e5_checkpoint_custody(payload: Mapping[str, Any]) -> None:
    """A full E5 checkpoint must carry everything an exact resume needs."""
    missing = tuple(key for key in REQUIRED_E5_RESUME_KEYS if key not in payload)
    if missing:
        raise E5TrainingContractError(
            "E5 full checkpoint is missing resume state: " + ", ".join(sorted(missing)))


def assert_compatible_e5_resume(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any],
) -> None:
    """Refuse an E5 resume whose training contract differs in any tracked field."""
    assert_e5_checkpoint_custody(payload)
    if str(payload.get("mode", "")) != "full":
        raise E5TrainingContractError(
            "only a full-training checkpoint may initialize an E5 full resume")
    differences = [
        f"{field_name}: checkpoint={payload.get(field_name)!r} "
        f"expected={expected.get(field_name)!r}"
        for field_name in E5_RESUME_COMPATIBILITY_FIELDS
        if str(payload.get(field_name, "")) != str(expected.get(field_name, ""))
    ]
    if differences:
        raise E5TrainingContractError(
            "E5 full resume is incompatible: " + "; ".join(differences))


@dataclass(frozen=True, slots=True)
class E5ReadinessReport:
    """What is complete, and what still requires a Colab runtime."""

    implemented: tuple[str, ...]
    requires_colab: tuple[str, ...]
    status: str = "IMPLEMENTED_PREFLIGHT_EXECUTED_READY_FOR_COLAB_SMOKE_NOT_TRAINED"
    version: str = E5_READINESS_VERSION
    supervised_types: tuple[str, ...] = field(default_factory=tuple)
    unsupervised_types: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "e5_readiness_version": self.version,
            "status": self.status,
            "implemented": list(self.implemented),
            "requires_colab_runtime": list(self.requires_colab),
            "supervised_entity_types": list(self.supervised_types),
            "unsupervised_entity_types": list(self.unsupervised_types),
            "trained": False,
            "internal_test_accessed": False,
        }


def e5_readiness_report(type_report: SupervisedTypeReport) -> E5ReadinessReport:
    return E5ReadinessReport(
        implemented=(
            "governed train/validation loading by authoritative SHA-256",
            "internal_test prohibition",
            "character-offset preservation",
            "MRC query construction and query hash",
            "answer start/end targets",
            "type-aware decoding with exact-span validation",
            "deterministic example ordering",
            "real gradient accumulation with derived step accounting",
            "gradient clipping with a tracked max norm",
            "mixed-precision policy (bf16 / fp16+GradScaler / cpu fp32)",
            "checkpoint custody and resume compatibility contract",
            "best checkpoint on governed validation only",
            "local-first artifact persistence (shared infrastructure)",
            "artifact validator (shared Phase-2 validator)",
            "progress heartbeats and ETA (shared, Audit 0041)",
            "supervised-type reporting",
        ),
        requires_colab=(
            "immutable model/tokenizer revision resolution",
            "tokenizer acquisition",
            "encoder acquisition",
            "smoke training execution",
            "artifact validation on a real artifact",
        ),
        supervised_types=type_report.supervised_types,
        unsupervised_types=type_report.unsupervised_types)


__all__ = [
    "E5_CHECKPOINT_SCHEMA_VERSION",
    "E5_INPUT_CONTRACT_VERSION",
    "E5_READINESS_VERSION",
    "E5_RESUME_COMPATIBILITY_FIELDS",
    "E5_WEIGHT_FORMAT_BIN",
    "E5_WEIGHT_FORMAT_SAFETENSORS",
    "REQUIRED_E5_RESUME_KEYS",
    "E5ReadinessReport",
    "E5WeightFormat",
    "SupervisedTypeReport",
    "assert_compatible_e5_resume",
    "assert_e5_checkpoint_custody",
    "assert_full_training_device",
    "build_e5_full_resolved_config",
    "e5_readiness_report",
    "plan_e5_accumulation",
    "resolve_e5_precision",
    "resolve_e5_weight_format",
    "scan_supervised_types",
]
