"""E4 identity, weight-format and artifact contracts (Audit 0045).

This is the *current* E4. The implementation that produced the collapsed run
audited in 0043/0044 has been removed; what survives here are the contracts that
were never implicated in the failure — the pinned revision, the atomic grid-word
coordinate system, the governed corpus identity, and the artifact/checkpoint
schema — carried forward under a stable, non-versioned path.

Two things are deliberately *not* carried forward:

* the per-example mean cross-entropy that made an all-background solution a
  stationary point (Audit 0044 §9). Loss now lives in :mod:`.recipes`;
* the single learning-rate optimizer signature. A backbone and a freshly
  initialized relation head must not share a learning rate, so the signature is
  defined over both groups in :mod:`.training`.

Nothing here trains, downloads a model, or reads internal_test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....mention_factory.w2ner import (
    ATOMIC_WORD_POLICY_VERSION,
    W2NER_CONFIG_VERSION,
    W2NERLabelVocab,
)
from ..artifacts import Phase2TrainingManifest, checkpoint_payload
from ..common import canonical_json_sha256, sha256_file

E4_STAGE_ID = "phase2-e4-clean-v1"
E4_EXPERT_ID = "E4_phobert_w2ner"
E4_MODEL_ID = "vinai/phobert-large"

# The revision the repository has governed since Audit 0038. Unchanged: the
# collapse was an optimization failure, not a base-model problem.
E4_PINNED_MODEL_REVISION = "1c7880f20db59c0054c6de5afd71b012369f6ee4"

# The grid coordinate system (Audit 0038) is unchanged and was proven exact by
# the Audit-0043 gold-grid round-trip: 13,711 governed entities reconstructed
# with P = R = F1 = 1.0. It is the one part of the failed run that measurement
# fully vindicated.
E4_INPUT_CONTRACT_VERSION = "e4-atomic-grid-word-v1"
ATOMIC_PROJECTION_VERSION = "atomic-projection-v1"
ATOMIC_FEATURE_DIM = 3

# A NEW checkpoint schema. Nothing written by the removed implementation may be
# loaded into this one, and the version bump is what makes that mechanical
# rather than a matter of discipline.
E4_CHECKPOINT_SCHEMA_VERSION = "phase2-e4-checkpoint-v3"
E4_REJECTED_CHECKPOINT_SCHEMA_VERSIONS: tuple[str, ...] = (
    "phase2-e4-checkpoint-v1",   # Audit 0037 segmented-model-word grid
    "phase2-e4-checkpoint-v2",   # Audit 0038-0044 collapsed implementation
    "",
)

E4_GOVERNED_TRAIN_SHA256 = (
    "892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a")
E4_GOVERNED_VALIDATION_SHA256 = (
    "ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103")
FORBIDDEN_SPLIT_NAMES = frozenset({"internal_test"})

# ---------------------------------------------------------------------------
# Supervision scope
# ---------------------------------------------------------------------------
#
# The organizer schema has five entity types and the relation grid keeps all
# seven labels, so the output space is unchanged. But the governed E4 corpus
# contains ZERO instances of TEST_NAME and TEST_RESULT in train and validation
# alike (Audit 0043 §6.2, re-measured), which means two of the seven classifier
# outputs have no training signal whatsoever.
#
# That absence is recorded, never papered over. No synthetic laboratory mention
# is generated, no class is dropped from the head, and no metric silently
# excludes them. Laboratory extraction is E1/E2's job and the resolver's; E4
# proposes only what it is actually supervised for.
E4_SUPERVISED_TYPES: tuple[str, ...] = ("DIAGNOSIS", "MEDICATION", "SYMPTOM")
E4_UNSUPERVISED_TYPES: tuple[str, ...] = ("TEST_NAME", "TEST_RESULT")
E4_UNSUPERVISED_TYPE_POLICY = (
    "the governed E4 corpus contains no TEST_NAME or TEST_RESULT mention; those "
    "classes remain in the output schema, receive no fabricated supervision, and "
    "are left to E1/E2 and the L4 resolver")

E4_FULL_AUTHORIZATION = "I_AUTHORIZE_E4_FULL_TRAINING"
E4_TINY_AUTHORIZATION = "I_AUTHORIZE_E4_TINY_RECIPE_ABLATION"
E4_SUBSET_AUTHORIZATION = "I_AUTHORIZE_E4_SUBSET_SMOKE"

# ---------------------------------------------------------------------------
# Pretrained weight format
# ---------------------------------------------------------------------------
#
# The pinned official revision publishes `pytorch_model.bin` and no
# `model.safetensors`; requesting safetensors is what broke a real Colab run in
# Audit 0039, and the Audit-0044 probe re-confirmed the repository file listing.
E4_WEIGHT_FORMAT_BIN = "pytorch_model.bin"
E4_WEIGHT_FORMAT_SAFETENSORS = "model.safetensors"
E4_SAFETENSORS_INDEX = "model.safetensors.index.json"
E4_KNOWN_BIN_ONLY_REVISIONS: tuple[str, ...] = (E4_PINNED_MODEL_REVISION,)


class E4ContractError(ValueError):
    """Raised when an E4 training or artifact contract is violated."""


def assert_split_allowed(split: str) -> str:
    if split in FORBIDDEN_SPLIT_NAMES:
        raise E4ContractError(f"refusing to open the frozen split {split!r}")
    return split


def supervised_types_present(entity_types: Sequence[str]) -> tuple[str, ...]:
    """Supervised types actually present, in the canonical order.

    Used by the stage gates so "this type was never predicted" and "this type
    was never in the data" stay distinguishable.
    """
    present = set(entity_types)
    return tuple(t for t in E4_SUPERVISED_TYPES if t in present)


@dataclass(frozen=True, slots=True)
class PhoBERTWeightFormat:
    """Which serialization the pinned revision actually publishes."""

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


def resolve_phobert_weight_format(
    model_id: str,
    model_revision: str,
    *,
    repository_files: Sequence[str] | None = None,
) -> PhoBERTWeightFormat:
    """Decide the weight format for a pinned revision, deterministically."""
    if repository_files is not None:
        listing = {str(name) for name in repository_files}
        if E4_WEIGHT_FORMAT_SAFETENSORS in listing or E4_SAFETENSORS_INDEX in listing:
            return PhoBERTWeightFormat(
                filename=E4_WEIGHT_FORMAT_SAFETENSORS, use_safetensors=True,
                model_revision=model_revision, resolved_by="repository_file_listing")
        if E4_WEIGHT_FORMAT_BIN not in listing:
            raise E4ContractError(
                f"{model_id}@{model_revision} publishes neither "
                f"{E4_WEIGHT_FORMAT_SAFETENSORS} nor {E4_WEIGHT_FORMAT_BIN}")
        return PhoBERTWeightFormat(
            filename=E4_WEIGHT_FORMAT_BIN, use_safetensors=False,
            model_revision=model_revision, resolved_by="repository_file_listing")
    resolved_by = (
        "known_bin_only_revision"
        if model_revision in E4_KNOWN_BIN_ONLY_REVISIONS
        else "official_phobert_default")
    return PhoBERTWeightFormat(
        filename=E4_WEIGHT_FORMAT_BIN, use_safetensors=False,
        model_revision=model_revision, resolved_by=resolved_by)


def assert_weight_format_loadable(weight_format: PhoBERTWeightFormat) -> None:
    """Refuse a safetensors request for a revision that does not publish it."""
    if (weight_format.use_safetensors
            and weight_format.model_revision in E4_KNOWN_BIN_ONLY_REVISIONS):
        raise E4ContractError(
            f"{E4_MODEL_ID}@{weight_format.model_revision} does not publish "
            f"{E4_WEIGHT_FORMAT_SAFETENSORS}; use_safetensors must be False")
    if weight_format.use_safetensors != (
            weight_format.filename == E4_WEIGHT_FORMAT_SAFETENSORS):
        raise E4ContractError(
            "use_safetensors disagrees with the resolved weight-format filename")


def validate_phobert_encoder_load_report(
    *, missing_keys: Sequence[str], unexpected_keys: Sequence[str],
) -> dict[str, Any]:
    """Accept only the MLM-head mismatch expected when loading the base encoder."""
    expected_prefixes = ("lm_head.", "roberta.lm_head.", "cls.", "roberta.cls.")

    def is_mlm_head(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in expected_prefixes)

    bad = [k for k in (*missing_keys, *unexpected_keys) if not is_mlm_head(k)]
    if bad:
        raise E4ContractError(
            f"PhoBERT base encoder load had unexpected keys: {sorted(bad)}")
    return {
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "ignored_mlm_head_keys": [
            k for k in (*missing_keys, *unexpected_keys) if is_mlm_head(k)],
        "w2ner_head_expected_from_base": False,
    }


# ---------------------------------------------------------------------------
# Governed split resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GovernedSplitResolution:
    split: str
    path: Path
    sha256: str


def resolve_governed_split_by_sha256(
    *, split: str, expected_sha256: str, search_roots: Sequence[str | Path],
) -> GovernedSplitResolution:
    """Locate a governed JSONL split by authoritative SHA-256, not by stale name."""
    assert_split_allowed(split)
    candidates: list[Path] = []
    for root in search_roots:
        path = Path(root)
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(sorted(path.rglob("*.jsonl")))
    for path in candidates:
        if path.is_file() and sha256_file(path) == expected_sha256:
            return GovernedSplitResolution(split=split, path=path, sha256=expected_sha256)
    raise FileNotFoundError(
        f"could not locate governed {split} split with SHA-256 {expected_sha256}")


def resolve_e4_governed_splits(
    search_roots: Sequence[str | Path],
) -> dict[str, GovernedSplitResolution]:
    return {
        "train": resolve_governed_split_by_sha256(
            split="train", expected_sha256=E4_GOVERNED_TRAIN_SHA256,
            search_roots=search_roots),
        "validation": resolve_governed_split_by_sha256(
            split="validation", expected_sha256=E4_GOVERNED_VALIDATION_SHA256,
            search_roots=search_roots),
    }


# ---------------------------------------------------------------------------
# Checkpoint identity
# ---------------------------------------------------------------------------


def e4_checkpoint_payload(
    *,
    mode: str,
    config_sha256: str,
    model_revision: str,
    tokenizer_revision: str,
    parameter_count: int,
    recipe_name: str,
) -> dict[str, Any]:
    """Shared Phase-2 payload plus this implementation's contract identity."""
    payload = checkpoint_payload(
        expert_id=E4_EXPERT_ID,
        mode=mode,
        config_sha256=config_sha256,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        parameter_count=parameter_count,
        label_space=W2NERLabelVocab().type_order,
    )
    payload["e4_input_contract_version"] = E4_INPUT_CONTRACT_VERSION
    payload["e4_checkpoint_schema_version"] = E4_CHECKPOINT_SCHEMA_VERSION
    payload["atomic_projection_version"] = ATOMIC_PROJECTION_VERSION
    payload["grid_word_surface"] = ATOMIC_WORD_POLICY_VERSION
    payload["atomic_feature_dim"] = ATOMIC_FEATURE_DIM
    payload["recipe"] = recipe_name
    payload["supervised_types"] = list(E4_SUPERVISED_TYPES)
    payload["unsupervised_types"] = list(E4_UNSUPERVISED_TYPES)
    return payload


def reject_superseded_checkpoint(payload: Mapping[str, Any]) -> None:
    """Refuse any checkpoint not written by this implementation.

    The collapsed run's checkpoints declare ``phase2-e4-checkpoint-v2``. They
    restore perfectly (Audit 0044 §6) and are therefore *loadable*, which is
    exactly why refusing them has to be mechanical: a v2 checkpoint carries the
    converged all-background solution, and initializing from it would reproduce
    the failure rather than retrain past it.
    """
    schema = str(payload.get("e4_checkpoint_schema_version", ""))
    if schema in E4_REJECTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise E4ContractError(
            f"checkpoint schema {schema!r} was written by a superseded E4 "
            f"implementation and must never initialize this one; expected "
            f"{E4_CHECKPOINT_SCHEMA_VERSION!r}. Train from pinned pretrained "
            "weights and a freshly initialized relation head instead.")
    if schema != E4_CHECKPOINT_SCHEMA_VERSION:
        raise E4ContractError(
            f"unknown E4 checkpoint schema {schema!r}; expected "
            f"{E4_CHECKPOINT_SCHEMA_VERSION!r}")
    contract = str(payload.get("e4_input_contract_version", ""))
    if contract != E4_INPUT_CONTRACT_VERSION:
        raise E4ContractError(
            f"checkpoint input contract {contract!r} is incompatible with "
            f"{E4_INPUT_CONTRACT_VERSION!r}")


def build_e4_manifest(
    *,
    mode: str,
    status: str,
    run_completed: bool,
    repository_commit: str,
    corpus_hashes: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    model_revision: str,
    tokenizer_revision: str,
    seed: int,
    completed_epochs: int,
    optimizer_steps: int,
    effective_batch_size: int,
    parameter_count: int,
    checkpoint_hashes: Mapping[str, str],
    best_metric: float,
    train_split_id: str,
    validation_split_id: str,
    safe_to_resume: bool,
    initialization_source: str,
    training_accounting: Mapping[str, Any] | None = None,
) -> Phase2TrainingManifest:
    return Phase2TrainingManifest(
        stage_id=E4_STAGE_ID,
        expert_id=E4_EXPERT_ID,
        mode=mode,
        status=status,
        run_completed=run_completed,
        interrupted_reason="",
        safe_to_resume=safe_to_resume,
        repository_commit=repository_commit,
        corpus_hashes=corpus_hashes,
        data_hashes=corpus_hashes,
        config_sha256=canonical_json_sha256(dict(resolved_config)),
        model_id=E4_MODEL_ID,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        query_revision="",
        query_hash="",
        seed=seed,
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        effective_batch_size=effective_batch_size,
        parameter_count=parameter_count,
        checkpoint_hashes=checkpoint_hashes,
        best_metric=best_metric,
        best_metric_name="validation_exact_f1",
        best_criterion="max_validation_exact_f1_governed_validation_only",
        train_split_id=train_split_id,
        validation_split_id=validation_split_id,
        internal_test_accessed=False,
        initialization_source=initialization_source,
        label_space=tuple(W2NERLabelVocab().type_order),
        threshold_config={},
        best_latest_identical_allowed=False,
        best_latest_identical_reason="",
        training_accounting=dict(training_accounting or {}),
    )


def build_e4_resolved_config(
    *,
    mode: str,
    recipe: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    schedule: Mapping[str, Any],
    order: Mapping[str, Any],
    accumulation: Mapping[str, Any],
    precision: Mapping[str, Any],
    weight_format: PhoBERTWeightFormat,
    model_revision: str,
    tokenizer_revision: str,
    seed: int,
    max_words: int,
    epochs: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    """Resolved config. Every field here changes the config hash by design."""
    assert_weight_format_loadable(weight_format)
    config: dict[str, Any] = {
        "stage_id": E4_STAGE_ID,
        "expert_id": E4_EXPERT_ID,
        "mode": mode,
        "model_id": E4_MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "seed": seed,
        "max_words": max_words,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "label_space": list(W2NERLabelVocab().type_order),
        "supervised_types": list(E4_SUPERVISED_TYPES),
        "unsupervised_types": list(E4_UNSUPERVISED_TYPES),
        "unsupervised_type_policy": E4_UNSUPERVISED_TYPE_POLICY,
        "freeze_base_model": False,
        "quantization": "none",
        "e4_input_contract_version": E4_INPUT_CONTRACT_VERSION,
        "e4_checkpoint_schema_version": E4_CHECKPOINT_SCHEMA_VERSION,
        "grid_word_surface": ATOMIC_WORD_POLICY_VERSION,
        "atomic_projection_version": ATOMIC_PROJECTION_VERSION,
        "atomic_feature_dim": ATOMIC_FEATURE_DIM,
        "w2ner_config_version": W2NER_CONFIG_VERSION,
        "internal_test_accessed": False,
    }
    config["recipe"] = dict(recipe)
    config["optimizer"] = dict(optimizer)
    config["schedule"] = dict(schedule)
    config["data_order"] = dict(order)
    config.update(dict(accumulation))
    config.update(dict(precision))
    config.update(weight_format.as_dict())
    return config


def build_e4_history_row(
    *,
    epoch: int,
    mode: str,
    recipe_name: str,
    losses: Mapping[str, float],
    validation_metrics: Mapping[str, Any],
    optimizer_steps: int,
    backward_passes: int,
    examples_processed: int,
    backbone_learning_rate: float,
    head_learning_rate: float,
) -> dict[str, Any]:
    """One per-epoch history record (spec §18.1 error/decision trail).

    Positive, background and total loss are recorded separately. A single total
    would have hidden the Audit-0044 collapse: the total kept falling while the
    positive term was the only thing that mattered and never improved.
    """
    return {
        "epoch": int(epoch),
        "mode": mode,
        "recipe": recipe_name,
        "train_loss": float(losses.get("total", 0.0)),
        "train_loss_positive": float(losses.get("positive", 0.0)),
        "train_loss_background": float(losses.get("background", 0.0)),
        "validation_exact_precision": float(
            validation_metrics.get("validation_exact_precision", 0.0)),
        "validation_exact_recall": float(
            validation_metrics.get("validation_exact_recall", 0.0)),
        "validation_exact_f1": float(
            validation_metrics.get("validation_exact_f1", 0.0)),
        "validation_predicted_total": int(
            validation_metrics.get("validation_predicted_total", 0)),
        "validation_gold_total": int(validation_metrics.get("validation_gold_total", 0)),
        "validation_thw_predictions": int(
            validation_metrics.get("validation_thw_predictions", 0)),
        "validation_nnw_predictions": int(
            validation_metrics.get("validation_nnw_predictions", 0)),
        "gold_positive_background_rate": float(
            validation_metrics.get("gold_positive_background_rate", 1.0)),
        "optimizer_steps": int(optimizer_steps),
        "backward_passes": int(backward_passes),
        "completed_examples": int(examples_processed),
        "backbone_learning_rate": float(backbone_learning_rate),
        "head_learning_rate": float(head_learning_rate),
        "internal_test_accessed": False,
    }


__all__ = [
    "ATOMIC_FEATURE_DIM",
    "ATOMIC_PROJECTION_VERSION",
    "E4_CHECKPOINT_SCHEMA_VERSION",
    "E4_EXPERT_ID",
    "E4_FULL_AUTHORIZATION",
    "E4_GOVERNED_TRAIN_SHA256",
    "E4_GOVERNED_VALIDATION_SHA256",
    "E4_INPUT_CONTRACT_VERSION",
    "E4_KNOWN_BIN_ONLY_REVISIONS",
    "E4_MODEL_ID",
    "E4_PINNED_MODEL_REVISION",
    "E4_REJECTED_CHECKPOINT_SCHEMA_VERSIONS",
    "E4_STAGE_ID",
    "E4_SUBSET_AUTHORIZATION",
    "E4_SUPERVISED_TYPES",
    "E4_TINY_AUTHORIZATION",
    "E4_UNSUPERVISED_TYPES",
    "E4_UNSUPERVISED_TYPE_POLICY",
    "E4_WEIGHT_FORMAT_BIN",
    "E4_WEIGHT_FORMAT_SAFETENSORS",
    "FORBIDDEN_SPLIT_NAMES",
    "E4ContractError",
    "GovernedSplitResolution",
    "PhoBERTWeightFormat",
    "assert_split_allowed",
    "assert_weight_format_loadable",
    "build_e4_history_row",
    "build_e4_manifest",
    "build_e4_resolved_config",
    "e4_checkpoint_payload",
    "reject_superseded_checkpoint",
    "resolve_e4_governed_splits",
    "resolve_governed_split_by_sha256",
    "resolve_phobert_weight_format",
    "supervised_types_present",
    "validate_phobert_encoder_load_report",
]
