"""Training-contract helpers for the E4 PhoBERT W2NER expert."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...mention_factory.w2ner import (
    EntitySpan,
    W2NERGrid,
    W2NERLabelVocab,
    build_w2ner_grid,
    decode_w2ner_grid,
    mask_padded_pairs,
)
from .artifacts import (
    MODE_SMOKE,
    Phase2TrainingManifest,
    checkpoint_payload,
    write_checkpoint_payload,
)
from .common import canonical_json_sha256, sha256_file

E4_STAGE_ID = "phase2-e4-phobert-w2ner-v1"
E4_MODEL_ID = "vinai/phobert-large"
E4_FULL_AUTHORIZATION = "I_AUTHORIZE_E4_FULL_TRAINING"


class E4TrainingContractError(ValueError):
    """Raised when an E4 training or artifact contract is invalid."""


@dataclass(frozen=True, slots=True)
class W2NERBatchContract:
    grid: W2NERGrid
    padded_labels: tuple[tuple[int, ...], ...]
    padded_pair_mask: tuple[tuple[bool, ...], ...]
    label_count: int

    @property
    def word_count(self) -> int:
        return len(self.grid.words)


def build_w2ner_batch_contract(
    document_id: str,
    original_text: str,
    entities: Sequence[EntitySpan],
    *,
    max_words: int,
    vocab: W2NERLabelVocab | None = None,
) -> W2NERBatchContract:
    grid = build_w2ner_grid(document_id, original_text, entities, vocab=vocab)
    if len(grid.words) > max_words:
        raise E4TrainingContractError("W2NER document exceeds configured max_words")
    padded_mask = mask_padded_pairs(len(grid.words), max_words)
    labels = [
        list(row) + [grid.vocab.none_id for _ in range(max_words - len(row))]
        for row in grid.labels
    ]
    labels.extend(
        [[grid.vocab.none_id for _ in range(max_words)] for _ in range(max_words - len(labels))]
    )
    return W2NERBatchContract(
        grid=grid,
        padded_labels=tuple(tuple(row) for row in labels),
        padded_pair_mask=padded_mask,
        label_count=len(grid.vocab.labels),
    )


def _log_softmax_loss(logits: Sequence[float], label_id: int) -> float:
    if label_id < 0 or label_id >= len(logits):
        raise E4TrainingContractError("label id outside relation-logit vector")
    maximum = max(float(value) for value in logits)
    total = sum(math.exp(float(value) - maximum) for value in logits)
    return -float(logits[label_id]) + maximum + math.log(total)


def w2ner_relation_loss(
    logits: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[Sequence[int]],
    pair_mask: Sequence[Sequence[bool]],
) -> float:
    """Cross-entropy over valid word-pairs using pure Python numeric inputs."""
    total = 0.0
    count = 0
    if len(logits) != len(labels) or len(labels) != len(pair_mask):
        raise E4TrainingContractError("W2NER logits, labels and masks must have same rows")
    for row_index, (logit_row, label_row, mask_row) in enumerate(
        zip(logits, labels, pair_mask, strict=True)
    ):
        if len(logit_row) != len(label_row) or len(label_row) != len(mask_row):
            raise E4TrainingContractError(f"W2NER row {row_index} has inconsistent width")
        for scores, label_id, valid in zip(logit_row, label_row, mask_row, strict=True):
            if not valid:
                continue
            total += _log_softmax_loss(scores, int(label_id))
            count += 1
    if count == 0:
        raise E4TrainingContractError("W2NER loss has no valid word-pairs")
    return total / count


def decode_argmax_relation_grid(
    batch: W2NERBatchContract,
    logits: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[int, ...], ...]:
    """Convert relation logits to a square label grid for deterministic decoding."""
    if len(logits) < batch.word_count:
        raise E4TrainingContractError("relation logits shorter than the word count")
    rows: list[tuple[int, ...]] = []
    for row_index in range(batch.word_count):
        row = logits[row_index]
        if len(row) < batch.word_count:
            raise E4TrainingContractError("relation logits row shorter than the word count")
        rows.append(
            tuple(
                max(
                    range(len(row[column_index])),
                    key=lambda label_index: float(row[column_index][label_index]),
                )
                for column_index in range(batch.word_count)
            )
        )
    return tuple(rows)


def decode_w2ner_logits(
    batch: W2NERBatchContract,
    logits: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[int, int, str], ...]:
    labels = decode_argmax_relation_grid(batch, logits)
    decoded_grid = W2NERGrid(
        document_id=batch.grid.document_id,
        original_text=batch.grid.original_text,
        words=batch.grid.words,
        labels=labels,
        pair_mask=batch.grid.pair_mask,
        vocab=batch.grid.vocab,
    )
    return tuple(
        (span.start, span.end, span.entity_type)
        for span in decode_w2ner_grid(decoded_grid)
    )


def assert_full_not_initialized_from_smoke(
    *,
    run_full_training: bool,
    resume_from_smoke_checkpoint: bool,
) -> None:
    if run_full_training and resume_from_smoke_checkpoint:
        raise E4TrainingContractError("E4 full training may not resume from a smoke checkpoint")


def build_e4_resolved_config(
    *,
    mode: str,
    model_revision: str,
    tokenizer_revision: str,
    seed: int,
    max_words: int,
    effective_batch_size: int,
) -> dict[str, Any]:
    return {
        "stage_id": E4_STAGE_ID,
        "expert_id": "E4_phobert_w2ner",
        "mode": mode,
        "model_id": E4_MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "seed": seed,
        "max_words": max_words,
        "effective_batch_size": effective_batch_size,
        "label_space": list(W2NERLabelVocab().type_order),
        "internal_test_accessed": False,
    }


def write_e4_checkpoint_stub(
    path: str | Path,
    *,
    mode: str,
    config_sha256: str,
    model_revision: str,
    tokenizer_revision: str,
    parameter_count: int,
) -> str:
    payload = checkpoint_payload(
        expert_id="E4_phobert_w2ner",
        mode=mode,
        config_sha256=config_sha256,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        parameter_count=parameter_count,
        label_space=W2NERLabelVocab().type_order,
    )
    write_checkpoint_payload(path, payload)
    return sha256_file(path)


def build_e4_manifest(
    *,
    mode: str,
    status: str,
    run_completed: bool,
    repository_commit: str,
    corpus_hashes: Mapping[str, str],
    data_hashes: Mapping[str, str],
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
) -> Phase2TrainingManifest:
    config_sha256 = canonical_json_sha256(dict(resolved_config))
    return Phase2TrainingManifest(
        stage_id=E4_STAGE_ID,
        expert_id="E4_phobert_w2ner",
        mode=mode,
        status=status,
        run_completed=run_completed,
        interrupted_reason="",
        safe_to_resume=safe_to_resume,
        repository_commit=repository_commit,
        corpus_hashes=corpus_hashes,
        data_hashes=data_hashes,
        config_sha256=config_sha256,
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
        best_latest_identical_allowed=mode == MODE_SMOKE,
        best_latest_identical_reason=(
            "bounded smoke may save identical best/latest after one validation point"
            if mode == MODE_SMOKE
            else ""
        ),
    )


__all__ = [
    "E4_FULL_AUTHORIZATION",
    "E4_MODEL_ID",
    "E4_STAGE_ID",
    "E4TrainingContractError",
    "W2NERBatchContract",
    "assert_full_not_initialized_from_smoke",
    "build_e4_manifest",
    "build_e4_resolved_config",
    "build_w2ner_batch_contract",
    "decode_argmax_relation_grid",
    "decode_w2ner_logits",
    "w2ner_relation_loss",
    "write_e4_checkpoint_stub",
]
