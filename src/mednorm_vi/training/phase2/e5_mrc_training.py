"""Training-contract helpers for the E5 XLM-R MRC-NER expert."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...mention_factory.mrc import (
    MRC_QUERY_VERSION,
    TYPE_QUERIES_V1,
    TYPE_QUERY_ORDER,
    MRCExample,
    build_mrc_examples,
    pair_start_end,
)
from ...mention_factory.spans import EntitySpan
from .artifacts import (
    MODE_SMOKE,
    Phase2TrainingManifest,
    checkpoint_payload,
    write_checkpoint_payload,
)
from .common import canonical_json_sha256, sha256_file

E5_STAGE_ID = "phase2-e5-xlmr-mrc-ner-v1"
E5_MODEL_ID = "xlm-roberta-large"
E5_FULL_AUTHORIZATION = "I_AUTHORIZE_E5_FULL_TRAINING"


class E5TrainingContractError(ValueError):
    """Raised when an E5 training or artifact contract is invalid."""


@dataclass(frozen=True, slots=True)
class MRCBatchContract:
    examples: tuple[MRCExample, ...]
    query_hash: str
    max_span_chars: int
    allow_overlaps: bool

    @property
    def negative_query_count(self) -> int:
        return sum(1 for example in self.examples if not example.gold_spans)


def query_hash() -> str:
    return canonical_json_sha256(
        {
            "query_version": MRC_QUERY_VERSION,
            "queries": {key: TYPE_QUERIES_V1[key] for key in TYPE_QUERY_ORDER},
        }
    )


def build_mrc_batch_contract(
    document_id: str,
    original_text: str,
    entities: Sequence[EntitySpan],
    *,
    max_span_chars: int,
    allow_overlaps: bool,
) -> MRCBatchContract:
    examples = build_mrc_examples(document_id, original_text, entities)
    if not any(not example.gold_spans for example in examples):
        raise E5TrainingContractError("MRC conversion must retain negative query examples")
    for example in examples:
        query_seen = False
        context_seen = False
        for token, is_query, is_context in zip(
            example.tokens,
            example.query_mask,
            example.context_mask,
            strict=True,
        ):
            if is_query:
                query_seen = True
                if token.start != 0 or token.end != 0:
                    raise E5TrainingContractError("query tokens must not carry context offsets")
            if is_context:
                context_seen = True
        if not query_seen or not context_seen:
            raise E5TrainingContractError("MRC example must contain query and context tokens")
    return MRCBatchContract(
        examples=examples,
        query_hash=query_hash(),
        max_span_chars=max_span_chars,
        allow_overlaps=allow_overlaps,
    )


def _binary_cross_entropy(logit: float, label: int) -> float:
    target = float(label)
    if logit >= 0:
        return (1.0 - target) * logit + math.log1p(math.exp(-logit))
    return -target * logit + math.log1p(math.exp(logit))


def mrc_start_end_loss(
    example: MRCExample,
    start_logits: Sequence[float],
    end_logits: Sequence[float],
) -> float:
    if len(start_logits) != len(example.tokens) or len(end_logits) != len(example.tokens):
        raise E5TrainingContractError("MRC logits must align with tokens")
    total = 0.0
    count = 0
    for index, is_context in enumerate(example.context_mask):
        if not is_context:
            continue
        total += _binary_cross_entropy(float(start_logits[index]), example.start_labels[index])
        total += _binary_cross_entropy(float(end_logits[index]), example.end_labels[index])
        count += 2
    if count == 0:
        raise E5TrainingContractError("MRC loss has no context tokens")
    return total / count


def decode_mrc_logits(
    example: MRCExample,
    start_scores: Sequence[float],
    end_scores: Sequence[float],
    *,
    threshold: float,
    max_span_chars: int,
    allow_overlaps: bool,
) -> tuple[tuple[int, int, str], ...]:
    spans = pair_start_end(
        example,
        start_scores,
        end_scores,
        threshold=threshold,
        max_span_chars=max_span_chars,
        allow_overlaps=allow_overlaps,
    )
    return tuple((span.start, span.end, span.entity_type) for span in spans)


def assert_full_not_initialized_from_smoke(
    *,
    run_full_training: bool,
    resume_from_smoke_checkpoint: bool,
) -> None:
    if run_full_training and resume_from_smoke_checkpoint:
        raise E5TrainingContractError("E5 full training may not resume from a smoke checkpoint")


def build_e5_resolved_config(
    *,
    mode: str,
    model_revision: str,
    tokenizer_revision: str,
    seed: int,
    max_length: int,
    max_span_chars: int,
    effective_batch_size: int,
    allow_overlaps: bool,
) -> dict[str, Any]:
    return {
        "stage_id": E5_STAGE_ID,
        "expert_id": "E5_xlmr_mrc_ner",
        "mode": mode,
        "model_id": E5_MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "query_revision": MRC_QUERY_VERSION,
        "query_hash": query_hash(),
        "queries": {key: TYPE_QUERIES_V1[key] for key in TYPE_QUERY_ORDER},
        "seed": seed,
        "max_length": max_length,
        "max_span_chars": max_span_chars,
        "allow_overlaps": allow_overlaps,
        "effective_batch_size": effective_batch_size,
        "label_space": list(TYPE_QUERY_ORDER),
        "internal_test_accessed": False,
    }


def write_e5_checkpoint_stub(
    path: str | Path,
    *,
    mode: str,
    config_sha256: str,
    model_revision: str,
    tokenizer_revision: str,
    parameter_count: int,
) -> str:
    payload = checkpoint_payload(
        expert_id="E5_xlmr_mrc_ner",
        mode=mode,
        config_sha256=config_sha256,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        query_revision=MRC_QUERY_VERSION,
        parameter_count=parameter_count,
        label_space=TYPE_QUERY_ORDER,
    )
    write_checkpoint_payload(path, payload)
    return sha256_file(path)


def build_e5_manifest(
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
    manifest = Phase2TrainingManifest(
        stage_id=E5_STAGE_ID,
        expert_id="E5_xlmr_mrc_ner",
        mode=mode,
        status=status,
        run_completed=run_completed,
        interrupted_reason="",
        safe_to_resume=safe_to_resume,
        repository_commit=repository_commit,
        corpus_hashes=corpus_hashes,
        data_hashes=data_hashes,
        config_sha256=config_sha256,
        model_id=E5_MODEL_ID,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        query_revision=MRC_QUERY_VERSION,
        query_hash=query_hash(),
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
        label_space=TYPE_QUERY_ORDER,
        threshold_config={
            "query_hash_float_probe": float(int(query_hash()[:8], 16)),
        },
        best_latest_identical_allowed=mode == MODE_SMOKE,
        best_latest_identical_reason=(
            "bounded smoke may save identical best/latest after one validation point"
            if mode == MODE_SMOKE
            else ""
        ),
    )
    return manifest


__all__ = [
    "E5_FULL_AUTHORIZATION",
    "E5_MODEL_ID",
    "E5_STAGE_ID",
    "E5TrainingContractError",
    "MRCBatchContract",
    "assert_full_not_initialized_from_smoke",
    "build_e5_manifest",
    "build_e5_resolved_config",
    "build_mrc_batch_contract",
    "decode_mrc_logits",
    "mrc_start_end_loss",
    "query_hash",
    "write_e5_checkpoint_stub",
]
