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
    WordToken,
    build_w2ner_grid,
    decode_w2ner_grid,
    mask_padded_pairs,
)
from ..phobert_alignment import SegmentedWord as GovernedSegmentedWord
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
E4_GOVERNED_TRAIN_SHA256 = "892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a"
E4_GOVERNED_VALIDATION_SHA256 = "ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103"
EXPECTED_PHOBERT_MLM_HEAD_PREFIXES = (
    "lm_head.",
    "roberta.lm_head.",
    "cls.",
    "roberta.cls.",
)


class E4TrainingContractError(ValueError):
    """Raised when an E4 training or artifact contract is invalid."""


@dataclass(frozen=True, slots=True)
class E4SegmentedWord:
    """One W2NER word with both tokenizer surface and original offsets."""

    index: int
    model_text: str
    original_text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise E4TrainingContractError("segmented word has invalid original offsets")
        if self.original_text[self.start:self.end] == "":
            raise E4TrainingContractError("segmented word maps to an empty original slice")

    @property
    def original_slice(self) -> str:
        return self.original_text[self.start:self.end]

    def to_word_token(self) -> WordToken:
        token = WordToken(self.index, self.original_slice, self.start, self.end)
        token.validate_against(self.original_text)
        return token


@dataclass(frozen=True, slots=True)
class PhoBERTWordEncoding:
    """Encoder-ready PhoBERT inputs plus deterministic word/subtoken mapping."""

    model_inputs: Mapping[str, tuple[int, ...]]
    word_ids: tuple[int | None, ...]
    subword_indices_by_word: tuple[tuple[int, ...], ...]
    tokenizer_class: str
    tokenizer_is_fast: bool
    consumed_offset_mapping: bool

    def __post_init__(self) -> None:
        if "offset_mapping" in self.model_inputs:
            raise E4TrainingContractError("offset_mapping must never be passed to the encoder")
        input_ids = self.model_inputs.get("input_ids")
        if input_ids is None:
            raise E4TrainingContractError("PhoBERT model inputs must include input_ids")
        if len(input_ids) != len(self.word_ids):
            raise E4TrainingContractError("word-id mapping must match encoded input length")
        attention_mask = self.model_inputs.get("attention_mask")
        if attention_mask is not None and len(attention_mask) != len(input_ids):
            raise E4TrainingContractError("attention_mask length must match input_ids")
        for word_index, subtoken_indices in enumerate(self.subword_indices_by_word):
            if not subtoken_indices:
                raise E4TrainingContractError(
                    f"segmented word {word_index} has no PhoBERT subtokens"
                )
            for index in subtoken_indices:
                if index < 0 or index >= len(input_ids):
                    raise E4TrainingContractError("subtoken index outside encoded sequence")
                if self.word_ids[index] != word_index:
                    raise E4TrainingContractError("subtoken index does not map back to its word")


@dataclass(frozen=True, slots=True)
class GovernedSplitResolution:
    split: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class W2NERBatchContract:
    grid: W2NERGrid
    padded_labels: tuple[tuple[int, ...], ...]
    padded_pair_mask: tuple[tuple[bool, ...], ...]
    label_count: int
    segmented_words: tuple[E4SegmentedWord, ...] = ()

    @property
    def word_count(self) -> int:
        return len(self.grid.words)


def _pad_grid(grid: W2NERGrid, max_words: int) -> W2NERBatchContract:
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
    segmented_words = tuple(
        E4SegmentedWord(
            index=word.index,
            model_text=word.text,
            original_text=grid.original_text,
            start=word.start,
            end=word.end,
        )
        for word in grid.words
    )
    return W2NERBatchContract(
        grid=grid,
        padded_labels=tuple(tuple(row) for row in labels),
        padded_pair_mask=padded_mask,
        label_count=len(grid.vocab.labels),
        segmented_words=segmented_words,
    )


def build_w2ner_batch_contract(
    document_id: str,
    original_text: str,
    entities: Sequence[EntitySpan],
    *,
    max_words: int,
    vocab: W2NERLabelVocab | None = None,
) -> W2NERBatchContract:
    grid = build_w2ner_grid(document_id, original_text, entities, vocab=vocab)
    return _pad_grid(grid, max_words)


def _coerce_segmented_word(
    *,
    original_text: str,
    index: int,
    word: GovernedSegmentedWord | E4SegmentedWord,
) -> E4SegmentedWord:
    if isinstance(word, E4SegmentedWord):
        if word.index != index:
            raise E4TrainingContractError("segmented word indices must be contiguous")
        if word.original_text != original_text:
            raise E4TrainingContractError("segmented word original_text does not match document")
        return word
    return E4SegmentedWord(
        index=index,
        model_text=word.model_text,
        original_text=original_text,
        start=word.original_start,
        end=word.original_end,
    )


def build_w2ner_batch_contract_from_segmented_words(
    document_id: str,
    original_text: str,
    entities: Sequence[EntitySpan],
    segmented_words: Sequence[GovernedSegmentedWord | E4SegmentedWord],
    *,
    max_words: int,
    vocab: W2NERLabelVocab | None = None,
) -> W2NERBatchContract:
    """Build W2NER labels over governed VnCoreNLP segmented words.

    The tokenizer-facing ``model_text`` may contain segmenter join characters
    (for example ``suy_tim``), but the W2NER grid stores exact original slices so
    decoding always returns ``original_text[start:end]``.
    """
    e4_words = tuple(
        _coerce_segmented_word(original_text=original_text, index=index, word=word)
        for index, word in enumerate(segmented_words)
    )
    if not e4_words:
        raise E4TrainingContractError("W2NER segmented contract has no words")
    if len(e4_words) > max_words:
        raise E4TrainingContractError("W2NER document exceeds configured max_words")
    word_tokens = tuple(word.to_word_token() for word in e4_words)
    grid = build_w2ner_grid(
        document_id,
        original_text,
        entities,
        words=word_tokens,
        vocab=vocab,
    )
    contract = _pad_grid(grid, max_words)
    return W2NERBatchContract(
        grid=contract.grid,
        padded_labels=contract.padded_labels,
        padded_pair_mask=contract.padded_pair_mask,
        label_count=contract.label_count,
        segmented_words=e4_words,
    )


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise E4TrainingContractError("PhoBERT encoding must be a single sequence")
        values = values[0]
    return tuple(int(value) for value in values)


def _as_offset_tuple(values: Any) -> tuple[tuple[int, int], ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        if len(values) != 1:
            raise E4TrainingContractError("PhoBERT offset mapping must be a single sequence")
        values = values[0]
    return tuple((int(start), int(end)) for start, end in values)


def _tokenizer_pad_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = 0
    return int(value)


def _special_mask(
    tokenizer: Any,
    body_ids: Sequence[int],
    built_ids: Sequence[int],
) -> tuple[int, ...]:
    get_mask = getattr(tokenizer, "get_special_tokens_mask", None)
    if callable(get_mask):
        mask = tuple(
            int(value)
            for value in get_mask(list(body_ids), already_has_special_tokens=False)
        )
        if len(mask) == len(built_ids):
            return mask
    if len(built_ids) == len(body_ids) + 2:
        return (1, *([0] * len(body_ids)), 1)
    if len(built_ids) == len(body_ids):
        return tuple(0 for _ in built_ids)
    raise E4TrainingContractError("cannot derive special-token mask for PhoBERT encoding")


def _pad_encoding(
    input_ids: tuple[int, ...],
    attention_mask: tuple[int, ...],
    word_ids: tuple[int | None, ...],
    *,
    tokenizer: Any,
    max_length: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int | None, ...]]:
    if max_length is None:
        return input_ids, attention_mask, word_ids
    if len(input_ids) > max_length:
        raise E4TrainingContractError(
            "PhoBERT token sequence exceeds max_length; W2NER training refuses silent truncation"
        )
    pad = max_length - len(input_ids)
    if pad <= 0:
        return input_ids, attention_mask, word_ids
    return (
        (*input_ids, *([_tokenizer_pad_id(tokenizer)] * pad)),
        (*attention_mask, *([0] * pad)),
        (*word_ids, *([None] * pad)),
    )


def _indices_by_word(
    word_ids: Sequence[int | None],
    word_count: int,
) -> tuple[tuple[int, ...], ...]:
    buckets: list[list[int]] = [[] for _ in range(word_count)]
    for index, word_index in enumerate(word_ids):
        if word_index is None:
            continue
        if word_index < 0 or word_index >= word_count:
            raise E4TrainingContractError("encoded subtoken maps to an invalid word index")
        buckets[word_index].append(index)
    return tuple(tuple(bucket) for bucket in buckets)


def _tokenize_slow_phobert_words(
    tokenizer: Any,
    segmented_words: Sequence[E4SegmentedWord],
    *,
    max_length: int | None,
) -> PhoBERTWordEncoding:
    body_ids: list[int] = []
    body_word_ids: list[int] = []
    for word in segmented_words:
        pieces = tokenizer.tokenize(word.model_text)
        if not pieces:
            raise E4TrainingContractError(
                "PhoBERT tokenizer produced no pieces for a segmented word"
            )
        token_ids = tokenizer.convert_tokens_to_ids(pieces)
        if not isinstance(token_ids, list) or len(token_ids) != len(pieces):
            raise E4TrainingContractError("PhoBERT tokenizer returned mismatched piece ids")
        body_ids.extend(int(value) for value in token_ids)
        body_word_ids.extend([word.index] * len(token_ids))
    build_inputs = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(build_inputs):
        input_ids = tuple(int(value) for value in build_inputs(list(body_ids)))
    else:
        cls_id = int(getattr(tokenizer, "cls_token_id", 0) or 0)
        sep_id = int(getattr(tokenizer, "sep_token_id", 2) or 2)
        input_ids = (cls_id, *tuple(body_ids), sep_id)
    special_mask = _special_mask(tokenizer, body_ids, input_ids)
    word_ids: list[int | None] = []
    body_cursor = 0
    for is_special in special_mask:
        if is_special:
            word_ids.append(None)
            continue
        if body_cursor >= len(body_word_ids):
            raise E4TrainingContractError(
                "special-token mask has more body positions than token ids"
            )
        word_ids.append(body_word_ids[body_cursor])
        body_cursor += 1
    if body_cursor != len(body_word_ids):
        raise E4TrainingContractError("special-token mask did not consume every body token")
    attention_mask = tuple(1 for _ in input_ids)
    input_ids, attention_mask, padded_word_ids = _pad_encoding(
        input_ids,
        attention_mask,
        tuple(word_ids),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return PhoBERTWordEncoding(
        model_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        word_ids=padded_word_ids,
        subword_indices_by_word=_indices_by_word(padded_word_ids, len(segmented_words)),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_is_fast=False,
        consumed_offset_mapping=False,
    )


def _segmented_join_intervals(
    segmented_words: Sequence[E4SegmentedWord],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    pieces: list[str] = []
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for word in segmented_words:
        if pieces:
            cursor += 1
        pieces.append(word.model_text)
        start = cursor
        end = start + len(word.model_text)
        intervals.append((start, end))
        cursor = end
    return " ".join(pieces), tuple(intervals)


def _word_id_for_joined_offset(
    start: int,
    end: int,
    intervals: Sequence[tuple[int, int]],
) -> int | None:
    if start == end == 0:
        return None
    if end <= start:
        raise E4TrainingContractError("fast tokenizer returned an invalid offset")
    for index, (word_start, word_end) in enumerate(intervals):
        if word_start <= start and end <= word_end:
            return index
    raise E4TrainingContractError(
        "fast tokenizer offset did not map inside one governed segmented word"
    )


def _tokenize_fast_phobert_words(
    tokenizer: Any,
    segmented_words: Sequence[E4SegmentedWord],
    *,
    max_length: int | None,
) -> PhoBERTWordEncoding:
    joined, intervals = _segmented_join_intervals(segmented_words)
    encoded = tokenizer(
        joined,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )
    if "offset_mapping" not in encoded:
        raise E4TrainingContractError("fast tokenizer did not return offset_mapping")
    offsets = _as_offset_tuple(encoded["offset_mapping"])
    input_ids = _as_int_tuple(encoded["input_ids"])
    attention_mask = _as_int_tuple(encoded.get("attention_mask", [1] * len(input_ids)))
    if len(offsets) != len(input_ids):
        raise E4TrainingContractError("fast tokenizer offsets and input_ids differ in length")
    word_ids = tuple(_word_id_for_joined_offset(start, end, intervals) for start, end in offsets)
    input_ids, attention_mask, padded_word_ids = _pad_encoding(
        input_ids,
        attention_mask,
        word_ids,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return PhoBERTWordEncoding(
        model_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        word_ids=padded_word_ids,
        subword_indices_by_word=_indices_by_word(padded_word_ids, len(segmented_words)),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_is_fast=True,
        consumed_offset_mapping=True,
    )


def prepare_phobert_word_inputs(
    tokenizer: Any,
    segmented_words: Sequence[E4SegmentedWord],
    *,
    max_length: int | None = None,
) -> PhoBERTWordEncoding:
    """Tokenize governed segmented words without relying on slow offsets.

    Official PhoBERT commonly resolves to the slow ``PhobertTokenizer``. That
    backend cannot emit ``offset_mapping``, so the mandatory path tokenizes each
    VnCoreNLP segmented word and maps subtokens directly to the governed original
    word spans. If a true fast tokenizer is supplied, offsets are consumed only
    after explicitly requesting them and are removed from encoder inputs.
    """
    if not segmented_words:
        raise E4TrainingContractError("PhoBERT encoding requires at least one segmented word")
    for expected, word in enumerate(segmented_words):
        if word.index != expected:
            raise E4TrainingContractError("segmented word indices must be contiguous")
        if word.original_text[word.start:word.end] != word.original_slice:
            raise E4TrainingContractError("segmented word original-offset invariant failed")
    if bool(getattr(tokenizer, "is_fast", False)):
        return _tokenize_fast_phobert_words(tokenizer, segmented_words, max_length=max_length)
    return _tokenize_slow_phobert_words(tokenizer, segmented_words, max_length=max_length)


def pool_subtoken_embeddings(sequence_output: Any, encoding: PhoBERTWordEncoding) -> Any:
    """Mean-pool encoder states into W2NER word embeddings."""
    import torch

    pooled = []
    for indices in encoding.subword_indices_by_word:
        pooled.append(sequence_output[list(indices)].mean(dim=0))
    return torch.stack(pooled).unsqueeze(0)


def validate_phobert_encoder_load_report(
    *,
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
) -> dict[str, Any]:
    """Accept only the MLM-head mismatch expected when loading the base encoder."""
    bad_missing = [key for key in missing_keys if not _is_expected_mlm_head_key(key)]
    bad_unexpected = [key for key in unexpected_keys if not _is_expected_mlm_head_key(key)]
    if bad_missing or bad_unexpected:
        raise E4TrainingContractError(
            "PhoBERT base encoder load had unexpected missing/unexpected keys"
        )
    return {
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "ignored_mlm_head_keys": [
            key for key in (*missing_keys, *unexpected_keys) if _is_expected_mlm_head_key(key)
        ],
        "w2ner_head_expected_from_base": False,
    }


def _is_expected_mlm_head_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in EXPECTED_PHOBERT_MLM_HEAD_PREFIXES)


def resolve_governed_split_by_sha256(
    *,
    split: str,
    expected_sha256: str,
    search_roots: Sequence[str | Path],
) -> GovernedSplitResolution:
    """Locate a governed JSONL split by authoritative SHA-256, not by stale name."""
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
        f"could not locate governed {split} split with SHA-256 {expected_sha256}"
    )


def resolve_e4_governed_splits(
    search_roots: Sequence[str | Path],
) -> dict[str, GovernedSplitResolution]:
    return {
        "train": resolve_governed_split_by_sha256(
            split="train",
            expected_sha256=E4_GOVERNED_TRAIN_SHA256,
            search_roots=search_roots,
        ),
        "validation": resolve_governed_split_by_sha256(
            split="validation",
            expected_sha256=E4_GOVERNED_VALIDATION_SHA256,
            search_roots=search_roots,
        ),
    }


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
    "E4_GOVERNED_TRAIN_SHA256",
    "E4_GOVERNED_VALIDATION_SHA256",
    "E4_FULL_AUTHORIZATION",
    "E4_MODEL_ID",
    "E4SegmentedWord",
    "E4_STAGE_ID",
    "E4TrainingContractError",
    "GovernedSplitResolution",
    "PhoBERTWordEncoding",
    "W2NERBatchContract",
    "assert_full_not_initialized_from_smoke",
    "build_e4_manifest",
    "build_e4_resolved_config",
    "build_w2ner_batch_contract",
    "build_w2ner_batch_contract_from_segmented_words",
    "decode_argmax_relation_grid",
    "decode_w2ner_logits",
    "pool_subtoken_embeddings",
    "prepare_phobert_word_inputs",
    "resolve_e4_governed_splits",
    "resolve_governed_split_by_sha256",
    "validate_phobert_encoder_load_report",
    "w2ner_relation_loss",
    "write_e4_checkpoint_stub",
]
