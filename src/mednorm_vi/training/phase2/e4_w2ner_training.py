"""Training-contract helpers for the E4 PhoBERT W2NER expert."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...mention_factory.w2ner import (
    ATOMIC_WORD_POLICY_VERSION,
    W2NER_CONFIG_VERSION,
    EntitySpan,
    W2NERGrid,
    W2NERLabelVocab,
    WordToken,
    build_w2ner_grid,
    decode_w2ner_grid,
    mask_padded_pairs,
    tokenize_atomic_words,
)
from ..phobert_alignment import SegmentedWord as GovernedSegmentedWord
from .artifacts import (
    MODE_SMOKE,
    Phase2TrainingManifest,
    checkpoint_payload,
    write_checkpoint_payload,
)
from .common import canonical_json_sha256, sha256_file

E4_STAGE_ID = "phase2-e4-phobert-w2ner-v2"

# Audit 0038 changed the W2NER grid coordinate system from VnCoreNLP segmented
# model words to atomic original-text words. Any checkpoint, resolved config, or
# artifact produced under the Audit-0037 contract describes a different input
# space and MUST NOT be resumed from or validated as compatible.
E4_INPUT_CONTRACT_VERSION = "e4-atomic-grid-word-v1"
E4_SUPERSEDED_INPUT_CONTRACT_VERSIONS = ("e4-segmented-model-word-v1", "")
# Scoped to E4 on purpose: the shared ``CHECKPOINT_SCHEMA_VERSION`` also covers E5
# and L4, whose input contracts did not change, so bumping it would invalidate
# artifacts this milestone did not touch.
E4_CHECKPOINT_SCHEMA_VERSION = "phase2-e4-checkpoint-v2"
E4_SUPERSEDED_CHECKPOINT_SCHEMA_VERSIONS = ("phase2-e4-checkpoint-v1", "")

# Deterministic per-atomic-word features appended to the pooled model-word vector
# so that two atomic words sharing one merged model token are never identical.
ATOMIC_PROJECTION_VERSION = "atomic-projection-v1"
ATOMIC_FEATURE_DIM = 3
E4_MODEL_ID = "vinai/phobert-large"

# ---------------------------------------------------------------------------
# Pretrained weight format (Audit 0039)
# ---------------------------------------------------------------------------
#
# The pinned official `vinai/phobert-large` revision publishes `pytorch_model.bin`
# and does NOT publish `model.safetensors`. Requesting safetensors raises:
#
#     OSError: vinai/phobert-large does not appear to have model.safetensors or
#     model.safetensors.index.json and cannot be loaded with safetensors.
#
# This is purely a serialization-format fact about the official repository. The
# model, architecture, tensor values and revision are identical; loading `.bin`
# is NOT a lower-quality model.
E4_WEIGHT_FORMAT_BIN = "pytorch_model.bin"
E4_WEIGHT_FORMAT_SAFETENSORS = "model.safetensors"
E4_SAFETENSORS_INDEX = "model.safetensors.index.json"

# The revision the real Colab smoke run resolved and trained against.
E4_PINNED_MODEL_REVISION = "1c7880f20db59c0054c6de5afd71b012369f6ee4"

# Revisions known from real execution not to publish safetensors. Listed so the
# resolver reaches the correct answer even when a repository file listing is not
# available (for example offline, or before the hub is queried).
E4_KNOWN_BIN_ONLY_REVISIONS: tuple[str, ...] = (E4_PINNED_MODEL_REVISION,)

# ---------------------------------------------------------------------------
# Mixed precision (Audit 0039)
# ---------------------------------------------------------------------------
PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISION_BF16 = "bf16"
SUPPORTED_PRECISION_MODES: tuple[str, ...] = (PRECISION_FP32, PRECISION_FP16, PRECISION_BF16)

DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

# ---------------------------------------------------------------------------
# Initialization sources
# ---------------------------------------------------------------------------
INITIALIZATION_PINNED_BASE = "pinned_pretrained_base"
INITIALIZATION_PINNED_BASE_SMOKE = "pinned_pretrained_base_bounded_smoke"
INITIALIZATION_FULL_RESUME = "compatible_full_training_checkpoint"

# Fields a full-training checkpoint must match exactly before it may be resumed.
FULL_RESUME_COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "e4_input_contract_version",
    "e4_checkpoint_schema_version",
    "atomic_projection_version",
    "config_sha256",
    "model_revision",
    "tokenizer_revision",
    "pretrained_weight_format",
    "precision_mode",
    "optimizer_signature",
    "accumulation_signature",
)

# State a full-training checkpoint must carry for an exact resume.
REQUIRED_FULL_RESUME_KEYS: tuple[str, ...] = (
    "model_state",
    "optimizer_state",
    "scaler_state",
    "epoch",
    "optimizer_steps",
    "best_metric",
    "best_checkpoint_sha256",
)
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
class AtomicWordProjection:
    """Deterministic PhoBERT subtoken -> model word -> atomic grid word mapping.

    The W2NER grid is indexed by ``atomic_words``; PhoBERT sees ``model words``.
    **Neither surface is a refinement of the other**, which the full-corpus scan
    proved rather than assumed:

    * one model word may cover several atomic words — ``gây_rối`` covers ``gây``
      and ``rối`` (the Audit-0037 failure);
    * one atomic word may span several model words — VnCoreNLP splits at
      letter/digit transitions, so ``beta1`` becomes ``beta`` + ``1`` while the
      atomic surface keeps it whole (31 such cases in the governed corpus).

    The projection is therefore **overlap-based**: an atomic word maps to every
    model word it overlaps, and pools the union of their subtokens. That needs no
    assumption about which tokenizer is finer, so a segmenter change cannot silently
    break it.

    **Pooling rule** (:data:`ATOMIC_PROJECTION_VERSION`), stated explicitly because
    it is a modelling decision and not an implementation detail: an atomic word's
    representation is the mean of the subtoken states of **every model word it
    overlaps**, with three deterministic features appended:

    ``start_ratio``   the atomic word's start, relative to its primary model word
    ``end_ratio``     the atomic word's end, relative to its primary model word
    ``index_ratio``   the atomic word's ordinal among those sharing that primary

    The *primary* model word is the one sharing the most characters with the atomic
    word, ties broken by the earliest index — fully deterministic.

    Atomic words that share a merged model token therefore share contextual content
    but are **not** identical: the appended features separate them, and a test
    asserts that separation. When a model word contains exactly one atomic word
    (the overwhelming majority) the features are ``(0.0, 1.0, 0.0)`` and the
    representation reduces to plain mean pooling.

    The subtoken-level alternative was rejected deliberately: the official slow
    ``PhobertTokenizer`` provides no subtoken character offsets, so slicing a merged
    word's subtokens per atomic word would require reconstructing BPE piece
    boundaries from the ``@@`` continuation marker — undocumented and fragile.
    """

    atomic_words: tuple[WordToken, ...]
    model_word_index_by_atomic: tuple[int, ...]
    overlapping_model_word_indices: tuple[tuple[int, ...], ...]
    atomic_indices_by_model_word: tuple[tuple[int, ...], ...]
    subtoken_indices_by_atomic: tuple[tuple[int, ...], ...]
    atomic_features: tuple[tuple[float, float, float], ...]
    projection_version: str = ATOMIC_PROJECTION_VERSION

    def __post_init__(self) -> None:
        count = len(self.atomic_words)
        if len(self.model_word_index_by_atomic) != count:
            raise E4TrainingContractError("atomic->model index length mismatch")
        if len(self.overlapping_model_word_indices) != count:
            raise E4TrainingContractError("atomic->overlap index length mismatch")
        if len(self.subtoken_indices_by_atomic) != count:
            raise E4TrainingContractError("atomic->subtoken index length mismatch")
        if len(self.atomic_features) != count:
            raise E4TrainingContractError("atomic feature length mismatch")
        for indices in self.subtoken_indices_by_atomic:
            if not indices:
                raise E4TrainingContractError("atomic word has no PhoBERT subtokens")

    @property
    def multi_model_word_atomic_count(self) -> int:
        """Atomic words spanning more than one model word (``beta1`` style)."""
        return sum(1 for group in self.overlapping_model_word_indices if len(group) > 1)

    @property
    def atomic_word_count(self) -> int:
        return len(self.atomic_words)

    @property
    def merged_model_word_count(self) -> int:
        """Model words covering more than one atomic word."""
        return sum(1 for group in self.atomic_indices_by_model_word if len(group) > 1)


@dataclass(frozen=True, slots=True)
class W2NERBatchContract:
    grid: W2NERGrid
    padded_labels: tuple[tuple[int, ...], ...]
    padded_pair_mask: tuple[tuple[bool, ...], ...]
    label_count: int
    segmented_words: tuple[E4SegmentedWord, ...] = ()
    atomic_words: tuple[WordToken, ...] = ()
    input_contract_version: str = E4_INPUT_CONTRACT_VERSION

    @property
    def word_count(self) -> int:
        return len(self.grid.words)


def _pad_grid(
    grid: W2NERGrid,
    max_words: int,
    *,
    segmented_words: tuple[E4SegmentedWord, ...] = (),
) -> W2NERBatchContract:
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
    if not segmented_words:
        # No segmenter was supplied: each atomic grid word is its own model word.
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
        atomic_words=grid.words,
    )


def build_w2ner_batch_contract(
    document_id: str,
    original_text: str,
    entities: Sequence[EntitySpan],
    *,
    max_words: int,
    vocab: W2NERLabelVocab | None = None,
) -> W2NERBatchContract:
    """Build the grid over atomic original-text words with no segmenter available."""
    grid = build_w2ner_grid(
        document_id,
        original_text,
        entities,
        words=tokenize_atomic_words(original_text),
        vocab=vocab,
    )
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
    """Build the W2NER grid over ATOMIC original-text words (Audit 0038).

    The VnCoreNLP segmented words are retained verbatim — including their
    ``model_text`` join characters — because PhoBERT consumes them. They are **not**
    the grid coordinate system: the segmenter may merge syllables across a gold
    entity boundary (``gây_rối`` versus a gold span starting at ``rối``), which made
    a correct governed entity unrepresentable under the Audit-0037 contract.

    Gold entities are used only to create labels. They never influence how atomic
    words are built.
    """
    e4_words = tuple(
        _coerce_segmented_word(original_text=original_text, index=index, word=word)
        for index, word in enumerate(segmented_words)
    )
    if not e4_words:
        raise E4TrainingContractError("W2NER segmented contract has no words")
    atomic_words = tokenize_atomic_words(original_text)
    if not atomic_words:
        raise E4TrainingContractError("W2NER atomic contract has no words")
    if len(atomic_words) > max_words:
        raise E4TrainingContractError("W2NER document exceeds configured max_words")
    grid = build_w2ner_grid(
        document_id,
        original_text,
        entities,
        words=atomic_words,
        vocab=vocab,
    )
    return _pad_grid(grid, max_words, segmented_words=e4_words)


def build_atomic_projection(
    original_text: str,
    segmented_words: Sequence[E4SegmentedWord],
    encoding: PhoBERTWordEncoding,
    *,
    atomic_words: Sequence[WordToken] | None = None,
) -> AtomicWordProjection:
    """Project PhoBERT subtokens through model words onto atomic grid words.

    Every atomic word must fall inside exactly one segmented model word; anything
    else means the two surfaces disagree about the original text and is a loud
    failure rather than a silent re-association.
    """
    words = tuple(atomic_words) if atomic_words is not None else tokenize_atomic_words(
        original_text)
    if not words:
        raise E4TrainingContractError("atomic projection requires at least one atomic word")
    if len(encoding.subword_indices_by_word) != len(segmented_words):
        raise E4TrainingContractError(
            "encoding subtoken buckets do not match the segmented word count")

    primary_index: list[int] = []
    overlapping_indices: list[tuple[int, ...]] = []
    atomic_by_model: list[list[int]] = [[] for _ in segmented_words]
    subtokens_by_atomic: list[tuple[int, ...]] = []
    raw_features: list[tuple[float, float]] = []

    for atomic_index, word in enumerate(words):
        overlaps = [
            model_index
            for model_index, model_word in enumerate(segmented_words)
            if model_word.start < word.end and word.start < model_word.end
        ]
        if not overlaps:
            raise E4TrainingContractError(
                f"atomic word {word.index} {word.start}:{word.end} overlaps no "
                "VnCoreNLP segmented model word")
        # Primary owner: the model word sharing the most characters with this
        # atomic word; ties resolve to the earliest, so the choice is deterministic.
        primary = min(
            overlaps,
            key=lambda model_index: (
                -(min(segmented_words[model_index].end, word.end)
                  - max(segmented_words[model_index].start, word.start)),
                model_index))
        model_word = segmented_words[primary]
        primary_index.append(primary)
        overlapping_indices.append(tuple(overlaps))
        atomic_by_model[primary].append(atomic_index)
        subtokens: list[int] = []
        for model_index in overlaps:
            subtokens.extend(encoding.subword_indices_by_word[model_index])
        subtokens_by_atomic.append(tuple(sorted(set(subtokens))))
        span = max(1, model_word.end - model_word.start)
        raw_features.append((
            round(min(1.0, max(0.0, (word.start - model_word.start) / span)), 6),
            round(min(1.0, max(0.0, (word.end - model_word.start) / span)), 6),
        ))

    features: list[tuple[float, float, float]] = []
    for atomic_index, (start_ratio, end_ratio) in enumerate(raw_features):
        group = atomic_by_model[primary_index[atomic_index]]
        ordinal = group.index(atomic_index)
        denominator = max(1, len(group) - 1)
        features.append((start_ratio, end_ratio, round(ordinal / denominator, 6)))

    return AtomicWordProjection(
        atomic_words=words,
        model_word_index_by_atomic=tuple(primary_index),
        overlapping_model_word_indices=tuple(overlapping_indices),
        atomic_indices_by_model_word=tuple(tuple(group) for group in atomic_by_model),
        subtoken_indices_by_atomic=tuple(subtokens_by_atomic),
        atomic_features=tuple(features),
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
    """Mean-pool encoder states into one vector per VnCoreNLP model word."""
    import torch

    pooled = []
    for indices in encoding.subword_indices_by_word:
        pooled.append(sequence_output[list(indices)].mean(dim=0))
    return torch.stack(pooled).unsqueeze(0)


def project_to_atomic_word_embeddings(
    sequence_output: Any, projection: AtomicWordProjection,
) -> Any:
    """One representation per ATOMIC grid word: ``[1, atomic_words, hidden + 3]``.

    Implements :data:`ATOMIC_PROJECTION_VERSION` exactly as documented on
    :class:`AtomicWordProjection`: the model word's mean-pooled subtoken states,
    concatenated with the atomic word's three deterministic position features. Two
    atomic words under one merged model token therefore differ in the final
    :data:`ATOMIC_FEATURE_DIM` dimensions and are never indistinguishable.
    """
    import torch

    pooled = []
    for atomic_index, indices in enumerate(projection.subtoken_indices_by_atomic):
        context = sequence_output[list(indices)].mean(dim=0)
        extras = torch.tensor(
            projection.atomic_features[atomic_index],
            dtype=context.dtype, device=context.device)
        pooled.append(torch.cat((context, extras), dim=-1))
    return torch.stack(pooled).unsqueeze(0)


def atomic_relation_head_input_dim(hidden_size: int) -> int:
    """Hidden size the relation-grid head must be built with under v2."""
    if hidden_size <= 0:
        raise E4TrainingContractError("hidden_size must be positive")
    return hidden_size + ATOMIC_FEATURE_DIM


# ---------------------------------------------------------------------------
# Pretrained weight format
# ---------------------------------------------------------------------------


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
    """Decide the weight format for a pinned revision, deterministically.

    When ``repository_files`` is supplied the decision is a plain file inspection.
    Without it, a revision known from real execution to be ``.bin``-only resolves
    to ``.bin``; anything else is also resolved to ``.bin``, because that is the
    format the official PhoBERT repository publishes and requesting safetensors is
    what broke the real Colab run.

    The resolution happens **before** training starts and does not change
    afterwards: there is no mid-run fallback across unrelated files.
    """
    if repository_files is not None:
        listing = {str(name) for name in repository_files}
        if E4_WEIGHT_FORMAT_SAFETENSORS in listing or E4_SAFETENSORS_INDEX in listing:
            return PhoBERTWeightFormat(
                filename=E4_WEIGHT_FORMAT_SAFETENSORS, use_safetensors=True,
                model_revision=model_revision, resolved_by="repository_file_listing")
        if E4_WEIGHT_FORMAT_BIN not in listing:
            raise E4TrainingContractError(
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
        raise E4TrainingContractError(
            f"{E4_MODEL_ID}@{weight_format.model_revision} does not publish "
            f"{E4_WEIGHT_FORMAT_SAFETENSORS}; use_safetensors must be False")
    if weight_format.use_safetensors != (
            weight_format.filename == E4_WEIGHT_FORMAT_SAFETENSORS):
        raise E4TrainingContractError(
            "use_safetensors disagrees with the resolved weight-format filename")


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AccumulationPlan:
    """Real optimizer-step accounting for one training run.

    Audit 0037/0038's loop called ``optimizer.step()`` after **every document**
    while the notebook declared ``EFFECTIVE_BATCH_SIZE = 8``, so the effective
    batch was one document and the recorded optimizer-step count was the document
    count. Everything below is derived arithmetic, never a relabelled constant.
    """

    example_count: int
    micro_batch_size: int
    accumulation_steps: int
    epochs: int

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    @property
    def micro_batches_per_epoch(self) -> int:
        return _ceil_div(self.example_count, self.micro_batch_size)

    @property
    def optimizer_steps_per_epoch(self) -> int:
        """Ceiling division: the trailing partial group still takes one step."""
        return _ceil_div(self.micro_batches_per_epoch, self.accumulation_steps)

    @property
    def expected_optimizer_steps(self) -> int:
        return self.optimizer_steps_per_epoch * self.epochs

    @property
    def expected_backward_passes(self) -> int:
        return self.micro_batches_per_epoch * self.epochs

    @property
    def final_partial_group_size(self) -> int:
        """Micro-batches in the last group; equals ``accumulation_steps`` if exact."""
        remainder = self.micro_batches_per_epoch % self.accumulation_steps
        return remainder or self.accumulation_steps

    @property
    def has_partial_final_group(self) -> bool:
        return self.micro_batches_per_epoch % self.accumulation_steps != 0

    def group_size_for(self, micro_batch_index: int) -> int:
        """Number of micro-batches in the accumulation group containing this index.

        The final group may be smaller, and the loss for every micro-batch in it
        must be divided by this size rather than by ``accumulation_steps`` —
        otherwise the last group's gradient is silently under-scaled.
        """
        if micro_batch_index < 0 or micro_batch_index >= self.micro_batches_per_epoch:
            raise E4TrainingContractError("micro-batch index outside the epoch")
        group_index = micro_batch_index // self.accumulation_steps
        remaining = self.micro_batches_per_epoch - group_index * self.accumulation_steps
        return min(self.accumulation_steps, remaining)

    def loss_scale_for(self, micro_batch_index: int) -> float:
        return 1.0 / self.group_size_for(micro_batch_index)

    def is_optimizer_step_boundary(self, micro_batch_index: int) -> bool:
        """True on the last micro-batch of its accumulation group."""
        if micro_batch_index < 0 or micro_batch_index >= self.micro_batches_per_epoch:
            raise E4TrainingContractError("micro-batch index outside the epoch")
        is_group_end = (micro_batch_index + 1) % self.accumulation_steps == 0
        is_epoch_end = micro_batch_index + 1 == self.micro_batches_per_epoch
        return is_group_end or is_epoch_end

    @property
    def signature(self) -> str:
        """Compact identity used for resume compatibility."""
        return (f"micro{self.micro_batch_size}"
                f"-accum{self.accumulation_steps}"
                f"-effective{self.effective_batch_size}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "micro_batch_size": self.micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "micro_batches_per_epoch": self.micro_batches_per_epoch,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "expected_optimizer_steps": self.expected_optimizer_steps,
            "expected_backward_passes": self.expected_backward_passes,
            "final_partial_group_size": self.final_partial_group_size,
            "has_partial_final_group": self.has_partial_final_group,
            "accumulation_signature": self.signature,
        }


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise E4TrainingContractError("denominator must be positive")
    return -(-numerator // denominator)


def plan_gradient_accumulation(
    example_count: int, *, micro_batch_size: int, accumulation_steps: int, epochs: int,
) -> AccumulationPlan:
    if example_count <= 0:
        raise E4TrainingContractError("example_count must be positive")
    if micro_batch_size <= 0:
        raise E4TrainingContractError("micro_batch_size must be positive")
    if accumulation_steps <= 0:
        raise E4TrainingContractError("accumulation_steps must be positive")
    if epochs <= 0:
        raise E4TrainingContractError("epochs must be positive")
    return AccumulationPlan(
        example_count=example_count, micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps, epochs=epochs)


def assert_optimizer_step_accounting(plan: AccumulationPlan, observed_steps: int) -> None:
    """Fail when recorded optimizer steps do not match real ``optimizer.step()`` calls."""
    if observed_steps != plan.expected_optimizer_steps:
        raise E4TrainingContractError(
            f"optimizer step accounting mismatch: expected "
            f"{plan.expected_optimizer_steps}, observed {observed_steps}")


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixedPrecisionPolicy:
    """Tracked autocast/GradScaler policy for one run."""

    mode: str
    device_type: str
    autocast_enabled: bool
    use_grad_scaler: bool

    @property
    def autocast_dtype_name(self) -> str:
        if not self.autocast_enabled:
            return ""
        return "torch.bfloat16" if self.mode == PRECISION_BF16 else "torch.float16"

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision_mode": self.mode,
            "precision_device_type": self.device_type,
            "autocast_enabled": self.autocast_enabled,
            "autocast_dtype": self.autocast_dtype_name,
            "use_grad_scaler": self.use_grad_scaler,
        }


def resolve_mixed_precision_policy(
    requested_mode: str, *, device_type: str, bf16_supported: bool = False,
) -> MixedPrecisionPolicy:
    """Resolve a safe precision policy for the runtime actually present.

    * CUDA + ``bf16`` needs no loss scaling, so no ``GradScaler``;
    * CUDA + ``fp16`` requires a ``GradScaler``;
    * CUDA + ``bf16`` requested on hardware without bf16 support degrades to
      ``fp16`` with a scaler rather than silently running an unsupported dtype;
    * CPU always resolves to fp32 — autocast on CPU is not used for E4, and full
      training on CPU is refused outright by
      :func:`assert_full_training_device`.
    """
    if requested_mode not in SUPPORTED_PRECISION_MODES:
        raise E4TrainingContractError(
            f"unsupported precision mode {requested_mode!r}; "
            f"expected one of {SUPPORTED_PRECISION_MODES}")
    if device_type != DEVICE_CUDA:
        return MixedPrecisionPolicy(
            mode=PRECISION_FP32, device_type=device_type,
            autocast_enabled=False, use_grad_scaler=False)
    if requested_mode == PRECISION_FP32:
        return MixedPrecisionPolicy(
            mode=PRECISION_FP32, device_type=DEVICE_CUDA,
            autocast_enabled=False, use_grad_scaler=False)
    if requested_mode == PRECISION_BF16 and bf16_supported:
        return MixedPrecisionPolicy(
            mode=PRECISION_BF16, device_type=DEVICE_CUDA,
            autocast_enabled=True, use_grad_scaler=False)
    return MixedPrecisionPolicy(
        mode=PRECISION_FP16, device_type=DEVICE_CUDA,
        autocast_enabled=True, use_grad_scaler=True)


def assert_full_training_device(device_type: str) -> None:
    """Full training must not run on CPU; bounded tests and smoke may."""
    if device_type != DEVICE_CUDA:
        raise E4TrainingContractError(
            "E4 full training requires a CUDA device; the CPU path is valid only "
            "for bounded smoke and local contract tests")


def optimizer_signature(*, name: str, learning_rate: float, weight_decay: float,
                        max_grad_norm: float) -> str:
    """Compact optimizer identity used for resume compatibility."""
    return f"{name}-lr{learning_rate:g}-wd{weight_decay:g}-clip{max_grad_norm:g}"


# ---------------------------------------------------------------------------
# Initialization and resume
# ---------------------------------------------------------------------------


def assert_full_initialization_source(
    *, run_full_training: bool, resume_from_smoke_checkpoint: bool,
    resume_from_full_checkpoint: bool, checkpoint_mode: str = "",
) -> str:
    """Return the initialization source for a full run, or fail loudly.

    A smoke checkpoint is bounded execution evidence, never an initializer.
    """
    if not run_full_training:
        return INITIALIZATION_PINNED_BASE_SMOKE
    if resume_from_smoke_checkpoint:
        raise E4TrainingContractError(
            "E4 full training may not resume from or initialize with a smoke checkpoint")
    if resume_from_full_checkpoint:
        if checkpoint_mode and checkpoint_mode != "full":
            raise E4TrainingContractError(
                f"full resume requires a full-training checkpoint, got mode "
                f"{checkpoint_mode!r}")
        return INITIALIZATION_FULL_RESUME
    return INITIALIZATION_PINNED_BASE


def assert_full_checkpoint_custody(payload: Mapping[str, Any]) -> None:
    """A full checkpoint must carry everything an exact resume needs."""
    missing = tuple(key for key in REQUIRED_FULL_RESUME_KEYS if key not in payload)
    if missing:
        raise E4TrainingContractError(
            "E4 full checkpoint is missing resume state: " + ", ".join(sorted(missing)))
    if not isinstance(payload.get("model_state"), Mapping):
        raise E4TrainingContractError("E4 checkpoint model_state must be a mapping")
    for key in ("base_model", "w2ner_head"):
        if key not in payload["model_state"]:
            raise E4TrainingContractError(f"E4 checkpoint model_state lacks {key!r}")


def assert_compatible_full_resume(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any],
) -> None:
    """Refuse a resume whose training contract differs in any tracked field.

    Precision and accumulation are included on purpose: silently changing either
    mid-run makes the recorded optimizer-step accounting meaningless.
    """
    reject_incompatible_e4_checkpoint(payload)
    assert_full_checkpoint_custody(payload)
    if str(payload.get("mode", "")) != "full":
        raise E4TrainingContractError(
            "only a full-training checkpoint may initialize a full-training resume")
    differences = [
        f"{field}: checkpoint={payload.get(field)!r} expected={expected.get(field)!r}"
        for field in FULL_RESUME_COMPATIBILITY_FIELDS
        if str(payload.get(field, "")) != str(expected.get(field, ""))
    ]
    if differences:
        raise E4TrainingContractError(
            "E4 full resume is incompatible: " + "; ".join(differences))


def reject_incompatible_e4_checkpoint(payload: Mapping[str, Any]) -> None:
    """Refuse a checkpoint or artifact built under the Audit-0037 input contract.

    The W2NER grid coordinate system changed, so an Audit-0037 checkpoint describes
    a different input space entirely. Resuming from one would silently train a head
    whose word indices mean something else.
    """
    contract = str(payload.get("e4_input_contract_version", ""))
    if contract != E4_INPUT_CONTRACT_VERSION:
        raise E4TrainingContractError(
            f"E4 checkpoint input contract {contract!r} is incompatible with "
            f"{E4_INPUT_CONTRACT_VERSION!r}; the W2NER grid word surface changed in "
            "Audit 0038 and an Audit-0037 artifact may not be resumed or validated")
    schema = str(payload.get("e4_checkpoint_schema_version", ""))
    if schema != E4_CHECKPOINT_SCHEMA_VERSION:
        raise E4TrainingContractError(
            f"E4 checkpoint schema {schema!r} is incompatible with "
            f"{E4_CHECKPOINT_SCHEMA_VERSION!r}")
    projection = str(payload.get("atomic_projection_version", ""))
    if projection != ATOMIC_PROJECTION_VERSION:
        raise E4TrainingContractError(
            f"E4 atomic projection {projection!r} is incompatible with "
            f"{ATOMIC_PROJECTION_VERSION!r}")


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
    weight_format: PhoBERTWeightFormat | None = None,
    accumulation: AccumulationPlan | None = None,
    precision: MixedPrecisionPolicy | None = None,
    progress: Mapping[str, Any] | None = None,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    optimizer_name: str = "AdamW",
) -> dict[str, Any]:
    """Resolved config. Every field here changes the config hash by design."""
    resolved_weight_format = weight_format or resolve_phobert_weight_format(
        E4_MODEL_ID, model_revision)
    assert_weight_format_loadable(resolved_weight_format)
    config: dict[str, Any] = {
        "stage_id": E4_STAGE_ID,
        "expert_id": "E4_phobert_w2ner",
        "mode": mode,
        "model_id": E4_MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "seed": seed,
        "max_words": max_words,
        "effective_batch_size": (
            accumulation.effective_batch_size if accumulation is not None
            else effective_batch_size),
        "label_space": list(W2NERLabelVocab().type_order),
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "optimizer_signature": optimizer_signature(
            name=optimizer_name, learning_rate=learning_rate,
            weight_decay=weight_decay, max_grad_norm=max_grad_norm),
        # PhoBERT is fully fine-tuned together with the W2NER head; nothing is
        # frozen and no quantization is applied (Audit 0039).
        "freeze_base_model": False,
        "quantization": "none",
        # Audit-0038 contract identity. These fields change the resolved-config
        # hash, so an Audit-0037 artifact can never validate against a v2 run.
        "e4_input_contract_version": E4_INPUT_CONTRACT_VERSION,
        "e4_checkpoint_schema_version": E4_CHECKPOINT_SCHEMA_VERSION,
        "grid_word_surface": ATOMIC_WORD_POLICY_VERSION,
        "atomic_projection_version": ATOMIC_PROJECTION_VERSION,
        "atomic_feature_dim": ATOMIC_FEATURE_DIM,
        "w2ner_config_version": W2NER_CONFIG_VERSION,
        "internal_test_accessed": False,
    }
    config.update(resolved_weight_format.as_dict())
    if accumulation is not None:
        config.update(accumulation.as_dict())
    if precision is not None:
        config.update(precision.as_dict())
    if progress is not None:
        # Observability-only settings (Audit 0041). They are recorded here as the
        # milestone requires, which does change the resolved-config hash; resume
        # SEMANTICS are unchanged, and no completed full checkpoint exists to
        # invalidate.
        config.update(dict(progress))
    return config


def build_e4_training_state_payload(
    *,
    mode: str,
    config_sha256: str,
    model_revision: str,
    tokenizer_revision: str,
    parameter_count: int,
    weight_format: PhoBERTWeightFormat,
    accumulation: AccumulationPlan,
    precision: MixedPrecisionPolicy,
    optimizer_signature_value: str,
    epoch: int,
    optimizer_steps: int,
    backward_passes: int,
    examples_processed: int,
    best_metric: float,
    best_checkpoint_sha256: str = "",
    model_state: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    scaler_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A checkpoint payload carrying everything an exact full resume needs.

    Audit 0038's payload held model weights only, so a full run could not truly be
    resumed: optimizer moments, the GradScaler scale, the step count and the best
    metric were all lost. ``scheduler_state`` stays ``None`` unless a scheduler is
    actually part of the tracked configuration — no arbitrary scheduler is added.
    """
    payload = e4_checkpoint_payload(
        mode=mode,
        config_sha256=config_sha256,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        parameter_count=parameter_count,
    )
    payload.update(weight_format.as_dict())
    payload.update(precision.as_dict())
    payload["accumulation_signature"] = accumulation.signature
    payload["accumulation"] = accumulation.as_dict()
    payload["optimizer_signature"] = optimizer_signature_value
    payload["epoch"] = int(epoch)
    payload["optimizer_steps"] = int(optimizer_steps)
    payload["backward_passes"] = int(backward_passes)
    payload["examples_processed"] = int(examples_processed)
    payload["best_metric"] = float(best_metric)
    payload["best_checkpoint_sha256"] = best_checkpoint_sha256
    payload["model_state"] = dict(model_state or {})
    payload["optimizer_state"] = dict(optimizer_state or {})
    payload["scaler_state"] = dict(scaler_state or {})
    payload["scheduler_state"] = dict(scheduler_state or {})
    payload["scheduler_configured"] = scheduler_state is not None
    return payload


def build_e4_history_row(
    *,
    epoch: int,
    mode: str,
    train_loss: float,
    validation_metrics: Mapping[str, Any],
    optimizer_steps: int,
    backward_passes: int,
    examples_processed: int,
    learning_rate: float,
    accumulation: AccumulationPlan,
    precision: MixedPrecisionPolicy,
) -> dict[str, Any]:
    """One governed per-epoch history record (spec §18.1 error/decision trail)."""
    return {
        "epoch": int(epoch),
        "mode": mode,
        "train_loss": float(train_loss),
        "validation_exact_precision": float(
            validation_metrics.get("validation_exact_precision", 0.0)),
        "validation_exact_recall": float(
            validation_metrics.get("validation_exact_recall", 0.0)),
        "validation_exact_f1": float(validation_metrics.get("validation_exact_f1", 0.0)),
        "optimizer_steps": int(optimizer_steps),
        "backward_passes": int(backward_passes),
        "completed_examples": int(examples_processed),
        "learning_rate": float(learning_rate),
        "micro_batch_size": accumulation.micro_batch_size,
        "accumulation_steps": accumulation.accumulation_steps,
        "effective_batch_size": accumulation.effective_batch_size,
        "precision_mode": precision.mode,
        "internal_test_accessed": False,
    }


def build_e4_training_accounting(
    *,
    accumulation: AccumulationPlan,
    precision: MixedPrecisionPolicy,
    weight_format: PhoBERTWeightFormat,
    observed_optimizer_steps: int,
    observed_backward_passes: int,
    observed_examples: int,
    max_grad_norm: float,
    gradient_clipping_enabled: bool = True,
) -> dict[str, Any]:
    """Manifest accounting that must match what the loop really did."""
    assert_optimizer_step_accounting(accumulation, observed_optimizer_steps)
    if observed_backward_passes != accumulation.expected_backward_passes:
        raise E4TrainingContractError(
            f"backward-pass accounting mismatch: expected "
            f"{accumulation.expected_backward_passes}, observed {observed_backward_passes}")
    return {
        **accumulation.as_dict(),
        **precision.as_dict(),
        **weight_format.as_dict(),
        "observed_optimizer_steps": int(observed_optimizer_steps),
        "observed_backward_passes": int(observed_backward_passes),
        "examples_processed": int(observed_examples),
        "gradient_clipping_enabled": bool(gradient_clipping_enabled),
        "max_grad_norm": float(max_grad_norm),
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
    payload = e4_checkpoint_payload(
        mode=mode,
        config_sha256=config_sha256,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        parameter_count=parameter_count,
    )
    write_checkpoint_payload(path, payload)
    return sha256_file(path)


def e4_checkpoint_payload(
    *,
    mode: str,
    config_sha256: str,
    model_revision: str,
    tokenizer_revision: str,
    parameter_count: int,
) -> dict[str, Any]:
    """Shared Phase-2 payload plus the Audit-0038 E4 contract identity.

    ``reject_incompatible_e4_checkpoint`` reads these fields, so every checkpoint
    written by the v2 path is self-describing and an Audit-0037 payload (which has
    none of them) is refused rather than silently resumed.
    """
    payload = checkpoint_payload(
        expert_id="E4_phobert_w2ner",
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
    return payload


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
    training_accounting: Mapping[str, Any] | None = None,
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
        training_accounting=dict(training_accounting or {}),
    )


__all__ = [
    "ATOMIC_FEATURE_DIM",
    "ATOMIC_PROJECTION_VERSION",
    "AccumulationPlan",
    "AtomicWordProjection",
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "E4_CHECKPOINT_SCHEMA_VERSION",
    "E4_INPUT_CONTRACT_VERSION",
    "E4_KNOWN_BIN_ONLY_REVISIONS",
    "E4_PINNED_MODEL_REVISION",
    "E4_SUPERSEDED_CHECKPOINT_SCHEMA_VERSIONS",
    "E4_SUPERSEDED_INPUT_CONTRACT_VERSIONS",
    "E4_WEIGHT_FORMAT_BIN",
    "E4_WEIGHT_FORMAT_SAFETENSORS",
    "FULL_RESUME_COMPATIBILITY_FIELDS",
    "INITIALIZATION_FULL_RESUME",
    "INITIALIZATION_PINNED_BASE",
    "INITIALIZATION_PINNED_BASE_SMOKE",
    "MixedPrecisionPolicy",
    "PRECISION_BF16",
    "PRECISION_FP16",
    "PRECISION_FP32",
    "PhoBERTWeightFormat",
    "REQUIRED_FULL_RESUME_KEYS",
    "SUPPORTED_PRECISION_MODES",
    "assert_compatible_full_resume",
    "assert_full_checkpoint_custody",
    "assert_full_initialization_source",
    "assert_full_training_device",
    "assert_optimizer_step_accounting",
    "assert_weight_format_loadable",
    "build_e4_history_row",
    "build_e4_training_accounting",
    "build_e4_training_state_payload",
    "optimizer_signature",
    "plan_gradient_accumulation",
    "resolve_mixed_precision_policy",
    "resolve_phobert_weight_format",
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
    "atomic_relation_head_input_dim",
    "build_atomic_projection",
    "build_w2ner_batch_contract",
    "build_w2ner_batch_contract_from_segmented_words",
    "e4_checkpoint_payload",
    "decode_argmax_relation_grid",
    "decode_w2ner_logits",
    "pool_subtoken_embeddings",
    "prepare_phobert_word_inputs",
    "project_to_atomic_word_embeddings",
    "reject_incompatible_e4_checkpoint",
    "resolve_e4_governed_splits",
    "resolve_governed_split_by_sha256",
    "validate_phobert_encoder_load_report",
    "w2ner_relation_loss",
    "write_e4_checkpoint_stub",
]
