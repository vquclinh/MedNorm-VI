"""E4 atomic grid-word alignment and PhoBERT projection (Audit 0045).

Carried forward **verbatim** from the removed implementation, deliberately.

The Audit-0043 gold-grid round-trip put this code through the whole governed
corpus with no model in the loop: 33,826 train and 1,045 validation examples,
13,711 entities, reconstructed at exact precision = recall = F1 = 1.0 with zero
failures. The collapse audited in 0044 was an optimization failure downstream of
here. Rewriting alignment would have discarded the one component measurement
fully vindicated, so it moved unchanged into the clean package.

The two coordinate systems stay decoupled (Audit 0038): the W2NER relation grid
is indexed by ATOMIC ORIGINAL-TEXT words, while PhoBERT consumes VnCoreNLP
segmented model words. Neither surface refines the other, so the projection
between them is overlap-based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ....mention_factory.w2ner import (
    EntitySpan,
    W2NERGrid,
    W2NERLabelVocab,
    WordToken,
    build_w2ner_grid,
    decode_w2ner_grid,
    mask_padded_pairs,
    tokenize_atomic_words,
)
from ...phobert_alignment import SegmentedWord as GovernedSegmentedWord
from .contracts import (
    ATOMIC_FEATURE_DIM,
    ATOMIC_PROJECTION_VERSION,
    E4_INPUT_CONTRACT_VERSION,
    E4ContractError,
)


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
            raise E4ContractError("segmented word has invalid original offsets")
        if self.original_text[self.start:self.end] == "":
            raise E4ContractError("segmented word maps to an empty original slice")

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
            raise E4ContractError("offset_mapping must never be passed to the encoder")
        input_ids = self.model_inputs.get("input_ids")
        if input_ids is None:
            raise E4ContractError("PhoBERT model inputs must include input_ids")
        if len(input_ids) != len(self.word_ids):
            raise E4ContractError("word-id mapping must match encoded input length")
        attention_mask = self.model_inputs.get("attention_mask")
        if attention_mask is not None and len(attention_mask) != len(input_ids):
            raise E4ContractError("attention_mask length must match input_ids")
        for word_index, subtoken_indices in enumerate(self.subword_indices_by_word):
            if not subtoken_indices:
                raise E4ContractError(
                    f"segmented word {word_index} has no PhoBERT subtokens"
                )
            for index in subtoken_indices:
                if index < 0 or index >= len(input_ids):
                    raise E4ContractError("subtoken index outside encoded sequence")
                if self.word_ids[index] != word_index:
                    raise E4ContractError("subtoken index does not map back to its word")
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
            raise E4ContractError("atomic->model index length mismatch")
        if len(self.overlapping_model_word_indices) != count:
            raise E4ContractError("atomic->overlap index length mismatch")
        if len(self.subtoken_indices_by_atomic) != count:
            raise E4ContractError("atomic->subtoken index length mismatch")
        if len(self.atomic_features) != count:
            raise E4ContractError("atomic feature length mismatch")
        for indices in self.subtoken_indices_by_atomic:
            if not indices:
                raise E4ContractError("atomic word has no PhoBERT subtokens")

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
        raise E4ContractError("W2NER document exceeds configured max_words")
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
            raise E4ContractError("segmented word indices must be contiguous")
        if word.original_text != original_text:
            raise E4ContractError("segmented word original_text does not match document")
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
        raise E4ContractError("W2NER segmented contract has no words")
    atomic_words = tokenize_atomic_words(original_text)
    if not atomic_words:
        raise E4ContractError("W2NER atomic contract has no words")
    if len(atomic_words) > max_words:
        raise E4ContractError("W2NER document exceeds configured max_words")
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
        raise E4ContractError("atomic projection requires at least one atomic word")
    if len(encoding.subword_indices_by_word) != len(segmented_words):
        raise E4ContractError(
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
            raise E4ContractError(
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
            raise E4ContractError("PhoBERT encoding must be a single sequence")
        values = values[0]
    return tuple(int(value) for value in values)


def _as_offset_tuple(values: Any) -> tuple[tuple[int, int], ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        if len(values) != 1:
            raise E4ContractError("PhoBERT offset mapping must be a single sequence")
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
    raise E4ContractError("cannot derive special-token mask for PhoBERT encoding")


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
        raise E4ContractError(
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
            raise E4ContractError("encoded subtoken maps to an invalid word index")
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
            raise E4ContractError(
                "PhoBERT tokenizer produced no pieces for a segmented word"
            )
        token_ids = tokenizer.convert_tokens_to_ids(pieces)
        if not isinstance(token_ids, list) or len(token_ids) != len(pieces):
            raise E4ContractError("PhoBERT tokenizer returned mismatched piece ids")
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
            raise E4ContractError(
                "special-token mask has more body positions than token ids"
            )
        word_ids.append(body_word_ids[body_cursor])
        body_cursor += 1
    if body_cursor != len(body_word_ids):
        raise E4ContractError("special-token mask did not consume every body token")
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
        raise E4ContractError("fast tokenizer returned an invalid offset")
    for index, (word_start, word_end) in enumerate(intervals):
        if word_start <= start and end <= word_end:
            return index
    raise E4ContractError(
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
        raise E4ContractError("fast tokenizer did not return offset_mapping")
    offsets = _as_offset_tuple(encoded["offset_mapping"])
    input_ids = _as_int_tuple(encoded["input_ids"])
    attention_mask = _as_int_tuple(encoded.get("attention_mask", [1] * len(input_ids)))
    if len(offsets) != len(input_ids):
        raise E4ContractError("fast tokenizer offsets and input_ids differ in length")
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
        raise E4ContractError("PhoBERT encoding requires at least one segmented word")
    for expected, word in enumerate(segmented_words):
        if word.index != expected:
            raise E4ContractError("segmented word indices must be contiguous")
        if word.original_text[word.start:word.end] != word.original_slice:
            raise E4ContractError("segmented word original-offset invariant failed")
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
        raise E4ContractError("hidden_size must be positive")
    return hidden_size + ATOMIC_FEATURE_DIM
def decode_argmax_relation_grid(
    batch: W2NERBatchContract,
    logits: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[int, ...], ...]:
    """Convert relation logits to a square label grid for deterministic decoding."""
    if len(logits) < batch.word_count:
        raise E4ContractError("relation logits shorter than the word count")
    rows: list[tuple[int, ...]] = []
    for row_index in range(batch.word_count):
        row = logits[row_index]
        if len(row) < batch.word_count:
            raise E4ContractError("relation logits row shorter than the word count")
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


__all__ = [
    "ATOMIC_FEATURE_DIM",
    "ATOMIC_PROJECTION_VERSION",
    "AtomicWordProjection",
    "E4SegmentedWord",
    "PhoBERTWordEncoding",
    "W2NERBatchContract",
    "atomic_relation_head_input_dim",
    "build_atomic_projection",
    "build_w2ner_batch_contract",
    "build_w2ner_batch_contract_from_segmented_words",
    "decode_argmax_relation_grid",
    "decode_w2ner_logits",
    "pool_subtoken_embeddings",
    "prepare_phobert_word_inputs",
    "project_to_atomic_word_embeddings",
]
