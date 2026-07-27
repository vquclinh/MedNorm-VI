"""E4 PhoBERT W2NER proposal path.

This module implements the architecture-level W2NER data contract without
downloading or training PhoBERT locally. The pure-Python pieces build governed
word grids and decode relation grids back to exact original offsets; the optional
model-head factory imports torch lazily only when a training or Colab runtime
asks for it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..lattice.models import EXPERT_PHOBERT_W2NER, ExpertSpanProposal
from ..schemas.constants import ENTITY_TYPES

W2NER_CONFIG_VERSION = "phobert-w2ner-v1"
W2NER_NONE = "NONE"
W2NER_NEXT_NEIGHBORING_WORD = "NNW"
W2NER_TAIL_HEAD_WORD_PREFIX = "THW:"
DEFAULT_TYPE_ORDER: tuple[str, ...] = (
    "DIAGNOSIS",
    "MEDICATION",
    "SYMPTOM",
    "TEST_NAME",
    "TEST_RESULT",
)


class W2NERError(ValueError):
    """Raised when W2NER conversion or decoding is not reversible."""


class W2NERUnavailableError(RuntimeError):
    """Raised when a disabled or untrained W2NER checkpoint is requested."""


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """Gold or decoded contiguous entity span in original character offsets."""

    start: int
    end: int
    entity_type: str
    text: str

    def validate_against(self, original_text: str) -> None:
        if self.entity_type not in ENTITY_TYPES:
            raise W2NERError(f"unsupported entity type {self.entity_type!r}")
        if self.end <= self.start:
            raise W2NERError(f"invalid entity offsets {self.start}:{self.end}")
        if original_text[self.start:self.end] != self.text:
            raise W2NERError("entity text is not an exact original_text slice")


@dataclass(frozen=True, slots=True)
class WordToken:
    index: int
    text: str
    start: int
    end: int

    def validate_against(self, original_text: str) -> None:
        if self.end <= self.start:
            raise W2NERError(f"invalid word offsets {self.start}:{self.end}")
        if original_text[self.start:self.end] != self.text:
            raise W2NERError("word text is not an exact original_text slice")


@dataclass(frozen=True, slots=True)
class WordSubwordAlignment:
    """Mapping from one whitespace word to its encoded subword indices."""

    word_index: int
    subword_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class W2NERLabelVocab:
    type_order: tuple[str, ...] = DEFAULT_TYPE_ORDER

    def __post_init__(self) -> None:
        unknown = set(self.type_order) - set(ENTITY_TYPES)
        if unknown:
            raise W2NERError(f"unknown W2NER entity types: {sorted(unknown)}")

    @property
    def labels(self) -> tuple[str, ...]:
        return (W2NER_NONE, W2NER_NEXT_NEIGHBORING_WORD) + tuple(
            f"{W2NER_TAIL_HEAD_WORD_PREFIX}{entity_type}" for entity_type in self.type_order
        )

    @property
    def none_id(self) -> int:
        return 0

    @property
    def nnw_id(self) -> int:
        return 1

    def thw_id(self, entity_type: str) -> int:
        if entity_type not in self.type_order:
            raise W2NERError(f"unknown THW type {entity_type!r}")
        return self.labels.index(f"{W2NER_TAIL_HEAD_WORD_PREFIX}{entity_type}")

    def thw_type(self, label_id: int) -> str | None:
        if label_id < 0 or label_id >= len(self.labels):
            raise W2NERError(f"relation label id {label_id} is outside the vocab")
        label = self.labels[label_id]
        if not label.startswith(W2NER_TAIL_HEAD_WORD_PREFIX):
            return None
        return label.removeprefix(W2NER_TAIL_HEAD_WORD_PREFIX)


@dataclass(frozen=True, slots=True)
class W2NERGrid:
    """Square word-word relation grid and valid-pair mask."""

    document_id: str
    original_text: str
    words: tuple[WordToken, ...]
    labels: tuple[tuple[int, ...], ...]
    pair_mask: tuple[tuple[bool, ...], ...]
    vocab: W2NERLabelVocab = field(default_factory=W2NERLabelVocab)

    def __post_init__(self) -> None:
        size = len(self.words)
        if len(self.labels) != size or len(self.pair_mask) != size:
            raise W2NERError("W2NER grid rows must match the word count")
        for row in self.labels:
            if len(row) != size:
                raise W2NERError("W2NER label grid must be square")
        for row in self.pair_mask:
            if len(row) != size:
                raise W2NERError("W2NER pair mask must be square")
        for word in self.words:
            word.validate_against(self.original_text)

    def config_hash(self) -> str:
        payload = {
            "version": W2NER_CONFIG_VERSION,
            "labels": self.vocab.labels,
            "word_count": len(self.words),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class W2NERDecodedSpan:
    start: int
    end: int
    text: str
    entity_type: str
    score: float
    word_start: int
    word_end: int

    def validate_against(self, original_text: str) -> None:
        if original_text[self.start:self.end] != self.text:
            raise W2NERError("decoded W2NER span is not reversible to original text")


@dataclass(frozen=True, slots=True)
class PhoBERTW2NERConfig:
    base_model_name: str = "vinai/phobert-large"
    model_revision: str = ""
    checkpoint_path: str = ""
    expected_checkpoint_sha256: str = ""
    max_words: int = 256
    threshold: float = 0.5
    enabled: bool = False
    config_version: str = W2NER_CONFIG_VERSION


def tokenize_words(original_text: str) -> tuple[WordToken, ...]:
    """Whitespace-preserving word scan over original character offsets."""
    words: list[WordToken] = []
    for index, match in enumerate(re.finditer(r"\S+", original_text)):
        token = WordToken(index, match.group(0), match.start(), match.end())
        token.validate_against(original_text)
        words.append(token)
    return tuple(words)


def align_words_to_subwords(
    words: Sequence[WordToken],
    subword_offsets: Sequence[tuple[int, int]],
    original_text: str,
) -> tuple[WordSubwordAlignment, ...]:
    """Assign subword offsets to containing words, validating reversibility."""
    alignments: list[WordSubwordAlignment] = []
    for word in words:
        indices: list[int] = []
        for subword_index, (start, end) in enumerate(subword_offsets):
            if start == end == 0:
                continue
            if end <= start or original_text[start:end] == "":
                raise W2NERError(f"invalid subword offset {start}:{end}")
            if word.start <= start and end <= word.end:
                indices.append(subword_index)
        alignments.append(WordSubwordAlignment(word.index, tuple(indices)))
    return tuple(alignments)


def _word_bounds_for_entity(words: Sequence[WordToken], entity: EntitySpan) -> tuple[int, int]:
    start_word = next((word.index for word in words if word.start == entity.start), None)
    end_word = next((word.index for word in words if word.end == entity.end), None)
    if start_word is None or end_word is None or end_word < start_word:
        raise W2NERError(
            f"entity {entity.entity_type} {entity.start}:{entity.end} is not word-aligned"
        )
    return start_word, end_word


def build_w2ner_grid(
    document_id: str,
    original_text: str,
    entities: Sequence[EntitySpan],
    *,
    words: Sequence[WordToken] | None = None,
    vocab: W2NERLabelVocab | None = None,
) -> W2NERGrid:
    """Create W2NER NNW/THW labels for contiguous governed spans."""
    word_tokens = tuple(words) if words is not None else tokenize_words(original_text)
    label_vocab = vocab or W2NERLabelVocab()
    size = len(word_tokens)
    labels = [[label_vocab.none_id for _ in range(size)] for _ in range(size)]
    pair_mask = [[True for _ in range(size)] for _ in range(size)]
    for entity in entities:
        entity.validate_against(original_text)
        start_word, end_word = _word_bounds_for_entity(word_tokens, entity)
        for word_index in range(start_word, end_word):
            labels[word_index][word_index + 1] = label_vocab.nnw_id
        labels[end_word][start_word] = label_vocab.thw_id(entity.entity_type)
    return W2NERGrid(
        document_id=document_id,
        original_text=original_text,
        words=word_tokens,
        labels=tuple(tuple(row) for row in labels),
        pair_mask=tuple(tuple(row) for row in pair_mask),
        vocab=label_vocab,
    )


def mask_padded_pairs(word_count: int, max_words: int) -> tuple[tuple[bool, ...], ...]:
    """Pair mask for padded grids: true only inside the real word square."""
    if word_count < 0 or max_words < word_count:
        raise W2NERError("max_words must be greater than or equal to word_count")
    return tuple(
        tuple(row < word_count and col < word_count for col in range(max_words))
        for row in range(max_words)
    )


def decode_w2ner_grid(
    grid: W2NERGrid,
    *,
    scores: Sequence[Sequence[float]] | None = None,
    threshold: float = 0.0,
) -> tuple[W2NERDecodedSpan, ...]:
    """Decode contiguous W2NER THW+NNW paths into exact original offsets."""
    decoded: list[W2NERDecodedSpan] = []
    for tail_index, row in enumerate(grid.labels):
        for head_index, label_id in enumerate(row):
            entity_type = grid.vocab.thw_type(label_id)
            if entity_type is None:
                continue
            if not grid.pair_mask[tail_index][head_index] or head_index > tail_index:
                continue
            if any(
                grid.labels[i][i + 1] != grid.vocab.nnw_id
                for i in range(head_index, tail_index)
            ):
                continue
            score = 1.0
            if scores is not None:
                score = float(scores[tail_index][head_index])
            if score < threshold:
                continue
            start = grid.words[head_index].start
            end = grid.words[tail_index].end
            text = grid.original_text[start:end]
            span = W2NERDecodedSpan(
                start=start,
                end=end,
                text=text,
                entity_type=entity_type,
                score=score,
                word_start=head_index,
                word_end=tail_index,
            )
            span.validate_against(grid.original_text)
            decoded.append(span)
    return tuple(sorted(decoded, key=lambda span: (span.start, span.end, span.entity_type)))


def decoded_spans_to_expert_proposals(
    document_id: str,
    original_text: str,
    spans: Sequence[W2NERDecodedSpan],
    *,
    config: PhoBERTW2NERConfig,
    route: str = "",
    section: str = "",
) -> tuple[ExpertSpanProposal, ...]:
    proposals: list[ExpertSpanProposal] = []
    for ordinal, span in enumerate(spans, start=1):
        span.validate_against(original_text)
        proposals.append(
            ExpertSpanProposal(
                document_id=document_id,
                start=span.start,
                end=span.end,
                text=span.text,
                type_scores={span.entity_type: span.score},
                local_score=span.score,
                expert_id=EXPERT_PHOBERT_W2NER,
                proposal_id=f"e4-{document_id}-{ordinal:04d}",
                route=route,
                section=section,
                original_start=span.start,
                original_end=span.end,
                features={
                    "w2ner_word_start": float(span.word_start),
                    "w2ner_word_end": float(span.word_end),
                    "w2ner_word_count": float(span.word_end - span.word_start + 1),
                },
                config_version=config.config_version,
                model_revision=config.model_revision,
                checkpoint_sha256=config.expected_checkpoint_sha256,
                config_sha256=_config_sha256(config),
            )
        )
    return tuple(proposals)


def _config_sha256(config: PhoBERTW2NERConfig) -> str:
    payload = {
        "base_model_name": config.base_model_name,
        "model_revision": config.model_revision,
        "max_words": config.max_words,
        "threshold": config.threshold,
        "config_version": config.config_version,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_w2ner_checkpoint(config: PhoBERTW2NERConfig) -> Path:
    if not config.enabled:
        raise W2NERUnavailableError("E4 PhoBERT W2NER is disabled by profile")
    if not config.checkpoint_path:
        raise W2NERUnavailableError("E4 PhoBERT W2NER has no configured checkpoint")
    path = Path(config.checkpoint_path)
    if not path.exists():
        raise W2NERUnavailableError(f"E4 checkpoint does not exist: {path}")
    if config.expected_checkpoint_sha256 and path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config.expected_checkpoint_sha256:
            raise W2NERUnavailableError("E4 checkpoint SHA-256 mismatch")
    return path


def build_relation_grid_head(hidden_size: int, relation_count: int) -> object:
    """Create a tiny relation-grid head; torch is imported only on demand."""
    if hidden_size <= 0 or relation_count <= 0:
        raise ValueError("hidden_size and relation_count must be positive")
    import torch
    from torch import nn

    class RelationGridHead(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(hidden_size, hidden_size)
            self.right = nn.Linear(hidden_size, hidden_size)
            self.classifier = nn.Linear(hidden_size * 2, relation_count)

        def forward(self, word_embeddings: Any, pair_mask: Any | None = None) -> Any:
            left = self.left(word_embeddings)
            right = self.right(word_embeddings)
            left_grid = left.unsqueeze(2).expand(-1, -1, word_embeddings.size(1), -1)
            right_grid = right.unsqueeze(1).expand(-1, word_embeddings.size(1), -1, -1)
            logits = self.classifier(torch.cat((left_grid, right_grid), dim=-1))
            if pair_mask is None:
                return logits
            return logits.masked_fill(~pair_mask.unsqueeze(-1), -1e4)

    return RelationGridHead()


@dataclass(frozen=True, slots=True)
class W2NERCheckpointMetadata:
    stage: str
    expert: str
    config_sha256: str
    corpus_sha256: str
    model_revision: str
    seed: int
    git_commit: str
    checkpoint_sha256: str
    parameter_count: int
    train_split_id: str
    validation_split_id: str
    internal_test_accessed: bool

    def validate(self) -> None:
        if self.expert != EXPERT_PHOBERT_W2NER:
            raise W2NERError("checkpoint metadata expert must be E4 PhoBERT W2NER")
        if self.internal_test_accessed:
            raise W2NERError("E4 checkpoint manifest must not access internal_test")


__all__ = [
    "DEFAULT_TYPE_ORDER",
    "EXPERT_PHOBERT_W2NER",
    "EntitySpan",
    "PhoBERTW2NERConfig",
    "W2NERCheckpointMetadata",
    "W2NER_CONFIG_VERSION",
    "W2NERDecodedSpan",
    "W2NERError",
    "W2NERGrid",
    "W2NERLabelVocab",
    "W2NERUnavailableError",
    "WordSubwordAlignment",
    "WordToken",
    "align_words_to_subwords",
    "build_relation_grid_head",
    "build_w2ner_grid",
    "decode_w2ner_grid",
    "decoded_spans_to_expert_proposals",
    "mask_padded_pairs",
    "tokenize_words",
    "validate_w2ner_checkpoint",
]
