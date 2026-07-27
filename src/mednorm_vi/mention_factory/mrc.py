"""E5 XLM-R query-based MRC-NER proposal path.

The MRC expert converts each governed document into one query/context example
per organizer type, masks query tokens out of span supervision, and decodes
start/end predictions back to exact original offsets. Full XLM-R training is
not performed locally; the runtime only validates already-present checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..lattice.models import EXPERT_XLMR_MRC, ExpertSpanProposal
from .w2ner import EntitySpan

MRC_CONFIG_VERSION = "xlmr-mrc-ner-v1"
MRC_QUERY_VERSION = "mrc-type-queries-v1"
TYPE_QUERY_ORDER: tuple[str, ...] = (
    "MEDICATION",
    "DIAGNOSIS",
    "SYMPTOM",
    "TEST_NAME",
    "TEST_RESULT",
)
TYPE_QUERIES_V1: Mapping[str, str] = {
    "MEDICATION": "Find medication, drug, dose, route, or treatment mentions.",
    "DIAGNOSIS": "Find confirmed diseases or clinical diagnoses.",
    "SYMPTOM": "Find patient symptoms, complaints, signs, or manifestations.",
    "TEST_NAME": "Find laboratory or clinical test names.",
    "TEST_RESULT": "Find test values, results, measurements, or interpretations.",
}


class MRCError(ValueError):
    """Raised when MRC conversion or decoding is not reversible."""


class MRCUnavailableError(RuntimeError):
    """Raised when the disabled or untrained MRC expert is requested."""


@dataclass(frozen=True, slots=True)
class MRCToken:
    index: int
    text: str
    start: int
    end: int
    is_query: bool

    def validate_against(self, original_text: str) -> None:
        if self.is_query:
            return
        if self.end <= self.start:
            raise MRCError(f"invalid context token offsets {self.start}:{self.end}")
        if original_text[self.start:self.end] != self.text:
            raise MRCError("context token is not an exact original_text slice")


@dataclass(frozen=True, slots=True)
class MRCExample:
    document_id: str
    entity_type: str
    query_version: str
    query: str
    original_text: str
    tokens: tuple[MRCToken, ...]
    query_mask: tuple[bool, ...]
    context_mask: tuple[bool, ...]
    start_labels: tuple[int, ...]
    end_labels: tuple[int, ...]
    gold_spans: tuple[EntitySpan, ...]

    def __post_init__(self) -> None:
        size = len(self.tokens)
        if not (
            len(self.query_mask)
            == len(self.context_mask)
            == len(self.start_labels)
            == len(self.end_labels)
            == size
        ):
            raise MRCError("MRC masks and labels must align with tokens")
        for token in self.tokens:
            token.validate_against(self.original_text)
        if any(label and not self.context_mask[i] for i, label in enumerate(self.start_labels)):
            raise MRCError("start labels may only appear on context tokens")
        if any(label and not self.context_mask[i] for i, label in enumerate(self.end_labels)):
            raise MRCError("end labels may only appear on context tokens")


@dataclass(frozen=True, slots=True)
class MRCDecodedSpan:
    start: int
    end: int
    text: str
    entity_type: str
    start_score: float
    end_score: float

    @property
    def score(self) -> float:
        return (self.start_score + self.end_score) / 2.0

    def validate_against(self, original_text: str) -> None:
        if original_text[self.start:self.end] != self.text:
            raise MRCError("decoded MRC span is not reversible to original text")


@dataclass(frozen=True, slots=True)
class XLMRMRCConfig:
    base_model_name: str = "xlm-roberta-large"
    model_revision: str = ""
    checkpoint_path: str = ""
    expected_checkpoint_sha256: str = ""
    max_length: int = 384
    max_span_chars: int = 96
    threshold: float = 0.5
    allow_overlaps: bool = True
    enabled: bool = False
    query_version: str = MRC_QUERY_VERSION
    config_version: str = MRC_CONFIG_VERSION


def tokenize_context(original_text: str) -> tuple[MRCToken, ...]:
    tokens: list[MRCToken] = []
    for index, match in enumerate(re.finditer(r"\S+", original_text)):
        tokens.append(MRCToken(index, match.group(0), match.start(), match.end(), False))
    return tuple(tokens)


def _query_tokens(query: str) -> tuple[MRCToken, ...]:
    return tuple(
        MRCToken(index, match.group(0), 0, 0, True)
        for index, match in enumerate(re.finditer(r"\S+", query))
    )


def _context_token_index(tokens: Sequence[MRCToken], offset: int, *, is_start: bool) -> int | None:
    for index, token in enumerate(tokens):
        if token.is_query:
            continue
        if is_start and token.start == offset:
            return index
        if not is_start and token.end == offset:
            return index
    return None


def build_mrc_examples(
    document_id: str,
    original_text: str,
    entities: Sequence[EntitySpan],
    *,
    queries: Mapping[str, str] = TYPE_QUERIES_V1,
    query_version: str = MRC_QUERY_VERSION,
) -> tuple[MRCExample, ...]:
    context_tokens = tokenize_context(original_text)
    examples: list[MRCExample] = []
    for entity_type in TYPE_QUERY_ORDER:
        query = queries[entity_type]
        query_tokens = _query_tokens(query)
        combined = query_tokens + tuple(
            MRCToken(
                index=len(query_tokens) + token.index,
                text=token.text,
                start=token.start,
                end=token.end,
                is_query=False,
            )
            for token in context_tokens
        )
        query_mask = tuple(token.is_query for token in combined)
        context_mask = tuple(not token.is_query for token in combined)
        start_labels = [0 for _ in combined]
        end_labels = [0 for _ in combined]
        gold_for_type: list[EntitySpan] = []
        for entity in entities:
            entity.validate_against(original_text)
            if entity.entity_type != entity_type:
                continue
            start_index = _context_token_index(combined, entity.start, is_start=True)
            end_index = _context_token_index(combined, entity.end, is_start=False)
            if start_index is None or end_index is None or end_index < start_index:
                raise MRCError(
                    f"entity {entity.entity_type} {entity.start}:{entity.end} "
                    "is not exactly recoverable from context tokens"
                )
            start_labels[start_index] = 1
            end_labels[end_index] = 1
            gold_for_type.append(entity)
        examples.append(
            MRCExample(
                document_id=document_id,
                entity_type=entity_type,
                query_version=query_version,
                query=query,
                original_text=original_text,
                tokens=combined,
                query_mask=query_mask,
                context_mask=context_mask,
                start_labels=tuple(start_labels),
                end_labels=tuple(end_labels),
                gold_spans=tuple(gold_for_type),
            )
        )
    return tuple(examples)


def _score_at(scores: Sequence[float], index: int) -> float:
    if index < 0 or index >= len(scores):
        raise MRCError("score vector length does not match tokens")
    return float(scores[index])


def pair_start_end(
    example: MRCExample,
    start_scores: Sequence[float],
    end_scores: Sequence[float],
    *,
    threshold: float = 0.5,
    max_span_chars: int = 96,
    allow_overlaps: bool = True,
) -> tuple[MRCDecodedSpan, ...]:
    """Pair valid context starts and ends under deterministic constraints."""
    if len(start_scores) != len(example.tokens) or len(end_scores) != len(example.tokens):
        raise MRCError("start/end score vectors must align with tokens")
    candidates: list[MRCDecodedSpan] = []
    for start_index, token in enumerate(example.tokens):
        if not example.context_mask[start_index]:
            continue
        start_score = _score_at(start_scores, start_index)
        if start_score < threshold:
            continue
        best_span: MRCDecodedSpan | None = None
        for end_index in range(start_index, len(example.tokens)):
            end_token = example.tokens[end_index]
            if not example.context_mask[end_index]:
                continue
            if end_token.end - token.start > max_span_chars:
                break
            end_score = _score_at(end_scores, end_index)
            if end_score < threshold:
                continue
            text = example.original_text[token.start:end_token.end]
            span = MRCDecodedSpan(
                start=token.start,
                end=end_token.end,
                text=text,
                entity_type=example.entity_type,
                start_score=start_score,
                end_score=end_score,
            )
            span.validate_against(example.original_text)
            if best_span is None or (span.score, -span.end) > (best_span.score, -best_span.end):
                best_span = span
        if best_span is not None:
            candidates.append(best_span)
    ordered = sorted(candidates, key=lambda span: (-span.score, span.start, span.end))
    if allow_overlaps:
        return tuple(sorted(ordered, key=lambda span: (span.start, span.end, span.entity_type)))
    selected: list[MRCDecodedSpan] = []
    for candidate in ordered:
        if any(candidate.start < kept.end and kept.start < candidate.end for kept in selected):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda span: (span.start, span.end, span.entity_type)))


def decoded_spans_to_expert_proposals(
    document_id: str,
    original_text: str,
    spans: Sequence[MRCDecodedSpan],
    *,
    config: XLMRMRCConfig,
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
                expert_id=EXPERT_XLMR_MRC,
                proposal_id=f"e5-{document_id}-{ordinal:04d}",
                route=route,
                section=section,
                original_start=span.start,
                original_end=span.end,
                features={
                    "mrc_start_score": span.start_score,
                    "mrc_end_score": span.end_score,
                    "mrc_span_length": float(span.end - span.start),
                },
                config_version=config.config_version,
                model_revision=config.model_revision,
                checkpoint_sha256=config.expected_checkpoint_sha256,
                config_sha256=config_sha256(config),
            )
        )
    return tuple(proposals)


def config_sha256(config: XLMRMRCConfig) -> str:
    payload = {
        "base_model_name": config.base_model_name,
        "model_revision": config.model_revision,
        "max_length": config.max_length,
        "max_span_chars": config.max_span_chars,
        "threshold": config.threshold,
        "allow_overlaps": config.allow_overlaps,
        "query_version": config.query_version,
        "queries": dict(TYPE_QUERIES_V1),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_mrc_checkpoint(config: XLMRMRCConfig) -> Path:
    if not config.enabled:
        raise MRCUnavailableError("E5 XLM-R MRC-NER is disabled by profile")
    if not config.checkpoint_path:
        raise MRCUnavailableError("E5 XLM-R MRC-NER has no configured checkpoint")
    path = Path(config.checkpoint_path)
    if not path.exists():
        raise MRCUnavailableError(f"E5 checkpoint does not exist: {path}")
    if config.expected_checkpoint_sha256 and path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config.expected_checkpoint_sha256:
            raise MRCUnavailableError("E5 checkpoint SHA-256 mismatch")
    return path


def build_mrc_span_head(hidden_size: int) -> object:
    """Create start/end classifiers; torch is imported only on demand."""
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    import torch
    from torch import nn

    class MRCSpanHead(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.start = nn.Linear(hidden_size, 1)
            self.end = nn.Linear(hidden_size, 1)

        def forward(self, sequence_output: Any, context_mask: Any) -> tuple[Any, Any]:
            start_logits = self.start(sequence_output).squeeze(-1)
            end_logits = self.end(sequence_output).squeeze(-1)
            masked_start = start_logits.masked_fill(~context_mask, -1e4)
            masked_end = end_logits.masked_fill(~context_mask, -1e4)
            return torch.sigmoid(masked_start), torch.sigmoid(masked_end)

    return MRCSpanHead()


@dataclass(frozen=True, slots=True)
class MRCCheckpointMetadata:
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
        if self.expert != EXPERT_XLMR_MRC:
            raise MRCError("checkpoint metadata expert must be E5 XLM-R MRC-NER")
        if self.internal_test_accessed:
            raise MRCError("E5 checkpoint manifest must not access internal_test")


__all__ = [
    "EXPERT_XLMR_MRC",
    "MRCCheckpointMetadata",
    "MRC_CONFIG_VERSION",
    "MRCDecodedSpan",
    "MRCError",
    "MRCExample",
    "MRC_QUERY_VERSION",
    "MRCUnavailableError",
    "MRCToken",
    "TYPE_QUERIES_V1",
    "TYPE_QUERY_ORDER",
    "XLMRMRCConfig",
    "build_mrc_examples",
    "build_mrc_span_head",
    "config_sha256",
    "decoded_spans_to_expert_proposals",
    "pair_start_end",
    "tokenize_context",
    "validate_mrc_checkpoint",
]
