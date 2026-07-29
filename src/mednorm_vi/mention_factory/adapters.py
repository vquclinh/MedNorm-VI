"""Unified L3 neural-expert adapter contracts.

The adapters define how ViHealthBERT, XLM-R MRC-NER, and GLiNER
checkpoints plug into the existing ``SpanProposal`` contract. They do not
fabricate predictions: missing checkpoints fail unless a caller has explicitly
configured a fallback-to-empty dry run. The production E7 Qwen3 proposer is the
disabled interface in ``mention_factory.qwen_proposer``; the anchored helper in
this file is retained only for older unit fixtures.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas.constants import ORGANIZER_LABEL_BY_TYPE
from .models import SpanProposal


class MissingCheckpointError(RuntimeError):
    """Raised when an enabled neural expert has no local checkpoint."""


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    model_id: str
    role: str
    path: str
    checkpoint_hash: str = ""
    tokenizer_hash: str = ""
    parameter_count: int = 0

    def validate_local(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise MissingCheckpointError(f"missing checkpoint for {self.model_id}: {p}")


@dataclass(frozen=True, slots=True)
class Window:
    start: int
    end: int
    text: str


def make_windows(text: str, *, window_chars: int, stride_chars: int) -> tuple[Window, ...]:
    """Long-document windowing with exact original offsets."""
    if window_chars <= 0 or stride_chars <= 0:
        raise ValueError("window_chars and stride_chars must be positive")
    windows: list[Window] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + window_chars)
        windows.append(Window(start=start, end=end, text=text[start:end]))
        if end == len(text):
            break
        start += stride_chars
    return tuple(windows)


def anchor_substring(text: str, substring: str, *, occurrence: int = 0) -> tuple[int, int]:
    """Locate an emitted substring in source text; required for Qwen proposals."""
    if not substring:
        raise ValueError("empty anchored substring")
    start = -1
    search_from = 0
    for _ in range(occurrence + 1):
        start = text.find(substring, search_from)
        if start < 0:
            raise ValueError(f"substring is not anchored in source text: {substring!r}")
        search_from = start + 1
    return start, start + len(substring)


@dataclass(frozen=True, slots=True)
class NeuralExpertAdapter:
    expert_id: str
    checkpoint: CheckpointManifest
    emits_types: tuple[str, ...]
    route_cases: tuple[str, ...] = field(default_factory=tuple)
    fallback_to_empty: bool = False

    def propose(self, document_id: str, text: str) -> tuple[SpanProposal, ...]:
        self.checkpoint.validate_local()
        raise MissingCheckpointError(
            f"{self.expert_id} inference backend is not installed in this offline milestone"
        )


@dataclass(frozen=True, slots=True)
class AnchoredQwenProposer(NeuralExpertAdapter):
    """Legacy anchored helper retained for pre-Phase-2 tests, not production E7."""

    def from_anchored_substrings(
        self,
        document_id: str,
        text: str,
        substrings: tuple[tuple[str, str], ...],
    ) -> tuple[SpanProposal, ...]:
        self.checkpoint.validate_local()
        proposals: list[SpanProposal] = []
        for _idx, (substring, internal_type) in enumerate(substrings, 1):
            start, end = anchor_substring(text, substring)
            label = ORGANIZER_LABEL_BY_TYPE[internal_type]
            digest = hashlib.sha256(f"{document_id}:{start}:{end}:{label}".encode()).hexdigest()
            proposals.append(
                SpanProposal(
                    proposal_id=f"qwen-{digest[:12]}",
                    document_id=document_id,
                    start=start,
                    end=end,
                    text=substring,
                    proposed_types=(label,),
                    source_specialist=self.expert_id,
                    source_node_id="qwen-window",
                    source_routes=tuple(self.route_cases),
                    local_score=0.0,
                    matched_rule="qwen:anchored_substring",
                    config_version=self.checkpoint.checkpoint_hash,
                    lexicon_version=self.checkpoint.tokenizer_hash,
                )
            )
        return tuple(sorted(proposals, key=lambda p: (p.start, p.end, p.proposal_id)))


def default_neural_adapters(
    checkpoint_root: str | Path, *, fallback_to_empty: bool = False
) -> tuple[NeuralExpertAdapter, ...]:
    root = Path(checkpoint_root)
    return (
        NeuralExpertAdapter(
            "vihealthbert-span-type",
            CheckpointManifest("vihealthbert-span-type", "span_type", str(root / "vihealthbert")),
            ("MEDICATION", "DIAGNOSIS", "SYMPTOM", "TEST_NAME", "TEST_RESULT"),
            ("C1", "C2", "C3", "C4", "C5"),
            fallback_to_empty,
        ),
        NeuralExpertAdapter(
            "xlmr-mrc-ner",
            CheckpointManifest("xlmr-mrc-ner", "mrc_ner", str(root / "xlmr_mrc")),
            ("MEDICATION", "DIAGNOSIS", "SYMPTOM", "TEST_NAME", "TEST_RESULT"),
            ("C3", "C4", "C5"),
            fallback_to_empty,
        ),
        NeuralExpertAdapter(
            "gliner",
            CheckpointManifest("gliner", "open_label_ner", str(root / "gliner")),
            ("MEDICATION", "DIAGNOSIS", "SYMPTOM", "TEST_NAME", "TEST_RESULT"),
            ("C3", "C4", "C5"),
            fallback_to_empty,
        ),
    )


__all__ = [
    "AnchoredQwenProposer",
    "CheckpointManifest",
    "MissingCheckpointError",
    "NeuralExpertAdapter",
    "Window",
    "anchor_substring",
    "default_neural_adapters",
    "make_windows",
]
