"""E6 GLiNER strict adapter.

GLiNER is treated as an external open-type mention proposer. This adapter never
downloads a model and never fabricates spans: it converts already-returned local
backend spans into the common L3 ``ExpertSpanProposal`` contract only after exact
substring and offset validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..lattice.models import EXPERT_GLINER, ExpertSpanProposal
from ..schemas.constants import ENTITY_TYPES

GLINER_CONFIG_VERSION = "gliner-adapter-v1"
GLINER_LABEL_DESCRIPTION_VERSION = "gliner-label-descriptions-v1"
GLINER_LABEL_DESCRIPTIONS_V1: Mapping[str, str] = {
    "MEDICATION": "Medication, drug, dose, route, frequency, or treatment mention.",
    "DIAGNOSIS": "Disease, disorder, confirmed diagnosis, or clinical condition.",
    "SYMPTOM": "Symptom, sign, patient complaint, or manifestation.",
    "TEST_NAME": "Laboratory, imaging, or clinical test name.",
    "TEST_RESULT": "Test result, numeric value, measurement, or interpretation.",
}


class GLiNERAdapterError(ValueError):
    """Raised when GLiNER returns a malformed or unsupported span."""


class GLiNERUnavailableError(RuntimeError):
    """Raised when the disabled or missing local GLiNER checkpoint is requested."""


class GLiNERBackend(Protocol):
    """Minimal local backend surface; implementations are injected by operators."""

    def predict_entities(
        self, texts: Sequence[str], labels: Sequence[str], threshold: float
    ) -> Sequence[Sequence[Mapping[str, object]]]:
        ...


@dataclass(frozen=True, slots=True)
class GLiNERConfig:
    local_model_path: str = ""
    model_revision: str = ""
    expected_checkpoint_sha256: str = ""
    threshold: float = 0.5
    enabled: bool = False
    label_descriptions: Mapping[str, str] = field(
        default_factory=lambda: dict(GLINER_LABEL_DESCRIPTIONS_V1)
    )
    label_description_version: str = GLINER_LABEL_DESCRIPTION_VERSION
    config_version: str = GLINER_CONFIG_VERSION

    def validate_labels(self) -> None:
        if set(self.label_descriptions) != set(ENTITY_TYPES):
            raise GLiNERAdapterError("GLiNER label descriptions must cover exactly five types")


@dataclass(frozen=True, slots=True)
class GLiNERRawSpan:
    start: int
    end: int
    text: str
    label: str
    score: float


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GLiNERAdapterError(f"GLiNER field {field_name!r} must be an integer")
    return value


def _require_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GLiNERAdapterError(f"GLiNER field {field_name!r} must be numeric")
    return float(value)


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise GLiNERAdapterError(f"GLiNER field {field_name!r} must be a string")
    return value


def parse_raw_span(raw: Mapping[str, object]) -> GLiNERRawSpan:
    return GLiNERRawSpan(
        start=_require_int(raw.get("start"), "start"),
        end=_require_int(raw.get("end"), "end"),
        text=_require_str(raw.get("text"), "text"),
        label=_require_str(raw.get("label"), "label"),
        score=_require_float(raw.get("score"), "score"),
    )


def config_sha256(config: GLiNERConfig) -> str:
    payload = {
        "local_model_path": config.local_model_path,
        "model_revision": config.model_revision,
        "threshold": config.threshold,
        "label_description_version": config.label_description_version,
        "label_descriptions": dict(config.label_descriptions),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_gliner_checkpoint(config: GLiNERConfig) -> Path:
    config.validate_labels()
    if not config.enabled:
        raise GLiNERUnavailableError("E6 GLiNER is disabled by profile")
    if not config.local_model_path:
        raise GLiNERUnavailableError("E6 GLiNER has no configured local model path")
    path = Path(config.local_model_path)
    if not path.exists():
        raise GLiNERUnavailableError(f"E6 local model path does not exist: {path}")
    if config.expected_checkpoint_sha256 and path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config.expected_checkpoint_sha256:
            raise GLiNERUnavailableError("E6 checkpoint SHA-256 mismatch")
    return path


def convert_gliner_span(
    document_id: str,
    original_text: str,
    raw: Mapping[str, object] | GLiNERRawSpan,
    *,
    config: GLiNERConfig,
    ordinal: int,
    route: str = "",
    section: str = "",
) -> ExpertSpanProposal:
    config.validate_labels()
    span = raw if isinstance(raw, GLiNERRawSpan) else parse_raw_span(raw)
    if span.label not in config.label_descriptions:
        raise GLiNERAdapterError(f"unsupported GLiNER label {span.label!r}")
    if span.end <= span.start:
        raise GLiNERAdapterError(f"invalid GLiNER offsets {span.start}:{span.end}")
    if not 0.0 <= span.score <= 1.0:
        raise GLiNERAdapterError("GLiNER confidence must be in [0, 1]")
    if original_text[span.start:span.end] != span.text:
        raise GLiNERAdapterError("GLiNER span text does not match original offsets")
    return ExpertSpanProposal(
        document_id=document_id,
        start=span.start,
        end=span.end,
        text=span.text,
        type_scores={span.label: span.score},
        local_score=span.score,
        expert_id=EXPERT_GLINER,
        proposal_id=f"e6-{document_id}-{ordinal:04d}",
        route=route,
        section=section,
        original_start=span.start,
        original_end=span.end,
        evidence_ids=(f"{GLINER_LABEL_DESCRIPTION_VERSION}:{span.label}",),
        features={
            f"gliner_confidence_{span.label}": span.score,
            "gliner_label_description_count": float(len(config.label_descriptions)),
        },
        config_version=config.config_version,
        model_revision=config.model_revision,
        checkpoint_sha256=config.expected_checkpoint_sha256,
        config_sha256=config_sha256(config),
    )


class GLiNERAdapter:
    """Forward-only adapter around an injected local GLiNER backend."""

    def __init__(self, config: GLiNERConfig, backend: GLiNERBackend | None = None) -> None:
        self.config = config
        self.backend = backend
        self.config.validate_labels()

    @property
    def available(self) -> bool:
        if not self.config.enabled or self.backend is None:
            return False
        try:
            validate_gliner_checkpoint(self.config)
        except GLiNERUnavailableError:
            return False
        return True

    def infer_batch(
        self,
        document_ids: Sequence[str],
        texts: Sequence[str],
    ) -> tuple[tuple[ExpertSpanProposal, ...], ...]:
        if len(document_ids) != len(texts):
            raise GLiNERAdapterError("document_ids and texts must have the same length")
        if not self.config.enabled:
            raise GLiNERUnavailableError("E6 GLiNER is disabled by profile")
        validate_gliner_checkpoint(self.config)
        if self.backend is None:
            raise GLiNERUnavailableError("E6 GLiNER has no injected local backend")
        labels = tuple(self.config.label_descriptions)
        raw_batches = self.backend.predict_entities(texts, labels, self.config.threshold)
        if len(raw_batches) != len(texts):
            raise GLiNERAdapterError("GLiNER backend returned the wrong batch size")
        converted: list[tuple[ExpertSpanProposal, ...]] = []
        for document_id, text, raw_spans in zip(document_ids, texts, raw_batches, strict=True):
            proposals = tuple(
                convert_gliner_span(
                    document_id,
                    text,
                    raw,
                    config=self.config,
                    ordinal=ordinal,
                )
                for ordinal, raw in enumerate(raw_spans, start=1)
            )
            converted.append(proposals)
        return tuple(converted)

    def infer_one(self, document_id: str, text: str) -> tuple[ExpertSpanProposal, ...]:
        return self.infer_batch((document_id,), (text,))[0]


__all__ = [
    "EXPERT_GLINER",
    "GLINER_CONFIG_VERSION",
    "GLINER_LABEL_DESCRIPTION_VERSION",
    "GLINER_LABEL_DESCRIPTIONS_V1",
    "GLiNERAdapter",
    "GLiNERAdapterError",
    "GLiNERBackend",
    "GLiNERConfig",
    "GLiNERRawSpan",
    "GLiNERUnavailableError",
    "config_sha256",
    "convert_gliner_span",
    "parse_raw_span",
    "validate_gliner_checkpoint",
]
