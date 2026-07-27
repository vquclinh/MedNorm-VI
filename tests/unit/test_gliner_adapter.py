"""E6 GLiNER adapter offset and availability contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from mednorm_vi.mention_factory.gliner import (
    GLiNERAdapter,
    GLiNERAdapterError,
    GLiNERConfig,
    GLiNERUnavailableError,
    convert_gliner_span,
    validate_gliner_checkpoint,
)


def test_gliner_valid_span_converts_to_l3_proposal() -> None:
    text = "Bệnh nhân ho khan"
    start = text.index("ho")
    proposal = convert_gliner_span(
        "doc",
        text,
        {"start": start, "end": len(text), "text": "ho khan", "label": "SYMPTOM", "score": 0.82},
        config=GLiNERConfig(model_revision="rev", expected_checkpoint_sha256="c" * 64),
        ordinal=1,
    )
    assert proposal.start == start
    assert proposal.end == len(text)
    assert proposal.text == "ho khan"
    assert proposal.type_scores == {"SYMPTOM": 0.82}
    assert proposal.evidence_ids == ("gliner-label-descriptions-v1:SYMPTOM",)


def test_gliner_rejects_malformed_or_non_reversible_spans() -> None:
    text = "suy tim"
    config = GLiNERConfig()
    with pytest.raises(GLiNERAdapterError):
        convert_gliner_span(
            "doc",
            text,
            {"start": 0, "end": 3, "text": "XXX", "label": "DIAGNOSIS", "score": 0.8},
            config=config,
            ordinal=1,
        )
    with pytest.raises(GLiNERAdapterError):
        convert_gliner_span(
            "doc",
            text,
            {"start": 0, "end": 7, "text": "suy tim", "label": "NOT_A_TYPE", "score": 0.8},
            config=config,
            ordinal=1,
        )


def test_gliner_requires_configured_local_checkpoint_when_enabled(tmp_path: Path) -> None:
    with pytest.raises(GLiNERUnavailableError):
        validate_gliner_checkpoint(
            GLiNERConfig(enabled=True, local_model_path=str(tmp_path / "missing"))
        )
    with pytest.raises(GLiNERUnavailableError):
        GLiNERAdapter(GLiNERConfig(enabled=True, local_model_path=str(tmp_path))).infer_one(
            "doc", "abc"
        )


def test_gliner_disabled_adapter_reports_unavailable_without_downloading() -> None:
    adapter = GLiNERAdapter(GLiNERConfig(enabled=False))
    assert not adapter.available
    with pytest.raises(GLiNERUnavailableError):
        adapter.infer_one("doc", "abc")
