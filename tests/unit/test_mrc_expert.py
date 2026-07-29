"""E5 XLM-R MRC-NER query conversion and pairing contracts."""

from __future__ import annotations

import pytest

from mednorm_vi.mention_factory.mrc import (
    TYPE_QUERIES_V1,
    MRCError,
    XLMRMRCConfig,
    build_mrc_examples,
    decoded_spans_to_expert_proposals,
    pair_start_end,
)
from mednorm_vi.mention_factory.spans import EntitySpan


def test_mrc_conversion_builds_five_versioned_query_examples() -> None:
    text = "uống aspirin và ho khan"
    medication_start = text.index("aspirin")
    examples = build_mrc_examples(
        "doc",
        text,
        (EntitySpan(medication_start, medication_start + 7, "MEDICATION", "aspirin"),),
    )
    assert [example.entity_type for example in examples] == list(TYPE_QUERIES_V1)
    medication = next(example for example in examples if example.entity_type == "MEDICATION")
    assert medication.query
    assert any(medication.query_mask)
    assert any(medication.context_mask)
    assert all(
        not medication.query_mask[i]
        for i, label in enumerate(medication.start_labels)
        if label
    )
    assert all(
        not medication.query_mask[i]
        for i, label in enumerate(medication.end_labels)
        if label
    )


def test_mrc_pairing_ignores_query_tokens_and_recovers_exact_offsets() -> None:
    text = "Bệnh nhân suy tim"
    start = text.index("suy")
    example = next(
        ex
        for ex in build_mrc_examples(
            "doc", text, (EntitySpan(start, len(text), "DIAGNOSIS", "suy tim"),)
        )
        if ex.entity_type == "DIAGNOSIS"
    )
    start_scores = [0.0 for _ in example.tokens]
    end_scores = [0.0 for _ in example.tokens]
    query_index = next(i for i, is_query in enumerate(example.query_mask) if is_query)
    start_scores[query_index] = 1.0
    end_scores[query_index] = 1.0
    for i, label in enumerate(example.start_labels):
        if label:
            start_scores[i] = 1.0
    for i, label in enumerate(example.end_labels):
        if label:
            end_scores[i] = 1.0
    decoded = pair_start_end(example, start_scores, end_scores, threshold=1.0)
    assert [(span.start, span.end, span.text) for span in decoded] == [
        (start, len(text), "suy tim")
    ]


def test_mrc_pairing_can_suppress_overlaps_deterministically() -> None:
    text = "suy tim nặng"
    example = next(
        ex
        for ex in build_mrc_examples("doc", text, ())
        if ex.entity_type == "DIAGNOSIS"
    )
    start_scores = [0.0 for _ in example.tokens]
    end_scores = [0.0 for _ in example.tokens]
    context_indices = [i for i, is_context in enumerate(example.context_mask) if is_context]
    start_scores[context_indices[0]] = 0.9
    end_scores[context_indices[1]] = 0.9
    end_scores[context_indices[2]] = 0.8
    decoded = pair_start_end(
        example,
        start_scores,
        end_scores,
        threshold=0.8,
        allow_overlaps=False,
    )
    assert len(decoded) == 1
    assert decoded[0].text == "suy tim"


def test_mrc_conversion_rejects_non_recoverable_entity_offsets() -> None:
    text = "suy tim"
    with pytest.raises(MRCError):
        build_mrc_examples("doc", text, (EntitySpan(1, 7, "DIAGNOSIS", "uy tim"),))


def test_mrc_decoded_spans_convert_to_expert_proposals() -> None:
    text = "uống aspirin"
    start = text.index("aspirin")
    example = next(
        ex
        for ex in build_mrc_examples(
            "doc", text, (EntitySpan(start, len(text), "MEDICATION", "aspirin"),)
        )
        if ex.entity_type == "MEDICATION"
    )
    scores = [float(label) for label in example.start_labels]
    ends = [float(label) for label in example.end_labels]
    decoded = pair_start_end(example, scores, ends, threshold=1.0)
    proposals = decoded_spans_to_expert_proposals(
        "doc",
        text,
        decoded,
        config=XLMRMRCConfig(model_revision="rev", expected_checkpoint_sha256="b" * 64),
    )
    assert proposals[0].text == "aspirin"
    assert proposals[0].model_revision == "rev"
    assert proposals[0].type_scores == {"MEDICATION": 1.0}
