"""E4 PhoBERT W2NER grid and decoding contracts."""

from __future__ import annotations

import unicodedata

import pytest

from mednorm_vi.mention_factory.w2ner import (
    EntitySpan,
    PhoBERTW2NERConfig,
    W2NERError,
    W2NERLabelVocab,
    build_w2ner_grid,
    decode_w2ner_grid,
    decoded_spans_to_expert_proposals,
    mask_padded_pairs,
)


def test_w2ner_grid_uses_nnw_and_tail_head_labels() -> None:
    text = "suy tim nặng"
    entity = EntitySpan(0, 7, "DIAGNOSIS", "suy tim")
    vocab = W2NERLabelVocab()
    grid = build_w2ner_grid("doc", text, (entity,), vocab=vocab)
    assert [word.text for word in grid.words] == ["suy", "tim", "nặng"]
    assert grid.labels[0][1] == vocab.nnw_id
    assert grid.labels[1][0] == vocab.thw_id("DIAGNOSIS")
    assert all(all(row) for row in grid.pair_mask)


def test_w2ner_decoder_supports_contiguous_overlapping_spans() -> None:
    text = "suy tim nặng"
    spans = (
        EntitySpan(0, 7, "DIAGNOSIS", "suy tim"),
        EntitySpan(0, len(text), "SYMPTOM", text),
    )
    decoded = decode_w2ner_grid(build_w2ner_grid("doc", text, spans))
    assert [(span.start, span.end, span.entity_type) for span in decoded] == [
        (0, 7, "DIAGNOSIS"),
        (0, len(text), "SYMPTOM"),
    ]
    assert all(text[span.start:span.end] == span.text for span in decoded)


def test_w2ner_rejects_non_word_aligned_gold_span() -> None:
    text = "suy tim"
    with pytest.raises(W2NERError):
        build_w2ner_grid("doc", text, (EntitySpan(1, 7, "DIAGNOSIS", text[1:7]),))


def test_w2ner_padded_pair_mask_marks_only_real_words() -> None:
    assert mask_padded_pairs(2, 4) == (
        (True, True, False, False),
        (True, True, False, False),
        (False, False, False, False),
        (False, False, False, False),
    )


def test_w2ner_decoded_spans_convert_to_lattice_contract_with_provenance() -> None:
    text = unicodedata.normalize("NFD", "sốt cao")
    entity = EntitySpan(0, len(text), "SYMPTOM", text)
    decoded = decode_w2ner_grid(build_w2ner_grid("doc", text, (entity,)))
    config = PhoBERTW2NERConfig(
        model_revision="rev",
        expected_checkpoint_sha256="a" * 64,
    )
    proposals = decoded_spans_to_expert_proposals("doc", text, decoded, config=config)
    proposal = proposals[0]
    assert proposal.start == 0 and proposal.end == len(text)
    assert proposal.text == text
    assert proposal.type_scores == {"SYMPTOM": 1.0}
    assert proposal.model_revision == "rev"
    assert proposal.checkpoint_sha256 == "a" * 64
