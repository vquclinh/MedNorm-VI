"""E4 atomic W2NER grid-word alignment and projection (Audit 0038).

The anchor case is the real Colab failure: a governed SYMPTOM whose start falls
inside a single VnCoreNLP segmented model word.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from mednorm_vi.mention_factory.w2ner import (
    ATOMIC_WORD_POLICY_VERSION,
    W2NER_CONFIG_VERSION,
    EntitySpan,
    W2NERError,
    build_w2ner_grid,
    decode_w2ner_grid,
    entity_atomic_alignment,
    is_atomic_boundary_character,
    tokenize_atomic_words,
)
from mednorm_vi.training.phase2.e4.alignment import (
    ATOMIC_FEATURE_DIM,
    ATOMIC_PROJECTION_VERSION,
    E4_INPUT_CONTRACT_VERSION,
    E4ContractError,
    atomic_relation_head_input_dim,
    build_atomic_projection,
    build_w2ner_batch_contract_from_segmented_words,
    prepare_phobert_word_inputs,
)
from mednorm_vi.training.phase2.e4.alignment_diagnostic import (
    ALIGNED,
    BOTH,
    LEFT_ONLY,
    RIGHT_ONLY,
    E4DiagnosticError,
    classify_alignment,
    run_alignment_diagnostic,
)
from mednorm_vi.training.phase2.e4.contracts import (
    E4_CHECKPOINT_SCHEMA_VERSION,
)
from mednorm_vi.training.phobert_alignment import SegmentedWord

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The exact governed example that failed in Colab after Audit 0037.
# ---------------------------------------------------------------------------
FAILED_EXAMPLE_ID = "vimedner:train:train-000054"
FAILED_TEXT = (
    "tìm kiếm các dấu hiệu của các bệnh khác , chẳng hạn như bệnh tuyến giáp , "
    "có thể gây rối loạn nhịp tim ."
)
FAILED_ENTITY = EntitySpan(85, 102, "SYMPTOM", "rối loạn nhịp tim")
DIAGNOSIS_ENTITY = EntitySpan(56, 71, "DIAGNOSIS", "bệnh tuyến giáp")

# VnCoreNLP output for that sentence, as observed in the real Colab run. The
# segmenter merges "gây rối" into one model word spanning 81:88.
FAILED_SEGMENTED_WORDS: tuple[tuple[str, int, int], ...] = (
    ("tìm_kiếm", 0, 8), ("các", 9, 12), ("dấu_hiệu", 13, 21), ("của", 22, 25),
    ("các", 26, 29), ("bệnh", 30, 34), ("khác", 35, 39), (",", 40, 41),
    ("chẳng_hạn", 42, 51), ("như", 52, 55), ("bệnh", 56, 60), ("tuyến_giáp", 61, 71),
    (",", 72, 73), ("có_thể", 74, 80), ("gây_rối", 81, 88), ("loạn", 89, 93),
    ("nhịp", 94, 98), ("tim", 99, 102), (".", 103, 104),
)


def _model_words(text: str, spec=FAILED_SEGMENTED_WORDS) -> list[SegmentedWord]:
    return [
        SegmentedWord(model_text=model_text, original_start=start, original_end=end)
        for model_text, start, end in spec
    ]


class _FakeSlowPhobertTokenizer:
    """Slow-tokenizer surface only: no ``offset_mapping``, one piece per word."""

    is_fast = False
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1

    def tokenize(self, text: str) -> list[str]:
        return [text]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [100 + index for index, _ in enumerate(tokens)]


def _projection(text: str = FAILED_TEXT, spec=FAILED_SEGMENTED_WORDS):
    contract = build_w2ner_batch_contract_from_segmented_words(
        FAILED_EXAMPLE_ID, text, (DIAGNOSIS_ENTITY, FAILED_ENTITY),
        _model_words(text, spec), max_words=256)
    encoding = prepare_phobert_word_inputs(
        _FakeSlowPhobertTokenizer(), contract.segmented_words, max_length=512)
    return contract, encoding, build_atomic_projection(
        text, contract.segmented_words, encoding, atomic_words=contract.atomic_words)


# ---------------------------------------------------------------------------
# The anchor regression
# ---------------------------------------------------------------------------


def test_the_real_colab_failure_is_fixed() -> None:
    contract, _encoding, _projection_result = _projection()
    decoded = {(span.start, span.end, span.entity_type)
               for span in decode_w2ner_grid(contract.grid)}
    assert (85, 102, "SYMPTOM") in decoded
    assert (56, 71, "DIAGNOSIS") in decoded


def test_gold_entity_offsets_and_text_are_untouched() -> None:
    assert FAILED_TEXT[85:102] == "rối loạn nhịp tim"
    contract, _encoding, _p = _projection()
    for span in decode_w2ner_grid(contract.grid):
        assert FAILED_TEXT[span.start:span.end] == span.text


def test_gay_roi_splits_into_two_atomic_words() -> None:
    words = tokenize_atomic_words(FAILED_TEXT)
    by_span = {(word.start, word.end): word.text for word in words}
    assert by_span[(81, 84)] == "gây"
    assert by_span[(85, 88)] == "rối"
    assert by_span[(89, 93)] == "loạn"
    assert by_span[(94, 98)] == "nhịp"
    assert by_span[(99, 102)] == "tim"


def test_entity_aligns_to_four_atomic_words() -> None:
    words = tokenize_atomic_words(FAILED_TEXT)
    covered = [w for w in words if FAILED_ENTITY.start <= w.start and w.end <= FAILED_ENTITY.end]
    assert [w.text for w in covered] == ["rối", "loạn", "nhịp", "tim"]
    assert covered[0].start == 85 and covered[-1].end == 102


def test_the_segmented_model_word_surface_still_cannot_represent_it() -> None:
    """The old grid genuinely could not express this entity — not a flaky failure."""
    model_word_tokens = tuple(
        __import__("mednorm_vi.mention_factory.w2ner", fromlist=["WordToken"]).WordToken(
            index, FAILED_TEXT[start:end], start, end)
        for index, (_text, start, end) in enumerate(FAILED_SEGMENTED_WORDS)
    )
    left, right = entity_atomic_alignment(model_word_tokens, FAILED_ENTITY)
    assert left is False and right is True
    with pytest.raises(W2NERError, match="not word-aligned"):
        build_w2ner_grid(
            FAILED_EXAMPLE_ID, FAILED_TEXT, (FAILED_ENTITY,), words=model_word_tokens)


def test_misalignment_error_names_the_straddling_word() -> None:
    module = __import__("mednorm_vi.mention_factory.w2ner", fromlist=["WordToken"])
    words = (module.WordToken(0, FAILED_TEXT[81:88], 81, 88),)
    with pytest.raises(W2NERError) as excinfo:
        build_w2ner_grid("x", FAILED_TEXT, (FAILED_ENTITY,), words=words)
    message = str(excinfo.value)
    assert "left_aligned=False" in message
    assert "straddling word 0 81:88" in message


# ---------------------------------------------------------------------------
# Atomic word construction
# ---------------------------------------------------------------------------


def test_punctuation_is_its_own_atomic_word() -> None:
    words = tokenize_atomic_words("sốt , ho .")
    assert [w.text for w in words] == ["sốt", ",", "ho", "."]


def test_punctuation_without_surrounding_spaces_still_splits() -> None:
    words = tokenize_atomic_words("suy tim,ho khan.")
    assert [w.text for w in words] == ["suy", "tim", ",", "ho", "khan", "."]


def test_repeated_spaces_do_not_change_the_word_set_but_do_shift_offsets() -> None:
    tight = tokenize_atomic_words("ho khan")
    loose = tokenize_atomic_words("ho     khan")
    assert [w.text for w in tight] == [w.text for w in loose]
    assert tight[1].start == 3
    assert loose[1].start == 7


def test_repeated_newlines_are_separators() -> None:
    text = "sốt\n\n\nho khan"
    words = tokenize_atomic_words(text)
    assert [w.text for w in words] == ["sốt", "ho", "khan"]
    for word in words:
        assert text[word.start:word.end] == word.text


def test_decomposed_unicode_is_preserved_exactly() -> None:
    text = unicodedata.normalize("NFD", "sốt cao")
    assert text != unicodedata.normalize("NFC", text)
    words = tokenize_atomic_words(text)
    assert len(words) == 2
    for word in words:
        assert text[word.start:word.end] == word.text
    assert "".join(w.text for w in words) == text.replace(" ", "")


def test_segmenter_join_character_stays_word_internal() -> None:
    """Pre-segmented governed sources contain ``_`` inside real words."""
    words = tokenize_atomic_words("đái_tháo_đường nặng")
    assert [w.text for w in words] == ["đái_tháo_đường", "nặng"]
    assert is_atomic_boundary_character("_") is False
    assert is_atomic_boundary_character(",") is True


def test_offsets_are_end_exclusive_and_exact() -> None:
    text = "  sốt cao  "
    for word in tokenize_atomic_words(text):
        assert text[word.start:word.end] == word.text
        assert word.end > word.start


def test_atomic_words_are_indexed_contiguously() -> None:
    words = tokenize_atomic_words("a b , c")
    assert [w.index for w in words] == [0, 1, 2, 3]


def test_tokenization_does_not_depend_on_gold_entities() -> None:
    """Identical text yields identical atomic words regardless of the labels."""
    baseline = tokenize_atomic_words(FAILED_TEXT)
    build_w2ner_grid(FAILED_EXAMPLE_ID, FAILED_TEXT, (FAILED_ENTITY,))
    build_w2ner_grid(FAILED_EXAMPLE_ID, FAILED_TEXT, ())
    build_w2ner_grid(
        FAILED_EXAMPLE_ID, FAILED_TEXT, (DIAGNOSIS_ENTITY, FAILED_ENTITY))
    assert tokenize_atomic_words(FAILED_TEXT) == baseline


def test_no_snapping_and_no_silent_exclusion_for_an_unalignable_entity() -> None:
    """An entity that cannot align raises; it is never moved to make it fit."""
    text = "abcdef"
    entity = EntitySpan(2, 4, "SYMPTOM", "cd")
    with pytest.raises(W2NERError):
        build_w2ner_grid("x", text, (entity,))


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_one_model_word_maps_to_two_atomic_words() -> None:
    _contract, _encoding, projection = _projection()
    gay = next(i for i, w in enumerate(projection.atomic_words) if w.text == "gây")
    roi = next(i for i, w in enumerate(projection.atomic_words) if w.text == "rối")
    assert projection.model_word_index_by_atomic[gay] == 14
    assert projection.model_word_index_by_atomic[roi] == 14
    assert projection.atomic_indices_by_model_word[14] == (gay, roi)
    assert projection.merged_model_word_count >= 1


def test_subtokens_reach_atomic_words_through_their_model_word() -> None:
    _contract, encoding, projection = _projection()
    gay = next(i for i, w in enumerate(projection.atomic_words) if w.text == "gây")
    assert projection.subtoken_indices_by_atomic[gay]
    assert set(projection.subtoken_indices_by_atomic[gay]) <= set(
        encoding.subword_indices_by_word[14])


def test_atomic_words_under_one_model_token_stay_distinguishable() -> None:
    _contract, _encoding, projection = _projection()
    gay = next(i for i, w in enumerate(projection.atomic_words) if w.text == "gây")
    roi = next(i for i, w in enumerate(projection.atomic_words) if w.text == "rối")
    assert projection.subtoken_indices_by_atomic[gay] == (
        projection.subtoken_indices_by_atomic[roi])
    assert projection.atomic_features[gay] != projection.atomic_features[roi]


def test_a_single_atomic_word_model_word_reduces_to_plain_pooling() -> None:
    _contract, _encoding, projection = _projection()
    tim = next(i for i, w in enumerate(projection.atomic_words) if w.text == "tim")
    assert projection.atomic_features[tim] == (0.0, 1.0, 0.0)


def test_projection_output_shape_matches_the_grid() -> None:
    contract, _encoding, projection = _projection()
    assert projection.atomic_word_count == contract.word_count
    assert len(projection.atomic_features) == contract.word_count
    for features in projection.atomic_features:
        assert len(features) == ATOMIC_FEATURE_DIM


def test_relation_head_input_dim_accounts_for_the_atomic_features() -> None:
    assert atomic_relation_head_input_dim(1024) == 1024 + ATOMIC_FEATURE_DIM
    with pytest.raises(E4ContractError):
        atomic_relation_head_input_dim(0)


def test_one_atomic_word_may_span_several_model_words() -> None:
    """VnCoreNLP splits letter/digit runs, so neither surface refines the other."""
    text = "beta1 cao"
    spec = (("beta", 0, 4), ("1", 4, 5), ("cao", 6, 9))
    contract = build_w2ner_batch_contract_from_segmented_words(
        "x", text, (), _model_words(text, spec), max_words=64)
    encoding = prepare_phobert_word_inputs(
        _FakeSlowPhobertTokenizer(), contract.segmented_words, max_length=64)
    projection = build_atomic_projection(
        text, contract.segmented_words, encoding, atomic_words=contract.atomic_words)
    assert [w.text for w in projection.atomic_words] == ["beta1", "cao"]
    assert projection.overlapping_model_word_indices[0] == (0, 1)
    assert projection.multi_model_word_atomic_count == 1
    assert len(projection.subtoken_indices_by_atomic[0]) == 2


def test_projection_fails_loudly_when_an_atomic_word_has_no_model_word() -> None:
    text = "sốt cao"
    spec = (("sốt", 0, 3),)
    contract = build_w2ner_batch_contract_from_segmented_words(
        "x", text, (), _model_words(text, spec), max_words=64)
    encoding = prepare_phobert_word_inputs(
        _FakeSlowPhobertTokenizer(), contract.segmented_words, max_length=64)
    with pytest.raises(E4ContractError, match="overlaps no"):
        build_atomic_projection(
            text, contract.segmented_words, encoding, atomic_words=contract.atomic_words)


def test_slow_tokenizer_path_never_requests_offset_mapping() -> None:
    _contract, encoding, _projection_result = _projection()
    assert encoding.tokenizer_is_fast is False
    assert encoding.consumed_offset_mapping is False
    assert "offset_mapping" not in encoding.model_inputs


def test_model_words_keep_their_join_characters_for_phobert() -> None:
    contract, _encoding, _projection_result = _projection()
    model_texts = [word.model_text for word in contract.segmented_words]
    assert "gây_rối" in model_texts
    assert "tuyến_giáp" in model_texts


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def test_alignment_categories_are_classified_correctly() -> None:
    words = tokenize_atomic_words("aa bb cc")
    assert classify_alignment(words, EntitySpan(0, 2, "SYMPTOM", "aa")) == ALIGNED
    assert classify_alignment(words, EntitySpan(0, 4, "SYMPTOM", "aa b")) == RIGHT_ONLY
    assert classify_alignment(words, EntitySpan(1, 5, "SYMPTOM", "a bb")) == LEFT_ONLY
    assert classify_alignment(words, EntitySpan(1, 4, "SYMPTOM", "a b")) == BOTH


def test_diagnostic_refuses_to_read_internal_test() -> None:
    with pytest.raises(E4DiagnosticError, match="internal_test"):
        run_alignment_diagnostic({"internal_test": "x.jsonl"}, segmenter=None)


def test_diagnostic_reports_a_misaligned_entity_without_repairing_it(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train.jsonl"
    split.write_text(json.dumps({
        "example_id": FAILED_EXAMPLE_ID,
        "source_dataset": "vimedner",
        "text": FAILED_TEXT,
        "entities": [{"start": 85, "end": 102, "target_type": "SYMPTOM",
                      "text": "rối loạn nhịp tim"}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    segmented = " ".join(text for text, _s, _e in FAILED_SEGMENTED_WORDS)
    diagnostic = run_alignment_diagnostic(
        {"train": split}, segmenter=lambda _text: segmented)
    totals = diagnostic.as_dict()["totals"]
    assert totals["entities"] == 1
    assert totals["segmented_misaligned_left_only"] == 1
    assert totals["entities_fixed_by_atomic_words"] == 1
    assert totals["entities_unalignable_after_atomic_words"] == 0
    assert totals["silent_exclusions"] == 0
    assert totals["atomic_projection_violations"] == 0
    assert diagnostic.passed is True

    record = diagnostic.as_dict()["mismatch_examples"][0]
    assert record["example_id"] == FAILED_EXAMPLE_ID
    assert (record["start"], record["end"]) == (85, 102)
    assert record["straddling_model_word"] == "gây_rối"
    assert (record["straddling_model_word_start"],
            record["straddling_model_word_end"]) == (81, 88)
    assert record["fixed_by_atomic_words"] is True


def test_diagnostic_reports_subtoken_statistics_as_unavailable_without_a_tokenizer(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train.jsonl"
    split.write_text(json.dumps(
        {"example_id": "x", "text": "sốt cao", "entities": []}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    totals = run_alignment_diagnostic(
        {"train": split}, segmenter=None).as_dict()["totals"]
    assert totals["phobert_subtoken_statistics_available"] is False
    assert totals["max_phobert_subtoken_count"] == -1
    assert totals["examples_exceeding_max_model_tokens"] == -1


def test_diagnostic_is_deterministic(tmp_path: Path) -> None:
    split = tmp_path / "train.jsonl"
    split.write_text(json.dumps(
        {"example_id": "x", "source_dataset": "s", "text": FAILED_TEXT,
         "entities": [{"start": 85, "end": 102, "target_type": "SYMPTOM",
                       "text": "rối loạn nhịp tim"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    segmented = " ".join(text for text, _s, _e in FAILED_SEGMENTED_WORDS)
    first = run_alignment_diagnostic({"train": split}, segmenter=lambda _t: segmented)
    second = run_alignment_diagnostic({"train": split}, segmenter=lambda _t: segmented)
    assert first.as_dict() == second.as_dict()


# ---------------------------------------------------------------------------
# Contract / schema versioning
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Contract identity under the clean implementation (Audit 0045)
# ---------------------------------------------------------------------------
#
# The checkpoint-compatibility and notebook-ordering tests that used to live here
# exercised the removed implementation: its `phase2-e4-checkpoint-v2` schema, its
# `reject_incompatible_e4_checkpoint` helper and its full-training notebook. All
# three are gone. Their replacements are covered by
# tests/unit/test_e4_clean_training.py; what stays here is the alignment surface,
# which Audit 0043 proved exact and which moved across unchanged.


def test_the_atomic_grid_contract_survived_the_replacement() -> None:
    assert W2NER_CONFIG_VERSION == "phobert-w2ner-v2"
    assert E4_INPUT_CONTRACT_VERSION == "e4-atomic-grid-word-v1"
    assert ATOMIC_PROJECTION_VERSION == "atomic-projection-v1"
    assert ATOMIC_WORD_POLICY_VERSION == "atomic-original-word-v1"


def test_the_checkpoint_schema_was_bumped_past_the_collapsed_implementation() -> None:
    assert E4_CHECKPOINT_SCHEMA_VERSION == "phase2-e4-checkpoint-v3"
