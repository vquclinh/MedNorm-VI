"""Slow-tokenizer (PhoBERT/ViHealthBERT-Word) alignment tests (Audit 0022).

No model, tokenizer, or VnCoreNLP download: a deterministic fake SLOW tokenizer
plus synthetic Vietnamese text. Proves the S1 encoder no longer needs
``offset_mapping``/``word_ids()`` and that supervision semantics are preserved.
"""

from __future__ import annotations

import pytest

from mednorm_vi.data_engine.annotation_coverage import SourceCoverage
from mednorm_vi.training.phobert_alignment import (
    ALIGNMENT_BACKEND,
    AlignmentError,
    align_subtokens,
    classify_truncated_entities,
    count_truncated_entities,
    describe_backend,
    find_boundary_violations,
    map_segmented_words,
    segmented_text_to_words,
    verify_tokenizer_equivalence,
)
from mednorm_vi.training.s1_mention_smoke import (
    ENTITY_TYPE_ORDER,
    encode_mention_example_slow,
)


class FakeSlowTokenizer:
    """Deterministic stand-in for PhobertTokenizer (slow; no offset mapping)."""

    is_fast = False
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1
    vocab_size = 64000

    def __init__(self, split_long: bool = True) -> None:
        self.split_long = split_long

    def tokenize(self, text: str) -> list[str]:
        if self.split_long and len(text) > 6:
            return [text[:3] + "@@", text[3:]]
        return [text]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [1000 + (hash(t) % 5000) for t in tokens]

    # Deliberately NOT provided: return_offsets_mapping, word_ids,
    # token_to_chars, char_to_token — the backend must not need them.


COVERAGE = {
    "vietmed_ner": SourceCoverage("vietmed_ner", span=True, entity_type=True),
    "vimedner": SourceCoverage("vimedner", span=True, entity_type=True),
}
MED = ENTITY_TYPE_ORDER.index("MEDICATION")


def _example(text: str, entities: list[tuple[str, int, int, str]],
             source: str = "vietmed_ner") -> dict:
    return {
        "example_id": "ex-1", "source_dataset": source, "text": text,
        "entities": [
            {"text": t, "start": s, "end": e, "target_type": ty,
             "mapping_status": "MAP_APPROXIMATE" if source == "vietmed_ner" else "MAP_EXACT"}
            for t, s, e, ty in entities
        ],
    }


# --- slow tokenizer is accepted, fast APIs unused -----------------------------

def test_slow_tokenizer_is_accepted() -> None:
    tok = FakeSlowTokenizer()
    assert tok.is_fast is False
    info = describe_backend(tok)
    assert info["tokenizer_is_fast"] is False
    assert info["alignment_backend"] == ALIGNMENT_BACKEND


def test_backend_does_not_use_fast_only_apis() -> None:
    tok = FakeSlowTokenizer()
    for attr in ("word_ids", "token_to_chars", "char_to_token"):
        assert not hasattr(tok, attr)
    text = "Bệnh nhân ho"
    feature = encode_mention_example_slow(
        _example(text, []), tok, coverage_by_source=COVERAGE, max_length=32)
    assert len(feature["input_ids"]) == len(feature["labels"])


# --- word -> original character mapping --------------------------------------

def test_single_word_entity() -> None:
    text = "Uống paracetamol"
    words = map_segmented_words(text, segmented_text_to_words(text))
    assert [(w.original_start, w.original_end) for w in words] == [(0, 4), (5, 16)]
    assert text[words[1].original_start:words[1].original_end] == "paracetamol"


def test_multi_syllable_underscore_word() -> None:
    text = "Bệnh nhân bị đái tháo đường"
    seg = "Bệnh_nhân bị đái_tháo_đường"
    words = map_segmented_words(text, segmented_text_to_words(seg))
    assert text[words[0].original_start:words[0].original_end] == "Bệnh nhân"
    assert text[words[2].original_start:words[2].original_end] == "đái tháo đường"


def test_repeated_word_occurrences_map_monotonically() -> None:
    text = "ho nhiều rồi ho khan"
    words = map_segmented_words(text, segmented_text_to_words(text))
    spans = [(w.original_start, w.original_end) for w in words]
    assert spans[0] == (0, 2) and spans[3] == (13, 15)   # distinct "ho" spans
    assert spans == sorted(spans)                        # monotonic


def test_repeated_whitespace_tabs_and_newlines() -> None:
    text = "sốt   cao\tvà\nho"
    words = map_segmented_words(text, segmented_text_to_words(text))
    for w in words:
        assert text[w.original_start:w.original_end] == w.model_text.replace("_", " ")


def test_punctuation_preserved() -> None:
    text = "Chẩn đoán: viêm phổi."
    seg = "Chẩn_đoán : viêm_phổi ."
    words = map_segmented_words(text, segmented_text_to_words(seg))
    assert text[words[0].original_start:words[0].original_end] == "Chẩn đoán"
    assert text[words[2].original_start:words[2].original_end] == "viêm phổi"


def test_missing_word_fails_fast() -> None:
    with pytest.raises(AlignmentError):
        map_segmented_words("chỉ có văn bản này", ["không_tồn_tại"])


def test_ambiguous_gap_inside_word_fails_fast() -> None:
    # syllables separated by non-whitespace must not be merged into one word
    with pytest.raises(AlignmentError):
        map_segmented_words("đái, tháo đường", ["đái_tháo_đường"])


# --- subtoken alignment -------------------------------------------------------

def test_bpe_word_split_into_multiple_pieces_shares_word_span() -> None:
    text = "uống paracetamol"
    words = map_segmented_words(text, segmented_text_to_words(text))
    result = align_subtokens(words, FakeSlowTokenizer(), max_length=32)
    body = [s for s in result.subtokens if not s.is_special]
    med_pieces = [s for s in body if s.source_word_index == 1]
    assert len(med_pieces) == 2                                   # split into 2 BPE pieces
    assert {(p.original_start, p.original_end) for p in med_pieces} == {(5, 16)}
    assert med_pieces[0].is_continuation is False
    assert med_pieces[1].is_continuation is True


def test_special_tokens_present_and_unsupervised() -> None:
    text = "ho"
    words = map_segmented_words(text, segmented_text_to_words(text))
    result = align_subtokens(words, FakeSlowTokenizer(), max_length=16)
    assert result.subtokens[0].is_special and result.subtokens[-1].is_special
    assert result.subtokens[0].source_word_index == -1
    assert (result.subtokens[0].original_start, result.subtokens[0].original_end) == (0, 0)


def test_adjacent_entities_get_distinct_labels() -> None:
    text = "sốt ho"
    ex = _example(text, [("sốt", 0, 3, "SYMPTOM"), ("ho", 4, 6, "SYMPTOM")])
    feature = encode_mention_example_slow(
        ex, FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=32)
    sym = ENTITY_TYPE_ORDER.index("SYMPTOM")
    supervised = [i for i, lab in enumerate(feature["labels"]) if lab[sym] == 1]
    assert len(supervised) == 2


def test_text_with_no_entities_has_no_positive_labels() -> None:
    feature = encode_mention_example_slow(
        _example("bệnh nhân ổn định", []), FakeSlowTokenizer(),
        coverage_by_source=COVERAGE, max_length=32)
    assert all(not any(lab) for lab in feature["labels"])
    assert sum(feature["label_mask"]) > 0                 # tokens still supervised (negatives)


# --- truncation policy --------------------------------------------------------

def test_entity_fully_retained_is_classified_retained() -> None:
    text = "uống paracetamol"
    words = map_segmented_words(text, segmented_text_to_words(text))
    result = align_subtokens(words, FakeSlowTokenizer(), max_length=32)
    report = classify_truncated_entities(words, result, [(5, 16)])
    assert report.fully_retained == (0,)
    assert report.fully_dropped_count == 0 and report.partially_truncated_count == 0
    assert report.truncated_entity_count == 0


def test_entity_fully_removed_is_classified_dropped() -> None:
    text = " ".join(["từ"] * 30) + " paracetamol"
    start = text.index("paracetamol")
    words = map_segmented_words(text, segmented_text_to_words(text))
    result = align_subtokens(words, FakeSlowTokenizer(), max_length=8)
    report = classify_truncated_entities(words, result, [(start, start + 11)])
    assert report.fully_dropped == (0,)
    assert report.partially_truncated_count == 0
    assert report.truncated_entity_count == 1


def test_entity_partially_cut_is_masked_not_partially_supervised() -> None:
    """A multi-word entity half-lost to truncation keeps NO labels."""
    # entity spans two words; truncation keeps only the first.
    text = "đái tháo cao huyết áp " + " ".join(["từ"] * 30)
    ent = (0, len("đái tháo"))
    ex = _example(text, [("đái tháo", ent[0], ent[1], "SYMPTOM")])
    words = map_segmented_words(text, segmented_text_to_words(text))
    # max_length chosen so only the first word's pieces survive
    result = align_subtokens(words, FakeSlowTokenizer(split_long=False), max_length=3)
    report = classify_truncated_entities(words, result, [ent])
    assert report.partially_truncated == (0,)
    assert report.fully_dropped_count == 0
    feature = encode_mention_example_slow(
        ex, FakeSlowTokenizer(split_long=False), coverage_by_source=COVERAGE, max_length=3)
    sym = ENTITY_TYPE_ORDER.index("SYMPTOM")
    assert all(lab[sym] == 0 for lab in feature["labels"]), "partial labels must be masked"
    assert feature["partially_truncated_entity_count"] == 1
    assert feature["truncated_entity_count"] >= 1


def test_multiple_entities_only_one_cut() -> None:
    text = "sốt " + " ".join(["từ"] * 30) + " paracetamol"
    med_start = text.index("paracetamol")
    ex = _example(text, [("sốt", 0, 3, "SYMPTOM"),
                         ("paracetamol", med_start, med_start + 11, "MEDICATION")])
    feature = encode_mention_example_slow(
        ex, FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=10)
    sym = ENTITY_TYPE_ORDER.index("SYMPTOM")
    assert any(lab[sym] == 1 for lab in feature["labels"])      # retained entity keeps labels
    assert all(lab[MED] == 0 for lab in feature["labels"])      # cut entity has none
    assert feature["fully_dropped_entity_count"] == 1
    assert feature["partially_truncated_entity_count"] == 0


def test_special_and_padding_positions_remain_ignored_after_truncation() -> None:
    text = "paracetamol " + " ".join(["từ"] * 30)
    ex = _example(text, [("paracetamol", 0, 11, "MEDICATION")])
    feature = encode_mention_example_slow(
        ex, FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=10)
    assert feature["label_mask"][0] == 0 and feature["label_mask"][-1] == 0
    assert not any(feature["labels"][0]) and not any(feature["labels"][-1])


def test_truncation_classification_is_deterministic() -> None:
    text = "đái tháo cao huyết áp " + " ".join(["từ"] * 20)
    ex = _example(text, [("đái tháo", 0, 8, "SYMPTOM")])
    a = encode_mention_example_slow(ex, FakeSlowTokenizer(split_long=False),
                                    coverage_by_source=COVERAGE, max_length=4)
    b = encode_mention_example_slow(ex, FakeSlowTokenizer(split_long=False),
                                    coverage_by_source=COVERAGE, max_length=4)
    assert a == b


# --- tokenizer equivalence helper --------------------------------------------

class CallableSlowTokenizer(FakeSlowTokenizer):
    """Adds a __call__ so the helper can tokenize the full sentence."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        assert add_special_tokens is False, "equivalence must use add_special_tokens=False"
        pieces: list[str] = []
        for word in text.split():
            pieces.extend(self.tokenize(word))
        return {"input_ids": self.convert_tokens_to_ids(pieces)}


class NonDecomposableTokenizer(CallableSlowTokenizer):
    """Whole-sentence tokenization differs from per-word tokenization."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        return {"input_ids": [42]}          # deliberately inconsistent


def test_tokenizer_equivalence_passes_for_decomposable_tokenizer() -> None:
    text = "uống paracetamol mỗi ngày"
    words = map_segmented_words(text, segmented_text_to_words(text))
    report = verify_tokenizer_equivalence(words, CallableSlowTokenizer())
    assert report["equivalent"] is True
    assert report["word_count"] == 4 and report["token_count"] > 0


def test_tokenizer_equivalence_fails_fast_on_mismatch() -> None:
    text = "uống paracetamol"
    words = map_segmented_words(text, segmented_text_to_words(text))
    with pytest.raises(AlignmentError, match="equivalence failed"):
        verify_tokenizer_equivalence(words, NonDecomposableTokenizer())


class WhitespaceSlowTokenizer(FakeSlowTokenizer):
    """No __call__; tokenize() splits on whitespace so it is per-word decomposable."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def test_tokenizer_equivalence_falls_back_to_tokenize_when_not_callable() -> None:
    text = "uống paracetamol"
    words = map_segmented_words(text, segmented_text_to_words(text))
    # No __call__ -> the helper falls back to tokenize() on the full sentence.
    report = verify_tokenizer_equivalence(words, WhitespaceSlowTokenizer())
    assert report["equivalent"] is True
    assert report["word_count"] == 2


def test_truncation_without_entity_loss() -> None:
    text = "paracetamol " + " ".join(["từ"] * 40)
    start, end = 0, len("paracetamol")
    ex = _example(text, [("paracetamol", start, end, "MEDICATION")])
    feature = encode_mention_example_slow(
        ex, FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=12)
    assert feature["truncated"] is True
    assert feature["truncated_entity_count"] == 0          # entity survives at the front
    assert any(lab[MED] == 1 for lab in feature["labels"])


def test_truncation_cutting_an_entity_is_counted() -> None:
    text = " ".join(["từ"] * 40) + " paracetamol"
    start = text.index("paracetamol")
    ex = _example(text, [("paracetamol", start, start + 11, "MEDICATION")])
    feature = encode_mention_example_slow(
        ex, FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=10)
    assert feature["truncated"] is True
    assert feature["truncated_entity_count"] == 1          # explicitly counted, not mislabeled
    assert all(lab[MED] == 0 for lab in feature["labels"])


# --- boundary-merge policy ----------------------------------------------------

def test_segmentation_merge_crossing_entity_boundary_is_masked_not_mislabeled() -> None:
    """Audit 0026: the straddling word is masked; the example is NOT discarded.

    Previously this raised and threw away every other entity in the example. The
    ambiguous supervision is now removed (labels zeroed, label_mask 0) and counted,
    which is the same rule PARTIAL_TRUNCATION_POLICY applies to truncation.
    """
    text = "đau đầu nhiều"
    # entity covers only "đau", but segmentation merges "đau_đầu" -> straddles boundary
    words = map_segmented_words(text, segmented_text_to_words("đau_đầu nhiều"))
    assert find_boundary_violations(words, [(0, 3)]) == [0]
    ex = _example(text, [("đau", 0, 3, "SYMPTOM")])
    feature = encode_mention_example_slow(
        ex, FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=32,
        segmented_text="đau_đầu nhiều")
    assert feature["boundary_merge_masked_word_count"] == 1
    assert feature["boundary_merge_affected_entity_count"] == 1
    symptom = ENTITY_TYPE_ORDER.index("SYMPTOM")
    # No token may carry the ambiguous label, and no masked token may carry any label.
    assert all(lab[symptom] == 0 for lab in feature["labels"])
    for lab, keep in zip(feature["labels"], feature["label_mask"], strict=True):
        if not keep:
            assert not any(lab)


def test_word_fully_inside_entity_is_not_a_violation() -> None:
    text = "đái tháo đường nặng"
    words = map_segmented_words(text, segmented_text_to_words("đái_tháo_đường nặng"))
    assert find_boundary_violations(words, [(0, 14)]) == []


# --- determinism, invariants, VietMed policy ----------------------------------

def test_alignment_is_deterministic() -> None:
    ex = _example("uống paracetamol mỗi ngày",
                  [("paracetamol", 5, 16, "MEDICATION")])
    a = encode_mention_example_slow(ex, FakeSlowTokenizer(),
                                    coverage_by_source=COVERAGE, max_length=32)
    b = encode_mention_example_slow(ex, FakeSlowTokenizer(),
                                    coverage_by_source=COVERAGE, max_length=32)
    assert a == b


def test_entity_offset_invariant_enforced() -> None:
    text = "uống paracetamol"
    bad = _example(text, [("aspirin", 5, 16, "MEDICATION")])   # text mismatch
    with pytest.raises(ValueError, match="offset/text invariant"):
        encode_mention_example_slow(bad, FakeSlowTokenizer(),
                                    coverage_by_source=COVERAGE, max_length=32)


def test_vietmed_masks_remain_span_and_type_only() -> None:
    feature = encode_mention_example_slow(
        _example("uống paracetamol", [("paracetamol", 5, 16, "MEDICATION")]),
        FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=32)
    assert feature["loss_mask"] == {
        "span": True, "entity_type": True,
        "assertions": False, "icd_candidates": False, "rxnorm_candidates": False,
    }


def test_dimensions_match_across_ids_labels_and_masks() -> None:
    feature = encode_mention_example_slow(
        _example("sốt cao và ho", [("sốt", 0, 3, "SYMPTOM")]),
        FakeSlowTokenizer(), coverage_by_source=COVERAGE, max_length=32)
    n = len(feature["input_ids"])
    assert len(feature["attention_mask"]) == n
    assert len(feature["labels"]) == n
    assert len(feature["label_mask"]) == n
    assert all(len(lab) == len(ENTITY_TYPE_ORDER) for lab in feature["labels"])


def test_count_truncated_entities_helper() -> None:
    text = "uống paracetamol"
    words = map_segmented_words(text, segmented_text_to_words(text))
    result = align_subtokens(words, FakeSlowTokenizer(), max_length=32)
    assert count_truncated_entities(result, [(5, 16)]) == 0
    assert count_truncated_entities(result, [(100, 110)]) == 1


# --- adversarial: removing the assertion alone would not work ------------------

def test_removing_is_fast_assertion_alone_would_fail() -> None:
    """A slow tokenizer cannot satisfy the old fast-only encoder path.

    Proves the fix required a real alignment backend, not just deleting the
    `assert tokenizer.is_fast` line: calling the slow tokenizer the old way
    raises, because it accepts no `return_offsets_mapping` kwarg.
    """
    from mednorm_vi.training.s1_mention_smoke import encode_mention_example
    tok = FakeSlowTokenizer()
    with pytest.raises(TypeError):
        # the old path calls tokenizer(text, return_offsets_mapping=True, ...)
        encode_mention_example(
            _example("uống paracetamol", []), tok,
            coverage_by_source=COVERAGE, max_length=32)
