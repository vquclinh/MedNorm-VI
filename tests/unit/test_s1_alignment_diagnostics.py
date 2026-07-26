"""Pre-segmented-source alignment fix and privacy-safe diagnostics (Audit 0026).

Deterministic and offline: no VnCoreNLP, no tokenizer download, no training.
The failing production pattern is reproduced with the governed corpus's own
shapes and a whitespace test tokenizer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mednorm_vi.training.colab_bootstrap import evaluate_full_training_readiness
from mednorm_vi.training.phobert_alignment import (
    BOUNDARY_MERGE_POLICY,
    REASON_GOVERNED_EXCLUSION,
    REASON_NON_SEPARATOR_GAP_INSIDE_WORD,
    REASON_SEGMENTED_SYLLABLE_NOT_FOUND,
    SEGMENTER_JOIN_CHARACTER,
    STAGE_SUBTOKEN_ENCODING,
    STAGE_WORD_MAPPING,
    AlignmentError,
    entities_touched_by_boundary_merge,
    is_legal_syllable_gap,
    map_segmented_words,
    segmented_text_to_words,
)
from mednorm_vi.training.s1_mention_smoke import (
    ENTITY_TYPE_ORDER,
    alignment_diagnostic,
    encode_mention_example_slow,
    governed_exclusion_diagnostic,
    load_governed_exclusions,
    privacy_safe_example_id,
    summarize_alignment_diagnostics,
)

REPO = Path(__file__).resolve().parents[2]
EXCLUSION_POLICY = REPO / "configs" / "training" / "s1_governed_exclusions.yaml"


class WhitespaceTokenizer:
    """Slow-tokenizer stand-in: one piece per whitespace-separated word."""

    is_fast = False
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1

    def tokenize(self, text: str) -> list[str]:
        return [t for t in text.split() if t]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [(abs(hash(t)) % 900) + 10 for t in tokens]


def _coverage():
    from mednorm_vi.data_engine.annotation_coverage import SourceCoverage
    return {"vimq": SourceCoverage(
        source_dataset="vimq", span=True, entity_type=True, assertions=False,
        icd_candidates=False, rxnorm_candidates=False, test_result_pairing=False,
        patient_context=False, relations=False)}


# --- the exact production failure: literal '_' in pre-segmented source text ----

def test_literal_underscore_in_source_text_is_now_mappable() -> None:
    """ViMQ/PhoNER ship ALREADY word-segmented text, so '_' is a real character.

    Before Audit 0026 this raised `non-whitespace gap inside segmented word`,
    which silently discarded ~93% of ViMQ examples.
    """
    original = "em bi dau_dau va ho_khan nhieu"
    words = map_segmented_words(original, segmented_text_to_words(original))
    assert [w.model_text for w in words] == [
        "em", "bi", "dau_dau", "va", "ho_khan", "nhieu"]
    # Every span must reproduce the original substring exactly, underscore included.
    for word in words:
        assert original[word.original_start:word.original_end] == word.model_text


def test_entity_spans_containing_an_underscore_align_and_label_correctly() -> None:
    """A ViMQ entity whose gold span includes the literal underscore."""
    text = "toi bi viem_hong nang"
    example = {
        "example_id": "vimq:train:000069", "source_dataset": "vimq", "text": text,
        "entities": [{"start": 7, "end": 16, "text": "viem_hong",
                      "target_type": "DIAGNOSIS", "mapping_status": "MAP_EXACT"}],
    }
    assert text[7:16] == "viem_hong"
    feature = encode_mention_example_slow(
        example, WhitespaceTokenizer(), coverage_by_source=_coverage(),
        max_length=32, segmented_text=text)
    diagnosis = ENTITY_TYPE_ORDER.index("DIAGNOSIS")
    labelled = [i for i, row in enumerate(feature["labels"]) if row[diagnosis]]
    assert labelled, "the entity produced no supervision"
    assert feature["boundary_merge_masked_word_count"] == 0
    # Exactly the entity's own word is labelled - not its neighbours.
    assert len(labelled) == 1


def test_segmenter_joined_words_still_map_when_the_source_uses_spaces() -> None:
    """The normal case is unchanged: RDRSegmenter joined two spaced syllables."""
    original = "benh nhan dau dau nhieu"
    words = map_segmented_words(original, segmented_text_to_words("benh_nhan dau_dau nhieu"))
    assert [(w.original_start, w.original_end) for w in words] == [(0, 9), (10, 17), (18, 23)]
    assert original[0:9] == "benh nhan"
    assert original[10:17] == "dau dau"


@pytest.mark.parametrize("gap,legal", [
    (" ", True), ("  ", True), ("\n", True), ("\t", True),
    (SEGMENTER_JOIN_CHARACTER, True), ("_ ", True), ("", True),
    (",", False), ("-", False), ("a", False), (" , ", False),
])
def test_only_whitespace_and_the_join_character_may_separate_syllables(gap, legal) -> None:
    assert is_legal_syllable_gap(gap) is legal


def test_a_genuinely_unmappable_word_still_raises_with_a_reason_code() -> None:
    """The fix must not turn real drift into silent success."""
    with pytest.raises(AlignmentError) as excinfo:
        map_segmented_words("benh nhan khoe", segmented_text_to_words("khong_co_o_day"))
    assert excinfo.value.reason_code == REASON_SEGMENTED_SYLLABLE_NOT_FOUND


def test_punctuation_between_syllables_is_still_illegal() -> None:
    with pytest.raises(AlignmentError) as excinfo:
        map_segmented_words("dau, dau nhieu", segmented_text_to_words("dau_dau nhieu"))
    assert excinfo.value.reason_code == REASON_NON_SEPARATOR_GAP_INSIDE_WORD


# --- boundary merge: mask, never discard the example --------------------------

def test_boundary_merge_masks_the_straddling_word_instead_of_dropping_the_example() -> None:
    """Two adjacent entities separated by one space, merged by the segmenter.

    This is the geometry present in the governed smoke subset. Previously the
    whole example was discarded; now only the ambiguous supervision is masked.
    """
    text = "sot cao ho nhieu ngay"
    example = {
        "example_id": "vimedner:train:train-002", "source_dataset": "vimq", "text": text,
        "entities": [
            {"start": 0, "end": 7, "text": "sot cao", "target_type": "SYMPTOM",
             "mapping_status": "MAP_EXACT"},
            {"start": 8, "end": 10, "text": "ho", "target_type": "SYMPTOM",
             "mapping_status": "MAP_EXACT"},
            {"start": 11, "end": 21, "text": "nhieu ngay", "target_type": "SYMPTOM",
             "mapping_status": "MAP_EXACT"},
        ],
    }
    # The segmenter merges "cao" + "ho" across the [0,7)/[8,10) boundary.
    feature = encode_mention_example_slow(
        example, WhitespaceTokenizer(), coverage_by_source=_coverage(),
        max_length=32, segmented_text="sot cao_ho nhieu_ngay")
    assert feature["boundary_merge_masked_word_count"] == 1
    assert feature["boundary_merge_affected_entity_count"] >= 2
    # The example survives and the UNAFFECTED entity keeps its supervision.
    symptom = ENTITY_TYPE_ORDER.index("SYMPTOM")
    assert any(row[symptom] for row in feature["labels"]), "all supervision was lost"
    # No masked position may carry a label.
    for row, keep in zip(feature["labels"], feature["label_mask"], strict=True):
        if not keep:
            assert not any(row), "a masked token still carries a label"


def test_boundary_merge_helper_reports_words_and_affected_entities() -> None:
    words = map_segmented_words("sot cao ho nhieu", segmented_text_to_words("sot cao_ho nhieu"))
    straddling, affected = entities_touched_by_boundary_merge(words, [(0, 7), (8, 10)])
    assert straddling == (1,)
    assert affected == (0, 1)


def test_policy_constant_documents_the_masking_behaviour() -> None:
    assert BOUNDARY_MERGE_POLICY == "mask_straddling_word_and_affected_entity_subtokens"


# --- privacy-safe diagnostics -------------------------------------------------

def _example() -> dict:
    return {"example_id": "vimq:dev:000958", "source_dataset": "vimq",
            "text": "benh nhan dau_bung nhieu", "entities": []}


def test_privacy_safe_example_id_is_a_stable_non_reversible_handle() -> None:
    handle = privacy_safe_example_id("vimq:dev:000958")
    assert handle == hashlib.sha256(b"vimq:dev:000958").hexdigest()[:16]
    assert len(handle) == 16 and handle == privacy_safe_example_id("vimq:dev:000958")
    assert privacy_safe_example_id("vimq:dev:000959") != handle


def test_diagnostic_records_only_the_permitted_privacy_safe_fields() -> None:
    example = _example()
    diagnostic = alignment_diagnostic(
        example, split="validation", stage=STAGE_WORD_MAPPING,
        error=AlignmentError("boom", REASON_NON_SEPARATOR_GAP_INSIDE_WORD))
    payload = diagnostic.as_dict()
    assert set(payload) == {"source", "split", "privacy_safe_example_id", "stage",
                            "reason_code", "exception_type"}
    assert payload["source"] == "vimq"
    assert payload["split"] == "validation"
    assert payload["stage"] == STAGE_WORD_MAPPING
    assert payload["reason_code"] == REASON_NON_SEPARATOR_GAP_INSIDE_WORD
    assert payload["exception_type"] == "AlignmentError"
    assert diagnostic.expected is False


def test_no_raw_text_or_verbatim_id_reaches_a_diagnostic() -> None:
    example = _example()
    serialized = json.dumps(alignment_diagnostic(
        example, split="validation", stage=STAGE_SUBTOKEN_ENCODING,
        error=AlignmentError("boom", REASON_SEGMENTED_SYLLABLE_NOT_FOUND)).as_dict())
    assert example["text"] not in serialized
    assert "dau_bung" not in serialized
    assert example["example_id"] not in serialized       # hashed, never verbatim
    for fragment in example["text"].split():
        assert fragment not in serialized


# --- counter reconciliation and readiness -------------------------------------

def test_unexpected_failures_and_governed_exclusions_use_different_counters() -> None:
    unexpected = alignment_diagnostic(
        _example(), split="train", stage=STAGE_WORD_MAPPING,
        error=AlignmentError("boom", REASON_SEGMENTED_SYLLABLE_NOT_FOUND))
    excluded = governed_exclusion_diagnostic(_example(), split="train")
    summary = summarize_alignment_diagnostics([unexpected, excluded])
    assert summary["unalignable_example_count"] == 1
    assert summary["governed_exclusion_count"] == 1
    assert summary["reason_code_counts"] == {
        REASON_GOVERNED_EXCLUSION: 1, REASON_SEGMENTED_SYLLABLE_NOT_FOUND: 1}
    assert len(summary["unalignable_examples"]) == 1
    assert len(summary["governed_exclusions"]) == 1
    assert excluded.expected is True


def test_governed_exclusions_never_appear_in_the_blocking_counter() -> None:
    summary = summarize_alignment_diagnostics(
        [governed_exclusion_diagnostic(_example(), split="train") for _ in range(3)])
    assert summary["unalignable_example_count"] == 0
    assert summary["governed_exclusion_count"] == 3


def _ready() -> dict:
    return {
        "production_segmentation": True, "tokenizer_equivalence_examples": 12,
        "tokenizer_equivalence_failures": 0, "unalignable_example_count": 0,
        "dependency_restart_completed": True, "numpy_abi_preflight_passed": True,
        "s1_dependency_closure_verified": True, "train_loss_finite": True,
        "backward_completed": True, "optimizer_step_completed": True,
        "validation_completed": True, "checkpoint_saved": True,
        "checkpoint_reloaded": True,
    }


def test_unexpected_unalignable_examples_still_block_readiness() -> None:
    assert evaluate_full_training_readiness(_ready()) is True
    assert evaluate_full_training_readiness(
        _ready() | {"unalignable_example_count": 1}) is False


def test_governed_exclusions_alone_do_not_block_readiness() -> None:
    """An explicitly tracked exclusion is a decision, not an unresolved failure."""
    evidence = _ready() | {"governed_exclusion_count": 2}
    assert evaluate_full_training_readiness(evidence) is True


# --- the tracked governed-exclusion policy ------------------------------------

def test_exclusion_policy_is_tracked_and_currently_empty() -> None:
    """Audit 0026 found an implementation bug, not invalid data: nothing is excluded."""
    assert load_governed_exclusions(EXCLUSION_POLICY) == {}


def test_exclusion_entries_require_a_justification(tmp_path: Path) -> None:
    bad = tmp_path / "exclusions.yaml"
    bad.write_text("exclusions:\n  - privacy_safe_example_id: abcdef0123456789\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="justification"):
        load_governed_exclusions(bad)


def test_exclusion_policy_carries_no_raw_text_or_verbatim_ids() -> None:
    text = EXCLUSION_POLICY.read_text(encoding="utf-8")
    assert "privacy_safe_example_id" in text
    for verbatim in ("vimq:train:", "vimq:dev:", "vimedner:train:", "vietmed_ner:"):
        assert verbatim not in text, f"policy leaks a verbatim example id: {verbatim}"
