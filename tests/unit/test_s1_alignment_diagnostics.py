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
    REASON_EMPTY_SEGMENTED_WORD,
    REASON_GOVERNED_EXCLUSION,
    REASON_NON_SEPARATOR_GAP_INSIDE_WORD,
    REASON_SEGMENTED_SYLLABLE_NOT_FOUND,
    REASON_SEGMENTER_ALTERED_CHARACTERS,
    SEGMENTATION_SOURCE_PRE_SEGMENTED,
    SEGMENTATION_SOURCE_SEGMENTER,
    SEGMENTER_JOIN_CHARACTER,
    STAGE_SUBTOKEN_ENCODING,
    STAGE_WORD_MAPPING,
    AlignmentError,
    entities_touched_by_boundary_merge,
    is_legal_syllable_gap,
    looks_pre_segmented,
    map_segmented_words,
    resolve_segmented_text,
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
    assert excinfo.value.reason_code == REASON_SEGMENTER_ALTERED_CHARACTERS


def test_a_dropped_punctuation_character_is_still_a_failure() -> None:
    """Separators may move; a real character may never disappear."""
    with pytest.raises(AlignmentError) as excinfo:
        map_segmented_words("dau, dau nhieu", segmented_text_to_words("dau_dau nhieu"))
    assert excinfo.value.reason_code == REASON_SEGMENTER_ALTERED_CHARACTERS


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


# --- Audit 0027: the two real smoke-v2 word-mapping failures -------------------
#
# Both patterns are reproduced structurally, using the SHAPE of the governed
# examples (never their text, and never their verbatim ids). Neither test
# references a privacy-safe id, so nothing here can degrade into a special case.

def test_standalone_join_token_is_mapped_not_treated_as_empty() -> None:
    """Real failure 1 (vimq / validation / EMPTY_SEGMENTED_WORD).

    RDRSegmenter re-segmenting ALREADY-segmented text splits the join character
    off as its own token, so the word list contains a bare "_".
    """
    original = "Co_the bo_sung vitamin cho tre khong ?"
    shredded = original.replace("_", " _ ")          # what the segmenter emitted
    words = map_segmented_words(original, segmented_text_to_words(shredded))
    # The separator-only tokens carry no characters, so they contribute no words.
    assert [w.model_text for w in words] == [
        "Co", "the", "bo", "sung", "vitamin", "cho", "tre", "khong", "?"]
    for word in words:
        assert original[word.original_start:word.original_end] == word.model_text


def test_segmenter_compound_merge_across_spaces_is_mapped() -> None:
    """Real failure 2 (vimedner / train / SEGMENTED_SYLLABLE_NOT_FOUND).

    The segmenter pulled a spaced hyphen compound together, so no emitted
    syllable existed verbatim in the original text.
    """
    original = "doan noi tam vi - thuc quan co the gay nuot nghen ."
    merged = "doan_noi tam_vi-thuc_quan co_the gay nuot_nghen ."
    words = map_segmented_words(original, segmented_text_to_words(merged))
    compound = next(w for w in words if "-" in w.model_text)
    # The word maps back onto the ORIGINAL spacing, exactly.
    assert original[compound.original_start:compound.original_end] == "tam vi - thuc quan"
    assert compound.model_text == "tam_vi-thuc_quan"


def test_both_real_failure_shapes_encode_end_to_end() -> None:
    """Both patterns now produce features instead of unalignable examples."""
    cases = [
        # (original text, what the segmenter returns, entity span)
        ("Co_the bo_sung vitamin cho tre", "Co _ the bo _ sung vitamin cho tre", (7, 14)),
        ("tam vi - thuc quan dau", "tam_vi-thuc_quan dau", (0, 18)),
    ]
    for text, segmented, (start, end) in cases:
        example = {
            "example_id": "synthetic", "source_dataset": "vimq", "text": text,
            "entities": [{"start": start, "end": end, "text": text[start:end],
                          "target_type": "DIAGNOSIS", "mapping_status": "MAP_EXACT"}],
        }
        feature = encode_mention_example_slow(
            example, WhitespaceTokenizer(), coverage_by_source=_coverage(),
            max_length=64, segmented_text=segmented)
        diagnosis = ENTITY_TYPE_ORDER.index("DIAGNOSIS")
        assert any(row[diagnosis] for row in feature["labels"]), text
        for row, keep in zip(feature["labels"], feature["label_mask"], strict=True):
            if not keep:
                assert not any(row)


# --- the general policy, not per-example patches ------------------------------

@pytest.mark.parametrize("text,pre_segmented", [
    ("Co_the bo_sung vitamin", True),          # ViMQ / PhoNER shape
    ("benh nhan dau bung", False),             # raw clinical text
    ("tam vi - thuc quan", False),             # spaced punctuation only
    ("a_1 b", True),                           # digits count as word characters
    ("_ leading separator", False),            # a bare separator is not segmentation
    ("trailing _", False),
    ("double __ join", False),                 # '_' is not a word character
])
def test_pre_segmented_detection_is_structural(text, pre_segmented) -> None:
    assert looks_pre_segmented(text) is pre_segmented


def test_pre_segmented_text_is_used_verbatim_and_raw_text_is_segmented() -> None:
    """One policy covers both source kinds without any per-source table."""
    calls = []

    def segmenter(text):
        calls.append(text)
        return text.replace(" ", "_")

    segmented, source = resolve_segmented_text("Co_the bo_sung", segmenter)
    assert source == SEGMENTATION_SOURCE_PRE_SEGMENTED
    assert segmented == "Co_the bo_sung"
    assert calls == [], "already-segmented text must not be re-segmented"

    segmented, source = resolve_segmented_text("benh nhan dau", segmenter)
    assert source == SEGMENTATION_SOURCE_SEGMENTER
    assert segmented == "benh_nhan_dau"
    assert calls == ["benh nhan dau"]


def test_alignment_contains_no_example_specific_special_cases() -> None:
    """No privacy-safe id or dataset name may steer the alignment backend.

    Dataset names may appear in prose explaining WHY a source is pre-segmented;
    they must never appear in executable code, which would make the fix a
    per-corpus patch instead of a general policy.
    """
    import ast
    import io
    import tokenize

    path = REPO / "src" / "mednorm_vi" / "training" / "phobert_alignment.py"
    source = path.read_text(encoding="utf-8")
    for handle in ("6c3b519cd0100bec", "c06995a162f6eb76"):
        assert handle not in source, "the backend references a specific example"

    # Strip comments and docstrings, leaving only executable code.
    code_lines = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        code_lines.append(token.string)
    executable = " ".join(code_lines).lower()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            executable = executable.replace(node.value.lower(), " ")
    for dataset in ("vimq", "vimedner", "vietmed", "phoner"):
        assert dataset not in executable, f"backend special-cases {dataset}"


# --- monotonic offset reconstruction ------------------------------------------

@pytest.mark.parametrize("original,segmented", [
    ("a b c d", "a_b c_d"),                        # plain merge
    ("a b c d", "a b c d"),                        # identity
    ("a_b c_d", "a _ b c _ d"),                    # shredded join characters
    ("x - y z", "x-y_z"),                          # punctuation pulled together
    ("p  q\nr", "p_q r"),                          # repeated / newline whitespace
    ("ab ab ab", "ab_ab ab"),                      # repeated identical surface forms
])
def test_offsets_are_monotonic_and_reconstruct_the_original(original, segmented) -> None:
    words = map_segmented_words(original, segmented_text_to_words(segmented))
    assert words
    for word in words:
        assert 0 <= word.original_start < word.original_end <= len(original)
        # every span is a real slice of the ORIGINAL text
        assert original[word.original_start:word.original_end].strip()
    for left, right in zip(words, words[1:], strict=False):
        assert left.original_end <= right.original_start, "spans overlap or go backwards"
    # every non-separator character of the original is covered exactly once
    covered = "".join(
        original[w.original_start:w.original_end] for w in words)
    strip = str.maketrans("", "", " \t\n_")
    assert covered.translate(strip) == original.translate(strip)


def test_segmentation_that_produces_only_separators_fails() -> None:
    with pytest.raises(AlignmentError) as excinfo:
        map_segmented_words("benh nhan", segmented_text_to_words("_ _ _"))
    assert excinfo.value.reason_code == REASON_EMPTY_SEGMENTED_WORD


def test_structural_mismatch_diagnostics_carry_no_characters() -> None:
    """The failure message locates the divergence without quoting the text."""
    with pytest.raises(AlignmentError) as excinfo:
        map_segmented_words("benh nhan khoe manh", segmented_text_to_words("benh nhon"))
    message = str(excinfo.value)
    assert excinfo.value.reason_code == REASON_SEGMENTER_ALTERED_CHARACTERS
    assert "unicode category" in message and "offset" in message
    for fragment in ("nhan", "nhon", "khoe", "manh"):
        assert fragment not in message
