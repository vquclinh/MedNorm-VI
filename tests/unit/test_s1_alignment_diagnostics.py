"""Pre-segmented-source alignment fix and privacy-safe diagnostics (Audit 0026).

Deterministic and offline: no VnCoreNLP, no tokenizer download, no training.
The failing production pattern is reproduced with the governed corpus's own
shapes and a whitespace test tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from mednorm_vi.training.colab_bootstrap import evaluate_full_training_readiness
from mednorm_vi.training.phobert_alignment import (
    ASTRAL_SENTINEL,
    BOUNDARY_MERGE_POLICY,
    EQUIVALENCE_CANONICAL,
    EQUIVALENCE_TONE_PLACEMENT,
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
    VIETNAMESE_VOWEL_BASES,
    AlignmentError,
    entities_touched_by_boundary_merge,
    is_legal_syllable_gap,
    looks_pre_segmented,
    map_segmented_words,
    orthographic_equivalence,
    protect_astral_characters,
    resolve_segmented_text,
    restore_astral_characters,
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
    assert "categories" in message and "offset" in message
    assert "same_base_letter=" in message
    for fragment in ("nhan", "nhon", "khoe", "manh"):
        assert fragment not in message


# --- Audit 0029: general Vietnamese orthographic equivalence ------------------
#
# The governed corpus stores some text in a partially DECOMPOSED form (a base
# letter followed by a standalone combining mark). The segmenter emits the
# composed spelling, and it may also move a tone mark within a vowel cluster.
# Everything below is written from those TRANSFORMATIONS - never from an
# example's text or privacy-safe id.

# Audit 0028's fixed 15-pair table, kept only as a coverage oracle: the general
# rule must still accept every pair the table used to enumerate.
TONE_PAIR_ORACLE = (
    ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
    ("oè", "òe"), ("oé", "óe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
    ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
)


def _aligns(original: str, segmented: str) -> bool:
    try:
        map_segmented_words(original, segmented_text_to_words(segmented))
    except AlignmentError:
        return False
    return True


def _decomposed(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def _composed(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# --- transformation 1: canonical composition (the four real v4 failures) ------

@pytest.mark.parametrize("base_letter,mark_name", [
    ("ơ", "COMBINING HOOK ABOVE"),        # U+01A1 + U+0309
    ("i", "COMBINING TILDE"),             # U+0069 + U+0303
    ("i", "COMBINING GRAVE ACCENT"),      # U+0069 + U+0300
    ("â", "COMBINING GRAVE ACCENT"),      # U+00E2 + U+0300
])
def test_partially_decomposed_source_aligns_with_the_composed_spelling(
    base_letter, mark_name,
) -> None:
    """Each of the four real failures, reproduced from its code points alone."""
    mark = unicodedata.lookup(mark_name)
    original = f"x {base_letter}{mark} y"          # the source's decomposed form
    segmented = _composed(original)                # what the segmenter emits
    assert len(segmented) < len(original), "the fixture must exercise composition"
    words = map_segmented_words(original, segmented_text_to_words(segmented))
    cluster = words[1]
    # The span covers BOTH original code points and slices untouched original_text.
    assert original[cluster.original_start:cluster.original_end] == f"{base_letter}{mark}"
    assert cluster.original_end - cluster.original_start == 2


def test_composition_is_accepted_in_both_directions() -> None:
    composed, decomposed = "ở", _decomposed("ở")
    assert len(decomposed) == 3 and len(composed) == 1
    for original, segmented in ((decomposed, composed), (composed, decomposed)):
        result = orthographic_equivalence(original, 0, segmented, 0)
        assert result is not None
        assert result[2] == EQUIVALENCE_CANONICAL
        assert result[0] == len(original) and result[1] == len(segmented)


def test_offsets_index_untouched_original_text_when_the_source_is_decomposed() -> None:
    """The aligner must never normalize the original: that shifts every offset."""
    original = f"benh {_decomposed('ở')} nhan"
    assert unicodedata.normalize("NFC", original) != original
    words = map_segmented_words(original, segmented_text_to_words(_composed(original)))
    # The trailing word still maps onto its true position in the ORIGINAL string.
    assert original[words[-1].original_start:words[-1].original_end] == "nhan"
    assert words[-1].original_end == len(original)


# --- transformation 2: tone placement, and both at once -----------------------

@pytest.mark.parametrize("first,second", TONE_PAIR_ORACLE)
def test_the_general_rule_covers_every_pair_the_old_table_enumerated(first, second) -> None:
    for original, segmented in ((first, second), (second, first)):
        result = orthographic_equivalence(original, 0, segmented, 0)
        assert result is not None, (original, segmented)
        assert result[2] == EQUIVALENCE_TONE_PLACEMENT


def test_the_general_rule_covers_clusters_absent_from_the_old_table() -> None:
    """A three-vowel cluster, which no two-character pair table could express."""
    result = orthographic_equivalence("uyế", 0, "úyê", 0)
    assert result is not None and result[2] == EQUIVALENCE_TONE_PLACEMENT


def test_a_decomposed_source_whose_tone_also_moves_aligns() -> None:
    """Both transformations in one cluster - the shape the full corpus exposed."""
    original = "u\u0309y"                          # decomposed, tone on the first vowel
    segmented = _composed("uỷ")                    # composed, tone on the second
    assert original != segmented
    words = map_segmented_words(original, segmented_text_to_words(segmented))
    assert len(words) == 1
    assert words[0].original_start == 0
    assert words[0].original_end == len(original)   # the whole ORIGINAL cluster


# --- the rule cannot accept a different tone or base letter -------------------

@pytest.mark.parametrize("original,segmented,label", [
    ("hóa", "hõa", "different tone"),
    ("hóa", "hóe", "different base letter"),
    ("hoa", "hao", "reordered base letters"),
    ("hoa", "hoaa", "inserted character"),
    ("hóa", "hó", "dropped character"),
    ("an", "án", "tone inserted"),
    ("án", "an", "tone removed"),
    ("uơ", "ưo", "non-tone quality mark moved to another letter"),
    ("hoa xyz", "hoa", "dropped trailing word"),
])
def test_unsupported_character_changes_still_fail(original, segmented, label) -> None:
    assert not _aligns(original, segmented), label


def test_a_tone_may_not_move_onto_a_consonant() -> None:
    """The carrier may only change inside a cluster of Vietnamese vowels."""
    assert all(base in VIETNAMESE_VOWEL_BASES for base in "aeiouy")
    assert orthographic_equivalence("ó" + "n", 0, "o" + "ń", 0) is None


def test_equivalence_requires_an_identical_tone_multiset() -> None:
    for original, segmented in (("óa", "õa"), ("óa", "òa"), ("óá", "oá"), ("óa", "oa")):
        assert orthographic_equivalence(original, 0, segmented, 0) is None, (
            original, segmented)


def test_equivalence_requires_identical_non_tone_marks() -> None:
    # circumflex vs horn on the same base letter: same letter, different vowel.
    assert orthographic_equivalence("ầu", 0, "àu", 0) is None
    assert orthographic_equivalence("ơi", 0, "oi", 0) is None


# --- non-BMP characters across the JVM segmenter boundary ---------------------
#
# VnCoreNLP runs on the JVM, where text is UTF-16. A non-BMP character is a
# surrogate pair there and the round trip truncates it to its low 16 bits, which
# silently replaces real clinical content with a different character. It is NOT
# an orthographic variant and must never be accepted as equivalent.

ASTRAL_CHARACTER = "\U0001d6c3"          # MATHEMATICAL BOLD SMALL BETA


def test_astral_characters_are_protected_across_the_segmenter() -> None:
    text = f"xet nghiem {ASTRAL_CHARACTER} - hCG"

    def jvm_like_segmenter(payload: str) -> str:
        # A JVM round trip preserves BMP characters exactly.
        assert all(ord(c) <= 0xFFFF for c in payload), "astral char reached the JVM"
        return payload.replace("xet nghiem", "xet_nghiem")

    segmented, source = resolve_segmented_text(text, jvm_like_segmenter)
    assert source == SEGMENTATION_SOURCE_SEGMENTER
    assert ASTRAL_CHARACTER in segmented          # restored verbatim
    words = map_segmented_words(text, segmented_text_to_words(segmented))
    carrier = next(w for w in words if ASTRAL_CHARACTER in w.model_text)
    assert text[carrier.original_start:carrier.original_end] == ASTRAL_CHARACTER


def test_a_truncated_astral_character_is_never_accepted_as_equivalent() -> None:
    """0x1D6C3 & 0xFFFF == 0xD6C3 - a different letter, so alignment must fail."""
    truncated = chr(ord(ASTRAL_CHARACTER) & 0xFFFF)
    assert truncated != ASTRAL_CHARACTER
    assert orthographic_equivalence(ASTRAL_CHARACTER, 0, truncated, 0) is None
    with pytest.raises(AlignmentError) as excinfo:
        map_segmented_words(f"a {ASTRAL_CHARACTER} b", segmented_text_to_words(f"a {truncated} b"))
    assert excinfo.value.reason_code == REASON_SEGMENTER_ALTERED_CHARACTERS


def test_protection_round_trips_and_preserves_length() -> None:
    text = f"{ASTRAL_CHARACTER} x {ASTRAL_CHARACTER}"
    protected, astral = protect_astral_characters(text)
    assert len(protected) == len(text)            # one code point for one code point
    assert astral == (ASTRAL_CHARACTER, ASTRAL_CHARACTER)
    assert all(ord(c) <= 0xFFFF for c in protected)
    assert restore_astral_characters(protected, astral) == text


def test_text_without_astral_characters_is_untouched() -> None:
    text = "benh nhan dau dau"
    protected, astral = protect_astral_characters(text)
    assert protected == text and astral == ()
    assert restore_astral_characters(text, ()) == text


def test_a_segmenter_that_loses_a_protected_character_fails_loudly() -> None:
    _protected, astral = protect_astral_characters(f"a {ASTRAL_CHARACTER} b")
    with pytest.raises(AlignmentError) as excinfo:
        restore_astral_characters("a b", astral)          # sentinel dropped
    assert excinfo.value.reason_code == REASON_SEGMENTER_ALTERED_CHARACTERS


def test_a_segmenter_that_duplicates_a_protected_character_fails_loudly() -> None:
    protected, astral = protect_astral_characters(f"a {ASTRAL_CHARACTER} b")
    with pytest.raises(AlignmentError) as excinfo:
        restore_astral_characters(protected + ASTRAL_SENTINEL, astral)
    assert excinfo.value.reason_code == REASON_SEGMENTER_ALTERED_CHARACTERS


def test_a_segmenter_that_reorders_protected_characters_fails_loudly() -> None:
    second_astral = "\U0001d6c4"
    protected, astral = protect_astral_characters(f"{ASTRAL_CHARACTER} x {second_astral}")
    first_sentinel = protected[0]
    second_sentinel = protected[-1]
    assert first_sentinel != second_sentinel
    reordered = f"{second_sentinel} x {first_sentinel}"
    with pytest.raises(AlignmentError) as excinfo:
        restore_astral_characters(reordered, astral)
    assert excinfo.value.reason_code == REASON_SEGMENTER_ALTERED_CHARACTERS


def test_text_already_containing_the_sentinel_is_refused() -> None:
    with pytest.raises(AlignmentError):
        protect_astral_characters(f"a {ASTRAL_SENTINEL} {ASTRAL_CHARACTER}")
