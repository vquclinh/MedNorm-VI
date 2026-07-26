"""Exact character-offset mention evaluator (Audit 0033).

Pure logic: no model, no corpus, no Torch. Every fixture states its offsets
explicitly so the invariant under test is visible.
"""

from __future__ import annotations

import pytest

from mednorm_vi.evaluation.exact_mention import (
    BOTH_BOUNDARY,
    DUPLICATE_OVERLAP,
    EXACT_MATCH,
    LEFT_BOUNDARY,
    MISSED,
    PROVENANCE_DETERMINISTIC,
    PROVENANCE_NEURAL,
    RIGHT_BOUNDARY,
    SPURIOUS,
    WRONG_TYPE,
    ExactMentionError,
    ExactMentionEvaluator,
    Mention,
    config_hash_of,
    evaluate_examples,
    length_bucket,
    privacy_safe_example_id,
    render_markdown,
    validate_mention,
)

TEXT = "benh nhan sot cao va ho nhieu ngay"
#       0123456789...
SOT_CAO = Mention(10, 17, "SYMPTOM", "sot cao")
HO = Mention(21, 23, "SYMPTOM", "ho")


def _score(gold, predicted, text=TEXT, source="vimedner"):
    evaluator = ExactMentionEvaluator()
    evaluator.update(text, gold, predicted, example_id="ex:0001", source=source)
    return evaluator.report()


# --- the invariant ------------------------------------------------------------

def test_span_must_slice_exactly_the_mention_text() -> None:
    assert TEXT[10:17] == "sot cao"
    validate_mention(SOT_CAO, TEXT, role="gold")
    with pytest.raises(ExactMentionError, match="original_text"):
        validate_mention(Mention(10, 16, "SYMPTOM", "sot cao"), TEXT, role="gold")


def test_end_offset_is_exclusive() -> None:
    tail = Mention(29, 34, "SYMPTOM", "ngay")
    with pytest.raises(ExactMentionError, match="original_text"):
        validate_mention(tail, TEXT, role="gold")        # 29:34 is " ngay", not "ngay"
    validate_mention(Mention(30, 34, "SYMPTOM", "ngay"), TEXT, role="gold")


def test_out_of_range_and_unsupported_type_are_rejected() -> None:
    with pytest.raises(ExactMentionError, match="exceeds"):
        validate_mention(Mention(30, 999, "SYMPTOM", "x"), TEXT, role="gold")
    with pytest.raises(ExactMentionError, match="unsupported type"):
        validate_mention(Mention(10, 17, "NOT_A_TYPE", "sot cao"), TEXT, role="gold")


def test_decomposed_unicode_offsets_are_respected() -> None:
    text = "benh nhan ởn dinh"          # decomposed base + combining mark
    mention = Mention(10, 13, "SYMPTOM", text[10:13])
    validate_mention(mention, text, role="gold")
    assert len(mention.text) == 3                    # 2 code points + 'n'
    report = _score([mention], [mention], text=text)
    assert report["micro"]["true_positive"] == 1


def test_repeated_spaces_and_newlines_do_not_shift_offsets() -> None:
    text = "sot  cao\nva ho"
    gold = Mention(0, 8, "SYMPTOM", "sot  cao")
    assert text[0:8] == "sot  cao"
    assert _score([gold], [gold], text=text)["micro"]["f1"] == 1.0


# --- exact matching and the wrong-type double penalty --------------------------

def test_exact_match_scores_a_true_positive() -> None:
    report = _score([SOT_CAO], [SOT_CAO])
    assert report["micro"]["true_positive"] == 1
    assert report["error_categories"][EXACT_MATCH] == 1
    assert report["by_type"]["SYMPTOM"]["f1"] == 1.0


def test_wrong_type_is_a_false_positive_and_a_false_negative() -> None:
    """Spec §1: a wrong type is double-penalised."""
    predicted = Mention(10, 17, "DIAGNOSIS", "sot cao")
    report = _score([SOT_CAO], [predicted])
    assert report["error_categories"][WRONG_TYPE] == 1
    assert report["micro"]["true_positive"] == 0
    assert report["micro"]["false_positive"] == 1
    assert report["micro"]["false_negative"] == 1
    # charged to the RIGHT types: FP on predicted, FN on gold
    assert report["by_type"]["DIAGNOSIS"]["false_positive"] == 1
    assert report["by_type"]["SYMPTOM"]["false_negative"] == 1
    assert report["by_type"]["DIAGNOSIS"]["false_negative"] == 0


@pytest.mark.parametrize("start,end,category", [
    (11, 17, LEFT_BOUNDARY),      # left edge slipped
    (10, 13, RIGHT_BOUNDARY),     # right edge slipped
    (11, 13, BOTH_BOUNDARY),      # both edges slipped
])
def test_boundary_errors_are_categorised_but_never_credited(start, end, category) -> None:
    predicted = Mention(start, end, "SYMPTOM", TEXT[start:end])
    report = _score([SOT_CAO], [predicted])
    assert report["error_categories"][category] == 1
    assert report["micro"]["true_positive"] == 0          # NOT a fuzzy match
    assert report["micro"]["false_positive"] == 1
    assert report["micro"]["false_negative"] == 1


def test_a_near_miss_is_never_silently_accepted() -> None:
    """Overlapping text must not be treated as an exact match."""
    predicted = Mention(10, 13, "SYMPTOM", "sot")
    assert _score([SOT_CAO], [predicted])["micro"]["f1"] == 0.0


def test_missed_and_spurious_are_separated() -> None:
    report = _score([SOT_CAO], [Mention(30, 34, "SYMPTOM", "ngay")])
    assert report["error_categories"][MISSED] == 1
    assert report["error_categories"][SPURIOUS] == 1


def test_a_prediction_colliding_with_a_match_is_a_duplicate_overlap() -> None:
    extra = Mention(10, 13, "DIAGNOSIS", "sot")
    report = _score([SOT_CAO], [SOT_CAO, extra])
    assert report["error_categories"][EXACT_MATCH] == 1
    assert report["error_categories"][DUPLICATE_OVERLAP] == 1
    assert report["error_categories"][SPURIOUS] == 0


# --- no text-only deduplication ------------------------------------------------

def test_repeated_identical_text_at_different_offsets_stays_separate() -> None:
    text = "ho nhieu va ho lai"
    first = Mention(0, 2, "SYMPTOM", "ho")
    second = Mention(12, 14, "SYMPTOM", "ho")
    assert text[0:2] == text[12:14] == "ho"
    report = _score([first, second], [first, second], text=text)
    assert report["micro"]["true_positive"] == 2      # two mentions, not one
    assert report["gold_mentions"] == 2


def test_predicting_only_one_of_two_identical_surface_forms_is_a_miss() -> None:
    text = "ho nhieu va ho lai"
    gold = [Mention(0, 2, "SYMPTOM", "ho"), Mention(12, 14, "SYMPTOM", "ho")]
    report = _score(gold, [gold[0]], text=text)
    assert report["micro"]["true_positive"] == 1
    assert report["error_categories"][MISSED] == 1


def test_adjacent_non_overlapping_spans_both_match() -> None:
    report = _score([SOT_CAO, HO], [SOT_CAO, HO])
    assert report["micro"]["true_positive"] == 2
    assert report["micro"]["f1"] == 1.0


# --- grouping and provenance ---------------------------------------------------

def test_results_are_grouped_by_provenance_and_length_and_source() -> None:
    neural = Mention(10, 17, "SYMPTOM", "sot cao", provenance=PROVENANCE_NEURAL)
    deterministic = Mention(21, 23, "SYMPTOM", "ho",
                            provenance=PROVENANCE_DETERMINISTIC)
    evaluator = ExactMentionEvaluator()
    evaluator.update(TEXT, [SOT_CAO, HO], [neural, deterministic],
                     example_id="ex:1", source="vimedner")
    report = evaluator.report()
    assert report["by_provenance"][PROVENANCE_NEURAL]["true_positive"] == 1
    assert report["by_provenance"][PROVENANCE_DETERMINISTIC]["true_positive"] == 1
    assert report["by_source"]["vimedner"]["true_positive"] == 2
    assert report["by_length_bucket"]["6-10"]["true_positive"] == 1   # "sot cao" = 7
    assert report["by_length_bucket"]["1-5"]["true_positive"] == 1    # "ho" = 2


@pytest.mark.parametrize("length,bucket", [
    (1, "1-5"), (5, "1-5"), (6, "6-10"), (11, "11-20"), (21, "21-40"), (500, "41+"),
])
def test_length_buckets(length, bucket) -> None:
    assert length_bucket(length) == bucket


def test_route_and_section_grouping_default_sensibly() -> None:
    report = _score([SOT_CAO], [SOT_CAO])
    assert "unrouted" in report["by_route"]
    assert "unsectioned" in report["by_section"]


# --- reporting -----------------------------------------------------------------

def test_report_states_it_is_not_the_complete_organizer_score() -> None:
    report = _score([SOT_CAO], [SOT_CAO])
    assert report["reproduces_complete_organizer_score"] is False
    assert any("Jaccard" in item for item in report["excluded_from_score"])
    markdown = render_markdown(report)
    assert "not** the complete" in markdown


def test_diagnostics_carry_hashed_ids_and_no_clinical_text() -> None:
    report = _score([SOT_CAO], [Mention(30, 34, "SYMPTOM", "ngay")])
    serialised = str(report["errors"])
    assert privacy_safe_example_id("ex:0001") in serialised
    assert "ex:0001" not in serialised
    for fragment in ("sot cao", "ngay", "benh nhan"):
        assert fragment not in serialised


def test_config_hash_is_deterministic_and_sensitive() -> None:
    assert config_hash_of({"a": 1}) == config_hash_of({"a": 1})
    assert config_hash_of({"a": 1}) != config_hash_of({"a": 2})


def test_evaluate_examples_reads_governed_rows() -> None:
    examples = [{
        "example_id": "vimedner:train:0001", "source_dataset": "vimedner", "text": TEXT,
        "entities": [{"start": 10, "end": 17, "text": "sot cao", "target_type": "SYMPTOM"}],
    }]
    report = evaluate_examples(
        examples, predictions={"vimedner:train:0001": [SOT_CAO]}, label="unit")
    assert report["micro"]["f1"] == 1.0
    assert report["label"] == "unit"
    assert report["by_source"]["vimedner"]["true_positive"] == 1


def test_an_example_with_no_predictions_counts_every_gold_as_missed() -> None:
    report = _score([SOT_CAO, HO], [])
    assert report["micro"]["recall"] == 0.0
    assert report["error_categories"][MISSED] == 2
