"""Full-corpus alignment preflight contract (Audit 0029).

Pure logic: the encode/verify callables are supplied by the caller, so these
tests need no tokenizer, no VnCoreNLP, no corpus and no model.
"""

from __future__ import annotations

import pytest

from mednorm_vi.training.phobert_alignment import (
    REASON_SEGMENTER_ALTERED_CHARACTERS,
    STAGE_SUBTOKEN_ENCODING,
    STAGE_TOKENIZER_EQUIVALENCE,
    AlignmentError,
)
from mednorm_vi.training.s1_mention_smoke import (
    privacy_safe_example_id,
    run_alignment_preflight,
)


def _rows(split: str, count: int, source: str = "src_a"):
    return [{"example_id": f"{split}:{i:04d}", "source_dataset": source,
             "text": "x", "entities": []} for i in range(count)]


def _splits(train: int = 5, validation: int = 3):
    return {"train": _rows("train", train), "validation": _rows("validation", validation)}


def _ok(row):
    return {"example_id": row["example_id"]}


def test_a_clean_corpus_passes_and_reconciles() -> None:
    report, features = run_alignment_preflight(_splits(), encode=_ok)
    assert report.passed is True
    assert report.counters_reconciled is True
    assert report.examples_considered == 8
    assert report.aligned_example_count == 8
    assert report.unalignable_example_count == 0
    assert report.governed_exclusion_count == 0
    assert len(features["train"]) == 5 and len(features["validation"]) == 3


def test_encoding_failures_block_and_are_recorded_by_stage() -> None:
    failing = {"train:0002"}

    def encode(row):
        if row["example_id"] in failing:
            raise AlignmentError("boom", REASON_SEGMENTER_ALTERED_CHARACTERS)
        return _ok(row)

    report, _ = run_alignment_preflight(_splits(), encode=encode)
    assert report.passed is False
    assert report.unalignable_example_count == 1
    assert report.aligned_example_count == 7
    assert report.counters_reconciled is True          # the books still balance
    assert report.stage_counts == {STAGE_SUBTOKEN_ENCODING: 1}
    assert report.reason_code_counts == {REASON_SEGMENTER_ALTERED_CHARACTERS: 1}
    assert report.by_split["train"]["unalignable"] == 1
    assert report.by_split["validation"]["unalignable"] == 0


def test_equivalence_failures_are_counted_separately_and_block() -> None:
    def verify(row):
        if row["example_id"] == "validation:0001":
            raise AlignmentError("mismatch", REASON_SEGMENTER_ALTERED_CHARACTERS)

    report, features = run_alignment_preflight(
        _splits(), encode=_ok, verify_equivalence=verify)
    assert report.passed is False
    assert report.equivalence_failure_count == 1
    assert report.equivalence_checked_count == 7
    assert report.unalignable_example_count == 1
    assert report.stage_counts == {STAGE_TOKENIZER_EQUIVALENCE: 1}
    # a failed example never reaches the feature set
    assert len(features["validation"]) == 2


def test_governed_exclusions_are_separate_and_do_not_block() -> None:
    excluded = privacy_safe_example_id("train:0001")
    report, _ = run_alignment_preflight(
        _splits(), encode=_ok,
        governed_exclusions={excluded: {"justification": "tracked decision"}})
    assert report.passed is True                       # a decision, not a failure
    assert report.governed_exclusion_count == 1
    assert report.unalignable_example_count == 0
    assert report.aligned_example_count == 7
    assert report.counters_reconciled is True
    assert report.governed_exclusions[0]["privacy_safe_example_id"] == excluded
    assert report.unalignable_examples == []


def test_unsupervised_examples_are_not_considered_at_all() -> None:
    def supervised(row):
        return not row["example_id"].endswith("0000")

    report, _ = run_alignment_preflight(_splits(), encode=_ok, supervised=supervised)
    assert report.examples_considered == 6             # one dropped from each split
    assert report.counters_reconciled is True


def test_counts_are_reported_per_split_and_per_source() -> None:
    splits = {"train": _rows("train", 4, "src_a") + _rows("train2", 2, "src_b"),
              "validation": _rows("validation", 3, "src_b")}
    report, _ = run_alignment_preflight(splits, encode=_ok)
    assert report.by_split["train"]["considered"] == 6
    assert report.by_split["validation"]["considered"] == 3
    assert report.by_source["src_a"]["considered"] == 4
    assert report.by_source["src_b"]["considered"] == 5
    assert sum(v["considered"] for v in report.by_split.values()) == report.examples_considered
    assert sum(v["considered"] for v in report.by_source.values()) == report.examples_considered


def test_diagnostics_stay_privacy_safe() -> None:
    def encode(row):
        raise AlignmentError("boom", REASON_SEGMENTER_ALTERED_CHARACTERS)

    report, _ = run_alignment_preflight(
        {"train": [{"example_id": "train:0007", "source_dataset": "src_a",
                    "text": "benh nhan dau bung", "entities": []}]},
        encode=encode)
    entry = report.unalignable_examples[0]
    assert set(entry) == {"source", "split", "privacy_safe_example_id", "stage",
                          "reason_code", "exception_type"}
    assert entry["privacy_safe_example_id"] != "train:0007"
    for leak in ("benh", "nhan", "dau", "bung", "train:0007"):
        assert leak not in str(report.as_dict())


@pytest.mark.parametrize("failures,expected", [(0, True), (1, False), (3, False)])
def test_passed_requires_zero_unexpected_failures(failures, expected) -> None:
    ids = {f"train:{i:04d}" for i in range(failures)}

    def encode(row):
        if row["example_id"] in ids:
            raise AlignmentError("boom", REASON_SEGMENTER_ALTERED_CHARACTERS)
        return _ok(row)

    report, _ = run_alignment_preflight(_splits(), encode=encode)
    assert report.passed is expected


def test_report_serializes_every_counter() -> None:
    report, _ = run_alignment_preflight(_splits(), encode=_ok)
    payload = report.as_dict()
    for key in ("alignment_preflight_passed", "counters_reconciled",
                "examples_considered", "aligned_example_count",
                "unalignable_example_count", "governed_exclusion_count",
                "equivalence_checked_count", "equivalence_failure_count",
                "by_split", "by_source", "reason_code_counts", "stage_counts",
                "unalignable_examples", "governed_exclusions"):
        assert key in payload
