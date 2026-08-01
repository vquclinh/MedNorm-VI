"""The linker-gold annotation contract (Audit 0065 §3).

This schema exists because Audit 0063 §7.1 found that the governed corpora carry no ICD or
RxNorm gold codes at all, which made retrieval Recall@k and MRR undefined. The records it
describes are the first ground truth this project will have, so the validator is strict:
a half-filled annotation is worse than no annotation, because it looks like evidence.

The two invariants these tests exist to protect:

* **silver never counts as gold** - `SILVER_UNIQUE_EXACT` is deliberately outside
  `HUMAN_GOLD_STATUSES`, so no benchmark can average an automatic exact-alias match
  together with a human decision;
* **ICD identity stays dotless** - the dot is display/submission metadata (Audit 0061), and
  a validator that accepted a dotted `code` would silently break the offered-set key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednorm_vi.annotation.linker_gold import (
    HUMAN_GOLD_STATUSES,
    LABEL_DIAGNOSIS,
    LABEL_MEDICATION,
    ONTOLOGY_ICD10,
    ONTOLOGY_RXNORM,
    SCHEMA_PATH,
    STATUS_HUMAN_GOLD,
    STATUS_SILVER_UNIQUE_EXACT,
    STATUS_UNLABELLED,
    LinkerGoldError,
    annotation_id,
    blank_record,
    classify_icd_specificity,
    document_hash,
    is_human_gold,
    validate_record,
)

REPO = Path(__file__).resolve().parents[2]
DOC = "a" * 64


def icd_candidate(code: str, rank: int = 1, score: float = 10.0) -> dict:
    return {
        "code": code,
        "display_code": code if len(code) <= 3 else f"{code[:3]}.{code[3:]}",
        "canonical_name": "name",
        "aliases": [],
        "linker_rank": rank,
        "score": score,
        "retrieval_channels": ["exact"],
        "structure": {
            "ontology": ONTOLOGY_ICD10,
            "parent_code": None,
            "child_codes": [],
            "hierarchy_depth": 1,
            "specificity_tier": "1",
            "specificity_class": classify_icd_specificity(code),
        },
    }


def rx_candidate(code: str, tty: str = "IN", rank: int = 1) -> dict:
    return {
        "code": code,
        "canonical_name": "name",
        "aliases": [],
        "linker_rank": rank,
        "score": 1.0,
        "retrieval_channels": ["char_ngram"],
        "structure": {
            "ontology": ONTOLOGY_RXNORM,
            "tty": tty,
            "ingredient": "x",
            "precise_ingredient": None,
            "strength": None,
            "dose_form": None,
            "brand": None,
            "graph_relations": [],
            "structured_match": [],
        },
    }


def diagnosis_record(**overrides) -> dict:
    record = blank_record(
        document_text_hash=DOC,
        source_dataset="vimedner",
        start=10,
        end=15,
        mention_text="abcde",
        context_before="before ",
        context_after=" after",
        entity_type=LABEL_DIAGNOSIS,
        candidate_snapshot_id="snap-1",
        offered_candidates=[icd_candidate("I269"), icd_candidate("I50", 2, 9.0)],
        provenance={
            "sampling_seed": 1,
            "stratum": "dx_exact_kb_alias",
            "pack_id": "p",
            "linker_version": "v",
        },
    )
    record.update(overrides)
    return record


# --- the schema file itself ----------------------------------------------------


def test_the_schema_file_is_valid_json_and_versioned() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$id"].endswith("linker_gold_v1.schema.json")
    assert set(schema["required"]) >= {
        "annotation_id",
        "source_dataset",
        "source_document_hash",
        "mention_start",
        "mention_end",
        "mention_text",
        "context_before",
        "context_after",
        "entity_type",
        "ontology",
        "candidate_snapshot_id",
        "offered_candidates",
        "selected_codes",
        "rejected_codes",
        "no_valid_candidate",
        "adjudication_status",
        "reviewer_id",
        "reviewer_confidence",
        "adjudication_notes",
        "provenance",
        "created_at",
        "updated_at",
    }


def test_the_schema_is_trackable_but_the_annotations_are_ignored() -> None:
    """The contract is public; the clinical text it describes must never be.

    Asserted as POLICY (gitignore) rather than as current tracking state: the schema may
    legitimately be uncommitted while a milestone is in flight, but a private annotation
    path being trackable would be a data-handling defect at any moment.
    """
    import subprocess

    def ignored(relative: str) -> bool:
        return (
            subprocess.run(
                ["git", "check-ignore", "-q", relative],
                cwd=REPO,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )

    assert not ignored("schemas/annotation/linker_gold_v1.schema.json"), (
        "the schema is a public contract and must be trackable"
    )
    for private in (
        "data/private_linker_gold/0065/pack.jsonl",
        "data/private_linker_gold/0065/manifest.json",
        "data/private_linker_gold/0065/silver/silver.jsonl",
    ):
        assert ignored(private), f"{private} carries clinical text and must be gitignored"

    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    assert not any(name.startswith("data/private_linker_gold/") for name in tracked)


# --- identity and derived fields ----------------------------------------------


def test_the_annotation_id_is_deterministic_for_a_mention() -> None:
    """Re-sampling the same mention must not create a second record for it."""
    assert annotation_id(DOC, 10, 15) == annotation_id(DOC, 10, 15)
    assert annotation_id(DOC, 10, 15) != annotation_id(DOC, 10, 16)
    assert annotation_id(DOC, 10, 15).startswith("lg1-")


def test_icd_specificity_classification() -> None:
    assert classify_icd_specificity("I50") == "category"
    assert classify_icd_specificity("I508") == "other"
    assert classify_icd_specificity("I509") == "unspecified"
    assert classify_icd_specificity("M4802") == "specific"
    with pytest.raises(LinkerGoldError):
        classify_icd_specificity("")


def test_document_hash_identifies_without_reproducing() -> None:
    assert len(document_hash("bệnh nhân sốt")) == 64


# --- ontology routing ----------------------------------------------------------


def test_a_blank_record_validates_and_carries_no_decision() -> None:
    record = diagnosis_record()
    assert validate_record(record).valid
    assert record["adjudication_status"] == STATUS_UNLABELLED
    assert record["selected_codes"] == [] and record["no_valid_candidate"] is False


def test_diagnosis_must_link_to_icd_and_medication_to_rxnorm() -> None:
    bad = diagnosis_record(ontology=ONTOLOGY_RXNORM)
    outcome = validate_record(bad)
    assert not outcome.valid
    assert any("forbids the cross-links" in p for p in outcome.problems)

    with pytest.raises(LinkerGoldError):
        blank_record(
            document_text_hash=DOC,
            source_dataset="vimedner",
            start=0,
            end=1,
            mention_text="x",
            context_before="",
            context_after="",
            entity_type="TRIỆU_CHỨNG",
            candidate_snapshot_id="s",
            offered_candidates=[],
            provenance={},
        )


def test_medication_records_route_to_rxnorm() -> None:
    record = blank_record(
        document_text_hash=DOC,
        source_dataset="vimq",
        start=0,
        end=3,
        mention_text="abc",
        context_before="",
        context_after="",
        entity_type=LABEL_MEDICATION,
        candidate_snapshot_id="s",
        offered_candidates=[rx_candidate("161")],
        provenance={"sampling_seed": 1, "stratum": "rx", "pack_id": "p", "linker_version": "v"},
    )
    assert record["ontology"] == ONTOLOGY_RXNORM
    assert validate_record(record).valid


# --- decision integrity --------------------------------------------------------


def test_selected_codes_and_no_valid_candidate_are_mutually_exclusive() -> None:
    record = diagnosis_record(
        selected_codes=["I269"],
        no_valid_candidate=True,
        adjudication_status=STATUS_HUMAN_GOLD,
        reviewer_id="r",
        reviewer_confidence="high",
    )
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("mutually exclusive" in p for p in outcome.problems)


def test_human_gold_must_carry_a_decision() -> None:
    record = diagnosis_record(
        adjudication_status=STATUS_HUMAN_GOLD, reviewer_id="r", reviewer_confidence="high"
    )
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("carries no decision" in p for p in outcome.problems)


def test_an_explicit_no_valid_candidate_is_a_complete_decision() -> None:
    """Audit 0063 measured a 5.4% empty-set rate; 'none is correct' must be recordable."""
    record = diagnosis_record(
        no_valid_candidate=True,
        adjudication_status=STATUS_HUMAN_GOLD,
        reviewer_id="r",
        reviewer_confidence="high",
    )
    assert validate_record(record).valid


def test_multiple_selected_codes_are_allowed() -> None:
    record = diagnosis_record(
        selected_codes=["I269", "I50"],
        adjudication_status=STATUS_HUMAN_GOLD,
        reviewer_id="r",
        reviewer_confidence="medium",
    )
    assert validate_record(record).valid


def test_a_reviewer_may_only_select_from_what_was_offered() -> None:
    record = diagnosis_record(
        selected_codes=["Z999"],
        adjudication_status=STATUS_HUMAN_GOLD,
        reviewer_id="r",
        reviewer_confidence="high",
    )
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("not among those offered" in p for p in outcome.problems)


def test_a_code_cannot_be_both_selected_and_rejected() -> None:
    record = diagnosis_record(
        selected_codes=["I269"],
        rejected_codes=["I269"],
        adjudication_status=STATUS_HUMAN_GOLD,
        reviewer_id="r",
        reviewer_confidence="high",
    )
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("both selected and rejected" in p for p in outcome.problems)


def test_an_unlabelled_record_may_not_carry_a_decision() -> None:
    record = diagnosis_record(selected_codes=["I269"])
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("UNLABELLED" in p for p in outcome.problems)


# --- ICD identity --------------------------------------------------------------


def test_selected_icd_codes_must_be_dotless() -> None:
    """The dot is display metadata (Audit 0061); the KB key is dotless."""
    record = diagnosis_record(
        selected_codes=["I26.9"],
        adjudication_status=STATUS_HUMAN_GOLD,
        reviewer_id="r",
        reviewer_confidence="high",
    )
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("DOTLESS" in p for p in outcome.problems)


def test_a_dotted_candidate_code_is_refused_but_display_code_is_fine() -> None:
    candidate = icd_candidate("I269")
    candidate["code"] = "I26.9"
    record = diagnosis_record(offered_candidates=[candidate])
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("dotless" in p.lower() for p in outcome.problems)


# --- snapshot membership -------------------------------------------------------


def test_a_selected_code_must_exist_in_the_frozen_snapshot() -> None:
    record = diagnosis_record(
        selected_codes=["I269"],
        adjudication_status=STATUS_HUMAN_GOLD,
        reviewer_id="r",
        reviewer_confidence="high",
    )
    assert validate_record(record, snapshot_codes=frozenset({"I269", "I50"})).valid
    outcome = validate_record(record, snapshot_codes=frozenset({"I50"}))
    assert not outcome.valid
    assert any("absent from snapshot" in p for p in outcome.problems)


# --- silver is never gold ------------------------------------------------------


def test_silver_is_not_counted_as_human_gold() -> None:
    assert STATUS_SILVER_UNIQUE_EXACT not in HUMAN_GOLD_STATUSES
    silver = diagnosis_record(
        selected_codes=["I269"],
        adjudication_status=STATUS_SILVER_UNIQUE_EXACT,
        reviewer_id="auto:unique_exact_alias",
    )
    assert validate_record(silver).valid, "a silver record is structurally valid"
    assert not is_human_gold(silver), "but it must never count as ground truth"


def test_human_gold_statuses_are_exactly_the_reviewed_ones() -> None:
    assert HUMAN_GOLD_STATUSES == {"HUMAN_GOLD", "SECOND_REVIEW_AGREED"}


# --- structural requirements on the evidence ----------------------------------


def test_rxnorm_candidates_must_record_tty() -> None:
    candidate = rx_candidate("161")
    del candidate["structure"]["tty"]
    record = blank_record(
        document_text_hash=DOC,
        source_dataset="vimq",
        start=0,
        end=3,
        mention_text="abc",
        context_before="",
        context_after="",
        entity_type=LABEL_MEDICATION,
        candidate_snapshot_id="s",
        offered_candidates=[candidate],
        provenance={"sampling_seed": 1, "stratum": "rx", "pack_id": "p", "linker_version": "v"},
    )
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("TTY" in p for p in outcome.problems)


def test_mention_text_must_match_the_span_width() -> None:
    outcome = validate_record(diagnosis_record(mention_text="too-long-for-the-span"))
    assert not outcome.valid
    assert any("does not match the span width" in p for p in outcome.problems)


def test_more_than_ten_offered_candidates_is_refused() -> None:
    record = diagnosis_record(
        offered_candidates=[icd_candidate("I269", rank=i + 1) for i in range(11)]
    )
    outcome = validate_record(record)
    assert not outcome.valid
    assert any("exceeds the ten" in p for p in outcome.problems)
