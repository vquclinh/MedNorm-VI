"""AI-assisted pre-annotation must never become human gold (Audit 0066 §2).

Audit 0065 built the first honest linker benchmark on the principle that only human
decisions count. Introducing a model that drafts labels is exactly the change that could
erode it - silently, and in a way no single record reveals. These tests hold the boundary
as a property of the vocabularies and the writer, not as a convention.

The five things that must remain true: an AI status is never a gold status; drafts cannot
unlock the benchmark; drafts cannot overwrite a human decision; a draft cannot invent a
code the linker never offered; and the ontology restriction survives.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mednorm_vi.annotation.linker_gold import (
    AI_DRAFT_STATUSES,
    HUMAN_GOLD_STATUSES,
    STATUS_AI_ASSISTED_DRAFT,
    STATUS_AI_WEAK_HIGH_CONFIDENCE,
    STATUS_HUMAN_GOLD,
    LinkerGoldError,
    assert_not_ai_gold,
    is_ai_draft,
    is_human_gold,
)

REPO = Path(__file__).resolve().parents[2]
WRITER = REPO / "scripts" / "build_ai_linker_drafts_0066.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _module(WRITER, "build_ai_linker_drafts_0066")


def pack_record(**overrides) -> dict:
    record = {
        "annotation_id": "lg1-0123456789abcdef",
        "ontology": "ICD10",
        "entity_type": "CHẨN_ĐOÁN",
        "adjudication_status": "UNLABELLED",
        "candidate_snapshot_id": "snap",
        "provenance": {"stratum": "s"},
        "offered_candidates": [
            {
                "code": "I269",
                "canonical_name": "n",
                "aliases": [],
                "linker_rank": 1,
                "score": 1.0,
                "retrieval_channels": ["exact"],
                "structure": {"ontology": "ICD10", "specificity_class": "unspecified"},
            }
        ],
    }
    record.update(overrides)
    return record


# --- the boundary --------------------------------------------------------------


def test_ai_statuses_are_never_human_gold_statuses() -> None:
    assert not (AI_DRAFT_STATUSES & HUMAN_GOLD_STATUSES)
    assert_not_ai_gold()


def test_neither_ai_status_passes_is_human_gold() -> None:
    for status in (STATUS_AI_ASSISTED_DRAFT, STATUS_AI_WEAK_HIGH_CONFIDENCE):
        record = {"adjudication_status": status, "selected_codes": ["I269"]}
        assert not is_human_gold(record), f"{status} must never count as gold"
        assert is_ai_draft(record)


def test_even_the_promoted_weak_label_is_not_gold() -> None:
    """AI_WEAK_HIGH_CONFIDENCE is the most a draft can become, and it is still not gold."""
    assert STATUS_AI_WEAK_HIGH_CONFIDENCE in AI_DRAFT_STATUSES
    assert STATUS_AI_WEAK_HIGH_CONFIDENCE not in HUMAN_GOLD_STATUSES


def test_ai_drafts_cannot_unlock_benchmark_reporting(tmp_path: Path) -> None:
    """Hundreds of AI drafts must leave the benchmark exactly as closed as zero would."""
    evaluator = _module(REPO / "scripts" / "evaluate_linker_gold.py", "evaluate_linker_gold")
    pack = tmp_path / "pack.jsonl"
    rows = []
    for index in range(200):
        rows.append(
            {
                "annotation_id": f"lg1-{index:016x}",
                "ontology": "ICD10" if index % 2 else "RXNORM",
                "adjudication_status": STATUS_AI_ASSISTED_DRAFT,
                "selected_codes": ["I269"],
                "no_valid_candidate": False,
                "offered_candidates": [
                    {"code": "I269", "linker_rank": 1, "structure": {"ontology": "ICD10"}}
                ],
                "reviewer_confidence": None,
                "provenance": {"stratum": "s"},
            }
        )
    pack.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    evaluator.main(["--pack", str(pack), "--output", str(out)])
    payload = json.loads((out / "linker_benchmark.json").read_text(encoding="utf-8"))
    assert payload["reportable"] is False
    assert payload["reviewed_by_ontology"] == {"ICD10": 0, "RXNORM": 0}
    assert payload["decision"]["next_intervention"] == "MORE_HUMAN_ADJUDICATION_REQUIRED"


# --- the writer's guards -------------------------------------------------------


def test_a_draft_cannot_select_a_code_that_was_never_offered() -> None:
    with pytest.raises(LinkerGoldError, match="never offered"):
        BUILDER.build_draft(
            pack_record(),
            {
                "decision": "SELECTED",
                "codes": ["Z999"],
                "confidence": "HIGH",
                "flag": "f",
                "rationale": "r",
            },
        )


def test_an_icd_draft_must_use_the_dotless_identity() -> None:
    record = pack_record()
    record["offered_candidates"][0]["code"] = "I26.9"
    with pytest.raises(LinkerGoldError, match="dotless"):
        BUILDER.build_draft(
            record,
            {
                "decision": "SELECTED",
                "codes": ["I26.9"],
                "confidence": "HIGH",
                "flag": "f",
                "rationale": "r",
            },
        )


def test_a_non_selected_decision_may_not_carry_codes() -> None:
    for decision in ("NO_VALID_CANDIDATE", "ABSTAIN"):
        with pytest.raises(LinkerGoldError, match="must not carry codes"):
            BUILDER.build_draft(
                pack_record(),
                {
                    "decision": decision,
                    "codes": ["I269"],
                    "confidence": "HIGH",
                    "flag": "f",
                    "rationale": "r",
                },
            )


def test_a_selected_decision_must_carry_a_code() -> None:
    with pytest.raises(LinkerGoldError, match="SELECTED with no codes"):
        BUILDER.build_draft(
            pack_record(),
            {
                "decision": "SELECTED",
                "codes": [],
                "confidence": "HIGH",
                "flag": "f",
                "rationale": "r",
            },
        )


def test_every_built_draft_carries_the_ai_status() -> None:
    draft = BUILDER.build_draft(
        pack_record(),
        {
            "decision": "SELECTED",
            "codes": ["I269"],
            "confidence": "HIGH",
            "flag": "f",
            "rationale": "r",
        },
    )
    assert draft["adjudication_status"] == STATUS_AI_ASSISTED_DRAFT
    assert not is_human_gold(draft)


def test_an_unknown_decision_or_confidence_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "d.tsv"
    bad.write_text("abcd1234\tMAYBE\t\tHIGH\tf\tr\n", encoding="utf-8")
    with pytest.raises(LinkerGoldError, match="unknown decision"):
        BUILDER.read_decisions([bad])
    bad.write_text("abcd1234\tSELECTED\tI269\tVERY\tf\tr\n", encoding="utf-8")
    with pytest.raises(LinkerGoldError, match="unknown confidence"):
        BUILDER.read_decisions([bad])


def test_a_duplicate_decision_for_one_record_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "d.tsv"
    bad.write_text(
        "abcd1234\tABSTAIN\t\tLOW\tf\tr\nabcd1234\tABSTAIN\t\tLOW\tf\tr\n", encoding="utf-8"
    )
    with pytest.raises(LinkerGoldError, match="duplicate decision"):
        BUILDER.read_decisions([bad])


# --- drafts never overwrite a human ---------------------------------------------


def test_a_human_decision_is_never_overwritten_by_a_draft(tmp_path: Path) -> None:
    """The writer skips any record a person has already adjudicated."""
    pack = tmp_path / "pack.jsonl"
    human = pack_record(
        adjudication_status=STATUS_HUMAN_GOLD,
        selected_codes=["I269"],
        no_valid_candidate=False,
    )
    pack.write_text(json.dumps(human) + "\n", encoding="utf-8")
    decisions = tmp_path / "d.tsv"
    decisions.write_text(
        f"{human['annotation_id'][4:12]}\tNO_VALID_CANDIDATE\t\tHIGH\tf\tr\n", encoding="utf-8"
    )
    out = tmp_path / "drafts"
    BUILDER.main(["--pack", str(pack), "--decisions", str(decisions), "--out", str(out)])
    written = [
        json.loads(line)
        for line in (out / "claude_code_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert written == [], "a record already adjudicated by a human must be skipped"
    # And the pack itself is untouched.
    assert json.loads(pack.read_text(encoding="utf-8").strip())["selected_codes"] == ["I269"]


def test_drafts_are_written_outside_the_pack() -> None:
    """The draft file is a separate artifact; pack.jsonl is never the drafts' home."""
    assert BUILDER.OUT.name == "ai_drafts"
    assert BUILDER.PACK.name == "pack.jsonl"
    assert BUILDER.OUT != BUILDER.PACK.parent


# --- promotion is never automatic -------------------------------------------------


def test_no_code_path_promotes_a_draft_to_human_gold() -> None:
    """Nothing in the writer may emit a human-gold status."""
    source = WRITER.read_text(encoding="utf-8")
    assert "STATUS_HUMAN_GOLD" not in source
    assert "SECOND_REVIEW_AGREED" not in source
    assert "STATUS_AI_ASSISTED_DRAFT" in source


def test_the_private_draft_artifacts_are_gitignored() -> None:
    for relative in (
        "data/private_linker_gold/0065/ai_drafts/claude_code_v1.jsonl",
        "data/private_linker_gold/0065/ai_drafts/manifest.json",
    ):
        assert (
            subprocess.run(
                ["git", "check-ignore", "-q", relative],
                cwd=REPO,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ), f"{relative} must be gitignored"
