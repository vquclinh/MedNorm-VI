"""AI-to-AI adjudication handoff (Audit 0067).

Two properties carry this milestone and both are easy to lose silently:

* **The first packet is blind.** A second opinion that has already seen the first is a
  ratification, and the agreement rate it produces would be meaningless. The export is
  therefore asserted to contain no decision, confidence, rationale or recommended
  candidate.
* **No AI status is human gold.** `AI_SECOND_OPINION` and `AI_CONSENSUS_HIGH` join the
  existing AI statuses outside `HUMAN_GOLD_STATUSES`. Two models agreeing is the strongest
  signal available without a person, and it is still not ground truth: models that share a
  blind spot agree just as readily as models that are right.
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
    STATUS_AI_CONSENSUS_HIGH,
    STATUS_AI_SECOND_OPINION,
    LinkerGoldError,
    assert_not_ai_gold,
    is_human_gold,
)

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data" / "private_linker_gold" / "0065"
BLIND = BASE / "chatgpt_handoff" / "minimum_benchmark_blind.jsonl"
QUEUE = BASE / "ai_drafts" / "queues" / "queue_minimum_benchmark.txt"
PACK = BASE / "pack.jsonl"


def _module(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXPORTER = _module("export_chatgpt_handoff_0067")
IMPORTER = _module("import_ai_second_opinion")
CONSENSUS = _module("build_ai_consensus_0067")

requires_packet = pytest.mark.skipif(
    not BLIND.is_file(), reason="blind packet not exported locally"
)


# --- statuses are never gold ---------------------------------------------------


def test_second_opinion_and_consensus_are_never_human_gold() -> None:
    assert_not_ai_gold()
    for status in (STATUS_AI_SECOND_OPINION, STATUS_AI_CONSENSUS_HIGH):
        assert status in AI_DRAFT_STATUSES
        assert status not in HUMAN_GOLD_STATUSES
        assert not is_human_gold({"adjudication_status": status, "selected_codes": ["I269"]})


def test_consensus_writer_never_emits_a_human_gold_status() -> None:
    source = (REPO / "scripts" / "build_ai_consensus_0067.py").read_text(encoding="utf-8")
    assert "STATUS_HUMAN_GOLD" not in source
    assert "SECOND_REVIEW_AGREED" not in source
    assert "STATUS_AI_CONSENSUS_HIGH" in source


# --- the blind export ----------------------------------------------------------


@requires_packet
def test_the_blind_export_contains_no_first_opinion() -> None:
    rows = [json.loads(line) for line in BLIND.read_text(encoding="utf-8").splitlines() if line]
    banned = set(EXPORTER.FORBIDDEN_KEYS)
    record_keys = set().union(*(set(r) for r in rows))
    assert not (record_keys & banned), sorted(record_keys & banned)
    candidate_keys = set().union(*(set(c) for r in rows for c in r["offered_candidates"]))
    assert not (candidate_keys & banned), sorted(candidate_keys & banned)
    # linker_rank IS allowed: it is factual metadata, not a recommendation.
    assert "linker_rank" in candidate_keys


@requires_packet
def test_every_queued_record_was_exported() -> None:
    expected = {
        line.strip() for line in QUEUE.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    exported = {
        json.loads(line)["annotation_id"]
        for line in BLIND.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert exported == expected
    assert len(exported) == 80


@requires_packet
def test_the_markdown_packet_does_not_name_the_first_opinion() -> None:
    text = (BASE / "chatgpt_handoff" / "minimum_benchmark_blind.md").read_text(encoding="utf-8")
    for needle in ("AI_ASSISTED_DRAFT", "ai_decision", "Claude", "recommended candidate"):
        assert needle not in text, needle


def test_the_exporter_refuses_to_emit_a_forbidden_field() -> None:
    """The guard is structural, not a convention."""
    assert "ai_decision" in EXPORTER.FORBIDDEN_KEYS
    assert "rationale" in EXPORTER.FORBIDDEN_KEYS
    assert "linker_top1" in EXPORTER.FORBIDDEN_KEYS


# --- privacy -------------------------------------------------------------------


def test_no_handoff_artifact_is_tracked_or_trackable() -> None:
    for relative in (
        "data/private_linker_gold/0065/chatgpt_handoff/minimum_benchmark_blind.jsonl",
        "data/private_linker_gold/0065/chatgpt_handoff/minimum_benchmark_blind.md",
        "data/private_linker_gold/0065/ai_second_opinion/chatgpt_second_opinion.jsonl",
        "data/private_linker_gold/0065/ai_consensus/consensus.jsonl",
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
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert not any(name.startswith("data/private_linker_gold/") for name in tracked)


def test_the_response_schema_is_tracked_but_the_responses_are_not() -> None:
    assert (REPO / "schemas" / "annotation" / "ai_second_opinion_v1.schema.json").is_file()
    assert (
        subprocess.run(
            ["git", "check-ignore", "-q", "schemas/annotation/ai_second_opinion_v1.schema.json"],
            cwd=REPO,
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ), "the schema is a public contract and must be trackable"


# --- importer validation -------------------------------------------------------


def packet_fixture(tmp_path: Path) -> Path:
    rows = [
        {
            "annotation_id": "lg1-0000000000000001",
            "ontology": "ICD10",
            "offered_candidates": [{"code_dotless": "I269", "linker_rank": 1}],
        },
        {
            "annotation_id": "lg1-0000000000000002",
            "ontology": "RXNORM",
            "offered_candidates": [{"rxcui": "161", "linker_rank": 1}],
        },
    ]
    path = tmp_path / "blind.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def decision(annotation_id: str, ontology: str, **overrides) -> dict:
    row = {
        "annotation_id": annotation_id,
        "ontology": ontology,
        "decision": "SELECTED",
        "selected_codes": ["I269" if ontology == "ICD10" else "161"],
        "confidence": "HIGH",
        "rationale": "r",
        "evidence_used": ["mention_text"],
        "competing_codes": [],
        "ambiguity_flags": [],
    }
    row.update(overrides)
    return row


def load_packet(path: Path) -> dict:
    return {
        json.loads(line)["annotation_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_importer_rejects_an_unknown_annotation_id(tmp_path: Path) -> None:
    packet = load_packet(packet_fixture(tmp_path))
    rows = [decision("lg1-000000000000dead", "ICD10")]
    with pytest.raises(LinkerGoldError, match="unknown annotation_id"):
        IMPORTER.validate(rows, packet, allow_partial=True)


def test_importer_rejects_a_duplicate_annotation_id(tmp_path: Path) -> None:
    packet = load_packet(packet_fixture(tmp_path))
    rows = [decision("lg1-0000000000000001", "ICD10")] * 2
    with pytest.raises(LinkerGoldError, match="duplicate decision"):
        IMPORTER.validate(rows, packet, allow_partial=True)


def test_importer_rejects_a_code_that_was_never_offered(tmp_path: Path) -> None:
    packet = load_packet(packet_fixture(tmp_path))
    rows = [decision("lg1-0000000000000001", "ICD10", selected_codes=["Z999"])]
    with pytest.raises(LinkerGoldError, match="never offered"):
        IMPORTER.validate(rows, packet, allow_partial=True)


def test_importer_rejects_a_dotted_icd_code(tmp_path: Path) -> None:
    path = packet_fixture(tmp_path)
    packet = load_packet(path)
    packet["lg1-0000000000000001"]["offered_candidates"][0]["code_dotless"] = "I26.9"
    rows = [decision("lg1-0000000000000001", "ICD10", selected_codes=["I26.9"])]
    with pytest.raises(LinkerGoldError, match="dotless"):
        IMPORTER.validate(rows, packet, allow_partial=True)


def test_importer_rejects_a_mismatched_ontology(tmp_path: Path) -> None:
    packet = load_packet(packet_fixture(tmp_path))
    rows = [decision("lg1-0000000000000001", "RXNORM", selected_codes=[])]
    rows[0]["decision"] = "NO_VALID_CANDIDATE"
    with pytest.raises(LinkerGoldError, match="does not match"):
        IMPORTER.validate(rows, packet, allow_partial=True)


def test_importer_requires_every_record_unless_partial_is_explicit(tmp_path: Path) -> None:
    packet = load_packet(packet_fixture(tmp_path))
    rows = [decision("lg1-0000000000000001", "ICD10")]
    with pytest.raises(LinkerGoldError, match="have no decision"):
        IMPORTER.validate(rows, packet, allow_partial=False)
    accepted = IMPORTER.validate(rows, packet, allow_partial=True)
    assert len(accepted) == 1
    assert accepted[0]["adjudication_status"] == STATUS_AI_SECOND_OPINION


def test_a_non_selected_decision_may_not_carry_codes(tmp_path: Path) -> None:
    packet = load_packet(packet_fixture(tmp_path))
    rows = [decision("lg1-0000000000000001", "ICD10", decision="ABSTAIN")]
    with pytest.raises(LinkerGoldError, match="must not carry selected codes"):
        IMPORTER.validate(rows, packet, allow_partial=True)


# --- consensus classification --------------------------------------------------


def opinion(decision_value: str, codes: list[str], confidence: str = "HIGH") -> dict:
    return {
        "ai_decision": decision_value,
        "selected_codes": codes,
        "ai_confidence": confidence,
        "ambiguity_reason": "",
        "ambiguity_flags": [],
    }


def test_consensus_categories() -> None:
    c = CONSENSUS
    assert c.classify(opinion("SELECTED", ["A"]), opinion("SELECTED", ["A"])) == c.EXACT_AGREEMENT
    assert (
        c.classify(opinion("SELECTED", ["A"]), opinion("SELECTED", ["B"]))
        == c.DECISION_AGREEMENT_CODE_DISAGREEMENT
    )
    assert (
        c.classify(opinion("SELECTED", ["A"]), opinion("NO_VALID_CANDIDATE", []))
        == c.CLAUDE_SELECTED_CHATGPT_REJECTED
    )
    assert (
        c.classify(opinion("NO_VALID_CANDIDATE", []), opinion("SELECTED", ["A"]))
        == c.CHATGPT_SELECTED_CLAUDE_REJECTED
    )
    assert c.classify(opinion("ABSTAIN", []), opinion("ABSTAIN", [])) == c.BOTH_ABSTAIN
    assert c.classify(opinion("ABSTAIN", []), opinion("SELECTED", ["A"])) == c.CLAUDE_ABSTAIN
    assert c.classify(opinion("SELECTED", ["A"]), opinion("ABSTAIN", [])) == c.CHATGPT_ABSTAIN
    assert (
        c.classify(opinion("NO_VALID_CANDIDATE", []), opinion("NO_VALID_CANDIDATE", []))
        == c.EXACT_AGREEMENT
    )


def test_high_consensus_requires_exact_code_agreement() -> None:
    record = {
        "ontology": "ICD10",
        "offered_candidates": [{"code_dotless": "A"}, {"code_dotless": "B"}],
    }
    ok, _ = CONSENSUS.eligible_for_high_consensus(
        opinion("SELECTED", ["A"]), opinion("SELECTED", ["A"]), record, CONSENSUS.EXACT_AGREEMENT
    )
    assert ok
    ok, reason = CONSENSUS.eligible_for_high_consensus(
        opinion("SELECTED", ["A"]),
        opinion("SELECTED", ["B"]),
        record,
        CONSENSUS.DECISION_AGREEMENT_CODE_DISAGREEMENT,
    )
    assert not ok and "category" in reason


def test_low_confidence_or_flagged_ambiguity_blocks_high_consensus() -> None:
    record = {"ontology": "ICD10", "offered_candidates": [{"code_dotless": "A"}]}
    ok, reason = CONSENSUS.eligible_for_high_consensus(
        opinion("SELECTED", ["A"]),
        opinion("SELECTED", ["A"], confidence="LOW"),
        record,
        CONSENSUS.EXACT_AGREEMENT,
    )
    assert not ok and "LOW confidence" in reason

    flagged = opinion("SELECTED", ["A"])
    flagged["ambiguity_flags"] = ["parent_child_ambiguity"]
    ok, reason = CONSENSUS.eligible_for_high_consensus(
        opinion("SELECTED", ["A"]), flagged, record, CONSENSUS.EXACT_AGREEMENT
    )
    assert not ok and "ambiguity" in reason


def test_a_consensus_code_absent_from_the_kb_offer_set_is_refused() -> None:
    record = {"ontology": "ICD10", "offered_candidates": [{"code_dotless": "A"}]}
    ok, reason = CONSENSUS.eligible_for_high_consensus(
        opinion("SELECTED", ["ZZZ"]),
        opinion("SELECTED", ["ZZZ"]),
        record,
        CONSENSUS.EXACT_AGREEMENT,
    )
    assert not ok and "frozen KB" in reason


# --- the pack is never mutated --------------------------------------------------


@requires_packet
def test_no_tool_in_this_milestone_writes_to_the_pack() -> None:
    for name in (
        "export_chatgpt_handoff_0067",
        "import_ai_second_opinion",
        "build_ai_consensus_0067",
    ):
        source = (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "pack.write_text" not in source
        assert "write_atomically" not in source
