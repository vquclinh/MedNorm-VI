"""Variant F: E3-constrained verifier guards (sprint 0076). No weights, no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mednorm_vi.reasoner.prompt import VERIFY_POLICY, verify_prompt
from mednorm_vi.reasoner.validator import ORGANIZER_TYPES
from mednorm_vi.reasoner.verifier import (
    VerifierDiagnostics,
    apply_decisions,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_e3_verified_8b_0076.py"
spec = importlib.util.spec_from_file_location("f0076", SCRIPT)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

E3 = [
    {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": []},
    {
        "text": "viêm phổi",
        "type": "CHẨN_ĐOÁN",
        "position": [5, 14],
        "assertions": [],
        "candidates": ["J18.9"],
    },
    {"text": "glucose", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [16, 23]},
]


def run(decisions: list[dict] | None) -> tuple[list[dict], VerifierDiagnostics]:
    diagnostics = VerifierDiagnostics()
    return apply_decisions(E3, decisions, diagnostics), diagnostics


def test_text_and_position_always_come_from_e3() -> None:
    """A hallucinated span is unrepresentable: the model never supplies text or offsets."""
    out, _ = run(
        [
            {
                "id": 0,
                "keep": True,
                "type": "TRIỆU_CHỨNG",
                "text": "HALLUCINATED",
                "position": [999, 1000],
            },
        ]
    )
    kept = next(e for e in out if e["text"] == "sốt")
    assert kept["position"] == [0, 3]
    assert "HALLUCINATED" not in [e["text"] for e in out]


def test_model_cannot_add_an_entity() -> None:
    out, diagnostics = run([{"id": 99, "keep": True, "type": "TRIỆU_CHỨNG"}])
    assert len(out) == len(E3)
    assert diagnostics.invalid_decisions["unknown_proposal_id"] == 1


def test_duplicate_proposal_id_is_rejected() -> None:
    out, diagnostics = run(
        [
            {"id": 0, "keep": False},
            {"id": 0, "keep": True},
        ]
    )
    assert diagnostics.invalid_decisions["duplicate_proposal_id"] == 1
    assert "sốt" not in [e["text"] for e in out]  # first decision applied


def test_drop_removes_the_proposal() -> None:
    out, diagnostics = run([{"id": 0, "keep": False}])
    assert [e["text"] for e in out] == ["viêm phổi", "glucose"]
    assert diagnostics.dropped == 1
    assert diagnostics.drop_by_type["TRIỆU_CHỨNG"] == 1


def test_unknown_type_is_rejected_and_e3_type_survives() -> None:
    out, diagnostics = run([{"id": 0, "keep": True, "type": "SYMPTOM"}])
    assert out[0]["type"] == "TRIỆU_CHỨNG"
    assert diagnostics.invalid_decisions["unknown_type"] == 1


def test_valid_type_change_is_applied_and_counted() -> None:
    out, diagnostics = run([{"id": 0, "keep": True, "type": "CHẨN_ĐOÁN"}])
    assert out[0]["type"] == "CHẨN_ĐOÁN"
    assert diagnostics.type_changes == 1
    assert diagnostics.type_change_pairs["TRIỆU_CHỨNG->CHẨN_ĐOÁN"] == 1


def test_non_boolean_assertion_is_rejected() -> None:
    out, diagnostics = run(
        [
            {"id": 0, "keep": True, "type": "TRIỆU_CHỨNG", "assertions": {"isNegated": "yes"}},
        ]
    )
    assert out[0]["assertions"] == []
    assert diagnostics.invalid_decisions["non_boolean_assertion"] == 1


def test_assertions_applied_for_assertion_types_only() -> None:
    out, _ = run(
        [
            {
                "id": 0,
                "keep": True,
                "type": "TRIỆU_CHỨNG",
                "assertions": {"isNegated": True, "isFamily": False, "isHistorical": False},
            },
            {
                "id": 2,
                "keep": True,
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "assertions": {"isNegated": True},
            },
        ]
    )
    assert out[0]["assertions"] == ["isNegated"]
    lab = next(e for e in out if e["type"] == "KẾT_QUẢ_XÉT_NGHIỆM")
    assert "assertions" not in lab


def test_candidates_are_always_empty() -> None:
    out, _ = run([{"id": 1, "keep": True, "type": "CHẨN_ĐOÁN"}])
    diagnosis = next(e for e in out if e["type"] == "CHẨN_ĐOÁN")
    assert diagnosis["candidates"] == []


def test_parse_failure_falls_back_to_e3_unchanged() -> None:
    """The fallback reproduces the currently best-scoring behaviour and invents nothing."""
    out, diagnostics = run(None)
    assert diagnostics.parse_fallbacks == 1
    assert [e["text"] for e in out] == [e["text"] for e in E3]
    assert all(e.get("candidates") == [] for e in out if "candidates" in e)


def test_diagnostics_capture_everything_needed_to_interpret_the_result() -> None:
    _, diagnostics = run([{"id": 0, "keep": False}])
    report = diagnostics.as_dict()
    for key in (
        "e3_proposals_in",
        "kept",
        "dropped",
        "type_changes",
        "assertion_changes",
        "invalid_decisions",
        "parse_fallbacks",
        "keep_by_type",
        "drop_by_type",
    ):
        assert key in report
    assert report["candidate_policy"] == "forced_all_null"


def test_prompt_is_keep_first_and_forbids_new_spans() -> None:
    text = verify_prompt("note", E3)
    assert VERIFY_POLICY in text
    lowered = text.lower()
    assert "the default for every proposal is keep" in lowered
    assert "uncertainty means keep" in lowered
    assert "correction layer, not a pruning layer" in lowered
    assert "removing a correct entity is a real loss" in lowered
    assert "drop only when there is strong affirmative evidence" in lowered
    assert "Do NOT output entity text or character positions" in text
    assert "Do NOT add new entities" in text
    for organizer_type in ORGANIZER_TYPES:
        assert organizer_type in text


def test_prompt_schema_is_omission_based_to_reduce_fallbacks() -> None:
    text = verify_prompt("note", E3)
    assert "Return an entry ONLY for a proposal you want to CHANGE" in text
    assert "is KEPT UNCHANGED" in text
    assert '{"decisions":[]}' in text


@pytest.mark.parametrize("reply", ["", "no json here", '{"decisions": "nope"}', "{bad"])
def test_unusable_replies_trigger_the_fallback(reply: str) -> None:
    assert runner.parse_decisions(reply) is None


def test_greedy_decoding_with_no_sampling_parameters() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "do_sample=False" in source
    assert 'for name in ("temperature", "top_p", "top_k")' in source
    assert "setattr(config, name, None)" in source


def test_no_4b_pair_and_no_candidate_retrieval_anywhere() -> None:
    """No 4B model is *used*. The strings may appear in the refusal message - that is the
    guard telling the operator why the illegal seed was rejected."""
    source = SCRIPT.read_text(encoding="utf-8")
    for banned in ("Qwen3Reranker(", "build_pool(", "load_index(", "torch.load("):
        assert banned not in source, banned
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    joined = " ".join(imports)
    for banned in ("reranker", "hybrid", "pool", "retrieval"):
        assert banned not in joined, banned
    assert 'if "semantic_s1_0072" in seed_text' in source
