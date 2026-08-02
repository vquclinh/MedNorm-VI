"""Guards for sprint submissions #3, #4, #5. No weights, no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mednorm_vi.inference.config import PipelineConfig
from mednorm_vi.linking.semantic import null_gate as ng
from mednorm_vi.linking.semantic import runtime as rt
from mednorm_vi.reasoner import budget

ROOT = Path(__file__).resolve().parents[2]
NULLIFY = ROOT / "scripts/make_null_candidates_0075.py"
REASONER = ROOT / "scripts/run_8b_reasoner_0075.py"

spec = importlib.util.spec_from_file_location("nullify_0075", NULLIFY)
assert spec and spec.loader
nullify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nullify)


# ------------------------------------------------------- #3 all-null candidate diagnostic


def test_nullify_empties_candidates_and_changes_nothing_else(tmp_path: Path) -> None:
    rows = [
        {
            "text": "viêm phổi",
            "type": "CHẨN_ĐOÁN",
            "position": [0, 9],
            "assertions": ["isNegated"],
            "candidates": ["J18.9", "J18.0"],
        },
        {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [10, 13], "assertions": []},
        {"text": "glucose", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [14, 21]},
    ]
    source = tmp_path / "s"
    source.mkdir()
    (source / "1.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "o"
    assert (
        nullify.main(
            ["--source-dir", str(source), "--output-dir", str(out), "--expected-documents", "1"]
        )
        == 0
    )
    after = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert after[0]["candidates"] == []
    assert after[0]["assertions"] == ["isNegated"]  # untouched
    assert after[0]["position"] == [0, 9]  # untouched
    assert after[1] == rows[1]  # symptom untouched entirely
    assert after[2] == rows[2]  # lab entity untouched entirely
    assert [e["text"] for e in after] == [r["text"] for r in rows]


def test_nullify_requires_only_e3() -> None:
    """No embedding, reranker or 8B is referenced by the #3 path."""
    text = NULLIFY.read_text(encoding="utf-8")
    for banned in ("Qwen3", "Embedding-4B", "Reranker", "torch"):
        assert banned not in text


def test_nullify_fails_closed_on_document_count(tmp_path: Path) -> None:
    empty = tmp_path / "s"
    empty.mkdir()
    assert (
        nullify.main(
            [
                "--source-dir",
                str(empty),
                "--output-dir",
                str(tmp_path / "o"),
                "--expected-documents",
                "100",
            ]
        )
        == 2
    )


# ----------------------------------------------------------- #4 conservative NULL gate


def test_conservative_profile_differs_from_public_best_only_in_null_policy() -> None:
    best = PipelineConfig.load(ROOT / "configs/pipeline/public_best_0073_legacy.yaml")
    gated = PipelineConfig.load(ROOT / "configs/pipeline/conservative_null_0075.yaml")
    assert best.linker_backend == gated.linker_backend
    assert best.semantic_linker["row_order"] == gated.semantic_linker["row_order"]
    assert best.semantic_linker["null_mode"] == "shadow"
    assert gated.semantic_linker["null_mode"] == "conservative"
    differing = {
        k
        for k in gated.semantic_linker
        if gated.semantic_linker.get(k) != best.semantic_linker.get(k)
    }
    assert differing == {"null_mode", "structured_rxnorm"}


def test_conservative_null_is_now_an_accepted_mode() -> None:
    settings = rt.SemanticLinkerSettings(null_mode="conservative")
    assert not [p for p in rt.missing_requirements(settings) if "null_mode" in p]
    bogus = rt.SemanticLinkerSettings(null_mode="aggressive")
    assert [p for p in rt.missing_requirements(bogus) if "null_mode" in p]


@pytest.mark.parametrize("tier", sorted(ng.PROTECTED_TIERS))
def test_conservative_gate_never_refuses_exact_governed_evidence(tier: str) -> None:
    decision = ng.evaluate(
        ng.NullGateEvidence(
            top_tier=tier, candidate_count=5, reranker_score=0.0, source_count=1, ontology="RXNORM"
        )
    )
    assert decision.emits


def test_conservative_gate_refuses_a_bare_drug_class() -> None:
    decision = ng.evaluate(
        ng.NullGateEvidence(
            candidate_count=8, ontology="RXNORM", mention_text="kháng sinh", reranker_score=0.9
        )
    )
    assert decision.decision == ng.DECISION_NO_VALID


def test_structured_contradiction_is_refusal_evidence_not_reordering() -> None:
    source = (ROOT / "src/mednorm_vi/linking/semantic/hybrid.py").read_text(encoding="utf-8")
    assert "unsupported_extra_tokens=unsupported_detail" in source
    # The rejected Audit-0074 reordering must NOT be wired into the emission path.
    assert "rs.reorder(" not in source


# --------------------------------------------------------------- #5 legal candidate_8b


def test_variant_d_refuses_a_seed_that_deploys_the_4b_pair() -> None:
    text = REASONER.read_text(encoding="utf-8")
    assert 'if "semantic_s1_0072" in seed_text' in text
    assert "16,369,296,389" in text
    assert "--seed-run-config" in text


def test_variant_d_parameter_total_is_legal() -> None:
    assert budget.assert_within_cap() == 8_325_737_477
    with pytest.raises(budget.ParameterBudgetExceeded):
        budget.assert_within_cap(with_semantic_s1=True)


def test_no_free_form_entity_discovery_in_variant_d() -> None:
    """cand_8b must consume the seed entities, never re-discover them."""
    text = REASONER.read_text(encoding="utf-8")
    assert 'if args.config == "cand_8b":\n            proposals = [dict(e) for e in seed]' in text
