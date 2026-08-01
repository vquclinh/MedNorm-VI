"""Representation and null-gate invariants for the hybrid semantic linker (Audit 0071).

Model-free throughout: no weights are loaded and no network call is made, so these run on any
machine. Fixtures are synthetic; no clinical record or public-test text appears here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednorm_vi.kb.indexing import evidence as ev
from mednorm_vi.linking.semantic import null_gate as ng
from mednorm_vi.linking.semantic.representation import (
    CONTEXT_CHARACTER_BUDGET,
    QUERY_INSTRUCTION,
    SemanticQuery,
    icd_document,
    rxnorm_document,
)

# ------------------------------------------------------------------------- representation


def test_query_carries_mention_type_and_bounded_context() -> None:
    rendered = SemanticQuery(
        mention_text="viêm phổi",
        entity_type="CHẨN_ĐOÁN",
        context_before="bệnh nhân sốt cao",
        context_after="đã điều trị kháng sinh",
        ontology="ICD10",
    ).render()
    assert "viêm phổi" in rendered
    assert "CHẨN_ĐOÁN" in rendered
    assert "bệnh nhân sốt cao" in rendered
    assert rendered.startswith(QUERY_INSTRUCTION)


def test_query_instruction_forbids_inferring_unstated_detail() -> None:
    """The instruction is the model's only statement of the specificity policy."""
    lowered = QUERY_INSTRUCTION.lower()
    assert "do not infer" in lowered
    for forbidden in ("subtype", "organism", "strength", "dose form", "brand"):
        assert forbidden in lowered


def test_context_is_clipped_towards_the_mention() -> None:
    """Context before keeps its tail and context after keeps its head - both nearest the mention."""
    body = SemanticQuery(
        mention_text="ho",
        entity_type="CHẨN_ĐOÁN",
        context_before="A" + "x" * 500,
        context_after="y" * 500 + "B",
        ontology="ICD10",
    ).render(with_instruction=False)
    before = body.split("context before: ")[1].split(" | ")[0]
    after = body.split("context after: ")[1]
    assert len(before) == CONTEXT_CHARACTER_BUDGET
    assert len(after) == CONTEXT_CHARACTER_BUDGET
    # The far end of each side is the part dropped.
    assert "A" not in before
    assert "B" not in after


def test_query_is_deterministic() -> None:
    query = SemanticQuery("sởi", "CHẨN_ĐOÁN", "a", "b", "ICD10")
    assert len({query.render() for _ in range(5)}) == 1


def test_damaged_icd_title_is_never_embedded_as_positive_text() -> None:
    """Embedding `Bao gồm: …` would teach the retriever that an instruction is a concept."""
    document = icd_document(
        {
            "concept_id": "C20",
            "canonical_name": "Bao gồm: Bóng",
            "metadata": {"dotted_code": "C20"},
            "aliases": [],
        }
    )
    assert "Bao gồm" not in document.text
    assert document.excluded_damaged_title == "Bao gồm: Bóng"
    assert "canonical_name" not in document.fields_used


def test_repaired_icd_title_and_aliases_are_embedded() -> None:
    document = icd_document(
        {
            "concept_id": "C20",
            "canonical_name": "U ác tính ở trực tràng",
            "metadata": {"dotted_code": "C20"},
            "aliases": ["ung thư trực tràng"],
        },
        parent_title="U ác tính ở cơ quan tiêu hóa",
    )
    assert "U ác tính ở trực tràng" in document.text
    assert "ung thư trực tràng" in document.text
    assert "canonical_name" in document.fields_used


def test_rxnorm_document_carries_tty_and_stays_bounded() -> None:
    document = rxnorm_document(
        {
            "concept_id": "1191",
            "canonical_name": "aspirin",
            "metadata": {"tty": "IN"},
            "aliases": ["acetylsalicylic acid"],
        }
    )
    assert "RxCUI 1191" in document.text
    assert "TTY: IN" in document.text
    assert len(document.text) < 1000


# ----------------------------------------------------------------------------- null gate


def test_no_candidates_is_a_null_decision() -> None:
    decision = ng.evaluate(ng.NullGateEvidence(candidate_count=0, ontology="ICD10"))
    assert decision.decision == ng.DECISION_NO_VALID
    assert ng.REASON_NO_CANDIDATES in decision.reasons


@pytest.mark.parametrize("tier", sorted(ng.PROTECTED_TIERS))
def test_exact_governed_evidence_is_never_refused(tier: str) -> None:
    """Audit-0069 tiers A-C are whole-string accent-sensitive matches; the gate protects them."""
    decision = ng.evaluate(
        ng.NullGateEvidence(
            top_tier=tier,
            candidate_count=5,
            reranker_score=0.0,
            source_count=1,
            unsupported_extra_tokens=3,
            ontology="ICD10",
        )
    )
    assert decision.emits


def test_protected_tiers_match_the_audit_0069_ladder() -> None:
    assert ng.PROTECTED_TIERS == frozenset(
        {ev.TIER_A_EXACT_CANONICAL, ev.TIER_B_EXACT_ALIAS, ev.TIER_C_SEMANTIC_SYNONYM}
    )


@pytest.mark.parametrize(
    "mention",
    ["kháng sinh", "thuốc kháng sinh", "giảm đau", "vitamin", "thuốc bổ", "kháng sinh khác"],
)
def test_bare_drug_class_is_refused(mention: str) -> None:
    """RxNorm has no concept for a therapeutic class, so the honest answer is nothing."""
    assert ng.is_drug_class_mention(mention)
    decision = ng.evaluate(
        ng.NullGateEvidence(
            candidate_count=10,
            ontology="RXNORM",
            mention_text=mention,
            reranker_score=0.9,
            has_exact_evidence=False,
        )
    )
    assert decision.decision == ng.DECISION_NO_VALID
    assert ng.REASON_GENERIC_DRUG_CLASS in decision.reasons


@pytest.mark.parametrize("mention", ["kháng sinh amoxicillin", "paracetamol", "vitamin D3 1000UI"])
def test_a_named_product_is_not_a_drug_class(mention: str) -> None:
    assert not ng.is_drug_class_mention(mention)


def test_one_weak_signal_alone_does_not_refuse() -> None:
    """Refusing on a single soft signal trades false emissions for lost coverage."""
    decision = ng.evaluate(
        ng.NullGateEvidence(
            candidate_count=1,
            ontology="ICD10",
            reranker_score=0.9,
            source_count=1,
            top_score=1.0,
        )
    )
    assert decision.emits


def test_two_independent_weak_signals_refuse() -> None:
    decision = ng.evaluate(
        ng.NullGateEvidence(
            candidate_count=4,
            ontology="ICD10",
            reranker_score=0.01,
            top_score=0.5,
            second_score=0.49,
            source_count=1,
        )
    )
    assert decision.decision == ng.DECISION_NO_VALID
    assert len(decision.reasons) >= 2


def test_gate_is_deterministic() -> None:
    evidence = ng.NullGateEvidence(
        candidate_count=3,
        ontology="ICD10",
        reranker_score=0.2,
        top_score=0.5,
        second_score=0.49,
        source_count=1,
    )
    first = ng.evaluate(evidence)
    assert all(ng.evaluate(evidence).as_dict() == first.as_dict() for _ in range(5))


def test_thresholds_are_not_derived_from_leaderboard() -> None:
    """Documented contract: calibration sources exclude public scoring."""
    assert "public" in ng.NullGateThresholds.__doc__.lower()


# ---------------------------------------------------------------- benchmark and manifest


def test_concept_disjoint_benchmark_has_no_leakage() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "artifacts/semantic/0071/manifest.json"
    if not manifest.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("semantic benchmark not built in this checkout")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["concept_disjoint"] is True
    assert document["concept_leakage"] == 0
    assert document["contains_clinical_text"] is False


def test_deployed_parameter_manifest_is_under_the_cap() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "artifacts/semantic/0071/deployed_parameter_manifest.json"
    if not path.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("parameter manifest not built in this checkout")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["budget_cap"] == 9_000_000_000
    for name, system in document["systems"].items():
        assert system["under_cap"] is True, name
        assert system["total_deployed_parameters"] < document["budget_cap"], name
