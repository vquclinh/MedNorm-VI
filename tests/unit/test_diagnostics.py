"""Diagnostic emission tests."""

from __future__ import annotations

from mednorm_vi.evaluation import evaluate_corpus
from mednorm_vi.evaluation.models import (
    EvaluationConfig,
    EvaluationDocument,
    EvaluationEntity,
    Provenance,
)

CFG = EvaluationConfig.from_mapping({"matching_strategy": "exact-text-occurrence"})


def _run(gt_ents, pred_ents, prov=Provenance.SYNTHETIC):
    gt = {"1": EvaluationDocument("1", tuple(gt_ents), provenance=prov)}
    pred = {"1": EvaluationDocument("1", tuple(pred_ents))}
    return evaluate_corpus(gt, pred, CFG).diagnostics.counts


def _e(text, etype, s, e, cands=(), asserts=()):
    return EvaluationEntity(text, etype, s, e, assertions=asserts, candidates=cands)


def test_missing_and_spurious() -> None:
    counts = _run([_e("sốt", "TRIỆU_CHỨNG", 0, 3)], [_e("ho", "TRIỆU_CHỨNG", 5, 7)])
    assert counts.get("missing_entity") == 1
    assert counts.get("spurious_entity") == 1


def test_wrong_type_diagnostic() -> None:
    counts = _run([_e("Ho khan", "TRIỆU_CHỨNG", 0, 7)],
                  [_e("Ho khan", "CHẨN_ĐOÁN", 0, 7, cands=("R05",))])
    assert counts.get("wrong_type") == 1


def test_duplicate_candidate_diagnostic() -> None:
    counts = _run([_e("dx", "CHẨN_ĐOÁN", 0, 2, cands=("A", "A"))],
                  [_e("dx", "CHẨN_ĐOÁN", 0, 2, cands=("A",))])
    assert counts.get("duplicate_candidate") == 1


def test_duplicate_assertion_diagnostic() -> None:
    counts = _run([_e("sốt", "TRIỆU_CHỨNG", 0, 3, asserts=("isNegated", "isNegated"))],
                  [_e("sốt", "TRIỆU_CHỨNG", 0, 3, asserts=("isNegated",))])
    assert counts.get("duplicate_assertion") == 1


def test_candidate_wrong_ontology_diagnostic() -> None:
    # THUỐC candidate must be numeric RxCUI; a non-numeric string is flagged.
    counts = _run([_e("para", "THUỐC", 0, 4, cands=("1049640",))],
                  [_e("para", "THUỐC", 0, 4, cands=("J18.9",))])
    assert counts.get("candidate_wrong_ontology") == 1


def test_text_token_substitution_diagnostic() -> None:
    counts = _run([_e("viêm phổi cấp", "CHẨN_ĐOÁN", 0, 13, cands=("J18.9",))],
                  [_e("viêm phổi cấp", "CHẨN_ĐOÁN", 0, 13, cands=("J18.9",))])
    assert "text_token_substitution" not in counts  # identical text


def test_silver_label_warning() -> None:
    counts = _run([_e("sốt", "TRIỆU_CHỨNG", 0, 3)],
                  [_e("sốt", "TRIỆU_CHỨNG", 0, 3)], prov=Provenance.SILVER)
    assert counts.get("silver_label_warning") == 1
