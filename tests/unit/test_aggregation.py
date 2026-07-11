"""Scoring + aggregation tests via evaluate_corpus."""

from __future__ import annotations

import dataclasses

from mednorm_vi.evaluation import evaluate_corpus
from mednorm_vi.evaluation.models import (
    EvaluationConfig,
    EvaluationDocument,
    EvaluationEntity,
    Provenance,
)

CFG = EvaluationConfig.from_mapping({
    "matching_strategy": "exact-text-occurrence",
    "cost_weights": {"token_wer": 0.5, "char_overlap": 0.3, "position_distance": 0.15,
                     "boundary_length": 0.05, "exact_text_bonus": 0.5},
})


def _doc(entities, doc_id="1", prov=Provenance.SYNTHETIC):
    return {doc_id: EvaluationDocument(doc_id, tuple(entities), provenance=prov)}


def _pred(entities, doc_id="1"):
    return {doc_id: EvaluationDocument(doc_id, tuple(entities))}


def _e(text, etype, s, e, cands=(), asserts=()):
    return EvaluationEntity(text, etype, s, e, assertions=asserts, candidates=cands)


def test_exact_match_perfect_final_score() -> None:
    ents = [_e("viêm phổi", "CHẨN_ĐOÁN", 0, 9, cands=("J18.9",))]
    out = evaluate_corpus(_doc(ents), _pred([dataclasses.replace(ents[0])]), CFG)
    assert out.corpus.final_score == 1.0
    assert out.corpus.n_matched == 1


def test_missing_prediction_penalized() -> None:
    ents = [_e("sốt", "TRIỆU_CHỨNG", 0, 3)]
    out = evaluate_corpus(_doc(ents), _pred([]), CFG)
    assert out.corpus.n_missing == 1
    assert out.corpus.text_score == 0.0


def test_spurious_prediction_penalized() -> None:
    out = evaluate_corpus(_doc([]), _pred([_e("sốt", "TRIỆU_CHỨNG", 0, 3)]), CFG)
    assert out.corpus.n_spurious == 1
    assert out.corpus.text_score == 0.0


def test_wrong_type_is_double_error() -> None:
    gt = [_e("Ho khan", "TRIỆU_CHỨNG", 0, 7)]
    pred = [_e("Ho khan", "CHẨN_ĐOÁN", 0, 7, cands=("R05",))]
    out = evaluate_corpus(_doc(gt), _pred(pred), CFG)
    assert out.corpus.n_missing == 1  # GT symptom unmatched
    assert out.corpus.n_spurious == 1  # pred diagnosis unmatched
    assert out.corpus.n_matched == 0


def test_candidate_weighting_partial() -> None:
    gt = [_e("dx", "CHẨN_ĐOÁN", 0, 2, cands=("A", "B", "C"))]
    pred = [_e("dx", "CHẨN_ĐOÁN", 0, 2, cands=("A",))]
    out = evaluate_corpus(_doc(gt), _pred(pred), CFG)
    # candidate jaccard = 1/3; only one slot, weight len(GT)+1=4 -> component 1/3
    assert abs(out.corpus.candidates_score - 1 / 3) < 1e-9


def test_aggregation_policies_differ() -> None:
    gt = _doc([_e("a", "THUỐC", 0, 1, cands=("1",)), _e("b", "THUỐC", 2, 3, cands=("2", "3"))])
    pred = _pred([_e("a", "THUỐC", 0, 1, cands=("1",)), _e("b", "THUỐC", 2, 3, cands=("2",))])
    prov = evaluate_corpus(gt, pred, dataclasses.replace(CFG, aggregation_policy="provisional-v1"))
    micro = evaluate_corpus(gt, pred,
                            dataclasses.replace(CFG, aggregation_policy="micro-entity-diagnostic"))
    # weighted (provisional) vs unweighted (micro) candidate aggregation differ here.
    assert prov.corpus.candidates_score != micro.corpus.candidates_score


def test_per_type_and_per_case_present() -> None:
    gt = _doc([_e("sốt", "TRIỆU_CHỨNG", 0, 3)])
    gt["1"] = dataclasses.replace(
        gt["1"], entities=(dataclasses.replace(gt["1"].entities[0], route_case="C3"),))
    out = evaluate_corpus(gt, _pred([_e("sốt", "TRIỆU_CHỨNG", 0, 3)]), CFG)
    assert "TRIỆU_CHỨNG" in out.corpus.per_type
    assert "C3" in out.corpus.per_case
