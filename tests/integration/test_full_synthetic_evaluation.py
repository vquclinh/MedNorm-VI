"""A complete synthetic 100-document evaluation run."""

from __future__ import annotations

import json
from pathlib import Path

from mednorm_vi.evaluation import evaluate_corpus
from mednorm_vi.evaluation.loading import load_ground_truth, load_predictions
from mednorm_vi.evaluation.models import EvaluationConfig
from mednorm_vi.evaluation.reporting_json import write_data_reports

CFG = EvaluationConfig.from_mapping({"matching_strategy": "exact-text-occurrence"})


def _doc(i: int) -> tuple[str, list[dict], list[dict]]:
    text = f"Bệnh nhân {i} sốt cao, chẩn đoán viêm phổi; táo bón rồi táo bón."
    a = text.index("táo bón")
    b = text.index("táo bón", a + 1)
    sot = text.index("sốt")
    vp = text.index("viêm phổi")
    gold = [
        {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [sot, sot + 3],
         "route_case": "C3"},
        {"text": "viêm phổi", "type": "CHẨN_ĐOÁN", "candidates": ["J18.9"], "assertions": [],
         "position": [vp, vp + len("viêm phổi")], "route_case": "C3"},
        {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": ["isHistorical"],
         "position": [a, a + 7], "route_case": "C7"},
        {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [b, b + 7],
         "route_case": "C7"},
    ]
    # Predictions: correct except every 5th doc drops the diagnosis candidate.
    pred = []
    for e in gold:
        pe = {k: v for k, v in e.items() if k != "route_case"}
        if i % 5 == 0 and e["type"] == "CHẨN_ĐOÁN":
            pe = {**pe, "candidates": []}
        pred.append(pe)
    return text, gold, pred


def _build_corpus(root: Path) -> tuple[Path, Path]:
    gt = root / "gold"
    pred = root / "pred"
    gt.mkdir()
    pred.mkdir()
    for i in range(1, 101):
        text, gold, prediction = _doc(i)
        for e in gold:
            s, en = e["position"]
            assert text[s:en] == e["text"]  # end-exclusive invariant
        (gt / f"{i}.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        (pred / f"{i}.json").write_text(json.dumps(prediction, ensure_ascii=False),
                                        encoding="utf-8")
    (gt / "manifest.yaml").write_text(
        "dataset_id: synth100\nversion: 1\nprovenance: SYNTHETIC\nlabeled: true\n",
        encoding="utf-8")
    return gt, pred


def test_full_100_document_run(tmp_path: Path) -> None:
    gt_dir, pred_dir = _build_corpus(tmp_path)
    gt, prov, gt_res = load_ground_truth(gt_dir)
    pred, pred_res = load_predictions(pred_dir)
    assert gt_res.ok and pred_res.ok
    assert len(gt) == 100 and len(pred) == 100

    out = evaluate_corpus(gt, pred, CFG)
    assert out.corpus.n_documents == 100
    # 4 entities/doc, all matched (text/positions align) -> 400 matched.
    assert out.corpus.n_matched == 400
    assert out.corpus.n_missing == 0
    assert out.corpus.n_spurious == 0
    # text + assertions perfect; candidates imperfect (20 docs drop a code).
    assert out.corpus.text_score == 1.0
    assert out.corpus.assertions_score == 1.0
    assert out.corpus.candidates_score < 1.0
    assert 0.9 < out.corpus.final_score < 1.0

    hashes = write_data_reports(tmp_path / "rep", out, {"x": 1})
    assert set(hashes) >= {"summary.json", "entity_pairs.jsonl", "diagnostics.jsonl"}


def test_perfect_predictions_score_one(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for i in range(1, 101):
        text, gold, _ = _doc(i)
        payload = json.dumps([{k: v for k, v in e.items() if k != "route_case"} for e in gold],
                             ensure_ascii=False)
        (gt_dir / f"{i}.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        (pred_dir / f"{i}.json").write_text(payload, encoding="utf-8")
    (gt_dir / "manifest.yaml").write_text(
        "dataset_id: s\nversion: 1\nprovenance: SYNTHETIC\nlabeled: true\n", encoding="utf-8")
    gt, _, _ = load_ground_truth(gt_dir)
    pred, _ = load_predictions(pred_dir)
    out = evaluate_corpus(gt, pred, CFG)
    assert out.corpus.final_score == 1.0
