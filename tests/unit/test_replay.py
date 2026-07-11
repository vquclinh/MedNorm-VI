"""Determinism, replay manifest, HTML escaping, and UTF-8 tests."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.evaluation import evaluate_corpus
from mednorm_vi.evaluation.loading import load_ground_truth, load_predictions
from mednorm_vi.evaluation.models import (
    EvaluationConfig,
    EvaluationDocument,
    EvaluationEntity,
    EvaluationRunMetadata,
    Provenance,
)
from mednorm_vi.evaluation.replay import (
    aggregate_dir_hash,
    build_replay_manifest,
    directory_file_hashes,
)
from mednorm_vi.evaluation.reporting_html import render_html
from mednorm_vi.evaluation.reporting_json import write_data_reports

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "evaluation"
CFG = EvaluationConfig.from_mapping({"matching_strategy": "exact-text-occurrence"})


def _outcome():
    gt, _, _ = load_ground_truth(FIX / "gold", provenance=None)
    pred, _ = load_predictions(FIX / "predictions")
    return evaluate_corpus(gt, pred, CFG)


def test_data_reports_are_deterministic(tmp_path: Path) -> None:
    out = _outcome()
    mapping = {"evaluator_version": "provisional-v1"}
    h1 = write_data_reports(tmp_path / "a", out, mapping)
    h2 = write_data_reports(tmp_path / "b", out, mapping)
    assert h1 == h2  # identical inputs -> identical report hashes


def test_scores_reproduce() -> None:
    a = _outcome().corpus
    b = _outcome().corpus
    assert a.final_score == b.final_score
    assert a.text_score == b.text_score
    assert a.candidates_score == b.candidates_score


def test_replay_manifest_stable_hashes() -> None:
    m1 = build_replay_manifest(
        config=CFG, config_mapping={"x": 1}, ground_truth_dir=FIX / "gold",
        predictions_dir=FIX / "predictions", ground_truth_provenance="SYNTHETIC",
        prediction_experiment_id=None, timestamp="fixed")
    m2 = build_replay_manifest(
        config=CFG, config_mapping={"x": 1}, ground_truth_dir=FIX / "gold",
        predictions_dir=FIX / "predictions", ground_truth_provenance="SYNTHETIC",
        prediction_experiment_id=None, timestamp="fixed")
    assert m1.config_hash == m2.config_hash
    assert m1.ground_truth_dir_hash == m2.ground_truth_dir_hash
    assert m1.ground_truth_file_hashes == m2.ground_truth_file_hashes


def test_dir_hash_matches_manual() -> None:
    fh = directory_file_hashes(FIX / "gold")
    assert "1.json" in fh
    assert aggregate_dir_hash(fh) == aggregate_dir_hash(dict(reversed(list(fh.items()))))


def test_html_escapes_user_text() -> None:
    gt = {"1": EvaluationDocument("1", (
        EvaluationEntity("<script>alert(1)</script>", "TRIỆU_CHỨNG", 0, 26),
    ), provenance=Provenance.SYNTHETIC)}
    pred = {"1": EvaluationDocument("1", (
        EvaluationEntity("<script>alert(1)</script>", "TRIỆU_CHỨNG", 0, 26),
    ))}
    out = evaluate_corpus(gt, pred, CFG)
    meta = EvaluationRunMetadata("v", "2026-01-01T00:00:00Z", "3.12", "linux", None, None)
    html = render_html(out, meta, "SYNTHETIC")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_html_preserves_vietnamese_without_escapes() -> None:
    out = _outcome()
    meta = EvaluationRunMetadata("v", "2026-01-01T00:00:00Z", "3.12", "linux", None, None)
    html = render_html(out, meta, "SYNTHETIC")
    assert "PROVISIONAL LOCAL EVALUATOR" in html
    assert "viêm phổi" in html  # Vietnamese preserved, not \u-escaped


def test_json_reports_preserve_vietnamese(tmp_path: Path) -> None:
    out = _outcome()
    write_data_reports(tmp_path, out, {"x": 1})
    text = (tmp_path / "entity_pairs.jsonl").read_text(encoding="utf-8")
    assert "viêm phổi" in text
    assert "\\u" not in text
