"""Evaluator CLI integration tests (exit codes + reports + fail-fast)."""

from __future__ import annotations

import json
from pathlib import Path

from mednorm_vi.evaluation.cli import main

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "evaluation"
CONFIG = REPO / "configs" / "evaluation" / "provisional_v1.yaml"


def _run(gt: Path, pred: Path, report: Path, extra: list[str] | None = None) -> int:
    argv = ["--ground-truth", str(gt), "--predictions", str(pred),
            "--config", str(CONFIG), "--report-dir", str(report)]
    return main(argv + (extra or []))


def test_cli_success_and_reports(tmp_path: Path, capsys) -> None:
    rc = _run(FIX / "gold", FIX / "predictions", tmp_path / "rep")
    assert rc == 0
    out = capsys.readouterr().out
    assert "PROVISIONAL LOCAL EVALUATOR" in out
    for name in ("summary.json", "per_document.json", "entity_pairs.jsonl",
                 "unmatched_entities.jsonl", "diagnostics.jsonl", "config_snapshot.json",
                 "replay_manifest.json", "report.html"):
        assert (tmp_path / "rep" / name).is_file(), name
    summary = json.loads((tmp_path / "rep" / "summary.json").read_text(encoding="utf-8"))
    assert summary["banner"] == "PROVISIONAL LOCAL EVALUATOR"
    assert 0.0 <= summary["final_score"] <= 1.0


def test_cli_rejects_malformed_predictions(tmp_path: Path) -> None:
    bad = tmp_path / "pred"
    bad.mkdir()
    (bad / "1.json").write_text("{not json", encoding="utf-8")
    rc = _run(FIX / "gold", bad, tmp_path / "rep")
    assert rc == 3


def test_cli_rejects_english_type_predictions(tmp_path: Path) -> None:
    bad = tmp_path / "pred"
    bad.mkdir()
    (bad / "1.json").write_text(
        json.dumps([{"text": "sốt", "type": "SYMPTOM", "assertions": [], "position": [0, 3]}]),
        encoding="utf-8")
    rc = _run(FIX / "gold", bad, tmp_path / "rep")
    assert rc == 3


def test_cli_rejects_organizer_test_provenance(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "1.json").write_text(
        json.dumps([{"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [0, 3]}]),
        encoding="utf-8")
    (gt / "manifest.yaml").write_text(
        "dataset_id: x\nversion: 1\nprovenance: ORGANIZER_TEST\nlabeled: true\n", encoding="utf-8")
    rc = _run(gt, FIX / "predictions", tmp_path / "rep")
    assert rc == 2


def test_cli_permits_low_quality_but_valid_predictions(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    # valid organizer entities but wrong everything -> low score, still rc 0
    (pred / "1.json").write_text(
        json.dumps([{"text": "zzz", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [0, 3]}]),
        encoding="utf-8")
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "1.json").write_text(
        json.dumps([{"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [0, 3]}]),
        encoding="utf-8")
    (gt / "manifest.yaml").write_text(
        "dataset_id: x\nversion: 1\nprovenance: SYNTHETIC\nlabeled: true\n", encoding="utf-8")
    rc = _run(gt, pred, tmp_path / "rep")
    assert rc == 0
