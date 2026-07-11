"""Experiment registry + leaderboard tracking tests (no network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from mednorm_vi.experiments.leaderboard import (
    attach_output,
    compare,
    record_leaderboard,
    record_local_score,
)
from mednorm_vi.experiments.models import LocalScoreKind
from mednorm_vi.experiments.registry import ExperimentExistsError, ExperimentRegistry


def _registry(tmp_path: Path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "registry")


def _zip(path: Path, payload: str = "[]") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("output/1.json", payload)
    return path


def test_create_and_deterministic_ids(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    a = reg.create(title="first")
    b = reg.create(title="second")
    assert a.experiment_id == "EXP-0001"
    assert b.experiment_id == "EXP-0002"


def test_refuse_overwrite(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    rec = reg.create(title="x")
    with pytest.raises(ExperimentExistsError):
        reg.save(rec, overwrite=False)


def test_attach_output_hash(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.create(title="x")
    z = _zip(tmp_path / "output.zip")
    updated = attach_output(reg, "EXP-0001", z)
    assert updated.output_zip_hash
    assert "output/1.json" in updated.prediction_file_hashes


def test_output_hash_changes_with_content(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.create(title="x")
    z1 = _zip(tmp_path / "a.zip", payload="[]")
    h1 = attach_output(reg, "EXP-0001", z1).output_zip_hash
    z2 = _zip(tmp_path / "b.zip", payload='[{"text":"x"}]')
    h2 = attach_output(reg, "EXP-0001", z2).output_zip_hash
    assert h1 != h2


def test_local_scores_separated_from_leaderboard(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.create(title="x")
    record_local_score(reg, "EXP-0001", kind=LocalScoreKind.GOLD, final_score=0.8)
    record_local_score(reg, "EXP-0001", kind=LocalScoreKind.SILVER, final_score=0.6)
    rec = record_leaderboard(reg, "EXP-0001", score=0.71, submission_id="s1")
    assert rec.score_of_kind(LocalScoreKind.GOLD).final_score == 0.8
    assert rec.score_of_kind(LocalScoreKind.SILVER).final_score == 0.6
    assert rec.leaderboard is not None and rec.leaderboard.score == 0.71


def test_record_local_replaces_same_kind(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.create(title="x")
    record_local_score(reg, "EXP-0001", kind=LocalScoreKind.GOLD, final_score=0.5)
    rec = record_local_score(reg, "EXP-0001", kind=LocalScoreKind.GOLD, final_score=0.9)
    golds = [s for s in rec.local_scores if s.kind is LocalScoreKind.GOLD]
    assert len(golds) == 1 and golds[0].final_score == 0.9


def test_git_state_preserved(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    rec = reg.create(title="x")
    # git_commit may be None outside a repo; the field must exist and round-trip.
    loaded = reg.load("EXP-0001")
    assert loaded.git_commit == rec.git_commit
    assert loaded.git_dirty == rec.git_dirty


def test_compare_two_experiments(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.create(title="a")
    reg.create(title="b")
    record_local_score(reg, "EXP-0001", kind=LocalScoreKind.GOLD, final_score=0.8)
    rows = compare([reg.load("EXP-0001"), reg.load("EXP-0002")])
    assert rows[0]["local_gold"] == 0.8
    assert rows[1]["local_gold"] is None


def test_registry_json_is_sorted_and_diff_friendly(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.create(title="x")
    text = (tmp_path / "registry" / "EXP-0001.json").read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["experiment_id"] == "EXP-0001"
    # sorted keys => deterministic diffs
    assert list(data.keys()) == sorted(data.keys())
