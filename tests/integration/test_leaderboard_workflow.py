"""End-to-end experiment/leaderboard workflow (create → attach → score → compare)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from mednorm_vi.experiments.cli import main
from mednorm_vi.experiments.models import LocalScoreKind
from mednorm_vi.experiments.registry import ExperimentRegistry


def _zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(1, 4):
            zf.writestr(f"output/{i}.json", "[]")
    return path


def test_full_workflow(tmp_path: Path, capsys) -> None:
    reg_dir = tmp_path / "registry"

    def run(*args: str) -> int:
        return main(["--registry", str(reg_dir), *args])

    assert run("create", "--title", "baseline deterministic", "--seed", "42") == 0
    assert run("create", "--title", "second experiment") == 0

    z = _zip(tmp_path / "output.zip")
    assert run("attach-output", "--experiment", "EXP-0001", "--zip", str(z)) == 0
    assert run("record-local", "--experiment", "EXP-0001", "--kind", "gold",
               "--score", "0.83") == 0
    assert run("record-local", "--experiment", "EXP-0001", "--kind", "synthetic",
               "--score", "0.90") == 0
    assert run("record-leaderboard", "--experiment", "EXP-0001", "--score", "0.71234",
               "--submission-id", "sub-1") == 0
    assert run("show", "--experiment", "EXP-0001") == 0
    assert run("compare", "EXP-0001", "EXP-0002") == 0

    out = capsys.readouterr().out
    assert "leaderboard score never replaces local error analysis" in out

    reg = ExperimentRegistry(reg_dir)
    rec = reg.load("EXP-0001")
    assert rec.seed == 42
    assert rec.output_zip_hash is not None
    assert rec.score_of_kind(LocalScoreKind.GOLD).final_score == 0.83
    assert rec.score_of_kind(LocalScoreKind.SYNTHETIC).final_score == 0.90
    assert rec.leaderboard is not None and rec.leaderboard.score == 0.71234


def test_no_network_import_surface() -> None:
    # The experiments package must not import networking libraries.
    import mednorm_vi.experiments as exp

    src = Path(exp.__file__).parent
    for py in src.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for banned in ("import requests", "import urllib.request", "http://", "https://"):
            assert banned not in text, f"{py.name} references {banned}"
