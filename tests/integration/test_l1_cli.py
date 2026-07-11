"""L1 CLI integration tests (summary, determinism, negative cases)."""

from __future__ import annotations

import json
from pathlib import Path

from mednorm_vi.document_intelligence.cli import main

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "document_intelligence" / "synthetic_multisection.txt"
CONFIG = REPO / "configs" / "document_intelligence" / "base.yaml"


def test_cli_success_and_output(tmp_path: Path, capsys) -> None:
    out = tmp_path / "graph.json"
    rc = main(["--input", str(FIXTURE), "--config", str(CONFIG), "--output", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "MedNorm-VI L1 Document Intelligence" in printed
    assert "graph validation  : OK" in printed
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "l1-graph-1"
    assert data["nodes"]


def test_cli_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    main(["--input", str(FIXTURE), "--config", str(CONFIG), "--output", str(a)])
    main(["--input", str(FIXTURE), "--config", str(CONFIG), "--output", str(b)])
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_cli_invalid_utf8_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00broken")
    rc = main(["--input", str(bad), "--config", str(CONFIG), "--output", str(tmp_path / "o.json")])
    assert rc == 2


def test_cli_inspect_mode(tmp_path: Path, capsys) -> None:
    rc = main(["--input", str(FIXTURE), "--config", str(CONFIG), "--inspect"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "section hierarchy" in printed
