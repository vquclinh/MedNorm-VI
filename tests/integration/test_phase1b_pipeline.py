"""Phase 1B pipeline + CLI integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from mednorm_vi.deterministic_baseline import Phase1BConfig, run_phase1b
from mednorm_vi.deterministic_baseline.cli import main
from mednorm_vi.deterministic_baseline.serialization import determinism_hash
from mednorm_vi.document_intelligence import analyze_document

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "phase1b"
L1 = REPO / "configs" / "document_intelligence" / "base.yaml"
ROUTER = REPO / "configs" / "case_router" / "base.yaml"
MED = REPO / "configs" / "medication" / "grammar_v1.yaml"
LAB = REPO / "configs" / "laboratory" / "parser_v1.yaml"
CONFIG = Phase1BConfig.load(ROUTER, MED, LAB)


def _graph(name: str):
    return analyze_document(FIX / name)


def test_multi_section_pipeline() -> None:
    g = _graph("synthetic_medical_document.txt")
    res = run_phase1b(g, CONFIG)
    assert res.l1_valid and res.proposals_valid
    assert res.medication_proposals() and res.laboratory_proposals()
    # has_result relations present and valid
    ids = {p.proposal_id for p in res.proposals}
    assert res.relations
    for r in res.relations:
        assert r.source_proposal_id in ids and r.target_proposal_id in ids


def test_multi_route_nodes_present() -> None:
    g = _graph("synthetic_medical_document.txt")
    res = run_phase1b(g, CONFIG)
    assert any(len(r.cases) >= 2 for r in res.routings)


def test_no_final_organizer_json_emitted() -> None:
    g = _graph("synthetic_medical_document.txt")
    res = run_phase1b(g, CONFIG)
    for p in res.proposals:
        # proposals are evidence: no candidates/assertions/position-final fields
        assert "candidates" not in p.features and "assertions" not in p.features


def test_medication_list_fixture_counts() -> None:
    g = _graph("synthetic_medication_list.txt")
    res = run_phase1b(g, CONFIG)
    names = {p.normalized_form for p in res.medication_proposals()}
    assert {"amlodipine", "paracetamol", "metformin", "amoxicillin"} <= names


def test_laboratory_fixture_pairs() -> None:
    g = _graph("synthetic_laboratory.txt")
    res = run_phase1b(g, CONFIG)
    by_id = {p.proposal_id: p.text for p in res.proposals}
    pairs = {(by_id[r.source_proposal_id], by_id[r.target_proposal_id]) for r in res.relations}
    assert ("WBC", "14,43") in pairs
    assert ("NEUT%", "76.4") in pairs


def test_hard_negative_fixture_no_false_proposals() -> None:
    g = _graph("synthetic_hard_negatives.txt")
    res = run_phase1b(g, CONFIG)
    # dates/ages/phones/addresses/percent-prose must not yield medical proposals
    assert res.medication_proposals() == ()
    assert res.laboratory_proposals() == ()


def test_cli_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    args = ["--input", str(FIX / "synthetic_medical_document.txt"),
            "--l1-config", str(L1), "--router-config", str(ROUTER),
            "--medication-config", str(MED), "--laboratory-config", str(LAB)]
    assert main([*args, "--output", str(a)]) == 0
    assert main([*args, "--output", str(b)]) == 0
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    data = json.loads(a.read_text(encoding="utf-8"))
    assert data["schema_version"] == "phase1b-1"


def test_cli_invalid_utf8_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00broken")
    rc = main(["--input", str(bad), "--l1-config", str(L1), "--router-config", str(ROUTER),
               "--medication-config", str(MED), "--laboratory-config", str(LAB)])
    assert rc == 2


def test_determinism_hash_stable() -> None:
    g = _graph("synthetic_medical_document.txt")
    assert determinism_hash(run_phase1b(g, CONFIG)) == determinism_hash(run_phase1b(g, CONFIG))
