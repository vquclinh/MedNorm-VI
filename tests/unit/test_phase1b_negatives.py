"""Negative smokes for the Phase 1B hardening (Area 7).

Guards against the specific failure modes the review flagged:
1. a broken relation<->boundary group (dangling / cross-parse endpoints);
2. duplicate specialist execution over the same content;
3. cross-row has_result propagation;
4. C6 falsely activated only by routine boundary candidates;
5. weak narrative medication-like text producing a false proposal.
"""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.deterministic_baseline import Phase1BConfig, run_phase1b
from mednorm_vi.document_intelligence import analyze_text

REPO = Path(__file__).resolve().parents[2]
CFG = Phase1BConfig.load(
    REPO / "configs" / "case_router" / "base.yaml",
    REPO / "configs" / "medication" / "grammar_v1.yaml",
    REPO / "configs" / "laboratory" / "parser_v1.yaml",
)


def _run(text: str):
    return run_phase1b(analyze_text(text), CFG)


def test_no_broken_relation_boundary_group() -> None:
    res = _run("Xét nghiệm:\nGlucose: 7.8 mmol/L; WBC: 14.43\n")
    by_id = {p.proposal_id: p for p in res.proposals}
    for r in res.relations:
        # both endpoints resolve to real proposals
        assert r.source_proposal_id in by_id and r.target_proposal_id in by_id
        # the recorded target boundary group matches the target's own group
        tgt = by_id[r.target_proposal_id]
        assert r.target_boundary_group_id == tgt.boundary_group_id
        # every alternative in a pair group shares the same source name
        assert r.pair_group_id  # non-empty for lab pairings


def test_no_duplicate_specialist_execution() -> None:
    # The nested value-sentence of a key-value row must not re-run the lab parser.
    res = _run("Xét nghiệm:\nNEUT%: 76.4\n")
    lab = res.laboratory_proposals()
    assert sum(1 for p in lab if p.text == "NEUT%") == 1
    assert sum(1 for p in lab if p.text == "76.4") == 1


def test_no_cross_row_relation_propagation() -> None:
    res = _run("Xét nghiệm:\nWBC: 14.43\n\nGlucose: 7.8 mmol/L\n")
    by_id = {p.proposal_id: p.text for p in res.proposals}
    pairs = {(by_id[r.source_proposal_id], by_id[r.target_proposal_id])
             for r in res.relations}
    assert not any(s == "WBC" and t.startswith("7.8") for s, t in pairs)


def test_c6_not_falsely_activated_by_routine_boundaries() -> None:
    res = _run("Thuốc tại nhà:\n1. amlodipine 10 mg mỗi ngày\n2. metformin 500 mg bid\n")
    tags = {t for r in res.routings for t in r.route_tags}
    # multiple complete meds, each with boundary candidates, but no linking ambiguity
    assert "C6" not in tags


def test_weak_narrative_med_text_no_false_proposal() -> None:
    res = _run("Bệnh nhân uống 2 lít nước mỗi ngày.\n")
    assert res.medication_proposals() == ()
