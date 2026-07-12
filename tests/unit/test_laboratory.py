"""Laboratory parser + pairing tests (Phase 1B proposals)."""

from __future__ import annotations

from pathlib import Path

from _routing_helpers import forced_routings

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.mention_factory.laboratory import load_lab_lexicon, parse_graph

REPO = Path(__file__).resolve().parents[2]
LEX = load_lab_lexicon(REPO / "configs" / "laboratory" / "parser_v1.yaml")


def _run(text: str):
    g = analyze_text(text)
    routings = forced_routings(g, ("C2",))  # force lab routing on every unit
    return parse_graph(g, routings, LEX, "lab-p1"), g


def _names(res) -> list[str]:
    return [p.text for p in res.proposals if "TÊN_XÉT_NGHIỆM" in p.proposed_types]


def _results(res) -> list[str]:
    return [p.text for p in res.proposals if "KẾT_QUẢ_XÉT_NGHIỆM" in p.proposed_types]


def _pairs(res) -> set[tuple[str, str]]:
    by_id = {p.proposal_id: p.text for p in res.proposals}
    return {(by_id[r.source_proposal_id], by_id[r.target_proposal_id]) for r in res.relations}


def test_simple_key_value() -> None:
    res, _ = _run("WBC: 14.43")
    assert "WBC" in _names(res) and "14.43" in _results(res)
    assert ("WBC", "14.43") in _pairs(res)


def test_decimal_comma() -> None:
    res, _ = _run("WBC: 14,43")
    rv = [c for p in res.proposals for c in p.components if c.role == "result_value"]
    assert any(c.text == "14,43" and c.normalized == "14.43" for c in rv)


def test_percent_and_unit() -> None:
    res, _ = _run("NEUT%: 76.4\nGlucose: 7.8 mmol/L")
    assert "NEUT%" in _names(res)
    assert any(c.role == "unit" and c.text == "mmol/L"
               for p in res.proposals for c in p.components)


def test_reference_range_and_flag() -> None:
    res, _ = _run("Glucose: 7.8 mmol/L (3.9-6.4)\nHGB: 120 g/L H")
    roles = {c.role for p in res.proposals for c in p.components}
    assert "reference_range" in roles and "flag" in roles


def test_qualitative_vietnamese() -> None:
    res, _ = _run("HIV: âm tính")
    assert "âm tính" in " ".join(_results(res))


def test_inequality_value() -> None:
    res, _ = _run("CRP: <5 mg/L")
    assert any("<5" in t for t in _results(res))


def test_semicolon_multiple_pairs() -> None:
    res, _ = _run("WBC: 14.43; NEUT%: 76.4; LYMPH%: 12.8")
    assert _pairs(res) == {("WBC", "14.43"), ("NEUT%", "76.4"), ("LYMPH%", "12.8")}


def test_tab_separated_row() -> None:
    res, _ = _run("HGB\t120\tg/L")
    assert "HGB" in _names(res) and "120" in _results(res)


def test_narrative_value() -> None:
    res, _ = _run("Glucose máu là 180 mg/dL.")
    assert ("Glucose máu", "180") in _pairs(res)


def test_no_fabrication_from_vague_narrative() -> None:
    res, _ = _run("Xét nghiệm glucose tăng.")
    assert _results(res) == []


def test_unknown_test_with_structure() -> None:
    res, _ = _run("XYZ99: 42")
    assert "XYZ99" in _names(res)
    assert any("unknown_test_name" in p.warnings for p in res.proposals)


def test_boundary_alternatives_value_and_unit() -> None:
    res, _ = _run("Glucose: 7.8 mmol/L")
    vals = _results(res)
    assert "7.8" in vals and "7.8 mmol/L" in vals


def test_pairing_cost_recorded() -> None:
    res, _ = _run("WBC: 14.43")
    assert all(isinstance(r.pairing_cost, float) for r in res.relations)


def test_unrelated_rows_do_not_pair() -> None:
    res, _ = _run("Xét nghiệm:\nWBC: 14.43\n\nGlucose: 7.8")
    pairs = _pairs(res)
    assert ("WBC", "7.8") not in pairs and ("Glucose", "14.43") not in pairs


def test_medication_dose_not_lab_result() -> None:
    res, _ = _run("Liều: 500 mg")  # bare med dose, not a lab unit
    assert _results(res) == []


def test_dates_ages_times_rejected() -> None:
    for line in ("Ngày: 12/05", "Tuổi: 54", "Giờ: 09:30"):
        res, _ = _run(line)
        assert _results(res) == [], line


def test_exact_offsets() -> None:
    res, g = _run("WBC: 14,43; Glucose: 7.8 mmol/L")
    for p in res.proposals:
        assert g.original_text[p.start : p.end] == p.text
        for c in p.components:
            assert g.original_text[c.start : c.end] == c.text


def test_repeated_tests_distinct_positions() -> None:
    res, _ = _run("Xét nghiệm:\nWBC: 1\n\nWBC: 2")
    wbc = [p for p in res.proposals if p.text == "WBC"]
    assert len({p.start for p in wbc}) == 2


def test_deterministic() -> None:
    a, _ = _run("WBC: 14.43; NEUT%: 76.4")
    b, _ = _run("WBC: 14.43; NEUT%: 76.4")
    assert [(p.proposal_id, p.start, p.end) for p in a.proposals] == \
           [(p.proposal_id, p.start, p.end) for p in b.proposals]
    assert [(r.relation_id, r.pairing_cost) for r in a.relations] == \
           [(r.relation_id, r.pairing_cost) for r in b.relations]
