"""Medication grammar tests (Phase 1B proposals)."""

from __future__ import annotations

from pathlib import Path

from _routing_helpers import forced_routings

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.mention_factory.medication import load_medication_lexicon, parse_graph

REPO = Path(__file__).resolve().parents[2]
LEX = load_medication_lexicon(REPO / "configs" / "medication" / "grammar_v1.yaml")


def _run(text: str):
    g = analyze_text(text)
    routings = forced_routings(g, ("C1",))  # force med routing on every unit
    return parse_graph(g, routings, LEX, "med-g1"), g


def _texts(text: str) -> list[str]:
    res, _ = _run(text)
    return [p.text for p in res.proposals]


def _kinds(text: str) -> set[str]:
    res, _ = _run(text)
    return {p.matched_rule.split(":")[-1] for p in res.proposals if p.matched_rule}


def test_name_only() -> None:
    assert "amlodipine" in _texts("amlodipine")


def test_name_strength() -> None:
    assert "amlodipine 10 mg" in _texts("amlodipine 10 mg")


def test_name_strength_route() -> None:
    assert "amlodipine 10 mg po" in _texts("amlodipine 10 mg po")


def test_full_boundary() -> None:
    assert "amlodipine 10 mg po daily" in _texts("amlodipine 10 mg po daily")


def test_multiple_boundary_candidates() -> None:
    res, _ = _run("amlodipine 10 mg po daily")
    same = [p for p in res.proposals if p.parse_ref == res.proposals[0].parse_ref]
    assert len({(p.start, p.end) for p in same}) >= 3


def test_decimal_dot_and_comma_preserved() -> None:
    res, _ = _run("amlodipine 2,5 mg")
    sv = [c for p in res.proposals for c in p.components if c.role == "strength_value"]
    assert any(c.text == "2,5" and c.normalized == "2.5" for c in sv)  # text unchanged


def test_attached_unit() -> None:
    assert "paracetamol 500mg" in _texts("paracetamol 500mg")


def test_range_and_endash() -> None:
    assert "paracetamol 325-650 mg" in _texts("paracetamol 325-650 mg")
    assert "paracetamol 325–650 mg" in _texts("paracetamol 325–650 mg")


def test_concentration() -> None:
    res, _ = _run("amoxicillin 250 mg/5 mL")
    assert any(c.role == "concentration" for p in res.proposals for c in p.components)


def test_release_dose_route_frequency_prn_duration() -> None:
    res, _ = _run("metformin 500 mg ER tablet po bid khi cần x 7 ngày")
    roles = {c.role for p in res.proposals for c in p.components}
    assert {"release", "dose_form", "route", "frequency", "prn", "duration"} <= roles


def test_multi_ingredient() -> None:
    res, _ = _run("paracetamol 500 mg and ibuprofen 200 mg")
    refs = {p.parse_ref for p in res.proposals}
    names = {p.normalized_form for p in res.proposals}
    assert len(refs) == 2 and {"paracetamol", "ibuprofen"} <= names


def test_list_marker_and_delimiter_excluded() -> None:
    texts = _texts("1. amlodipine 10 mg")
    assert "amlodipine 10 mg" in texts
    assert not any(t.startswith("1.") for t in texts)


def test_exact_offsets() -> None:
    res, g = _run("Dùng amlodipine 10 mg mỗi sáng.")
    for p in res.proposals:
        assert g.original_text[p.start : p.end] == p.text
        for c in p.components:
            assert g.original_text[c.start : c.end] == c.text


def test_repeated_names_distinct_offsets() -> None:
    res, _ = _run("aspirin 81 mg và aspirin 81 mg")
    names = [p for p in res.proposals if p.matched_rule and p.matched_rule.endswith("name_only")]
    starts = {p.start for p in names}
    assert len(starts) == 2


def test_unknown_name_with_structure() -> None:
    res, _ = _run("Zynovate 20 mg tablet")
    assert res.proposals
    assert all("unknown_medication_name" in p.warnings for p in res.proposals)


def test_incomplete_parse_warning() -> None:
    res, _ = _run("amlodipine")
    assert any("incomplete_parse_no_strength" in p.warnings for p in res.proposals)


def test_hard_negatives_no_medication() -> None:
    for line in ("Bệnh nhân 54 tuổi.", "cao 170 cm", "nặng 68 kg", "Vào viện lúc 09:30"):
        assert _texts(line) == [], line


def test_bare_unit_without_name() -> None:
    # A bare unit / number with no preceding word must not invent a medication.
    assert _texts("mg") == []
    assert _texts("... 10 mg") == []


def test_deterministic() -> None:
    a, _ = _run("1. amlodipine 10 mg po daily")
    b, _ = _run("1. amlodipine 10 mg po daily")
    assert [(p.proposal_id, p.start, p.end) for p in a.proposals] == \
           [(p.proposal_id, p.start, p.end) for p in b.proposals]
