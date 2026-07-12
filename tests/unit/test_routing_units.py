"""Canonical L1 routing units + duplicate-specialist prevention (Area 3).

The router consumes canonical routable units (list-item content, table/key-value
rows, sentence-like spans, line fallback) — not only whole lines — applying a
specificity policy so a parent line is never routed through a specialist that an
eligible child already represents.
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


def _routings(text: str):
    return run_phase1b(analyze_text(text), CFG)


def _unit(res, needle: str):
    return next(r for r in res.routings if needle in r.text)


def test_numbered_med_list_item_routes_item_content() -> None:
    res = _routings("Thuốc tại nhà:\n1. amlodipine 10 mg mỗi ngày\n")
    unit = _unit(res, "amlodipine")
    assert unit.node_kind == "list_item"
    assert unit.has_case("C1")
    assert not unit.text.startswith("1.")  # item CONTENT, marker excluded


def test_key_value_lab_row_routes_c2() -> None:
    res = _routings("Xét nghiệm:\nWBC: 14.43\n")
    unit = _unit(res, "WBC")
    assert unit.node_kind == "table_row"
    assert unit.has_case("C2")


def test_narrative_value_in_key_value_routes_narrative() -> None:
    res = _routings("Lý do vào viện: đau ngực dữ dội, khó thở nhiều giờ.\n")
    # The narrative value segment is routed for narrative/assertion, not C1/C2.
    narr = [r for r in res.routings if "đau ngực" in r.text and r.node_kind == "sentence"]
    assert narr
    assert any(r.has_case("C3") or r.has_case("C4") for r in narr)


def test_parent_line_not_double_run_by_specialist() -> None:
    # A key-value lab row whose value also forms a sentence must run the lab
    # specialist ONCE (the row), not again on the nested value sentence.
    res = _routings("Xét nghiệm:\nNEUT%: 76.4\n")
    lab = res.laboratory_proposals()
    neut_names = [p for p in lab if p.text == "NEUT%"]
    assert len(neut_names) == 1  # not duplicated by the nested sentence unit
    # exactly one logical pair
    assert len({rel.pair_group_id for rel in res.relations}) == 1


def test_semicolon_lab_row_one_pairing_group() -> None:
    res = _routings("Xét nghiệm:\nWBC: 14.43; NEUT%: 76.4\n")
    # One row unit; three names -> three logical pairs, all from the same row node.
    row_nodes = {rel.source_node_id for rel in res.relations}
    assert len(row_nodes) == 1
    assert len({rel.pair_group_id for rel in res.relations}) == 2  # WBC, NEUT%


def test_sentence_only_doc_uses_sentence_or_line_fallback() -> None:
    res = _routings("Bệnh nhân đau ngực và khó thở nhiều giờ trước khi vào viện.\n")
    kinds = {r.node_kind for r in res.routings}
    assert kinds <= {"sentence", "line"}
    assert "table_row" not in kinds and "list_item" not in kinds


def test_repeated_text_distinct_units() -> None:
    res = _routings("Xét nghiệm:\nWBC: 1\n\nWBC: 2\n")
    wbc_units = [r for r in res.routings if r.text.strip().startswith("WBC")]
    assert len({(r.start, r.end) for r in wbc_units}) == 2  # distinct positions


def test_routing_deterministic() -> None:
    a = _routings("Thuốc:\n1. amlodipine 10 mg\nWBC: 14.43\n")
    b = _routings("Thuốc:\n1. amlodipine 10 mg\nWBC: 14.43\n")
    ka = [(r.decision_id, r.node_id, r.node_kind, r.start, r.end, r.route_tags)
          for r in a.routings]
    kb = [(r.decision_id, r.node_id, r.node_kind, r.start, r.end, r.route_tags)
          for r in b.routings]
    assert ka == kb


def test_routing_preserves_provenance_fields() -> None:
    res = _routings("Thuốc tại nhà:\n1. amlodipine 10 mg\n")
    unit = _unit(res, "amlodipine")
    assert unit.node_id and unit.decision_id
    assert unit.parent_line_id is not None  # child unit records its parent line
    assert res.document_id[:4] == unit.document_id[:4]
    # absolute offsets address the original text exactly
    assert unit.text == analyze_text(
        "Thuốc tại nhà:\n1. amlodipine 10 mg\n").original_text[unit.start:unit.end]
