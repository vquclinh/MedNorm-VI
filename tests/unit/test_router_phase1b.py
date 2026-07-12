"""Case Router (Phase 1B) multi-label routing tests."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.case_router import CaseRouter, load_router_config, validate_routings
from mednorm_vi.document_intelligence import analyze_text

REPO = Path(__file__).resolve().parents[2]
CFG = load_router_config(REPO / "configs" / "case_router" / "base.yaml")


def _route(text: str):
    g = analyze_text(text)
    return CaseRouter(CFG).route_graph(g), g


def _cases_for(routings, needle: str) -> set[str]:
    for r in routings:
        if needle in r.text:
            return set(r.route_tags)
    return set()


def test_c1_medication_list() -> None:
    routings, _ = _route("Thuốc tại nhà:\n1. amlodipine 10 mg po daily")
    assert "C1" in _cases_for(routings, "amlodipine")


def test_c2_lab_semi_structured() -> None:
    routings, _ = _route("Xét nghiệm:\nWBC: 14,43")
    assert "C2" in _cases_for(routings, "WBC")


def test_c3_clinical_narrative_fallback() -> None:
    routings, _ = _route("Bệnh nhân mệt mỏi và ăn uống kém trong nhiều ngày qua.")
    assert "C3" in _cases_for(routings, "Bệnh nhân")


def test_c4_assertion_heavy() -> None:
    routings, _ = _route("Bệnh nhân không sốt, không ho.")
    assert "C4" in _cases_for(routings, "không sốt")


def test_c5_abbreviation_noise() -> None:
    routings, _ = _route("THA, DTD type 2 chua on dinh.")
    tags = _cases_for(routings, "THA")
    assert "C5" in tags


def test_multi_label_no_forced_exclusivity() -> None:
    # A family-history sentence with negation → C4; plus narrative fallback.
    routings, _ = _route("Tiền sử gia đình: bố không mắc đái tháo đường nhưng mẹ thì có.")
    tags = _cases_for(routings, "gia đình")
    assert "C4" in tags and len(tags) >= 1


def test_ordinary_prose_has_no_strong_structured_route() -> None:
    routings, _ = _route("Hôm nay trời đẹp và bệnh nhân cảm thấy khỏe hơn nhiều.")
    tags = _cases_for(routings, "Hôm nay")
    assert "C1" not in tags and "C2" not in tags  # C3 fallback allowed


def test_deterministic_route_scores() -> None:
    a, _ = _route("1. amlodipine 10 mg po daily")
    b, _ = _route("1. amlodipine 10 mg po daily")
    sa = [(c.case, c.score) for r in a for c in r.cases]
    sb = [(c.case, c.score) for r in b for c in r.cases]
    assert sa == sb


def test_section_prior_is_evidence() -> None:
    routings, _ = _route("Tiền sử:\ntăng huyết áp nhiều năm")
    for r in routings:
        if "tăng huyết áp" in r.text:
            assert "medical_history" == r.section_category
            assert "C4" in r.section_priors  # prior recorded as evidence


def test_activated_specialists() -> None:
    routings, _ = _route("1. amlodipine 10 mg")
    for r in routings:
        if "amlodipine" in r.text:
            assert "medication" in r.activated_specialists()


def test_repeated_segments_remain_distinct() -> None:
    routings, _ = _route("WBC: 1\n\nWBC: 2")
    rows = [r for r in routings if r.text.startswith("WBC")]
    assert len(rows) == 2
    assert rows[0].decision_id != rows[1].decision_id
    assert rows[0].node_id != rows[1].node_id


def test_routings_validate() -> None:
    routings, g = _route("Xét nghiệm:\nWBC: 14,43\n1. amlodipine 10 mg")
    assert validate_routings(routings, g.original_text).ok


def test_key_value_prose_not_routed_c2_without_lab_evidence() -> None:
    # A colon line without numbers/units must not become a lab route.
    routings, _ = _route("Chẩn đoán: viêm phổi cộng đồng")
    assert "C2" not in _cases_for(routings, "Chẩn đoán")
