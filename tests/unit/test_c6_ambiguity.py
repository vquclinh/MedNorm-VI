"""C6 ambiguous-linking is meaningful ambiguity, not routine boundaries (Area 4).

C6 must NOT fire merely because a medication produced several progressive
boundary candidates. It fires on real linking ambiguity: a known medication
named without an identity-critical strength, an unknown medication name, or
conflicting strength evidence for the same ingredient within a unit.
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


def _tags(text: str) -> set[str]:
    res = run_phase1b(analyze_text(text), CFG)
    return {t for r in res.routings for t in r.route_tags}


def _c6_signals(text: str) -> set[str]:
    res = run_phase1b(analyze_text(text), CFG)
    out: set[str] = set()
    for r in res.routings:
        for c in r.cases:
            if c.case == "C6":
                out |= {s.name for s in c.fired_signals}
    return out


def test_routine_boundaries_do_not_trigger_c6() -> None:
    # A complete medication with strength + several boundary candidates: no C6.
    assert "C6" not in _tags("Thuốc tại nhà:\n1. amlodipine 10 mg mỗi ngày\n")


def test_incomplete_medication_triggers_c6() -> None:
    # Known drug named without an identity-critical strength.
    assert "C6" in _tags("Thuốc tại nhà:\n1. amlodipine mỗi ngày\n")
    assert "incomplete_identity" in _c6_signals("Thuốc tại nhà:\n1. amlodipine mỗi ngày\n")


def test_unknown_medication_triggers_c6() -> None:
    assert "C6" in _tags("Thuốc tại nhà:\n1. Xzytorbin 10 mg mỗi ngày\n")
    assert "unknown_medication" in _c6_signals("Thuốc tại nhà:\n1. Xzytorbin 10 mg mỗi ngày\n")


def test_conflicting_strength_triggers_c6() -> None:
    text = "Bệnh nhân dùng amlodipine 5 mg và amlodipine 10 mg.\n"
    assert "C6" in _tags(text)
    assert "conflicting_strength" in _c6_signals(text)


def test_c6_absent_when_no_medication() -> None:
    assert "C6" not in _tags("Xét nghiệm:\nWBC: 14.43\n")
