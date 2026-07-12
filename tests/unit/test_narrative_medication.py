"""Strong-narrative medication activation with hard negatives (Area 5).

The medication grammar runs on a C3 narrative segment only with STRONG
deterministic evidence: a known ingredient + (strength|route|frequency), an
administration predicate + a strength, or medication-section context + a known
ingredient. Ordinary numbers/units, lab results, food/weight/age, and narrative
words merely resembling drug names must NOT produce medication proposals.
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


def _med_texts(text: str) -> list[str]:
    res = run_phase1b(analyze_text(text), CFG)
    return [p.text for p in res.medication_proposals()]


def _narrative_activated(text: str) -> list[str]:
    res = run_phase1b(analyze_text(text), CFG)
    return [p.text for p in res.medication_proposals()
            if p.features.get("narrative_activation") == 1.0]


# --- positive activations ---

def test_admin_predicate_plus_known_name_and_freq() -> None:
    # "đang dùng amlodipine mỗi ngày" — known name + frequency, no strength.
    texts = _med_texts("Bệnh nhân đang dùng amlodipine mỗi ngày.\n")
    assert any(t == "amlodipine" for t in texts)
    assert "amlodipine" in _narrative_activated("Bệnh nhân đang dùng amlodipine mỗi ngày.\n")


def test_known_name_plus_strength_in_narrative() -> None:
    texts = _med_texts("Bệnh nhân đang dùng amlodipine 10 mg mỗi ngày.\n")
    assert any(t.startswith("amlodipine") for t in texts)


# --- hard negatives: strong evidence absent ---

def test_ordinary_number_unit_not_medication() -> None:
    assert _med_texts("Bệnh nhân uống 2 lít nước mỗi ngày.\n") == []


def test_lab_result_not_medication() -> None:
    assert _med_texts("Xét nghiệm:\nGlucose: 7.8 mmol/L\n") == []


def test_food_weight_age_rejected() -> None:
    for line in ("Bệnh nhân ăn 2 quả táo mỗi ngày.\n",
                 "Cân nặng 68 kg, cao 170 cm.\n",
                 "Bệnh nhân nam 54 tuổi.\n"):
        assert _med_texts(line) == [], line


def test_narrative_word_resembling_name_without_evidence() -> None:
    # A bare mention with no strength/route/frequency and no admin predicate must
    # not fabricate a medication in narrative context.
    assert _med_texts("Bệnh nhân khỏe mạnh, không có triệu chứng bất thường.\n") == []
