"""The exact four-way ablation, driven by injected E3 spans (Audit 0034).

No checkpoint and no Torch: the neural expert is supplied as data, which is what
makes the whole ablation reproducible offline.
"""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.deterministic_baseline.models import Phase1BConfig
from mednorm_vi.evaluation.ablation import (
    ARM_DETERMINISTIC_ONLY,
    ARM_LATTICE_AFTER_L4,
    ARM_LATTICE_BEFORE_L4,
    ARM_NEURAL_ONLY,
    ARMS,
    run_ablation,
)
from mednorm_vi.mention_factory.neural.decoding import NeuralSpan
from mednorm_vi.resolution.config_v1 import DEFAULT_CONFIG_PATH, load_resolver_v1_config

REPO = Path(__file__).resolve().parents[2]
PHASE1B = Phase1BConfig.load(
    REPO / "configs" / "case_router" / "base.yaml",
    REPO / "configs" / "medication" / "grammar_v1.yaml",
    REPO / "configs" / "laboratory" / "parser_v1.yaml")
RESOLVER = load_resolver_v1_config(REPO / DEFAULT_CONFIG_PATH)

TEXT_A = "Bệnh nhân bị suy tim và ho khan."
TEXT_B = "Glucose: 5.6 mmol/L"

# Offsets are derived from the text itself, so a hand-typed index can never make
# a fixture violate the spec §4 invariant it is meant to protect.
DIAGNOSIS_START = TEXT_A.index("suy tim")
DIAGNOSIS_END = DIAGNOSIS_START + len("suy tim")
SYMPTOM_START = TEXT_A.index("ho khan")
SYMPTOM_END = SYMPTOM_START + len("ho khan")

EXAMPLES = [
    {
        "example_id": "ex-a", "text": TEXT_A, "source_dataset": "synthetic",
        "entities": [
            {"start": DIAGNOSIS_START, "end": DIAGNOSIS_END,
             "target_type": "DIAGNOSIS", "text": "suy tim"},
            {"start": SYMPTOM_START, "end": SYMPTOM_END,
             "target_type": "SYMPTOM", "text": "ho khan"},
        ],
    },
    {
        "example_id": "ex-b", "text": TEXT_B, "source_dataset": "synthetic",
        "entities": [
            {"start": 0, "end": 7, "target_type": "TEST_NAME", "text": "Glucose"},
            {"start": 9, "end": 12, "target_type": "TEST_RESULT", "text": "5.6"},
        ],
    },
]

NEURAL = {
    "ex-a": (
        NeuralSpan(DIAGNOSIS_START, DIAGNOSIS_END, "DIAGNOSIS", "suy tim", 0.95, 2),
        NeuralSpan(SYMPTOM_START, SYMPTOM_END, "SYMPTOM", "ho khan", 0.90, 2),
    ),
    "ex-b": (),
}


def _report():
    return run_ablation(
        EXAMPLES, PHASE1B, RESOLVER, neural_spans_for=lambda eid: NEURAL.get(eid, ()))


def test_all_four_arms_are_executed() -> None:
    report = _report()
    assert set(report["arms"]) == set(ARMS)
    for arm in ARMS:
        assert report["arms"][arm]["examples"] == 2


def test_arm1_scores_only_the_neural_expert() -> None:
    arm = _report()["arms"][ARM_NEURAL_ONLY]
    assert arm["final_count"] == 2
    assert arm["error_categories"]["exact_match"] == 2
    assert arm["by_type"]["TEST_NAME"]["true_positive"] == 0


def test_arm2_scores_only_the_deterministic_experts() -> None:
    arm = _report()["arms"][ARM_DETERMINISTIC_ONLY]
    assert arm["by_type"]["TEST_NAME"]["true_positive"] == 1
    assert arm["by_type"]["DIAGNOSIS"]["true_positive"] == 0


def test_arm3_is_the_union_of_both_expert_families() -> None:
    report = _report()
    before = report["arms"][ARM_LATTICE_BEFORE_L4]
    assert before["error_categories"]["exact_match"] >= (
        report["arms"][ARM_NEURAL_ONLY]["error_categories"]["exact_match"])
    assert before["proposal_count"] >= report["arms"][ARM_NEURAL_ONLY]["proposal_count"]


def test_arm4_resolves_boundary_alternatives_and_reports_its_own_counters() -> None:
    report = _report()
    after = report["arms"][ARM_LATTICE_AFTER_L4]
    before = report["arms"][ARM_LATTICE_BEFORE_L4]
    assert after["final_count"] <= before["final_count"]
    assert after["resolver_suppressed"] >= 1  # the value+unit alternative
    assert after["error_categories"]["exact_match"] >= 3


def test_every_arm_reports_the_required_fields() -> None:
    for arm in _report()["arms"].values():
        for key in ("micro", "by_type", "error_categories", "proposal_count",
                    "final_count", "boundary_error_total", "delta_vs_arm1"):
            assert key in arm
        for key in ("precision", "recall", "f1"):
            assert key in arm["micro"]
        for key in ("missed", "spurious", "wrong_type", "left_boundary",
                    "right_boundary", "both_boundary", "exact_match"):
            assert key in arm["error_categories"]


def test_deltas_are_measured_against_arm1() -> None:
    report = _report()
    baseline = report["arms"][ARM_NEURAL_ONLY]
    assert baseline["delta_vs_arm1"]["f1"] == 0.0
    for arm_name, arm in report["arms"].items():
        expected = round(arm["micro"]["f1"] - baseline["micro"]["f1"], 6)
        assert arm["delta_vs_arm1"]["f1"] == expected, arm_name


def test_report_states_it_is_not_the_organizer_score() -> None:
    report = _report()
    assert report["reproduces_complete_organizer_score"] is False
    assert report["resolver_config_sha256"] == RESOLVER.config_sha256


def test_ablation_is_deterministic() -> None:
    assert _report()["arms"] == _report()["arms"]


def test_symptom_attribution_is_produced_and_privacy_safe() -> None:
    report = _report()
    assert report["symptom_attribution"]["total"] >= 0
    for record in report["symptom_attributions"]:
        assert len(record["privacy_safe_example_id"]) == 16
        assert "ho khan" not in str(record)
