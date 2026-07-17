"""Phase 1C-A doctor readiness report — determinism + no network (1C-A)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.phase1c_foundation import DoctorPaths, build_report, render_report

REPO = Path(__file__).resolve().parents[2]


def _paths() -> DoctorPaths:
    return DoctorPaths(
        organizer_dir=REPO / "configs" / "organizer",
        position_config=REPO / "configs" / "organizer" / "position_policies_v1.yaml",
        resolver_config=REPO / "configs" / "resolution" / "resolver_v1.yaml",
        manifests_dir=REPO / "data" / "manifests",
        resource_templates_dir=REPO / "configs" / "resources",
        ner_manifests_dir=REPO / "configs" / "resources" / "ner",
        rxnorm_dir=REPO / "data" / "external" / "rxnorm",
        icd_dir=REPO / "data" / "external" / "icd10_vi")


def test_doctor_reports_registries() -> None:
    r = build_report(_paths())
    assert r.data["organizer_registry"]["confirmed_facts"] >= 10
    assert r.data["position_policies"]["default"] == "raw-codepoint-half-open"
    assert r.data["no_network"] is True
    assert r.data["resolver"]["ready"] is True


def test_doctor_flags_missing_kb() -> None:
    r = build_report(_paths())
    # no real KB snapshot is committed, so both are missing locally
    assert r.data["rxnorm_snapshot"]["available"] is False
    assert r.data["icd_snapshot"]["available"] is False
    assert any("RxNorm" in m for m in r.missing_local_resources)


def test_doctor_deterministic_hash_and_render() -> None:
    a = build_report(_paths())
    b = build_report(_paths())
    assert a.determinism_hash == b.determinism_hash
    assert render_report(a) == render_report(b)
