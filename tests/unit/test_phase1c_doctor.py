"""Phase 1C-A doctor readiness report — determinism + no network (1C-A)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.phase1c_foundation import DoctorPaths, build_report, render_report

REPO = Path(__file__).resolve().parents[2]
RXNORM_FIX = REPO / "tests" / "fixtures" / "kb" / "rxnorm" / "snapshot_a"


def _paths(tmp_path: Path) -> DoctorPaths:
    manifests = tmp_path / "manifests"
    rxnorm = tmp_path / "rxnorm"
    icd = tmp_path / "icd10_vi"
    manifests.mkdir(exist_ok=True)
    rxnorm.mkdir(exist_ok=True)
    icd.mkdir(exist_ok=True)
    return DoctorPaths(
        organizer_dir=REPO / "configs" / "organizer",
        position_config=REPO / "configs" / "organizer" / "position_policies_v1.yaml",
        resolver_config=REPO / "configs" / "resolution" / "resolver_v1.yaml",
        manifests_dir=manifests,
        resource_templates_dir=REPO / "configs" / "resources",
        ner_manifests_dir=REPO / "configs" / "resources" / "ner",
        rxnorm_dir=rxnorm,
        icd_dir=icd)


def test_doctor_reports_registries(tmp_path: Path) -> None:
    r = build_report(_paths(tmp_path))
    assert r.data["organizer_registry"]["confirmed_facts"] >= 10
    assert r.data["position_policies"]["default"] == "raw-codepoint-half-open"
    assert r.data["no_network"] is True
    assert r.data["resolver"]["ready"] is True


def test_doctor_flags_missing_kb(tmp_path: Path) -> None:
    r = build_report(_paths(tmp_path))
    # no real KB snapshot is committed, so both are missing locally
    assert r.data["rxnorm_snapshot"]["available"] is False
    assert r.data["icd_snapshot"]["available"] is False
    assert r.data["icd_source_artifacts"]["available"] is False
    assert any("RxNorm" in m for m in r.missing_local_resources)


def test_doctor_detects_nested_rxnorm_and_icd_source_pdfs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    nested = paths.rxnorm_dir / "prescribable-2026-07-06" / "raw" / "rrf"
    nested.mkdir(parents=True)
    for source in RXNORM_FIX.iterdir():
        if source.is_file():
            (nested / source.name).write_text(source.read_text(encoding="utf-8"),
                                             encoding="utf-8")
    icd_source = paths.icd_dir / "tt06-2026-official"
    icd_source.mkdir()
    (icd_source / "06-byt.pdf").write_bytes(b"%PDF-1.3\n%%EOF\n")
    (icd_source / "06-byt-kem.pdf").write_bytes(b"%PDF-1.3\n%%EOF\n")

    r = build_report(paths)
    assert r.data["rxnorm_prescribable_snapshot"]["available"] is True
    assert r.data["rxnorm_full_snapshot"]["available"] is False
    assert r.data["icd_source_artifacts"]["available"] is True
    assert r.data["icd_snapshot"]["available"] is False


def test_doctor_deterministic_hash_and_render(tmp_path: Path) -> None:
    a = build_report(_paths(tmp_path))
    b = build_report(_paths(tmp_path))
    assert a.determinism_hash == b.determinism_hash
    assert render_report(a) == render_report(b)
