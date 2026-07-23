"""RxNorm Full intake, governance, coexistence, and selection (Audit 0014).

Synthetic fixtures only — no real RxNorm rows or raw package files are used.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mednorm_vi.kb.rxnorm import build_snapshot, discover_rrf
from mednorm_vi.phase1c_foundation import DoctorPaths, build_report
from mednorm_vi.resources import load_manifest, validate_manifest

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "kb" / "rxnorm" / "snapshot_a"
FULL_MANIFEST = REPO / "data" / "manifests" / "rxnorm-full-2026-07-06.yaml"
PIPELINE_CFG = REPO / "configs" / "pipeline" / "full_v1.yaml"
SNAP_CFG = REPO / "configs" / "linking" / "rxnorm_snapshots_v1.yaml"

_CORE = ("RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF")


def _copy(files: tuple[str, ...], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in files:
        (dst / name).write_text((FIX / name).read_text(encoding="utf-8"), encoding="utf-8")


# --- Governance ---------------------------------------------------------------

def test_full_manifest_is_usable_and_restricted() -> None:
    m = load_manifest(FULL_MANIFEST)
    assert m.resource_id == "rxnorm-full-2026-07-06"
    assert m.license.status == "REDISTRIBUTION_RESTRICTED"
    assert m.is_usable is True                      # usable for internal use
    assert m.license.permits_redistribution is False
    vr = validate_manifest(m)
    assert not vr.errors, vr.errors


def test_full_manifest_archive_checksums_present() -> None:
    m = load_manifest(FULL_MANIFEST)
    a = m.archive
    assert a is not None and a.archive_present is True
    assert a.expected_published_md5.lower() == "33acdc0176af35808f91b3fc74ff2bb4"
    assert a.locally_calculated_md5.lower() == a.expected_published_md5.lower()
    assert a.locally_verified_archive_md5 == "true"
    assert len(a.archive_sha256) == 64


def test_full_manifest_stores_no_credentials() -> None:
    # Look for real secret *values* (not prose stating none are stored).
    import re
    text = FULL_MANIFEST.read_text(encoding="utf-8")
    patterns = [
        r"[\w.+-]+@[\w-]+\.\w{2,}",              # email address
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",   # private key block
        r"\b(sk|ghp|hf)_[A-Za-z0-9]{16,}\b",     # api/token prefixes
        r"AKIA[0-9A-Z]{16}",                      # AWS key
        r"(?i)\b(password|api[_-]?key|token|secret)\s*[:=]\s*\S+",  # key: value secret
    ]
    for pat in patterns:
        assert re.search(pat, text) is None, f"credential-like value matched {pat!r}"


# --- Snapshot semantics -------------------------------------------------------

def test_full_snapshot_has_semantic_types_prescribable_does_not(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    presc_dir = tmp_path / "presc"
    _copy(_CORE + ("RXNSTY.RRF",), full_dir)          # Full: incl RXNSTY
    _copy(_CORE, presc_dir)                            # Prescribable: no RXNSTY
    full = build_snapshot(full_dir)
    presc = build_snapshot(presc_dir)
    assert len(full.semantic_types) > 0
    assert len(presc.semantic_types) == 0
    assert discover_rrf(full_dir).sty is not None
    assert discover_rrf(presc_dir).sty is None


# --- Doctor coexistence -------------------------------------------------------

def test_doctor_detects_both_snapshots_and_indexes(tmp_path: Path) -> None:
    rxnorm = tmp_path / "rxnorm"
    indices = tmp_path / "indices"
    _copy(_CORE, rxnorm / "prescribable-2026-07-06" / "raw" / "rrf")
    _copy(_CORE + ("RXNSTY.RRF",), rxnorm / "full-2026-07-06" / "raw" / "rrf")
    for snap in ("prescribable-2026-07-06", "full-2026-07-06"):
        d = indices / snap
        d.mkdir(parents=True)
        (d / "index.json").write_text("{}", encoding="utf-8")

    paths = DoctorPaths(
        organizer_dir=REPO / "configs" / "organizer",
        position_config=REPO / "configs" / "organizer" / "position_policies_v1.yaml",
        resolver_config=REPO / "configs" / "resolution" / "resolver_v1.yaml",
        manifests_dir=tmp_path / "m",
        resource_templates_dir=REPO / "configs" / "resources",
        ner_manifests_dir=REPO / "configs" / "resources" / "ner",
        rxnorm_dir=rxnorm,
        rxnorm_indices_dir=indices,
        icd_dir=tmp_path / "icd")
    (tmp_path / "m").mkdir()
    r = build_report(paths)
    presc = r.data["rxnorm_prescribable_snapshot"]
    full = r.data["rxnorm_full_snapshot"]
    assert presc["available"] is True and presc["rxnsty_available"] is False
    assert presc["index_available"] is True
    assert full["available"] is True and full["rxnsty_available"] is True
    assert full["semantic_types_available"] is True and full["index_available"] is True
    assert "available" in full["status"]
    assert r.data["rxnorm_snapshot"]["active"] == "prescribable"


def test_doctor_full_absent_when_only_prescribable(tmp_path: Path) -> None:
    rxnorm = tmp_path / "rxnorm"
    _copy(_CORE, rxnorm / "prescribable-2026-07-06" / "raw" / "rrf")
    paths = DoctorPaths(
        organizer_dir=REPO / "configs" / "organizer",
        position_config=REPO / "configs" / "organizer" / "position_policies_v1.yaml",
        resolver_config=REPO / "configs" / "resolution" / "resolver_v1.yaml",
        manifests_dir=tmp_path / "m",
        resource_templates_dir=REPO / "configs" / "resources",
        ner_manifests_dir=REPO / "configs" / "resources" / "ner",
        rxnorm_dir=rxnorm,
        rxnorm_indices_dir=tmp_path / "indices",
        icd_dir=tmp_path / "icd")
    (tmp_path / "m").mkdir()
    r = build_report(paths)
    assert r.data["rxnorm_prescribable_snapshot"]["available"] is True
    assert r.data["rxnorm_full_snapshot"]["available"] is False


# --- Selection configuration --------------------------------------------------

def test_snapshot_selection_config_is_conservative_and_complete() -> None:
    cfg = yaml.safe_load(SNAP_CFG.read_text(encoding="utf-8"))
    assert cfg["active"] == "prescribable"
    assert cfg["current_prescribable_preference"] is True
    assert set(cfg["snapshots"]) == {"prescribable", "full"}
    assert cfg["snapshots"]["full"]["semantic_types"] is True
    assert cfg["snapshots"]["prescribable"]["semantic_types"] is False


def test_pipeline_config_registers_both_snapshots_default_prescribable() -> None:
    doc = yaml.safe_load(PIPELINE_CFG.read_text(encoding="utf-8"))
    assert doc["rxnorm_index"].endswith("prescribable-2026-07-06/index.json")
    sel = doc["rxnorm_selection"]
    assert sel["active"] == "prescribable"
    for mode in ("prescribable_only", "full_only", "full_prefer_prescribable",
                 "ablation_full_vs_prescribable"):
        assert mode in sel["modes"]
    assert set(doc["rxnorm_snapshots"]) == {"prescribable", "full"}
