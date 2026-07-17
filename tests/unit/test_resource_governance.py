"""External-resource governance validation (Phase 1C-A)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.resources import (
    ChecksumRecord,
    LicenseRecord,
    RedistributionPolicy,
    ResourceManifest,
    SnapshotIdentity,
    SourceRecord,
    load_manifest,
    validate_manifest,
)
from mednorm_vi.resources.ner import (
    LabelMapping,
    label_mapping_set,
    load_ner_manifest,
    validate_ner_manifest,
)

REPO = Path(__file__).resolve().parents[2]


def _complete_manifest(**overrides) -> ResourceManifest:
    base = dict(
        resource_id="rxnorm-test", title="t",
        source=SourceRecord(organization="NLM", url="http://example"),
        snapshot=SnapshotIdentity(snapshot_id="s1", version="2026-01"),
        license=LicenseRecord(name="UMLS", status="PERMITTED_INTERNAL_USE"),
        redistribution=RedistributionPolicy(permission="restricted"),
        files=(ChecksumRecord(path="RXNCONSO.RRF", sha256="abc", bytes=10),),
        intended_use=("linking",), permitted_use=("linking",))
    base.update(overrides)
    return ResourceManifest(**base)  # type: ignore[arg-type]


def test_complete_manifest_valid_and_usable() -> None:
    m = _complete_manifest()
    assert validate_manifest(m).ok
    assert m.is_usable


def test_missing_version_fails() -> None:
    m = _complete_manifest(snapshot=SnapshotIdentity())
    r = validate_manifest(m)
    assert not r.ok
    assert any(i.code == "resource.missing_version" for i in r.errors)


def test_missing_checksum_fails() -> None:
    m = _complete_manifest(files=())
    r = validate_manifest(m)
    assert any(i.code == "resource.missing_checksums" for i in r.errors)


def test_missing_intended_use_fails() -> None:
    m = _complete_manifest(intended_use=())
    assert any(i.code == "resource.missing_intended_use" for i in validate_manifest(m).errors)


def test_unknown_license_status_not_usable() -> None:
    m = _complete_manifest(license=LicenseRecord(name="x", status="LICENSE_UNKNOWN"))
    assert not m.is_usable
    warns = {i.code for i in validate_manifest(m).warnings}
    assert "resource.not_yet_usable" in warns


def test_rejected_license_errors() -> None:
    m = _complete_manifest(license=LicenseRecord(name="x", status="REJECTED"))
    assert any(i.code == "resource.rejected" for i in validate_manifest(m).errors)


def test_intended_exceeds_permitted_fails() -> None:
    m = _complete_manifest(intended_use=("linking", "ner_training"), permitted_use=("linking",))
    assert any(i.code == "resource.intended_exceeds_permitted"
               for i in validate_manifest(m).errors)


def test_template_manifest_loads() -> None:
    m = load_manifest(REPO / "configs" / "resources" / "rxnorm_snapshot.manifest.template.yaml")
    assert m.license.status == "REVIEW_REQUIRED"
    assert not m.is_usable  # a template is never usable as-is


def test_ner_manifest_and_label_mapping() -> None:
    m = load_ner_manifest(REPO / "configs" / "resources" / "ner" / "phoner_covid19.manifest.yaml")
    assert validate_ner_manifest(m).ok  # REVIEW mapping target is allowed
    assert m.license.status == "REVIEW_REQUIRED"


def test_bad_label_mapping_target_fails() -> None:
    from mednorm_vi.resources.ner import NerDatasetManifest

    m = NerDatasetManifest(
        dataset_id="d", title="t", source=SourceRecord(organization="x"),
        license=LicenseRecord(), redistribution=RedistributionPolicy(),
        label_mappings=(LabelMapping("SRC", "inter", "NOT_A_TYPE"),))
    assert any(i.code == "ner.bad_mapping_target" for i in validate_ner_manifest(m).errors)


def test_label_mapping_set_deterministic_hash() -> None:
    m = [LabelMapping("DRUG", "medication", "MEDICATION")]
    a = label_mapping_set("d", 1, m)
    b = label_mapping_set("d", 1, m)
    assert a.config_hash == b.config_hash and a.config_hash
