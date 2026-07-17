"""External-resource governance validation (Phase 1C-A)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.resources import (
    ArchiveRecord,
    ChecksumRecord,
    ExtractedSnapshotRecord,
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
        files=(ChecksumRecord(path="RXNCONSO.RRF", sha256="a" * 64, bytes=10),),
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


def test_deleted_archive_unknown_sha_is_explicitly_represented() -> None:
    m = _complete_manifest(
        archive=ArchiveRecord(
            original_filename="RxNorm_full_prescribe_07062026.zip",
            package_type="zip",
            expected_published_md5="767678e3b5b1d6fe358b61c21659f3ef",
            locally_verified_archive_md5="unknown",
            archive_present=False,
            archive_deleted_before_manifest_completion=True,
        ),
        extracted_snapshot=ExtractedSnapshotRecord(
            local_path="data/external/rxnorm/prescribable-2026-07-06",
            tree_hash_sha256="a" * 64,
            file_count=16,
            total_bytes=507963140,
            content_snapshot_id="rxnorm-local-test",
            validation_status="OK",
            core_relative_paths=("raw/rrf/RXNCONSO.RRF",),
        ),
    )
    result = validate_manifest(m)
    assert result.ok
    assert any(i.code == "resource.archive_sha256_unknown" for i in result.warnings)


def test_present_archive_requires_sha256() -> None:
    m = _complete_manifest(
        archive=ArchiveRecord(
            original_filename="rxnorm.zip",
            package_type="zip",
            expected_published_md5="767678e3b5b1d6fe358b61c21659f3ef",
            locally_calculated_md5="767678e3b5b1d6fe358b61c21659f3ef",
            locally_verified_archive_md5="true",
            archive_present=True,
        )
    )
    assert any(i.code == "resource.archive_missing_sha256"
               for i in validate_manifest(m).errors)


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
