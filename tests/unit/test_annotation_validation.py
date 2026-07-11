"""Annotation + manifest validation tests."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.annotation.models import AnnotationEntity, Provenance, ReviewStatus
from mednorm_vi.annotation.validation import (
    validate_annotation_entity,
    validate_dataset_dir,
    validate_manifest,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "annotation" / "gold"


def _ann(**over):
    base = {
        "document_id": "1", "text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": [],
        "position": [0, 3], "review_status": "ADJUDICATED", "provenance": "GOLD",
    }
    base.update(over)
    return base


def test_valid_gold_annotation() -> None:
    assert validate_annotation_entity(_ann()).ok


def test_valid_silver_annotation() -> None:
    assert validate_annotation_entity(_ann(provenance="SILVER",
                                           review_status="SINGLE_REVIEWED")).ok


def test_invalid_offset_end_before_start() -> None:
    r = validate_annotation_entity(_ann(position=[5, 2]))
    assert not r.ok


def test_invalid_organizer_label() -> None:
    r = validate_annotation_entity(_ann(type="SYMPTOM"))  # English enum
    assert not r.ok
    assert any(i.code == "organizer.english_type" for i in r.errors)


def test_invalid_assertion() -> None:
    r = validate_annotation_entity(_ann(assertions=["isChronic"]))
    assert not r.ok


def test_unsupported_field_for_type() -> None:
    # TÊN_XÉT_NGHIỆM cannot carry assertions.
    r = validate_annotation_entity(_ann(type="TÊN_XÉT_NGHIỆM", assertions=[]))
    assert not r.ok
    assert any(i.code == "organizer.unsupported_field" for i in r.errors)


def test_bad_review_status() -> None:
    r = validate_annotation_entity(_ann(review_status="MAYBE"))
    assert not r.ok
    assert any(i.code == "annotation.bad_review_status" for i in r.errors)


def test_organizer_test_provenance_rejected() -> None:
    r = validate_annotation_entity(_ann(provenance="ORGANIZER_TEST"))
    assert not r.ok
    assert any(i.code == "annotation.organizer_test_label" for i in r.errors)


def test_only_adjudicated_counts_as_gold() -> None:
    a = AnnotationEntity("1", "sốt", "TRIỆU_CHỨNG", 0, 3,
                         review_status=ReviewStatus.DOUBLE_REVIEWED, provenance=Provenance.GOLD)
    assert not a.is_accepted_gold()
    b = AnnotationEntity("1", "sốt", "TRIỆU_CHỨNG", 0, 3,
                         review_status=ReviewStatus.ADJUDICATED, provenance=Provenance.GOLD)
    assert b.is_accepted_gold()


def test_manifest_organizer_test_labeled_rejected() -> None:
    r = validate_manifest({"dataset_id": "x", "version": "1",
                           "provenance": "ORGANIZER_TEST", "labeled": True})
    assert not r.ok
    assert any(i.code == "manifest.organizer_test_labeled" for i in r.errors)


def test_manifest_organizer_test_unlabeled_ok() -> None:
    r = validate_manifest({"dataset_id": "x", "version": "1", "provenance": "ORGANIZER_TEST",
                           "labeled": False, "includes_organizer_test_inputs": True})
    assert r.ok


def test_fixture_dataset_valid() -> None:
    assert validate_dataset_dir(FIXTURE).ok
