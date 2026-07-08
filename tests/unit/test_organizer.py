"""Organizer-facing serialization & validation tests (confirmed labels/fields)."""

from __future__ import annotations

import json

from mednorm_vi.schemas import (
    ORGANIZER_LABEL_BY_TYPE,
    EntityPrediction,
    Span,
    SpanCoordinates,
)
from mednorm_vi.validator import validate_organizer_document, validate_organizer_entity


def _pred(
    text: str,
    etype: str,
    start: int,
    end: int,
    assertions: tuple[str, ...] = (),
    candidates: tuple[str, ...] = (),
) -> EntityPrediction:
    return EntityPrediction(
        text=text,
        type=etype,
        coords=SpanCoordinates(absolute=Span(start, end)),
        assertions=assertions,
        candidates=candidates,
    )


# --- 1. Internal enum -> exact Vietnamese label ------------------------------

def test_every_internal_enum_maps_to_exact_vietnamese_label() -> None:
    assert ORGANIZER_LABEL_BY_TYPE == {
        "MEDICATION": "THUỐC",
        "DIAGNOSIS": "CHẨN_ĐOÁN",
        "SYMPTOM": "TRIỆU_CHỨNG",
        "TEST_NAME": "TÊN_XÉT_NGHIỆM",
        "TEST_RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
    }


def test_serializer_emits_vietnamese_type() -> None:
    d = _pred("paracetamol", "MEDICATION", 0, 11, candidates=("1049640",)).to_submission_dict()
    assert d["type"] == "THUỐC"


# --- 2. UTF-8 round-trip, no escaped output ----------------------------------

def test_vietnamese_labels_roundtrip_without_ascii_escapes() -> None:
    d = _pred("ho", "SYMPTOM", 0, 2).to_submission_dict()
    encoded = json.dumps(d, ensure_ascii=False)
    assert "TRIỆU_CHỨNG" in encoded
    assert "\\u" not in encoded  # not escaped
    assert json.loads(encoded)["type"] == "TRIỆU_CHỨNG"


# --- 3. English type strings rejected in organizer JSON ----------------------

def test_english_type_is_rejected() -> None:
    entity = {"text": "ho", "type": "SYMPTOM", "assertions": [], "position": [0, 2]}
    result = validate_organizer_entity(entity)
    assert not result.ok
    assert any(i.code == "organizer.english_type" for i in result.errors)


def test_unknown_type_is_rejected() -> None:
    entity = {"text": "ho", "type": "FOO", "position": [0, 2]}
    result = validate_organizer_entity(entity)
    assert not result.ok
    assert any(i.code == "organizer.invalid_type" for i in result.errors)


# --- 4. Allowed / forbidden fields per type ----------------------------------

def test_medication_allowed_fields_pass() -> None:
    d = _pred("amlodipine", "MEDICATION", 0, 10,
              assertions=("isHistorical",), candidates=("1049640",)).to_submission_dict()
    assert set(d) == {"text", "type", "candidates", "assertions", "position"}
    assert validate_organizer_entity(d).ok


def test_diagnosis_allowed_fields_pass() -> None:
    d = _pred("viêm phổi", "DIAGNOSIS", 0, 9, candidates=("J18.9",)).to_submission_dict()
    assert set(d) == {"text", "type", "candidates", "assertions", "position"}
    assert validate_organizer_entity(d).ok


def test_symptom_has_no_candidates_field() -> None:
    d = _pred("sốt", "SYMPTOM", 0, 3, assertions=("isNegated",)).to_submission_dict()
    assert set(d) == {"text", "type", "assertions", "position"}
    assert "candidates" not in d
    assert validate_organizer_entity(d).ok


def test_test_name_has_only_text_type_position() -> None:
    d = _pred("WBC", "TEST_NAME", 0, 3).to_submission_dict()
    assert set(d) == {"text", "type", "position"}
    assert validate_organizer_entity(d).ok


def test_test_result_has_only_text_type_position() -> None:
    d = _pred("14.43", "TEST_RESULT", 0, 5).to_submission_dict()
    assert set(d) == {"text", "type", "position"}
    assert validate_organizer_entity(d).ok


def test_symptom_carrying_candidates_field_is_rejected() -> None:
    entity = {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": [],
              "candidates": ["J18.9"], "position": [0, 3]}
    result = validate_organizer_entity(entity)
    assert not result.ok
    assert any(i.code == "organizer.unsupported_field" for i in result.errors)


def test_test_name_carrying_assertions_is_rejected() -> None:
    entity = {"text": "WBC", "type": "TÊN_XÉT_NGHIỆM", "assertions": [], "position": [0, 3]}
    result = validate_organizer_entity(entity)
    assert not result.ok
    assert any(i.code == "organizer.unsupported_field" for i in result.errors)


def test_unsupported_field_never_reaches_final_json() -> None:
    entity = {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": [],
              "position": [0, 2], "confidence": 0.9}
    result = validate_organizer_entity(entity)
    assert not result.ok
    assert any(
        i.code == "organizer.unsupported_field" and i.context.get("field") == "confidence"
        for i in result.errors
    )


# --- Candidate ontology syntax (light, known-only) ---------------------------

def test_medication_candidate_must_be_numeric_rxcui() -> None:
    entity = {"text": "amlodipine", "type": "THUỐC", "assertions": [],
              "candidates": ["J18.9"], "position": [0, 10]}
    result = validate_organizer_entity(entity)
    assert not result.ok
    assert any(i.code == "organizer.rxnorm_not_numeric" for i in result.errors)


def test_diagnosis_candidate_icd10_string_ok() -> None:
    entity = {"text": "viêm phổi", "type": "CHẨN_ĐOÁN", "assertions": [],
              "candidates": ["J18.9"], "position": [0, 9]}
    assert validate_organizer_entity(entity).ok


def test_missing_required_field_flagged() -> None:
    entity = {"type": "TRIỆU_CHỨNG", "assertions": [], "position": [0, 2]}  # no text
    result = validate_organizer_entity(entity)
    assert not result.ok
    assert any(i.code == "organizer.missing_field" for i in result.errors)


def test_validate_document_requires_list_root() -> None:
    result = validate_organizer_document({"not": "a list"})
    assert not result.ok
    assert any(i.code == "organizer.root_not_list" for i in result.errors)


def test_validate_document_over_list_of_entities() -> None:
    docs = [
        _pred("ho", "SYMPTOM", 0, 2).to_submission_dict(),
        _pred("viêm phổi", "DIAGNOSIS", 5, 14, candidates=("J18.9",)).to_submission_dict(),
    ]
    assert validate_organizer_document(docs).ok
