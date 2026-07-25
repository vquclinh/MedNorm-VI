"""Organizer output-contract, position-boundary, and lab-boundary review tests
(Audit 0015 correction). Validates the CONFIRMED half-open [start, end) position
convention and the per-type field contract. Synthetic text only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _routing_helpers import forced_routings

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.mention_factory.laboratory import load_lab_lexicon, parse_graph
from mednorm_vi.schemas.constants import (
    ASSERTION_LABELS,
    CANDIDATE_ONTOLOGY_BY_TYPE,
    ORGANIZER_LABELS,
    POSITION_IS_END_EXCLUSIVE,
)
from mednorm_vi.schemas.prediction import EntityPrediction
from mednorm_vi.schemas.spans import Span, SpanCoordinates

REPO = Path(__file__).resolve().parents[2]
LEX = load_lab_lexicon(REPO / "configs" / "laboratory" / "parser_v1.yaml")


def _pred(text: str, start: int, end: int, typ: str = "SYMPTOM",
          assertions=(), candidates=()) -> EntityPrediction:
    return EntityPrediction(
        text=text, type=typ, coords=SpanCoordinates(Span(start, end)),
        assertions=tuple(assertions), candidates=tuple(candidates))


# --- position convention (half-open, end-exclusive; original[start:end]==text) ---

def test_position_is_end_exclusive_flag() -> None:
    assert POSITION_IS_END_EXCLUSIVE is True


@pytest.mark.parametrize("doc,start,end,expected", [
    ("A", 0, 1, "A"),                              # one-character entity
    ("abc", 0, 1, "a"),                            # entity starting at zero
    ("abc", 2, 3, "c"),                            # entity ending at last character
    ("Đau đầu nhiều", 4, 7, "đầu"),                # Vietnamese Unicode (code points)
    ("WBC\n14,43", 4, 9, "14,43"),                 # entity after a line break
    ("abc", 1, 1, ""),                             # empty span
])
def test_position_slice_matches_text(doc, start, end, expected) -> None:
    p = _pred(expected, start, end)
    assert p.position == (start, end)
    assert doc[start:end] == expected            # code-point end-exclusive invariant


def test_invalid_span_rejected() -> None:
    with pytest.raises(ValueError):
        Span(5, 3)
    with pytest.raises(ValueError):
        Span(-1, 2)


def test_confirmed_organizer_example_end_exclusive() -> None:
    # BTC-confirmed: a 25-char span "amlodipine 10 mg po daily" at [58, 83) with
    # original_text[58:83] == the span text. Synthetic equivalent (no real sample).
    prefix = "x" * 58
    span_text = "amlodipine 10 mg po daily"          # 25 chars
    doc = prefix + span_text + " tiếp tục"
    assert len(span_text) == 25
    p = _pred(span_text, 58, 83, "MEDICATION")
    assert p.position == (58, 83)
    assert doc[58:83] == span_text                    # end-exclusive invariant


def test_span_ending_at_len_text() -> None:
    doc = "Bệnh nhân ho"
    p = _pred("ho", len(doc) - 2, len(doc), "SYMPTOM")
    assert doc[p.position[0]:p.position[1]] == "ho"


def test_repeated_surface_forms_at_different_positions() -> None:
    doc = "ho nhiều, sau đó ho khan"
    first = _pred("ho", 0, 2, "SYMPTOM")
    second = _pred("ho", 17, 19, "SYMPTOM")
    assert doc[0:2] == "ho" and doc[17:19] == "ho"
    assert first.position != second.position          # distinct positions preserved


def test_inclusive_end_output_would_be_wrong() -> None:
    # An inclusive-end emitter would output end-1; assert we do NOT do that.
    doc = "paracetamol"
    p = _pred("paracetamol", 0, 11, "MEDICATION")
    assert p.position == (0, 11)                       # end-exclusive == len
    assert p.position != (0, 10)                       # inclusive-end would be wrong
    assert doc[0:11] == "paracetamol"


# --- per-type organizer field contract ---

def test_medication_and_diagnosis_carry_candidates_and_assertions() -> None:
    med = _pred("paracetamol", 0, 11, "MEDICATION",
                assertions=("isHistorical",), candidates=["161"]).to_submission_dict()
    assert med["type"] == "THUỐC"
    assert set(med) == {"text", "type", "candidates", "assertions", "position"}
    diag = _pred("viêm phổi", 0, 9, "DIAGNOSIS",
                 assertions=("isNegated",), candidates=["J18.9"]).to_submission_dict()
    assert diag["type"] == "CHẨN_ĐOÁN"
    assert set(diag) == {"text", "type", "candidates", "assertions", "position"}


def test_symptom_has_assertions_but_no_candidates() -> None:
    sym = _pred("ho", 0, 2, "SYMPTOM", assertions=("isFamily",)).to_submission_dict()
    assert sym["type"] == "TRIỆU_CHỨNG"
    assert set(sym) == {"text", "type", "assertions", "position"}
    assert "candidates" not in sym


def test_test_name_and_result_have_no_assertions_or_candidates() -> None:
    tn = _pred("WBC", 0, 3, "TEST_NAME").to_submission_dict()
    tr = _pred("14,43", 0, 5, "TEST_RESULT").to_submission_dict()
    assert tn["type"] == "TÊN_XÉT_NGHIỆM" and set(tn) == {"text", "type", "position"}
    assert tr["type"] == "KẾT_QUẢ_XÉT_NGHIỆM" and set(tr) == {"text", "type", "position"}


def test_candidate_ontology_mapping_and_assertion_labels() -> None:
    assert CANDIDATE_ONTOLOGY_BY_TYPE["DIAGNOSIS"] == "ICD10"
    assert CANDIDATE_ONTOLOGY_BY_TYPE["MEDICATION"] == "RXNORM"
    assert CANDIDATE_ONTOLOGY_BY_TYPE["SYMPTOM"] is None
    assert ASSERTION_LABELS == {"isNegated", "isHistorical", "isFamily"}


def test_no_patient_information_or_relation_types_in_contract() -> None:
    # Patient-information and relations are UNSUPPORTED_BY_OUTPUT_SCHEMA.
    assert "THÔNG_TIN_BỆNH_NHÂN" not in ORGANIZER_LABELS
    assert "patient_information" not in ORGANIZER_LABELS
    # No relation field is emitted for any type (relations are out of schema).
    for typ in ("MEDICATION", "DIAGNOSIS", "SYMPTOM", "TEST_NAME", "TEST_RESULT"):
        keys = _pred("x", 0, 1, typ).to_submission_dict()
        assert "relations" not in keys and "relation" not in keys


# --- laboratory result boundaries (value-only, value+unit, qualitative, comparison) ---

def _lab_results(text: str) -> list[str]:
    g = analyze_text(text)
    res = parse_graph(g, forced_routings(g, ("C2",)), LEX, "lab")
    return [p.text for p in res.proposals if "KẾT_QUẢ_XÉT_NGHIỆM" in p.proposed_types]


@pytest.mark.parametrize("text,needle", [
    ("WBC: 14,43", "14,43"),                 # value only, decimal comma
    ("Glucose: 7.8 mmol/L", "7.8"),          # value with unit (value part is a result)
    ("HIV: âm tính", "âm tính"),             # qualitative negative
    ("Test: dương tính", "dương tính"),      # qualitative positive
    ("Kết quả: bình thường", "bình thường"), # normal (Audit 0015 lexicon add)
    ("Men gan: tăng nhẹ", "tăng nhẹ"),       # slight increase (Audit 0015 lexicon add)
    ("CRP: <5 mg/L", "5"),                    # comparison result
])
def test_lab_result_boundaries(text, needle) -> None:
    results = " | ".join(_lab_results(text))
    assert needle in results, f"{needle!r} not found in results: {results!r}"
