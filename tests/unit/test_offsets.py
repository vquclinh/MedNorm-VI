"""Offset-invariant tests (spec section 4)."""

from __future__ import annotations

from mednorm_vi.schemas import EntityPrediction, Span, SpanCoordinates
from mednorm_vi.validator import validate_offset_invariant

TEXT = "Bệnh nhân bị ho, không sốt; tiền sử ho từ nhỏ."


def _entity(text: str, start: int, end: int, etype: str = "SYMPTOM") -> dict:
    return {
        "text": text,
        "type": etype,
        "assertions": [],
        "candidates": [],
        "position": [start, end],
    }


def test_original_substring_invariant_holds() -> None:
    start = TEXT.index("ho")
    e = _entity("ho", start, start + 2)
    assert validate_offset_invariant(TEXT, e).ok


def test_substring_mismatch_is_rejected() -> None:
    # Same text, wrong position -> substring at that position differs.
    e = _entity("ho", 0, 2)  # TEXT[0:2] == "Bệ", not "ho"
    result = validate_offset_invariant(TEXT, e)
    assert not result.ok
    assert any(i.code == "offset.substring_mismatch" for i in result.errors)


def test_end_exclusive_indexing() -> None:
    # [start, end) must span exactly len(text). Off-by-one end is caught.
    start = TEXT.index("ho")
    too_long = _entity("ho", start, start + 3)  # includes an extra char
    result = validate_offset_invariant(TEXT, too_long)
    assert not result.ok
    codes = {i.code for i in result.errors}
    # Either a length mismatch or a substring mismatch flags the error.
    assert codes & {"offset.length_mismatch", "offset.substring_mismatch"}


def test_end_exclusive_full_string() -> None:
    e = _entity(TEXT, 0, len(TEXT), etype="DIAGNOSIS")
    # position end == len(text) is valid (end-exclusive), substring matches.
    assert validate_offset_invariant(TEXT, e).ok


def test_empty_text_rejected() -> None:
    e = _entity("", 5, 5)
    result = validate_offset_invariant(TEXT, e)
    assert not result.ok
    assert any(i.code == "offset.empty_text" for i in result.errors)


def test_out_of_bounds_rejected() -> None:
    e = _entity("nhỏ.", len(TEXT) - 2, len(TEXT) + 5)
    result = validate_offset_invariant(TEXT, e)
    assert not result.ok
    assert any(i.code == "offset.out_of_bounds" for i in result.errors)


def test_entity_prediction_object_is_accepted() -> None:
    start = TEXT.index("sốt")
    coords = SpanCoordinates(absolute=Span(start, start + len("sốt")))
    pred = EntityPrediction(text="sốt", type="SYMPTOM", coords=coords)
    assert validate_offset_invariant(TEXT, pred).ok


def test_code_point_indexing_with_combining_accents() -> None:
    # Vietnamese diacritics: indices are code points, so slicing must match text.
    text = "paracetamol sốt"
    start = text.index("sốt")
    e = _entity("sốt", start, start + len("sốt"))
    assert validate_offset_invariant(text, e).ok


def test_published_offset_golden_example() -> None:
    # Organizer-published example confirming Python-style end-exclusive character
    # offsets: text "amlodipine 10 mg po daily", position [58, 83], length 25.
    entity_text = "amlodipine 10 mg po daily"
    assert len(entity_text) == 25
    start, end = 58, 83
    assert end - start == 25  # end-exclusive: end - start == len(text)

    # Build an original text (with Vietnamese diacritics) where the token sits
    # exactly at [58, 83]. Prefix is padded to exactly `start` code points.
    prefix = "Bệnh nhân được kê toa: "
    prefix = prefix + "x" * (start - len(prefix))
    assert len(prefix) == start
    original = prefix + entity_text + " một lần mỗi ngày."
    assert original[start:end] == entity_text  # code-point indexing

    e = _entity(entity_text, start, end, etype="MEDICATION")
    assert validate_offset_invariant(original, e).ok
