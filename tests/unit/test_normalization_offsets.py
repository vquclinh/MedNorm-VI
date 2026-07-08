"""Normalization must never alter original output offsets (spec section 4).

The normalized view exists for matching only. Output ``position`` is always
derived from ABSOLUTE coordinates, and the alignment maps any normalized index
back to the original.
"""

from __future__ import annotations

import unicodedata

from mednorm_vi.schemas import (
    EntityPrediction,
    OffsetAlignment,
    Span,
    SpanCoordinates,
)
from mednorm_vi.validator import validate_offset_invariant


def test_output_position_comes_from_absolute_not_normalized() -> None:
    original = "paracetamol 500mg"
    start = original.index("paracetamol")
    end = start + len("paracetamol")
    coords = SpanCoordinates(
        absolute=Span(start, end),
        # Pretend the normalized view uppercased/relocated the token; these
        # coordinates differ but must never leak into output.
        normalized=Span(999, 1010),
    )
    pred = EntityPrediction(text="paracetamol", type="MEDICATION", coords=coords)
    assert pred.position == (start, end)
    assert validate_offset_invariant(original, pred).ok


def test_nfc_normalization_preserves_output_offsets() -> None:
    # Decomposed (NFD) original; matching may normalize to NFC on a COPY, but the
    # emitted span still slices the ORIGINAL text and must match exactly.
    original = unicodedata.normalize("NFD", "sốt cao")
    token = unicodedata.normalize("NFD", "sốt")
    start = original.index(token)
    coords = SpanCoordinates(absolute=Span(start, start + len(token)))
    pred = EntityPrediction(text=token, type="SYMPTOM", coords=coords)
    # The invariant uses the ORIGINAL text; normalization on a copy is irrelevant.
    assert validate_offset_invariant(original, pred).ok
    assert original[pred.start : pred.end] == pred.text


def test_offset_alignment_identity_roundtrip() -> None:
    align = OffsetAlignment(original_length=10, normalized_length=10)
    assert align.is_identity
    for i in range(10):
        assert align.to_original(align.to_normalized(i)) == i


def test_offset_alignment_explicit_roundtrip() -> None:
    # Normalized view dropped a combining char: original len 4 -> normalized len 3.
    # original idx: 0 1 2 3  ; normalized idx: 0 1 . 2  (idx 2 collapses into 1)
    o2n = (0, 1, 1, 2)
    n2o = (0, 1, 3)
    align = OffsetAlignment(
        original_length=4,
        normalized_length=3,
        original_to_normalized=o2n,
        normalized_to_original=n2o,
    )
    assert not align.is_identity
    # A match found at normalized index 2 maps back to original index 3.
    assert align.to_original(2) == 3
    assert align.to_normalized(3) == 2
