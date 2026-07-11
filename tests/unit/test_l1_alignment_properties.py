"""Exhaustive small-string property tests for L1 alignment.

For representative transforms and small inputs, checks EVERY boundary and EVERY
valid span:
  * endpoints are anchored and boundary maps are monotonic;
  * original→normalized maps stay in bounds and match forward slicing;
  * normalized→original yields the SMALLEST valid covering original span;
  * exact original substring recovery reads only from original text;
  * composed stages remain consistent.

Deterministic and dependency-free (no property-testing framework).
"""

from __future__ import annotations

import unicodedata

from mednorm_vi.document_intelligence.alignment import CharAlignment
from mednorm_vi.document_intelligence.normalization import build_view

# (name, stage list, small inputs) — representative transformations.
_CASES: list[tuple[str, tuple[str, ...], list[str]]] = [
    ("identity", (), ["", "a", "abc", "áé", "  ", "a\nb"]),
    ("casefold", ("casefold",), ["", "A", "ẞ", "STRASSE", "BỆNH", "Aß b"]),
    ("whitespace", ("whitespace_collapse",), ["", " ", "a  b", "  a   b  ", "a\t\tb"]),
    ("nfc", ("nfc",), [unicodedata.normalize("NFD", s) for s in ["sốt", "Chẩn", "áé"]]),
    ("punct", ("punctuation_alias",), ["“x”", "a–b—c", "•y"]),
    ("decimal", ("decimal_view",), ["14,43", "a,b", "1,2,3"]),
    ("accent", ("accent_strip",), ["Chẩn đoán", "sốt", "PHỔI"]),
    ("match", ("nfc", "casefold", "whitespace_collapse", "punctuation_alias", "decimal_view"),
     ["Tiền  SỬ: 14,5", "“BỆNH”  phổi", "  Chẩn đoán  "]),
]


def _check_alignment(a: CharAlignment, src: str, dst: str) -> None:
    # endpoints + lengths
    assert a.original_length == len(src)
    assert a.normalized_length == len(dst)
    assert not a.consistency_errors()
    assert a.o2n[0] == 0
    assert a.o2n[-1] == len(dst)
    # monotonic
    assert all(x <= y for x, y in zip(a.o2n, a.o2n[1:], strict=False))

    # every original span → in-bounds normalized span matching forward slicing
    for s in range(len(src) + 1):
        for e in range(s, len(src) + 1):
            ns, ne = a.original_span_to_normalized(s, e)
            assert 0 <= ns <= ne <= len(dst)
            assert (ns, ne) == (a.o2n[s], a.o2n[e])

    # every normalized span → covering original span; recovery reads only original
    for qs in range(len(dst) + 1):
        for qe in range(qs, len(dst) + 1):
            os, oe = a.normalized_span_to_original(qs, qe)
            assert 0 <= os <= oe <= len(src)
            # recovery reads ONLY from original text (never normalized)
            assert a.recover_original(src, qs, qe) == src[os:oe]
            if qs < qe:  # smallest covering (well-defined for non-empty spans)
                assert a.o2n[os] <= qs and a.o2n[oe] >= qe  # covering
                assert os == len(src) or a.o2n[os + 1] > qs  # os maximal at/before qs
                assert oe == 0 or a.o2n[oe - 1] < qe  # oe minimal at/after qe
            else:  # empty normalized span → zero-width original point
                assert os == oe


def test_exhaustive_alignment_properties() -> None:
    for _name, stages, inputs in _CASES:
        for src in inputs:
            view = build_view("v", stages, src)
            _check_alignment(view.alignment, src, view.text)


def test_identity_covering_is_exact() -> None:
    a = CharAlignment.identity(6)
    for qs in range(7):
        for qe in range(qs, 7):
            assert a.normalized_span_to_original(qs, qe) == (qs, qe)


def test_expansion_covering_minimal() -> None:
    a = build_view("v", ("casefold",), "ß").alignment  # ß -> ss
    assert a.normalized_span_to_original(0, 1) == (0, 1)  # first 's' covers 'ß'
    assert a.normalized_span_to_original(1, 2) == (0, 1)  # second 's' covers 'ß'
    assert a.normalized_span_to_original(0, 2) == (0, 1)


def test_composition_consistency() -> None:
    src = "Tiền  SỬ"
    full = build_view("v", ("nfc", "casefold", "accent_strip"), src)
    # composed alignment full-range covers the original exactly
    assert full.alignment.normalized_span_to_original(0, len(full.text)) == (0, len(src))
    assert not full.alignment.consistency_errors()
