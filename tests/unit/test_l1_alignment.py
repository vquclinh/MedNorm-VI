"""L1 alignment + normalization tests."""

from __future__ import annotations

import unicodedata

from mednorm_vi.document_intelligence.alignment import CharAlignment
from mednorm_vi.document_intelligence.normalization import build_view


def _match(text: str) -> object:
    return build_view(
        "match",
        ("nfc", "casefold", "whitespace_collapse", "punctuation_alias", "decimal_view"),
        text,
    )


def test_identity_mapping() -> None:
    a = CharAlignment.identity(5)
    assert a.is_identity
    assert a.original_span_to_normalized(1, 4) == (1, 4)
    assert a.normalized_span_to_original(1, 4) == (1, 4)
    assert not a.consistency_errors()


def test_casefold_view_alignment_and_recovery() -> None:
    text = "BỆNH PHỔI"
    view = build_view("m", ("casefold",), text)
    assert view.text == text.casefold()
    # normalized 'bệnh' recovers original 'BỆNH'
    idx = view.text.index("bệnh") if "bệnh" in view.text else 0
    assert view.alignment.recover_original(text, idx, idx + 4) == text[idx : idx + 4]


def test_composed_nfd_nfc_alignment() -> None:
    original = unicodedata.normalize("NFD", "sốt")  # decomposed
    view = build_view("m", ("nfc",), original)
    assert view.text == unicodedata.normalize("NFC", original)
    # normalized composed 'sốt' maps back to the covering decomposed original
    os, oe = view.alignment.normalized_span_to_original(0, len(view.text))
    assert original[os:oe] == original
    assert not view.alignment.consistency_errors()


def test_many_to_one_whitespace_collapse() -> None:
    view = build_view("m", ("whitespace_collapse",), "a   b")
    assert view.text == "a b"
    # smallest covering of the single normalized space is one original space...
    os, oe = view.alignment.normalized_span_to_original(1, 2)
    assert "a   b"[os:oe] == " "
    # ...but a span across the words covers the whole collapsed run.
    fos, foe = view.alignment.normalized_span_to_original(0, 3)
    assert "a   b"[fos:foe] == "a   b"


def test_one_to_many_expansion_casefold() -> None:
    # German sharp s casefolds to 'ss' (1 original -> 2 normalized chars).
    view = build_view("m", ("casefold",), "ß")
    assert view.text == "ss"
    os, oe = view.alignment.normalized_span_to_original(0, 2)
    assert "ß"[os:oe] == "ß"
    # selecting only the second 's' still covers the whole original 'ß'
    os2, oe2 = view.alignment.normalized_span_to_original(1, 2)
    assert "ß"[os2:oe2] == "ß"


def test_decimal_comma_view() -> None:
    view = build_view("m", ("decimal_view",), "WBC 14,43")
    assert view.text == "WBC 14.43"


def test_original_span_to_normalized() -> None:
    view = _match("Chẩn ĐOÁN")
    # original 'Chẩn' maps to a normalized span whose text is the casefolded form
    ns, ne = view.alignment.original_span_to_normalized(0, 4)
    assert view.text[ns:ne] == "chẩn"


def test_composition_multi_stage_consistent() -> None:
    view = _match("Tiền   SỬ:  14,5")
    assert not view.alignment.consistency_errors()
    # full range covering
    os, oe = view.alignment.normalized_span_to_original(0, len(view.text))
    assert (os, oe) == (0, len("Tiền   SỬ:  14,5"))


def test_broken_alignment_rejected() -> None:
    bad = CharAlignment(original_length=4, normalized_length=3, o2n=(0, 2, 1, 0, 3))
    assert bad.consistency_errors()  # non-monotonic o2n


def test_exact_original_substring_recovery() -> None:
    text = "Không sốt, ho khan."
    view = _match(text)
    # every normalized position recovers a covering original substring
    os, oe = view.alignment.normalized_span_to_original(0, len(view.text))
    assert text[os:oe] == text


def test_accent_strip_view() -> None:
    view = build_view("m", ("accent_strip",), "Đái tháo đường")
    assert view.text == "Ðai thao đuong" or "ai thao" in view.text  # accents removed
    assert not view.alignment.consistency_errors()
