"""L1 exact-input loading tests."""

from __future__ import annotations

import pytest

from mednorm_vi.document_intelligence.io import DocumentLoadError, from_text, load_text


def test_valid_utf8_roundtrip() -> None:
    doc = load_text("Chẩn đoán: viêm phổi".encode())
    assert doc.original_text == "Chẩn đoán: viêm phổi"
    assert doc.source.encoding == "utf-8"


def test_invalid_utf8_fails() -> None:
    with pytest.raises(DocumentLoadError):
        load_text(b"\xff\xfe\x00abc")


def test_lf_preserved() -> None:
    doc = from_text("a\nb\nc")
    assert doc.original_text == "a\nb\nc"
    assert doc.source.newline_style == "LF"


def test_crlf_preserved() -> None:
    doc = from_text("a\r\nb\r\nc")
    assert doc.original_text == "a\r\nb\r\nc"
    assert doc.source.newline_style == "CRLF"


def test_mixed_newlines_detected() -> None:
    assert from_text("a\r\nb\nc").source.newline_style == "MIXED"


def test_no_trailing_newline_preserved() -> None:
    doc = from_text("no newline at end")
    assert doc.original_text.endswith("end")
    assert doc.source.newline_style == "NONE"


def test_leading_trailing_whitespace_preserved() -> None:
    doc = from_text("   leading and trailing   ")
    assert doc.original_text == "   leading and trailing   "


def test_tabs_and_repeated_blanks_preserved() -> None:
    raw = "a\tb\n\n\nc"
    doc = from_text(raw)
    assert doc.original_text == raw
    assert "\t" in doc.original_text
    assert "\n\n\n" in doc.original_text


def test_bom_preserve_policy_keeps_bom() -> None:
    doc = load_text("﻿hello".encode(), bom_policy="preserve")
    assert doc.original_text.startswith("﻿")
    assert doc.source.has_bom is True


def test_bom_strip_policy_removes_only_at_load() -> None:
    doc = load_text("﻿hello".encode(), bom_policy="strip")
    assert doc.original_text == "hello"
    assert doc.source.has_bom is False


def test_bom_error_policy_raises() -> None:
    with pytest.raises(DocumentLoadError):
        load_text("﻿hello".encode(), bom_policy="error")


def test_source_metadata() -> None:
    doc = from_text("abc\ndef")
    assert doc.source.char_length == 7
    assert doc.source.byte_length == 7
    assert len(doc.source.sha256) == 64
