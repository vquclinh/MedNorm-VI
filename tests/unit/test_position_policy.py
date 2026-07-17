"""Position-policy mapping, reversibility, and forensics (Phase 1C-A)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.position import (
    Observation,
    PositionEncodingError,
    analyze,
    encode_span,
    load_position_registry,
    validate_encoding,
)

REPO = Path(__file__).resolve().parents[2]
REG = load_position_registry(REPO / "configs" / "organizer" / "position_policies_v1.yaml")

# Multi-byte Vietnamese text with CRLF line endings.
TEXT = "Bệnh nhân dùng amlodipine.\r\nHuyết áp cao.\r\n"


def _span(word: str) -> tuple[int, int]:
    i = TEXT.index(word)
    return i, i + len(word)


def test_raw_codepoint_is_identity_and_reversible() -> None:
    s, e = _span("amlodipine")
    r = encode_span(REG.policy("raw-codepoint-half-open"), TEXT, s, e)
    assert (r.start, r.end) == (s, e)
    assert r.text == "amlodipine"
    assert REG.round_trips("raw-codepoint-half-open", TEXT, s, e)


def test_utf8_byte_differs_and_round_trips() -> None:
    s, e = _span("amlodipine")
    r = encode_span(REG.policy("utf8-byte-half-open"), TEXT, s, e)
    # multi-byte prefix shifts the byte offset beyond the code-point offset
    assert r.start > s
    assert REG.round_trips("utf8-byte-half-open", TEXT, s, e)
    assert validate_encoding(REG, TEXT, r).ok


def test_utf8_byte_distinct_from_codepoint() -> None:
    s, e = _span("Huyết")
    cp = encode_span(REG.policy("raw-codepoint-half-open"), TEXT, s, e)
    by = encode_span(REG.policy("utf8-byte-half-open"), TEXT, s, e)
    assert cp.encoded_position != by.encoded_position


def test_canonical_lf_collapses_crlf() -> None:
    # A span after the first CRLF shifts by 1 per preceding CRLF under LF canon.
    s, e = _span("Huyết áp")
    raw = encode_span(REG.policy("raw-codepoint-half-open"), TEXT, s, e)
    lf = encode_span(REG.policy("canonical-lf-codepoint"), TEXT, s, e)
    assert lf.start == raw.start - 1  # one CRLF collapsed before this span
    assert REG.round_trips("canonical-lf-codepoint", TEXT, s, e)


def test_encoding_never_edits_text() -> None:
    s, e = _span("amlodipine")
    for pid in REG.policy_ids:
        if pid == "normalized-view-codepoint":
            continue
        r = encode_span(REG.policy(pid), TEXT, s, e)
        assert r.text == TEXT[s:e]


def test_normalized_view_requires_alignment() -> None:
    s, e = _span("amlodipine")
    try:
        encode_span(REG.policy("normalized-view-codepoint"), TEXT, s, e)
        raise AssertionError("expected PositionEncodingError")
    except PositionEncodingError:
        pass


def test_forensics_detects_byte_offsets() -> None:
    obs = []
    for w in ("amlodipine", "Huyết"):
        s, e = _span(w)
        b = encode_span(REG.policy("utf8-byte-half-open"), TEXT, s, e)
        obs.append(Observation(w, b.start, b.end))
    report = analyze(REG, TEXT, obs)
    assert report.best_policy_id == "utf8-byte-half-open"
    assert report.byte_vs_codepoint == "byte"


def test_forensics_detects_crlf_reconstruction() -> None:
    # Observations produced under LF-canonical: the line-ending evidence should
    # point at LF canonicalization (raw CRLF over-counts by 1 per preceding CRLF).
    obs = []
    for w in ("Bệnh", "Huyết"):
        s, e = _span(w)
        lf = encode_span(REG.policy("canonical-lf-codepoint"), TEXT, s, e)
        obs.append(Observation(w, lf.start, lf.end))
    report = analyze(REG, TEXT, obs)
    assert report.line_ending_evidence == "lf_canonical"
