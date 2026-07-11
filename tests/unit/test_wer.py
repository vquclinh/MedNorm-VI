"""WER + tokenization tests (provisional evaluator)."""

from __future__ import annotations

from mednorm_vi.evaluation.tokenization import (
    tokenize_character_diagnostic,
    tokenize_whitespace,
    tokenize_whitespace_punctuation,
)
from mednorm_vi.evaluation.wer import compute_wer


def _wer(ref: str, hyp: str, tok: str = "whitespace", clip: bool = False):
    return compute_wer(ref, hyp, tokenization=tok, clipping_enabled=clip)


def test_exact_match_perfect_score() -> None:
    w = _wer("amlodipine 10 mg", "amlodipine 10 mg")
    assert w.raw_wer == 0.0
    assert w.raw_text_score == 1.0
    assert (w.substitutions, w.deletions, w.insertions) == (0, 0, 0)


def test_substitution() -> None:
    w = _wer("viêm phổi cấp", "viêm phổi mạn")
    assert w.substitutions == 1
    assert w.deletions == 0 and w.insertions == 0
    assert abs(w.raw_wer - 1 / 3) < 1e-9


def test_insertion() -> None:
    w = _wer("sốt", "sốt cao")
    assert w.insertions == 1
    assert w.raw_wer == 1.0  # 1 insertion / 1 ref token


def test_deletion() -> None:
    w = _wer("sốt cao", "sốt")
    assert w.deletions == 1
    assert abs(w.raw_wer - 0.5) < 1e-9


def test_empty_ref_empty_hyp() -> None:
    w = _wer("", "")
    assert w.ref_tokens == 0
    assert w.raw_wer == 0.0
    assert w.raw_text_score == 1.0


def test_empty_ref_nonempty_hyp_preserves_wer_gt_one() -> None:
    w = _wer("", "a b c")
    assert w.insertions == 3
    assert w.raw_wer == 3.0  # denominator floored at 1
    assert w.raw_text_score == -2.0  # NOT clamped


def test_wer_greater_than_one_preserved() -> None:
    w = _wer("a", "b c d e f")
    assert w.raw_wer > 1.0
    assert w.raw_text_score < 0.0


def test_clipping_is_explicit_and_preserves_raw() -> None:
    w = _wer("", "a b c", clip=True)
    assert w.raw_text_score == -2.0  # raw always preserved
    assert w.clipped_text_score == 0.0
    assert w.text_score == 0.0  # uses clipped when enabled


def test_no_clipping_uses_raw() -> None:
    w = _wer("", "a b c", clip=False)
    assert w.text_score == w.raw_text_score == -2.0


def test_vietnamese_unicode() -> None:
    w = _wer("đái tháo đường", "đái tháo đường")
    assert w.raw_text_score == 1.0


def test_punctuation_tokenization_separates_marks() -> None:
    assert tokenize_whitespace_punctuation("WBC:14.43") == ("WBC", ":", "14", ".", "43")
    # whitespace tokenizer keeps it as one token -> different WER behavior.
    w_ws = _wer("WBC:14.43", "WBC:14.5", tok="whitespace")
    w_wp = _wer("WBC:14.43", "WBC:14.5", tok="whitespace-punctuation")
    assert w_ws.raw_wer == 1.0  # whole token substituted
    assert w_wp.raw_wer < 1.0   # only the '43'->'5' token differs


def test_whitespace_collapses_repeated_spaces() -> None:
    assert tokenize_whitespace("a   b\t c") == ("a", "b", "c")


def test_character_diagnostic_tokenizer() -> None:
    assert tokenize_character_diagnostic("ab c") == ("a", "b", " ", "c")
