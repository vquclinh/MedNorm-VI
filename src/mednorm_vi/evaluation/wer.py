"""Token-sequence Levenshtein Word Error Rate (WER).

Confirmed: the text component uses WER on the ``text`` field and conceptually
``1 - WER``. PROVISIONAL: tokenization and whether ``1 - WER`` is clipped are not
published; both are configurable and audited.

WER = (substitutions + deletions + insertions) / max(ref_tokens, 1)

Edge cases (documented and tested):
  * ref empty, hyp empty  -> WER 0.0, 1-WER 1.0
  * ref empty, hyp N>0    -> WER N (denominator floored at 1), 1-WER negative
  * ref M>0, hyp empty    -> WER 1.0 (all deletions), 1-WER 0.0
  * WER may exceed 1.0 and the raw value is always preserved.

Raw values are never silently clamped. Clipping (if enabled) is a separate,
explicit field; the raw score is retained alongside it.
"""

from __future__ import annotations

from .models import WERBreakdown
from .tokenization import get_tokenizer


def _levenshtein_counts(ref: tuple[str, ...], hyp: tuple[str, ...]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) via edit-distance backtrace.

    Deterministic tie-break preference: substitution, then deletion, then
    insertion (stable and reproducible).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]; op[i][j] records the
    # chosen operation for backtrace: 'M' match, 'S' sub, 'D' del, 'I' ins.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
                continue
            sub = dp[i - 1][j - 1]
            dele = dp[i - 1][j]
            ins = dp[i][j - 1]
            best = min(sub, dele, ins)
            dp[i][j] = best + 1
            # Deterministic preference order: S, D, I.
            if best == sub:
                op[i][j] = "S"
            elif best == dele:
                op[i][j] = "D"
            else:
                op[i][j] = "I"
    subs = dels = inss = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "M":
            i, j = i - 1, j - 1
        elif o == "S":
            subs += 1
            i, j = i - 1, j - 1
        elif o == "D":
            dels += 1
            i -= 1
        else:  # "I"
            inss += 1
            j -= 1
    return subs, dels, inss


def compute_wer(
    reference_text: str,
    hypothesis_text: str,
    *,
    tokenization: str,
    clipping_enabled: bool,
) -> WERBreakdown:
    """Compute a full WER breakdown for a (reference, hypothesis) text pair."""
    tokenizer = get_tokenizer(tokenization)
    ref = tokenizer(reference_text)
    hyp = tokenizer(hypothesis_text)
    subs, dels, inss = _levenshtein_counts(ref, hyp)
    ref_tokens = len(ref)
    denom = ref_tokens if ref_tokens > 0 else 1
    raw_wer = (subs + dels + inss) / denom
    raw_text_score = 1.0 - raw_wer
    clipped: float | None = None
    if clipping_enabled:
        clipped = max(0.0, min(1.0, raw_text_score))
    return WERBreakdown(
        tokenization=tokenization,
        substitutions=subs,
        deletions=dels,
        insertions=inss,
        ref_tokens=ref_tokens,
        raw_wer=raw_wer,
        raw_text_score=raw_text_score,
        clipping_enabled=clipping_enabled,
        clipped_text_score=clipped,
    )


__all__ = ["compute_wer"]
