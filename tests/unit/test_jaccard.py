"""Jaccard tests: empty-set rules, dedup diagnostics, candidate weighting."""

from __future__ import annotations

from mednorm_vi.evaluation.jaccard import jaccard_breakdown


def test_both_empty_is_one() -> None:
    b = jaccard_breakdown("assertions", (), ())
    assert b.jaccard == 1.0


def test_gt_empty_pred_nonempty_is_zero() -> None:
    b = jaccard_breakdown("assertions", (), ("isNegated",))
    assert b.jaccard == 0.0


def test_normal_jaccard() -> None:
    b = jaccard_breakdown("candidates", ("A", "B"), ("B", "C"))
    assert b.intersection == ("B",)
    assert set(b.union) == {"A", "B", "C"}
    assert b.jaccard == 1 / 3
    assert b.missing == ("A",)
    assert b.extra == ("C",)


def test_duplicate_candidate_surfaced() -> None:
    b = jaccard_breakdown("candidates", ("A", "A", "B"), ("B",))
    assert b.gt_duplicates == ("A",)
    assert b.gt_deduped == ("A", "B")
    # dedup happens before set conversion -> jaccard uses deduped sets
    assert b.jaccard == 0.5


def test_duplicate_assertion_surfaced() -> None:
    b = jaccard_breakdown("assertions", ("isNegated", "isNegated"), ("isNegated",))
    assert b.pred_duplicates == ()
    assert b.gt_duplicates == ("isNegated",)
    assert b.jaccard == 1.0


def test_candidate_weight_is_len_gt_plus_one() -> None:
    b = jaccard_breakdown("candidates", ("A", "B", "C"), ("A",))
    assert b.weight == 4.0  # len(GT)+1
    empty = jaccard_breakdown("candidates", (), ("X",))
    assert empty.weight == 1.0  # 0 + 1


def test_assertions_have_unit_weight() -> None:
    b = jaccard_breakdown("assertions", ("isNegated",), ())
    assert b.weight == 1.0
