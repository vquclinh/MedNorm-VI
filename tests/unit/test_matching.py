"""Entity matching strategy tests."""

from __future__ import annotations

import dataclasses

from mednorm_vi.evaluation.matching import build_matcher
from mednorm_vi.evaluation.models import EvaluationConfig, EvaluationEntity

BASE = EvaluationConfig.from_mapping({
    "cost_weights": {"token_wer": 0.5, "char_overlap": 0.3, "position_distance": 0.15,
                     "boundary_length": 0.05, "exact_text_bonus": 0.5, "position_scale": 50.0},
    "max_matching_cost": 0.9,
})


def _ent(text: str, etype: str, s: int, e: int) -> EvaluationEntity:
    return EvaluationEntity(text=text, type=etype, start=s, end=e)


def _match(strategy: str, gt, pred):
    cfg = dataclasses.replace(BASE, matching_strategy=strategy)
    return build_matcher(cfg).match(tuple(gt), tuple(pred))


def test_same_text_different_position_distinct() -> None:
    gt = [_ent("táo bón", "TRIỆU_CHỨNG", 397, 404), _ent("táo bón", "TRIỆU_CHỨNG", 443, 450)]
    pred = [_ent("táo bón", "TRIỆU_CHỨNG", 443, 450), _ent("táo bón", "TRIỆU_CHỨNG", 397, 404)]
    r = _match("exact-text-occurrence", gt, pred)
    assert len(r.pairs) == 2
    # occurrence order is by position -> gt0(397)->pred1(397), gt1(443)->pred0(443)
    mapping = {d.gt_index: d.pred_index for d in r.pairs}
    assert mapping == {0: 1, 1: 0}


def test_wrong_type_never_matches() -> None:
    gt = [_ent("Ho khan", "TRIỆU_CHỨNG", 0, 7)]
    pred = [_ent("Ho khan", "CHẨN_ĐOÁN", 0, 7)]
    for strat in ("exact-position", "exact-text-occurrence", "min-cost-bipartite"):
        r = _match(strat, gt, pred)
        assert r.pairs == ()
        assert r.unmatched_gt == (0,)
        assert r.unmatched_pred == (0,)


def test_exact_position_requires_same_span() -> None:
    gt = [_ent("sốt", "TRIỆU_CHỨNG", 10, 13)]
    pred = [_ent("sốt", "TRIỆU_CHỨNG", 11, 14)]
    assert _match("exact-position", gt, pred).pairs == ()
    same = [_ent("sốt", "TRIỆU_CHỨNG", 10, 13)]
    assert len(_match("exact-position", gt, same).pairs) == 1


def test_min_cost_bipartite_matches_near_span() -> None:
    gt = [_ent("viêm phổi", "CHẨN_ĐOÁN", 10, 19)]
    pred = [_ent("viêm phổi", "CHẨN_ĐOÁN", 10, 19)]
    r = _match("min-cost-bipartite", gt, pred)
    assert len(r.pairs) == 1
    assert r.pairs[0].cost < 0  # exact text bonus makes cost negative


def test_min_cost_bipartite_rejects_over_max_cost() -> None:
    gt = [_ent("aaaa", "CHẨN_ĐOÁN", 0, 4)]
    pred = [_ent("zzzzzz", "CHẨN_ĐOÁN", 500, 506)]  # far, different text
    r = _match("min-cost-bipartite", gt, pred)
    assert r.pairs == ()  # cost exceeds max_matching_cost


def test_matching_is_deterministic() -> None:
    gt = [_ent("x", "THUỐC", 0, 1), _ent("x", "THUỐC", 5, 6)]
    pred = [_ent("x", "THUỐC", 5, 6), _ent("x", "THUỐC", 0, 1)]
    r1 = _match("min-cost-bipartite", gt, pred)
    r2 = _match("min-cost-bipartite", gt, pred)
    assert r1 == r2
