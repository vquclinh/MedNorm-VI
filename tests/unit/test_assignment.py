"""Shared minimum-cost assignment + Phase-1B lab pairing optimality (Area 1).

The laboratory matcher must return the GLOBAL minimum-cost one-to-one assignment,
not a greedy pairing. These tests include a regression matrix where a greedy
nearest-first pairing is strictly worse than the global optimum and verify both
the shared solver and the end-to-end lab matcher pick the global optimum.
"""

from __future__ import annotations

from pathlib import Path

from _routing_helpers import forced_routings

from mednorm_vi.assignment import assign_rectangular, min_cost_assignment
from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.mention_factory.laboratory import load_lab_lexicon, parse_graph

REPO = Path(__file__).resolve().parents[2]
LEX = load_lab_lexicon(REPO / "configs" / "laboratory" / "parser_v1.yaml")


def _greedy(cost: list[list[float]], max_cost: float) -> float:
    """Reference greedy pairing: repeatedly take the globally cheapest free cell."""
    rows, cols = len(cost), len(cost[0])
    used_r: set[int] = set()
    used_c: set[int] = set()
    total = 0.0
    cells = sorted(((cost[r][c], r, c) for r in range(rows) for c in range(cols)),
                   key=lambda t: (t[0], t[1], t[2]))
    for value, r, c in cells:
        if r in used_r or c in used_c or value > max_cost:
            continue
        used_r.add(r)
        used_c.add(c)
        total += value
    return total


def test_min_cost_beats_greedy_regression_matrix() -> None:
    # Classic assignment where greedy grabs a cheap cell that forces an expensive
    # remainder. Greedy total = 1 + 9 = 10; optimum = 4 + 5 = 9 ... tuned so the
    # gap is unambiguous.
    cost = [
        [1.0, 4.0],
        [3.0, 9.0],
    ]
    # greedy takes (0,0)=1 then must take (1,1)=9 -> 10
    assert _greedy(cost, max_cost=100.0) == 10.0
    matches, ur, uc = assign_rectangular(cost, max_cost=100.0)
    total = sum(m[2] for m in matches)
    # optimum takes (0,1)=4 + (1,0)=3 = 7
    assert total == 7.0 < 10.0
    assert ur == [] and uc == []


def test_min_cost_larger_regression() -> None:
    cost = [
        [1.0, 2.0, 9.0],
        [2.0, 9.0, 9.0],
        [9.0, 3.0, 2.0],
    ]
    matches, _, _ = assign_rectangular(cost, max_cost=100.0)
    opt = sum(m[2] for m in matches)
    assert opt <= _greedy(cost, 100.0)
    assert opt == 6.0  # (0,1)=2 + (1,0)=2 + (2,2)=2 — global optimum


def test_rectangular_unmatched_and_max_cost() -> None:
    # 3 rows, 2 cols -> at least one row unmatched; one pair exceeds max_cost.
    cost = [
        [0.1, 5.0],
        [5.0, 0.2],
        [9.0, 9.0],
    ]
    matches, ur, uc = assign_rectangular(cost, max_cost=1.0)
    assert {(r, c) for r, c, _ in matches} == {(0, 0), (1, 1)}
    assert ur == [2] and uc == []


def test_assignment_deterministic_tie_break() -> None:
    cost = [[1.0, 1.0], [1.0, 1.0]]
    a = min_cost_assignment([row[:] for row in cost])
    b = min_cost_assignment([row[:] for row in cost])
    assert a == b  # identical inputs -> identical output


# --- end-to-end: the lab matcher returns the global optimum ---

def _pairs(res) -> set[tuple[str, str]]:
    by_id = {p.proposal_id: p.text for p in res.proposals}
    return {(by_id[r.source_proposal_id], by_id[r.target_proposal_id])
            for r in res.relations if r.is_primary}


def test_lab_matcher_global_optimum_on_crossing_layout() -> None:
    # Two names and two results where a greedy nearest pairing would cross-link,
    # but the global optimum keeps each name with its own value.
    g = analyze_text("WBC: 14.43; NEUT%: 76.4")
    res = parse_graph(g, forced_routings(g, ("C2",)), LEX, "lab-p1")
    assert _pairs(res) == {("WBC", "14.43"), ("NEUT%", "76.4")}


def test_lab_no_cross_group_pairing() -> None:
    # Distinct semicolon groups must never pair a name to another group's value.
    g = analyze_text("WBC: 14.43; NEUT%: 76.4; LYMPH%: 12.8")
    res = parse_graph(g, forced_routings(g, ("C2",)), LEX, "lab-p1")
    assert _pairs(res) == {("WBC", "14.43"), ("NEUT%", "76.4"), ("LYMPH%", "12.8")}
