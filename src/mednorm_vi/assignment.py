"""Shared deterministic minimum-cost assignment (pure Python, no dependencies).

A single Hungarian (Kuhn–Munkres, O(n³)) implementation reused by the provisional
evaluator's ``min-cost-bipartite`` matcher and the Phase 1B laboratory
test-result pairing, so the two never drift into subtly different algorithms.

``min_cost_assignment`` solves a square matrix (row → col). ``assign_rectangular``
adapts it to rectangular cost matrices with a ``max_cost`` rejection threshold and
returns matched ``(row, col, cost)`` triples plus the unmatched row/col indices.
Deterministic for a fixed input (ties are resolved by the algorithm's stable scan
order, then by ascending ``(row, col)`` at the rejection step).
"""

from __future__ import annotations

_PAD = 1e9


def min_cost_assignment(cost: list[list[float]]) -> list[int]:
    """Minimum-cost perfect assignment on a square matrix; returns row → col.

    Returns a list ``assign`` of length ``n`` where ``assign[row] == col``. An
    empty matrix yields ``[]``.
    """
    n = len(cost)
    if n == 0:
        return []
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assign[p[j] - 1] = j - 1
    return assign


def assign_rectangular(
    cost: list[list[float]], *, max_cost: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Globally minimum-cost one-to-one assignment over a rectangular matrix.

    ``cost[r][c]`` is the cost of matching row ``r`` to column ``c``; forbidden
    pairs must already carry a cost greater than ``max_cost`` (e.g. a large
    sentinel). Pairs whose optimal cost exceeds ``max_cost`` are rejected, leaving
    both endpoints unmatched.

    Returns ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is a
    list of ``(row, col, cost)`` sorted by ``(row, col)``.
    """
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    n = max(rows, cols)
    square = [[_PAD] * n for _ in range(n)]
    for r in range(rows):
        for c in range(cols):
            square[r][c] = cost[r][c]
    assign = min_cost_assignment(square)
    matches: list[tuple[int, int, float]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r in range(rows):
        c = assign[r]
        if 0 <= c < cols and cost[r][c] <= max_cost:
            matches.append((r, c, cost[r][c]))
            matched_rows.add(r)
            matched_cols.add(c)
    matches.sort(key=lambda m: (m[0], m[1]))
    unmatched_rows = [r for r in range(rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


__all__ = ["min_cost_assignment", "assign_rectangular"]
