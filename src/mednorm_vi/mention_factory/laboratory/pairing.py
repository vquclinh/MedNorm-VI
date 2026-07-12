"""Deterministic TEST_NAME -> TEST_RESULT pairing (global min-cost assignment).

Pairing is a **globally minimum-cost one-to-one assignment** (shared Hungarian in
``mednorm_vi.assignment``) between TEST_NAME proposals and result GROUPS within a
row/line. A result group is one logical value with its boundary alternatives
(value-only / value+unit); pairing uses one representative per group so the
logical pair count is not inflated by boundary alternatives.

Cross-group pairs are forbidden (large sentinel cost); pairs above ``max_cost``
are rejected, leaving names/results unmatched. Deterministic tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...assignment import assign_rectangular
from ..models import SpanProposal

_FORBID = 1e6


@dataclass(frozen=True, slots=True)
class ResultGroup:
    """One logical result value: a representative proposal + its alternatives."""

    group_id: str
    representative: SpanProposal  # the value-only proposal
    members: tuple[SpanProposal, ...]  # all boundary alternatives (incl. representative)
    group_idx: int
    has_unit: bool


@dataclass(frozen=True, slots=True)
class LogicalPair:
    """One logical TEST_NAME -> result-group pairing."""

    name_id: str
    result_group_id: str
    cost: float
    same_group: bool
    unit_match: bool
    members: tuple[SpanProposal, ...] = field(default_factory=tuple)


def pair_cost(
    name: SpanProposal, result: SpanProposal, *, same_group: bool, unit_match: bool,
    weights: dict[str, float], line_len: int,
) -> float:
    dist = abs(result.start - name.end) / max(line_len, 1)
    order = 0.0 if result.start >= name.end else weights.get("order", 0.2)
    cost = weights.get("distance", 0.4) * min(1.0, dist) + order
    if same_group:
        cost -= weights.get("same_row_bonus", 0.5)
    if unit_match:
        cost -= weights.get("unit_bonus", 0.2)
    return round(cost, 6)


def pair_names_to_groups(
    names: list[SpanProposal], groups: list[ResultGroup], group_of: dict[str, int],
    *, weights: dict[str, float], max_cost: float, line_len: int,
) -> list[LogicalPair]:
    """Globally minimum-cost assignment of names to result groups (within a line)."""
    if not names or not groups:
        return []
    cost: list[list[float]] = []
    same_flags: list[list[bool]] = []
    unit_flags: list[list[bool]] = []
    for n in names:
        row: list[float] = []
        s_row: list[bool] = []
        u_row: list[bool] = []
        for grp in groups:
            same = group_of.get(n.proposal_id) == grp.group_idx
            unit_match = grp.has_unit
            c = pair_cost(n, grp.representative, same_group=same, unit_match=unit_match,
                          weights=weights, line_len=line_len)
            row.append(c if same else _FORBID)  # never pair across groups
            s_row.append(same)
            u_row.append(unit_match)
        cost.append(row)
        same_flags.append(s_row)
        unit_flags.append(u_row)
    matches, _unmatched_names, _unmatched_groups = assign_rectangular(cost, max_cost=max_cost)
    out: list[LogicalPair] = []
    for r, c, mc in matches:
        out.append(LogicalPair(
            name_id=names[r].proposal_id, result_group_id=groups[c].group_id, cost=mc,
            same_group=same_flags[r][c], unit_match=unit_flags[r][c],
            members=groups[c].members))
    out.sort(key=lambda p: (p.name_id, p.result_group_id))
    return out


__all__ = ["ResultGroup", "LogicalPair", "pair_cost", "pair_names_to_groups"]
