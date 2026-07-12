# Test-Result Pairing (Phase 1B)

Pairing `TEST_NAME --has_result--> TEST_RESULT` is a **globally minimum-cost
one-to-one assignment** between test-name proposals and result *groups* within a
single pairing group. The relation is **internal only** and must never appear in
organizer output.

## Assignment algorithm

Pairing uses the shared Hungarian (Kuhn–Munkres) solver in
`mednorm_vi.assignment` — the *same* implementation the provisional evaluator's
`min-cost-bipartite` matcher uses, so the two never drift into subtly different
algorithms. `assign_rectangular` builds a rectangular `names × groups` cost
matrix, pads it to a square with a large sentinel, solves for the assignment that
minimises **total** cost, then rejects any matched pair whose cost exceeds
`max_cost`.

This is a true global optimum, not a greedy nearest-first pairing. Where a greedy
pass would grab a locally cheap pair and force an expensive remainder, the
Hungarian solver minimises the sum. `tests/unit/test_assignment.py` includes a
regression matrix where greedy is strictly worse and asserts the Phase 1B matcher
returns the global optimum.

Complexity is `O(n³)` in `n = max(#names, #groups)` per pairing group. Pairing
groups are small (a row / semicolon chunk), so this is negligible in practice.

## Pairing groups (never cross)

Names and results are extracted per **pairing group** — a semicolon chunk, a
key/value pair, a tab-cell row, or a narrative fragment — and a name is **never**
matched to a result in another group (cross-group cells carry a forbidding
sentinel cost `_FORBID`). Pairing also never crosses L1 rows/sections/documents,
because each canonical routable unit is parsed independently.

```
WBC: 14.43; NEUT%: 76.4; LYMPH%: 12.8
  WBC    -> 14.43     (group 0)
  NEUT%  -> 76.4      (group 1)
  LYMPH% -> 12.8      (group 2)
```

## Result groups and boundary alternatives

One logical result value produces several **boundary alternatives** — a
value-only span (`7.8`) and, when a unit follows, a value+unit span
(`7.8 mmol/L`). These share one `boundary_group_id` and form a **result group**.
Assignment uses a single representative (the value-only span) per group, so the
**logical pair count is not inflated** by boundary alternatives.

Once a name is paired to a result group, one `has_result` relation is emitted per
alternative in that group, all sharing a `pair_group_id`; `is_primary` marks the
value-only alternative and `target_boundary_group_id` records the group. Whichever
boundary L4 selects later, a valid `has_result` endpoint still exists. Count
**logical pairs** by distinct `pair_group_id`; count **concrete relation
alternatives** by relation.

## Cost matrix

```
cost = w_distance·norm_distance + w_order·order_penalty
       − w_same_row·same_group − w_unit·unit_present
```

`norm_distance` is the gap between the name end and the result start over the unit
length; `order_penalty` applies when the result precedes the name. Same-group and
unit-present bonuses reduce cost. Weights live in
`configs/laboratory/parser_v1.yaml::pairing.weights`; costs are rounded to 6
decimals for stable comparison.

## Unmatched names/results and rejection

The matrix may be rectangular (more names than results or vice versa), so some
endpoints are legitimately **unmatched** and simply carry no relation. Any matched
pair whose optimal cost exceeds `max_cost` is rejected, leaving both endpoints
unmatched — so unrelated names/results never pair.

## Tie-breaking / determinism

The Hungarian solver is deterministic for a fixed matrix (stable scan order);
rejection and emission then sort by ascending `(name_id, result_group_id)` and
relations are emitted in `(name_id, result_group_id, member index)` order. Costs
are rounded before comparison. Identical input always yields identical relations
(`test_deterministic_relation_serialization`).

## Limitations

- Costs are heuristic evidence, not calibrated probabilities.
- Grouping is syntactic (delimiter-based); a malformed row may under- or
  over-group, changing which names/results compete in one assignment.
- Only value-only spans drive assignment; an exotic result whose only span is
  value+unit would need its representative reconsidered (not observed in the
  seed lexicon).
