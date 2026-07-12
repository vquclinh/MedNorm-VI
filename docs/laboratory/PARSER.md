# Laboratory Parser (Phase 1B)

A deterministic laboratory / test-result parser over C2-routed L1 lines. It emits
`TEST_NAME` and `TEST_RESULT` **proposals** plus internal `has_result` relations.
It **never** emits final organizer entities and **never fabricates** a numeric
result from vague narrative.

## Sources (recorded in provenance)

`key_value` rows, `table_like` (tab) rows, `semicolon_row` (multi-pair), and
`narrative` (`<TEST> <copula> <value> <unit>`). The source kind is part of each
proposal's `matched_rule`.

## Test names

A small representative seed lexicon (`test_lexicon_seed_v1.yaml`) with aliases
(e.g. `HGB`↔`Hb`↔`huyết sắc tố`, `Glucose`↔`đường huyết`). An unknown test name in
a strongly structured row is proposed with an `unknown_test_name` warning. No lab
ontology is downloaded; a local git-ignored production lexicon may be merged.

## Result values

Integers, decimal dot/comma (text preserved, normalized stored separately),
percentages, inequalities (`<5`, `≥10`), ranges, and qualitative terms
(`positive/negative`, `âm tính/dương tính`, `trace`, ...). A **reference range**
in parentheses is captured separately and excluded from the observed value.

## Boundary alternatives

A result span does **not** automatically swallow the unit. The parser emits a
`value_only` boundary and, when a unit follows, a `value_unit` boundary — both
sharing one `boundary_group_id` (a **result group**). A `has_result` pairing
targets the group and is propagated to every alternative (see `PAIRING.md`), so
whichever boundary L4 selects later still has a valid relation endpoint.

## Components (exact offsets)

`test_name`, `result_value` / `result_percent` / `result_inequality` /
`result_qualitative`, `unit`, `flag` (`H`/`L`/`↑`/`↓`/`cao`/`thấp`), and
`reference_range` — each with exact original coordinates.

## Units are evidence

A number followed by a unit is **not** automatically a lab result without
sufficient row/context evidence. Hard negatives (dates/ages/times/phones and bare
medication doses like `500 mg`) are rejected (checked over the whole name+value
region). See `KNOWN_LIMITATIONS.md`; pairing in `PAIRING.md`.
