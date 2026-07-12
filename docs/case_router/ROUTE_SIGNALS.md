# L2 Route Signals (v1)

Signals live in `configs/case_router/signals_v1.yaml` (versioned, extensible).
Three signal families:

- **`cue.<group>`** — a regex from a named cue group matched (case-insensitively)
  against the line's original text. Accented **and** unaccented Vietnamese
  variants are listed explicitly, plus English forms (bilingual support).
- **`structural.<name>`** — computed from the L1 graph context (is-list-item,
  key-value/table row, semicolon pairs, numeric present, narrative, upper-case
  short form, unaccented, mixed language, section prior).
- **`derived.<name>`** — computed from specialist proposals after they run.
  **C6** (meaningful linking ambiguity): `med_incomplete_identity` (known drug,
  no strength), `unknown_medication`, `conflicting_strength`. **C7** (genuine
  overlap): `cross_parse_overlap` (overlap between *different* parses/specialists,
  never the boundary alternatives of one parse) and `repeated_surface`. Routine
  progressive boundary candidates of a single parse are **not** an ambiguity
  signal and trigger neither C6 nor C7.

## Cue groups (representative, not exhaustive)

`units, routes, frequencies, release_forms, dose_forms, prn, lab_units,
reference_range, flags, qualitative, negation, history, family, contrast,
temporal, abbreviations`.

Each is a versioned list of regexes. Grow them via regression tests; adding a cue
must be checked for both recall and false positives.

## Section priors

`section_priors` maps an L1 section category → per-case prior weight (e.g.
`home_medications → C1 0.40`, `laboratory → C2 0.45`, `family_history → C4 0.45`).
These are **evidence**; a local sentence may override them in later layers.

## Per-case signals & weights

Each case lists signals with a `kind` (soft/structural/derived) and a `weight`.
Notable balances:

- **C2** requires *lab-specific* evidence: a key/value or table row alone is not
  enough; a numeric value, lab unit, reference range, flag, or lab section must
  also fire (so ordinary `key: value` prose is not routed to labs).
- **C4** negation is a strong single cue (weight 0.50) so a pure negation
  sentence activates C4; history/family combine with section priors.

## Determinism

Cue groups are matched in config order; scores are summed deterministically and
capped at 1.0. The same input + config always yields the same routing.
