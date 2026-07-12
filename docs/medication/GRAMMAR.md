# Medication Grammar (Phase 1B)

A deterministic, high-recall medication grammar over medication-activated
canonical routable units. It is a **proposal source**, not the final boundary
authority (L4 decides). It never emits final entities, assertions, or RxNorm
codes.

```
MED = NAME [SALT] [RELEASE] [STRENGTH|CONCENTRATION] [DOSE_FORM]
          [ROUTE] [FREQUENCY] [DURATION] [PRN]
```

## Structured fields (each with exact original offsets)

`name`, `salt`, `release`, `strength_value`, `strength_unit`, `concentration`,
`dose_form`, `route`, `frequency`, `duration`, `prn`. A parsed field does **not**
imply it belongs to the final organizer span.

## Strength / concentration

Supports `10 mg`, `0.5 mg`, `0,5 mg`, `325-650 mg`, `325–650 mg`, `500mg`, `1 g`,
`5 mg/ml`, `0.4 mg/mL`, `20 mg/5 mL`, percentages, ratios. The **original text is
never rewritten**; a normalized numeric (comma→dot) is stored separately on the
`strength_value` component.

## Inventories (versioned config)

Release forms, dose forms, routes, frequencies, PRN forms, and duration templates
live in `configs/medication/grammar_v1.yaml`; route/frequency/release abbreviation
hints in `abbreviations_v1.yaml` (evidence only, never a committed final meaning).

## Names & unknowns

A small representative **seed** lexicon (`lexicon_seed_v1.yaml`) plus an optional
local, git-ignored production lexicon path. A name **not** in the lexicon is
proposed only when followed by real medication structure (strength/dose/route)
and carries an `unknown_medication_name` warning. See `LEXICON_POLICY.md`.

## Activation (structured vs narrative)

The grammar runs on a unit routed **C1/C5** (structured medication context: list
items, medication sections) with the full grammar including the unknown-name
heuristic. On a **C3 narrative** unit it runs **only with strong deterministic
evidence** and restricted to **known** ingredients (the unknown-name heuristic is
disabled), so ordinary prose never fabricates a drug. Strong narrative evidence is
any of:

- a **known ingredient** + (strength | route | frequency); or
- an **administration predicate** (`dùng`, `uống`, `kê`, `chỉ định`, `prescribed`,
  `taking`, …) + a **strength**; or
- **medication-section context** + a known ingredient.

Example — `Bệnh nhân đang dùng amlodipine 10 mg mỗi ngày.` produces proposals with
no neural NER; `Bệnh nhân uống 2 lít nước mỗi ngày.` produces none. Narrative
proposals carry a `narrative_activation` feature. The unknown-name heuristic
anchors only on a **strength/concentration** (not a bare route/dose cue), so a
route verb like `uống` plus a frequency cannot invent an unknown medication.

## Multi-ingredient

`paracetamol 500 mg and ibuprofen 200 mg` yields two parses (one per ingredient),
each with its own component scope and boundary candidates.

## Scoring

Deterministic heuristic evidence (name lexicon match, section prior, list
structure, and the components **inside** each boundary candidate). Wider
boundaries score differently from `name_only`. These are **not** probabilities.

## Hard negatives

Ages, vitals (`cao 170 cm`, `nặng 68 kg`), dates/times, and bare `mg` without a
name never become medications. See `KNOWN_LIMITATIONS.md`. Boundary policy is in
`BOUNDARY_PROPOSALS.md`.
