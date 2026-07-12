# Medication Boundary Proposals (Phase 1B)

The grammar generates **multiple boundary candidates** per ingredient because the
organizer annotation convention is not yet fixed. **Phase 1B never selects a
single final boundary** — L4 will.

## Progressive right-extension

For `1. amlodipine 10 mg po daily`, candidates (all excluding the list marker):

- `amlodipine` — `name_only`
- `amlodipine 10 mg` — `name_strength`
- `amlodipine 10 mg po` — `name_strength_route`
- `amlodipine 10 mg po daily` — `full`

(`name_strength_form` is added when a dose form follows.) Identical `(start,end)`
candidates are de-duplicated, keeping the earliest kind.

## Each candidate

- **excludes** the list marker (`1.`) and leading delimiters;
- **preserves** exact original offsets (`original_text[start:end] == text`);
- **references the same `MedicationParse`** (`parse_ref`) and carries the full
  component set for downstream use;
- records the boundary `kind` (`matched_rule = med_grammar:<kind>`) and a
  completeness feature;
- gets a deterministic score reflecting the components **inside** that span.

## Why keep them all

L4 (Boundary & Type Resolver) will learn the organizer's convention and choose
among these alternatives. Emitting a single boundary now would bake in a guess.
Overlapping/adjacent alternatives are preserved and surfaced as C7 evidence and
merge diagnostics.
