# L2 Case Router — Overview (Phase 1B)

The Case Router assigns each routable L1 node zero or more **case tags** (C1-C7).

> **A route tag is a PROCESSING CASE, not an entity type and not a final
> prediction.** It only *activates* downstream specialists. Routing is
> **multi-label**: a node may carry several cases at once (e.g. C1+C4+C6). Route
> scores are **deterministic heuristic evidence**, never calibrated probabilities.

## What it consumes and emits

- **Input:** the immutable L1 `DocumentGraph` (sections, lines, list items,
  table/key-value rows, sentences, tokens — all with exact offsets).
- **Output:** one `NodeRouting` per **canonical routable unit** (see below), with
  per-case `CaseScore`s (score + fired `RouteSignal`s + activated specialists),
  the unit's `node_kind` and `parent_line_id`, absolute offsets, section category
  + section priors (evidence), a deterministic `decision_id`, and warnings.

## Canonical routable units (not only lines)

The router routes **canonical routable units**, not whole lines. For each line it
applies a deterministic *specificity policy* and routes the most specific child
structures first:

1. **list-item content** (marker excluded) — `node_kind = list_item`;
2. **table / key-value rows** — `node_kind = table_row`;
3. **sentence-like spans**, including a key-value narrative *value* segment —
   `node_kind = sentence`;
4. the whole **line** as a fallback only when no more specific unit exists —
   `node_kind = line`.

A parent line is **never** routed through a specialist that an eligible child
already represents, so a specialist runs **once** per content region (no duplicate
proposals from multiple L1 views). A sentence nested inside a table/key-value row
is **narrative-only**: it is routed for narrative cases (C3/C4/C5) but not for the
structured cases (C1/C2) the row already covers — otherwise the lab/medication
specialist would double-run on the same value. Each `NodeRouting` records the
source `node_id`/`node_kind`, `parent_line_id`, absolute offsets, section context,
and a deterministic `decision_id`.

## The seven cases

| Case | Meaning | Activates |
| ---- | ------- | --------- |
| C1 | Medication-list | medication grammar |
| C2 | Lab semi-structured | laboratory parser |
| C3 | Clinical narrative | future-NER hook (placeholder only) |
| C4 | Assertion-heavy | assertion-evidence hook (cues only, no labels) |
| C5 | Abbreviation/noise | abbreviation hook |
| C6 | Ambiguous linking | linking hook (evidence only, no codes) |
| C7 | Overlap/duplicates | overlap/metric hooks |

C1-C5 come from structural/lexical signals; **C6/C7 are derived** from specialist
proposals after they run (`augment_derived`). C3 is a low-structure fallback when
no structured route fires on a non-empty unit.

**C6 means *meaningful* linking ambiguity — never routine boundary candidates.**
A complete medication that merely produced several progressive boundary spans does
**not** get C6. C6 fires (per parse) on: a known medication named without an
identity-critical **strength** (`incomplete_identity`), an **unknown** medication
name (`unknown_medication`), or **conflicting strength** evidence for the same
ingredient in one unit (`conflicting_strength`). **C7** fires on genuine overlap
between *different* parses/specialists (`cross_parse_overlap`) or the same surface
at different positions (`repeated_surface`) — not on the boundary alternatives of
a single parse.

## Scoring

Per case, `score = min(1.0, Σ fired-signal weights)`. A tag is attached when the
score reaches `thresholds.activate` (default 0.50). Nothing is forced to be
exclusive; a node with no strong route may legitimately carry only C3 or nothing.

## What it must not do

- It does not classify entities, assign assertions, or emit final predictions.
- It does not deduplicate by text; repeated lines stay distinct nodes.
- Section priors are **evidence** later layers may weigh — never final labels.

See `ROUTE_SIGNALS.md` (signal inventory) and `KNOWN_LIMITATIONS.md`. Config:
`configs/case_router/{base,signals_v1}.yaml`.
