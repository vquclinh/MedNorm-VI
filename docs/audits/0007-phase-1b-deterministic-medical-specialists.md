# Audit 0007 — Phase 1B: Deterministic Medical Structure Specialists

- **Date:** 2026-07-12
- **Author:** Claude (AI agent), for human review
- **Change type:** First medical-processing layer (L2 Case Router + L3 medication
  grammar + laboratory parser). Deterministic; no models, datasets, KB indices,
  network, or heavy ML dependencies.
- **Status:** Implemented **and post-review hardened** (see §33). All facts below
  describe the final, hardened implementation.

## 1. Objective and scope

Implement the first deterministic medical components that consume the L1
`DocumentGraph` and emit auditable medical **proposals**: an initial multi-label
Case Router (C1-C7), a medication grammar, and a laboratory/test-result parser
with `has_result` pairing. Phase 1B produces **no** final organizer predictions,
assertions, or ontology codes. Every proposal preserves exact original offsets
(`original_text[start:end] == proposal.text`).

Out of scope (deferred): neural NER (ViHealthBERT/PhoBERT/W2NER/MRC/GLiNER), LLMs,
ICD-10/RxNorm linking, final assertions/resolution, organizer JSON / `output.zip`.

## 2. Git state before implementation

Branch `main`, clean working tree, latest commit
`af69c30 feat: implement L1 document intelligence`.

## 3. Files created

- **Configs:** `configs/case_router/{base,signals_v1}.yaml`;
  `configs/medication/{grammar_v1,lexicon_seed_v1,abbreviations_v1}.yaml`;
  `configs/laboratory/{parser_v1,test_lexicon_seed_v1,units_v1}.yaml`.
- **assignment.py**: shared deterministic minimum-cost (Hungarian) assignment,
  reused by the evaluator's `min-cost-bipartite` matcher and lab pairing.
- **case_router/**: `models.py`, `signals.py`, `rules.py`, `router.py`,
  `validation.py`, `serialization.py`.
- **mention_factory/**: `models.py`, `merge.py`, `validation.py`;
  `medication/{models,lexicon,patterns,parser,scoring,validation,__init__}.py`;
  `laboratory/{models,lexicon,patterns,parser,pairing,scoring,validation,__init__}.py`.
- **deterministic_baseline/**: `models.py`, `pipeline.py`, `serialization.py`,
  `validation.py`, `cli.py`, `__init__.py`.
- **Docs:** `docs/case_router/{OVERVIEW,ROUTE_SIGNALS,KNOWN_LIMITATIONS}.md`;
  `docs/medication/{GRAMMAR,LEXICON_POLICY,BOUNDARY_PROPOSALS,KNOWN_LIMITATIONS}.md`;
  `docs/laboratory/{PARSER,PAIRING,LEXICON_POLICY,KNOWN_LIMITATIONS}.md`;
  `docs/deterministic_baseline/PHASE1B.md`.
- **Fixtures:** `tests/fixtures/phase1b/{synthetic_medical_document,
  synthetic_medication_list,synthetic_laboratory,synthetic_hard_negatives}.txt`.
- **Tests:** `tests/unit/test_{router_phase1b,medication,laboratory,proposals}.py`;
  `tests/unit/_routing_helpers.py`; the hardening suites
  `tests/unit/test_{assignment,relation_boundaries,routing_units,c6_ambiguity,
  narrative_medication,phase1b_negatives}.py`;
  `tests/integration/test_phase1b_{pipeline,stress}.py`.
- This audit.

## 4. Files modified

- `src/mednorm_vi/case_router/__init__.py`, `src/mednorm_vi/mention_factory/__init__.py`
  (Phase-0 stubs → real Phase 1B APIs).
- `src/mednorm_vi/evaluation/matching/min_cost_bipartite.py` — now imports the
  shared `assignment.min_cost_assignment` instead of a local Hungarian copy, so
  the evaluator and lab pairing cannot drift into different algorithms.
- `README.md`, `docs/architecture/LAYER_CONTRACTS.md` (L2/L3 marked implemented),
  `docs/architecture/CASE_ROUTING.md`, `docs/audits/README.md`.

## 5. Case Router design

Deterministic, multi-label, over **canonical routable units** (not whole lines).
For each line the router applies a specificity policy and routes the most specific
child structure first — list-item content, table/key-value row, sentence-like span
(incl. a key-value narrative *value* segment) — and falls back to the whole line
only when no more specific unit exists. A parent line is **never** routed through a
specialist an eligible child already represents, so a specialist runs once per
content region. A sentence nested inside a table/key-value row is *narrative-only*:
routable for C3/C4/C5 but not the structured C1/C2 the row already covers. Each
`NodeRouting` records `node_kind`, `parent_line_id`, absolute offsets, section
context, and a deterministic `decision_id`.

Reuses `schemas.routing.RouteTag/RouteDecision`; adds `RouteSignal`, `CaseScore`,
`NodeRouting`. Signals are `cue.<group>` (regex over original text, accented +
unaccented + English), `structural.<name>` (from the L1 graph), and
`derived.<name>` (from proposals). Score = min(1, Σ fired weights); a tag attaches
at `activate` (0.50). A route tag is a **processing case**, not an entity type or
final prediction.

## 6. C1-C7 route signals

C1 medication-list (list item, strength/route/frequency/dose/release/PRN cues,
med section prior); C2 lab (key-value/table/semicolon rows **plus** numeric/lab-
unit/reference/flag/lab-section — structure alone is insufficient); C3 narrative
(fallback hook); C4 assertion (negation 0.50 activates; history/family + section
priors; evidence only, no labels); C5 abbreviation/noise.

**C6 and C7 are derived from *meaningful* signals, never routine boundary
candidates.** A complete medication that merely produced several progressive
boundary spans triggers neither. Computed per parse (`augment_derived`):

- **C6 ambiguous linking:** `med_incomplete_identity` (a *known* drug named
  without an identity-critical strength), `unknown_medication` (unknown name), or
  `conflicting_strength` (same ingredient, different strengths in one unit).
- **C7 overlap/duplicates:** `cross_parse_overlap` (overlap between *different*
  parses/specialists — the boundary alternatives of one parse are excluded) and
  `repeated_surface` (same surface at different positions).

Section priors are evidence.

## 7. Multi-label routing behavior

No forced exclusivity; a node may carry several cases (verified C1+…, C4+C3,
etc.). Ordinary `key: value` prose (e.g. family history) is **not** routed to C2.
Repeated identical lines remain distinct nodes with distinct decision ids.

## 8. Medication grammar design

`MED = NAME [SALT] [RELEASE] [STRENGTH|CONCENTRATION] [DOSE_FORM] [ROUTE]
[FREQUENCY] [DURATION] [PRN]`. Over C1/C5 units it emits a `MedicationParse` +
multiple boundary-candidate `SpanProposal`s per ingredient. Original text is never
rewritten; a normalized numeric (comma→dot) is stored on `strength_value`.

**Narrative (C3) activation.** On a C3 narrative unit the grammar runs **only with
strong deterministic evidence** and restricted to **known** ingredients (unknown-
name heuristic disabled). Strong evidence: a known ingredient + (strength | route |
frequency); OR an administration predicate (`dùng`/`uống`/`kê`/`prescribed`/…) + a
strength; OR medication-section context + a known ingredient. So
`Bệnh nhân đang dùng amlodipine 10 mg mỗi ngày.` yields proposals with no neural
NER, while `Bệnh nhân uống 2 lít nước mỗi ngày.` yields none. Narrative proposals
carry a `narrative_activation` feature. The unknown-name heuristic anchors only on
a **strength/concentration** (not a bare route/dose cue), so a route verb plus a
frequency cannot fabricate a drug.

## 9. Medication structured fields

`name, salt, release, strength_value, strength_unit, concentration, dose_form,
route, frequency, duration, prn` — each a `ComponentSpan` with exact absolute
offsets. Multi-ingredient (`A and B`) yields two parses.

## 10. Medication boundary-candidate policy

Progressive right-extension: `name_only` → `name_strength` →
`name_strength_form`/`name_strength_route` → `full`. The list marker (`1.`) and
delimiters are excluded; identical spans de-duplicated; each candidate references
the same parse and is scored by the components **inside** its span. **No single
final boundary is chosen** (L4's job). See `docs/medication/BOUNDARY_PROPOSALS.md`.

## 11. Medication lexicon & abbreviation policy

Small representative seed lexicon (`lexicon_seed_v1.yaml`,
`provenance: representative_seed`) + optional local git-ignored
`production_lexicon_path` (absent). Unknown names proposed only with following
structure + `unknown_medication_name` warning. Abbreviations
(`abbreviations_v1.yaml`) are evidence-only hints. No drug DB downloaded.

## 12. Laboratory parser design

Over C2 canonical units, sources = key_value / table_like (tab) / semicolon_row /
narrative. Emits `TEST_NAME` / `TEST_RESULT` proposals (organizer labels
`TÊN_XÉT_NGHIỆM` / `KẾT_QUẢ_XÉT_NGHIỆM`) with components (value/unit/flag/
reference_range, exact offsets). Reference ranges are separated from the observed
value; no numeric is fabricated from vague narrative.

## 13. Test-result pairing design (global minimum-cost assignment)

`TEST_NAME --has_result--> TEST_RESULT` is a **globally minimum-cost one-to-one
assignment** — the shared Hungarian solver in `mednorm_vi.assignment` (the *same*
implementation the evaluator's `min-cost-bipartite` matcher uses), **not** a greedy
pairing. Within a pairing group (semicolon chunk / key-value / tab pair /
narrative) it builds a `names × result-groups` cost matrix, forbids cross-group
cells with a sentinel, solves for minimum **total** cost, and rejects pairs above
`max_cost` (leaving endpoints unmatched). Cost = distance + order − same-group −
unit. Never crosses groups/rows/sections/documents. `O(n³)` per small group.
Deterministic tie-break: stable solver + ascending `(name_id, result_group_id)`.
`has_result` is **internal only**. `test_assignment.py` includes a regression
matrix where greedy is strictly worse and asserts the matcher returns the global
optimum. See `docs/laboratory/PAIRING.md`.

## 14. Laboratory boundary-candidate policy (relations survive the choice)

A result span does not auto-swallow the unit: `value_only` and (when a unit
follows) `value_unit` boundary alternatives are emitted, sharing one
`boundary_group_id` (a **result group**). Assignment uses one representative per
group (the value-only span), so the **logical pair count is not inflated** by
alternatives. Once a name is paired to a group, one `has_result` relation is
emitted **per alternative**, all sharing a `pair_group_id`; `is_primary` marks the
value-only alternative and `target_boundary_group_id` records the group. Whichever
boundary L4 selects later, a valid relation endpoint still exists. Count **logical
pairs** by distinct `pair_group_id`; **concrete relation alternatives** by
relation.

## 15. Proposal and relation contracts

`SpanProposal` (id, document, `[start,end)`, text, proposed types, specialist,
source node, source routes, `local_score`, matched rule, normalized form,
components, `parse_ref`, `boundary_group_id`, `source_node_kind`, `parent_line_id`,
evidence ids, config/lexicon versions, warnings, features). `RelationProposal` (id,
type, source/target proposal ids, score, pairing cost, `pair_group_id`,
`is_primary`, `target_boundary_group_id`, provenance). `merge.collect` orders
deterministically, keeps identical-span duplicates (preserving provenance), and
flags overlaps/repeats — **never** deduplicates by text.

## 16. Deterministic scoring policy

`local_score`s combine rule/lexicon/section/structure/completeness evidence and
pairing cost. They are **deterministic heuristics, not calibrated probabilities**
(documented throughout). No final calibrated probability exists in Phase 1B.

## 17. Validation invariants

`mention_factory.validation` fails fast on: out-of-bounds spans, text/offset
mismatch, missing provenance/route, unsupported proposed types (only
THUỐC/TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM), final ontology codes in features,
duplicate proposal/relation ids, broken/cross-document relation endpoints.
Component spans are re-checked against `original_text`. Warnings for unknown
names / ambiguity. Router + L1 validation compose in the pipeline.

## 18. Configuration files and versions

Router `router_version=r1b`, `signals` v1; medication `grammar=med-g1`,
`lexicon=med-lex-seed-1`, `abbrev=med-abbrev-1`; laboratory `parser=lab-p1`,
`lexicon=lab-lex-seed-1`, `units=lab-units-1`. All versioned, extensible; no
lexicons hidden in Python.

## 19. Synthetic fixtures

Medication list, laboratory, mixed clinical (`synthetic_medical_document.txt`),
and hard-negative fixtures — all synthetic. No organizer test inputs committed.

## 20. Tests and exact results

```
python3 -m pytest -q   ->  383 passed in ~6s
ruff check .           ->  All checks passed!
python3 -m mypy        ->  Success: no issues found in 105 source files
git diff --check       ->  clean
```

Covers router C1-C7 + multi-label + no-exclusivity + determinism + section-prior
evidence + activated specialists + repeated distinct; medication name/strength/
route/frequency/PRN/duration/decimal(dot,comma)/attached-unit/ranges/en-dash/
concentration/release/dose/multi-ingredient/list-marker-excluded/multiple
boundaries/exact offsets/normalized-without-altering-text/repeated names/unknown
+ structure/incomplete warning/hard negatives/determinism; laboratory key-value/
decimal-comma/percent/unit/reference/flag/qualitative/inequality/semicolon-multi/
tab/narrative/unknown-with-structure/no-fabrication/boundary-alts/pairing-cost/
unrelated-not-paired/med-dose-not-lab/dates-ages-times-rejected/exact-offsets/
repeated-distinct/determinism; proposal & relation contracts + merge + broken/
cross-doc/final-code rejection + round-trip byte-deterministic JSON; integration
pipeline + CLI determinism; stress (CRLF/NFC-NFD/no-diacritics/mixed/space-tab/
long/semicolon-heavy/duplicates/unicode-dashes/abbrev/empty/no-structure).

The **post-review hardening suites** additionally cover: (1) shared min-cost
assignment incl. a greedy-vs-optimum regression matrix, rectangular/unmatched,
max-cost rejection, tie-break determinism, and end-to-end lab global optimum;
(2) `has_result` across value-only/value+unit alternatives, same logical result
parse, either boundary a valid endpoint, logical pairs not inflated, no cross-row
propagation, deterministic serialization; (3) canonical routing units (numbered
med list item → C1 on content, key-value lab row → C2, narrative value → C3/C4,
parent-line no duplicate specialist, semicolon one pairing group, sentence/line
fallback, repeated-distinct, deterministic, provenance preserved); (4) C6 fires on
incomplete/unknown/conflicting but **not** on routine boundaries; (5) narrative
medication activation with hard negatives (numbers/units, lab results, food/weight/
age, look-alike words). Plus five negative smokes (broken relation-boundary group,
duplicate specialist execution, cross-row propagation, falsely-activated C6, weak
narrative false proposal).

## 21. Positive CLI smoke-test output

```
MedNorm-VI Phase 1B — Deterministic Medical Specialists
input path                 : tests/fixtures/phase1b/synthetic_medical_document.txt
L1 graph validation        : OK
canonical routable units   : 24
  unit[list_item ]      : 4
  unit[table_row ]      : 6
  unit[sentence  ]      : 14
  unit[line      ]      : 0
  C1 : 4   C2 : 6   C3 : 6   C4 : 4   C5 : 1   C6 (derived) : 0   C7 (derived) : 0
multi-route units          : 5
specialist exec[medication]: 4     specialist exec[laboratory]: 4
duplicate-exec prevented   : 15
medication parses          : 4     medication boundary props  : 13
laboratory parses          : 4     test-name proposals        : 6
test-result proposals      : 8
logical test-result pairs  : 6     relation alternatives      : 8
duplicate-id groups        : 0     overlap pairs              : 17
proposal validation        : OK    warnings                   : 3
determinism hash           : b21659d39092cf316c48ec43ac2b36a2ef44a797a7447d5a3601d016b28268dd
```

The CLI now reports canonical routable-unit counts by node kind, C1-C7 (with the
derived C6/C7 flagged), per-specialist execution counts, the duplicate-specialist-
execution-prevention count, medication parses / boundary proposals, laboratory
parses, and the **logical test-result pair** count distinct from the **concrete
relation-alternative** count.

## 22. Determinism result

Running the CLI twice on the fixture produced **byte-identical** JSON and identical
determinism hash (`b21659d3…`), verified across all four fixtures. Asserted by
`test_phase1b_pipeline.test_cli_deterministic` and
`test_proposals.test_roundtrip_and_byte_deterministic_json`.

## 23. Negative smoke-test results

1. out-of-bounds medication proposal → `proposal.out_of_bounds` ✅
2. medication text/offset mismatch → `proposal.text_offset_mismatch` ✅
3. broken lab relation endpoint → `relation.broken_endpoint` ✅
4. cross-document relation → `relation.cross_document` ✅
5. unsupported ontology code (feature `rxnorm`) → `proposal.final_code` ✅
6. organizer-final entity type (`CHẨN_ĐOÁN` proposed) → `proposal.unsupported_type` ✅
7. hard-negative document (`synthetic_hard_negatives.txt`) → 0 medication and 0
   lab proposals, validation OK ✅

## 24. Hard-negative regression results

Medication: ages/vitals (`cao 170 cm`, `nặng 68 kg`), dates/times, bare `mg`
produce no medication. Laboratory: dates/ages/times/phones and bare med doses
(`500 mg`) produce no result; keys like `Tuổi:` are rejected over the whole
name+value region; `120 g/L` and `180 mg/dL` correctly parse as lab (lookahead
excludes concentration units). Pinned by tests.

## 25. Items intentionally deferred

Neural NER (E3-E7), LLM critic/adjudicator, L4 boundary/type resolution, assertion
classification, ICD-10/RxNorm linking, evidence graph, organizer JSON/`output.zip`.
C3 narrative and C4/C5/C6 hooks emit evidence only.

## 26. Risks and known limitations

Heuristic scores are not probabilities; unknown-name recall vs precision is gated
by routing; representative (non-exhaustive) lexicons/cues; pairing groups are
syntactic (a malformed row may under/over-group before assignment); aligned non-tab
tables not fully celled; C6/C7 are evidence hooks (no linker runs yet). All in the
`KNOWN_LIMITATIONS.md` docs. None violate the offset invariant or emit final
predictions.

## 27. No models/datasets/KBs/external APIs

Confirmed. `grep` over the new packages for
`torch|transformers|faiss|requests|urllib.request` returns nothing. Only stdlib +
`PyYAML`. No network calls; no ICD/RxNorm files.

## 28. AI-assistant files untracked

`.claude/AGENTS.md` remains git-ignored (`git check-ignore` confirms). No
`CLAUDE.md`/`AGENTS.md` is tracked.

## 29. Architecture PDFs unchanged

`git status --short -- '*.pdf'` empty; both PDFs remain at `docs/…` and were not
edited or moved.

## 30. `git status --short`

```
 M README.md
 M docs/architecture/CASE_ROUTING.md
 M docs/architecture/LAYER_CONTRACTS.md
 M docs/audits/README.md
 M src/mednorm_vi/case_router/__init__.py
 M src/mednorm_vi/evaluation/matching/min_cost_bipartite.py
 M src/mednorm_vi/mention_factory/__init__.py
?? configs/case_router/  configs/laboratory/  configs/medication/
?? docs/case_router/  docs/deterministic_baseline/  docs/laboratory/  docs/medication/
?? docs/audits/0007-phase-1b-deterministic-medical-specialists.md
?? src/mednorm_vi/assignment.py
?? src/mednorm_vi/case_router/{models,router,rules,serialization,signals,validation}.py
?? src/mednorm_vi/deterministic_baseline/
?? src/mednorm_vi/mention_factory/{laboratory,medication}/
?? src/mednorm_vi/mention_factory/{merge,models,validation}.py
?? tests/fixtures/phase1b/
?? tests/integration/test_phase1b_{pipeline,stress}.py
?? tests/unit/_routing_helpers.py
?? tests/unit/test_{router_phase1b,medication,laboratory,proposals}.py
?? tests/unit/test_{assignment,relation_boundaries,routing_units,c6_ambiguity,narrative_medication,phase1b_negatives}.py
```
(Everything uncommitted, left for human review.)

## 31. Is Phase 1B complete?

**Yes.** The Case Router, medication grammar, laboratory parser + pairing,
proposal merging, validation, deterministic serialization, and CLI are
implemented over the L1 graph; pytest/ruff/mypy/`git diff --check` are green;
positive, determinism, and all seven negative smoke tests pass; hard negatives
hold; no models/data/KB/APIs; PDFs untouched; AI-assistant files ignored.

## 32. Recommended next milestone

**Phase 1C — deterministic entity resolution and exact/fuzzy normalization
baseline:** an L4 boundary/type resolver over these proposals, exact/fuzzy ICD-10
and RxNorm retrieval over locally frozen **permitted** KB snapshots, and the first
valid organizer-compatible baseline output.

## 33. Post-review hardening summary

A focused review pass hardened five areas; all facts in §§1-32 already reflect the
final state, so this section is a pointer, not a separate contract:

1. **Laboratory matching (§13-14):** replaced greedy pairing with the shared
   Hungarian **global minimum-cost assignment** (`assignment.py`, reused by the
   evaluator); result groups keep logical-pair counts un-inflated; regression
   matrix proves optimum < greedy.
2. **has_result across boundaries (§14):** relations propagate to every value-only/
   value+unit alternative via `pair_group_id` / `target_boundary_group_id`, so any
   L4 boundary choice keeps a valid endpoint; logical pairs vs relation
   alternatives are counted separately.
3. **Canonical routing units (§5):** the router routes list-item content, table/
   key-value rows, sentences, and a line fallback — with a specificity policy and
   narrative-only nesting that prevents duplicate specialist execution.
4. **C6/C7 redefinition (§6):** C6 fires on meaningful linking ambiguity
   (incomplete identity / unknown name / conflicting strength), C7 on cross-parse
   overlap or repeated surface — routine progressive boundaries trigger neither.
5. **Narrative medication (§8):** C3 grammar activates only on strong deterministic
   evidence (known name + strength/route/frequency, admin predicate + strength, or
   med-section + known name), with hard negatives; the unknown-name heuristic
   anchors on strength only.

**Validation after hardening:** `pytest -q` → 383 passed; `ruff check .` → clean;
`python3 -m mypy` → clean (105 files); `git diff --check` → clean; CLI twice →
byte-identical on all four fixtures; the five negative smokes pass. No Phase 1C
work started; no models/datasets/KBs/outputs added; PDFs untouched.
