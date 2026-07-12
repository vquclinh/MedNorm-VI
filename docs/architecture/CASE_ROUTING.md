# MedNorm-VI — Case Routing (L2)

> Describes the seven routed case families from `docs/MedNorm-VI_Architecture.pdf`
> (section 5) and how the multi-label router activates specialist modules.
> Declarative placeholders live in `configs/routes.yaml`. Cue lists there are
> **seed examples, not the closed medical lexicon** (spec section 4.3).
>
> **Phase 1B status:** an initial deterministic multi-label router is implemented
> in `src/mednorm_vi/case_router/`, configured by `configs/case_router/` (not the
> `configs/routes.yaml` placeholder). It routes L1 LINE nodes for C1-C7 as
> processing cases (evidence only) and activates the medication grammar (C1/C5)
> and laboratory parser (C2). See `docs/case_router/`.

## Multi-label routing

The router **does not** classify a whole document (or even a segment) into
exactly one case. A single segment may simultaneously carry several cases —
e.g. a pre-admission medication list containing a drug with ambiguous strength
is **C1 + C4 + C6**. There is no forced exclusivity. When signals are weak, the
default fallback route is **C3 (narrative)**.

Routing is versioned (`lexicon_version` in `configs/routes.yaml`); every cue
change must run regression tests so recall gains do not silently cost precision.

## The seven case families

### C1 — Medication-list
- **What:** numbered/bulleted medication lines; repeated route–frequency
  patterns; repeated drug names.
- **Routing signals:** structural (numbering, bullets, repeated route/frequency
  pattern); lexical (dose-form/route/frequency tokens); section priors
  (pre-admission / home / current medication list → `isHistorical` prior when
  the section implies it).
- **Preferred flow (spec):** medication grammar + section propagation +
  structured RxNorm linker.
- **Activates:** `mention_factory.medication_grammar` (E1), `specialists.rxnorm`,
  `boundary_type`.

### C2 — Lab semi-structured
- **What:** `name: value unit [flag] [reference-range]` rows; colons/semicolons,
  units, decimals, reference ranges.
- **Routing signals:** structural (colon/semicolon separation, units present,
  decimals, reference ranges); section priors (laboratory, investigations,
  hematology, chemistry).
- **Preferred flow (spec):** finite-state parser + test-result bipartite
  matching (`TEST_NAME –has_result→ VALUE [UNIT] [FLAG] [REFERENCE_RANGE]`).
- **Activates:** `mention_factory.lab_parser` (E2), `boundary_type`.

### C3 — Clinical narrative
- **What:** long sentences, multiple symptoms/diagnoses, soft boundaries.
- **Routing signals:** structural (long sentences, multiple clauses, soft
  boundaries); section priors (current examination, symptoms, diagnosis,
  conclusion). Default fallback route.
- **Preferred flow (spec):** span-NER ensemble + MRC/W2NER + contextual type
  scorer.
- **Activates:** `mention_factory.span_ner_ensemble` (E3–E6), `boundary_type`,
  `specialists.assertion`.

### C4 — Assertion-heavy
- **What:** negation / history / family cues; conjunctions and contrast.
- **Routing signals:** lexical cue families (negation/history/family — open
  lexicon per §4.3); structural (conjunctions, contrast markers, coordinated
  negation); section priors (medical history, family history).
- **Preferred flow (spec):** cue detector + scope model + section prior +
  relation classifier.
- **Activates:** `specialists.assertion`, `boundary_type`.

### C5 — Abbreviation/noise
- **What:** abbreviations (THA, DTD, qhs), accent errors, English-Vietnamese
  code switching.
- **Routing signals:** structural (unaccented tokens, mixed language, OCR-like
  artifacts); lexical (abbreviation alias families). Lower activation threshold
  (noise is easy to miss, cheap to propose).
- **Preferred flow (spec):** abbreviation lattice + character retrieval + LLM
  expansion verifier.
- **Activates:** `mention_factory.abbreviation_lattice`, `confidence_cascade`
  (LLM expansion verifier for hard cases).

### C6 — Ambiguous linking
- **What:** ICD parent/child ambiguity; same RxNorm ingredient with different
  strength/form.
- **Routing signals:** competing ontology candidates; strength/form variation.
- **Preferred flow (spec):** ontology hard negatives + cross-encoder +
  constrained judge.
- **Activates:** `specialists.icd`, `specialists.rxnorm`, `confidence_cascade`.

### C7 — Overlap/duplicates
- **What:** same text at multiple positions; nested/adjacent spans.
- **Routing signals:** repeated surface form, nested spans, adjacent spans.
- **Preferred flow (spec):** absolute offsets + interval/graph optimizer +
  **no text-only deduplication**.
- **Activates:** `evidence_graph`, `metric_decoder`. Repeated mentions at
  different offsets stay distinct.

## Suggested routing signals (hints, not a committed feature set)

- **Lexical:** cue-family hits (exact/regex/fuzzy against versioned lexicons),
  abbreviation density, unit-token density.
- **Structural:** list/bullet layout, colon-value pattern, line-length
  distribution, delimiter profile.
- **Discourse:** section prior (from L1), repeated surface forms (supports C7).
- **Learned:** segment-classifier logits (contextual multi-label classifier —
  future work).

## Case → specialist activation matrix

| Case | RxNorm | ICD | Assertion | Med grammar | Lab parser | Span-NER ensemble | Abbrev lattice | Evidence graph | Confidence cascade | Metric decoder |
| ---- | :----: | :-: | :-------: | :---------: | :--------: | :---------------: | :------------: | :------------: | :----------------: | :------------: |
| C1   | ✅     |     |           | ✅          |            |                   |                |                |                    |                |
| C2   |        |     |           |             | ✅         |                   |                |                |                    |                |
| C3   |        |     | ✅        |             |            | ✅                |                |                |                    |                |
| C4   |        |     | ✅        |             |            |                   |                |                |                    |                |
| C5   |        |     |           |             |            |                   | ✅             |                | ✅                 |                |
| C6   | ✅     | ✅  |           |             |            |                   |                |                | ✅                 |                |
| C7   |        |     |           |             |            |                   |                | ✅             |                    | ✅             |

`boundary_type` (L4) runs for every route that produces spans; L4–L9 always
execute on the merged span lattice regardless of case. The matrix records the
**case-specific** activations from spec section 5.
