# MedNorm-VI — Layer Contracts

> Per-layer input/output contracts for the nine processing layers. Derived from
> `docs/MedNorm-VI_Architecture.pdf` (sections 3–13, 16, 19). These are **interface
> contracts**, not implementations. Every layer is mandatory — do not silently
> simplify or bypass a layer.

## Cross-cutting invariants (apply to every layer)

- **Offset preservation.** `original_text` is immutable and is the only source
  for emitted `text`/`position`. For every emitted span,
  `original_text[start:end] == text` and `position` is `[start, end)`
  (end-exclusive). Normalization happens on a copy with a preserved alignment;
  **normalized coordinates never become output coordinates.**
- **Coordinate triplet.** Every span-bearing object distinguishes:
  1. **local** — coordinates within its segment,
  2. **absolute** — coordinates into `original_text` (authoritative for output),
  3. **normalized** — coordinates into `normalized_view` (optional; matching only).
- **Provenance.** Every proposal/decision records where it came from
  (`source_model`/rule ids/route) and a score, for replay and ablation.
- **Confidence.** Scores are explicit and calibratable; downstream layers may
  abstain. No layer emits a final entity except through L8 → L9.
- **Fail-fast.** On a substring/end-index violation or an out-of-KB code, stop
  that document and write an audit log; never silently repair output.

---

## L1 — Document Intelligence
- **Input:** raw UTF-8 `original_text` of one document.
- **Output:** `DocumentGraph` (sections, sentences, list rows, table-like cells,
  segments) + `NormalizedView` with an `OffsetAlignment` back to `original_text`.
- **Responsibility:** segmentation; section/discourse parsing with priors;
  reversible Unicode/abbreviation/unit normalization; list splitting that
  retains separators and absolute offsets.
- **Must not:** trim whitespace, re-case, or rewrite punctuation on the original;
  lose or approximate original offsets; treat a section prior as an absolute rule.
- **Provenance/confidence:** each node stores its absolute offsets; section nodes
  store the matched pattern and a `prior_strength`.
- **Offsets:** builds the alignment used by every later layer; guarantees every
  normalized code point maps back to an original code point.

## L2 — Case Router
- **Input:** `DocumentGraph` segments (+ section priors).
- **Output:** for each segment, a set of `RouteTag`s (a `RouteDecision`) with
  per-case scores and applied priors. **Multi-label** — a segment may carry
  several cases (e.g. C1+C4+C6).
- **Responsibility:** score C1–C7 from lexical/structural/section/learned
  signals; attach priors (e.g. `isHistorical` for pre-admission sections).
- **Must not:** force single-label exclusivity; drop a segment for lacking a
  strong case (fallback route `C3`); emit spans or entities.
- **Provenance/confidence:** each `RouteTag` stores the case id, score, the
  signals that fired, and `lexicon_version`.
- **Offsets:** carries segment offsets through unchanged.

## L3 — Mention Factory
- **Input:** routed segments.
- **Output:** a **span lattice** — many overlapping `SpanProposal`s from experts
  E1–E7. High recall by design.
- **Responsibility:** propose every plausible entity span (grammar, lexicon,
  span/W2NER/MRC/GLiNER encoders, LLM proposer for hard cases).
- **Must not:** emit final entities; deduplicate prematurely; let any single
  expert decide a boundary unilaterally; let the LLM proposer invent text that
  is not an exact substring.
- **Provenance/confidence:** each `SpanProposal` stores `source` (expert id),
  `local_score`, `route`, `proposed_types`, and grammar/lexicon features
  (`SpanProvenance`).
- **Offsets:** every proposal carries the absolute coordinate triplet; proposer
  substrings are validated against `original_text`.

## L4 — Boundary & Type Resolver
- **Input:** the span lattice (overlapping `SpanProposal`s) + route features.
- **Output:** `TypedHypothesis` objects (selected/trimmed/merged spans with a
  `type_distribution`).
- **Responsibility:** boundary offset selection to the organizer's convention;
  type scoring; **abstention** on high-entropy or type-ontology conflicts;
  global span optimization (weighted interval/graph).
- **Must not:** deduplicate by text alone; merge spans at different positions;
  assign ICD to MEDICATION or RxNorm to DIAGNOSIS; guess a type when wrong-type
  risk outweighs the gain.
- **Provenance/confidence:** stores `φ(H)` feature contributions, the chosen
  boundary rationale, and a calibratable type distribution.
- **Offsets:** boundary changes are re-validated against `original_text`.

## L5 — Specialists (Assertion / ICD / RxNorm)
- **Input:** `TypedHypothesis` objects (+ `DocumentGraph` context).
- **Output:** `EvidenceBundle` per hypothesis containing `AssertionEvidence` and
  `CandidateEvidence` (scored, not final).
- **Responsibility:**
  - *Assertion:* cue + scope + section + experiencer → multi-label assertion
    distribution.
  - *ICD:* candidate ICD-10 codes for DIAGNOSIS via multi-index retrieval +
    rerank; specificity control; frozen KB.
  - *RxNorm:* candidate RxCUIs for MEDICATION via structured parse + graph
    search; SCD/SBD handling; frozen KB.
- **Must not:** edit spans (may only emit scored boundary *feedback*); emit codes
  absent from the frozen KB; recall IDs from LLM memory; cross-link ontologies.
- **Provenance/confidence:** each candidate stores retrieval source(s), scores,
  hierarchy/graph provenance; each assertion stores contributing cues/scope.
- **Offsets:** consumes offsets read-only; never mutates them.

## L6 — Evidence Graph
- **Input:** hypotheses + evidence bundles.
- **Output:** a clinical graph (nodes: spans/sections/cues/list-items/lab-values/
  candidates; edges: `has_result`, `modified_by`, `in_section`, `treats`,
  `overlaps`, `same_surface`, `candidate_of`) with global consistency features.
- **Responsibility:** relation reasoning and cross-entity consistency
  (section consistency, test–result pairing, drug–indication, overlap
  competition, repeated-mention handling).
- **Must not:** collapse repeated mentions at different offsets; overwrite
  specialist evidence (adds features, does not replace).
- **Provenance/confidence:** each edge/feature records its supporting nodes.
- **Offsets:** node identity includes absolute offsets; `same_surface` links do
  not merge distinct-offset nodes.

## L7 — Confidence Cascade
- **Input:** hypotheses + evidence graph (+ margins/disagreement signals).
- **Output:** adjudicated hypotheses with calibrated confidences.
- **Responsibility:** route only difficult/high-impact cases to heavier models
  (critic Qwen3-1.7B, adjudicator Qwen3-4B); easy cases bypass the LLM.
- **Must not:** let the LLM change `start`/`end` unless the span is in the locked
  option set; let it emit IDs outside the retrieved top-K; store chain-of-thought
  (store only the final structured decision + auditable evidence).
- **Provenance/confidence:** records entry condition, model used, and the locked
  option set presented.
- **Offsets:** offsets are immutable inputs to the LLM prompt contract.

## L8 — Metric Decoder
- **Input:** calibrated hypotheses (spans, assertion dists, candidate dists).
- **Output:** the metric-optimal prediction set (final `EntityPrediction`s
  before validation).
- **Responsibility:** expected-WER span selection; expected-Jaccard candidate-set
  decoding (no fixed top-K); per-label assertion thresholds; entity
  existence/type utility net of wrong-type risk.
- **Must not:** apply a fixed 0.5 threshold or blind top-1; deduplicate by text
  alone.
- **Provenance/confidence:** records the expected-utility rationale per decision.
- **Offsets:** selected spans keep their exact absolute offsets.

## L9 — Validator
- **Input:** final `EntityPrediction`s per document.
- **Output:** the CONFIRMED submission — `output.zip` containing one top-level
  `output/` directory with `1.json` … `100.json`, or a fail-fast audit.
- **Serialization:** `EntityPrediction.to_submission_dict` emits the
  **organizer-facing** shape — `type` becomes the exact Vietnamese label
  (`THUỐC`, `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`,
  `KẾT_QUẢ_XÉT_NGHIỆM`) and **only the fields allowed for that type** are
  written (see `ORGANIZER_FIELDS_BY_TYPE`). JSON is UTF-8 with
  `ensure_ascii=false`.
- **Responsibility:** deterministic checks — substring/offset invariant, schema,
  allowed types & assertion labels, per-type field policy, duplicate policy,
  ordering, packaging (dir + ZIP), correct-ontology-for-type (RxNorm=numeric,
  ICD-10=string); parameter-budget verification. Organizer-facing validation
  (`validate_organizer_entity` / `validate_organizer_document`) rejects English
  `type` values and unsupported fields; submission validation
  (`validate_output_directory` / `validate_submission_zip`) enforces the file
  set, one top-level `output/` dir, UTF-8, list roots, and rejects unsafe/extra
  paths.
- **Must not:** repair invalid output silently; accept a code outside the frozen
  KB; accept `original_text[start:end] != text`; emit English `type` values or
  unsupported fields.
- **Deferred:** KB-membership of candidate codes (until frozen KB snapshots
  exist); candidate syntax is kept light (numeric RxCUIs; ICD-10 as strings).
- **Provenance/confidence:** emits a structured validation report; failures name
  the document and offending entity.
- **Offsets:** the final gatekeeper for the substring/end-exclusive invariant.

---

## Data-contract summary (see `src/mednorm_vi/schemas/`)

```
SpanProposal:   start, end, text, proposed_types, source, local_score, route, features
TypedHypothesis: span, type_distribution, ... , calibrated_score, evidence_ids
EvidenceBundle: assertion_evidence[], candidate_evidence[]
EntityPrediction (internal): text, type(enum), assertions, candidates, position  # [start, end)
  .to_submission_dict() -> organizer shape: type=Vietnamese label, per-type fields only
```

Every span-bearing object carries the (local, absolute, normalized) coordinate
triplet defined above.
