# MedNorm-VI — Architecture Digest

> Faithful engineering summary of the authoritative English specification
> `MedNorm-VI_Architecture.pdf` (Super Architecture Specification, Version 1.1,
> 08 July 2026). This digest **does not invent requirements**; where the PDF is
> silent or defers to organizer data, that is stated explicitly. The PDF remains
> the source of truth — read it before any architecture-sensitive change.

## 1. Task

MedNorm-VI converts free-form **Vietnamese clinical text** into a set of
structured entity hypotheses:

```
H = (start, end, text, type, assertions, candidates)
```

- **Scope:** qualification round, **100 input documents → 100 JSON outputs**.
- Each output entity carries: `text`, `type`, `assertions`, `candidates`, `position`.
- `position` is `[start, end)` (end-exclusive) into the original text.

### Metric (section 1)

| Component  | Scoring              | Weight |
| ---------- | -------------------- | ------ |
| text       | 1 − Word Error Rate  | 30%    |
| assertions | multi-label Jaccard  | 30%    |
| candidates | Jaccard              | 40%    |
| type       | no direct weight, but a **wrong type is double-penalized** (false positive + false negative) |

Optimization chain (spec): entity recall → correct type/boundary → correct
assertion/linking → metric-optimal output set. The decoder maximizes **expected**
score, not a fixed 0.5 threshold.

Practical objective (section 1.1):
`S(H) = α·Sspan + β·Stype + γ·Sassert + δ·Slink + ε·Ssection + ζ·Sconsistency − λ·Risk(wrong_type)`.

## 2. Fixed vocabularies

- **Five entity types (internal enums):** `MEDICATION`, `DIAGNOSIS`, `SYMPTOM`,
  `TEST_NAME`, `TEST_RESULT`. (The PDF names "five entity categories"; these five
  surface throughout sections 4.2 and 7.2.) Organizer-facing JSON serializes
  `type` to the confirmed Vietnamese labels `THUỐC`, `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`,
  `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM` (see `LAYER_CONTRACTS.md` L9 and ADR 0001).
- **Three assertion labels:** `isNegated`, `isHistorical`, `isFamily` (multi-label).
- **Candidates:** `DIAGNOSIS → ICD-10 codes`, `MEDICATION → RxCUIs`. Cross-links
  forbidden (no ICD on MEDICATION, no RxNorm on DIAGNOSIS — section 7.3).

## 3. Core principles (section 2)

- **P1** Specialized encoders for spans; LLMs only for ambiguity.
- **P2** NER as a multi-representation ensemble (W2NER, MRC-NER, T2-NER).
- **P3** Linking is multi-stage and ontology-aware (BioSyn, SapBERT, KRISSBERT, GenBioEL).
- **P4** Assertion = cue + scope + section + experiencer.
- **P5** Pipeline and joint inference coexist (specialist pipelines → global decoder).
- **P6** Synthetic data must be controlled/filtered and must preserve exact spans.
- **P7** LLMs must not recall ontology IDs from memory — only select from retrieved candidates.
- **P8** Route by case; do not apply one model to every structure.

## 4. Nine processing layers (section 3)

| Layer | Name                     | Responsibility                                   | Primary output          |
| ----- | ------------------------ | ------------------------------------------------ | ----------------------- |
| L1    | Document Intelligence    | Segmentation, sections, lists, sentences, reversible normalization | DocumentGraph |
| L2    | Case Router              | Assign one or more cases per segment (multi-label) | Route tags + priors   |
| L3    | Mention Factory          | Propose every plausible entity span (high recall) | Span lattice           |
| L4    | Boundary & Type Resolver | Select, trim, merge, and type spans              | Typed hypotheses        |
| L5    | Specialists              | Assertion, ICD, and RxNorm processing            | Evidence bundles        |
| L6    | Evidence Graph           | Connect entity–section–cue–result–candidate evidence | Clinical graph      |
| L7    | Confidence Cascade       | Invoke heavy models only when needed             | Adjudicated hypotheses  |
| L8    | Metric Decoder           | Select entity/assertion/candidate sets           | Optimal prediction set  |
| L9    | Validator                | Check schema, offsets, packaging                 | 100 JSON files + ZIP    |

## 5. Document Intelligence & offset preservation (section 4)

Three text representations:

| Representation   | Purpose                                          | May change? |
| ---------------- | ------------------------------------------------ | ----------- |
| `original_text`  | The **only** source used to emit text and position | **No**    |
| `normalized_view`| Matching, retrieval, abbreviation/unit normalization | Yes, but every char maps back |
| token/segment graph | Sentences, sections, list items, delimiters, cues, table-like rows | Yes; every node stores indexes into `original_text` |

Invariant: `assert original_text[start:end] == entity["text"]`. Never trim
whitespace / rewrite punctuation before storing a span. Unicode normalization is
performed on a copy with preserved alignment. Every span proposal stores
`source_model`, `local_score`, `route`, normalized form, and original
coordinates. Reject output on any substring/end-index violation.

**Sections are evidence, not absolute rules** (section 4.2): e.g. a
"pre-admission medication list" creates an `isHistorical` prior, but a local
"started today" may override it. Lexicons (section 4.3) are open, versioned by
semantic group, with provenance and regression tests; a cue only *contributes*
evidence — keyword presence must never label every entity.

## 6. Case Router (section 5) — see `CASE_ROUTING.md`

Multi-label. C1 Medication-list, C2 Lab semi-structured, C3 Clinical narrative,
C4 Assertion-heavy, C5 Abbreviation/noise, C6 Ambiguous linking, C7
Overlap/duplicates. A segment may carry several (e.g. C1+C4+C6).

## 7. Mention Factory (section 6)

Produces a unified **span lattice** from multiple experts; **no module emits a
final entity directly**. Experts: E1 medication grammar, E2 lab parser, E3
ViHealthBERT span model, E4 PhoBERT W2NER, E5 XLM-R MRC-NER, E6 GLiNER, E7
Qwen3-1.7B proposer (proposal-only, hallucination-prone).

- Medication grammar (6.1): `MED = NAME [SALT] [RELEASE] [STRENGTH|CONCENTRATION] [DOSE_FORM] [ROUTE] [FREQUENCY] [PRN]`; generate multiple boundary candidates, let the resolver learn the organizer's convention.
- Lab parser (6.2): `TEST_NAME –has_result→ VALUE [UNIT] [FLAG] [REFERENCE_RANGE]`; internal relation, not emitted in round-one output.

## 8. Boundary & Type Resolver (section 7)

Feature vector `φ(H) = [model logits, grammar completeness, lexicon match,
section prior, boundaries, ontology evidence, route features]`. Boundary-offset
head predicts left/right expand/contract; loss combines exact-span, token IoU,
WER surrogate. **Abstention:** because wrong type is double-penalized, high-
entropy or type-ontology-conflicting hypotheses may be dropped. Global span
optimization: never deduplicate by text alone; merge only when position **and**
semantic identity match; forbid ICD on MEDICATION / RxNorm on DIAGNOSIS.

## 9. Specialists (L5)

- **Assertion Hydra (section 8):** stages A1 section prior → A2 cue detector →
  A3 scope resolver → A4 entity-cue classifier → A5 LLM adjudicator (only on
  disagreement) → A6 set calibration. Multi-label, not exclusive softmax.
  Vietnamese scope rules (8.1): negation typically stops at contrast markers;
  coordinated negation creates multiple scopes; family experiencer may be far
  from the entity (use dependency/discourse edges).
- **ICD-10 Super Linker (section 9):** KB frozen to organizer version. Multi-
  representation indexes (exact alias, char n-gram, BM25/sparse, BGE-M3, Qwen3
  embedding, SapBERT/BioSyn, hierarchy graph). Candidate generation → weighted
  RRF fusion → parent/child/sibling expansion → cross-encoder rerank → Qwen3-4B
  selects only from top 5–10 (cannot invent a code) → calibrated set decoding.
  Specificity controller balances parent vs child codes.
- **RxNorm Super Linker (section 10):** structured drug representation
  `D = {ingredient[], salt, strength[], unit, concentration, dose_form, release, brand, route, frequency, prn}`.
  Graph search over ingredient/component/SCD/brand/dose-form. Hard negatives:
  same ingredient/different strength (most important), then same strength/
  different release/form. Term types SCD vs SBD honored.

## 10. Clinical Evidence Graph (section 11)

Nodes: spans, sections, cues, list items, lab values, ontology candidates.
Edges: `has_result`, `modified_by`, `in_section`, `treats`, `overlaps`,
`same_surface`, `candidate_of`. Global inference favors beam search / weighted
intervals first (debuggability); ILP is a research branch. Repeated mentions at
different offsets are kept as distinct nodes.

## 11. Confidence Cascade & LLM committee (section 12)

Levels: Fast path (exact grammar/alias, high margin → accept) → Specialist path
(disagreement → evidence bundle) → Critic (Qwen3-1.7B) → Adjudicator
(Qwen3-4B, hard/high-impact) → Global decoder (no LLM). Prompt contract (12.1):
the LLM cannot change `start`/`end` unless the span is in the option set, cannot
emit IDs outside retrieved top-K, and **chain-of-thought is not stored** — only
the final structured decision + auditable evidence.

## 12. Metric-aware Set Decoder (section 13)

- **Text/boundary:** maximize expected `1 − WER` over competing spans.
- **Candidate set:** search prefixes/top combinations to maximize expected
  Jaccard `P* = argmax_P E[|P∩G| / |P∪G|]`. Top-1 not always optimal; no fixed top-K.
- **Assertion set:** per-label calibrated thresholds under class imbalance.
- **Entity existence/type:** utility subtracts wrong-type risk; a near-correct
  span with uncertain type may be worse than omission.

## 13. Data Engine, training, inference, budget

- **Data Engine (14):** organizer gold, ontologies, public VN corpora, synthetic
  templates, weak labeling (Snorkel-style), teacher consensus. Controlled
  synthesis by case; active-learning priority `= disagreement · metric_weight ·
  uncertainty · novelty · frequency`.
- **Training (15):** S0 domain adaptation → S1 mention → S2 assertion → S3
  retrieval → S4 reranking → S5 LLM critic/judge → S6 calibration/stacking.
  Shared backbones with multiple adapters/heads; group-split CV against leakage.
- **Inference (16):** 10-step flow, **fail-fast** on any
  `original_text[start:end] != text` or code absent from the frozen KB.
- **9B budget (17):** Safe Submission Stack ≈ **8.852B base**. Adapters/heads
  excluded. Operate near ~8.85B for margin. Variants: Safe-8.85B (default),
  Alt-8.99B, Lean-6.5B, No-LLM-baseline. See `configs/parameter_budget.yaml`.

## 14. Open decisions requiring organizer data (section 21.1)

These are **unresolved in the spec** and must not be guessed:

- Does entity matching use position, or only text + type?
- What exact tokenization/normalization is used for WER?
- Which ICD-10 version and RxNorm release are provided?
- Is the candidate list length limited, and does ordering matter?
- Are nested/overlapping entities present?
- Will later rounds include an explicit relation field?
- Are there time/memory limits for private rebuild?

## 15. Repository & roadmap (sections 19–20)

Modules should be pure-ish functions over an immutable `DocumentGraph`; every
decision stores provenance for replay/ablation; linking modules may not silently
edit spans (only emit scored boundary feedback). Roadmap order is **fixed**:
Phase 0 (contract + evaluator + validator) must complete **before** heavy models,
"otherwise model changes cannot be judged against the organizer's true metric."
This bootstrap implements the deterministic foundation of Phase 0.
