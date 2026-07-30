# MedNorm-VI — Active Runtime Architecture Manifest

**Supplemental to `docs/MedNorm-VI_Architecture.pdf`, never a replacement for it.**
The PDF is the primary source of truth and is unmodified
(SHA-256 `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`). It
describes a candidate *super*-architecture. This file records what the repository
**actually** implements today, so a reader can tell a designed component from a
built one without running the code.

- **Established by:** Audit 0051 (repository-wide architecture review and cleanup)
- **Last updated by:** Audit 0054 (deterministic L5-L7: structured RxNorm linking, ICD
  hierarchy + specificity, all L6 edges, the typed consistency contract, L7 locked-option
  escalation, the inference run manifest, the code-linking evaluation contract). Audit
  0054 is a **partial** milestone: of Milestone 2B's seven carried-forward tasks plus
  container verification, **six are complete**, the old-L4 deletion is **characterized
  but not migrated**, and the **Docker build is blocked by the environment** — §13 below
  marks both.
- **Date:** 2026-07-30
- **Rule this file obeys:** a module is never described as implemented when it has
  no trained checkpoint, no wiring, or only a contract. "Implemented" here means
  code that runs and is exercised by tests; it does **not** mean measured against
  the organizer metric.

---

## 0. Headline status

```text
E4 PhoBERT-W2NER:
RETIRED_FROM_ACTIVE_ARCHITECTURE
```

Its implementation, notebook, configs and tests were deleted in Audit 0051. The
surviving record is `src/mednorm_vi/governance/e4_retirement.py` plus Audits
0043-0048 and 0051. It is absent from `AVAILABLE_EXPERTS`, from every feature-flag
set, from every registry, from every ablation arm and from every parameter ledger,
and `PipelineConfig.load` **refuses** any profile that declares its flag.

The historical record, stated precisely:

```text
E4 produced training, diagnostic and reproduction checkpoints.
No E4 checkpoint passed the acceptance gate.
No E4 checkpoint is retained in the current tree or active deployment.
Obsolete E4 artifacts were deleted after evidence was recorded in the
append-only audits.

encoder / base parameters       369,163,264
W2NER head parameters             2,125,897
total parameters                371,289,161
count status                    PROGRAMMATICALLY_VERIFIED (Audit 0044)
```

Audit 0044 restored `best.pt` and `latest.pt` and reconciled both instantiated
models exactly to 371,289,161 parameters, with zero missing and zero unexpected
keys on encoder and head alike. **This measured total is not the same number as the
~370M planning estimate** that spec §17 lists for PhoBERT-large; §12 keeps the two
apart deliberately.

| Fact | Value |
| --- | --- |
| Layers with a trained checkpoint | **1 of 9** (L3, the E3 mention expert) |
| **E3 runs through the canonical runner** | **YES, since Audit 0052** — real inference, verified checkpoint, provenance recorded |
| Canonical L3 lattice wired into the runner | **YES, since Audit 0052** |
| Canonical L4 entry points | **1** (`resolution.canonical`), since Audit 0052 |
| **Deterministic experts are route-gated with recorded suppression** | **YES, since Audit 0053** — `required_evidence` per case in `signals_v1`; suppression reasons carried on `NodeRouting.gate_reasons` |
| **L8 no longer claims metric-aware decoding** | **Renamed and re-implemented in Audit 0053** — `decode_entities`, evidence-tier + relative-score band; the calibrated path raises `CalibratedDecoderUnavailable` |
| **L9 enforces KB membership independently of the linker** | **YES, since Audit 0053** — `validator/kb_membership.py`, imports no linker or model, and the canonical packaging path invokes it before writing anything |
| Layers implemented and wired | L1, L2, L3 (E1+E2+E3), L4 (canonical over the lattice), L5 assertion (deterministic, §8.1 scope), **L5 linking (structured RxNorm + ICD hierarchy)**, **L6 (9 edges + typed consistency)**, **L7 (deterministic fallback + locked-option escalation contract)**, L8 (deterministic, honestly named, consuming L6/L7), L9 (incl. KB membership) |
| Layers that are contract-or-scaffold only | L7 **model** stages (no local weights), L8 **calibrated** expected-Jaccard/WER decoding (fails loudly), L6 global *optimization* (beam/ILP — the graph is now read, but not searched) |
| Reproducibility infrastructure | `Dockerfile` + `requirements.lock` + **`requirements-image.lock`** (Audit 0054). The lock's `torch==2.13.0+cu126` pin **cannot install from PyPI**, which Audit 0054's first real build proved; the image file records the deviation. `output.zip` is now **byte-reproducible** (fixed ZIP entry timestamps). **The image has still never been built — see §10.** |
| `output.zip` ever produced | **No** |
| Organizer metric ever computed | **No** — only exact mention span/type |
| Organizer inference ever run | **No** |
| Candidate Recall@K ever measured | **No** — the governed corpus carries **zero** ontology codes. Audit 0054 added the strict evaluation contract (`evaluation/code_linking.py`), which **refuses to score** rather than returning zeros; the status constant is `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD` |
| **RxNorm consumes E1's structured components** | **YES, since Audit 0054** — `linking/structured_medication.py` + `linking/rxnorm.py`; ingredient/strength/unit/concentration/dose-form/release/route conflicts are hard negatives |
| **RxNorm graph traversal against the real snapshot** | **YES, since Audit 0054** — IN → SCDC → SCD → SBD by TTY role. The graph carries **no relation labels**, so the walk is TTY-inferred and every path is recorded (see §5.3) |
| **ICD-10 hierarchy + specificity controller** | **YES, since Audit 0054** — ancestor/descendant/sibling expansion, and a descendant is kept only when the mention states the detail it adds |
| **All supportable L6 edge types** | **YES, since Audit 0054** — 9 relations; `treats` requires explicit evidence and co-occurrence is refused |
| **A typed deterministic consistency report exists and L7/L8 consume it** | **YES, since Audit 0054** — `evidence_graph/consistency.py`, 12 rules, 4 verdicts; a fatal contradiction can no longer be silently accepted |
| **L7 locked-option escalation contract** | **YES, since Audit 0054** — decisions are expressed as option ids only, so a code, label or coordinate that was not offered cannot be returned. **No model is loaded** |
| **Deterministic run manifest** | **YES, since Audit 0054**, opt-in via `--run-manifest`; no clinical text, byte-identical apart from two documented runtime fields |
| **Docker image built and smoke-tested** | **NO** — blocked by the environment, not by the Dockerfile (§10 below) |
| **Obsolete non-canonical L4 deleted** | **NO** — characterized in 13 tests, not yet migrated (§13 below) |

**Organizer readiness is not claimed.** One mention expert now runs end to end; the
candidate-set and assertion halves of the metric remain unimplemented and unmeasured.

### Measured, Audit 0052 (bounded: 200 of 1,045 governed validation examples)

Exact character-offset span **and** type, through the canonical runner:

```text
arm                       P        R       F1     TP    FP    FN
E1+E2 deterministic   0.0000   0.0000   0.0000     0    11   406
E3 only               0.6220   0.5025   0.5559   204   124   202
E1+E2+E3 lattice      0.6018   0.5025   0.5477   204   135   202
```

Two things this says plainly. **E1/E2 score zero on this corpus** — it is narrative
DIAGNOSIS/SYMPTOM text and they are medication/laboratory parsers, so their 11
proposals are all false positives; this reproduces Audit 0034's Arm-2 result exactly.
And **adding them to E3 lowers F1** (0.5477 against 0.5559), which is why
`deterministic` and `specialist` are separate modes rather than one merged profile.

This figure is **not comparable** to Audit 0032's `validation_span_micro_f1`
0.7194: that was a token-index span proxy over the full 1,045 rows computed during
training, this is exact character offsets **plus** exact type over 200 rows. Neither
is the organizer's complete metric.

### Measured, Audit 0053 (same bounded 200 rows, same digest-resolved split)

Route gating changed only E1/E2 behaviour, so the E3-only arm is unchanged:

```text
arm                                        P        R       F1     TP    FP    FN
E1+E2, before gating (equivalent)     0.0000   0.0000   0.0000     0    11   406
E1+E2, after gating                   0.0000   0.0000   0.0000     0     5   406
E3 only (reference, unchanged)        0.6220   0.5025   0.5559   204   124   202
E1+E2 gated + E3 lattice              0.6126   0.5025   0.5521   204   130   202
```

E1/E2 still score zero — this corpus is narrative DIAGNOSIS/SYMPTOM text and they
are medication/laboratory parsers, so **zero is the correct score and the only
improvement available is fewer false positives**. Gating delivered that:

```text
E2 proposals on 200 narrative docs      16  ->   6     (-62%)
documents producing E1/E2 proposals      8  ->   4
E1/E2 false positives after L4          11  ->   5
C2 routes suppressed for lack of positive evidence      7 nodes
fixture recall (medication/laboratory/mixed/hard-neg)   UNCHANGED, exactly
merged-arm F1                       0.5477  -> 0.5521
```

Adding gated E1/E2 to E3 still costs F1 (0.5521 against E3-only 0.5559), so
`deterministic` and `specialist` remain separate modes; gating narrowed the gap
without closing it.

L8 candidate-set sizes over the same 200 documents, where the old decoder returned
an unconditional 10:

```text
size    0:133   1:7   2:4   3:4   4:1   5:4   6:1   7:3   8:1   9:6  10:2
       11:4  12:4  13:1  14:1  15:1  16:2  17:2  18:3  19:4  20:145
candidates offered by L6 4,000  ->  emitted 3,390     (610 dropped by band/tier)
L9 KB membership violations                     0
```

The 145 sets of exactly 20 are the **retriever's** top-K cap, not a decoder rule:
those mentions had 20 same-tier candidates inside the score band. The 133 empty sets
are SYMPTOM/TEST\_\* entities, which take no candidates at all (spec §7.3).

**Candidate Recall@K remains unmeasurable and is not reported.** Every one of the
406 gold entities in these rows carries `target_type` and `mapping_status:
MAP_EXACT` but **no ontology code** — the governed corpus has no ICD-10 or RxNorm
gold, and no weak-evidence code field either. Recall@K needs a gold code set that
does not exist yet.

---

## 1. L1 — Document Intelligence

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/document_intelligence/` (16 files, 2,109 lines); offsets `src/mednorm_vi/position/` (6 files, 641 lines) |
| Status | `IMPLEMENTED_AND_WIRED` |
| Deterministic implementation | Yes — full: Unicode-safe normalization with reversible alignment, sections, sentences, lines, list items, table-like rows, key–value, BOM handling |
| Pretrained adapter | None required |
| Trained checkpoint | None required |
| Future checkpoint slot | None |
| Enabled | Always; `configs/document_intelligence/base.yaml` |
| Contract version | `L1Config` + `SectionLexicon`, `configs/document_intelligence/section_lexicon_v1.yaml` |
| Parameter-ledger behaviour | Contributes 0; deterministic |
| Known missing | Section lexicons are seed-sized against spec §4.3's "high-coverage expression lexicon" requirement; no discourse-override model |

## 2. L2 — Case Router

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/case_router/` (7 files, 743 lines) |
| Status | `IMPLEMENTED_AND_WIRED` |
| Deterministic implementation | Yes — multi-label C1–C7 routing with signals and priors |
| Pretrained adapter | None |
| Trained checkpoint | None |
| Future checkpoint slot | A learned router is **not** in the active plan |
| Enabled | Always; `configs/case_router/base.yaml`, `configs/case_router/signals_v1.yaml`, `configs/routes.yaml` |
| Contract version | `signals_v1` (extended in Audit 0053 with `required_evidence`) |
| Parameter-ledger behaviour | Contributes 0 |
| Positive-evidence gate | **Added in Audit 0053.** A case may declare `required_evidence`; reaching `activate` is then necessary but not sufficient. C2 requires one of `numeric_key_value`, `lab_unit`, `reference_range`, `flag`, `lab_section`. `numeric_value` is deliberately **excluded** — it fires on any numeral and was present in every measured false positive |
| Suppression is recorded | `NodeRouting.gate_reasons` names each suppressed case, its score and what it required, so a routing defect is visible without reading parser source |
| Known missing | **Routing accuracy has never been measured.** No labelled route data exists, so the router is unvalidated rather than validated-and-good. The Audit-0053 gate was validated by its *effect* (false positives down 62% at unchanged fixture recall), not against route gold |

## 3. L3 — Mention Factory

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/mention_factory/` (registry + experts + specialists); unified lattice `src/mednorm_vi/lattice/` |
| Status | `PARTIALLY_IMPLEMENTED_AND_INTEGRATED` — three experts run (E1, E2, E3); three await weights or training |
| Expert entry point | `mention_factory/registry.py` — the runner consults the registry and names no expert of its own, so a new expert needs **no** runner edit |
| Contract version | `mention-span-v1` (`spans.py`), `mention-offsets-v1` (`offsets.py`), lattice `EXPERT_FAMILY`/`AVAILABLE_EXPERTS`, `route-gate-diagnostics-v1` (`route_gate.py`) |
| Route eligibility is reported | **Audit 0053.** `mention_factory/route_gate.py` declares each expert's routes (E1 → C1/C5/C3, E2 → C2) and `PipelineResult.route_gate` carries eligible/skipped nodes, per-expert eligibility, proposals per expert and suppression reasons. It carries **no clinical text** — a test asserts this |
| Parameter-ledger behaviour | E1/E2 contribute 0; every neural expert must be counted programmatically before deployment |

Per expert:

| Expert | Status | Checkpoint | Enabled | Notes |
| --- | --- | --- | --- | --- |
| E1 medication grammar | `IMPLEMENTED_AND_WIRED` | none needed | **yes** | Finite-state grammar per spec §6.1; `configs/medication/grammar_v1.yaml` |
| E2 laboratory parser | `IMPLEMENTED_AND_WIRED` | none needed | **yes** | `TEST_NAME –has_result→ VALUE` per spec §6.2; `configs/laboratory/parser_v1.yaml` |
| E3 ViHealthBERT span+type | **`IMPLEMENTED_AND_INTEGRATED`** | **exists and RUNS** — `checkpoint/s1_mention_full_training_v1/best.pt`, SHA-256 `a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c`, 1,615,513,303 bytes, git-ignored; byte-identical Drive backup verified by MD5 in Audit 0052 | **yes, in `specialist`** | Adapter `mention_factory/experts/e3_vihealthbert.py`; lazy, `local_files_only=True`, backbone never downloaded (the checkpoint carries the full state dict). Exact span+type F1 **0.5559** on 200 governed validation rows (Audit 0052). Audit 0032's 0.7194 / 0.746182 are token-index proxies and are not the same measurement |
| E4 PhoBERT W2NER | **`RETIRED_FROM_ACTIVE_ARCHITECTURE`** | **none retained.** Training, diagnostic and reproduction checkpoints WERE produced; none passed the acceptance gate; none remains in the tree | **removed** | See §0 |
| E5 XLM-R MRC-NER | `REQUIRES_FUTURE_TRAINING` | none | no | Contract + training path implemented (`mention_factory/mrc.py`, `training/phase2/e5_mrc_training.py`). Its task head is **randomly initialized**; it must never run at inference untrained |
| E6 GLiNER open-type | `REQUIRES_PRETRAINED_CHECKPOINT` | none locally | no | Strict adapter `mention_factory/gliner.py` fails closed without a verified local checkpoint; loader `llm/backends.OpenTypeSpanBackend` |
| E7 Qwen proposer | `REQUIRES_PRETRAINED_CHECKPOINT` | none locally | no | Proposal-only. Loader `llm/backends.CausalLMBackend`; offsets are never taken from it — resolved by `mention_factory/offsets.resolve_occurrence` and re-verified |

Known missing: E5's head is untrained; E6/E7 have no local weights; TEST_NAME and
TEST_RESULT have **zero** governed supervision, so two of the five spec types are
unlearned by E3 (confirmed by the Audit-0052 per-type table: E3 predicts only
DIAGNOSIS and SYMPTOM).

## 4. L4 — Boundary & Type Resolver

| Field | Value |
| --- | --- |
| Implementation | **canonical entry point `resolution/canonical.py`** over `SpanLattice`, wrapping `resolution/resolver_v1.py`'s decision logic; `resolution/learned_v2.py` is the future learned slot |
| Status | `IMPLEMENTED_AND_INTEGRATED` (deterministic); learned v2 implemented, untrained |
| Canonical entry points | **1.** Audit 0052 replaced the two-live-L4 situation. `resolution/resolver.py` is off the canonical path (a test asserts the runner does not import it) and **still exists**: Audit 0054 characterized its unique behaviour in 13 tests (`tests/unit/test_old_l4_characterization.py`) but did **not** migrate or delete it — see §13 |
| Deterministic implementation | Yes: `resolver_v1.py`, `boundary.py`, `typing.py`, `overlap.py` (interval/graph selection, abstention, coordinate-identity merge) |
| Pretrained adapter | None |
| Trained checkpoint | **None** |
| Future checkpoint slot | `resolution/learned_l4_v2` (`resolution/learned_v2.py`, 749 lines; trainer `training/phase2/l4_training.py`; notebook `MedNorm_L4_Learned_Resolver_v2_Training.ipynb`) |
| Enabled | **Both off.** `enable_l4_deterministic_v1: false`, `enable_l4_learned_v2: false` |
| Contract version | `boundary-type-resolver-v1`, `learned-l4-v2` |
| Configuration source | `configs/resolution/resolver_v1.yaml`, `configs/resolution/boundary_type_resolver_v1.yaml`, `configs/resolution/learned_l4_v2.yaml` |
| Parameter-ledger behaviour | v1 contributes 0; v2 is a small MLP that **is** counted when loaded |
| Known missing | No boundary-offset head (spec §7.1). No learned v2 checkpoint. `resolution/resolver.py` still exists as a second, non-canonical implementation pending test migration |

Note: the former `src/mednorm_vi/boundary_type/` bootstrap stub was a second, dead
L4 import path raising `NotImplementedError`; Audit 0051 deleted it. `resolution/`
is the one L4 module.

## 5. L5 — Specialists

### 5.1 Assertion Hydra (spec §8)

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/specialists/assertion/` — `cues.py` owns the lexicons and the §8.1 scope model; `hydra.py` is the wired per-hypothesis path and delegates every decision to it |
| Status | `PARTIALLY_IMPLEMENTED_AND_INTEGRATED` — deterministic A1-A3 only |
| Deterministic implementation | Yes, and **corrected in Audit 0052**: directional cues, sentence + clause boundaries, contrast markers (§8.1), word-boundary cue matching, entity-type eligibility, section priors (§4.2), multi-label output, empty-on-uncertainty |
| Trained checkpoint | **None** |
| Future checkpoint slot | S2 assertion head (`training/phase2/s2_assertion_training.py`, 754 lines; `MedNorm_S2_Assertion.ipynb`) |
| Enabled | Deterministic path is wired into `inference/pipeline.py` |
| Contract version | `assertion-cue-scope-v1` |
| Parameter-ledger behaviour | Deterministic path 0; the S2 head shares the ViHealthBERT backbone and is counted as base + adapter |
| Known missing | **The governed corpus has zero assertion supervision** (Audit 0042), so no assertion metric can be computed at all — the Audit-0052 fix is verified by regression tests against constructed cases, not by a gold score. Learned A1/A3, A4 (entity–cue relation) and A6 (per-label calibration) are unimplemented. |
| Fixed in Audit 0052 | A measured false-positive defect: the old ±80-character symmetric window assigned all three labels to **10 entities** in one fixture. Three causes — no direction/boundary, no contrast handling, and bare-substring matching ("ông" inside "không"). 32 regression tests now hold the fix |

### 5.2 ICD-10 Super Linker (spec §9)

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/linking/icd10.py`, `linking/snapshot.py`; KB `src/mednorm_vi/kb/` (32 files, 2,230 lines) |
| Status | `IMPLEMENTED_AND_WIRED` (deterministic) — lexical retrieval **plus hierarchy traversal and the §9.3 specificity controller**, since Audit 0054 |
| Deterministic implementation | Yes: alias/lexical retrieval, snapshot membership enforced twice, cross-ontology linking refused |
| Pretrained adapter | None local |
| Trained checkpoint | **None** |
| Future checkpoint slot | S3 dense (`retrieval/icd_dense`), S4 reranker (`reranker/cross_encoder`) |
| Enabled | Yes — `indices/icd10_vi/tt06-2026/index.json` (generated, git-ignored) |
| Contract version | `snapshot-linking-v1` |
| Configuration source | `configs/pipeline/full_v1.yaml` (`icd_index`), `data/manifests/icd10-vi-tt06-2026-official.yaml` |
| Parameter-ledger behaviour | 0 today; BGE-M3 / Qwen3-Embedding / reranker counted when added |
| Hierarchy | `linking/icd10_hierarchy.py`. The graph is **untyped and symmetric** (`dict[str, list[str]]`, 0 asymmetric edges), so **direction is reconstructed from code length**; `metadata.specificity` equals `len(code) - 3` for every record and is therefore reported but **excluded from ranking**. The hierarchy is incomplete by design of the source: 410 codes have no parent record, 454 three-char codes have no children, 774 records are absent from the graph — each is reported, never raised |
| Specificity rule | A descendant outranks its ancestor **only when the mention's text expresses the detail the descendant adds** (token difference between the two canonical names). Otherwise it is suppressed as `DROP_UNSUPPORTED_SPECIFICITY` with the missing tokens recorded, and the ancestor is the conservative fallback |
| Measured, Audit 0054 (200 rows) | 3,328 unsupported-specificity suppressions, 11,609 no-lexical-support suppressions, 478 supported-specific retentions, 193 broader fallbacks, 290 sibling-competition retentions, 10 exact-name hits. Candidate-set size spans **21 distinct values (0-20)** |
| Known missing | No BM25, no dense embedder, no cross-encoder. **Candidate accuracy is still unmeasured and unmeasurable** — zero ICD codes in the governed corpus. The extracted `canonical_name` of some records is visibly truncated (`'- Bao gồm: viêm'`), which is a KB-intake data-quality gap, not a linker gap |

### 5.3 RxNorm Super Linker (spec §10)

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/linking/rxnorm.py` (TTY priority), `kb/rxnorm/` |
| Status | `IMPLEMENTED_AND_WIRED` (deterministic) — **structured linking + TTY-role graph traversal + hard negatives**, since Audit 0054 |
| Trained checkpoint | **None** |
| Future checkpoint slot | Same S3/S4 slots as ICD |
| Enabled | Yes — `indices/rxnorm/prescribable-2026-07-06/index.json`; a Full snapshot index also exists and is selectable by config |
| Configuration source | `configs/linking/rxnorm_snapshots_v1.yaml`; active selection `prescribable` |
| Structured representation | `linking/structured_medication.py` — spec §10.1's `D = {ingredient, salt, strength, concentration, dose_form, release, brand, route, frequency, prn, duration}`, built **only** from E1 ComponentSpans that Audit 0052 preserved through L4. Missing evidence is distinguished from negative evidence: an unstated field lands in `unresolved_fields` and cannot conflict |
| Graph traversal | `linking/rxnorm_graph.py` — IN → SCDC → SCD → SBD. **The generated index stores an unlabeled symmetric adjacency**: the builder saw 2,563,978 relations and kept none of their names, so direction is inferred from endpoint **TTY**, and every emitted candidate records the TTY path it was reached by. Rebuilding the index with labeled edges would let this tighten from "a neighbour with the right TTY" to "a neighbour across the right relation" |
| Hard negatives | Conflicting strength (incl. cross-unit mass conversion), unit **family** (mass vs volume is a category error, not a near miss), concentration (mg/mL), dose form, release form and route each suppress the candidate with a distinct reason code. Suppressed and obsoleted concepts are never offered; grouper TTYs are evidence, never codes |
| INN bridge | **A governed-data gap found by measurement:** `paracetamol` occurs in **zero** records of the locked snapshot — RxNorm is a US vocabulary and names it `Acetaminophen`. A 12-entry INN→USAN bridge widens *retrieval* only, is marked `REQUIRES_CLINICAL_REVIEW`, is recorded on every affected decision, and is listed in full in Audit 0054 §3 for clinical review. The real fix is a governed crosswalk at KB-intake time |
| Measured, Audit 0054 (medication fixture) | 59 structured matches, 74 strength conflicts, 9 dose-form conflicts, 9 concentration conflicts, 146 grouper suppressions, 5 suppressed-concept drops. Candidate-set size per mention `[14, 1, 1, 1, 2, 2]` where the old lexical linker returned 20/1/20/20/1/20 |
| Known missing | No dense retrieval, no reranker. **Candidate accuracy is unmeasured and unmeasurable** — zero RxCUIs in the governed corpus |

Note: `specialists/icd/` and `specialists/rxnorm/` were duplicate bootstrap
packages whose `link()` returned `[]`; Audit 0051 deleted them. `linking/` — the
path spec §19 names — is the one L5 linking module.

## 6. L6 — Clinical Evidence Graph

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/evidence_graph/` — `graph.py` (v2) + **`consistency.py`** (Audit 0054) |
| Status | `IMPLEMENTED_AND_WIRED` — all supportable edge types, plus a typed consistency contract that L7 and L8 consume |
| Deterministic implementation | 9 relations: `supports`, `has_assertion`, `has_candidate`, **`has_result`** (from E2 pair groups preserved through L4), **`modified_by`** (E1 components only), **`in_section`** (L1 section nodes), **`overlaps`** (verified character intervals), **`same_surface`** (normalized repeats, separate coordinates preserved), **`treats`**. Every edge carries `evidence_source`, `confidence_tier`, `provenance` and a rule version; edges are de-duplicated by (source, target, relation) |
| `treats` is hard to create, on purpose | It requires an explicit Vietnamese trigger **between** the two mentions inside one sentence-scale window (120 chars, broken by any sentence terminator), or a governed KB relation. **Co-occurrence produces no edge at all**, and the declined pair is recorded so the consistency layer can report a refused claim |
| Trained checkpoint | None; none planned |
| Enabled | Yes |
| Parameter-ledger behaviour | 0 |
| Consistency contract | `GraphConsistencyReport` / `ConsistencyDecision` / `ConsistencyIssue` / `SupportSignal`, contract `graph-consistency-v1`. **12 rules** (C01 section compatibility … C12 unsafe `treats`), each returning one of **four verdicts** — `SUPPORTED`, `CONTRADICTED`, `UNRESOLVED`, `NOT_APPLICABLE`. **Uncertainty is never encoded as false**: folding `UNRESOLVED` into `CONTRADICTED` invents contradictions and folding it into `SUPPORTED` hides them |
| Fatal vs advisory | An issue is fatal only when the evidence *contradicts*. A fatal issue naming **candidate codes** condemns those codes; a fatal issue naming **no codes** condemns the entity. Conflating the two is a defect Audit 0054 introduced and its own tests caught — a duplicate RxCUI must not delete a correctly-found medication |
| Measured, Audit 0054 (200 rows, specialist arm) | edges `supports` 333, `has_assertion` 333, `has_candidate` 2,099, `in_section` 333, `same_surface` 15, `has_result` 1, `modified_by` 7. Consistency: 1,193 SUPPORTED / 81 UNRESOLVED / 1 CONTRADICTED / 1,005 NOT_APPLICABLE. Issues by rule: C02 3, C04 12, C05 28, C09 4, C10 35. **`overlaps` and `treats` are 0 on this corpus** — L4 leaves no surviving same-type overlap, and narrative DIAGNOSIS/SYMPTOM text contains no medications to relate |
| Known missing | **No global optimization**: no beam search with global features, no ILP. The graph is now read by L7 and L8, but it is not *searched*. `treats` has never fired on real governed data, so its trigger list is untested outside unit fixtures |

## 7. L7 — Confidence Cascade

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/confidence_cascade/` — `cascade.py` + **`escalation.py`** (Audit 0054); local-only loaders and constrained schemas `src/mednorm_vi/llm/` |
| Status | `PARTIAL` — deterministic thresholds **plus a complete locked-option escalation contract**; **no model stage runs, and none can be loaded from here** |
| Deterministic implementation | `apply_confidence_cascade` accepts/rejects on resolver score, **and** `escalation.py` provides `CascadeTier`, `EvidenceBundle`, `LockedBoundaryOption` / `LockedTypeOption` / `LockedAssertionOption` / `LockedCandidateOption` / `LockedOptionSet`, `EscalationRequest` / `EscalationDecision` / `EscalationValidationResult`, and the four dispositions `ACCEPT` / `REJECT` / `UNRESOLVED` / `ESCALATE` |
| Spec P7 lives in the plumbing, not a prompt | An `EscalationDecision` names its choices by **option id**, never by value, so it *cannot* express a code, label or coordinate that was not offered. `validate_escalation_decision` re-checks every id and refuses anything else; a refused decision degrades to the deterministic fallback rather than propagating |
| Entry conditions | Ten, explicit and deterministic: L4 confidence, expert disagreement, graph contradiction, graph unresolved, assertion uncertainty, candidate ambiguity, boundary competition, wrong-type risk, missing structured evidence, repeated-mention conflict. **Both the fired and the not-fired conditions are recorded**, so "why did this *not* escalate?" is answerable |
| No-model fallback | Explicit: `deterministic_fallback` keeps what L4 chose and returns `UNRESOLVED` with reason `no_decision_source` when a condition fired, `ACCEPT` when none did. It never guesses and never silently accepts |
| Measured, Audit 0054 (200 rows, specialist arm) | 239 ACCEPT, 94 UNRESOLVED, 0 REJECT. Conditions fired: assertion uncertainty 35, candidate ambiguity 33, graph unresolved 66, repeated-mention conflict 8, graph contradiction 1. An earlier threshold choice escalated 6/6 subjects on the medication fixture; `boundary_competition` was tightened from `> 0` to `> 1` retained alternatives and `graph_unresolved` narrowed to issues whose own recommendation is ESCALATE — both recorded rather than tuned silently |
| Thresholds | Fixed and written down (`LOW_CONFIDENCE_BELOW = 0.35`, `AMBIGUOUS_CANDIDATE_COUNT = 8`). **Not searched** — this milestone forbids broad threshold optimization, and an unsearched threshold that is documented is more honest than a tuned one that is not |
| Pretrained adapter | `llm/backends.CausalLMBackend`, `local_files_only=True` always, fails closed via `LocalCheckpointMissing` |
| Trained checkpoint | **None** |
| Future checkpoint slot | S5 Qwen LoRA critic/adjudicator (`qwen_lora`) |
| Enabled | Deterministic path only. `enable_e7_qwen_proposer: false` |
| Contract version | `local-backend-v1`, `structured-output-v1` |
| Parameter-ledger behaviour | 0 today. A shared backbone is counted **once**; a LoRA counts base **plus** adapter; peak concurrent residency is reported separately (spec §21) |
| Known missing | No critic model, no adjudicator model, no local Qwen weights. The contract, the entry conditions, the locked option sets and the refusal path exist and are tested; **the models do not**. A future backend implements one `EscalationDecisionSource` method and needs no change to the canonical runner. `ESCALATE` is therefore never returned today — the fallback answers `UNRESOLVED` instead, which is the truthful state |

## 8. L8 — Metric-aware Set Decoder

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/metric_decoder/` — rewritten in Audit 0053 |
| Status | `PARTIAL` — deterministic evidence-ranked selection. **Still not metric-aware, and no longer claims to be** |
| Deterministic implementation | `decode_entities(accepted, cascade, assertions, links)`, contract `l8-deterministic-evidence-ranked-v1`: candidates are grouped into evidence tiers (`exact_alias` > `strong_lexical` > `weak_lexical`); an exact alias alone wins alone; otherwise the best tier is kept down to a relative score floor of 0.60 of its top score. `CANDIDATE_SAFETY_BOUND = 25` is a **bound**, not the rule — a per-candidate `CandidateDecision` records the reason (`KEEP_EXACT`, `KEEP_WITHIN_TIER`, `DROP_BELOW_BAND`, `DROP_WEAKER_TIER`, `DROP_SAFETY_BOUND`, `DROP_DUPLICATE`, `DROP_TYPE_MISMATCH`) |
| Calibrated path | `decode_expected_jaccard_calibrated` exists and **raises `CalibratedDecoderUnavailable`**. It fails loudly rather than returning a plausible-looking set, because a silent stand-in for §13 is what the old name did |
| Trained checkpoint | **None** |
| Future checkpoint slot | S6 calibration meta-model (`calibration/full_v1`) |
| Enabled | Yes; `decoder_status()` reports the active contract |
| Consumes L6 and L7 (Audit 0054) | A **fatal** contradiction withholds the entity; codes a fatal issue names are dropped as `dropped_l6_fatal_contradiction`; an L7 `REJECT` withholds the entity; every consistency verdict appears in `retention_reason`. Omitting both arguments reproduces the Audit-0053 behaviour exactly, so the wiring is additive rather than a silent policy change |
| Measured, Audit 0054 (200 rows) | Candidate-set size spans **15 distinct values** (0-18). Per-candidate reasons: 617 kept-within-band, 10 kept-exact, 812 dropped-below-band, 660 dropped-weaker-tier |
| Parameter-ledger behaviour | Small meta-model counted when it exists |
| Fixed in Audit 0053 | `decode_expected_jaccard` was the most misleading name in the repository: it applied a **fixed top-10**, exactly the "fixed top-K" spec §13.2 rules out, under a name promising expected-Jaccard search. The name is gone; a test asserts it cannot be imported |
| Known missing | Spec §13 proper: expected-Jaccard candidate-set search, expected-WER boundary utility, per-label assertion thresholds, entity-retention utility subtracting wrong-type risk. All of it needs calibrated probabilities, which **no stage produces** — so S6 remains the true blocker. `DecodedEntity.as_dict()` asserts `performs_expected_jaccard_decoding: False` |

## 9. L9 — Validator & Packager

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/validator/` (10 files, incl. `kb_membership.py` added in Audit 0053); `organizer_policy/` (3 files, 257 lines); packaging `inference/packaging.py` |
| Status | `IMPLEMENTED_AND_WIRED` — never run to a real submission |
| Deterministic implementation | Yes, fail-fast: exact-substring/offset checks, organizer label + per-type field policy, duplicate detection, candidate syntax, `1.json`..`100.json` set, UTF-8, deterministic ordering, `output.zip` containing exactly one top-level `output/` |
| Pretrained adapter | None |
| Trained checkpoint | None |
| Enabled | Always |
| Contract version | Confirmed organizer layout (Audit 0002); `schemas/constants.py` is authoritative for labels, per-type fields and end-exclusive positions |
| Parameter-ledger behaviour | 0 |
| KB membership | **Enforced in Audit 0053** by `validator/kb_membership.py` (`l9-kb-membership-v1`). Deliberately **model- and retriever-independent**: it imports no linker, no index builder and no model — a test asserts this, because the linker is the component whose bug this gate exists to catch. Checks: candidate present in the snapshot its type requires; correct ontology per type (`icd10_vi` / `rxnorm`, wrong-snapshot detected); no candidates on a type that takes none (spec §7.3); duplicates; missing snapshot treated as **cannot validate**, never as pass; and `offered_codes`, so a code that exists but was not retrieved for that mention fails under spec P7 |
| **The gate is on the canonical path** | **YES.** `inference/pipeline._gate_kb_membership` runs inside `run_input_dir` on the **serialized** organizer JSON, **before** anything is written, and raises `KbMembershipViolation`. A violating run therefore leaves **no `output/` and no `output.zip`**. As first written in Audit 0053 this validator had **no caller under `inference/`** — it was wired in the same audit's correction pass, with an integration test that injects an out-of-snapshot code at the linker and asserts packaging stops |
| Known missing | It **never repairs** — the caller stops the run, and the run-manifest wiring that *records* the stop is not built (Audit 0053 item 10, not attempted). Consequently the wired gate enforces membership and ontology but **not** P7's `offered_codes`: nothing records which codes were offered per mention. Candidate *ordering* determinism is asserted by running the pipeline twice in tests, not by this validator. No packaging-stress suite per spec §18.3. `output.zip` has never been produced for a real submission |

Audit 0051 deleted a second, divergent packager/validator (`zs0/submission.py`)
that required the 100 files at the **ZIP root** and expected a `{"entities": …}`
object with `start_pos`/`end_pos` — contradicting the confirmed layout above on
three points. `validator/` is the one L9 module.

---

## 9a. The canonical L3 -> L4 contract lineage (Audit 0052)

One versioned lineage, one owner per type. Four types were named ``SpanProposal``;
they are not interchangeable, so each now declares whether it is runtime.

```text
L1   document_intelligence.models.DocumentGraph            RUNTIME  (8 fields)
     schemas.document.DocumentGraph                        non-runtime sketch

L2   case_router.models.NodeRouting                        RUNTIME
     schemas.routing.RouteDecision                         non-runtime sketch

L3   E1/E2  -> mention_factory.models.SpanProposal         RUNTIME  (22 fields,
                                                             + ComponentSpan,
                                                             + RelationProposal)
     E3..E7 -> lattice.models.ExpertSpanProposal            RUNTIME  (20 fields,
                                                             + model_revision,
                                                             + checkpoint_sha256)
     schemas.hypotheses.SpanProposal                        non-runtime sketch
                    |
                    v  lattice.builder.build_span_lattice
     merged -> lattice.models.SpanProposal                  RUNTIME  (10 fields,
                                                             sources[] keeps every
                                                             contributing expert)
                    |
                    v  resolution.canonical.resolve_lattice_to_hypotheses
L4   resolution.models.EntityHypothesis                     RUNTIME  (+ expert_ids,
                                                             + components)
                    |
                    v
L5-L9  assertions / linking / graph / cascade / decoder / validator
```

**Invariants the lineage enforces**, each checked in code rather than documented:

| Invariant | Where |
| --- | --- |
| `original_text[start:end] == text` | expert (`propose`), registry, lattice `register`, L4, L9 — four independent checks |
| end-exclusive absolute coordinates preserved | `ExpertSpanProposal.__post_init__` rejects a mismatched `original_start`/`original_end` |
| full provenance preserved | `SourceEvidence` per contributing expert; never averaged into the node |
| E1 `ComponentSpan` survives to L5 | `SourceEvidence.components` -> `SpanProposal.components()` -> `EntityHypothesis.components` |
| E2 `RelationProposal` pairing survives to L5 | `SourceEvidence.relation_refs` -> `EntityHypothesis.has_result_pair_group_ids` |
| model revision + checkpoint SHA recorded | `ExpertSpanProposal`, and `ExpertRunRecord` per document |
| no expert emits `EntityHypothesis` or organizer JSON | the `L3Expert` protocol has no such method; `run_registered_experts` accepts only `ExpertSpanProposal` |
| a new expert needs no runner edit | `mention_factory/registry.py`; a test asserts the runner names no expert |

Before Audit 0052 the third and fourth rows were false: the lattice reduced E1's
components to a single float and dropped the relation pairing entirely, one layer
above the linker that spec §10.1/§6.2 needs them for.

## 10. Cross-cutting components

| Component | Path | Status | Notes |
| --- | --- | --- | --- |
| Local evaluator | `evaluation/` (24 files, 4,113 lines) | `IMPLEMENTED_AND_WIRED` | Exact character-offset mention evaluator (`exact_mention.py`), WER, Jaccard, matching strategies, HTML/JSON reporting. **Byte-pinned** in `tests/unit/test_e5_s2_readiness_and_budget.py` |
| Parameter budget / 9B gate | `governance/` (3 files, 800 lines) | `IMPLEMENTED_AND_WIRED` | Candidate vs deployment inventories kept apart; fails **closed** on any unverified count; shared backbone counted once; peak concurrency reported |
| Governed corpus + split identity | `data_engine/`, `training/governed_splits.py` | `IMPLEMENTED_AND_WIRED` | Splits resolved by SHA-256, never by name; `internal_test` refused by name |
| KB intake | `kb/`, `data/manifests/*.yaml` | `IMPLEMENTED_AND_WIRED` | ICD-10 TT06-2026 derived deterministically from the protected source PDFs; RxNorm Full + Prescribable 2026-07-06 |
| Experiment registry | `experiments/` | `IMPLEMENTED_NOT_WIRED` | `EXP-*.json` records tracked; no leaderboard decision has used it |
| Reproducibility | `Dockerfile`, `requirements.lock`, `requirements-image.lock` | `AUDITED_AND_HARDENED__BUILD_BLOCKED` | Audit 0054 **attempted the build** and found two real defects. (1) `requirements.lock` pins `torch==2.13.0+cu126`, a local version identifier absent from PyPI, so `pip install -r requirements.lock` cannot resolve it — `requirements-image.lock` now installs the **same upstream version** from the CPU index, with both indexes named explicitly and the deviation recorded. (2) `package_output_zip` embedded file mtimes, so two identical runs produced different ZIP digests; entries now use a fixed timestamp and mode, and the archive is byte-reproducible. **The build itself is blocked by the environment, not the Dockerfile**: `~/.docker/config.json` holds stale Docker Hub tokens that the daemon sends on every pull, so even an anonymous pull of the public base image returns `401 Unauthorized`. Nothing was faked and the Dockerfile was not weakened to work around it |
| Inference run manifest | `inference/manifest.py` | `IMPLEMENTED_AND_WIRED` (opt-in) | Audit 0054. `--run-manifest PATH`; nothing is written without it. Records git commit + dirty state, the PDF hash, every contract version, config hashes, readiness and fail-closed state, checkpoint roles/SHAs/revisions, snapshot ids, and aggregate counts for route gating, proposals, lattice, L4, assertions, linking, L6 edges, consistency, L7 dispositions, decoder candidate sizes and L9 issues. **No clinical text** (asserted by test), home-directory paths reduced to `role:` identifiers, byte-identical apart from `runtime_seconds` and `peak_memory_gib`. Written **even when L9 stops the run**, with `l9_stopped_the_run: true` |
| Code-linking evaluation contract | `evaluation/code_linking.py` | `CONTRACT_ONLY__NO_GOLD_EXISTS` | Audit 0054. Strict record schemas, Recall@1/@5/@10, candidate Jaccard, exact-set match, and a 9-way error taxonomy separating broader-than-gold from wrong. It **refuses to score** without gold (`GoldSetUnavailable`) rather than returning zeros, and refuses `PENDING`/`DISPUTED` annotations. **No gold data ships**, no function turns predictions into labels, and `REQUIRED_ANNOTATION_ARTIFACT` specifies the human round that must happen first |

## 11. Training stages (spec §15)

| Stage | Status | Checkpoint | Blocker |
| --- | --- | --- | --- |
| S0 domain adaptation | `SCAFFOLD_ONLY` | none | Notebook is a `DESIGN_DRAFT` with placeholders |
| **S1 mention** | **`EXECUTED`** | **exists, validated** | Complete; see L3/E3 |
| S2 assertion | `IMPLEMENTED_NOT_RUN` | none | **No assertion supervision exists** in the governed corpus |
| S3 retrieval | `SCAFFOLD_ONLY` | none | No embedder weights; no index beyond lexical |
| S4 reranking | `SCAFFOLD_ONLY` | none | Depends on S3 |
| S5 Qwen LoRA | `NOT_STARTED` | none | No local Qwen weights |
| S6 calibration | `NOT_STARTED` | none | Needs out-of-fold predictions, which no stage produces yet |

## 12. Parameter budget, as actually configured

```text
cap (spec §17)                      9,000,000,000
active planned profile              Safe-8.85B-minus-E4
summed in_profile base params       8,482,000,000        = 8,852,000,000 − 370,000,000
soft target                         8,900,000,000        (advisory, satisfied)
margin to the cap                     518,000,000
```

The 370M is PhoBERT-large's **planning estimate**, withdrawn with retired E4. Its
row is kept in `configs/parameter_budget.yaml` with `in_profile: false` so the
spec's own stack table stays visible and only the *summed active plan* changed.

**Two different numbers, kept apart on purpose:**

| Number | Provenance | Where it lives |
| --- | --- | --- |
| 370,000,000 | spec §17 approximate **planning estimate** for PhoBERT-large | `configs/parameter_budget.yaml`, labelled, `in_profile: false`, not summed |
| **371,289,161** | **measured**, `PROGRAMMATICALLY_VERIFIED` in Audit 0044 (369,163,264 encoder + 2,125,897 head) | `governance/e4_retirement.py` |

Citing the planning row as E4's historical size would be wrong. The measured figure
is the historical size; it contributes to no ledger because E4 is retired.

**Every neural count in the active plan is a published estimate, not a verified
one.** The deployment gate in `governance/parameter_budget.py` fails closed on any
unverified component, so no deployment can be certified until each model is counted
programmatically from its real configuration or checkpoint.

## 13. Ranked remaining architecture work

**Updated by Audit 0054.** Of Milestone 2B's seven carried-forward tasks plus container
verification, **six are complete** (structured RxNorm linking, ICD hierarchy +
specificity, L6 edges, the typed consistency contract consumed by L7/L8, the L7
locked-option contract, the run manifest) plus the code-linking evaluation contract.
**Two are not**, and they head the list.

1. **Migrate and delete `resolution/resolver.py`** — Audit 0054 completed step 1 of the
   five §10 asked for: its unique behaviour is characterized in 13 tests
   (`tests/unit/test_old_l4_characterization.py`). What is unique is a **configurable
   per-type boundary policy** (`medication_boundary` full/name_only/name_strength,
   `test_result_boundary` value_only/value_unit, `abstain_on_conflict`) that the
   canonical L4 has no equivalent of. Deleting the module also means migrating
   `phase1c_foundation/cli.py`, `phase1c_foundation/doctor.py`, the
   `resolution/__init__.py` re-export and `tests/unit/test_resolution.py`. Two L4
   implementations must not coexist indefinitely.
2. **Build and offline-smoke the Docker image** — blocked in Audit 0054 by stale Docker
   Hub credentials in `~/.docker/config.json`, which make even an anonymous pull of the
   public base image fail with `401`. The owner clears it with `docker logout` (or by
   removing the `auths` entries) and then runs the build; the Dockerfile and both lock
   files are already corrected and the static contract is tested. Required for the
   private-test rebuild.
3. **A code-bearing evaluation set** — still the single most consequential blocker in the
   repository, and a *data* blocker. Candidates are 40% of the organizer metric and it
   is **unmeasurable**: zero ICD-10 codes and zero RxCUIs in the governed corpus. Audit
   0054 built the strict contract and the refusal path; what is missing is the human
   annotation round specified in `evaluation.code_linking.REQUIRED_ANNOTATION_ARTIFACT`.
   Everything below item 4 is being built blind until this exists.
4. **Calibration (S6)** — nothing downstream of L4 has calibrated probabilities, so the
   real spec §13 decoder has nothing to optimize over and L8 stays a deterministic
   stand-in. S6 needs out-of-fold predictions no stage produces.
5. **A governed INN ↔ RxNorm crosswalk at KB-intake time** — `paracetamol` occurs in
   **zero** records of the locked snapshot. Audit 0054's 12-entry bridge is a stopgap
   marked `REQUIRES_CLINICAL_REVIEW`; a large hand-written drug-name table is a
   patient-safety hazard and must not be the answer.
6. **Rebuild the KB indexes with labeled graph edges** — both graphs store unlabeled
   symmetric adjacency, so ICD direction is inferred from code length and RxNorm
   direction from endpoint TTY. Both work and both are recorded as inferences; labeled
   edges would make them assertions.
7. **L6 global optimization** — the graph is now read by L7 and L8 but not *searched*.
   No beam search with global features, no ILP (spec §11).
8. **L4 boundary-offset head** (spec §7.1) and a re-run ablation: deterministic L4 v1
   still measures below the E3-only baseline (0.5521 against 0.5559).
9. **Assertion supervision** — build an assertion corpus; until then no assertion metric
   is computable and S2 cannot be validated.
10. **L7 model stages** — the contract, entry conditions and refusal path are done, so a
    critic/adjudicator backend is now a drop-in. Last, per spec §20's ordering.

## 14. Maintenance rule

Update this file in the same change that alters a layer's status, and record the
change in the numbered audit. If this file and the code disagree, the code is
right and this file is a defect.
