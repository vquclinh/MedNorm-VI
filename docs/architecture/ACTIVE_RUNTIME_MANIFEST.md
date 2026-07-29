# MedNorm-VI — Active Runtime Architecture Manifest

**Supplemental to `docs/MedNorm-VI_Architecture.pdf`, never a replacement for it.**
The PDF is the primary source of truth and is unmodified
(SHA-256 `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`). It
describes a candidate *super*-architecture. This file records what the repository
**actually** implements today, so a reader can tell a designed component from a
built one without running the code.

- **Established by:** Audit 0051 (repository-wide architecture review and cleanup)
- **Last updated by:** Audit 0053 (route gating + honest L8 + L9 KB membership +
  reproducibility infrastructure). Audit 0053 is a **partial** milestone: of the
  **thirteen implementation tasks** in the Milestone 2 prompt (its sections 2-14;
  sections 1 and 15 were verification and the audit itself), **seven were not
  attempted**, and §13 below marks them.
- **Date:** 2026-07-29
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
| Layers implemented and wired | L1, L2, L3 (E1+E2+E3), L4 (canonical over the lattice), L5 assertion (deterministic, §8.1 scope), L5 linking (lexical), L6, L7 (deterministic fallback), L8 (deterministic, honestly named), L9 (incl. KB membership) |
| Layers that are contract-or-scaffold only | L7 LLM cascade (no local weights), L8 **calibrated** expected-Jaccard/WER decoding (fails loudly), L6 global reasoning |
| Reproducibility infrastructure | **Authored in Audit 0053** — `Dockerfile` + `requirements.lock`. **The image has never been built.** |
| `output.zip` ever produced | **No** |
| Organizer metric ever computed | **No** — only exact mention span/type |
| Organizer inference ever run | **No** |
| Candidate Recall@K ever measured | **No** — the governed corpus carries **zero** ontology codes (see §0 measurement note) |

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
| Canonical entry points | **1.** Audit 0052 replaced the two-live-L4 situation: `resolution/resolver.py` (Phase-1C-A) is no longer on the canonical path and is retained only until its remaining per-type boundary policy is provably covered |
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
| Status | `PARTIAL` — lexical retrieval over a locked snapshot |
| Deterministic implementation | Yes: alias/lexical retrieval, snapshot membership enforced twice, cross-ontology linking refused |
| Pretrained adapter | None local |
| Trained checkpoint | **None** |
| Future checkpoint slot | S3 dense (`retrieval/icd_dense`), S4 reranker (`reranker/cross_encoder`) |
| Enabled | Yes — `indices/icd10_vi/tt06-2026/index.json` (generated, git-ignored) |
| Contract version | `snapshot-linking-v1` |
| Configuration source | `configs/pipeline/full_v1.yaml` (`icd_index`), `data/manifests/icd10-vi-tt06-2026-official.yaml` |
| Parameter-ledger behaviour | 0 today; BGE-M3 / Qwen3-Embedding / reranker counted when added |
| Known missing | No BM25, no character n-gram index, no dense embedder, no cross-encoder, **no hierarchy-graph specificity controller** (spec §9.3). Recall@K has never been measured |

### 5.3 RxNorm Super Linker (spec §10)

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/linking/rxnorm.py` (TTY priority), `kb/rxnorm/` |
| Status | `PARTIAL` — lexical retrieval + TTY preference |
| Trained checkpoint | **None** |
| Future checkpoint slot | Same S3/S4 slots as ICD |
| Enabled | Yes — `indices/rxnorm/prescribable-2026-07-06/index.json`; a Full snapshot index also exists and is selectable by config |
| Configuration source | `configs/linking/rxnorm_snapshots_v1.yaml`; active selection `prescribable` |
| Known missing | **No structured drug parse feeding the linker** (spec §10.1 `D = {ingredient, strength, dose_form, …}`), no RxNorm relation graph traversal (§10.2), no strength/dose-form hard negatives. These are the highest-value ICD/RxNorm gaps, since candidates carry 40% of the metric |

Note: `specialists/icd/` and `specialists/rxnorm/` were duplicate bootstrap
packages whose `link()` returned `[]`; Audit 0051 deleted them. `linking/` — the
path spec §19 names — is the one L5 linking module.

## 6. L6 — Clinical Evidence Graph

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/evidence_graph/` (2 files, 100 lines) |
| Status | `PARTIAL` — node/edge construction and a deterministic graph hash |
| Deterministic implementation | Builds proposal → hypothesis → assertion → candidate nodes and `supports` / `has_assertion` / `has_candidate` edges; wired into `inference/pipeline.py` |
| Trained checkpoint | None; none planned |
| Enabled | Yes |
| Parameter-ledger behaviour | 0 |
| Measured, Audit 0053 (200 rows) | Exactly three relations occur: `has_assertion` 333, `has_candidate` 4,000, `supports` 333. There is **no consistency-check module and no typed public contract**, so "consistency violations" is not a measurable quantity today and Audit 0053 reports it as unmeasurable rather than as zero |
| Known missing | Most of spec §11. Absent edges: `has_result`, `modified_by`, `in_section`, `treats`, `overlaps`, `same_surface`. **No global reasoning at all** — no beam search with global features, no ILP, no consistency constraints. The graph currently records evidence rather than using it, and nothing downstream reads it: L7 and L8 both take L4 output directly (Audit 0053 items 5-6, not attempted) |

## 7. L7 — Confidence Cascade

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/confidence_cascade/` (2 files, 59 lines); local-only loaders and constrained schemas `src/mednorm_vi/llm/` (3 files, 422 lines) |
| Status | `PARTIAL` — deterministic threshold fallback implemented; **no LLM stage runs** |
| Deterministic implementation | Yes: `apply_confidence_cascade` accepts/rejects on resolver score |
| Pretrained adapter | `llm/backends.CausalLMBackend`, `local_files_only=True` always, fails closed via `LocalCheckpointMissing` |
| Trained checkpoint | **None** |
| Future checkpoint slot | S5 Qwen LoRA critic/adjudicator (`qwen_lora`) |
| Enabled | Deterministic path only. `enable_e7_qwen_proposer: false` |
| Contract version | `local-backend-v1`, `structured-output-v1` |
| Parameter-ledger behaviour | 0 today. A shared backbone is counted **once**; a LoRA counts base **plus** adapter; peak concurrent residency is reported separately (spec §21) |
| Known missing | No critic stage, no adjudicator stage, no entry conditions per spec §12, no local Qwen weights. Constrained-decision *parsing* and the *option-set* constraints exist and are tested; the models do not |

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
| Reproducibility | `Dockerfile`, `requirements.lock` | `AUTHORED_NOT_BUILT` | Added in Audit 0053. `python:3.14.5-slim-bookworm`, JRE only, `HF_HUB_OFFLINE=1`, `PYTHONHASHSEED=0`, non-root uid 10001, one `ENTRYPOINT` (the canonical inference CLI), weights and KB payloads **mounted, never baked in**. `requirements.lock` pins every version from `importlib.metadata` in the environment that produced these measurements. **The image has never been built and no organizer inference has run**; building it is a separate authorized step |

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

**Carried forward from Audit 0053, which did not attempt them.** Audit 0053
completed six of the **thirteen implementation tasks** in its scope — prompt sections
2, 8, 9, 11, 13 and 14 (sections 1 and 15 were verification and the audit itself, so
the denominator is thirteen, not fifteen). The following seven were declared out of
scope for that turn and remain open. Its acceptance criteria are therefore
**not met**.

1. **L5 RxNorm structured linking** (Audit 0053 item 4) — consume
   `EntityHypothesis.components` (the E1 parse that Audit 0052 preserved to L5 and
   the linker still ignores), traverse `ingredient → component → SCD → SBD` over
   the loaded `index["graph"]` (77,055 nodes), add strength/dose-form/release hard
   negatives. Highest expected value: candidates are 40% of the score and this is
   the one gap where the required inputs already exist and are unused.
2. **L5 ICD-10 hierarchy + specificity controller** (item 3, spec §9.3) —
   `LocalIndex.graph` carries 14,534 nodes, is materialized, is loaded, and
   `linking/icd10.py` never reads it.
3. **Calibration (S6)** — nothing downstream of L4 has calibrated probabilities, so
   the real §13 decoder has nothing to optimize over. Audit 0053's L8 is honest
   about being a deterministic stand-in; only S6 removes the stand-in.
4. **L6 edges, consistency checks and a typed downstream contract** (items 5-6,
   spec §11) — six missing relations, eight deterministic consistency checks, and a
   contract L7/L8 actually consume. Until then L6 is write-only.
5. **L7 deterministic escalation contract with locked option sets** (item 7,
   spec §12.1) — the constrained-decision *parsing* exists; the locked
   assertion/candidate/boundary option sets an escalation must choose from do not.
6. **Inference run manifest** (item 10) — deterministic, no clinical-text leakage.
   It closes two loops on L9's wired gate: the gate now stops a violating run but
   nothing **records** the stop, and it cannot enforce spec P7 until something
   records which codes were offered per mention.
7. **Migrate and delete `resolution/resolver.py`** (item 12) — its per-type
   boundary policy must be provably covered by the canonical L4 first, with
   equivalence tests, then the file goes. Two L4 implementations must not coexist.
8. **L4 boundary-offset head** (spec §7.1) and a re-run ablation: deterministic L4
   v1 still measures below the E3-only baseline (0.5521 against 0.5559).
9. **Assertion supervision** — build an assertion corpus; until then no assertion
   metric is computable and S2 cannot be validated.
10. **Build and verify the Docker image** — the file exists; the image does not.
    Required for the private-test rebuild, and a separate authorized step.
11. **L7 LLM cascade** — last, per spec §20's ordering: evaluator, offsets and
    validator before heavy architecture.

## 14. Maintenance rule

Update this file in the same change that alters a layer's status, and record the
change in the numbered audit. If this file and the code disagree, the code is
right and this file is a defect.
