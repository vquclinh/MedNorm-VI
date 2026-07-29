# MedNorm-VI — Active Runtime Architecture Manifest

**Supplemental to `docs/MedNorm-VI_Architecture.pdf`, never a replacement for it.**
The PDF is the primary source of truth and is unmodified
(SHA-256 `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`). It
describes a candidate *super*-architecture. This file records what the repository
**actually** implements today, so a reader can tell a designed component from a
built one without running the code.

- **Established by:** Audit 0051 (repository-wide architecture review and cleanup)
- **Last updated by:** Audit 0052 (canonical L3 lattice + E3 activation + assertion fix)
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
| Layers implemented and wired | L1, L2, L3 (E1+E2+E3), L4 (canonical over the lattice), L5 assertion (deterministic, §8.1 scope), L5 linking (lexical), L6, L7 (deterministic fallback), L8 (deterministic), L9 |
| Layers that are contract-or-scaffold only | L7 LLM cascade (no local weights), L8 expected-Jaccard/WER decoding |
| `output.zip` ever produced | **No** |
| Organizer metric ever computed | **No** — only exact mention span/type |
| Organizer inference ever run | **No** |

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
| Contract version | `signals_v1` |
| Parameter-ledger behaviour | Contributes 0 |
| Known missing | **Routing accuracy has never been measured.** No labelled route data exists, so the router is unvalidated rather than validated-and-good |

## 3. L3 — Mention Factory

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/mention_factory/` (registry + experts + specialists); unified lattice `src/mednorm_vi/lattice/` |
| Status | `PARTIALLY_IMPLEMENTED_AND_INTEGRATED` — three experts run (E1, E2, E3); three await weights or training |
| Expert entry point | `mention_factory/registry.py` — the runner consults the registry and names no expert of its own, so a new expert needs **no** runner edit |
| Contract version | `mention-span-v1` (`spans.py`), `mention-offsets-v1` (`offsets.py`), lattice `EXPERT_FAMILY`/`AVAILABLE_EXPERTS` |
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
| Known missing | Most of spec §11. Absent edges: `has_result`, `modified_by`, `in_section`, `treats`, `overlaps`, `same_surface`. **No global reasoning at all** — no beam search with global features, no ILP, no consistency constraints. The graph currently records evidence rather than using it |

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
| Implementation | `src/mednorm_vi/metric_decoder/` (2 files, 53 lines) |
| Status | `PARTIAL` — deterministic selection only, **not metric-aware** |
| Deterministic implementation | `decode_expected_jaccard` selects cascade-accepted entities and truncates candidates at `max_candidates=10` |
| Trained checkpoint | **None** |
| Future checkpoint slot | S6 calibration meta-model (`calibration/full_v1`) |
| Enabled | Yes |
| Parameter-ledger behaviour | Small meta-model counted when it exists |
| Known missing | **The function name overstates the implementation and this is the most misleading gap in the repository.** Spec §13 requires expected-Jaccard candidate-set search, an expected-WER boundary utility, per-label assertion thresholds and an entity-retention utility that subtracts wrong-type risk. None is implemented: there is no probability calibration, no set search, and the fixed top-10 truncation is exactly the "fixed top-K" spec §13.2 rules out. Renaming/implementing this is next-milestone work |

## 9. L9 — Validator & Packager

| Field | Value |
| --- | --- |
| Implementation | `src/mednorm_vi/validator/` (9 files, 1,370 lines); `organizer_policy/` (3 files, 257 lines); packaging `inference/packaging.py` |
| Status | `IMPLEMENTED_AND_WIRED` — never run to a real submission |
| Deterministic implementation | Yes, fail-fast: exact-substring/offset checks, organizer label + per-type field policy, duplicate detection, candidate syntax, `1.json`..`100.json` set, UTF-8, deterministic ordering, `output.zip` containing exactly one top-level `output/` |
| Pretrained adapter | None |
| Trained checkpoint | None |
| Enabled | Always |
| Contract version | Confirmed organizer layout (Audit 0002); `schemas/constants.py` is authoritative for labels, per-type fields and end-exclusive positions |
| Parameter-ledger behaviour | 0 |
| Known missing | **KB membership of emitted candidate codes is not enforced by the L9 validator** (it is enforced inside `linking/snapshot.py`); the two should meet. No packaging-stress suite per spec §18.3. `output.zip` has never been produced |

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
| Reproducibility | `reproducibility/`, `Dockerfile` | **MISSING** | There is **no Dockerfile and no `requirements.lock`**, both required by spec §19 and Appendix A |

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

1. **L8 metric-aware decoding** — candidates are 40% of the score and the decoder
   currently applies a fixed top-10. Highest expected value per unit of work.
2. **L5 RxNorm structured parse + graph search** and **L5 ICD hierarchy
   specificity controller** — the two linkers are lexical-only.
3. **Calibration (S6)** — nothing downstream of L4 has calibrated probabilities, so
   L8 has nothing to optimize over.
4. **L4 boundary-offset head** and a re-run ablation, since deterministic L4 v1
   currently measures below the E3-only baseline.
5. **Assertion supervision** — build an assertion corpus; until then no assertion
   metric is computable and S2 cannot be validated.
6. **L6 global reasoning** — edges and constraints from spec §11.
7. **Reproducibility (Phase 7)** — Dockerfile, lockfile, offline weights,
   one-command inference. Required for the private-test rebuild.
8. **L7 LLM cascade** — last, per spec §20's ordering: evaluator, offsets and
   validator before heavy architecture.

## 14. Maintenance rule

Update this file in the same change that alters a layer's status, and record the
change in the numbered audit. If this file and the code disagree, the code is
right and this file is a defect.
