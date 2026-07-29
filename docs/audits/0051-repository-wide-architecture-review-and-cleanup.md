# Audit 0051 — Repository-wide Architecture Review and Cleanup

- **Date:** 2026-07-29
- **Author:** Claude (AI agent), for human review
- **Change type:** Repository-wide review, consolidation and deletion. **No commit.
  No push.** No training, no fine-tuning, no benchmarking, no validation inference.
  No model downloaded. No `internal_test` access. No organizer inference. No
  `output.zip`. No baseline created. No Docker image built. No assistant control
  file created or tracked. No audit deleted or edited.
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — read in full (886 extracted
  lines, 14 pages), **unmodified**.
- **Status:** `REPOSITORY_CLEAN_AND_READY_FOR_REAL_ARCHITECTURE_IMPLEMENTATION`

---

## 1. Initial Git state

```text
pwd     /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch  main   (## main...origin/main)
HEAD    e7ca86b fix: repair ZS0 Stage-3 imports, field names and label mapping
        0dc83cc fix: implement real ZS0 Stage-3 arm execution
        75bc73a feat: retire E4 and add the ZS0 zero-shot submission baseline
        cbc756a feat: replace the E4 Stage-2 objectives and fix best-state reproduction
        6dc6b9b fix: repair the E4 tiny-overfit stopping and reproduction contract

tracked files at start   627
tracked files at end     587   (net -40; +12 added, -52 removed)
branches                 main, backup/pre-audit-0013, origin/main
```

The working tree already carried the staged deletions and modifications of the
in-flight ZS0/E4 work when this audit began (`git status` showed 44 staged/modified
paths). Those were part of the same uncommitted milestone; this audit continued from
that state rather than resetting it, and §18 lists the complete final set.

## 2. Architecture PDF hash, before and after

```text
before  0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b
after   0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b
git diff --stat -- docs/MedNorm-VI_Architecture.pdf   (empty)
```

**Byte-identical.** The Vietnamese copy remains untracked and git-ignored. Text was
extracted read-only with `pdftotext -layout` into the session scratchpad; the file
itself was never opened for writing.

## 3. Method

The PDF was read in full first, then the entire repository: all 50 prior audits,
`src/mednorm_vi/` (267 modules), `configs/` (56 files), `tests/` (95 modules),
`notebooks/` (18 on disk), `scripts/`, `schemas/`, `typings/`, the model and
checkpoint registries, the governance and parameter-budget code, the local
evaluator, the validators and packagers, the data/ontology/index code, tracked
generated artifacts, ignored artifact roots, and the Git history of the abandoned
implementations.

Two mechanical passes did the work that reading cannot:

- a **module-level import graph** over every `src/` module plus every test and
  notebook, which produced the orphan set — modules no `src/` module imports — and
  is what identified `boundary_type`, `specialists.icd`, `specialists.rxnorm`,
  `mention_factory.qwen_proposer` and `data_engine.review` as unreachable;
- a **whole-package import sweep** (`pkgutil.walk_packages` + `import_module`) run
  after every deletion, which is how each broken import was found and fixed rather
  than discovered later by a test.

Nothing was inferred from the latest audit alone; §4's classifications come from the
real files.

## 4. Repository implementation matrix

Classification per the task's vocabulary. Layer column is the spec layer the item
serves. "Imported/executed" means reachable from `src/` (not merely from a test).

### 4.1 L1–L9 runtime modules

| Path | Layer | Actual responsibility | Imported/executed | Duplicate of | Tests validate | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| `document_intelligence/` (16f, 2,109L) | L1 | Normalization, offsets, sections, sentences, lines, lists, table rows | yes, wired | — | real behaviour + property tests | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `position/` (6f, 641L) | L1 | Offset encoders, forensics, policy registry | yes | — | real behaviour | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `case_router/` (7f, 743L) | L2 | Multi-label C1–C7 routing, signals, priors | yes, wired | — | real behaviour, **no accuracy data** | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `mention_factory/medication/`, `laboratory/` | L3 | E1 grammar, E2 parser | yes, wired | — | real behaviour + stress | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `mention_factory/neural/` (3f, 601L) | L3 | E3 runtime + decoding | yes | — | real behaviour | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `mention_factory/gliner.py` (245L) | L3 | E6 strict adapter, fails closed | via tests only | — | contract | `REQUIRES_PRETRAINED_CHECKPOINT` → **KEEP** |
| `mention_factory/mrc.py` (410L) | L3 | E5 MRC contract | yes | — | contract | `REQUIRES_FUTURE_TRAINING` → **KEEP** |
| `mention_factory/w2ner.py` (570L) | L3 | **E4 only** | only by E4 + 3 generic symbols | — | contract | `OBSOLETE` → **MIGRATE_THEN_DELETE** |
| `mention_factory/qwen_proposer.py` (105L) | L3 | E7 stub whose `propose()` raises `NoReturn` | **no consumer** | superseded by `llm/backends` | contract only | `OBSOLETE` → **DELETE** |
| `lattice/` (4f, 826L) | L3 | Unified span lattice, coordinate identity | yes, wired | — | real behaviour | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `resolution/` (13f, 2,486L) | L4 | Deterministic v1 + learned v2 | yes, wired (v1 disabled by measurement) | — | real behaviour | `PARTIAL` → **KEEP** |
| `boundary_type/` (1f, 36L) | L4 | Protocol + `resolve()` raising `NotImplementedError` | **no consumer** | `resolution/` | none | `DUPLICATED` → **DELETE** |
| `specialists/assertion/` | L5 | Deterministic cue/scope assertions | yes, wired | — | via pipeline | `PARTIAL` → **KEEP** |
| `specialists/icd/`, `specialists/rxnorm/` (2f, 56L) | L5 | `link()` returning `[]`, "compatibility wrapper for the historical bootstrap protocol" | **no consumer** | `linking/icd10.py`, `linking/rxnorm.py` | none | `DUPLICATED` → **DELETE** |
| `linking/` (4f, 399L) | L5 | Real ICD/RxNorm retrieval over locked snapshots | yes, wired | — | real behaviour | `PARTIAL` → **KEEP** |
| `kb/` (32f, 2,230L) | L5 | ICD/RxNorm snapshot intake, indexing, forensics | yes | — | real behaviour | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `evidence_graph/` (2f, 100L) | L6 | Nodes/edges + graph hash | yes, wired | — | real behaviour | `PARTIAL` → **KEEP** |
| `confidence_cascade/` (2f, 59L) | L7 | Deterministic threshold fallback | yes, wired | — | real behaviour | `PARTIAL` → **KEEP** |
| `metric_decoder/` (2f, 53L) | L8 | Selection + fixed top-10 truncation | yes, wired | — | real behaviour | `PARTIAL` (**name overstates it**) → **KEEP** |
| `validator/` (9f, 1,370L) | L9 | Offsets, organizer policy, duplicates, submission layout, ZIP | yes, wired | — | real behaviour | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `organizer_policy/` (3f, 257L) | L9 | Confirmed/unresolved organizer facts | yes | — | real behaviour | `IMPLEMENTED_AND_WIRED` → **KEEP** |
| `inference/` (6f, 392L) | L1–L9 | The one end-to-end pipeline | yes | — | integration | `IMPLEMENTED_AND_WIRED` → **KEEP** |

### 4.2 Abandoned workflow packages

| Path | Layer | Actual responsibility | Imported by `src/` | Duplicate of | Tests validate | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| `zs0/profile.py` (211L) | — | ZS0 arm definitions, forbidden-component list | **no** | — | contract only | `OBSOLETE` → **DELETE** |
| `zs0/runner.py` (290L) | — | ZS0 arm execution + per-arm metrics | **no** | `evaluation/` | contract only | `OBSOLETE` → **DELETE** |
| `zs0/submission.py` (327L) | L9 | Arm selection, organizer gate, **its own** payload/ZIP validator | **no** | `validator/submission.py` + `validator/organizer.py` — **and contradicts them** (§10.2) | contract only | `DUPLICATED` → **DELETE** |
| `zs0/resolver.py` (272L) | L4 | Conservative deterministic resolver over `ExpertSpanProposal` | **no** | `resolution/` | contract only | `DUPLICATED` → **DELETE** |
| `zs0/ledger.py` (229L) | — | 9B gate, `MAX_ACTIVE_PARAMETERS`, `LedgerEntry`, `build_ledger` | **no** | `governance/parameter_budget.py` | contract only | `DUPLICATED` → **MIGRATE_THEN_DELETE** |
| `zs0/backends.py` (285L) | L3/L7 | Lazy GLiNER + Qwen loaders, JSON extraction | **no** | second `GLiNERBackend` name | contract only | **MIGRATE_THEN_DELETE** |
| `zs0/proposals.py` (344L) | L3 | Offset verification, substring resolution, rejection ledger, GLiNER label map | **no** | second `convert_gliner_spans` | real behaviour | **MIGRATE_THEN_DELETE** |
| `zs0/linking.py` (287L) | L5 | Snapshot membership, constrained code selection | **no** | — (genuinely new) | real behaviour | **MIGRATE_THEN_DELETE** |
| `zs0/assertions.py` (226L) | L5 | Cue+scope, constrained adjudication | **no** | overlaps `specialists/assertion/hydra.py` lexicons | real behaviour | **MIGRATE_THEN_DELETE** |
| `training/phase2/e4/` (10f, 5,595L) | L3 | E4 alignment, recipes, sampling, training, gates, Colab IO, progress | **no** | — | contract + real behaviour | `OBSOLETE` (retired) → **MIGRATE_THEN_DELETE** |
| `governance/post_e4_gates.py` (164L) | — | Blocks 9 downstream tasks on an E4 full artifact | **no** | — | real behaviour | `OBSOLETE` → **DELETE** (§9.3) |

### 4.3 Supporting code kept as-is

| Path | Classification | Reason |
| --- | --- | --- |
| `training/phobert_alignment.py` (783L) | `IMPLEMENTED_AND_WIRED` → **KEEP** | **Not an E4 artifact.** It is the VnCoreNLP-segmented-word ↔ character aligner used by S1 (`s1_mention_smoke.py`, `s1_full_training.py`) and by `mention_factory/neural/runtime.py`. Named for the tokenizer family (ViHealthBERT-Word uses the slow `PhobertTokenizer`), not for E4. Renaming it would touch 4 notebooks and 3 tests for no behavioural gain; recorded as naming debt in the manifest instead |
| `training/s1_*.py`, `training/colab_bootstrap.py` | `IMPLEMENTED_AND_EXECUTED` → **KEEP** | The only executed training path in the project |
| `training/phase2/{e5_*,l4_training,s2_assertion_training}.py` | `REQUIRES_FUTURE_TRAINING` → **KEEP** | Architecture-declared (spec §6 E5, §7 learned L4, §8 S2); implemented, untrained |
| `training/phase2/training_contracts.py` | `IMPLEMENTED_AND_WIRED` → **KEEP** | Already expert-independent; only its E4-flavoured docstrings were neutralized |
| `data_engine/review.py`, `evaluation/ablation_cli.py`, `*/cli.py` | `REVIEW_REQUIRED` → **KEEP** | Orphaned by the import graph, but CLI modules are entry points invoked via `python -m`, and the review module is a documented operator tool. Deleting an entry point because no library imports it would be a category error. Flagged, not removed |
| `experiments/` | `IMPLEMENTED_NOT_WIRED` → **KEEP** | Registry records are tracked and diff-friendly; no leaderboard decision has used it yet |

## 5. Mapping of current code to L1–L9

See `docs/architecture/ACTIVE_RUNTIME_MANIFEST.md` (created by this audit) for the
full per-layer manifest with checkpoints, contract versions, config sources,
enabled/disabled state and per-layer gaps. Summary:

```text
L1 Document Intelligence   document_intelligence/ + position/       IMPLEMENTED_AND_WIRED
L2 Case Router             case_router/                             IMPLEMENTED_AND_WIRED (accuracy unmeasured)
L3 Mention Factory         mention_factory/ + lattice/              PARTIAL — 1 of 6 experts trained
L4 Boundary & Type         resolution/                              PARTIAL — v1 disabled on measurement, v2 untrained
L5 Assertion               specialists/assertion/                   PARTIAL — deterministic only, zero supervision
L5 ICD-10 / RxNorm         linking/ + kb/                           PARTIAL — lexical only
L6 Evidence Graph          evidence_graph/                          PARTIAL — records evidence, no global reasoning
L7 Confidence Cascade      confidence_cascade/ + llm/               PARTIAL — no LLM stage runs
L8 Metric Decoder          metric_decoder/                          PARTIAL — NOT metric-aware
L9 Validator & Packager    validator/ + organizer_policy/            IMPLEMENTED_AND_WIRED, never run to a submission
```

## 6. Implementations that are genuinely complete

Complete meaning: implemented, wired, exercised by tests against real behaviour, and
requiring no checkpoint.

1. **L1 Document Intelligence** — including the hardest part of the whole spec, the
   §4 offset invariant, with astral-plane protection and canonical-equivalence
   handling proven on the full 24,844-example corpus preflight (Audits 0027–0030).
2. **L9 Validator** — schema, offsets, organizer per-type field policy, duplicates,
   the `1.json`..`100.json` set, UTF-8, deterministic ordering, ZIP layout.
3. **The local evaluator** — exact character-offset mention scoring, WER, Jaccard,
   matching strategies, HTML/JSON audit export. Byte-pinned against silent drift.
4. **L2 Case Router** — implemented completely as specified; its *accuracy* is
   simply unknown, which is a data gap, not an implementation gap.
5. **E1 medication grammar and E2 laboratory parser** — the full spec §6.1/§6.2
   grammars with stress suites.
6. **KB intake and governance** — ICD-10 TT06-2026 derived deterministically from
   the protected source PDFs; RxNorm Full and Prescribable snapshots; manifests
   hashed; the 9B budget gate that fails closed.
7. **S1 mention training** — the only stage with a validated artifact.

## 7. Implementations that are partial or contract-only

| Item | What exists | What does not |
| --- | --- | --- |
| L8 metric decoder | Selection of cascade-accepted entities | **Everything spec §13 asks for**: expected-Jaccard set search, expected-WER boundary utility, per-label assertion thresholds, wrong-type-risk-adjusted retention. The fixed top-10 truncation is the "fixed top-K" §13.2 rules out |
| L7 cascade | Deterministic threshold fallback; local-only loaders; constrained schemas | Critic stage, adjudicator stage, entry conditions, local Qwen weights |
| L6 graph | Nodes, 3 edge kinds, deterministic hash | 6 of 9 spec edge kinds; **all** global reasoning (beam search / ILP / constraints) |
| L5 assertion | Cue families, scope window, multi-label, empty-on-uncertainty | Learned section prior (A1), learned scope (A3), entity–cue relation (A4), per-label calibration (A6). **Zero assertion supervision exists** |
| L5 ICD | Alias + lexical retrieval, snapshot enforcement | BM25, character n-gram, dense, cross-encoder, hierarchy specificity controller (§9.3) |
| L5 RxNorm | Lexical retrieval + TTY priority | Structured drug parse (§10.1), relation-graph traversal (§10.2), strength/dose-form hard negatives |
| L4 | Deterministic v1, learned v2 implementation | Boundary-offset head (§7.1); v2 checkpoint; v1 measures *below* baseline |
| L3 | E1/E2/E3 working | E5 head untrained; E6/E7 have no local weights; TEST_NAME/TEST_RESULT unsupervised |
| Reproducibility | `reproducibility/repository_checkout.py` | **No Dockerfile, no `requirements.lock`** — both required by spec §19 and Appendix A |

## 8. Items not present in the architecture

| Item | Why it is not in the architecture | Action |
| --- | --- | --- |
| `zs0/` (whole package) | The PDF contains no zero-shot arm profile, no `ZS0-A/B/C`, and no arm-selection rule. ZS0 was a milestone-local baseline direction | Deleted after migration |
| `zs0/submission.py` arm selection + organizer gate | Selection between baseline arms is not an architecture concern; §16 defines one inference flow | Deleted |
| `zs0/submission.py` payload/ZIP validator | A **second packager for one profile**, contradicting the confirmed layout | Deleted (§10.2) |
| `governance/post_e4_gates.py` | Nothing in the PDF gates work on one expert's artifact | Deleted |
| `boundary_type/`, `specialists/icd/`, `specialists/rxnorm/` | The PDF names the *layers*; it does not sanction two import paths per layer, one of which returns `[]` | Deleted |
| `mention_factory/qwen_proposer.py` | An interface whose only method raises. §6 E7 is a real expert; a permanently-raising stub is not an implementation of it | Deleted |
| E4 in active profiles/registries/ledgers/arms | E4 *is* in the PDF as a candidate expert, but the active stack is a validated subset and E4 is retired from it | Purged from every active path |

Nothing invented a new architecture layer, and no module emitted final organizer
JSON directly. Three items that would have been the worst kind of violation were
checked for specifically and **not found**: a model backend writing organizer JSON,
an expert bypassing `SpanProposal`/L4, and hard-coded fake metrics.

One genuine instance of "random untrained task heads available at inference" was
found and remains correctly gated: E5's MRC head is randomly initialized, and
`enable_e5_xlmr_mrc: false` plus the missing-checkpoint gate keep it out of
inference. It is recorded in the manifest rather than removed, because E5 is an
architecture-declared expert awaiting training.

## 9. E4 retirement and cleanup

### 9.1 Status

```text
E4 PhoBERT-W2NER:
RETIRED_FROM_ACTIVE_ARCHITECTURE
```

Audit 0048 recorded the owner's decision but kept the implementation "as historical
evidence". This audit removes it. The reasoning: the *evidence* is the measured
result and the audits, both of which are preserved exactly. 5,595 lines of training
code, a 67 KB notebook, a 12.5 KB config and 3,548 lines of tests were not evidence —
they were a live, importable path to an expert that will never be trained, and every
future reader would have had to re-derive that it is dead.

### 9.2 Withdrawn from

| Surface | Before | After |
| --- | --- | --- |
| `DEFAULT_FEATURE_FLAGS` | `enable_e4_phobert_w2ner: False` | **key absent** |
| `CHECKPOINT_BY_FEATURE_FLAG` | maps to `mention/phobert_w2ner` | **key absent** |
| `configs/pipeline/full_v1.yaml` | flag `false` in 4 places | **flag removed everywhere**; loader refuses its return |
| `lattice.AVAILABLE_EXPERTS` | 7 experts | 6; `RETIRED_EXPERTS` names E4 and the lattice refuses it as an unknown expert |
| `configs/model_registry/models_v1.yaml` | `phobert_large_w2ner` role, 370M+3M, in 4 profiles | **entry removed** |
| `configs/models/candidate_model_registry.yaml` | `e4_phobert_w2ner` component | **entry removed** |
| `configs/parameter_budget.yaml` | PhoBERT-large `in_profile: true`, summed | `in_profile: false`, **not summed**; active plan 8,852,000,000 → **8,482,000,000** |
| Ablation arms | 3 of 8 arms required E4 | **0 arms**; `ARM_E3_E4`, `ARM_E3_E4_E5` removed, `ARM_E3_E5_E6` added |
| `configs/training/phase2_frozen_proposal_generation.yaml` | `e4_phobert_w2ner` entry | removed |
| `configs/routes.yaml` | C3 comment "E3-E6 (…W2NER…)" | "E3/E5/E6" |
| Docs | `ARCHITECTURE_DIGEST`, `CASE_ROUTING`, notebook integrity report | retirement stated explicitly |

**A new enforcement, not just a removal:** `PipelineConfig.load` now calls
`assert_e4_absent_from_flags` on the top-level flags **and on every nested per-mode
profile**, and `assert_no_e4_checkpoint_required` on both checkpoint lists. A config
that declares the flag is refused **whether it is set true or false** — a dead flag
is a flag somebody can flip.

### 9.3 `post_e4_gates.py` deserved its own decision

It blocked nine downstream tasks — including `train_learned_l4_v2` and
`submit_leaderboard_predictions` — on a validated E4 full artifact. Since no such
artifact can ever exist, the gate was permanently shut and was blocking live work
behind a retired expert. That is worse than dead code. Deleted.

### 9.4 Correction to this audit's own first draft

This audit initially recorded two **false** statements about E4. Both are corrected
here, in this same append-only document, with the corrected text carried into
`governance/e4_retirement.py`, `tests/unit/test_e4_retirement.py`,
`docs/architecture/ACTIVE_RUNTIME_MANIFEST.md` and
`configs/parameter_budget.yaml`:

| Claimed (wrong) | Actual |
| --- | --- |
| "370,000,000 base + 3,000,000 adapter", `DECLARED_ESTIMATE_NEVER_PROGRAMMATICALLY_VERIFIED` | **369,163,264 encoder + 2,125,897 W2NER head = 371,289,161 total**, `PROGRAMMATICALLY_VERIFIED` |
| "no checkpoint was ever produced"; "E4 never reached a stage that writes one" | E4 **did** produce checkpoints; Audit 0044 restored and probed two of them |

The error came from reading the *planning* row in `configs/parameter_budget.yaml`
(spec §17's approximate 0.370B for PhoBERT-large) as though it were E4's measured
size, and from inferring "no checkpoint" from "no checkpoint passed the gate". The
two are different claims and only the second is true.

### 9.5 The exact historical record

```text
E4 produced training, diagnostic and reproduction checkpoints.
No E4 checkpoint passed the acceptance gate.
No E4 checkpoint is retained in the current tree or active deployment.
Obsolete E4 artifacts were deleted after evidence was recorded in the
append-only audits.

encoder / base parameters       369,163,264
W2NER head parameters             2,125,897
total parameters                371,289,161
count status                    PROGRAMMATICALLY_VERIFIED
evidence                        Audit 0044
```

Audit 0044 restored `best.pt` and `latest.pt`, instantiated the model twice, and
reconciled the instantiated total against each checkpoint's own declared count —
**zero missing and zero unexpected keys on both encoder and head**, all four figures
agreeing exactly. Its recorded digests, verifiable in that audit:

```text
full_training/checkpoints/best.pt
    4cc934eb5d072bcf827e46745bbcc308beda3552b4156c4c4504f571aa0bd16f
full_training/checkpoints/latest.pt
    22b9017da3e5a56b7086c7f03dda1aed7e78b5e7b9f844d6171ee9333355bf07
```

The later tiny-overfit execution also serialized and reloaded checkpoints. One
further digest is **operator-reported** for the hard-negative recipe:

```text
tiny_overfit/hard_negative_ce
    93495b355a52d9742e89247f4afd07ddf876b24a5338e4359c30172f60e8f558
```

It is recorded in `E4_OPERATOR_REPORTED_CHECKPOINT_SHA256`, deliberately **separate**
from the audited digests, because it appears in no prior audit and therefore cannot
be independently checked from this repository. Presenting it as audit-verified would
repeat exactly the kind of error §9.4 corrects. A test asserts the separation holds.

**Two different numbers, never to be conflated:**

| Number | Provenance | Where it lives |
| --- | --- | --- |
| 370,000,000 | spec §17 approximate **planning estimate** for PhoBERT-large | `configs/parameter_budget.yaml`, explicitly labelled, `in_profile: false`, not summed |
| **371,289,161** | **measured**, `PROGRAMMATICALLY_VERIFIED`, Audit 0044 | `governance/e4_retirement.py` |

The planning row may remain, and does, only because it now says which of the two it
is. A test asserts `base_params != E4_TOTAL_PARAMETERS` and that the row's note
contains the word "planning".

### 9.6 Preserved as historical evidence

- Audits 0043–0048, byte-identical, plus this audit.
- `governance/e4_retirement.py` (`e4-retirement-v3`), a **concise record**: status,
  the complete Stage-2 result (12 examples, 22 gold mentions, 200 epochs, 600
  optimizer steps, warmup served, peak LR reached, save/reload reproduced,
  `reference_ce` 0.3448 / `group_balanced_ce` 0.7333 / `hard_negative_ce` 0.3704
  against the 0.95 gate, no recipe selected), the exact measured parameter counts,
  `CHECKPOINT_HISTORY`, both digest sets, and four assertions that keep E4
  unreachable.
- `configs/parameter_budget.yaml` keeps the PhoBERT-large planning row with
  `in_profile: false`, so the withdrawal is *visible* inside the spec's own stack
  table rather than silently absent.
- `tests/unit/test_e4_retirement.py` asserting all of the above, including that the
  three measured figures are findable in Audit 0044 and that the audited digests
  really are recorded there.

### 9.7 E4 artifacts on disk

Searched for E4 checkpoints, best/latest model files, optimizer states, scheduler
states, training-state files, prediction dumps, temporary exports and cached
tensors. **None remains.** They existed and were removed after their evidence was
recorded in the append-only audits — which is the correct order, and is why Audits
0043/0044 can still be read as complete probes of files that no longer exist. A test
asserts the only `*.pt` in the tree is the E3/S1 checkpoint.

`reports/e4_alignment/` (78,537 bytes), the one generated E4 report, was deleted; its
content is summarized in Audits 0037/0038.

## 10. ZS0 / pretrained-only cleanup

### 10.1 What was there

Audits 0048–0050 built ZS0 (three arms: deterministic, +GLiNER, +Qwen) and repaired
its notebook three times. **It never completed a run.** No arm result, no selection,
no `output.zip`, no measured number — so there was no result to preserve, only
implementation. The direction is no longer the project goal.

### 10.2 The defect that settled the packager question

`zs0/submission.py` was not merely a duplicate of `validator/`; it **disagreed with
the confirmed organizer contract on three points**:

| Aspect | Confirmed (Audit 0002, `validator/`, `schemas/constants.py`) | `zs0/submission.py` |
| --- | --- | --- |
| ZIP layout | `output.zip` → `output/` → `1.json`… | files at the **ZIP root**; a directory entry was an **error** |
| Document root | a JSON **list** of entities | an **object** with an `"entities"` list |
| Position fields | `position: [start, end]` | `start_pos` / `end_pos` |
| Field policy | exact per-type field set enforced | not enforced |

Had ZS0 ever produced a submission, it would have validated it against a layout the
organizer does not use. This is the concrete cost of a second packager per profile,
and it is why `validator/` is now the only L9 packager.

### 10.3 Disposition

| Component | Disposition |
| --- | --- |
| `notebooks/MedNorm_ZS0_Baseline_Submission.ipynb` | **DELETE_OBSOLETE** — obsolete notebook |
| `zs0/profile.py`, `runner.py` | **DELETE_OBSOLETE** — abandoned baseline profile and runner |
| `zs0/submission.py` | **DELETE_OBSOLETE** — duplicate + divergent validator/packager |
| `zs0/resolver.py` | **DELETE_OBSOLETE** — duplicate L4 |
| `zs0/ledger.py` | **MIGRATE_THEN_DELETE** — duplicate of `governance/parameter_budget.py` |
| `zs0/backends.py`, `proposals.py`, `linking.py`, `assertions.py` | **MIGRATE_THEN_DELETE** |
| `configs/pipeline/zs0_baseline.yaml`, `configs/resolution/zs0_conservative_v1.yaml`, `configs/models/zs0_parameter_ledger.yaml` | **DELETE_OBSOLETE** — abandoned baseline profiles |
| `tests/unit/test_zs0_baseline.py` (1,413 lines) | **MIGRATE_THEN_DELETE** — the arm/selection/packager assertions preserved an abandoned workflow; the contract assertions were migrated |
| Audits 0048–0050 | **PRESERVE_AS_HISTORICAL_EVIDENCE** — unmodified |

## 11. Complete cleanup inventory

| Category | Count | Items |
| --- | --- | --- |
| **KEEP** | — | All of §4.1 marked KEEP, §4.3, all audits, the governed corpus, locked ontology snapshots, generated indices, the E3 checkpoint |
| **CONSOLIDATE** | 3 | Cue lexicons → one source of truth; L4 → `resolution/` only; L9 packaging → `validator/` only |
| **MIGRATE_THEN_DELETE** | 7 | `zs0/{backends,proposals,linking,assertions,ledger}.py`, `training/phase2/e4/contracts.py` (split identity), `mention_factory/w2ner.py` (span contract) |
| **DELETE_OBSOLETE** | 45 tracked files | §11.1 |
| **DELETE_GENERATED_ARTIFACT** | 2 paths | `caches/` (1,075,833,774 B), `reports/e4_alignment/` (78,537 B) |
| **PRESERVE_AS_HISTORICAL_EVIDENCE** | 51 | All numbered audits, unmodified; the retirement record; the measured Stage-2 result; the historical parameter figure |
| **DELETE_OBSOLETE** (untracked, owner decision) | 1 | `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`, 61,226 B — §11.3 |
| **REVIEW_REQUIRED** | 2 | §11.3 items 2-3 |

### 11.1 Deleted paths, sizes and reasons

Tracked source (all removed with `git rm`; every byte recoverable from history):

| Path | Tracked | Size | Files/Lines | Reason | Surviving source of truth | Evidence preserved |
| --- | --- | --- | --- | --- | --- | --- |
| `src/mednorm_vi/zs0/` | yes | 236,810 B | 10 / 2,594 | Abandoned baseline package; no `src/` consumer | migrated modules (§13) | Audits 0048–0050 |
| `src/mednorm_vi/training/phase2/e4/` | yes | 508,849 B | 10 / 5,595 | Retired expert's training package | `training/governed_splits.py` | Audits 0043–0048 |
| `src/mednorm_vi/governance/post_e4_gates.py` | yes | 6,166 B | 1 / 164 | Permanently-shut gate blocking live work | — | Audit 0042 |
| `src/mednorm_vi/mention_factory/w2ner.py` | yes | 21,440 B | 1 / 570 | E4-only expert | `mention_factory/spans.py` | Audits 0038–0047 |
| `src/mednorm_vi/mention_factory/qwen_proposer.py` | yes | 3,299 B | 1 / 105 | Permanently-raising stub, no consumer | `llm/backends.CausalLMBackend` | Audit 0035 |
| `src/mednorm_vi/boundary_type/` | yes | 4,491 B | 1 / 36 | Dead second L4 path | `resolution/` | Audit 0001 |
| `src/mednorm_vi/specialists/icd/` | yes | 3,274 B | 1 / 28 | Dead wrapper returning `[]` | `linking/icd10.py` | Audit 0011 |
| `src/mednorm_vi/specialists/rxnorm/` | yes | 3,311 B | 1 / 28 | Dead wrapper returning `[]` | `linking/rxnorm.py` | Audit 0011 |

Tracked tests (3,548 lines of E4 tests + 1,413 of ZS0):

| Path | Size | Lines | Reason |
| --- | --- | --- | --- |
| `tests/unit/test_e4_clean_training.py` | 78,154 B | 1,792 | Tests a deleted package |
| `tests/unit/test_e4_runtime_io.py` | 22,801 B | 586 | Tests a deleted package |
| `tests/unit/test_e4_progress_observability.py` | 21,142 B | 496 | Tests a deleted package |
| `tests/unit/test_e4_atomic_grid_alignment.py` | 18,729 B | 443 | Tests a deleted package |
| `tests/unit/test_e4_phobert_alignment_fix.py` | 8,421 B | 231 | Tests a deleted package |
| `tests/unit/test_w2ner_expert.py` | 2,543 B | 75 | Tests a deleted module |
| `tests/unit/test_zs0_baseline.py` | 58,232 B | 1,413 | Preserved an abandoned workflow; contract coverage migrated |

Tracked configs and notebooks:

| Path | Size | Reason |
| --- | --- | --- |
| `configs/training/phase2_e4.yaml` | 12,566 B | Retired expert's runtime config |
| `configs/mention_factory/phobert_w2ner_v1.yaml` | 865 B | Retired expert's profile |
| `configs/pipeline/zs0_baseline.yaml` | 3,606 B | Abandoned baseline profile |
| `configs/models/zs0_parameter_ledger.yaml` | 2,841 B | Abandoned baseline ledger |
| `configs/resolution/zs0_conservative_v1.yaml` | 587 B | Abandoned baseline resolver config |
| `notebooks/MedNorm_E4_Clean_Training.ipynb` | 67,573 B | Retired expert's only execution path |
| `notebooks/MedNorm_ZS0_Baseline_Submission.ipynb` | 31,676 B | Abandoned baseline; never completed a run |

**Total tracked deletion: 45 files, ~1.20 MB, 12,236 lines of source + 5,036 lines
of tests.**

Untracked generated artifacts:

| Path | Size | Reason | Reproducible from |
| --- | --- | --- | --- |
| `caches/` | **1,075,833,774 B (1.1 GB)** | pip HTTP cache, empty HF cache dir, torchinductor temp, `torch_install.log`. Stale cache inside a repository path | `pip install` / a Colab run |
| `reports/e4_alignment/` | 78,537 B | Generated E4 alignment diagnostic, already summarized | Audits 0037/0038 |
| `**/__pycache__/` | — | Stale bytecode, including `.pyc` for 8 test modules that no longer exist | Any import |

### 11.2 Never deleted, verified

```text
docs/MedNorm-VI_Architecture.pdf            byte-identical (§2)
docs/audits/*.md                            51 files, contiguous 0001-0051, none edited
data/ (governed corpus, 3.4 GB)             untouched
indices/ (660 MB, locked snapshots)         untouched
checkpoint/s1_mention_full_training_v1/     untouched — see below
credentials / .env / secrets                none present, none touched
files outside the repository                none touched
```

**The E3 checkpoint was not deleted.** `checkpoint/s1_mention_full_training_v1/best.pt`
(1,615,513,303 bytes, SHA-256 `a64cc173…1017c`) is the only trained model the project
has. Audit 0032 recorded a byte-identical Drive copy, but that copy could not be
verified from here, so the local file is treated as an unverified-only-copy and kept
untouched. Its mtime is unchanged.

### 11.3 Owner decisions

**1. `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb` — `DELETE_OBSOLETE`, deleted.**

This file appeared on disk during this audit. It is the pre-Audit-0045 E4 notebook
path, deleted from git in commit `7a84c8c`. The first draft of this audit classified
it `REVIEW_REQUIRED` and left it in place because its only-copy status could not be
proven. **The owner has since decided explicitly: `DELETE_OBSOLETE`.** Its facts,
recorded before deletion:

```text
path              notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb
tracked status    UNTRACKED  (git ls-files: 0 entries; git status: ??)
size              61,226 bytes
sha256            80cf751e33ea98764715f1fc024882893ee3253868553699c8f08810a76fa227
mtime             2026-07-29 06:10:30 +0700
in any commit     NO  (git log --all --find-object: 0 commits)
last tracked at   7a84c8c^  (blob 8b9230200210eb4f8b6b1483e7d3d1f00f3af2e2,
                  sha256 85040fbb824521582d5d04d7c45af293db7927b8a746a4d5b52558b32ce92813
                  — a DIFFERENT digest, so the on-disk copy was not that version)
lint state        5 ruff errors (2x F401, 3x F821): it imports
                  mednorm_vi.training.phase2.e4_runtime_io, a flat module path that
                  has not existed since Audit 0045
purpose           training E4, which is RETIRED_FROM_ACTIVE_ARCHITECTURE
```

The lint state is independent evidence it was dead superseded code rather than
anything in use. **Deleted. Not replaced.** No filename-specific ignore rule was
added, and the quarantine mechanism was removed with it (§11.4), so the notebook
inventory test now rejects **every** unexpected untracked notebook with no
exceptions.

**2. CLI entry points with no library importer** — `training/cli.py`,
`training/phase2/cli.py`, `model_registry/cli.py`, `kb/indexing/cli.py`,
`kb/icd10/conversion/cli.py`, `phase1c_foundation/cli.py`, `round2/cli.py`,
`evaluation/ablation_cli.py`, `data_engine/review.py`, `training/manifests.py`.
The import graph shows no `src/` consumer, which is expected for `python -m` entry
points. Kept. A future milestone should confirm each is still a documented command
and delete any that is not.

**3. `reports/` (2.3 MB, git-ignored)** — generated reports from Audits 0009–0033.
Reproducible, but several are cited by tracked docs. Kept; a future milestone should
decide whether cited reports belong in the audits instead.

### 11.4 Exemptions removed with the stray notebook

Deleting the file made three carve-outs unnecessary, and all three are gone:

```text
tests/unit/test_notebooks.py   QUARANTINED_UNTRACKED_NOTEBOOKS removed entirely,
                               along with the two tests that read it. The inventory
                               test now fails on ANY untracked notebook.
pyproject.toml                 extend-exclude removed; ruff again lints every
                               notebook in the directory with no exceptions.
docs/notebooks/...integrity.md the REVIEW_REQUIRED paragraph replaced by a plain
                               statement that the path was deleted.
```

## 12. Consolidations applied

| Duplication | Before | After |
| --- | --- | --- |
| L4 resolver | `boundary_type/` (raises) + `resolution/` (real) | `resolution/` only |
| L5 ICD/RxNorm linking | `specialists/{icd,rxnorm}/` (returns `[]`) + `linking/` (real) | `linking/` only |
| L9 packaging/validation | `zs0/submission.py` (divergent) + `validator/` (confirmed) | `validator/` only |
| Deployment ledger | `zs0/ledger.py` + `governance/parameter_budget.py` | `governance/parameter_budget.py`, with concurrency + safetensors counting folded in |
| GLiNER adapter | `zs0/backends.GLiNERBackend` (concrete) + `mention_factory/gliner.GLiNERBackend` (protocol) — **two classes, same name** | protocol in `mention_factory/gliner.py`, loader in `llm/backends.OpenTypeSpanBackend` |
| Assertion cue lexicons | `zs0/assertions.py` + `specialists/assertion/hydra.py`, differing (`phủ định` vs `phủ nhận`) | one union in `specialists/assertion/cues.py`; `hydra.py` imports it |
| Governed split identity | inside `e4/contracts.py` | `training/governed_splits.py` |
| `EntitySpan` / type order | inside `mention_factory/w2ner.py` (E4), imported by E5 | `mention_factory/spans.py` |

`SpanProposal` remains defined in three places (`schemas/spans.py`,
`mention_factory/models.py`, and `lattice/ExpertSpanProposal`). Audit 0050 flagged
this. They are **not** interchangeable — one is the spec contract, one is the Phase-1B
deterministic output, one is the Phase-2 expert output — so collapsing them is a
design change requiring a measured migration, not a cleanup. Recorded as a gap.

## 13. Migrated reusable components

Every migration is a move to a path the spec §19 module contract names, or into an
existing canonical package. No new parallel package was created.

| From | To | What moved | Why it is reusable |
| --- | --- | --- | --- |
| `zs0/backends.py` | **`src/mednorm_vi/llm/backends.py`** (spec §19 `llm/`) | Lazy `OpenTypeSpanBackend`, `CausalLMBackend`, `require_local_directory`, `assert_no_external_api` | Model-independent local loaders; no baseline concept |
| `zs0/backends.py` | **`src/mednorm_vi/llm/structured_output.py`** | `extract_json_object`, `normalize_json_completion`, `require_json_object` | Generic constrained-output parsing (spec §12.1) |
| `zs0/proposals.py` | **`src/mednorm_vi/mention_factory/offsets.py`** | `verify_span`, `resolve_occurrence`, `resolve_in_document`, `RejectionLedger`, 9 reason codes | The spec §4 invariant and §1's "an LLM must never calculate offsets freely", for any source |
| `zs0/linking.py` | **`src/mednorm_vi/linking/snapshot.py`** | `OntologySnapshot`, `Candidate`, `generate_candidates`, `constrain_selection`, `assert_codes_in_snapshot`, `link_against_snapshot` | Spec P7 enforcement for either ontology, any retriever, any model |
| `zs0/assertions.py` | **`src/mednorm_vi/specialists/assertion/cues.py`** | Cue families, `find_cues`, `decide_from_cues`, `constrain_adjudication`, `build_adjudication_payload` | Spec §8 stages A2/A3 deterministic form |
| `zs0/ledger.py` | **`governance/parameter_budget.py`** | `concurrent_with_other_models` + `concurrently_resident_parameters` (spec §21), `counted_from_safetensors_index` | Folded into the existing canonical ledger |
| `training/phase2/e4/contracts.py` | **`src/mednorm_vi/training/governed_splits.py`** | `resolve_governed_split_by_sha256`, `resolve_governed_splits`, `assert_split_allowed`, corpus digests | Split identity by hash is not expert-specific |
| `mention_factory/w2ner.py` | **`src/mednorm_vi/mention_factory/spans.py`** | `EntitySpan`, `DEFAULT_TYPE_ORDER`, `assert_type_order_complete` | E5 already imported these *from E4*; the contract was never E4's |

**Two behavioural improvements were made during migration**, both required by the
task's §8:

1. `local_files_only=True` is now passed **unconditionally** and cannot be
   overridden; the old ZS0 backends resolved a **hub id** (`urchade/gliner_medium-v2.1`,
   `Qwen/Qwen3-1.7B`) and would have downloaded on first use. The new backends take a
   **local directory** and raise `LocalCheckpointMissing`, naming the exact path, when
   it is absent.
2. Model-specific naming was removed from architecture-neutral code: `QwenBackend` →
   `CausalLMBackend`, `GLiNERBackend` → `OpenTypeSpanBackend`, `qwen_payload` →
   `selection_payload`, `parse_qwen_assertions` → `constrain_adjudication`.

### Migrated tests

`tests/unit/test_zs0_baseline.py` (1,413 lines) was replaced by two files rather than
deleted outright:

- **`tests/unit/test_migrated_contracts.py`** (40 tests) — the surviving behavioural
  coverage, re-pointed at the new modules: span contract, offset invariant with a
  reason code per failure mode, ambiguous-surface-form refusal, anchor
  disambiguation, document-coordinate re-verification, rejection ledger carrying no
  clinical text, cross-ontology refusal, alias-wins-outright, drifted-index
  filtering, empty-beats-invented, **a model may not emit a code that exists in the
  snapshot but was not offered**, malformed-selection fallback, cue/scope decisions
  including out-of-scope uncertainty, adjudicator label constraints, local-only
  fail-closed loading, constrained JSON extraction, split-by-digest, and ledger
  concurrency/shared-backbone/LoRA accounting.
- **`tests/unit/test_e4_retirement.py`** (82 tests) — retirement and cleanup
  completeness (§17).

Not migrated, deliberately: ZS0 arm construction, arm selection ordering, the
organizer authorization string, and the ZS0 package validator. Those described an
abandoned workflow, and the last of them contradicted the confirmed contract (§10.2).

## 14. Preserved historical evidence

- **All 51 audits**, contiguous `0001`–`0051`, none deleted, none edited. A test
  asserts contiguity.
- **E4's complete measured result**, verbatim in `governance/e4_retirement.py` and
  asserted field-by-field by test.
- **E4's exact measured parameter count** — 369,163,264 encoder + 2,125,897 head =
  371,289,161, `PROGRAMMATICALLY_VERIFIED` in Audit 0044 — kept distinct from spec
  §17's ~370M planning estimate, and **E4's checkpoint history**, which records that
  checkpoints were produced, none passed the gate, and none is retained.
- **The E3 checkpoint** and every governed-corpus digest.
- **`docs/notebooks/notebook_execution_integrity.md`** gained a "Notebooks removed in
  Audit 0051" section with the reason for each, and a `REVIEW_REQUIRED` note for the
  stray in §11.3. One inaccurate sentence this audit initially wrote there — claiming
  the path "had not existed for several audits" — was corrected once the untracked
  file was found.

## 15. Active architecture manifest

Created: **`docs/architecture/ACTIVE_RUNTIME_MANIFEST.md`**.

For each L1–L9 component it records implementation path, status, active deterministic
implementation, available pretrained adapter, existing trained checkpoint, future
checkpoint slot, enabled/disabled state, contract version, configuration source,
parameter-ledger behaviour, and known missing implementation. It states explicitly:

```text
E4 PhoBERT-W2NER:
RETIRED_FROM_ACTIVE_ARCHITECTURE
```

It does not describe any untrained or missing-checkpoint module as implemented. Three
places where a name previously flattered the implementation are called out in it:
`metric_decoder.decode_expected_jaccard` (does no expected-Jaccard decoding),
`evidence_graph` (records evidence, performs no reasoning), and
`training/phobert_alignment.py` (generic aligner, tokenizer-family name).

## 16. Remaining architecture gaps, ranked

*Ranked by expected value if taken in isolation. The **ordering §23 recommends is
different**: complete every layer to a pretrained-ready state first, because items
1 and 3 below depend on calibrated probabilities that no finished layer yet
produces. Read this as the gap inventory, and §23 as the plan.*

1. **L8 metric-aware decoding** (spec §13) — candidates carry 40% of the score and
   the decoder truncates at a fixed top-10, exactly what §13.2 rules out. No
   calibrated probabilities exist anywhere, so this also blocks §13.3 and §13.4.
2. **L5 RxNorm structured parse + graph search** (§10.1/§10.2) and **ICD hierarchy
   specificity controller** (§9.3) — both linkers are lexical-only.
3. **Calibration / S6** — nothing downstream of L4 is calibrated, so L8 has nothing
   to optimize over. §13 depends on this.
4. **L4 boundary-offset head** (§7.1) and a re-run ablation; deterministic v1
   currently measures 0.7039 against the E3-only 0.7103.
5. **Assertion supervision** — the governed corpus has zero. Until it exists, no
   assertion metric is computable and S2 cannot be validated. 30% of the metric.
6. **TEST_NAME / TEST_RESULT supervision** — two of five spec types are unlearned by
   the only trained expert.
7. **L6 global reasoning** (§11) — 6 of 9 edge kinds and all constraints missing.
8. **Reproducibility / Phase 7** — **no Dockerfile, no `requirements.lock`**. Spec
   §19 and Appendix A both require them; the private-test rebuild depends on it.
9. **L7 LLM cascade** (§12) — last, per §20's explicit ordering rule.
10. **Router accuracy** — never measured.
11. **`SpanProposal` ×3** — three contracts sharing one name (§12).

## 17. Tests and static checks

Run after all deletions, migrations and repairs. Full results in §17.1.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q          see §17.1
ruff check .                                              see §17.1
ruff check notebooks                                      see §17.1
env PYTHONPATH=src .venv/bin/python -m mypy               see §17.1
env PYTHONPATH=src .venv/bin/python -m compileall -q src  see §17.1
git diff --check                                          see §17.1
```

Focused checks proving the milestone's specific claims — all in
`tests/unit/test_e4_retirement.py` unless noted:

```text
E4 unreachable from all active and future profiles   every configs/pipeline/*.yaml loaded
                                                     and asserted flag-free and
                                                     checkpoint-free; a config that
                                                     re-declares the flag (true OR
                                                     false) is REFUSED; a nested
                                                     per-mode profile is refused too
no active registry contains E4                       candidate registry + models_v1.yaml
no active config requires an E4 checkpoint           both checkpoint lists, all profiles
no deployment ledger counts E4                       retired status is NON_DEPLOYABLE;
                                                     planned budget sums 8,482,000,000
                                                     = 8,852,000,000 - 370,000,000
no ablation arm requires E4                          PHASE2_ABLATION_ARMS + both configs
the lattice refuses a retired expert                 by the unknown-expert check, with
                                                     no special-case branch
obsolete ZS0 entry points removed                    15 deleted paths absent from disk
                                                     AND from git ls-files
deleted files have no remaining imports              AST scan of every tracked .py for
                                                     Import/ImportFrom of 8 deleted
                                                     modules; 17-needle text scan of
                                                     every tracked notebook
canonical architecture contracts import              10 modules imported explicitly
missing pretrained checkpoints fail closed           empty path and missing path both
                                                     raise LocalCheckpointMissing naming
                                                     the exact path; unloaded backends
                                                     refuse to predict or count
no test triggers model downloading                   no tracked test calls
                                                     from_pretrained without
                                                     local_files_only; none sets it False
no Hugging Face download begins                      no network call exists in the suite;
                                                     every loader takes a local directory
architecture inspection works without CUDA           whole-package import sweep passes on
or model assets                                      CPU with no model present
docs/MedNorm-VI_Architecture.pdf unchanged           SHA-256 pinned by test
no assistant control file exists                     CLAUDE.md / AGENTS.md / .claude/
                                                     absent from git ls-files
no audit was deleted                                 audit numbers contiguous 1..51
```

No test was weakened to make the cleanup pass. Four assertions changed because the
**fact** changed, each with the new number and its derivation recorded in the test:

| Test | Was | Now | Why |
| --- | --- | --- | --- |
| `test_budget.py::test_shipped_budget_config_is_valid_and_within_cap` | `== 8_852_000_000` | `== 8_482_000_000` **and** `== 8_852_000_000 - 370_000_000` | PhoBERT-large left the active plan |
| `test_phase2_flags_registry_notebooks.py::…safe_stack…` | `== 8_852_000_000` | `== 8_482_000_000` | same |
| `…::test_phase2_default_feature_flags_preserve_measured_baseline` | flag `is False` | flag `not in` | a dead flag was removed, not set false — a stronger assertion |
| `test_notebooks.py::test_notebook_inventory_matches_expected` | `== 18` | `== 16` | two notebooks deleted |

Three tests were **strengthened**: the notebook inventory now reads from `git
ls-files` rather than a directory glob (an untracked notebook is not part of a
reproducible workflow); a new test fails on any undocumented untracked notebook; and
`llm/backends.py` is held to a *stricter* rule than the module list it was moved out
of — every `from_pretrained` must pass `local_files_only=LOCAL_FILES_ONLY`, counted.

One genuine bug was found and fixed while re-pointing imports:
`notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb` imported `EntitySpan` from the
E4 module. That import would have failed on the next Colab run; it now reads from
`mention_factory/spans.py`.

### 17.1 Executed results

Final run, after every correction in §9.4, §11.3 and §23:

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q
    1504 passed, 1 skipped in 133.70s
    (skip: tests/unit/test_vietmed_adapter.py:399 — pyarrow not installed locally,
     pre-existing and unrelated)

ruff check .                                            All checks passed!
ruff check notebooks                                    All checks passed!
env PYTHONPATH=src .venv/bin/python -m mypy             Success: no issues found
                                                        in 264 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src   clean
git diff --check                                        clean

whole-package import sweep
(pkgutil.walk_packages + import_module)                 258 modules, 0 broken
```

Test-count movement, reconciled:

```text
before this audit (Audit 0050)      1792 passed, 1 skipped
E4 tests removed                    -3,548 lines across 6 modules
ZS0 tests removed                   -1,413 lines (1 module)
new/migrated tests added            +133  (test_e4_retirement 88,
                                           test_migrated_contracts 45)
after this audit                    1504 passed, 1 skipped
```

The net drop is entirely the deleted E4/ZS0 modules. **No surviving assertion was
removed or weakened** — every test that previously passed and still has a subject
still runs. §17's table lists the four assertions whose expected *value* changed
because the underlying fact changed, and the three that were strengthened; §17.3
lists the two factual corrections and the six tests that now hold them.

### 17.1.1 Required verifications, executed

```text
1  no untracked E4 notebook exists
       notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb        ABSENT
       untracked notebooks in notebooks/ at all                 0
       notebooks on disk == notebooks tracked by git            16 == 16

2  pyproject.toml has no E4-specific lint exclusion
       grep -c 'extend-exclude|PhoBERT_W2NER' pyproject.toml    0
       ruff check notebooks                                     All checks passed!

3  no E4 source, training path, active flag, registry entry or
   checkpoint requirement exists
       src/mednorm_vi/training/phase2/e4/                        absent
       src/mednorm_vi/mention_factory/w2ner.py                    absent
       8 deleted E4/ZS0 modules importable                        0 of 8
       'enable_e4_phobert_w2ner' in any config                    0 occurrences
       'component_id: e4_phobert' / 'role: l3_e4' in configs      0 occurrences
       'mention/phobert_w2ner' in configs                         1 occurrence,
           configs/pipeline/full_v1.yaml:55 — an explanatory COMMENT. Verified by
           loading the profile: full_requires_checkpoints and
           specialist_requires_checkpoints both exclude it; `contains E4 key: False`
       PipelineConfig.load refuses a config declaring the flag    asserted by test

4  the exact verified historical parameter count is preserved
       369,163,264 encoder + 2,125,897 head = 371,289,161
       status PROGRAMMATICALLY_VERIFIED
       all three figures re-read from Audit 0044 by test
       both audited checkpoint digests re-read from Audit 0044 by test
       operator-reported tiny-overfit digest kept separate, asserted by test

5  the active planning budget still excludes E4
       summed in_profile base params                8,482,000,000
       = 8,852,000,000 - 370,000,000 (planning row, in_profile: false)
       margin to the 9B cap                          518,000,000
       PhoBERT-large row labelled a planning estimate, asserted != 371,289,161

6  architecture PDF SHA-256 unchanged
       0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b
       git diff HEAD -- docs/MedNorm-VI_Architecture.pdf         0 lines

7  no model download occurred
       no network call exists on any path; every loader takes a local directory and
       passes local_files_only=True unconditionally (asserted by test)
       no tracked test calls from_pretrained without local_files_only

8  no internal_test or organizer inference occurred
       internal_test never opened; assert_split_allowed refuses it by name
       no organizer inference, no output.zip, no Docker image build

9  repository hygiene
       docs/audits/                                 51 files, contiguous 0001-0051
       Audits 0001-0050 vs HEAD                     no diff
       git ls-files: .pt/.pth/.ckpt/.safetensors/.bin/.zip       none
       git ls-files: caches/ checkpoint/ artifacts/ weights/     none
       git ls-files: CLAUDE.md / AGENTS.md / .claude/            none
       only *.pt in the tree      checkpoint/s1_mention_full_training_v1/best.pt
                                  (E3/S1; asserted by test)
       E3 checkpoint after all operations
           SHA-256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
           size    1,615,513,303 bytes
           mtime   2026-07-27 00:53:41  (unchanged)
```

### 17.2 No linter scope change is retained

An earlier draft of this audit added
`extend-exclude = ["notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb"]` to
`pyproject.toml`, because that untracked file produced 5 ruff errors (2 x F401,
3 x F821) from importing `mednorm_vi.training.phase2.e4_runtime_io` — a module path
removed in Audit 0045 — and therefore could not pass lint by construction.

The owner then decided to delete the file (§11.3). **The exclusion was removed with
it**, so `ruff check notebooks` again lints every notebook in the directory with no
exceptions. `pyproject.toml` is now **byte-identical to `HEAD`**
(`git diff HEAD --quiet -- pyproject.toml` succeeds), so it is not in the changed-file
set and not in the staging command — this audit leaves the lint configuration exactly
as it found it.

Those 5 errors remain worth recording: they were independent evidence that the file
was dead superseded code rather than anything in use.

### 17.3 Corrections applied after the first draft

Both were factual errors in this audit's own text, corrected in place with the
corrected values carried into code, tests, the manifest and the config:

| Corrected | Was | Now |
| --- | --- | --- |
| E4 parameter count | 370,000,000 + 3,000,000, `DECLARED_ESTIMATE_NEVER_PROGRAMMATICALLY_VERIFIED` | 369,163,264 + 2,125,897 = **371,289,161**, `PROGRAMMATICALLY_VERIFIED` (Audit 0044) |
| E4 checkpoints | "none was ever produced"; "never reached a stage that writes one" | produced (training, diagnostic, reproduction); none passed the gate; none retained |

Six tests now hold these facts in place, including two that read Audit 0044 directly
and assert the three figures and both audited digests are genuinely recorded there —
so the corrected numbers cannot drift back without the corroborating audit changing
too.

## 18. Changed files

**Added (12):**
```text
docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
docs/audits/0051-repository-wide-architecture-review-and-cleanup.md
src/mednorm_vi/llm/__init__.py
src/mednorm_vi/llm/backends.py
src/mednorm_vi/llm/structured_output.py
src/mednorm_vi/linking/snapshot.py
src/mednorm_vi/mention_factory/offsets.py
src/mednorm_vi/mention_factory/spans.py
src/mednorm_vi/specialists/assertion/cues.py
src/mednorm_vi/training/governed_splits.py
tests/unit/test_e4_retirement.py
tests/unit/test_migrated_contracts.py
```

**Renamed (1):**
```text
tests/unit/test_parallel_post_e4_readiness.py -> tests/unit/test_e5_s2_readiness_and_budget.py
```

**Deleted (45):** listed in §11.1.

**Modified (33):**
```text
.gitignore                                              /caches/ anchored rule
README.md                                               layer -> module map corrected
configs/evaluation/l3_l4_ablation_v2.yaml               E4 arms removed
configs/evaluation/phase2_validation_ablation.yaml      E4 arms removed
configs/model_registry/models_v1.yaml                   E4 role removed
configs/models/candidate_model_registry.yaml            E4 component removed
configs/models/deployment_budget_template.yaml          note
configs/parameter_budget.yaml                           PhoBERT-large out of the plan
configs/pipeline/full_v1.yaml                           E4 flag removed everywhere
configs/routes.yaml                                     real module paths
configs/training/phase2_frozen_proposal_generation.yaml E4 entry removed
docs/architecture/ARCHITECTURE_DIGEST.md                retirement stated
docs/architecture/CASE_ROUTING.md                       real module paths
docs/notebooks/notebook_execution_integrity.md          removals + REVIEW_REQUIRED
docs/repository-layout.md                               naming inconsistency resolved
docs/audits/README.md                                   0051 index entry
notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb        broken EntitySpan import fixed
src/mednorm_vi/evaluation/l3_l4_ablation_v2.py          E4 arms removed
src/mednorm_vi/governance/__init__.py                   exports retirement, drops gates
src/mednorm_vi/governance/e4_retirement.py              rewritten as the record
src/mednorm_vi/governance/parameter_budget.py           concurrency, safetensors, status
src/mednorm_vi/inference/config.py                      flag removed + load-time refusal
src/mednorm_vi/lattice/__init__.py                      E4 export removed
src/mednorm_vi/lattice/builder.py                       docstring
src/mednorm_vi/lattice/models.py                        E4 out of AVAILABLE_EXPERTS
src/mednorm_vi/mention_factory/adapters.py              E4 manifest removed
src/mednorm_vi/mention_factory/mrc.py                   EntitySpan from spans.py
src/mednorm_vi/specialists/__init__.py                  documents the removals
src/mednorm_vi/specialists/assertion/hydra.py           shared cue lexicons
src/mednorm_vi/training/phase2/artifacts.py             E4 kind/branches/validator out
src/mednorm_vi/training/phase2/e5_mrc_training.py       EntitySpan from spans.py
src/mednorm_vi/training/phase2/e5_readiness.py          docstrings neutralized
src/mednorm_vi/training/phase2/proposal_generation.py   has_e4 -> has_e6
src/mednorm_vi/training/phase2/training_contracts.py    docstrings neutralized
src/mednorm_vi/training/phase2/validation_ablation.py   arm list updated
tests/unit/test_budget.py                               active plan total
tests/unit/test_e5_s2_readiness_and_budget.py           section D removed, pins repointed
tests/unit/test_l3_multi_expert_phase2.py               E4 exemplar -> E5
tests/unit/test_mrc_expert.py                           EntitySpan from spans.py
tests/unit/test_notebooks.py                            tracked inventory; no exceptions
tests/unit/test_phase2_flags_registry_notebooks.py      E4 out, loader rule added
tests/unit/test_phase2_training_readiness.py            E4 exemplar -> E5
```

## 19. Genuine blockers

1. **No assertion supervision exists.** 30% of the organizer metric cannot be
   measured or trained at all. This is a data-acquisition blocker, not a code one,
   and it is the single largest gap.
2. **No local pretrained weights.** E6, E7, the dense embedder, the reranker and both
   Qwen models have no verified local checkpoint, so L7 cannot be exercised and the
   parameter ledger cannot certify any deployment (it fails closed by design). This
   audit deliberately did not resolve it — downloading was out of scope.
3. **No Dockerfile and no `requirements.lock`.** Spec §19 and Appendix A require
   both; the private-test rebuild cannot be attempted without them.
4. **`checkpoint/best.pt` is an unverified only-copy.** Audit 0032 recorded a
   byte-identical Drive copy but it cannot be verified from here. Losing this file
   loses the project's only trained model. **This is now the only open blocker that
   risks losing work.**

*Resolved since the first draft:* the untracked E4 notebook — the owner decided
`DELETE_OBSOLETE` and it was deleted (§11.3).

## 20. Optional future hardening

- Rename `metric_decoder.decode_expected_jaccard` when the real §13 decoder lands, or
  sooner — the current name is the most misleading identifier in the repository.
- Rename `training/phobert_alignment.py` to something architecture-neutral; deferred
  here because it touches 4 notebooks and 3 tests for no behavioural gain.
- Collapse or clearly namespace the three `SpanProposal` contracts (§12).
- Make L9 enforce KB membership of emitted candidate codes, meeting
  `linking/snapshot.py` in the middle.
- Add the spec §18.3 stress suites that do not yet exist (packaging, ICD, RxNorm).
- Decide whether cited `reports/` artifacts belong inside the audits.
- Audit the ten CLI entry points in §11.3 and delete any that is no longer documented.

## 21. Safe-to-commit verdict

```text
SAFE_TO_COMMIT
```

All owner decisions are resolved and applied. Verified:

```text
architecture PDF                    byte-identical (§2)
Audits 0001-0050                    unmodified; 0051 added
governed corpus / ontologies         untouched
E3 checkpoint                        untouched, same mtime and digest
credentials / secrets                none present, none touched
files outside the repository         none touched
Git history                          not rewritten
assistant control files              none created, none tracked
training / fine-tuning / benchmark   none
model downloads                      none; no network call exists in any path
internal_test                        never opened
organizer inference                  did not run
output.zip                           not created
Docker image                         not built
nothing artifact-shaped staged       verified by test (no .pt/.bin/.zip/caches/...)
untracked notebooks                  NONE; the stray was deleted by owner decision
E4 historical record                 corrected and test-asserted (§9.4, §9.5)
lint/type/compile/whitespace         clean (§17.1)
```

## 22. Exact commit commands

Stage explicitly. **Do not use `git add -A`** — that is the habit that let an
untracked E4 notebook sit one command away from being committed in the first place.

```bash
# Staged deletions are already recorded by `git rm`. Stage the rest explicitly.
git add \
  docs/audits/0051-repository-wide-architecture-review-and-cleanup.md \
  docs/audits/README.md \
  docs/architecture/ACTIVE_RUNTIME_MANIFEST.md \
  docs/architecture/ARCHITECTURE_DIGEST.md \
  docs/architecture/CASE_ROUTING.md \
  docs/notebooks/notebook_execution_integrity.md \
  docs/repository-layout.md \
  README.md \
  .gitignore \
  src/mednorm_vi/llm/ \
  src/mednorm_vi/linking/snapshot.py \
  src/mednorm_vi/mention_factory/offsets.py \
  src/mednorm_vi/mention_factory/spans.py \
  src/mednorm_vi/specialists/assertion/cues.py \
  src/mednorm_vi/training/governed_splits.py \
  src/mednorm_vi/evaluation/l3_l4_ablation_v2.py \
  src/mednorm_vi/governance/ \
  src/mednorm_vi/inference/config.py \
  src/mednorm_vi/lattice/ \
  src/mednorm_vi/mention_factory/adapters.py \
  src/mednorm_vi/mention_factory/mrc.py \
  src/mednorm_vi/specialists/__init__.py \
  src/mednorm_vi/specialists/assertion/hydra.py \
  src/mednorm_vi/training/phase2/ \
  configs/ \
  notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb \
  tests/unit/test_e4_retirement.py \
  tests/unit/test_migrated_contracts.py \
  tests/unit/test_e5_s2_readiness_and_budget.py \
  tests/unit/test_budget.py \
  tests/unit/test_l3_multi_expert_phase2.py \
  tests/unit/test_mrc_expert.py \
  tests/unit/test_notebooks.py \
  tests/unit/test_phase2_flags_registry_notebooks.py \
  tests/unit/test_phase2_training_readiness.py

# Preflight: nothing artifact-shaped, no untracked notebook, no E4 identifier in a
# staged active path.
git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors|bin|zip)$' && echo FAIL || echo "no weights staged"
git status --porcelain notebooks | grep '^??' && echo FAIL || echo "no untracked notebooks"
git diff --cached --name-only | grep -c 'MedNorm_E4_PhoBERT_W2NER_Training'   # must print 0

git commit -m "chore: retire E4 from the architecture and consolidate the repository

Delete the retired E4 PhoBERT-W2NER implementation, the abandoned ZS0 baseline
and four dead duplicate layer packages; migrate their reusable contracts into
llm/, mention_factory/, linking/, specialists/assertion/ and training/; add the
active runtime architecture manifest.

Audit: docs/audits/0051-repository-wide-architecture-review-and-cleanup.md"

git push origin main
```

## 23. Recommended next consolidated architecture implementation milestone

### COMPLETE THE PRETRAINED-READY L1-L9 ARCHITECTURE

The goal is a repository where **every** L1-L9 layer is implemented, deterministic
where it can be, contract-complete where it cannot, and ready to receive pretrained
weights — *without* a single model download, training run or submission. Materializing
checkpoints is the milestone after this one, and it should find nothing left to build.

This supersedes the earlier draft recommendation (S6 calibration + L8 +
`output.zip`). That draft was wrong about ordering: calibration needs upstream
probabilities that no completed layer yet produces, and a first `output.zip` should
come from a *finished* deterministic pipeline rather than from one with four partial
layers underneath it.

**Scope:**

1. **Consolidate and version the three `SpanProposal` contracts.** `schemas/spans.py`,
   `mention_factory/models.py` and `lattice/ExpertSpanProposal` share one name and
   three meanings (§12). Give them one versioned lineage with explicit conversions, so
   every later layer is written against a single span contract.
2. **Complete the one canonical L1-L9 runner.** `inference/pipeline.py` is the only
   end-to-end path; make it *the* path, with every layer entered through a declared
   interface and full provenance recorded for replay and ablation (spec §19.1).
3. **Complete the remaining deterministic architecture logic** in every layer that
   currently stops short — the L4 global span optimizer's constraint set, L2 route
   priors feeding L4 weighting, L1 section-lexicon coverage.
4. **Complete ICD hierarchy and specificity logic** (spec §9.1 hierarchy graph, §9.3
   specificity controller): parent/child/sibling expansion with graph provenance, and
   the general-versus-specific decision that cosine similarity cannot make.
5. **Complete RxNorm structured medication parsing and graph traversal** (spec §10.1,
   §10.2): the full `D = {ingredient, salt, strength, unit, concentration, dose_form,
   release, brand, route, frequency, prn}` parse feeding the linker, ingredient →
   component → SCD → SBD traversal, and the strength/release hard-negative ordering.
6. **Complete the missing L6 evidence-edge types and deterministic consistency
   logic** (spec §11): `has_result`, `modified_by`, `in_section`, `treats`, `overlaps`,
   `same_surface`, plus the deterministic constraint checks (section consistency,
   test-result pairing, drug-indication, candidate-type consistency, overlap
   competition, repeated mention).
7. **Complete L7 interfaces and locked-option escalation paths** (spec §12): the
   cascade's entry conditions, the evidence-bundle construction, and the **locked
   option set** an adjudicator would be given — built and tested end to end with a
   stub decision source, so the day weights arrive nothing about the contract is new.
8. **Provide a truthful deterministic L8 fallback.** Replace the fixed top-10
   truncation with honest deterministic set selection, and **name it for what it
   does** — `decode_expected_jaccard` must not keep a name it has not earned. Leave
   the expected-Jaccard/WER machinery behind a disabled interface until calibrated
   probabilities exist.
9. **Leave learned S6 calibration disabled** until valid training data and upstream
   probabilities exist. Ship the interface and the fail-closed gate, not a fitted
   model.
10. **Implement real lazy pretrained adapters and checkpoint manifests** for E5, E6,
    E7, the dense embedder and the reranker: `local_files_only=True`, fail closed on a
    missing local directory, verified digest, and a manifest entry per role.
11. **Identify the exact pretrained checkpoints** the following materialization
    milestone should download — model id, pinned immutable revision, expected file
    layout, licence, and expected parameter count — recorded in the registry as a
    shopping list, with **nothing fetched**.
12. **Create `Dockerfile` and `requirements.lock`** as reproducibility
    infrastructure (spec §19, Appendix A). Author them; **do not build the submission
    image**.
13. **Keep every future trained/fine-tuned slot disabled and fail-closed** — no flag
    flips, and every missing checkpoint still raises by name.

**Explicitly not in this milestone:**

```text
calibration training          threshold search
organizer inference           output.zip
leaderboard submission        model downloads
```

**Exit criterion:** every L1-L9 row in
`docs/architecture/ACTIVE_RUNTIME_MANIFEST.md` reads `IMPLEMENTED_AND_WIRED` or
`REQUIRES_PRETRAINED_CHECKPOINT` — none reads `PARTIAL` or `SCAFFOLD_ONLY` — with the
manifest updated in the same change and no layer's status overstated.
