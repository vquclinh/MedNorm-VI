# Audit 0063: Post-Score Freeze, Artifact Hygiene, and Linker Diagnosis

Milestone: 4D - Post-score checkpoint freeze, safe artifact hygiene, candidate-linking root-cause audit
Date: 2026-08-01
Verdict: `ARTIFACTS_FROZEN_LINKER_DIAGNOSED`

## 1. Initial Git and Resource State

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `8faa445 feat: refine E3 boundaries and activate the validated checkpoint` |
| Working tree | clean |
| Staged files | none |
| `origin/main...HEAD` | `0 0` |
| Audit 0062 committed | yes |
| RAM | 14 GiB total, 7.0 GiB available |
| Disk (before) | 100 G, **11 G free (90% used)** |
| GPU | RTX 4060 Laptop, 8,188 MiB, 13 MiB used |
| Containers | 0 |

All four expected digests verified before any work began:

```text
docs/MedNorm-VI_Architecture.pdf                    0d5eaa20…81e09b   unchanged
runs/e3_boundary_v2_final_dotted/output.zip         eae8a348…4c05     matches
checkpoint/experiments/…/R3_alpha050/best.pt        524ece1e…dde3a    matches
checkpoint/s1_mention_full_training_v1/best.pt      a64cc173…1017c    matches
```

## 2. Leaderboard Result

The Audit-0062 candidate was submitted and scored.

| Submission | score | WER | J_assertion | J_candidates |
| --- | ---: | ---: | ---: | ---: |
| Dotted baseline (old E3) | 9.5736 | 87.0976 | 16.0234 | 2.2396 |
| **Refined E3 (Audit 0062)** | **11.9188** | **82.7090** | **18.6187** | **2.8646** |
| Delta | **+2.3452** | **−4.3886** | **+2.5953** | **+0.6250** |

`num_scored: 100`, `num_records: 100`. Both scores reproduce exactly under
`0.3·(100−WER) + 0.3·J_assert + 0.4·J_cand` — 9.5736 and 11.9187 against a published
11.9188 — confirming the reverse-engineered metric for the third time.

Score contribution by component:

| Component | contribution |
| --- | ---: |
| text (1−WER) | +1.3166 |
| J_assertion | +0.7786 |
| J_candidates | +0.2500 |
| **total** | **+2.3452** |

**This is the single most informative measurement in the milestone, and §7 turns on it.**
The KB, both linkers, the assertion logic, the evidence graph and the set decoder were
byte-identical across the two runs. The only thing that changed was the E3 mention
checkpoint. Yet **all three** scored components moved. J_assertion (+2.5953) and
J_candidates (+0.6250) are therefore *entirely* downstream consequences of better mention
extraction — measured, not inferred.

Audit 0062 is unmodified. The result arrived after it was written and is recorded here.

## 3. Artifact Inventory

656 files inventoried across `checkpoint/`, `models/`, `runs/`, `reports/`
(`caches/` and `local-artifacts/` do not exist). Full per-file inventory — path, size,
mtime, owning milestone, config/code/test/audit references, git-tracked flag,
reproducibility, runtime/rollback/resume need, disposition and reason — is written to the
ignored path `runs/diagnostics/0063_inventory/artifact_inventory.json`.

| Disposition | files | bytes | GiB |
| --- | ---: | ---: | ---: |
| KEEP_ACTIVE | 106 | 232,443 | 0.00 |
| KEEP_REPRODUCIBILITY_METADATA | 520 | 8,627,528 | 0.01 |
| KEEP_ROLLBACK | 1 | 1,615,513,303 | 1.50 |
| NEEDS_OWNER_BACKUP | 1 | 1,615,513,431 | 1.50 |
| SAFE_TO_DELETE | 27 | 12,468,869,350 | 11.61 |
| UNKNOWN_DO_NOT_DELETE | 1 | 718 | 0.00 |
| **TOTAL** | **656** | **15,708,756,773** | **14.63** |

Absence from git was **not** treated as evidence of safety — every generated artifact in
this project is untracked by design, so the test applied was live reference, not
tracking. The single `UNKNOWN_DO_NOT_DELETE` entry is `models/README.md` (git-tracked,
retained).

## 4. Canonical Active Checkpoint

The selected checkpoint lived under a scratch experiment path whose name encodes a
comparison that is now finished. It has been **copied** (never moved) to a permanent
location and the source left in place until every check passed.

```text
from  checkpoint/experiments/0062_e3_boundary_refinement/R3_alpha050/best.pt
to    checkpoint/e3_boundary_refinement_0062/best.pt
both  524ece1e7d190838cb8b1ce3b0a0f337bc5b8b7cc7cef70c4c3e0b0310adde3a  ✅
```

The canonical directory carries the smallest complete reproducibility contract:

| File | Bytes | Content |
| --- | ---: | --- |
| `best.pt` | 1,615,513,431 | the weights |
| `training_manifest.json` | 2,603 | git commit, PDF hash, split hashes, base model + revision, source checkpoint, tokenizer identity, label vocabulary, seed, every hyperparameter, config hash |
| `resolved_config.json` | 1,315 | the exact resolved hyperparameters and provenance hashes |
| `validation_metrics.json` | 1,752 | selection metric and rule, E3-only and L1→L4 gate measurements |
| `training_history.jsonl` | 12,479 | per-epoch loss, LR and full validation records |
| `checkpoint-manifest.json` | 1,821 | per-file digests, experiment origin, load mode, scored leaderboard result |

Three configs now reference the canonical path, and a test asserts the pipeline config and
the active profile cannot drift apart:

* `configs/models/e3_checkpoint_profiles.yaml` — `boundary_refinement_0062.path`
* `configs/pipeline/full_v1.yaml` — `expert_settings.e3_checkpoint_path`
* `configs/models/candidate_model_registry.yaml` — `active_checkpoint_path` + digest

The rollback profile is unchanged and still resolves the original artifact. Exactly one
active profile exists. No weight was altered — the digest is identical on both sides of
the copy.

**Bounded load validation** (one document, real weights, canonical path):

```text
configured path : checkpoint/e3_boundary_refinement_0062/best.pt
specialist ready: True
E3 executed     : True   proposals: 4
recorded digest : 524ece1e…dde3a  == configured digest  ✅
```

Fail-closed behaviour re-verified: wrong pinned hash is refused twice (at
`assert_load_is_permitted` and again at `fingerprint_checkpoint`), a missing checkpoint is
reported by readiness rather than raised, an empty path is refused, and an unverified
checkpoint can never reach the pickle path.

### Backup status — an explicit, deliberate exception

The experiment source `checkpoint/experiments/0062_e3_boundary_refinement/R3_alpha050/best.pt`
was **kept**, not deleted, and is classified `NEEDS_OWNER_BACKUP`.

The brief permits deleting it once the canonical copy validates. It was retained because
no second *external* backup could be verified, and this is the artifact behind the scored
11.9188 submission. The canonical copy and the experiment source sit on the same
filesystem, so this protects against accidental deletion and corruption, not against disk
failure. It costs 1.50 GiB against 22 G free.

**Action for the owner:** take one off-machine copy of
`checkpoint/e3_boundary_refinement_0062/` (1.5 GiB), after which the experiment source can
be deleted safely.

## 5. Cleanup

Three groups, each with paths, byte totals and proof of non-reference printed before
deletion. Every candidate was checked against every `.py`/`.yaml`/`.json`/`.sh`/`.toml`
file under `configs/`, `src/`, `scripts/` and `tests/`; **all returned zero references**.

| Group | Paths | Bytes | GiB |
| --- | ---: | ---: | ---: |
| A — resume-only state (`latest.pt` × 4) | 4 | 6,462,187,388 | 6.02 |
| B — non-selected recipe weights (R1, R2, confirmation seed) | 3 | 4,846,540,293 | 4.51 |
| C — GLiNER + companion tokenizer | 20 | 1,160,141,669 | 1.08 |
| **Total reclaimed** | **27** | **12,468,961,280** | **11.61** |

Reasons, recorded per group:

* **A** — optimizer and scheduler state exists only to resume. The R0/R1/R2/R3 comparison
  is complete and no run will be resumed; each run's metrics survive in its retained
  `experiment_manifest.json` and `training_history.jsonl`.
* **B** — R1 and R2 are non-selected recipes and the seed-2 run is a confirmation artifact
  whose purpose (satisfying the gate's seed criterion at +9.89 F1) is discharged. None is
  referenced by active or rollback config; all metrics are preserved in Audit 0062 §7–8
  and in the retained manifests; each is reproducible from the tracked config, script and
  seed in about 5.5 minutes of GPU time.
* **C** — GLiNER remains `EXCLUDED_BY_ABLATION` (Audit 0061) with `enable_e6_gliner: false`,
  is referenced by no active config, and is reacquirable from its pinned immutable revision
  `443d26d654e0324125a96bebd8e796c14ff2efe6` through the governed acquisition tool. This
  was the **project-owned revision-addressed copy** under `models/mention/`, not a shared
  cache.

**The global Hugging Face cache was not touched.** `~/.cache/huggingface` measured 936 M
before and 936 M after, and still contains `models--microsoft--mdeberta-v3-base` and
`models--demdecuong--vihealthbert-base-word`, which makes reacquisition cheaper still.

Nothing was deleted for looking old. The scored submission evidence, all three historical
baseline ZIPs and the dotted probe were retained untouched.

### Retained, with reasons

| Path | Why |
| --- | --- |
| `checkpoint/e3_boundary_refinement_0062/` | the active artifact and its reproducibility contract |
| `checkpoint/experiments/.../R3_alpha050/best.pt` | `NEEDS_OWNER_BACKUP` — see §4 |
| `checkpoint/s1_mention_full_training_v1/best.pt` | deterministic rollback target |
| `checkpoint/experiments/*/experiment_manifest.json`, `training_history.jsonl` | complete metric and provenance record for all four runs, kilobytes not gigabytes |
| `runs/e3_boundary_v2_final_dotted/` | the scored 11.9188 submission and its manifests |
| `runs/baseline_v0_profile_{a,b,c}/`, `runs/probes/` | hash-pinned scored leaderboard evidence |
| `runs/diagnostics/`, `reports/` | evidence behind audit numbers |
| `indices/` (809 M) | active runtime KB, referenced by the pipeline config |

## 6. Post-Cleanup Integrity

All twelve required conditions pass.

| Condition | Result | Observed |
| --- | --- | --- |
| active checkpoint exists | PASS | `checkpoint/e3_boundary_refinement_0062/best.pt` |
| active checkpoint hash matches | PASS | `524ece1e…dde3a` |
| rollback checkpoint exists | PASS | `checkpoint/s1_mention_full_training_v1/best.pt` |
| rollback checkpoint hash matches | PASS | `a64cc173…1017c` |
| active profile resolves canonical path | PASS | profile == pipeline config |
| rollback profile resolves original | PASS | unchanged |
| no rejected checkpoint is active | PASS | all 3 removed |
| E4 remains absent | PASS | 3 `.pt` files, 0 E4 |
| GLiNER remains disabled | PASS | `enable_e6_gliner: false` |
| parameter budget below 9B | PASS | 135,004,814 |
| scored submission ZIP hash unchanged | PASS | `eae8a348…4c05` |
| `git diff --check` | PASS | clean |

Remaining weight files (3):

```text
1,615,513,431  checkpoint/e3_boundary_refinement_0062/best.pt          ACTIVE
1,615,513,431  checkpoint/experiments/…/R3_alpha050/best.pt            NEEDS_OWNER_BACKUP
1,615,513,303  checkpoint/s1_mention_full_training_v1/best.pt          ROLLBACK
```

| | Before | After | Change |
| --- | ---: | ---: | ---: |
| Disk used | 90 G (90%) | 79 G (79%) | −11 G |
| Disk free | 11 G | **22 G** | **+11 G** |
| `checkpoint/` | 16 G | 4.6 G | −11.4 G |
| `models/` | 1.1 G | 4.0 K | −1.1 G |

## 7. Candidate-Linking Root-Cause Audit

### 7.1 The measurement the brief asks for cannot be computed — and why

The brief requires gold-code Recall@1/@2/@5/@10, MRR, top-1 accuracy and gold-code rank.
**No ICD or RxNorm gold codes exist anywhere in this project.** This was checked
exhaustively, not assumed:

| Source | Finding |
| --- | --- |
| `manifests/annotation_coverage.json` | all four governed sources declare `icd_candidates: false` and `rxnorm_candidates: false` |
| governed validation entities | keys are `start, end, text, target_type, source_label, mapping_status, mapping_confidence, provenance, quality_flags` — **no code field** |
| `data/dev_gold/`, `data/dev_silver/` | empty (`.gitkeep` only) |
| organizer archives | `input.zip` / `input_turn2_vong1.zip` contain 100 `.txt` files each — inputs, no labels |

Rank-based retrieval metrics are therefore **undefined**, not skipped: there is no
denominator. Reporting a Recall@k here would require inventing gold codes, and the brief
forbids using public organizer output as pseudo-gold.

This directly triggers the brief's own decision rule: *"too few mention-matched samples:
do not claim the linker is the sole bottleneck."* Here the count is zero.

### 7.2 What was measured instead

The linkers were run on **gold mention spans and types** from the governed validation
split — 1,302 DIAGNOSIS and 156 MEDICATION mentions. Feeding gold mentions is what the
brief demands ("do not mix mention-extraction failures into retrieval recall"): every
empty candidate set below is unambiguously a linker or KB failure, with zero contribution
from E3. No model was loaded; only the KB indices and the deterministic L5B/L5C linkers ran.

Detail: `runs/diagnostics/0063_linker/linker_audit.json`.

### 7.3 ICD-10 / DIAGNOSIS

| Measure | Value |
| --- | ---: |
| KB searchable records | 15,308 |
| KB normalized alias strings | 37,362 |
| Gold mentions linked | 1,302 |
| **Empty candidate set** | **70 (5.4%)** |
| At least one candidate | 1,232 (94.6%) |
| **Surface present verbatim in the KB alias table** | **114 (8.8%)** |
| …of those, still empty | 0 |
| …of those, not ranked top-1 | 0 |
| top-1 score median | 3,105.0 |
| top1−top2 margin median | 16.66 |
| exact ranking ties | 267/1,199 (22.3%) |
| margin ≤ 0.01 | 483/1,199 (40.3%) |

Retrieval channel that produced the offered set:

| Channel | Count |
| --- | ---: |
| char_ngram | 1,066 |
| sparse | 1,066 |
| exact | 114 |
| exact_ascii | 114 |

Tier the top-1 came from:

| Tier | Count | Share |
| --- | ---: | ---: |
| supported_specific | 950 | 77.1% |
| exact_name | 114 | 9.3% |
| lexical | 88 | 7.1% |
| broader | 80 | 6.5% |

Depth / specificity of the top-1 code:

| | Count | Share |
| --- | ---: | ---: |
| 3-char category only | 105 | 8.5% |
| 4-char | 1,088 | 88.3% |
| 5-char | 39 | 3.2% |
| 4-char ending `.8`/`.9` ("other"/"unspecified") | 460 | 37.3% |
| **vague-or-unspecified top-1** | **565** | **45.9%** |

Empty rate by surface length — the failures are concentrated in short surfaces:

| Words | n | empty | rate |
| --- | ---: | ---: | ---: |
| 1 | 37 | 19 | **51.4%** |
| 2–3 | 631 | 39 | 6.2% |
| 4–7 | 620 | 12 | 1.9% |
| 8+ | 14 | 0 | 0.0% |

**Reading.** Retrieval coverage is not the headline problem: 94.6% of gold diagnoses get
*something*. Two real weaknesses show up instead:

1. **Vietnamese synonym coverage is thin.** Only 8.8% of gold surfaces exist verbatim in
   the alias table; 82% of successes come from character-n-gram/sparse fuzzy matching,
   which finds *a* code with no semantic guarantee. Where an exact alias does exist the
   linker is flawless — 114/114 retrieved and 114/114 ranked first — so the exact channel
   is not broken, it is starved.
2. **Almost half the top-1 codes are vague** (45.9% category-only or `.8`/`.9`), which is
   exactly the specificity-controller behaviour spec §9.3 describes, and 40.3% of ranking
   decisions are effectively undetermined (margin ≤ 0.01).

### 7.4 RxNorm / MEDICATION

| Measure | Value |
| --- | ---: |
| KB searchable records | 211,949 |
| KB normalized alias strings | 151,861 |
| Gold mentions linked | 156 |
| **Empty candidate set** | **0 (0.0%)** |
| Sets saturating the 20-candidate bound | 130 (83.3%) |
| Surface present verbatim in the KB alias table | 15 (9.6%) |
| …present but **not** ranked top-1 | **4** |
| top1−top2 margin median | **0.0010** |
| exact ranking ties | **72/156 (46.2%)** |
| margin ≤ 0.01 | 101/156 (64.7%) |

TTY and tier of the top-1 concept:

| TTY | Count | Share | | Tier | Count | Share |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| IN | 94 | 60.3% | | ingredient_only | 107 | 68.6% |
| SCD | 48 | 30.8% | | lexical | 49 | 31.4% |
| MIN | 12 | 7.7% | | | | |
| PIN | 1 | 0.6% | | | | |
| SBD | 1 | 0.6% | | | | |

Governed INN bridge: **0** mentions in this population triggered a bridge note; all 156
top-1 concepts carry a graph path.

**Reading.** RxNorm never fails to retrieve — it fails to *decide*. 83.3% of sets are
saturated at the candidate bound and **46.2% of top-1 choices are exact score ties**, so
under `competition_top1` the emitted code is close to arbitrary among equals. Four of the
fifteen verbatim alias matches are outranked, which is a pure ranking defect. The TTY mix
is ingredient-dominated (IN 60.3%), meaning strength/dose-form evidence is largely not
being used to discriminate SCD candidates.

### 7.5 Error distribution against the brief's taxonomy

Categories requiring a gold code — `GOLD_CODE_NOT_IN_KB`, `GOLD_CODE_NOT_RETRIEVED`,
`GOLD_CODE_RETRIEVED_WRONG_RANK`, `AMBIGUOUS_GOLD` — are **NOT_ENOUGH_EVIDENCE** for the
reason in §7.1. What the evidence does support:

| Category | ICD | RxNorm | Basis |
| --- | ---: | ---: | --- |
| `MENTION_NOT_MATCHED` | n/a (gold mentions used by design) | n/a | isolated deliberately |
| `ICD_SYNONYM_OR_LANGUAGE_ERROR` | **1,188 / 1,302 (91.2%)** | — | surface absent from the alias table; matched only fuzzily or not at all |
| `ICD_HIERARCHY_OR_SPECIFICITY_ERROR` | **565 / 1,302 (45.9%)** | — | top-1 is category-only or `.8`/`.9` unspecified |
| `GOLD_CODE_NOT_RETRIEVED` (proxy: empty set) | 70 (5.4%) | 0 (0.0%) | linker returned nothing |
| `RXNORM_TTY_ERROR` | — | **107 / 156 (68.6%)** | resolved to ingredient-only where an SCD may be required |
| `RXNORM_STRENGTH_OR_FORM_ERROR` | — | 101 / 156 (64.7%) margin ≤ 0.01 | structured evidence not discriminating |
| `RXNORM_LOCAL_INN_BRIDGE_ERROR` | — | 0 | no bridge fired |
| `OUTPUT_SERIALIZATION_ONLY` | 0 | 0 | dotted transform verified in Audits 0061/0062 |
| `GOLD_CODE_NOT_IN_KB` / `…WRONG_RANK` / `AMBIGUOUS_GOLD` | NOT_ENOUGH_EVIDENCE | NOT_ENOUGH_EVIDENCE | no gold codes exist |

### 7.6 Retrieval versus ranking

* **ICD** — retrieval works (94.6% non-empty) but is *lexically shallow* (8.8% true alias
  coverage) and *imprecise* (45.9% vague top-1, 40.3% near-ties). Primary weakness:
  **synonym coverage**, secondary: ranking/specificity.
* **RxNorm** — retrieval is complete (0% empty, 83.3% saturated). Primary weakness:
  **ranking**, decisively — 46.2% exact ties.

Neither can be repaired by a learned model in the next milestone, because **the
supervision required to train or validate one does not exist**.

### 7.7 The scope question, recorded

69.7% of the scored run's entities (799 of 1,147: 627 SYMPTOM, 137 TEST_RESULT, 35
TEST_NAME) receive no candidates *by design*. That policy is recorded in Audit 0059 as
"no candidates on SYMPTOM / TEST_NAME / TEST_RESULT". Spec §7.3 only forbids the
cross-links (ICD on MEDICATION, RxNorm on DIAGNOSIS); it does **not** state that symptoms
carry none, and ICD-10 chapter XVIII (R00–R99) exists precisely for symptoms and signs.

This is flagged, not acted on. Resolving it requires organizer evidence this milestone may
not use, and getting it wrong is symmetric: under Jaccard, emitting candidates against an
empty gold list scores zero just as surely as omitting them against a populated one. It is
carried as the top open question in §10.

## 8. Next Intervention

**Selected: `MENTION_LAYER_STILL_BLOCKING`.**

This is chosen on measurement, not by default, and it is the brief's own rule for the
situation the data creates.

**The evidence.** §2 is decisive: between two scored submissions whose KB, linkers,
assertion logic and decoder were byte-identical, changing only the E3 mention checkpoint
moved **every** component — WER −4.3886, J_assertion +2.5953, **J_candidates +0.6250**.
The candidate score rose by a quarter of a point *without a single change to retrieval or
ranking*. Mention quality demonstrably gates the candidate component.

Supporting facts:

1. **No code supervision exists** (§7.1). `DENSE_RETRIEVER`, `LEARNED_RERANKER` and
   `HYBRID_RETRIEVER_PLUS_RERANKER` all require gold codes to train against and to
   validate. Selecting one would mean committing to a model that cannot be measured — the
   failure mode Audit 0061 already paid for once with GLiNER.
2. **The linker is not failing to retrieve** — 94.6% ICD and 100% RxNorm non-empty on gold
   mentions.
3. **Mention headroom remains large.** Validation micro F1 is 0.6799 against a measured
   label-projection ceiling of 0.9713 (Audit 0062 §4.1) — 29 F1 points still reachable
   inside the current architecture. WER is still 82.71, so the 30%-weighted text component
   sits at 17.29/100.

**Runner-up, and why it lost.** `KB_SYNONYM_REPAIR_FIRST` is the strongest linker-side
option and the only one needing no supervision — 8.8% verbatim alias coverage is a real,
fixable defect, and where an alias exists the linker is already perfect (114/114 top-1).
It loses this round only because its payoff is bounded by the same gate: candidates score
through matched entities, and 29 F1 points of mention headroom sit upstream of every code
the KB could offer. It should be the milestone after next, and it needs no new model.

### Specification of the selected intervention

| Field | Value |
| --- | --- |
| **Exact failure addressed** | Mention span/type quality, which gates all three scored components (§2). Concretely: validation F1 0.6799 vs oracle 0.9713; 236 gold entities still never proposed; 298 boundary errors; 8+ word spans at 7.1% exact |
| **ICD/RxNorm scope** | Indirect — improves the matched-entity population both linkers operate on. No change to retrieval or ranking |
| **Proposed model ID** | **None. No new model.** The intervention is further supervised refinement of the existing `demdecuong/vihealthbert-base-word` @ `f89e80b461e86f9cfc1c84019bd819830c24b6c5` head |
| **Immutable revision strategy** | Unchanged pinned revision; new checkpoints hash-pinned and declared in `e3_checkpoint_profiles.yaml` exactly as Audit 0063 does |
| **License** | Unchanged (existing approved base model) |
| **Parameter count estimate** | 135,002,117 — **unchanged**; refinement alters weights, never architecture |
| **Training supervision source** | Governed training split only (`892dc22d…`, 23,799 supervised examples). No public organizer input, no pseudo-labels |
| **Negative-sampling strategy** | Not applicable to a token classifier. The analogous lever is the class-weighting axis Audit 0062 opened: α = 0.50 is a measured midpoint, not a proven optimum, and boundary-token reweighting (spec §7.1) is untried |
| **Validation metrics** | Exact character-span + type micro P/R/F1 through `evaluation/exact_mention.py`, plus boundary-error counts and offset violations — the Audit-0062 §8 gate, unchanged and not weakened |
| **Local RTX 4060 feasibility** | **Proven.** Four runs completed in 21.7 min total, 2.929 GiB peak VRAM, 5.4 min each |
| **Colab feasibility** | Supported and unnecessary; `scripts/run_e3_refinement_0062.sh` carries L4/A100 commands |
| **Integration stage** | L3-E3 weights only, through the existing profile registry; L4–L9 untouched |
| **Rollback plan** | Two hash-pinned profiles already exist; rollback is a two-line config edit to `s1_full_training_v1`, verified by test |
| **Expected 9B budget impact** | **Zero.** Deployment ledger stays at 135,004,814 |

Nothing was downloaded and nothing was trained in this milestone.

## 9. Tests and Static Validation

```text
targeted   tests/unit/test_e3_checkpoint_profiles_0062.py           17 passed
targeted   e4_retirement + e3_profiles + e3_canonical_integration  117 passed
bounded    canonical-path load + digest agreement                   PASS
full (1)   pytest tests/ -q     1 failed, 1969 passed, 2 skipped in 500.41s
full (2)   pytest tests/ -q            1970 passed, 2 skipped in 484.08s
ruff check .                                                        All checks passed!
ruff format --check tests/unit/test_e4_retirement.py                added lines clean
mypy src/mednorm_vi                     Success: no issues found in 291 source files
compileall -q src tests scripts                                     OK
git diff --check                                                    OK
```

**The first full run failed, and it failed correctly.**
`test_e4_retirement::test_no_e4_checkpoint_remains_on_disk` — written in Audit 0062 to
require every `.pt` to sit in a known governed location — rejected the new canonical
directory created in §4. That is precisely the job it exists to do: a weight file in an
undeclared location should stop the suite.

The fix does not add a third hardcoded path. The governed locations are now **derived from
the checkpoint-profile registry**, so declaring a checkpoint in config is what makes it
legitimate and an undeclared one still fails. This was verified by negative control: a
planted `checkpoint/stray_unmanaged/best.pt` fails the test, and passes again once removed.
The guarantee is strictly stronger than before.

After the second run started, whitespace-only formatting was applied to that one test file
to match `ruff format` on the added lines; the file was re-verified on its own (88 passed).
`tests/unit/test_e4_retirement.py` is not `ruff format`-clean at `HEAD` and was left that
way — only the lines this milestone touched were written in the formatter's style, since
repository-wide formatting is out of scope.

The `GLiNER not acquired locally` skip is new and expected: it is the §5 cleanup, and the
test's `skipif` guard correctly detected the absence rather than failing.

No `src/` file changed, which is why the mypy surface stays at 291 files. The full suite
was justified because the active checkpoint path in `configs/pipeline/full_v1.yaml` is read
by tests.

## 10. Remaining Risks

1. **The active checkpoint has no off-machine backup.** Both local copies are on one
   filesystem. This is the highest-severity item and needs one owner action (§4).
2. **No ICD/RxNorm supervision exists.** Until some is created — manual gold on a sample,
   or organizer clarification — no learned linker can be trained *or honestly evaluated*.
   This blocks three of the six intervention options outright.
3. **Symptoms receive no candidates** (§7.7). 55% of emitted entities. Whether that is
   correct is unresolved and is worth one governed leaderboard experiment.
4. **RxNorm top-1 is close to arbitrary** — 46.2% exact ties under a top-1 decoder.
5. **ICD alias coverage is 8.8%**; 82% of matches are fuzzy, so a plausible-looking code
   may carry no real evidence.
6. **45.9% of ICD top-1 codes are vague**, which may be correct behaviour under an absent
   specificity signal, or may be systematically forfeiting precision. Undecidable without
   gold codes.
7. **Validation is still not iid with training** (Audit 0062 §4.6) — unchanged here.
8. **GLiNER weights are gone locally.** Reacquisition needs network access and the pinned
   revision; the record in `candidate_model_registry.yaml` is intact.

## 11. Changed-File Inventory

New:

```text
docs/audits/0063-post-score-freeze-artifact-hygiene-and-linker-diagnosis.md
```

Modified (tracked):

```text
configs/models/e3_checkpoint_profiles.yaml      active profile -> canonical path; scored result
configs/pipeline/full_v1.yaml                   e3_checkpoint_path -> canonical path
configs/models/candidate_model_registry.yaml    active_checkpoint_path + digest recorded
tests/unit/test_e4_retirement.py                governed weight locations derived from the
                                                profile registry instead of hardcoded
```

New untracked artifacts (ignored): `checkpoint/e3_boundary_refinement_0062/`,
`runs/diagnostics/0063_inventory/`, `runs/diagnostics/0063_linker/`.

Deleted (untracked generated artifacts only, 11.61 GiB): 4 × `latest.pt`, 3 × non-selected
`best.pt`, and the project-owned GLiNER/mdeberta copies under `models/mention/`.

**No `src/` file was modified.** Audit 0062 and all earlier audits are untouched. The
architecture PDF is unmodified and was not moved.

## 12. Verdict

**`ARTIFACTS_FROZEN_LINKER_DIAGNOSED`**

The scored checkpoint is frozen at a canonical, hash-pinned, fully documented location
with a working two-line rollback; 11.61 GiB was reclaimed with zero live references
touched; all twelve integrity conditions pass. The linker was diagnosed as far as the data
permits, and the honest limit of that diagnosis — no gold codes exist — is recorded as a
finding rather than papered over with an invented metric.
