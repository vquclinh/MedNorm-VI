# Audit 0064: Source-Balanced and Boundary-Weighted E3 Refinement

Milestone: 4E - Source-balanced and boundary-weighted E3 refinement, strict model selection
Date: 2026-08-01
Verdict: `E3_4E_NOT_ACCEPTED`

## 1. Initial Git and Safety State

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `3ee22b9 chore: freeze the scored E3 checkpoint and clean local artifacts` |
| Working tree | clean |
| Staged files | none |
| `origin/main...HEAD` | `0 0` |
| Audit 0063 committed | yes |
| RAM | 14 GiB total, 8.3 GiB available |
| Disk (start) | 22 G free (79% used) |
| GPU | RTX 4060 Laptop, 8,188 MiB, 13 MiB used |
| Containers | 0 |

Digests verified before any work:

```text
docs/MedNorm-VI_Architecture.pdf                 0d5eaa20…81e09b   unchanged
checkpoint/e3_boundary_refinement_0062/best.pt   524ece1e…dde3a    matches
checkpoint/s1_mention_full_training_v1/best.pt   a64cc173…1017c    matches
```

## 2. Duplicate Cleanup

The owner confirmed a hash-verified off-machine backup of
`checkpoint/e3_boundary_refinement_0062/`, releasing the `NEEDS_OWNER_BACKUP` hold from
Audit 0063 §4.

Preconditions checked before deleting
`checkpoint/experiments/0062_e3_boundary_refinement/R3_alpha050/best.pt`:

| Check | Result |
| --- | --- |
| canonical checkpoint exists | yes |
| canonical digest matches `524ece1e…dde3a` | yes |
| reproducibility metadata present in canonical dir | 6 files |
| R3 experiment provenance retained | manifest + history |

**Reclaimed: 1,615,513,431 bytes (1.50 GiB).** Disk 79% → 77%.

**A process fault, recorded rather than smoothed over.** The non-reference check printed
`YES — ABORT`, and the shell script had no `exit`, so the deletion proceeded anyway. The
outcome was nevertheless correct, verified afterwards: the only match is
`experiment_origin` in `e3_checkpoint_profiles.yaml`, a **provenance field** recording
where the checkpoint came from, not a load path. The active path resolves to the canonical
file, and the runtime was re-verified end to end after the deletion (readiness true, E3
executed, recorded digest equal to the configured digest, 105 targeted tests green). The
guard was too coarse — a substring grep cannot distinguish a load path from a
documentation field — and it did not stop the run. Both are worth fixing before the next
cleanup.

## 3. Reproduced Active Baseline

**Exact. No tolerance required.**

```text
micro span+type   P=0.6546  R=0.7072  F1=0.6799   TP=1408  FP=743  FN=583
exact SPAN ONLY   P=0.6774  R=0.7318  F1=0.7035
  DIAGNOSIS       P=0.7073  R=0.7312  F1=0.7190
  SYMPTOM         P=0.5359  R=0.6585  F1=0.5909
  MEDICATION      P=0.7895  R=0.6731  F1=0.7266
  TEST_NAME       0 gold, 3 FP        TEST_RESULT  0 gold, 14 FP

boundary errors 298 (right 192, left 91, both 15)
missed 236 | spurious 387 | wrong_type 49 | offset violations 0
predictions 2151 | empty documents 54
deterministic repeat agreement 120/120
```

Every target figure matches: P 0.6546, R 0.7072, F1 0.6799, boundary 298, offsets 0.

### 3.1 Source distribution — the finding that shaped R5

| Source | Train (supervised) | Entity types in that source | Validation |
| --- | ---: | --- | ---: |
| vietmed_ner | 9,267 (38.9%) | **MEDICATION 100%** | **0 examples** |
| vimq | 8,736 (36.7%) | **MEDICATION 100%** | 139 ex / 156 entities |
| vimedner | 5,796 (**24.4%**) | DIAGNOSIS 71% / SYMPTOM 29% | 906 ex / **1,835 entities** |
| phoner_covid19 | excluded (10,027, unsupervised) | — | — |

**The governed sources are type-disjoint.** All DIAGNOSIS and SYMPTOM supervision comes
from a single source holding 24.4% of training, while those two types are **92.2% of
validation entities**. MEDICATION receives 75.6% of training for 7.8% of validation
entities.

In this corpus, therefore, *source balancing is type rebalancing*. The audit states that
plainly rather than letting the label "source-balanced" imply something narrower.

Per-source and per-length behaviour of the active checkpoint:

| Source | n | gold | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vimedner | 906 | 1,835 | 0.6460 | 0.7101 | **0.6765** ← worst |
| vimq | 139 | 156 | 0.7836 | 0.6731 | 0.7241 |

| Gold length | gold | exact | rate |
| --- | ---: | ---: | ---: |
| 1 word | 132 | 84 | 63.6% |
| 2–3 words | 1,065 | 793 | 74.5% |
| 4–7 words | 752 | 521 | 69.3% |
| 8+ words | 42 | 10 | **23.8%** |

Mean gold length is 3.4 words in vimedner and 2.1–2.3 in the medication sources, and the
8+ word population lives almost entirely in vimedner (195 of 796 in training, 42 in
validation) — so long-span failure and worst-source failure are the same population.

## 4. Recipe Definitions

All recipes continue the **active** Audit-0062 checkpoint (`524ece1e…`, scored 11.9188),
preserve architecture, label vocabulary, alignment and decoding, keep `focal_alpha` at the
scored 0.50, and use only governed training gold. Each differs from the control by exactly
one switch.

| Recipe | Change | Rationale |
| --- | --- | --- |
| **R0** | none (control) | the active checkpoint |
| **R4a** | boundary weight 2.0 | 298 boundary errors remain, right outnumbering left 2:1 |
| **R4b** | boundary weight 3.0 | the second of one small predefined set |
| **R5** | source-balanced sampling | §3.1: the types that are 92.2% of validation get 24.4% of training |
| **R6** | boundary 2.0 **+** source-balanced | conditional; run because R4 and R5 each showed a beneficial, interpretable effect |

**Boundary weighting** multiplies the loss on tokens at the first or last position of a
gold run. The mask is derived from the labels themselves, so it cannot drift from the
supervision it weights, and it never marks a token the labels call negative (tested). The
normalizer is unchanged, so the intervention alters the gradient's *shape*, not its
magnitude.

**Source-balanced sampling** draws each epoch uniformly across governed sources, keeping
epoch length at 743 optimizer steps and effective batch at 32 so the schedule stays
comparable to Audit 0062. Small sources are oversampled rather than large ones truncated —
no governed example is discarded. It is uniform **by source**, deliberately *not* matched
to the validation mixture: matching validation would be fitting the evaluation set, which
Audit 0062 §4.6 declined to do.

Common hyperparameters: 2 epochs, early-stopping patience 1, LR 1e-5 / head 3e-5, batch
8 × 4 accumulation, weight decay 0.01, warmup 0.06, clip 1.0, max length 256, decision
threshold 0.5, 23,799 supervised examples, 743 steps/epoch, development seed 20260801.

## 5. Metrics

All measured through one evaluator on the L1→L4 path, the same instrument as Audit 0062
plus per-source and worst-source figures.

| Recipe | P | R | **F1** | ΔF1 | boundary | worst-src F1 | DIAGNOSIS | SYMPTOM | MEDICATION | missed | spurious |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 control | 0.6546 | 0.7072 | 0.6799 | — | 298 | 0.6765 | 0.7190 | 0.5909 | 0.7266 | 236 | 387 |
| R4a (w 2.0) | 0.6430 | 0.7338 | 0.6854 | +0.56 | 288 | 0.6833 | 0.7346 | 0.5795 | 0.7157 | 197 | 459 |
| R4b (w 3.0) | 0.6257 | 0.7363 | 0.6765 | −0.34 | 302 | 0.6746 | 0.7336 | 0.5564 | 0.7020 | 178 | 513 |
| **R5** | **0.6773** | 0.7263 | **0.7009** | **+2.11** | **280** | **0.6997** | **0.7524** | 0.5915 | 0.7192 | 228 | **361** |
| R5 seed 2 | 0.6630 | 0.7182 | 0.6895 | +0.96 | 294 | 0.6886 | 0.7407 | 0.5783 | 0.7059 | 228 | 388 |
| R6 combined | 0.6560 | 0.7338 | 0.6927 | +1.29 | 282 | 0.6912 | 0.7405 | 0.5893 | 0.7205 | 205 | 427 |

Every recipe: **0 offset violations**, **120/120 deterministic repeat agreement**, no
entity-type collapse, 135,002,117 parameters. TEST_NAME and TEST_RESULT remain 0 gold with
3 and 13–14 false positives respectively across all arms — unchanged, and owned by E2.

Long-span (8+ words) exact rate: R0 23.8% → R4a 33.3%, R4b 38.1%, R5 35.7%, R6 35.7%.
Every intervention helps the population that was worst; none of them helps enough.

### What the two levers actually did

* **Boundary weighting works mechanically and costs more than it earns.** w = 2.0 reduced
  boundary errors 298 → 288 and cut missed entities 236 → 197 — exactly its design intent
  — but precision fell 1.15 points and both SYMPTOM and MEDICATION regressed. w = 3.0
  degraded monotonically further (precision −2.89, boundary errors *up* to 302). The
  mechanism is real and the dose-response is clean; the trade is simply unfavourable.
* **Source balancing was the only intervention to improve precision and recall together**
  (+2.27 P, +1.91 R), and it produced the best boundary count (280), the best worst-source
  F1 (0.6997) and the fewest spurious predictions (361). On the development seed it passed
  every criterion.
* **The combination was worse than R5 alone** (0.6927 vs 0.7009). R4's precision cost
  survives the combination, so the two levers are not additive.

## 6. Acceptance Gate

| Criterion | Required | R4a | R4b | R5 (dev) | R6 | **R5 seed 2** |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| micro-F1 | ≥ +2.0 | +0.56 ❌ | −0.34 ❌ | **+2.11 ✅** | +1.29 ❌ | +0.96 |
| precision | drop ≤ 0.5 | −1.15 ❌ | −2.89 ❌ | **+2.27 ✅** | +0.15 ✅ | +0.84 |
| recall | must not decrease | +2.66 ✅ | +2.91 ✅ | **+1.91 ✅** | +2.66 ✅ | +1.10 |
| DIAGNOSIS F1 | must not decrease | +1.56 ✅ | +1.45 ✅ | **+3.33 ✅** | +2.15 ✅ | +2.17 |
| SYMPTOM F1 | must not decrease | −1.14 ❌ | −3.45 ❌ | **+0.06 ✅** | −0.16 ❌ | −1.26 |
| MEDICATION F1 | drop ≤ 1.0 | −1.09 ❌ | −2.47 ❌ | **−0.75 ✅** | −0.61 ✅ | −2.08 |
| boundary errors | < 298 | 288 ✅ | 302 ❌ | **280 ✅** | 282 ✅ | 294 |
| worst-source F1 | drop ≤ 1.0 | +0.00 ✅ | +0.00 ✅ | **+0.00 ✅** | +0.00 ✅ | +0.00 |
| offset violations | 0 | 0 ✅ | 0 ✅ | **0 ✅** | 0 ✅ | 0 |
| type collapse | none | none ✅ | none ✅ | **none ✅** | none ✅ | none |
| parameter budget | < 9B | ✅ | ✅ | **✅** | ✅ | ✅ |
| **confirmation seed** | **≥ +1.5 F1** | — | — | **+0.96 ❌** | — | — |
| L4–L9 contract | unchanged | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Result** | | NOT ACCEPTED | NOT ACCEPTED | **NOT ACCEPTED** | NOT ACCEPTED | |

### The confirmation seed is why nothing ships

R5 passed all eleven development-seed criteria. Its independent confirmation seed
(20260802, identical in every other respect) returned **+0.96 F1** against a required
**+1.5**, and independently failed SYMPTOM (−1.26) and MEDICATION (−2.08).

| | seed 20260801 | seed 20260802 | spread |
| --- | ---: | ---: | ---: |
| F1 | 0.7009 (+2.11) | 0.6895 (+0.96) | **1.15 points** |
| precision | 0.6773 | 0.6630 | 0.0143 |
| boundary errors | 280 | 294 | 14 |
| SYMPTOM F1 | 0.5915 | 0.5783 | 0.0132 |

**More than half the headline gain is seed variance.** A 1.15-point spread on a claimed
2.11-point effect is not a result; it is noise with a favourable draw. For contrast, the
Audit-0062 selection reproduced within 0.40 points across seeds — which is why that one
was accepted and this one is not.

The gate was not weakened, not re-parameterised, and no criterion was reinterpreted after
seeing the number. R5 is the most promising direction this project has for the mention
layer, and it still does not ship, because a single-seat improvement that does not
replicate would be spent on a leaderboard attempt that cannot be taken back.

## 7. Selected Checkpoint

**None.** `checkpoint/e3_boundary_refinement_0062/best.pt` (`524ece1e…dde3a`) remains the
active checkpoint, unmodified, and `checkpoint/s1_mention_full_training_v1/best.pt`
(`a64cc173…1017c`) remains the historical rollback. No config, profile, registry or
runtime file was changed.

## 8. Resources

| Run | wall | peak VRAM | steps/epoch | features |
| --- | ---: | ---: | ---: | ---: |
| R4a | 323.4 s | 2.927 GiB | 743 | 23,799 |
| R4b | 320.2 s | 2.927 GiB | 743 | 23,799 |
| R5 | 333.4 s | 2.935 GiB | 743 | 23,799 |
| R5 seed 2 | 336.7 s | 2.927 GiB | 743 | 23,799 |
| R6 | 317.1 s | 2.935 GiB | 743 | 23,799 |

Total training **26.9 minutes** on the local RTX 4060; peak RSS under 4 GiB; one GPU
process at a time throughout. Evaluation is ~85 s per recipe (1,045 documents plus a
120-document determinism repeat).

## 9. Artifact Hygiene

Provenance preserved for every run before any deletion: `experiment_manifest.json` and
`training_history.jsonl` in each run directory, plus the six full gate reports and
per-example predictions under `runs/diagnostics/0064_recipes/`.

All ten weight files were checked against `configs/` and `src/`; **every one returned zero
references**.

| Group | Files | Bytes |
| --- | ---: | ---: |
| rejected `best.pt` (R4a, R4b, R5, R5 seed 2, R6) | 5 | 8,077,567,155 |
| `latest.pt` resume states (no resume planned) | 5 | 8,077,734,235 |
| Audit-0063 duplicate (§2) | 1 | 1,615,513,431 |
| **Total reclaimed this milestone** | **11** | **≈ 15.15 GiB** |

Measured free-space recovery: 1.50 GiB (§2) + **13.54 GiB** (§9) — the difference from the
raw byte sum is filesystem overhead.

| | Before | After |
| --- | ---: | ---: |
| Disk used | 89% | **79%** |
| Disk free | 12 G | **22 G** |

Remaining weight files — exactly two, both protected:

```text
1,615,513,431  checkpoint/e3_boundary_refinement_0062/best.pt      ACTIVE   524ece1e…
1,615,513,303  checkpoint/s1_mention_full_training_v1/best.pt      ROLLBACK a64cc173…
```

Scored submission ZIP unchanged: `eae8a348…4c05`.

Every deleted checkpoint is reproducible from the tracked config, script and seed in about
5.5 minutes of GPU time.

## 10. Integration

**None performed**, as the gate requires. No candidate passed, so:

* the active profile, pipeline config and model registry are untouched;
* dotted ICD serialization, `competition_top1`, the candidate linker, KB data and L4–L9
  ownership are unchanged;
* GLiNER remains disabled and `EXCLUDED_BY_ABLATION`;
* the deployment ledger stays at 135,004,814.

The two new training levers are nevertheless covered by
`tests/unit/test_e3_source_boundary_levers_0064.py` (13 tests). An unaccepted *result* is
not a reason to ship an untested *sampler*: the next milestone will reuse both functions,
and one of them decides which examples the model sees.

## 11. Validation

```text
targeted   tests/unit/test_e3_source_boundary_levers_0064.py        13 passed
targeted   e3 profiles + e4 retirement (post-deletion)             105 passed
full       pytest tests/ -q                    1983 passed, 2 skipped in 566.80s
ruff check .                                                       All checks passed!
ruff format --check <3 changed Python files>                       3 files already formatted
mypy src/mednorm_vi                    Success: no issues found in 291 source files
compileall -q src tests scripts                                    OK
git diff --check                                                   OK
bash -n scripts/run_e3_source_boundary_0064.sh                     OK
```

No `src/` file changed, so the mypy surface stays at 291 files.

**A tooling fault worth recording:** the first attempt to launch R4a and R4b failed with
`unrecognized arguments: --boundary-weight 2.0`. The interactive shell is **zsh**, which
does not word-split unquoted variables, so `$ARGS` reached argparse as a single token. R5
was unaffected because `--source-balanced` is one word. The runs were relaunched with
literal flags; the shipped `scripts/run_e3_source_boundary_0064.sh` uses literal flags per
recipe and is `bash -n` clean.

## 12. Public Run

**Not run.** The acceptance gate did not pass, and §9 of the brief permits public organizer
inference only after it does. `runs/e3_source_boundary_v3_final_dotted/` does not exist and
no ZIP was produced. The leaderboard attempt is preserved.

## 13. Changed-File Inventory

New (tracked):

```text
configs/training/e3_source_boundary_0064.yaml
scripts/train_e3_source_boundary_0064.py
scripts/evaluate_e3_checkpoint_0064.py
scripts/run_e3_source_boundary_0064.sh
tests/unit/test_e3_source_boundary_levers_0064.py
docs/audits/0064-source-balanced-boundary-weighted-e3-refinement.md
```

Modified (tracked): **none.**

New untracked artifacts (ignored): `checkpoint/experiments/0064_e3_source_boundary_refinement/`
(manifests and histories only, no weights), `runs/diagnostics/0064_recipes/`.

Deleted (untracked generated weights): 11 files, ≈ 15.15 GiB (§2, §9).

`scripts/train_e3_boundary_refinement_0062.py` was **forked, not modified** — Audit 0062
cites it as the exact provenance of the active checkpoint.

## 14. Remaining Risks and Next Step

1. **Seed variance is large enough to fake a 2-point gain.** Any future single-seed result
   in this range must be treated as unproven until a second seed agrees. Two seeds may not
   be enough; three would cost 11 minutes.
2. **Source balancing remains the best-evidenced direction** — it is the only intervention
   that improved precision and recall together and it won on four secondary measures. The
   right follow-up is not to abandon it but to run it across three seeds and, if the mean
   holds above +1.5, to consider a longer schedule than two epochs. It was still improving
   at epoch 2 in both seeds.
3. **Boundary weighting should not be revisited at the token level.** Two doses gave a
   clean monotone precision cost. The boundary problem is real (298 errors) but this lever
   trades against it rather than solving it; spec §7.1's boundary-offset *head* is a
   different mechanism and remains untried.
4. **The type-disjoint corpus is the deeper constraint.** DIAGNOSIS and SYMPTOM have one
   source and 24.4% of training. No sampler can create supervision that does not exist;
   more DIAGNOSIS/SYMPTOM data would raise the ceiling that R5 is pushing against.
5. **Validation has only two sources**, so "worst-source F1" is a two-way minimum and a
   weaker guard than it sounds. It passed for every recipe and discriminated nothing.
6. **The §2 guard fault** (a substring grep that could not distinguish a load path from a
   provenance field, in a script that continued past its own abort message) should be fixed
   before the next cleanup.

## 15. Verdict

**`E3_4E_NOT_ACCEPTED`**

Four recipes and one confirmation seed were run. R5 passed all eleven development-seed
criteria and then failed the criterion that exists precisely to catch a lucky seed. The
active Audit-0062 checkpoint is retained unchanged, no public inference was run, the
leaderboard attempt is preserved, and 15.15 GiB of superseded weights were removed with
full provenance kept.
