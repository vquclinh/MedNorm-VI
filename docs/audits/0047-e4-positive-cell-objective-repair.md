# Audit 0047 - E4 Positive-Cell Objective Repair and Best-State Consistency

Date: 2026-07-28

## 1. Objective and Scope

The repaired Stage-2 tiny ablation ran to completion on a Colab T4. **This run is
valid.** It is recorded here as a genuine three-way recipe failure, not a runtime
failure, and it is the evidence on which the Stage-2 matrix is revised.

Two things follow: a proven defect in how reproduction was measured, and a
replacement objective matrix aimed at the imbalance the run actually exposed.

**No claim is made that either new objective works.** Neither has been run.

No training ran locally. `internal_test` was never opened. Stage 3, Stage 4 and
organizer inference did not run. No `output.zip` was produced. The tiny gate is
unchanged at exact F1 >= 0.95 and the epoch bound is unchanged at 200. The
one-current-E4 layout is preserved.

### Status

| Subject | Status |
| --- | --- |
| Stage-2 run of 2026-07-28 | `VALID_COMPLETED_RUN` — all three recipes failed |
| `balanced_focal` | `RETIRED` on completed evidence |
| Reproduction measurement | `DEFECT_PROVEN_AND_FIXED` |
| `group_balanced_ce`, `hard_negative_ce` | `IMPLEMENTED_NOT_RUN` |
| Stage 3 / Stage 4 | `BLOCKED` — no Stage-2 recipe has passed |

Audits 0043-0046 and the architecture PDF are unchanged.

## 2. The Completed Run, Interpreted

```text
epochs run                200 / 200      every recipe
optimizer steps           600 / 600      every recipe
warmup steps served        60 /  60      every recipe
peak learning rates       reached        every recipe
stopped inside warmup     none
```

Every condition Audit 0046 repaired held. This is what a valid Stage-2 result
looks like.

```text
recipe                    exact F1  recall  predicted  pos-cell acc  loss+     loss bg
reference_ce                0.1600  0.0909   3 / 22        0.2394    1.793879  0.005660
reference_ce_resampled      0.2308  0.1364   4 / 22        0.2958    1.784675  0.005362
balanced_focal              0.0000  0.0000   0 / 22        0.0000    0.468997  0.000521
```

**All three failed the unchanged 0.95 gate.** The reading:

* The **CE variants learned real relations but could not memorize.** Positive-cell
  accuracy of 0.24-0.30 and 3-4 correct mentions out of 22 is a model that has
  started to work and stalled — not a collapse. Compare Audit 0044's failure,
  which produced zero of everything.
* `reference_ce_resampled` scored marginally above `reference_ce`. On a tiny set
  of **positive examples only**, the two differ solely in data order and there is
  no zero-entity block to reorder, so this gap is noise, not a finding.
* **`balanced_focal` remained all-background for the entire run.** Its background
  loss fell to 0.000521 while its positive loss sat at 0.468997: it optimized the
  easy majority to near zero and never emitted a relation.

### 2.1 What the loss numbers say

For the CE variants, positive loss ~1.79 against background loss ~0.0055 — a
factor of ~325 — while the **total** sat near 0.016, tracking the background.
Audit 0044's batch-global reduction fixed *per-example* normalization; it did
not, and was never going to, change the fact that background cells outnumber
positive cells 577:1 and therefore still dominate the mean.

That is the target of this milestone.

## 3. Proven Save/Reload Root Cause

The hypothesis was that `best_state` is saved but compared against final-epoch
metrics. **Proven by code trace, not assumed.**

`train_recipe` in the notebook, by indentation:

```text
line 164   indent 4     for epoch in range(1, epochs + 1):
line 222   indent 8         metrics = evaluate(...)        <- rebound EVERY epoch
line 285   indent 4     return {"metrics": metrics, ...}   <- OUTSIDE the loop
```

`metrics` at the return is therefore the **final** epoch's evaluation. And
`best_state` is assigned only inside `if selector.observe(...)`, so it holds the
**best** epoch's weights. Stage 2 then did:

```python
metrics = outcome["metrics"]                    # FINAL epoch
payload["model_state"] = outcome["best_state"]  # BEST epoch
...
reproduction = ReproductionCheck(
    metrics_before=reproduction_metrics(metrics),        # FINAL metrics
    metrics_after=reproduction_metrics(after_metrics),   # BEST state, re-evaluated
)
```

The check compared **two different models**. Its verdict carried no information
about serialization at all.

This also explains the third row exactly: `balanced_focal` predicted nothing at
every epoch, so its final metrics and its best-state metrics were both all-zero
and matched trivially. It reported `save_reload_reproduced: true` for the same
reason the other two reported false — the comparison was meaningless in all
three cases, and only looked correct where the model was uniformly dead.

### 3.1 The corrected contract

`BestFinalRecord` names all four quantities and refuses to conflate them:

```text
best_epoch        the epoch whose state is saved
best_metrics      what the gate judges and what reproduction compares
final_epoch       the last epoch run
final_metrics     retained for trajectory reporting ONLY
best_state_changed_at   every epoch at which the best state moved
```

Reproduction is now, in order:

```text
1  evaluate best_state in a FRESH model, before serialization
2  save best_state
3  instantiate ANOTHER fresh model
4  restore the saved best_state; record missing/unexpected keys
5  evaluate again on the same 12 examples
6  compare the two evaluations of the SAME state, tolerance 1e-6
7  require eval mode and deterministic evaluation
8  never compare best_state against final_metrics
```

Step 8 is mechanical, not a convention: `ReproductionCheck` carries
`metrics_before_source` and `metrics_after_source`, and raises unless they are
exactly `best_state_pre_serialization` and `best_state_post_reload`. Passing
`final_epoch_metrics` raises with the words "Audit-0047 defect".
`assert_gate_uses_best_metrics` enforces the same thing for the gate.

The notebook no longer exposes a bare `outcome["metrics"]` key at all — there are
`best_metrics`, `best_epoch`, `final_metrics`, `final_epoch` and `best_final`.
Stages 2, 3 and 4 all judge `best_metrics`.

## 4. Revised Stage-2 Matrix

Stage 2 now compares exactly:

```text
1  reference_ce         baseline, unchanged
2  group_balanced_ce    new
3  hard_negative_ce     new
```

**`reference_ce_resampled` is no longer a Stage-2 objective.** It differs from
`reference_ce` only in data order, and the tiny set is entirely positive
examples, so zero-entity resampling has nothing to reorder — comparing them there
was comparing a recipe against itself. It remains defined and available for
Stage 3 and full training, where the ordering is real, and a test asserts that.

**`balanced_focal` is retired.** A completed 200-epoch run with full warmup and
peak learning rates produced zero predicted mentions. `build_recipe` raises for
it, `select_recipe` refuses any non-Stage-2 recipe, and the config records the
evidence under `retired_recipes`.

## 5. `group_balanced_ce`

```text
positive_mean   = mean CE over all valid non-NONE gold cells
background_mean = mean CE over valid NONE gold cells
loss            = 0.5 * positive_mean + 0.5 * background_mean
```

Each group is averaged **before** it is weighted, which is why a 577:1 imbalance
changes neither weight. This is not inverse-frequency class weighting, and
`GroupWeights` refuses a positive:background ratio at or above 100:1.

On the run's own measured numbers:

```text
positive_mean 1.79, background_mean 0.0055, 5 positive cells in 1000
  natural-frequency valid-cell mean   0.0144   <- tracks the background
  group-balanced                      0.8978   <- positives carry half
```

Requirements met: every positive cell participates; positive and background means
are recorded independently in the manifest and telemetry; an effective batch with
**no positive cell** falls back explicitly to `background_mean` — tested, because
returning 0.0 would hand the optimizer a free batch; an all-positive batch falls
back to `positive_mean`; the weights are configurable and manifest-recorded.

## 6. `hard_negative_ce`

Every positive cell participates. Background cells are retained only if they are
among the highest-loss negatives, bounded at **3:1** against the positive count.

```text
1 positive cell in a 100-cell grid -> 1 + 3 = 4 cells enter the loss, not 100
5 positives, 10,000 background     -> 15 negatives kept
5 positives, 4 background          -> 4 kept (min with what exists)
```

Selection is by `(-loss, row, column)`: the positional tiebreak makes it
reproducible **without consulting any RNG**, so it is stable across processes
rather than merely stable under the recorded seed.

`no positive cells` uses an explicit bounded rule — keep the
`no_positive_background_cap` (default 32) hardest negatives. Not all of them,
which would restore natural-frequency CE exactly where it does most damage, and
not none, which would drop the batch silently. Both branches are tested.

A ratio above 50 raises: an unbounded ratio is natural-frequency CE by another
name. Selected and total background counts are logged every epoch. The reduction
stays effective-batch global over the selected cells.

## 7. Telemetry

`EpochTelemetry` records, per epoch, aggregate only:

```text
best and final exact F1                     positive and background loss
positive-cell accuracy                      positive and background cell counts
per-positive-class accuracy for             selected hard-negative count and
  NNW, THW:DIAGNOSIS,                         total background candidates
  THW:MEDICATION, THW:SYMPTOM                 (hard_negative_ce only)
gold-positive predicted-as-NONE rate        relation-head gradient norm
mean NONE logit on gold-positive cells      backbone gradient norm
strongest non-NONE margin on those cells    best_epoch, best_state_changed
```

The two gradient norms come from clipping each parameter group separately, which
also makes the differential learning rates observable. No clinical text is ever
recorded; a test asserts the payload carries counts, rates and fixed names only.

## 8. Stage-2 Stopping and Gates — Unchanged

```text
TINY_EPOCH_BOUND                     200        unchanged
exact train F1 >= 0.95               unchanged
predicted mentions > 0               unchanged
positive-cell accuracy > 0           unchanged
every present supervised type        unchanged
real same-state save/reload          strengthened (section 3)
validation-patience stopping         still not used
collapse guard                       still disabled for tiny
Stage 3                              still not automatic
```

## 9. Tests and Static Checks

`tests/unit/test_e4_clean_training.py` grew from 150 to 180 tests. The new ones
prove: reproduction refuses final-epoch metrics as an operand and refuses a wrong
post-reload source; both sides declare the best state; eval mode and determinism
are required; `BestFinalRecord` requires both halves and rejects a best epoch
after the final one; the gate must be evaluated from the best state; the notebook
exposes no bare `outcome["metrics"]` and evaluates the best state before
serializing; the retired candidate cannot be built or selected; Stage-2 selection
refuses a non-Stage-2 recipe; `reference_ce` remains the baseline;
`reference_ce_resampled` stays available for Stage 3/full; group-balanced gives
the two groups independent influence on the run's own measured numbers; group
weights are not inverse frequency; the no-positive and all-positive fallbacks are
explicit; hard-negative selection is deterministic and never discards a positive;
the ratio is bounded by default and by guard; hard-negative does not silently
become all-cell CE; telemetry records every required quantity and stays
aggregate; and the tiny gate and epoch bound are unchanged.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q            1677 passed, 1 skipped
tests/unit/test_e4_clean_training.py                        180 passed
ruff check .                                                All checks passed
ruff check notebooks                                        All checks passed
env PYTHONPATH=src .venv/bin/python -m mypy                 Success: no issues found
                                                            in 271 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src    clean
git diff --check                                            clean
```

**No local training was performed.** No optimizer was constructed and no backward
pass executed by this milestone's code or tests. `internal_test` was not opened.
Stage 3, Stage 4 and organizer inference did not run.

## 10. Limitations

* **Neither new objective is claimed to work.** Both are implemented and
  unrun. `group_balanced_ce` and `hard_negative_ce` are hypotheses aimed at the
  measured 325:1 loss ratio; only a Colab run can test them.
* The two objectives attack the same problem from opposite directions —
  reweighting versus discarding. If the tiny set is unlearnable for a reason that
  is not the positive/background ratio, both will fail together, and that itself
  will be informative.
* The 3:1 hard-negative ratio and the 0.5/0.5 group weights are conventional
  starting points, not measurements. They are configurable and recorded.
* `reference_ce`'s 0.16 and `reference_ce_resampled`'s 0.23 were **final-epoch**
  numbers under the old reporting. The repaired code reports best-epoch metrics,
  so the re-run's baseline figure for `reference_ce` may differ from 0.16 without
  anything having changed in the objective.
* The tiny set is 12 positive-only examples. A recipe that memorizes it is not
  thereby a good recipe on the governed corpus — that is what Stages 3 and 4 are
  for.

## 11. Exact Fresh-Colab Stage-2 Rerun

```text
0.  New Colab notebook -> Runtime -> Change runtime type -> T4 GPU.
    Open notebooks/MedNorm_E4_Clean_Training.ipynb from the updated repository.

1.  Runtime -> Run all. Cell 2 installs py_vncorenlp==0.1.4; cell 3 runs the
    preflight. With every flag False the notebook stops there.

2.  Enable Stage 2 only:
        RUN_STAGE2_TINY_ABLATION = True
        CONFIRM_STAGE2 = "I_AUTHORIZE_E4_TINY_RECIPE_ABLATION"
        RUN_STAGE3_SUBSET_SMOKE  = False
        RUN_STAGE4_FULL_TRAINING = False
    Runtime -> Run all. Three recipes now run: reference_ce, group_balanced_ce,
    hard_negative_ce.

3.  Watch the per-epoch telemetry. The quantities that matter most:
        loss_positive vs loss_background      is the gap closing?
        positive_cell_accuracy                is it climbing past 0.30?
        per_positive_class_accuracy           which of NNW / THW:* is stuck?
        gold_positive_predicted_as_none_rate  is it falling below 0.7?
        head_grad_norm, backbone_grad_norm    is the head actually moving?
        selected_hard_negatives               (hard_negative_ce only)

4.  For each recipe confirm in stage2_tiny_ablation.json:
        schedule.peak_learning_rate_reached is true
        stopped_reason is tiny_gate_met or
          reached_epoch_bound_without_meeting_the_gate
        best_epoch and final_epoch are BOTH recorded
        reproduction.metrics_before_source == "best_state_pre_serialization"
        reproduction.metrics_after_source  == "best_state_post_reload"
        reproduction.reproduced is true and differences are within 1e-6
    A reproduction False now means a real serialization problem, because both
    sides are the same state.

5.  If a recipe passes the 0.95 gate, proceed to Stage 3 with the selected
    recipe. If none passes, STOP and report the telemetry — the next decision
    depends on which quantity moved and which did not.

6.  Return with stage2_tiny_ablation.json and recipe_comparison.md. Report the
    outcome exactly as recorded; do not describe a completed failure as a
    runtime problem.
```

## 12. Safe-to-Commit Verdict

Safe to commit after review. Audits 0043-0046 and the architecture PDF are
byte-identical; the tiny gate is unchanged at 0.95 and the epoch bound at 200;
validation-patience stopping is still not used for tiny; the collapse guard is
still disabled there; the one-current-E4 layout is preserved; no existing test
was weakened; nothing artifact-shaped is staged; no Git history was rewritten.

```bash
git add \
  docs/audits/0047-e4-positive-cell-objective-repair.md \
  docs/audits/README.md \
  configs/training/phase2_e4.yaml \
  notebooks/MedNorm_E4_Clean_Training.ipynb \
  src/mednorm_vi/training/phase2/e4/__init__.py \
  src/mednorm_vi/training/phase2/e4/gates.py \
  src/mednorm_vi/training/phase2/e4/recipes.py \
  src/mednorm_vi/training/phase2/e4/training.py \
  tests/unit/test_e4_clean_training.py

git commit -m "feat: replace the E4 Stage-2 objectives and fix best-state reproduction"
git push origin main
```
