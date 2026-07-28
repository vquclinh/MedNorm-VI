# Audit 0046 - E4 Tiny-Overfit Execution Contract Repair

Date: 2026-07-28

## 1. Objective and Scope

The clean E4 Stage-2 tiny ablation was executed on Colab T4. All three recipes
were reported failed. **That result is invalid for recipe comparison**: every
recipe was stopped by the full-training early stopper while still inside its
learning-rate warmup, at one fifth of the configured rates.

This milestone repairs the Stage-2 execution contract. It makes **no claim about
any recipe** — the ablation has not yet been run under a correct contract.

No training ran locally. `internal_test` was never opened. Stage 3, Stage 4 and
organizer inference did not run. No `output.zip` was produced. The exact-F1 gate
is unchanged at 0.95. The one-current-E4 layout from Audit 0045 is preserved.

### Status

| Subject | Status |
| --- | --- |
| First Stage-2 run | `INVALID_FOR_RECIPE_COMPARISON` |
| Stage-2 execution contract | `REPAIRED` |
| Recipe verdicts | **none** — Stage 2 must be re-run |
| Stage 3 / Stage 4 | `BLOCKED` — the stale gate cannot authorize them |

Audits 0043, 0044 and 0045 and the architecture PDF are unchanged.

## 2. The Colab Evidence

```text
TINY_EPOCH_BOUND                   200
epochs actually run                  4      (every recipe)
optimizer steps actually taken      12      (every recipe)
planned optimizer steps            600
warmup steps (10% of planned)       60
warmup actually served              12  =  20%
backbone LR at the stop           1e-6      configured target 5e-6
head LR at the stop               2e-4      configured target 1e-3
```

All three recipes were reported failed. None of them had reached the learning
rates it was configured to train at.

### 2.1 Why the result says nothing about the recipes

Stage 2 asks one question: *can this recipe memorize 12 examples?* A recipe that
was never allowed to leave warmup has not answered it. Reporting three failures
from that run would repeat, in miniature, the error Audits 0043 and 0044 exist to
prevent — treating an artefact of the harness as a fact about the model.

The arithmetic is fully reproduced in
`tests/unit/test_e4_clean_training.py::test_the_reproduced_stop_lands_inside_warmup_at_a_fifth_of_target_lr`,
which derives 600, 60, 12, 0.2, 1e-6 and 2e-4 from the shipped code.

## 3. Exact Root Cause

`train_recipe` in `notebooks/MedNorm_E4_Clean_Training.ipynb` applied the
**full-training** stopper to every run, Stage 2 included:

```python
selector = BestCheckpointSelector(patience=3)
...
if selector.observe(epoch=epoch, exact_f1=metrics["validation_exact_f1"]):
    best_state = ...
...
if selector.should_stop:
    break
```

With exact F1 pinned at 0.0 during warmup:

```text
epoch 1   0.0 > -1.0        -> new best, best_epoch=1, without_improvement=0
epoch 2   0.0 > 0.0 is False -> without_improvement=1
epoch 3                      -> without_improvement=2
epoch 4                      -> without_improvement=3 == patience -> STOP
```

12 examples with accumulation 4 give `ceil(12/4) = 3` optimizer steps an epoch,
so four epochs are 12 steps.

**Answers to the questions asked:**

```text
early-stopping condition       BestCheckpointSelector.should_stop
                               (epochs_without_improvement >= patience)
configured patience            3
best epoch                     1
best metric                    0.0
planned optimizer steps        600   (3/epoch x 200 epochs)
warmup steps                    60   (10% of planned)
actual optimizer steps          12   (3/epoch x 4 epochs)
generic full-training early
stopper applied to tiny?       YES - this is the defect
```

`BestCheckpointSelector` is not itself wrong. It is the correct stopper for full
training, where a plateau in validation F1 is real information. On a tiny
memorization run during warmup, "F1 has not improved yet" is the *expected*
state, and the stopper was answering a question nobody asked.

### 3.1 Regression test written before the fix

`test_the_generic_stopper_reproduces_the_four_epoch_stop` replays the shipped
selector against a flat 0.0 F1 and asserts it stops at **epoch 4** with **12**
optimizer steps, `best_epoch == 1`, `best_metric == 0.0`. It exercises
`BestCheckpointSelector` directly, so it keeps passing after the repair and
documents the behaviour that made the Colab run invalid.

## 4. Corrected Tiny Stopping Policy

`TinyOverfitStopPolicy` in `src/mednorm_vi/training/phase2/e4/training.py`.

```text
stop when   the tiny gate is met                    -> tiny_gate_met
            the epoch bound is reached              -> reached_epoch_bound_...
            the loss is NaN/Inf (or CUDA fails)     -> numeric_failure
            [opt-in] proven not learning AFTER the
            full configured warmup                  -> proven_not_learning_...

never stop  because validation F1 has not improved
            because F1 is still 0.0
            before the configured warmup is fully served
```

The gate is unchanged and unweakened:

```text
exact train F1 >= 0.95
predicted mentions > 0
positive-cell accuracy > 0
every supervised type PRESENT in the tiny set is predicted
```

`TinyEpochSignal.gate_met` evaluates all four; a recipe meeting them may stop
immediately, otherwise it runs to `TINY_EPOCH_BOUND`.

```text
allow_fail_fast                     False by default
fail_fast_patience_after_warmup     25 epochs, and only counted from the epoch
                                    at which optimizer_steps >= warmup_steps
collapse guard                      disabled for tiny-overfit
heartbeat                           epoch 1, every 5 epochs, and the last epoch
```

The fail-fast window is measured **after** warmup by construction, so a fail-fast
can never be caused by an under-warmed schedule. A test drives 19 dead epochs
inside warmup and asserts no stop, then continues past warmup and asserts the
opt-in fail-fast fires.

Stage 4's stopper is untouched: `test_full_training_early_stopping_is_unchanged`
asserts `BestCheckpointSelector` still stops after three non-improving epochs and
still selects on governed validation only.

## 5. Scheduler Accounting

`plan_schedule(examples, accumulation_steps, epoch_bound, warmup_ratio)` derives
the budget from the **full requested bound**, and `SchedulePlan` records planned
*and* realized values:

```text
examples 12, accumulation 4, epoch_bound 200, warmup_ratio 0.10
  optimizer_steps_per_epoch          3
  planned_total_optimizer_steps    600
  warmup_steps                      60

after the old premature stop (12 steps, 4 epochs)
  warmup_completed               False
  warmup_fraction_served           0.2
  peak_learning_rate_reached     False
```

`RecipeResult.stopped_inside_warmup` derives from that, and `select_recipe` now
marks a failed ablation `result_is_invalid_for_recipe_comparison` and names the
denied recipes when any of them stopped inside warmup — so this exact failure can
never again be read as a recipe verdict. A genuine failure after full warmup is
**not** flagged, and a test asserts both directions.

Learning rates and warmup ratio are unchanged and remain configurable. No
mathematical defect was found in them: at step 59 of 60 the multiplier is exactly
1.0, giving backbone 5e-6 and head 1e-3 as configured. The defect was never
reaching that step.

## 6. Real Save/Reload Reproduction

The removed check:

```python
reload_ok = set(reloaded["model_state"]) == {"base_model", "w2ner_head"}
```

That compares two dictionary key names. It passes for a checkpoint of zeros, for
the wrong recipe's weights, or for a payload whose tensors never loaded.

`ReproductionCheck` replaces it:

```text
1  save the candidate checkpoint and hash it
2  instantiate a FRESH architecture (fresh_model())
3  restore base_model and w2ner_head
4  record missing and unexpected keys from both load reports
5  re-evaluate the SAME 12 examples
6  compare exact precision / recall / F1, predicted mentions and positive-cell
   accuracy against the pre-save values, tolerance 1e-6
7  only then does `reproduced` become true
8  delete the temporary checkpoint
```

`RecipeResult` no longer carries a `save_reload_reproduced` boolean at all — it
holds the `ReproductionCheck` (or `None`), and the boolean is derived. There is
no constructor path that accepts a key-name comparison: omitting the metrics
raises `GateError`. `assert_real_reproduction(None)` raises rather than
defaulting to success.

Stage 3 carried the identical defect and was repaired the same way — its subset
best checkpoint is now restored into a fresh model and re-evaluated against the
disjoint validation subset. That was found by the Stage-2 test, not by
inspection.

## 7. Stale Gate Handling

The previous Stage-2 gate was written by the premature-stop implementation. It
cannot authorize anything now, for two independent reasons:

```text
1  it recorded passed=false, and assert_full_training_allowed refuses a gate
   that did not pass;
2  its code_sha256 was computed over the old e4/ package. Every file in that
   package changed, so the hash no longer matches and the gate is refused with
   "re-run the gated stages" even if its verdict had been positive.
```

`GateArtifact.write` is now atomic: it stages to `<name>.tmp` on the same
filesystem and `os.replace`s it, so a rerun replaces the old verdict in one step
and never leaves both. A test writes a failing gate, overwrites it with a passing
one, and asserts the directory contains exactly one file with the new contents.

**Stage 3 and Stage 4 remain blocked** until Stage 2 is re-run under the repaired
contract.

## 8. Tests and Static Checks

`tests/unit/test_e4_clean_training.py` grew from 115 to 150 tests. The new ones
prove: the four-epoch stop is reproduced by the generic stopper and lands at 20%
of warmup with the exact reported learning rates; the tiny policy does not stop
at epoch 4 on zero F1, and does not stop at any epoch on a flat zero-F1 run;
successful memorization may stop before the bound; unsuccessful training reaches
the bound; the exact-F1 gate is still 0.95 and still requires every present
supervised type; numeric failure stops immediately; fail-fast is off by default
and can only fire after full warmup; full-training early stopping is unchanged;
the schedule is planned from the full bound; planned and realized step counts are
both recorded; the scheduler reaches peak LR after warmup; a run stopped inside
warmup is flagged and its ablation marked invalid for comparison; reproduction
requires a fresh-model evaluation; **state-dict key names alone cannot construct
a passing reproduction check**; drift beyond tolerance and missing/unexpected
keys both fail; a recipe without a reproduction check cannot pass; the stale
failed gate cannot authorize Stage 4 and neither can a passing gate from the old
code hash; gates are replaced atomically; and the notebook uses every repaired
contract.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q            1647 passed, 1 skipped
tests/unit/test_e4_clean_training.py                        150 passed
ruff check .                                                All checks passed
ruff check notebooks                                        All checks passed
env PYTHONPATH=src .venv/bin/python -m mypy                 Success: no issues found
                                                            in 271 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src    clean
git diff --check                                            clean
```

**No local training was performed.** No optimizer was constructed and no backward
pass executed by this milestone's code or tests — asserted by token-level scans
over every `e4/` module and the test file itself. `internal_test` was not opened.
Stage 3, Stage 4 and organizer inference did not run. No `output.zip` was created.

## 9. Limitations

* **No recipe is claimed to work.** Stage 2 has not been run under the repaired
  contract. Whether any of the three can memorize 12 examples is still unknown.
* The repair addresses the stopper, the schedule accounting and the reproduction
  check. If the recipes also have a substantive problem, the re-run will show it
  — and now it will be a real result rather than a harness artefact.
* Learning rates and warmup are deliberately untouched. This was a code-only
  milestone and no mathematical defect was found in them.
* `fail_fast_patience_after_warmup` (25 epochs) is a judgement, not a measurement.
  It is off by default precisely because it has no evidence behind it yet.

## 10. Exact Colab Stage-2 Rerun

```text
0.  New Colab notebook -> Runtime -> Change runtime type -> T4 GPU.
    Open notebooks/MedNorm_E4_Clean_Training.ipynb from the updated repository.

1.  Runtime -> Run all.
    Cell 2 installs py_vncorenlp==0.1.4; cell 3 runs the preflight. With every
    flag False the notebook stops here.

2.  Enable Stage 2 only:
        RUN_STAGE2_TINY_ABLATION = True
        CONFIRM_STAGE2 = "I_AUTHORIZE_E4_TINY_RECIPE_ABLATION"
        RUN_STAGE3_SUBSET_SMOKE  = False
        RUN_STAGE4_FULL_TRAINING = False
    Runtime -> Run all.

3.  Watch the heartbeats. Each recipe prints, at epoch 1 and every 5 epochs:
        exact_f1, predicted_mentions, positive_cell_accuracy, types_predicted,
        optimizer_steps, warmup_fraction_served, peak_learning_rate_reached
    Confirm `peak_learning_rate_reached` becomes true near optimizer step 60.
    A recipe now runs to epoch 200 unless it meets the gate first.

4.  Read `stage2_tiny_ablation.json` and the Markdown table. For every recipe
    check `schedule.peak_learning_rate_reached` is true and `stopped_reason` is
    `tiny_gate_met` or `reached_epoch_bound_without_meeting_the_gate`. Any other
    reason means the run is again not a recipe verdict.

5.  Check `reproduction.reproduced` per recipe, and that
    `reproduction.differences` are all within 1e-6.

6.  Only if a recipe passes, proceed to Stage 3. The old failed gate is replaced
    atomically by this run; it cannot authorize anything.

7.  Return to the repository with stage2_tiny_ablation.json and
    recipe_comparison.md. Report the outcome exactly as recorded.
```

## 11. Safe-to-Commit Verdict

Safe to commit after review. Audits 0043-0045 and the architecture PDF are
byte-identical; the exact-F1 gate is unchanged; the one-current-E4 layout is
preserved; full-training early stopping is unchanged; no existing test was
weakened; no artifact, checkpoint or cache is staged; no Git history was
rewritten.

```bash
git add \
  docs/audits/0046-e4-tiny-overfit-execution-contract-repair.md \
  docs/audits/README.md \
  src/mednorm_vi/training/phase2/e4/__init__.py \
  src/mednorm_vi/training/phase2/e4/gates.py \
  src/mednorm_vi/training/phase2/e4/training.py \
  notebooks/MedNorm_E4_Clean_Training.ipynb \
  tests/unit/test_e4_clean_training.py

git commit -m "fix: repair the E4 tiny-overfit stopping and reproduction contract"
git push origin main
```
