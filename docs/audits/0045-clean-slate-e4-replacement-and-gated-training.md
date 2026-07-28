# Audit 0045 - Clean-Slate E4 Replacement and Gated T4 Training Pipeline

Date: 2026-07-28

## 1. Objective and Scope

Audit 0044 proved `ALL_BACKGROUND_LOSS_COLLAPSE` by direct measurement. This
milestone **removes** the implementation that produced it and replaces it with
one current E4: one source package, one notebook, one configuration family.

Audits 0043 and 0044 remain untouched. They are the record of why the removal
happened, and they are the reason nothing here is a guess.

No training ran locally. `internal_test` was never opened. No organizer inference
ran. No `output.zip` was produced. No assistant-specific file was created. No Git
history was rewritten and nothing was force-pushed.

### Status

| Subject | Status |
| --- | --- |
| Removed E4 implementation | `DELETED_FROM_CURRENT_TREE` |
| Current E4 implementation | `IMPLEMENTED_NOT_AUTHORIZED` |
| Stage 2 / 3 / 4 | `NOT_EXECUTED` — every run flag ships False |
| E4 model quality | **no claim** |

## 2. Initial Git State

```text
branch main, working tree clean
a2848ae fix: gate the collapse verdict on head restoration and harden the source scan
80f12b4  E4 post-training collapse
b7400a6 feat: diagnose E4 post-training collapse with bounded evidence
```

Audit 0044 is committed at `80f12b4`/`a2848ae`. Architecture PDF read in full;
SHA-256 `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`,
unchanged.

## 3. Obsolete Tracked Files Removed

```text
src/mednorm_vi/training/phase2/e4_w2ner_training.py        1,669 lines
src/mednorm_vi/training/phase2/e4_collapse_diagnosis.py    1,803
src/mednorm_vi/training/phase2/e4_checkpoint_probe.py      1,047
src/mednorm_vi/training/phase2/e4_tiny_overfit.py            546
scripts/diagnose_e4_collapse.py                              300
notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb
notebooks/MedNorm_E4_TinyOverfit_Diagnostic.ipynb
configs/training/phase2_e4_phobert_w2ner_colab.yaml
configs/training/phase2_e4_tiny_overfit_diagnostic.yaml
tests/unit/test_e4_collapse_diagnosis.py                   1,078
tests/unit/test_e4_checkpoint_probe.py                       647
tests/unit/test_e4_full_training_hardening.py                762
```

The diagnosis and probe modules go with the implementation they diagnosed: their
subject no longer exists, and their findings are permanently recorded in Audits
0043 and 0044. Keeping them would leave code whose only purpose is to analyse a
deleted artifact.

Test blocks that asserted the *structure of the deleted notebooks* were removed
from `test_e4_runtime_io.py` (14 blocks) and `test_e4_progress_observability.py`
(14). The module-level tests in both files survive untouched. The v1
checkpoint-compatibility tail of `test_e4_atomic_grid_alignment.py` was replaced
by two contract-identity assertions.

## 4. Local Failure Artifacts Removed

Deleted after confirming Audit 0044 was committed, with sizes recorded first:

```text
local-artifacts/e4_phobert_w2ner_full_v1/                      8.3 GB total
  checkpoints/best.pt                                    4,447,490,233 bytes
  checkpoints/latest.pt                                  4,447,559,137
  e4_alignment_diagnostic.json                                  78,536
  grid_target_statistics.json                                      647
  logs/training_history.jsonl                                    4,895
  logs/training_progress.jsonl                               3,712,246
  resolved_config.json                                           1,926
  training_manifest.json                                         4,359
  validation_metrics.json                                        2,829

local-artifacts/e4_tiny_overfit_diagnostic_v1/            DID NOT EXIST
```

Every measurement those files supported is transcribed into Audits 0043 and 0044,
including both checkpoint digests, so the evidence survives the bytes.

**Preserved, untouched** — these belong to other experts:

```text
checkpoint/s1_mention_full_training_v1/    1.6 GB   E3/S1
reports/                                   2.4 MB
```

## 5. Shared Code Retained, and Why

| kept | reason |
| --- | --- |
| `mention_factory/w2ner.py` | the canonical grid builder and decoder; used by the L3 lattice, adapters, the ablation and E1/E2/E3 |
| `e4/alignment.py` (from `e4_w2ner_training.py`) | Audit 0043 round-tripped it over 33,826 train + 1,045 validation examples and 13,711 entities at exact P = R = F1 = 1.0. It was the one component measurement fully vindicated, so it moved **verbatim** |
| `e4/runtime_io.py` | Drive health, local materialization, bounded streaming — Audit 0040 work, orthogonal to the collapse |
| `e4/progress.py` | heartbeats and ETA — Audit 0041 work, orthogonal |
| `e4/alignment_diagnostic.py` | full-corpus alignment preflight |
| `phase2/training_contracts.py` | **new**: accumulation, precision and optimizer-signature primitives that E5 previously imported *from E4*. Extracting them means no expert depends on another expert's training module |

Deliberately **not** carried forward: the per-example mean loss, the single-LR
optimizer signature, `loss_scale_for` (per-microbatch group scaling — the defect
itself), and the smoke-checkpoint initialization path.

## 6. Final E4 Source Layout

```text
src/mednorm_vi/training/phase2/
├── training_contracts.py          shared with E5; no expert-to-expert dependency
└── e4/
    ├── __init__.py                one public surface
    ├── contracts.py               identity, pinned revision, weight format,
    │                              checkpoint schema v3, supervision scope
    ├── alignment.py               atomic grid words + PhoBERT projection (verbatim)
    ├── recipes.py                 the three candidate training contracts
    ├── sampling.py                deterministic epoch order
    ├── training.py                accumulation, precision, collapse guard, resume
    ├── gates.py                   four-stage fail-closed authorization chain
    ├── runtime_io.py              (moved)
    ├── progress.py                (moved)
    └── alignment_diagnostic.py    (moved)

configs/training/phase2_e4.yaml                  one config family
notebooks/MedNorm_E4_Clean_Training.ipynb        one notebook, four stages
tests/unit/test_e4_clean_training.py             the new contract tests
```

A stable, non-versioned path. Reproducibility lives in internal version fields
(`e4-recipe-v1`, `e4-data-order-v1`, `phase2-e4-checkpoint-v3`), not in directory
names.

### Supervision scope

The output schema keeps all five organizer types and all seven grid labels. The
governed E4 corpus contains **zero** TEST_NAME and TEST_RESULT mentions in train
and validation alike, so two classifier outputs have no training signal. That is
recorded (`E4_UNSUPERVISED_TYPES`, `E4_UNSUPERVISED_TYPE_POLICY`), never
fabricated: no synthetic laboratory mention is generated, no class is dropped
from the head, and no stage gate fails for not predicting them. Laboratory
extraction stays with E1/E2 and the L4 resolver.

## 7. Recipe 1 — `reference_ce`

**Batch-global valid-cell cross entropy.**

```text
loss(effective batch) = (sum of per-cell CE over every valid cell in every
                         microbatch of the batch)
                      / (total valid cells in the effective batch)
```

One division, at the end. Not `mean(per-example means)`. `BatchGlobalAccumulator`
accumulates the numerator and the cell count; `microbatch_scale()` gives the
`1 / expected_valid_cells` factor that makes accumulation mathematically
identical to a single large batch rather than an approximation of it.

```text
optimizer            AdamW, weight decay 0.01
backbone LR          5e-6      (configurable, recorded)
relation head LR     1e-3      (configurable, recorded)
schedule             linear warmup then linear decay, warmup ratio 0.10
gradient clipping    5.0
precision            fp16 + GradScaler on T4 (resolved from capability, not name)
microbatch           1, accumulation configurable (default 8)
data order           deterministic shuffle + stratified source interleaving
selection            best exact span-and-type F1 on governed validation only
early stopping       patience 3
```

`OptimizerGroups` **refuses** equal learning rates and refuses a head LR below
the backbone's — sharing one rate between 24 pretrained layers and a randomly
initialized head is what the collapsed run did.

## 8. Recipe 2 — `reference_ce_resampled`

Identical loss and optimizer contract, plus positive-aware ordering.

The governed train split is 7,698 positive and 26,128 zero-entity examples —
77.2% negative. A first implementation used a fixed 50% cap and produced an
**18,431-example zero-entity tail**: the same defect relocated. The cap is
arithmetically unsatisfiable, so the merge is proportional (Bresenham-style)
instead, and `max_zero_entity_fraction` became a *guard* that raises when
unsatisfiable rather than being silently approximated.

Measured on the real corpus layout:

```text
                              longest zero-entity streak   longest same-source run
file order (the failed run)              10,027                     10,027
shuffled_source_interleaved                  18                          2
positive_aware_resampled                      4                          3
```

Both orders are **pure permutations** — `assert_order_preserves_corpus` proves no
example is dropped, duplicated or invented. Zero-entity documents are real
negatives and stay in the corpus. The realized composition is measured per epoch
and written to the manifest.

The stratified interleave replaced a credit-based round-robin that was wrong: on
its first pass no source had accrued a full unit of credit, its "nothing
progressed" fallback fired, and it drained every bucket in file order —
reproducing exactly the 10,027-example block it existed to break up. Measurement
caught it; the docstring records it.

## 9. Recipe 3 — `balanced_focal`

Same data order, optimizer, scheduler and precision as `reference_ce`; a
configurable focal objective over valid cells.

```text
focal(cell) = -w * (1 - p_t)**gamma * log p_t
w = alpha for a positive class, (1 - alpha) for background

gamma  2.0    alpha  0.25    effective positive:background weight  0.333
```

* every positive cell keeps weight `alpha` — positives always participate;
* easy background (`p_t` near 1) is suppressed by `(1 - p_t)**gamma`;
* `FocalConfig` **refuses** `alpha > 0.9` and any effective ratio at or above
  100:1 — the raw 577:1 inverse-frequency regime is explicitly rejected;
* `Recipe.__post_init__` refuses any reduction other than
  `batch_global_valid_cell_mean`, so no recipe can silently fall back to the
  failed per-example mean;
* positive, background and total loss are logged separately. A single total is
  what let the collapse hide: it fell steadily while the positive term — the only
  one that can produce a mention — never moved.

## 10. Dependency and Preflight Fix

The prior tiny diagnostic died in Colab on a missing `py_vncorenlp`. Cell 2 of
the new notebook, before any project import:

```python
%pip install -q py_vncorenlp==0.1.4
```

Cell 3 is an executable preflight that verifies and prints:

```text
py_vncorenlp imports; version and package location printed
VnCoreNLP model download OR existing-cache resolution succeeds; jar presence
torch imports; version
transformers imports; version
CUDA available; GPU name; compute capability
T4-compatible CUDA runtime (capability >= 7.0, read from the runtime not the name)
pinned PhoBERT revision printed
internal_test prohibited
```

Every failure is collected and reported together, then `SystemExit`. The notebook
runs from a fresh Colab runtime with no manually added install cell.

## 11. Stage 2 — Tiny-Overfit Recipe Ablation

12 deterministic governed examples, all three recipes under identical examples,
revision, seed, epoch bound, evaluation code, decoder and precision policy.

Recorded per recipe: exact P/R/F1, predicted and gold mention counts, false
positives, positive-cell accuracy, gold-positive background rate, NNW count, THW
count by supervised type, total/positive/background loss, seconds, peak VRAM, and
save/reload reproduction.

```text
pass requires   exact train F1 >= 0.95
                predicted mentions > 0
                positive-cell accuracy > 0
                every supervised type PRESENT in the tiny set is predicted
                save/reload reproduces the result

grid-cell accuracy is NEVER a pass criterion
```

That last line is load-bearing: an all-background model already scores ~0.998
grid-cell accuracy on this corpus, so a gate keyed on it would have passed the
collapsed run.

Selection order: highest exact F1, highest recall, fewest false positives, lower
runtime, lower peak VRAM, simpler recipe. `reference_ce` is the simplest and
therefore wins any exact tie by construction — no special case needed.

Outputs a JSON comparison and a Markdown table. **Checkpoint hygiene**: each
candidate's checkpoint is written only long enough to validate save/reload, then
deleted; only compact metrics survive for non-selected recipes. If no recipe
passes, the stage writes a failed gate artifact and stops — subset and full
training do not run.

## 12. Stage 3 — Representative Subset Smoke

Only the recipe Stage 2 selected. Deterministic subset covering DIAGNOSIS,
MEDICATION and SYMPTOM, positive and zero-entity examples, every governed
training source, with a **disjoint** validation subset and recorded IDs and
hashes.

```text
pass requires   validation predicted mentions > 0
                validation recall > 0
                NNW > 0 and THW > 0
                every supervised type present in the subset is predicted
                gold-positive background rate < 0.98   (materially below 1)
                the collapse guard did not fire
                best checkpoint save/reload reproduces metrics
                the artifact validator passes
```

Non-best checkpoints are deleted; only the best checkpoint, the compact state
needed for a same-run resume, metrics, manifest and hashes are kept. A failed
gate stops the chain.

## 13. Stage 4 — Full T4 Training

`assert_full_training_allowed` checks, in order: authorization string and run
flag; both gate artifacts exist; both passed; both name a known recipe; both
agree on the recipe; and the config, code and corpus hashes recorded in each
match the current ones. Flipping `RUN_STAGE4_FULL_TRAINING` satisfies exactly one
of those.

Training initializes from pinned pretrained PhoBERT plus a freshly initialized
relation head. `reject_superseded_checkpoint` refuses `phase2-e4-checkpoint-v1`
and `-v2` by schema version. This matters precisely because the collapsed
checkpoints **restore perfectly** (Audit 0044 §6) — being loadable is exactly why
refusing them has to be mechanical rather than a matter of discipline.

Local checkpoint first, hash-verified, reload-checked, then Drive with a digest
comparison. Resume is permitted only for a same-run interruption; `run_id` is in
the compatibility set so a resume cannot silently continue a different run.

## 14. Collapse Guards

```text
all four required, simultaneously, after warmup:
    predicted mentions == 0
    THW predictions == 0
    recall == 0
    gold-positive background rate >= 0.999
patience: 2 consecutive post-warmup validations
```

All four are required because any one alone is survivable early — an epoch-1
model legitimately predicts nothing. "Loss decreasing while collapsed" is
reported but **not** required: the audited run's loss actually *rose* into the
attractor, so requiring a falling loss would have missed it entirely.

Replayed against the real audited validation history:

```text
guard fires at epoch 5 (first collapsed epoch 4, streak 2)
epochs saved            7
backward passes saved   236,782
```

`assert_not_collapsed_when_marking_trained` refuses to record `FULLY_TRAINED` for
a collapsed run; the status is `COLLAPSED_NOT_TRAINED`.

## 15. Repository and Artifact Size Policy

`.gitignore` covers `checkpoint/`, `reports/`, `data/derived/`, `artifacts/`,
`weights/`, `caches/`, `local-artifacts/`, `model_cache/`, `.venv/` and every
`*.pt`/`*.bin`/`*.safetensors`/`*.zip`. A test walks `git ls-files` and asserts
no tracked path ends in a weight suffix and no artifact directory is tracked.

The final tree holds one E4 implementation, one E4 notebook, one E4 config, the
current tests, and the historical audits. No Git history was rewritten.

## 16. Tests and Static Checks

`tests/unit/test_e4_clean_training.py` — new. Proves the obsolete imports and
paths are gone; one canonical implementation path; batch-global valid-cell
reduction correctness; gradient accumulation numerically equal to a single
effective-batch reduction; distinct backbone/head learning rates; deterministic
shuffle and source interleaving; bounded zero-entity streak in resampled mode;
focal downweighting of easy background; no fabricated TEST_NAME/TEST_RESULT
supervision; refusal of any superseded checkpoint; the collapse guard stopping an
all-NONE synthetic run; the tiny stage comparing all three recipes; full training
unable to bypass the tiny and subset gates; `py_vncorenlp==0.1.4` installed before
the first import; a fresh-runtime preflight cell; no `internal_test`; no optimizer
or backward call in the implementation; no tracked artifacts; and Audits 0043 and
0044 plus the architecture PDF byte-identical.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q            1612 passed, 1 skipped
tests/unit/test_e4_clean_training.py                        115 passed
ruff check .                                                All checks passed
ruff check notebooks                                        All checks passed
env PYTHONPATH=src .venv/bin/python -m mypy                 Success: no issues found
                                                            in 271 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src    clean
git diff --check                                            clean
```

The single skip is the long-standing `pyarrow not installed locally` in
`tests/unit/test_vietmed_adapter.py`. The suite shrank from 1,704 to 1,612 tests
because 2,487 lines of tests for the deleted probe, diagnosis and v1 training
path were removed with their subject; 115 new tests replace them.

### 16.1 Two defects the milestone's own measurements caught

Both were in work added here, and both are recorded rather than quietly fixed:

1. **The source interleaver drained in file order.** A credit-based round-robin
   accrued less than one unit of credit on its opening pass, so its "nothing
   progressed" fallback fired immediately and emptied every bucket in file order
   — reproducing the exact 10,027-example zero-entity block it existed to break
   up. Measuring the realized order against the real corpus layout exposed it;
   the stratified fractional-position sort replaced it.

2. **The positive-aware cap was unsatisfiable.** A fixed 50% zero-entity cap
   against a 77.2% zero-entity corpus forced every remaining negative into an
   18,431-example tail. The proportional merge replaced it, and the fraction
   became a guard that raises rather than approximates.

Neither would have been caught by inspection. Both were found by running the
ordering over the governed corpus composition and measuring the result.

**No training was run locally.** No optimizer was constructed and no backward
pass executed anywhere in this milestone, in the implementation or in the tests.

## 17. Limitations

* **No E4 quality is claimed.** Nothing has been trained under any recipe. Stage
  2 has not run, so which recipe wins is unknown.
* The recipes are *candidates*, not a validated fix. That `reference_ce`'s
  batch-global reduction removes the measured defect is arithmetic; that it
  produces a good model is a hypothesis Stage 2 exists to test.
* The four corrections address the four causes Audit 0044 measured. If the
  collapse also had a cause the probe did not reach, these will not fix it — and
  the Stage-2 gate is what will reveal that, cheaply, on 12 examples.
* Focal defaults (gamma 2.0, alpha 0.25) are conventional, not tuned for this
  corpus. Stage 2 compares them against plain CE rather than assuming them.
* The tiny selection requires DIAGNOSIS, SYMPTOM and MEDICATION; no governed
  example carries all three, so the set is built by per-type quota.
* `phoner_covid19` and `vietmed_ner` are absent from governed validation
  (Audit 0043 §7). That corpus-composition issue is **not** addressed here.

## 18. Exact Fresh-Colab Steps

```text
0.  New Colab notebook -> Runtime -> Change runtime type -> T4 GPU.
    Open notebooks/MedNorm_E4_Clean_Training.ipynb.

1.  Runtime -> Run all.
    Cell 2 installs py_vncorenlp==0.1.4. Cell 3 runs the preflight and prints
    torch/transformers versions, GPU name, compute capability and the pinned
    revision. With every flag False the notebook stops here and trains nothing.

2.  STAGE 2 - tiny recipe ablation.
    RUN_STAGE2_TINY_ABLATION = True
    CONFIRM_STAGE2 = "I_AUTHORIZE_E4_TINY_RECIPE_ABLATION"
    Run all. Read the Markdown comparison table. If no recipe passes, STOP —
    the notebook raises and writes a failed gate artifact.

3.  STAGE 3 - subset smoke (only after Stage 2 passes).
    RUN_STAGE3_SUBSET_SMOKE = True
    CONFIRM_STAGE3 = "I_AUTHORIZE_E4_SUBSET_SMOKE"
    Run all. It reads the Stage-2 gate and uses the selected recipe.

4.  STAGE 4 - full training (only after Stages 2 and 3 pass).
    RUN_STAGE4_FULL_TRAINING = True
    CONFIRM_STAGE4 = "I_AUTHORIZE_E4_FULL_TRAINING"
    Run all. Stage 4 re-verifies both gate artifacts and their config/code/corpus
    hashes before a single weight is loaded.

5.  Return to the repository with gates/stage2_tiny_ablation.json,
    gates/stage3_subset_smoke.json, validation_metrics.json and
    logs/training_history.jsonl. Report the status exactly as recorded:
    COLLAPSED_NOT_TRAINED is a real outcome and is never a trained model.
```

Artifacts live under `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_current`.
Nothing under it is ever tracked by Git.

## 19. Safe-to-Commit Verdict

Safe to commit after review. No protected architecture, audit, evaluator or
governed corpus file changed; Audits 0043 and 0044 and the architecture PDF are
byte-identical; no existing test was weakened (the removed blocks asserted the
structure of deleted files, and two compile checks were taught to strip IPython
magics rather than dropped); no weight, checkpoint, corpus, cache or archive is
staged; no Git history was rewritten.

```bash
git add -A \
  docs/audits/0045-clean-slate-e4-replacement-and-gated-training.md \
  docs/audits/README.md \
  configs/training/phase2_e4.yaml \
  notebooks/MedNorm_E4_Clean_Training.ipynb \
  src/mednorm_vi/training/phase2/e4/ \
  src/mednorm_vi/training/phase2/training_contracts.py \
  src/mednorm_vi/training/phase2/e5_readiness.py \
  tests/unit/

git add -u   # records the deletions listed in section 3

git commit -m "feat: replace E4 with a clean gated training pipeline"
git push origin main
```
