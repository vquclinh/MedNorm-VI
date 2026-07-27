# Audit 0039 - E4 Real Smoke Evidence, Persistent PhoBERT Weight-Format Fix, and Full-Training Loop Hardening

Date: 2026-07-27

## 1. Objective and Scope

This append-only audit records three connected things:

1. the **real E4 Colab smoke run that succeeded** after Audit 0038;
2. the **persistent repository fix** for the PhoBERT weight-format failure the
   operator had to patch by hand in Colab;
3. **full-training loop hardening**: real gradient accumulation, mixed precision,
   gradient clipping, and true checkpoint/resume custody.

Not started here: E5, learned L4, S2 assertion, S3 retrieval, S4 reranking, Qwen,
internal_test, organizer inference, packaging. Nothing was trained locally.

`docs/MedNorm-VI_Architecture.pdf`, Audit 0036, Audit 0037, Audit 0038 and
`src/mednorm_vi/evaluation/exact_mention.py` are unmodified.

### Status

| Subject | Status |
| --- | --- |
| Previous real artifact (`e4_phobert_w2ner_smoke_v1`) | `SMOKE_EXECUTED`, `ARTIFACT_VALIDATED` |
| Newly changed repository | `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, `READY_FOR_COLAB_SMOKE` |
| Full training | **NOT** `READY_FOR_FULL_TRAINING` — that status is earned only after the Audit-0039 regression smoke passes |
| E4 trained model | `UNAVAILABLE_UNTRAINED` |

**No full-training claim is made.**

## 2. Initial Git State

```text
pwd                     /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
git branch --show-current   main
git status -sb          ## main...origin/main
git status --short      <clean>
git diff --check        <clean>

git log --oneline -5
d5ad5d1 fix: decouple E4 W2NER grid words from VnCoreNLP model words
020cae7 fix: repair E4 Colab bootstrap and PhoBERT alignment
03fc55f feat: complete phase2 Colab training readiness
c6107a4 feat: add phase2 mention ensemble and learned l4 v2 contracts
23065f9 feat: implement L3 span lattice and L4 resolver ablation
```

Audit 0038 is committed at `d5ad5d1` and left immutable. Architecture PDF read in
full before any change; SHA-256
`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`, unchanged.

## 3. Real Smoke Evidence

The E4 Colab smoke run completed successfully on the Audit-0038 repository state.

```text
model_id            vinai/phobert-large
model_revision      1c7880f20db59c0054c6de5afd71b012369f6ee4
tokenizer_revision  1c7880f20db59c0054c6de5afd71b012369f6ee4
tokenizer_class     PhobertTokenizer
tokenizer_is_fast   false

stage               training_completed
mode                smoke
validation_exact_f1 0.0
internal_test_accessed  false

best   checkpoint SHA-256  bd689ec9bdc824b5abb3c0fa6373a3a6461781ad88a004870a27e2b396b8bbb1
latest checkpoint SHA-256  42265aedb53bc39e729056bd0ec073eda27f7b3b15d136c29b170280686f0999

artifact_dir  /content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1
status        SMOKE_EXECUTED
validator.ok  true
validator.failures  []
validator.warnings  []
manifest SHA-256    d2c36a1d15b395d86638fc3cdab2983689d0d3aef4c8e65ce41d3298c4c765ac
```

**The smoke exact F1 of 0.0 is smoke-only and is not a readiness failure.** The
smoke objective was to prove that preprocessing, alignment, forward/backward,
optimizer, validation, checkpoint save/reload, hashing and artifact validation
execute on a bounded subset. It is **not** a full-training quality signal and no
quality claim is derived from it.

All of the above is now recorded in the tracked config
`configs/training/phase2_e4_phobert_w2ner_colab.yaml` under `observed_smoke_run`,
and asserted by test so the numbers cannot drift.

## 4. Observed Weight-Format Failure and the Persistent Fix

Before the successful run the notebook failed at
`AutoModel.from_pretrained(..., use_safetensors=True)` with:

```text
OSError: vinai/phobert-large does not appear to have model.safetensors or
model.safetensors.index.json and cannot be loaded with safetensors.
```

The operator manually changed `use_safetensors=True` to `use_safetensors=False`;
the model then loaded from the official pinned `pytorch_model.bin` and the run
completed. That manual edit is now **persisted**, so no future Colab run needs it.

`src/mednorm_vi/training/phase2/e4_w2ner_training.py`:

```text
E4_WEIGHT_FORMAT_BIN          = "pytorch_model.bin"
E4_WEIGHT_FORMAT_SAFETENSORS  = "model.safetensors"
E4_PINNED_MODEL_REVISION      = "1c7880f20db59c0054c6de5afd71b012369f6ee4"

resolve_phobert_weight_format(model_id, revision, *, repository_files=None)
assert_weight_format_loadable(weight_format)
```

A deterministic repository-file inspection is supported: when a file listing is
supplied the decision is a plain lookup. For the pinned revision the resolved
behaviour is, and is asserted to be:

```text
weight_format   = pytorch_model.bin
use_safetensors = false
```

`assert_weight_format_loadable` refuses a safetensors request for a revision known
not to publish it, so the pinned revision **cannot accidentally request
safetensors** even if a caller passes one. Resolution happens once, before
acquisition; there is no mid-run fallback across unrelated files.

`use_safetensors=True` no longer appears on any active E4 code path — the only
remaining occurrence in the notebook is inside an explanatory comment quoting the
original error, and the regression test strips comment lines before asserting.

Recorded in resolved config and manifest: `pretrained_weight_format`,
`use_safetensors`, `model_revision`, `tokenizer_revision`.

**Loading `pytorch_model.bin` is not a lower-quality model.** It is the
serialization format the official pinned repository publishes; the model,
architecture, tensor values and revision are unchanged.

The expected-only MLM-head load-report validation is unchanged and still tested:
`lm_head.*` keys may be unused when loading `AutoModel`; unexpected missing
encoder keys fail; unexpected non-MLM keys fail.

## 5. Training-Loop Defects Found

Inspection of the notebook, `e4_w2ner_training.py` and the Colab config found five
defects. All are corrected here.

| # | Defect | Effect |
| --- | --- | --- |
| 1 | `loss.backward(); optimizer.step(); optimizer.zero_grad()` after **every document** while `EFFECTIVE_BATCH_SIZE = 8` was declared | True effective batch was **one document**; the recorded optimizer-step count was the document count |
| 2 | No gradient clipping | Unbounded update norms |
| 3 | No mixed precision | fp32 PhoBERT-large plus an O(n²) relation grid is a real Colab OOM risk |
| 4 | Checkpoints held **model weights only** | No optimizer moments, no scaler scale, no step count, no best metric — a true resume was impossible |
| 5 | `RESUME_FROM_FULL_CHECKPOINT` was read but **never loaded anything** — it only decided whether to truncate `training_history.jsonl` | Setting it to `True` would silently restart from scratch while appending to the old history |

A sixth, smaller issue: `best_payload = latest_payload` aliased one object, so
`best.pt` and `latest.pt` could be byte-identical — which the artifact validator
rejects for full mode unless explicitly justified.

## 6. Real Gradient Accumulation

`AccumulationPlan` + `plan_gradient_accumulation` in `e4_w2ner_training.py`. Every
number is **derived arithmetic**, never a relabelled constant:

```text
micro_batch_size          explicit (1: W2NER grids are variable-sized per document)
accumulation_steps        explicit (8)
effective_batch_size      micro_batch_size * accumulation_steps  (derived)
micro_batches_per_epoch   ceil(example_count / micro_batch_size)
optimizer_steps_per_epoch ceil(micro_batches_per_epoch / accumulation_steps)
expected_optimizer_steps  optimizer_steps_per_epoch * epochs
expected_backward_passes  micro_batches_per_epoch * epochs
final_partial_group_size  micro_batches_per_epoch mod accumulation_steps (or the full size)
```

Loop behaviour, implemented in the notebook and pinned by tests:

- the optimizer steps **only** at accumulation boundaries
  (`is_optimizer_step_boundary`), including the trailing partial group;
- each micro-batch loss is divided by its group's **actual** size
  (`loss_scale_for`), so the final partial group is **not** under-scaled — every
  group's scales sum to exactly 1.0, asserted by a loop-shaped simulation test;
- `optimizer.zero_grad(set_to_none=True)` runs before the epoch and after each
  optimizer step;
- `torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)` with a tracked
  `max_grad_norm: 1.0`, applied after `scaler.unscale_` when a scaler is active;
- `optimizer_steps_total` increments only on real `optimizer.step()` calls.

`build_e4_training_accounting` **raises** if observed optimizer steps or backward
passes disagree with the plan, so a manifest cannot merely relabel a batch size.

History and manifest record: examples processed, backward passes, optimizer steps,
micro-batch size, accumulation steps, effective batch size, final partial-group
size, and the gradient-clipping setting.

## 7. Mixed Precision and Resource Safety

`MixedPrecisionPolicy` + `resolve_mixed_precision_policy(requested, device_type,
bf16_supported)`:

| Runtime | Resolved | Autocast | GradScaler |
| --- | --- | --- | --- |
| CUDA, bf16 requested and supported | `bf16` | yes | no (bf16 needs no loss scaling) |
| CUDA, bf16 requested, unsupported | `fp16` | yes | **yes** |
| CUDA, fp16 requested | `fp16` | yes | **yes** |
| CUDA, fp32 requested | `fp32` | no | no |
| CPU, anything requested | `fp32` | no | no |

`assert_full_training_device` refuses a CPU full-training run outright; the CPU
path stays valid for bounded smoke and local contract tests. The precision mode
and dtype are recorded in resolved config, manifest and every checkpoint, and a
resume whose precision differs is rejected — precision cannot silently change
after resume.

No quantization is introduced (`quantization: none`, asserted). PhoBERT is **not**
frozen: `freeze_base_model: false`, full fine-tuning plus the W2NER head, as the
architecture intends.

## 8. Checkpoint Custody and Resume

`build_e4_training_state_payload` writes everything an exact resume needs:

```text
model_state {base_model, w2ner_head}   optimizer_state   scaler_state
epoch   optimizer_steps   backward_passes   examples_processed
best_metric   best_checkpoint_sha256
pretrained_weight_format   precision_mode   accumulation_signature
optimizer_signature   scheduler_state (empty)   scheduler_configured=false
```

No arbitrary scheduler was added: `scheduler: none` is the tracked configuration,
`scheduler_configured` is `false`, and the field exists so that a scheduler added
later must also be checkpointed.

`assert_full_checkpoint_custody` fails on any missing resume key.
`assert_compatible_full_resume` rejects a resume unless **all ten** tracked fields
match: input contract, checkpoint schema, atomic projection, resolved config hash,
model revision, tokenizer revision, weight format, precision mode, optimizer
signature, accumulation signature.

`assert_full_initialization_source` enforces the initialization policy:

- full training initializes from the **pinned pretrained PhoBERT base plus a newly
  initialized compatible W2NER v2 head**;
- `RESUME_FROM_SMOKE_CHECKPOINT` stays `False` and any attempt to set it for a
  full run raises;
- a smoke-mode checkpoint is refused as a full-resume source.

`best.pt` and `latest.pt` are now built as **separate payload objects**, and
`latest.pt` records the current `best_checkpoint_sha256`, so both hashes stay
independently recomputable and the two files are not aliases.

The full artifact contract now also requires `e4_alignment_diagnostic.json`. The
Hugging Face cache stays at `DRIVE_ROOT/model_cache/huggingface`, outside the
artifact directory (asserted by test).

The existing `e4_phobert_w2ner_smoke_v1` artifact remains valid historical runtime
evidence and **must not** initialize full training.

## 9. Expected Full-Training Optimizer-Step Accounting

Derived from the implementation with ceiling division, not hardcoded:

```text
training examples          33,826   (governed train split)
micro_batch_size                1
accumulation_steps              8
effective_batch_size            8
epochs                         12

micro_batches_per_epoch    33,826
optimizer_steps_per_epoch   ceil(33,826 / 8) = 4,229
expected_optimizer_steps    4,229 * 12       = 50,748
expected_backward_passes    33,826 * 12      = 405,912
final_partial_group_size    33,826 mod 8     = 2      (loss scale 1/2, not 1/8)
```

For comparison, the Audit-0038 loop would have reported **405,912** "optimizer
steps" for the same run.

## 10. Validation and Model Selection

Unchanged and preserved: governed **validation-only** model selection, best
criterion `max_validation_exact_f1_governed_validation_only`,
`internal_test_allowed: false`. Nothing reads internal_test.

Each completed epoch records via `build_e4_history_row`: training loss, governed
validation exact precision / recall / F1, optimizer steps, backward passes,
completed examples, learning rate, micro-batch size, accumulation steps, effective
batch size, precision mode, and `internal_test_accessed: false`.

E4 remains **disabled** in every profile feature flag; full-training code existing
does not enable it.

## 11. Required Final Regression Smoke

The optimizer and accumulation path changed, so the previously validated smoke
artifact no longer characterizes the code that would run. One more smoke is
required before full training, and it must prove:

- the official `.bin` weight path needs no manual edit;
- the full-corpus preflight still passes;
- the slow `PhobertTokenizer` path still works;
- gradient accumulation executes;
- optimizer-step accounting is correct;
- the final partial accumulation group is correct;
- mixed precision works on the selected Colab GPU;
- checkpoint save/reload works;
- the artifact validator passes;
- internal_test remains untouched.

The existing smoke checkpoint is **not** reused as initialization. The tracked
smoke directory is now a **fresh** `e4_phobert_w2ner_smoke_v2`, with
`e4_phobert_w2ner_smoke_v1` recorded as `archived_smoke_dir`, so the two bodies of
evidence cannot be mixed. `observed_smoke_run.superseded_by_audit_0039_loop_changes`
is `true`.

## 12. Notebook Operator Settings

Committed defaults are the smoke settings:

```text
RUN_SMOKE_TRAINING = True
RUN_FULL_TRAINING = False
CONFIRM_FULL = ""
RESUME_FROM_SMOKE_CHECKPOINT = False
RESUME_FROM_FULL_CHECKPOINT = False
```

Full and resume modes are documented in the notebook's OPERATOR SETTINGS block.
That block deliberately describes them **in prose** rather than as literal
assignments, because a repository-wide notebook guard
(`tests/unit/test_notebooks.py`) forbids the literal full-training assignment from
appearing in any committed notebook. The guard is correct and was not weakened.

A **resolved execution summary is printed before model acquisition**, containing:
run mode, output directory, model revision, tokenizer revision, weight format,
micro-batch size, accumulation steps, effective batch size, epochs, expected
optimizer steps, mixed-precision policy, resume source, device type, config hash,
and `internal_test_accessed: false`.

`resolved_config.json` is finalized in that same cell rather than in the run gate,
because the accumulation plan needs the real training example count and the
precision policy needs the real device. The config hash therefore covers
accumulation and precision, which is what makes the resume compatibility check
meaningful.

Notebook run order is unchanged from Audit 0038 and still acquires the encoder only
after the alignment preflight passes.

## 13. Tests and Static Checks

New: `tests/unit/test_e4_full_training_hardening.py` (56 tests) covering the
active loading path using `use_safetensors=False`; resolved config recording the
`.bin` format; the pinned revision being unable to request safetensors;
expected-only MLM-head mismatch accepted and unexpected encoder mismatch rejected;
real gradient accumulation; the optimizer not stepping before a boundary; stepping
at each complete boundary; the correct final partial group and its loss scale; the
ceiling-division step calculation; manifest accounting matching real optimizer
steps; gradient clipping; every mixed-precision resolution including the bf16
fallback and CPU refusal; incompatible precision/accumulation/weight-format/config
resume rejection; smoke checkpoint rejected as full initialization; the compatible
full-resume state contract; optimizer/scaler checkpoint custody; notebook
smoke/full settings and the resolved summary ordering; no internal_test access;
and no model/cache/checkpoint file tracked in Git.

```text
env PYTHONPATH=src python -m pytest -q
1430 passed, 1 skipped in 124.27s (0:02:04)

env PYTHONPATH=src python -m pytest -q tests/unit/test_e4_full_training_hardening.py
56 passed

ruff check .
All checks passed!

ruff check notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb
All checks passed!

env PYTHONPATH=src python -m mypy
Success: no issues found in 257 source files

env PYTHONPATH=src python -m compileall -q src
<passed>

git diff --check
<passed>
```

The single skip is the pre-existing local `pyarrow` skip in
`tests/unit/test_vietmed_adapter.py`. Every modified notebook code cell was parsed
with `compile(...)`; an AST scan found no undefined or unused names.

## 14. Unchanged Protected Files and Hygiene

```text
docs/MedNorm-VI_Architecture.pdf        0d5eaa20…1e09b   unchanged
docs/audits/0036-…md                                     unchanged
docs/audits/0037-…md                                     unchanged
docs/audits/0038-…md                                     unchanged
src/mednorm_vi/evaluation/exact_mention.py               unchanged

892dc22d…2a4a  splits/train.jsonl
ed7cdd2d…f103  splits/validation.jsonl
e23acde0…e135  splits/internal_test.jsonl
a3fd365d…b8e3  manifests/corpus_manifest.json

tracked *.pt/*.pth/*.ckpt/*.safetensors/*.bin/*.zip   0
tracked artifacts|weights|caches|checkpoint paths     0
output.zip                                            absent
```

Nothing was trained locally, internal_test was not read, no organizer inference
ran, and no `output.zip` was produced.

## 15. Changed Files

Added:

- `docs/audits/0039-e4-smoke-evidence-and-full-training-hardening.md`
- `tests/unit/test_e4_full_training_hardening.py`

Modified:

- `configs/training/phase2_e4_phobert_w2ner_colab.yaml`
- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `src/mednorm_vi/training/phase2/e4_w2ner_training.py`
- `src/mednorm_vi/training/phase2/artifacts.py` (optional `training_accounting`
  manifest field, defaulted so E5 and L4 manifests are unaffected)

## 16. Limitations

1. **Full training has not run.** `READY_FOR_FULL_TRAINING` is deliberately not
   claimed; it is earned only after the regression smoke passes.
2. The Audit-0038 smoke artifact is valid evidence for the code as of that audit,
   **not** for the hardened loop.
3. `micro_batch_size` is 1 because W2NER grids are variable-sized per document.
   Larger micro-batches would need padded-grid batching, which is not implemented
   and is not claimed.
4. The precision policy is resolved against the runtime; the actual Colab GPU's
   bf16 support is only known in Colab.
5. No scheduler is configured, so no learning-rate schedule is claimed.

## 17. Safe-to-Commit Verdict

Safe to commit after review. Required tests and static checks pass, no protected
architecture/audit/evaluator file changed, governed corpus hashes are unchanged,
default feature flags are unchanged, and no weight, checkpoint, corpus, cache or
archive artifact is staged.

```bash
git add \
  docs/audits/0039-e4-smoke-evidence-and-full-training-hardening.md \
  docs/audits/README.md \
  configs/training/phase2_e4_phobert_w2ner_colab.yaml \
  notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb \
  src/mednorm_vi/training/phase2/e4_w2ner_training.py \
  src/mednorm_vi/training/phase2/artifacts.py \
  tests/unit/test_e4_full_training_hardening.py

git commit -m "fix: persist PhoBERT bin weight format and harden E4 training loop"
git push origin main
```
