# Audit 0031 — S1 Full-Training Run: Artifact Validation and Internal-Test Evaluation Path

- **Date:** 2026-07-27
- **Author:** Claude (AI agent), for human review
- **Change type:** Read-only validator for a completed S1 full-training artifact, plus the
  simplest Colab evaluation path for the governed `internal_test` split. **No commit. No push.
  No training and no retraining. No training artifact modified or regenerated. No model weights
  copied into the repository. No organizer inference. No `output.zip`.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — §15 (S1 objective), §15.2 ("every
  leaderboard decision records config, seed, Git commit, and artifact hash"), §18.1 (mandatory
  local evaluator, report by type). The PDF was not modified or moved.
- **Status:** `IMPLEMENTED_UNEXECUTED` — the validator has **not** been run against the real
  artifact, which lives on Drive and is not reachable from this machine.

Audit 0030 is committed (`36c520d`) and is not modified. The next immutable number is 0031.

## 1. Reported run

The first full S1 mention-extraction training run completed at
`artifacts/s1_mention_full_training_v1`:

```text
run_completed              true
status                     FULL_TRAINING
completed_epochs           4
completed_optimizer_steps  2976
best_metric_key            validation_span_micro_f1
best_metric                0.7194053623573136

best.pt    a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
latest.pt  0011cae4e8725a3133b3099a2f5bb1dc8b87e06261f3d6cdf9b9c5be510ed563
```

These values are **as reported by the operator**. They match the tracked plan exactly — 4 epochs
and `744 × 4 = 2976` optimizer steps from `derive_schedule()` over 23,799 supervised train
examples at effective batch 32 — which is consistent, but consistency is not verification.

**Nothing in this milestone asserts the artifact is valid.** The artifact directory is a Colab
Drive path and is not reachable from the machine this audit was prepared on, so the validator
was built and unit-tested here and must be **executed against the real files** before any claim
is made. See §7.

## 2. Read-only artifact validator

`validate_full_training_artifact()` in `src/mednorm_vi/training/s1_full_training.py`. It only
reads: no file in the artifact directory is created, modified, regenerated or copied, and no
weights ever enter the repository.

Required files, all six checked for existence and reported individually:

```text
checkpoints/best.pt          checkpoints/latest.pt
logs/training_history.jsonl  resolved_config.json
validation_metrics.json      training_manifest.json
```

Conditions, each checked against a recorded value rather than key presence:

| Group | Conditions |
| --- | --- |
| Run identity | `stage_id == S1`; `status == FULL_TRAINING`; `smoke_only_not_full_training is false`; `run_completed is true`; `interrupted_reason` empty; `safe_to_resume is true` |
| Accounting | `completed_epochs == hyperparameters.num_epochs`; `completed_optimizer_steps == schedule.total_optimizer_steps` |
| Revision | pinned revision is an immutable 40-hex hash, equals the operator-supplied revision, and equals `tokenizer_revision` |
| Initialization | `initialize_from == pretrained_base` **and** `initialized_from_smoke_checkpoint is false` |
| Criterion | `best_checkpoint_criterion` is `validation_span_micro_f1` / `max`; best metric is a real number in `[0, 1]` |
| Config hash | `resolved_config.json` re-hashes to `manifest.config_sha256`, and pins the same revision |
| Metrics file | `validation_metrics.json` agrees with the manifest on the best metric |
| History | `training_history.jsonl` parses and holds exactly one validation record per completed epoch |
| Checkpoint hashes | three-way agreement per checkpoint (recomputed from bytes == manifest == operator-supplied); `best.pt` and `latest.pt` must not be byte-identical |
| Checkpoint schema | every `CHECKPOINT_REQUIRED_KEYS` field present; `mode == FULL_TRAINING`; the pinned revision and label space match; `config_sha256` matches the manifest |
| Hygiene | no base-model cache files inside the artifact directory |

**No run-specific digest is hardcoded** — a test parses the module and asserts no 64-hex string
appears in it. A missing or malformed expected hash never passes: the validator reports the
recomputed digest inside the failure message so the operator can confirm it without editing
source. Every failed condition is reported, not just the first.

### Confirming the smoke checkpoint was not the initializer

Three independent checks, any of which fails the artifact:

1. `model.initialize_from` must equal `pretrained_base`;
2. `model.initialized_from_smoke_checkpoint` must be exactly `False`;
3. neither checkpoint payload may carry `mode == SMOKE_ONLY`.

### Torch boundary

The module stays free of Torch. Checkpoint *tensors* are loaded by the caller, which passes only
the metadata (`mode`, `epoch`, `global_step`, `pinned_model_revision`, `config_sha256`,
`entity_type_order`, and booleans marking which `*_state_dict` keys exist) into
`checkpoint_payloads`. When omitted, the file and manifest checks still run and
`checkpoint_schema_checked` is reported as `false` rather than silently passing.

## 3. Entry point

```bash
env PYTHONPATH=src python3 -m mednorm_vi.training.cli validate-full-training \
  --artifact-dir /content/drive/MyDrive/MedNorm-VI/artifacts/s1_mention_full_training_v1 \
  --revision f89e80b461e86f9cfc1c84019bd819830c24b6c5 \
  --expected-best-sha256   a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c \
  --expected-latest-sha256 0011cae4e8725a3133b3099a2f5bb1dc8b87e06261f3d6cdf9b9c5be510ed563 \
  --load-checkpoints \
  --report reports/s1_full_training_validation/validation.json
```

Exit status 0 when every condition passes, 1 otherwise. `--load-checkpoints` is the only mode
that needs Torch; without it the file, manifest, hash, config and history checks all still run.
`--report` is optional and writes under `reports/**`, which is git-ignored.

## 4. Internal-test evaluation path

`notebooks/MedNorm_S1_Mention_InternalTest_Evaluation.ipynb` — read-only, GPU, two-pass.

Order: two-pass bootstrap → forced restart → NumPy/AdamW ABI preflight → scoped dependency
health → GPU/Drive → **full-training artifact validation gate** → governed corpus gate →
VnCoreNLP production segmentation (asserted) → slow tokenizer at the pinned revision →
alignment preflight over `internal_test` → load `best.pt` → forward-only scoring → report.

Guarantees, all enforced by test:

* no `backward`, no `optimizer.step`, no scheduler; `model.eval()`, `torch.no_grad()`, and every
  parameter set to `requires_grad_(False)`;
* the artifact-validation gate precedes model acquisition and `load_state_dict`;
* results are written **only** to a separate directory, `artifacts/s1_mention_internal_test_eval_v1/internal_test_evaluation.json`;
  the training artifact is never a write target, and the report records
  `training_artifact_modified: false`;
* no organizer inference and no `output.zip`; both recorded as `false` in the report;
* no digest or revision is hardcoded — both come from runtime inputs
  (`MEDNORM_EXPECTED_BEST_CHECKPOINT_SHA256`, `MEDNORM_PINNED_MODEL_REVISION`), with the pinned
  revision falling back to the artifact's own manifest and still validated against it.

Metrics are `MentionMetrics` — per-type token precision/recall/F1 plus micro/macro aggregates
and span-level micro P/R/F1 — the same accumulator the training run used, so the internal-test
number is directly comparable to the reported validation figure. Per-type reporting is what
spec §18.1 requires.

**This is not the organizer metric.** Span-level micro F1 over contiguous typed token runs is a
training-time proxy; character-exact scoring, assertions and candidate Jaccard belong to L4/L8
and the mandatory local evaluator.

## 5. Tests and static validation

`tests/unit/test_s1_full_training_validation.py` (51, new): a complete run validates; the
checkpoint schema is verified when payloads are supplied and reported as unchecked when not;
each of the six required files is individually required; a missing manifest stops immediately;
hashes must match the manifest **and** the operator-supplied value; a missing or malformed
expected hash blocks and echoes the recomputed digest; no 64-hex digest exists in the module;
byte-identical best/latest rejected; run state and accounting enforced (7 parametrized cases);
pinned-revision rules (3 cases); smoke-checkpoint initialization rejected two ways plus a
`SMOKE_ONLY` payload; every `CHECKPOINT_REQUIRED_KEYS` field required (parametrized); revision
and label-space drift rejected; config-hash and metrics-file agreement; implausible best metric
rejected (4 cases); history must hold one validation record per epoch; base-model cache
rejected; all failures reported, not just the first.

`tests/unit/test_notebook_placeholders.py` (+5) and `tests/unit/test_notebooks.py` (inventory
12 → 13) cover the evaluation notebook's read-only guarantees.

```text
env PYTHONPATH=src python3 -m pytest -q          -> 1099 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 224 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
S1 notebook code-cell AST parse                   -> 5 notebooks, 0 syntax errors
```

## 6. Exact changed files

**Added (3):**
```text
docs/audits/0031-s1-full-training-run-validation-and-internal-test-evaluation.md
notebooks/MedNorm_S1_Mention_InternalTest_Evaluation.ipynb
tests/unit/test_s1_full_training_validation.py
```

**Modified (5):**
```text
docs/audits/README.md                              (0031 index entry)
src/mednorm_vi/training/cli.py                     (validate-full-training subcommand)
src/mednorm_vi/training/s1_full_training.py        (read-only artifact validator)
tests/unit/test_notebook_placeholders.py           (+5 evaluation-notebook assertions)
tests/unit/test_notebooks.py                       (notebook inventory 12 -> 13)
```

Counted from git: **3 added, 5 modified — 8 changed tracked files.**

Model weights, caches, corpora and generated reports remain outside Git: `models/**`, `data/**`,
`reports/**` and `.venv/` are all ignored, and nothing from the artifact directory was copied in.

## 7. Limitations and honest status

1. **The artifact has NOT been validated.** It lives at a Colab Drive path unreachable from this
   machine. The reported metrics and hashes are the operator's, unverified here. Running §3 is
   what turns this into a validated result.
2. **The best metric is a validation figure, not a test figure.** `validation_span_micro_f1`
   0.7194 was measured on the split used for checkpoint selection, so it is optimistically
   biased. The internal-test evaluation in §4 gives the first held-out number.
3. **Span-level token-index F1 is a proxy** for the organizer's character-exact entity scoring.
4. Only 3 of the 5 spec entity types occur in the governed corpus; TEST_NAME and TEST_RESULT
   received no supervision, so their metrics are undefined rather than poor.
5. This audit covers S1 mention extraction only. Assertions (S2), linking (S3/S4), the metric
   decoder (L8) and the validator (L9) are untouched.

**Status:** `IMPLEMENTED_UNEXECUTED`. The validator and evaluation notebook are implemented and
unit-tested; neither has been executed against the real artifact. No retraining was performed and
no training artifact was modified.

## 8. Recommended next step

Run §3 against the real artifact. If it validates, run the evaluation notebook to obtain the
first held-out `internal_test` numbers, then record them in a follow-up audit before moving to
S2 or to the L4 boundary/type resolver.
