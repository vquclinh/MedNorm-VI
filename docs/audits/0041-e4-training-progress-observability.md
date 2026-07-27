# Audit 0041 - E4 Training Progress Observability, Epoch/Sample Heartbeats, and Training/Validation ETA

Date: 2026-07-27

## 1. Objective and Scope

This append-only audit records a real T4 High-RAM E4 full-training attempt that
started correctly and then went silent for roughly an hour, and the observability
changes that make progress visible.

**This is not a correctness fix.** The training loop was working; it simply had no
logging between the first micro-batch and the end-of-epoch summary, so an operator
could not distinguish progress from a hang.

Nothing in the model, data, optimizer, loss, accumulation semantics, precision
policy, checkpoint compatibility, corpus governance, internal-test policy or
artifact format is altered. Nothing was trained locally, internal_test was not
read, no organizer inference ran and no `output.zip` was produced.

### Status

| Subject | Status |
| --- | --- |
| Previous T4 attempt | `TRAINING_STARTED`, `FIRST_MICRO_BATCH_PASSED`, `STOPPED_BY_OPERATOR`, `CHECKPOINT_UNAVAILABLE` |
| Newly changed repository | `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, `READY_FOR_FRESH_A100_FULL_RUN` |

**No training completion is claimed.** A100 is the operator's chosen runtime for
the next attempt; the architectural requirement established in Audit 0040 remains
**CUDA**, with T4 High-RAM, L4 and A100 all permitted.

## 2. Initial Git State

```text
pwd                       /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
git branch --show-current main
git status -sb            ## main...origin/main
git status --short        <clean>

git log --oneline -4
2dade89 fix: harden E4 Colab corpus IO and stream W2NER contracts
7423eb1 fix: persist PhoBERT bin weight format and harden E4 training loop
d5ad5d1 fix: decouple E4 W2NER grid words from VnCoreNLP model words
020cae7 fix: repair E4 Colab bootstrap and PhoBERT alignment
```

**Audit 0040 is committed** at `2dade89`, so this milestone creates the
append-only Audit 0041 rather than amending 0040. Audit 0040 is left untouched.

Architecture PDF read in full before any change; SHA-256
`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`, unchanged.

## 3. The Real T4 High-RAM Attempt

A fresh T4 High-RAM full-training attempt successfully reached:

```text
PhoBERT-large encoder loaded             YES
first forward pass completed             YES
first backward pass completed            YES
memory_after_first_micro_batch printed   YES
host RSS remained low                    YES   (the Audit-0040 streaming fix held)
GPU memory remained stable               YES
```

The training cell then produced **no new output for roughly one hour**. At that
point:

```text
checkpoints/            empty
logs/                   empty
epochs completed        0
validations run         0
checkpoint available    NO
traceback / CUDA OOM    none shown
```

The operator stopped and deleted the runtime.

`checkpoints/` and `logs/` were empty for a simple, expected reason: **epoch 1 had
not completed**, and every artifact in the Audit-0040 design is written at an
epoch boundary. Nothing was lost that had been produced; nothing had been produced
yet.

**No full checkpoint exists from that run, and no full-training success is
claimed.** The Audit-0040 host-RAM fix is separately confirmed working: RSS stayed
low where the previous attempt had reached ~30 GB.

## 4. Why the Committed Interval Is 100, Not 5

Logging every five samples was considered and rejected as the committed full-run
default:

```text
full-run backward passes                     405,912
heartbeats at every 5 samples          ~81,000 per run   (6,766 per epoch)
heartbeats at the committed policy       4,188 per run   (349 per epoch)
```

Five would flood the notebook, bloat the progress log, and read GPU counters far
more often than is useful. The committed policy instead logs **each of the first
ten samples individually** — so motion is visible within seconds of an epoch
starting, which is exactly what the T4 run lacked — and then **every 100 training
samples**, plus always the final sample of the epoch.

An operator may temporarily set `PROGRESS_LOG_EVERY_N_TRAIN_SAMPLES = 5` for a
smoke or debug run. Training semantics are identical either way; only the log
volume changes.

## 5. Progress Configuration

Committed defaults, validated and recorded in both `resolved_config.json` and
`training_manifest.json`:

```text
PROGRESS_ENABLED = True
PROGRESS_LOG_FIRST_N_SAMPLES = 10
PROGRESS_LOG_EVERY_N_TRAIN_SAMPLES = 100
PROGRESS_LOG_EVERY_N_VALIDATION_SAMPLES = 50
PROGRESS_BAR_ENABLED = True
PROGRESS_ROLLING_LOSS_WINDOW = 100
```

`ProgressConfig` rejects a negative first-N and any non-positive or non-integer
interval. With `PROGRESS_ENABLED = False` no heartbeat is emitted and training is
otherwise unchanged.

Recording the progress settings in `resolved_config.json` changes
`CONFIG_SHA256`. That is the intended behaviour of a changed configuration; the
**resume compatibility contract itself is unchanged** (the same ten fields), and
no completed full checkpoint exists to invalidate.

## 6. Training Heartbeat

`src/mednorm_vi/training/phase2/e4_progress.py` emits a `train_progress` record
for each of the first ten samples, then every 100, and always the final sample of
the epoch. Every heartbeat carries all 29 required fields:

```text
stage run_mode epoch total_epochs sample total_samples epoch_percent
accumulation_slot gradient_accumulation_steps
epoch_backward_passes global_backward_passes
epoch_optimizer_steps global_optimizer_steps
current_loss rolling_mean_loss
samples_per_second optimizer_steps_per_second
epoch_elapsed_seconds run_elapsed_seconds epoch_eta_seconds run_eta_seconds
learning_rate precision_mode gpu_name
gpu_allocated_gib gpu_reserved_gib gpu_peak_allocated_gib
process_rss_gib system_available_gib
```

Epoch and sample numbers are **1-based**. `time.monotonic()` is used for all
elapsed and ETA arithmetic. Percentages are clamped to `[0, 100]`, ETAs are
clamped to `>= 0`, and zero elapsed time returns `0.0` instead of dividing by
zero.

The optimizer-step boundary is reported through `accumulation_slot` and the step
counters inside the same heartbeat rather than as a second log stream, so there is
no duplicate noise.

**No synchronization was added.** The loop already computed
`float(loss.detach().float().cpu())` for its existing mean-loss calculation; the
bounded `RollingLoss` (window 100) reuses that value. Individual losses beyond the
window are never retained.

All output uses `print(..., flush=True)`. No clinical text, gold entity or corpus
content is ever logged — asserted by a test that greps the module for corpus
accessors and checks that no heartbeat value is a list, dict or tuple.

## 7. Validation Heartbeat

Validation over 1,045 examples could also look like a hang, so a
`validation_progress` record is emitted at sample 1, every 50 samples and the last
sample, carrying: epoch, total_epochs, sample, total_samples, validation_percent,
elapsed_seconds, samples_per_second, eta_seconds, predicted_mentions_so_far,
gold_mentions_so_far, gpu_allocated_gib, gpu_reserved_gib, process_rss_gib,
system_available_gib.

Only **counts** are reported — never predictions, spans or text. Nothing is
retained across examples: `predicted` and `gold` sets are deleted as the loop
advances, exactly as before. The exact validation metric calculation and
best-checkpoint selection are byte-for-byte unchanged, and internal_test is never
touched.

## 8. Progress Bar

One `tqdm.auto` bar per training epoch and one per validation pass:

- total is the **streamed** source example count (`source.example_count`), so the
  iterator is never materialized to measure it;
- updated once per consumed sample;
- postfix shows rolling loss, optimizer steps, samples/second and ETA;
- closed in a `finally` block at pass completion;
- disabled by `PROGRESS_BAR_ENABLED`.

tqdm is already present transitively (Transformers depends on it) and was **not**
added as a new hard requirement. When it is unavailable, `make_progress_bar`
returns an interface-compatible `NullProgressBar` so a training run never fails
for want of a progress bar. The structured heartbeat is emitted regardless of
whether the bar is active.

## 9. Epoch Lifecycle Records

```text
epoch_start                  epoch, total_epochs, train_examples,
                             expected_optimizer_steps_this_epoch,
                             expected_backward_passes_this_epoch,
                             resume_start_epoch, precision_mode, gpu_name,
                             timestamp_utc
epoch_training_complete      epoch, backward_passes, optimizer_steps,
                             mean_training_loss, elapsed_seconds,
                             samples_per_second
epoch_validation_complete    epoch, exact_precision, exact_recall, exact_f1,
                             best_f1_before_epoch, is_new_best
epoch_checkpoint_persisted   epoch, latest_checkpoint_path,
                             latest_checkpoint_sha256, best_checkpoint_updated,
                             best_checkpoint_path, persistent_custody_verified,
                             elapsed_seconds
```

Each potentially slow persistence step announces itself first, so checkpoint
serialization or a Drive sync can never be mistaken for a training hang:

```text
saving checkpoint to local staging
validating checkpoint reload
checking Drive health
syncing checkpoint to Drive
verifying persistent SHA-256
persistence complete
```

At clean completion a `full_training_complete` record reports epochs_completed,
global_backward_passes, global_optimizer_steps, best_validation_exact_f1,
best_epoch, total_elapsed_seconds, average_samples_per_second,
best_checkpoint_sha256, latest_checkpoint_sha256, artifact_validator_ok and
`internal_test_accessed: false`.

## 10. Local Progress JSONL

A structured log is appended locally, one JSON object per heartbeat, flushed per
record:

```text
/content/mednorm_vi_runtime/logs/training_progress.jsonl
```

It is **never** written to Drive per sample. It is synced through the **unchanged**
Audit-0040 local-first verified persistence mechanism at exactly three governed
boundaries — epoch completion, clean training completion, and controlled exception
handling — landing at `logs/training_progress.jsonl` in the artifact.

`ProgressLog.append` is best-effort: a write failure is counted and reported in
`progress_write_failures` / `progress_last_error` but never raises, so telemetry
can never corrupt model or optimizer state. Checkpoint correctness stays strictly
more important than saving a heartbeat.

## 11. Controlled Exception Reporting

The epoch body is wrapped so that any failure prints a concise `training_failed`
record — epoch, sample, global_backward_passes, global_optimizer_steps,
exception_type, exception_message, GPU allocated/reserved/peak GiB, process RSS,
system available, local_progress_log_path, latest_persistent_checkpoint_available
— attempts one best-effort progress-log sync, and then **re-raises**.

The error is never caught and suppressed. `latest_persistent_checkpoint_available`
is computed from whether a persistent `latest.pt` actually exists on disk, so
resumability is never claimed without a verified checkpoint.

## 12. Training Semantics Are Unchanged

Verified by test and by inspection:

```text
full train examples                 33,826
validation examples                  1,045
full epochs                             12
micro_batch_size                         1
gradient_accumulation_steps              8
effective_batch_size                     8
expected optimizer steps            50,748
expected backward passes           405,912
final partial accumulation group         2
learning rate / optimizer / max grad norm    unchanged
validation exact-F1 selection                unchanged
model / tokenizer revisions                  unchanged
use_safetensors                              False
pretrained weight format                     pytorch_model.bin
internal_test access                         forbidden
T4 -> fp16 + GradScaler, L4/A100 -> bf16     unchanged
```

No mid-epoch optimizer checkpoint was added. Resume semantics, the streaming
contract source and the local materialization policy are untouched. No contract
list is reconstructed, and **no `gc.collect()` or `torch.cuda.empty_cache()` per
sample** was introduced — asserted by test.

Heartbeat construction is a pure read of the counters: a test passes a counter
dictionary in and asserts it is unmodified afterwards.

## 13. Tests and Static Checks

New: `tests/unit/test_e4_progress_observability.py` (46 tests) covering the
committed interval being 100 and not 5; the first ten samples each emitting;
samples 11–99 not emitting; every hundredth sample emitting; the final epoch
sample always emitting; disabled progress emitting nothing; bounded heartbeat
volume (349/epoch vs >6,000 at interval 5); every required heartbeat field;
1-based epoch and sample numbers; zero-elapsed not dividing by zero; ETA
non-negative and zero at completion; percentages bounded; bounded rolling loss
under 10,000 observations; validation emitting at 1, every 50 and the last sample;
validation reporting counts only; all four epoch lifecycle records; every
persistence phase message; the final-run summary; valid local JSONL; a log
failure being reported but not fatal; `emit` printing flushed; the log staying
local and syncing only at the three boundaries; the failure record being complete
and not claiming resumability; report-then-re-raise; progress bar disable, total
and postfix; one bar per epoch and per validation pass; no materialization of a
streamed source; a new iterator per epoch and per validation pass; unchanged
optimizer/backward accounting; unchanged precision resolution; no corpus text
logged; no internal_test access; notebook cells parsing; and no
checkpoint/model/cache/archive tracked in Git.

```text
env PYTHONPATH=src python -m pytest -q
1532 passed, 1 skipped in 126.13s (0:02:06)          [was 1486]

env PYTHONPATH=src python -m pytest -q tests/unit/test_e4_progress_observability.py
46 passed

ruff check .                                             All checks passed!
ruff check notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb   All checks passed!
env PYTHONPATH=src python -m mypy      Success: no issues found in 259 source files
env PYTHONPATH=src python -m compileall -q src           <passed>
git diff --check                                         <passed>
```

The single skip is the pre-existing local `pyarrow` skip. Every notebook code cell
was parsed with `compile(...)`; an AST scan found no undefined or unused names. No
existing test was weakened — 46 added, none removed or relaxed.

One tooling note: `tqdm` ships no type stubs, so it was added to the existing
`pyproject.toml` mypy override list alongside the other optional heavy frameworks
(`torch`, `numpy`, `transformers`, `py_vncorenlp`). No runtime dependency was
added.

## 14. Protected Files and Hygiene

```text
docs/MedNorm-VI_Architecture.pdf     0d5eaa20…1e09b        unchanged
committed audits (0038, 0039, 0040)                        unchanged
src/mednorm_vi/evaluation/exact_mention.py                 unchanged
governed corpus + manifest                                 unchanged

892dc22d…2a4a  splits/train.jsonl
ed7cdd2d…f103  splits/validation.jsonl
e23acde0…e135  splits/internal_test.jsonl
a3fd365d…b8e3  manifests/corpus_manifest.json

tracked *.pt/*.pth/*.ckpt/*.safetensors/*.bin/*.zip        0
tracked artifacts|weights|caches|checkpoint|.claude paths  0
CLAUDE.md / AGENTS.md tracked                              0
output.zip                                                 absent
```

Committed notebook smoke defaults are retained; full mode is not enabled in Git.

## 15. Changed Files

Added:

- `docs/audits/0041-e4-training-progress-observability.md`
- `src/mednorm_vi/training/phase2/e4_progress.py`
- `tests/unit/test_e4_progress_observability.py`

Modified:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `src/mednorm_vi/training/phase2/e4_w2ner_training.py` (optional `progress=`
  argument to `build_e4_resolved_config`)
- `pyproject.toml` (mypy override for the stub-less optional `tqdm`)

## 16. Limitations

1. **No full training has run**, and none is claimed.
2. The one-hour silence was never reproduced locally; it is the operator's
   observation. The change addresses its cause — absent telemetry — not a measured
   throughput figure.
3. Heartbeat cadence is a judgement call, not an optimum. 349 records per epoch
   was chosen for readability; the interval is configurable.
4. Progress output cannot prove throughput in advance. If the next run is slow,
   the heartbeats will now say so with `samples_per_second` and an ETA instead of
   printing nothing.

## 17. Remaining Colab Work

A **fresh full run** is required. The operator has chosen A100 for the next
attempt; CUDA remains the only architectural requirement (T4 High-RAM, L4 and A100
are all permitted per Audit 0040).

The run should show: `epoch_start`, heartbeats for samples 1–10 within seconds,
then every 100 samples with live `samples_per_second` and ETA, a per-epoch
progress bar, validation heartbeats every 50 samples, the four epoch lifecycle
records, named persistence phases, 50,748 optimizer steps and 405,912 backward
passes, and a final `full_training_complete` summary — with internal_test
untouched throughout.

## 18. Safe-to-Commit Verdict

Safe to commit after review. Required tests and static checks pass, no protected
architecture, audit, evaluator or corpus file changed, governed hashes are
unchanged, no existing test was weakened, and no weight, checkpoint, corpus, cache
or archive artifact is staged.

```bash
git add \
  docs/audits/0041-e4-training-progress-observability.md \
  docs/audits/README.md \
  notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb \
  src/mednorm_vi/training/phase2/e4_progress.py \
  src/mednorm_vi/training/phase2/e4_w2ner_training.py \
  pyproject.toml \
  tests/unit/test_e4_progress_observability.py

git commit -m "feat: add E4 training progress heartbeats and ETA"
git push origin main
```
