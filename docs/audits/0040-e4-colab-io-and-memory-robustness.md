# Audit 0040 - E4 Colab Full-Run I/O Robustness, Local Governed-Corpus Materialization, and Memory-Bounded W2NER Streaming

Date: 2026-07-27

## 1. Objective and Scope

This append-only audit records a real A100 full-training attempt that passed the
complete alignment preflight and then died on a Google Drive FUSE transport
failure, plus the repository changes that remove both that failure mode and the
memory design that made a retry unsafe.

Not started here: E5, learned L4, S2 assertion, S3 retrieval, S4 reranking, Qwen,
internal_test, organizer inference, packaging, `output.zip`. Nothing was trained
locally and no smoke or full run was executed locally.

`docs/MedNorm-VI_Architecture.pdf`, Audit 0038, Audit 0039,
`src/mednorm_vi/evaluation/exact_mention.py`, the governed corpus files and the
corpus manifest are all unmodified.

### Status

| Subject | Status |
| --- | --- |
| Audit-0039 regression smoke v2 | `SMOKE_EXECUTED`, `ARTIFACT_VALIDATED` |
| Failed real full attempt | `ALIGNMENT_PREFLIGHT_PASSED`, `TRAINING_NOT_STARTED`, `ARTIFACT_UNAVAILABLE` |
| Newly changed repository | `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, `READY_FOR_FRESH_CUDA_FULL_RUN` |

**No full-training success is claimed.**

## 2. Initial Git State

```text
pwd                       /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
git branch --show-current main
git status -sb            ## main...origin/main
git status --short        <clean>
git diff --check          <clean>

git log --oneline -4
7423eb1 fix: persist PhoBERT bin weight format and harden E4 training loop
d5ad5d1 fix: decouple E4 W2NER grid words from VnCoreNLP model words
020cae7 fix: repair E4 Colab bootstrap and PhoBERT alignment
03fc55f feat: complete phase2 Colab training readiness
```

Audit 0039 is committed at `7423eb1` and left immutable. Architecture PDF read in
full before any change; SHA-256
`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`, unchanged.

## 3. Audit-0039 Regression Smoke v2 — Passed

```text
mode                        smoke
backward_passes             8
optimizer_steps             1
effective_batch_size        8
precision_mode              bf16
pretrained_weight_format    pytorch_model.bin
validator.ok                true
failures                    []
internal_test_accessed      false
```

Real gradient accumulation, bf16 precision and the `.bin` weight path all
executed. This audit does not disturb any of it.

## 4. The Failed Real Full Attempt

A fresh A100 full-training attempt reached and **passed** the complete alignment
diagnostic:

```text
stage                                     full_corpus_alignment_preflight
diagnostic_sha256   86ad2fc50bb9084d3a6ba2328c9b571a77aa0067c82fcc0481bb70b5d2c34bc7
passed                                    true

examples                                  34,871
entities                                  13,711
entities_fixed_by_atomic_words               159
entities_unalignable_after_atomic_words        0
atomic_projection_violations                   0
silent_exclusions                              0
examples_exceeding_max_model_tokens            0
examples_exceeding_max_words                   0
max_atomic_word_count                        162
max_model_word_count                         162
max_phobert_subtoken_count                   213
encoder_downloaded                         false
internal_test_accessed                     false
```

`max_phobert_subtoken_count = 213` is the value Audit 0038 could not measure
locally, now measured in Colab with the real tokenizer, comfortably inside the
512-token ceiling.

Immediately afterwards, while `load_governed_w2ner_contracts` opened the governed
train split a **second** time, Drive FUSE failed:

```text
OSError: [Errno 107] Transport endpoint is not connected

failing operation:
    with Path(split_path).open("r", encoding="utf-8") as handle:

source paths still on:
    /content/drive/MyDrive/MedNorm-VI/...
```

The operator manually remounted Drive and copied the governed splits to local
Colab storage:

```text
/content/mednorm_vi_runtime/splits/train.jsonl
/content/mednorm_vi_runtime/splits/validation.jsonl
```

That manual recovery succeeded. **A correction on the reported digests:** the
train value quoted in the milestone report is 63 hex characters —
`892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4` — one short of a
SHA-256. The authoritative governed digest, recomputed from the corpus in this
repository, is:

```text
train       892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a
validation  ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103
```

The pasted validation digest is correct. The truncation is called out because the
new materialization code compares digests exactly and rejects any expectation that
is not 64 hex characters — a 63-character value would fail every run. A test pins
that rejection.

A second problem became visible during the recovery:

- the first failed contract-construction attempt raised system RAM to roughly
  **30 GB**;
- the exception traceback and partially constructed objects could remain reachable
  in the notebook process, so rerunning the cell began allocating on top of that
  ~30 GB instead of returning to the original level;
- constructing all 33,826 train W2NER contracts and their O(n²) relation grids in
  one Python list is therefore not an acceptable full-training design.

The operator deleted the runtime before training began.

```text
encoder acquired          NO
optimizer step            NO
full checkpoint created   NO
internal_test accessed    NO
```

## 4a. GPU Runtime Policy

Recorded explicitly, because the previous wording implied a hardware requirement
that does not exist:

* the previous real failed attempt **did** happen on an **A100**;
* **A100 is not an architectural requirement**. Full training requires **CUDA**,
  nothing more;
* **T4 High-RAM is permitted**, as are L4 and A100;
* a runtime is **never** rejected because its device name is T4 or L4. The only
  device gate in the code is `assert_full_training_device(device_type)`, which
  rejects CPU alone — no GPU-name check exists anywhere on the active path;
* the Audit-0040 streaming design fixes **system-RAM** growth, which is
  independent of GPU type. The ~30 GB problem was host RAM, not VRAM, so it is
  fixed identically on T4, L4 and A100;
* a **T4 may still fail from GPU VRAM exhaustion, or be substantially slower**.
  PhoBERT-large plus an O(n²) relation grid is not guaranteed to fit a 16 GB T4;
* such a failure **must be reported as a runtime resource limitation**, never as
  a corpus-streaming or system-RAM failure. The two are distinguishable in the
  logs: streaming and materialization report their own errors, and a CUDA OOM
  surfaces as a CUDA error after the encoder is resident.

Precision resolution is unchanged and is derived from the runtime capability,
never from the device name:

| Runtime | `is_bf16_supported()` | Resolved precision |
| --- | --- | --- |
| T4 High-RAM (compute capability 7.5) | false | `fp16` + `GradScaler` |
| L4 | true | `bf16` |
| A100 | true | `bf16` |
| CPU | n/a | `fp32`, and full training is **rejected** |

The notebook reports `gpu_name` in the resolved execution summary for
observability, alongside `gpu_name_used_as_gate: false`.

## 5. Root Causes

1. Governed split files were read repeatedly across a long-lived Drive FUSE mount.
2. The alignment preflight read the corpus, then full contract construction read
   it **again**, directly from Drive.
3. Full mode built and retained `train_contracts` and `validation_contracts` as
   complete lists; each contract carries an O(n²) grid.
4. An exception during list construction could leave partial objects reachable
   through the notebook traceback, so a retry allocated a second large body.
5. The full path required the operator to paste manual remount, local-copy and
   cleanup snippets.

## 6. Drive Health Check and Bounded Remount

`src/mednorm_vi/training/phase2/e4_runtime_io.py`:

```text
DriveAdapter                 injected mount / flush_and_unmount / force_unmount
default_drive_probe          real stat + scandir + 1-byte read, not Path.exists()
ensure_drive_healthy         at most ONE remount, then fail clearly
is_transport_failure         errno 107 / EIO / ESHUTDOWN, or the message text
DRIVE_HEALTH_PROBE_VERSION   e4-drive-health-probe-v1
```

Behaviour, all pinned by tests:

- a healthy Drive is **never** remounted;
- errno 107 or `Transport endpoint is not connected` triggers exactly one
  remount: `flush_and_unmount()` when available, a bounded force-unmount only if
  the flush itself fails, then `drive.mount("/content/drive",
  force_remount=True)`, then the probe re-runs;
- a failed post-remount probe raises `DriveHealthError`; there is no retry loop
  and exactly two probes occur;
- a **non-transport** failure — a missing or corrupted source file — is re-raised
  untouched and never masked by a remount.

The probe does real filesystem work because a broken FUSE endpoint frequently
still answers `Path.exists()` while every real operation raises ENOTCONN, which is
exactly how the real run failed.

Mount operations are injected, so the tests never import `google.colab`.

Recorded in the runtime summary and manifest: `drive_health_checked`,
`drive_remount_attempted`, `drive_remount_succeeded`, `drive_health_probe_version`.

## 7. Local Governed-Split Materialization

After Drive mount, repository checkout, corpus resolution and source-hash
validation — and **before** the alignment diagnostic, contract construction,
encoder acquisition and training — only train and validation are copied to:

```text
/content/mednorm_vi_runtime/splits/train.jsonl
/content/mednorm_vi_runtime/splits/validation.jsonl
```

`materialize_split` / `materialize_governed_splits`:

- streams the copy in 1 MiB chunks; a JSONL is never loaded whole into memory;
- writes to `<name>.jsonl.tmp-<pid>`, then `os.replace` — an atomic swap;
- verifies copied **size** and exact **SHA-256** before the replace, and re-verifies
  after it, deleting the destination on any mismatch;
- reuses an existing local copy only when its digest matches;
- rejects an expected digest that is not 64 hex characters;
- refuses `internal_test` **by name**, both per split and in the batch entry point,
  so no directory-wide copy can expose it.

The active `TRAIN_SPLIT_PATH` and `VALIDATION_SPLIT_PATH` are switched to the local
paths before the preflight, and the notebook asserts neither still starts with
`/content/drive`. No later stage reopens the corpus from Drive.

Recorded in resolved config and manifest: `drive_source_train_path`,
`drive_source_validation_path`, `runtime_train_path`, `runtime_validation_path`,
`train_sha256`, `validation_sha256`, `train_size_bytes`, `validation_size_bytes`,
`local_materialization_version`, `local_materialization_reused`.

The governed SHA-256 values remain authoritative, so copying cannot modify
examples, ordering, text, offsets, entities or split identity. This was executed
locally against the real governed corpus: both splits materialized and reproduced
their authoritative digests, and a second pass reused them.

## 8. Memory-Bounded W2NER Streaming

`GovernedW2NERContractSource` replaces the full-corpus lists. It accepts the split
path, expected SHA-256, expected example count, tokenizer, contract-build
callable, `max_words`, `max_model_tokens`, the input contract version and an
optional bounded `max_rows`.

- iterating opens the local JSONL and yields **one** contract at a time;
- every `iter_contracts()` is an independent pass, so each epoch and each
  validation sweep gets a fresh iterator;
- the yielded contract is released as the consumer advances — nothing accumulates;
- deterministic file order is preserved;
- `verify_identity()` re-checks the local digest before streaming;
- a short stream is rejected against the governed example count;
- `internal_test` is refused by name.

`assert_not_materialized()` is a structural guard: handing training a `list`,
`tuple`, `set` or `dict` of contracts raises. The notebook applies it to both
sources and again inside `run_training`.

`list(...)` is not used around the iterator. No `gc.collect()` per example and no
per-example CUDA cache clearing were added. Smoke uses the **same** streaming path
with `max_rows = SMOKE_ROWS`, so smoke and full share data semantics.

## 9. Bounded Reporting and Grid Statistics

`BoundedAlignmentReport` keeps a fixed sample (three entries) plus scalar
aggregates — examples, max atomic words, max model words, max encoded tokens,
label count. The previous notebook appended one report dictionary per example,
i.e. 34,871 of them, on top of the contracts.

`grid_target_statistics.json` still reports train and validation example counts,
maximum atomic words, label count, train and validation hashes, tokenizer class,
`tokenizer_is_fast`, input contract version, grid word surface and the diagnostic
SHA-256 — now sourced from the completed diagnostic plus one bounded streaming
scan that retains nothing, and additionally records
`contracts_materialized: false`.

The full alignment diagnostic is unchanged and still proves 13,711 / 13,711
entities aligned, zero unalignable, zero silent exclusions and zero projection
violations.

## 10. Memory-Safety Reporting

`memory_snapshot(label)` reads `/proc/self/status` (`VmRSS`) and `/proc/meminfo`
(`MemAvailable`) with no new dependency, and degrades to `-1` rather than
guessing. The notebook prints it before the preflight, after the preflight, after
materialization, after the bounded statistics scan, before encoder acquisition,
after the first training micro-batch, and after artifact persistence.

**No hard universal RAM ceiling is claimed from local tests.** What is asserted is
structural: full mode does not materialize all contracts. Because the full-corpus
list no longer exists, a failed source-open cannot retain one, so no manual
IPython traceback cleanup is required and none was introduced.

## 11. Local-First Checkpoint and Artifact Persistence

The corpus is read locally, but persistent artifacts still land in
`/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_full_v1`.

`persist_artifact` implements the required order:

1. serialize to a local staging path (`/content/mednorm_vi_runtime/staging`);
2. compute the local SHA-256;
3. verify the checkpoint reloads (`validate_checkpoint_after_save_reload`, which
   also re-runs the Audit-0038 contract rejection);
4. run the Drive health check;
5. copy to a temporary persistent path;
6. atomically replace the target, falling back to a copy when the filesystem
   refuses the rename and recording `atomic_replace_used`;
7. recompute the persistent SHA-256;
8. require it to equal the local digest.

A transport failure buys **one** bounded remount and one retry. If persistence
still fails the run **stops**: it does not continue to later epochs, it preserves
and reports the verified local staged path and digest, and it states explicitly
that the epoch must not be treated as resumable from Drive.

The same policy is applied to `checkpoints/best.pt`, `checkpoints/latest.pt`,
`logs/training_history.jsonl`, `resolved_config.json`, `validation_metrics.json`,
`training_manifest.json`, `e4_alignment_diagnostic.json` and
`grid_target_statistics.json`. The Hugging Face cache stays at
`DRIVE_ROOT/model_cache/huggingface` and is never part of the artifact. The
Audit-0039 full-resume compatibility contract is unchanged.

## 12. Unchanged Training Semantics and Accounting

Nothing about the learning problem changed. Training data, epoch count, learning
rate, optimizer, model, tokenizer revision, labels, span rules and the metric are
all as in Audit 0039:

```text
micro_batch_size                 1
gradient_accumulation_steps      8
effective_batch_size             8
full epochs                     12
expected optimizer steps    50,748     (ceil(33,826 / 8) * 12, derived)
expected backward passes   405,912
final partial group              2     (loss scale 1/2, not 1/8)
precision                     bf16, fp16 + GradScaler fallback
weight format      pytorch_model.bin, use_safetensors = False
base model         full PhoBERT fine-tuning plus the W2NER head
```

The arithmetic remains derived. The only change is its input: the accumulation
plan is now built from `TRAIN_SOURCE.example_count` instead of
`len(train_contracts)`. Partial-group loss scaling, gradient clipping at the
tracked boundary, optimizer-step counting on real steps only, governed
validation-only selection by exact F1, and complete avoidance of internal_test are
all preserved and re-asserted by tests.

## 13. Notebook Execution Order

1. standard library plus smoke-default operator settings;
2. mount Drive (basic) and, in the first cell that can use the repository, run the
   tracked Drive health check;
3. clone/checkout the repository;
4. install dependencies and import the repository package;
5. resolve the governed corpus and validate source hashes;
6. materialize only train and validation to local `/content`;
7. verify local hashes and switch the active paths to local;
8. acquire the tokenizer and the VnCoreNLP singleton;
9. run the complete alignment preflight from the local files;
10. build only bounded metadata and statistics — no contract lists;
11. resolve CUDA and the precision policy;
12. print the resolved execution summary;
13. acquire the PhoBERT encoder;
14. stream train contracts during each epoch;
15. stream validation contracts during each validation pass;
16. save locally, then sync and verify persistent artifacts;
17. run the full artifact validator.

One ordering note, stated plainly: the *helper-based* Drive health check cannot
precede the repository clone, because the helper lives in the repository. The
bootstrap cell performs the plain `drive.mount`, and the tracked probe runs in the
first cell after import — still before any governed corpus I/O, which is the
property that matters. The probe is re-run against the resolved split files
immediately before the copy.

The encoder is still acquired only after the preflight passes, guarded by
`PREFLIGHT_PASSED`. Committed smoke defaults are retained:

```text
RUN_SMOKE_TRAINING = True
RUN_FULL_TRAINING = False
CONFIRM_FULL = ""
RESUME_FROM_SMOKE_CHECKPOINT = False
RESUME_FROM_FULL_CHECKPOINT = False
```

No full-mode authorization value is committed.

## 14. Tests and Static Checks

New: `tests/unit/test_e4_runtime_io.py` (51 tests) covering a healthy Drive needing
no remount; errno 107 and the transport message each triggering exactly one
remount; a failed post-remount probe raising; no infinite retry; a missing source
file never being hidden by a remount; force-unmount used only when the flush
fails; temporary-file plus atomic replace; copied hashes equalling governed
hashes; hash mismatch rejecting the copy; the 63-character digest being rejected;
reuse of a valid local copy; replacement of a stale one; `internal_test` never
copied or streamed; the real governed splits materializing with their
authoritative digests; one contract at a time; a new iterator per epoch and per
validation pass; deterministic order; short-stream rejection; bounded smoke rows;
identity verification; the structural non-materialization guard; bounded sample
reporting and scalar maxima; memory snapshots; local-first save with the reload
check running before the Drive sync; persistent hash verification; one bounded
remount retry during persistence; persistence failure stopping and preserving the
local copy; notebook cell order and wiring; `pytorch_model.bin` with
`use_safetensors=False` still active; unchanged accumulation semantics; no
internal_test access; and no model, checkpoint, cache, archive, `CLAUDE.md`,
`AGENTS.md` or `.claude/` path tracked in Git.

```text
env PYTHONPATH=src python -m pytest -q
1481 passed, 1 skipped in 141.84s (0:02:21)

env PYTHONPATH=src python -m pytest -q tests/unit/test_e4_runtime_io.py
51 passed

ruff check .                                             All checks passed!
ruff check notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb   All checks passed!
env PYTHONPATH=src python -m mypy      Success: no issues found in 258 source files
env PYTHONPATH=src python -m compileall -q src           <passed>
git diff --check                                         <passed>
```

The single skip is the pre-existing local `pyarrow` skip in
`tests/unit/test_vietmed_adapter.py`. Every modified notebook code cell was parsed
with `compile(...)`; an AST scan found no undefined or unused names. No existing
test was weakened: the pre-existing suite grew from 1,430 to 1,481 passing tests
with no deletions or relaxations.

## 15. Protected Files and Hygiene

```text
docs/MedNorm-VI_Architecture.pdf     0d5eaa20…1e09b        unchanged
docs/audits/0038-…md                                       unchanged
docs/audits/0039-…md                                       unchanged
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

## 16. Changed Files

Added:

- `docs/audits/0040-e4-colab-io-and-memory-robustness.md`
- `src/mednorm_vi/training/phase2/e4_runtime_io.py`
- `tests/unit/test_e4_runtime_io.py`

Modified:

- `configs/training/phase2_e4_phobert_w2ner_colab.yaml`
- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`

## 17. Limitations

1. **No full training has run.** The repository is ready for a fresh CUDA
   attempt on T4 High-RAM, L4 or A100; success is not claimed.
2. The ~30 GB figure is the operator's observation from the failed run, not a
   measured ceiling reproduced here. The fix is structural, not a tuned budget.
3. Local execution cannot exercise Drive FUSE, `google.colab`, or a CUDA GPU.
   The Drive and persistence paths are proven by injected adapters and simulated
   transport failures.
4. `micro_batch_size` remains 1; padded-grid batching is still not implemented.
5. The bounded statistics scan costs one extra streaming pass over the corpus
   before training. That is a deliberate trade: it retains nothing.
6. **T4 VRAM is not proven.** Host-RAM growth is fixed by streaming, but whether
   PhoBERT-large plus the O(n²) W2NER grid fits a 16 GB T4 is unmeasured. If it
   does not, that is a runtime resource limitation of that GPU, not a
   corpus-streaming or system-RAM regression in this milestone.

## 18. Remaining Colab Work

A **fresh CUDA full run on T4 High-RAM, L4, or A100** is required after this
change. It must show: the Drive health check passing, local materialization with
matching digests, the alignment
preflight passing from local files, bounded statistics without a contract list,
the resolved execution summary, encoder acquisition only afterwards, streaming
epochs, 50,748 optimizer steps and 405,912 backward passes, local-first verified
checkpoint persistence, and a passing artifact validator — with internal_test
untouched throughout.

If the chosen runtime runs out of **GPU VRAM**, that is a runtime resource
limitation of that GPU, not a corpus-streaming or system-RAM failure, and it must
be reported as such.

## 19. Safe-to-Commit Verdict

Safe to commit after review. Required tests and static checks pass, no protected
architecture, audit, evaluator or corpus file changed, governed hashes are
unchanged, default feature flags are unchanged, no existing test was weakened, and
no weight, checkpoint, corpus, cache or archive artifact is staged.

```bash
git add \
  docs/audits/0040-e4-colab-io-and-memory-robustness.md \
  docs/audits/README.md \
  configs/training/phase2_e4_phobert_w2ner_colab.yaml \
  notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb \
  src/mednorm_vi/training/phase2/e4_runtime_io.py \
  tests/unit/test_e4_runtime_io.py

git commit -m "fix: harden E4 Colab corpus IO and stream W2NER contracts"
git push origin main
```
