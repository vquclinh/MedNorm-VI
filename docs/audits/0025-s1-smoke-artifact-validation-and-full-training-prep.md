# Audit 0025 — S1 Smoke-Artifact Validation and Full-Training Preparation

- **Date:** 2026-07-26
- **Author:** Claude (AI agent), for human review
- **Change type:** One consolidated milestone — a real smoke-artifact validator plus an
  unexecuted production S1 full-training notebook and configuration. **No commit. No push.
  No local training, no local GPU, no model/tokenizer/VnCoreNLP downloads, no external API,
  no organizer inference, no `output.zip`. Governed corpus files untouched.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — §15 (S1 objective), §15.2 (leakage and
  reproducibility), §1/§13 (metric consequences), §18.1 (per-type error analysis), §21
  (Colab VRAM/time). The PDF was not modified, moved, renamed, or regenerated.
- **Status:** `IMPLEMENTED_UNEXECUTED`.

## 1. Audit numbering

`git log` shows Audit 0024 committed as `613ffca fix: scope S1 Colab dependency health`, on
top of `8fa4281` (0023) and `187ffe5` (0022). The working tree was clean. The next immutable
number is therefore **0025**. Audits 0017–0024 are unmodified.

## 2. Colab evidence supplied by the user

The S1 first-run smoke notebook completed end-to-end on a fresh Colab GPU runtime:

```text
artifact dir : /content/drive/MyDrive/MedNorm-VI/artifacts/s1_mention_first_run_smoke
files        : checkpoint/s1_mention_smoke_model.pt, training_manifest.json
checkpoint   : 7310f69acfae278f36753c4b356979737f17388c74f6703a362bb22788892213
final output : status = SMOKE_ONLY, checkpoint_reload_succeeded = true
```

**The screenshot is not treated as evidence.** Nothing in this milestone asserts that the
artifact is valid; a validator was built so the real files can prove it.

## 3. Artifact-validation contract

`src/mednorm_vi/training/s1_artifact_validation.py` (pure Python; no Torch, no network) reads
the actual `training_manifest.json`, recomputes the checkpoint SHA-256 from the bytes on
disk, and reports **every** failed condition rather than stopping at the first.

The checkpoint hash must equal **both** the hash recorded inside the manifest **and** the
observed hash above. Because the hash is recomputed rather than read, tampering with the
checkpoint after the manifest was written is detected.

Validated conditions (each against a recorded value, never mere key presence):

| Group | Conditions |
| --- | --- |
| Run identity | `status == SMOKE_ONLY`; `smoke_only_not_full_training`; 40-hex repository commit |
| Governed corpus | corpus-manifest SHA, train-split SHA, total/train/validation/internal-test counts, VietMed status — all equal to the approved corpus; cross-split family leakage `== 0`; validation/internal-test approximate mappings `== 0` |
| Model | registry id `vihealthbert_span_type`; HF id `demdecuong/vihealthbert-base-word`; requested revision recorded; **resolved revision immutable and non-empty** |
| Tokenizer | class is `PhobertTokenizer`; `tokenizer_is_fast is False`; alignment backend recorded |
| Alignment | equivalence checked; ≥ 1 example covered; zero equivalence failures; zero unalignable examples |
| Segmentation | `segmenter_mode == vncorenlp`; VnCoreNLP RDRSegmenter used; `degraded_fallback is False`; non-empty resource hashes |
| Environment | S1 dependency closure verified; **blocking** conflicts empty; NumPy ABI preflight passed; restart completed |
| Training evidence | finite train loss; backward completed; AdamW step completed; tiny validation completed |
| Readiness | `full_training_readiness is True` |
| Artifact | checkpoint exists; recomputed hash matches manifest **and** expected; no base-model cache files inside the artifact |

**Scoped dependency health is preserved.** A global `pip_check_passed == false` does not
invalidate the artifact — Audit 0024 established that `pip check` audits the whole Colab
image. `pip_check_output` and `non_blocking_dependency_conflicts` are carried through as
diagnostics; only `blocking_dependency_conflicts` (raised by the S1 closure itself) fail.

On any failure the entry point prints every failed condition, refuses to claim validation,
and refuses full-training readiness.

## 4. Has the real artifact been validated?

**No — this remains PENDING.** The validator is implemented and unit-tested, but the artifact
lives on the user's Drive and this environment has no access to it. Running
`notebooks/MedNorm_S1_Smoke_Artifact_Validation.ipynb` in Colab is the one action that can
promote the status to `SMOKE_ARTIFACT_VALIDATED_FULL_TRAINING_UNEXECUTED`.

The validation notebook is read-only: no installation, no kernel restart, no GPU, no
`from_pretrained`, no training (enforced by test).

## 5. Model-revision policy — the one open blocker

Full training must pin an **immutable 40-hex commit hash**. `main` resolves today and moves
tomorrow, which would make a leaderboard run unreproducible (spec §15.2).

The immutable revision can only be read from the validated manifest field
`model.resolved_model_revision`. It is **not** in the evidence the user supplied, so it was
**not invented**:

- `configs/training/s1_mention_full_training.yaml` ships `pinned_revision: ""`;
- `load_full_training_config()` **raises** `FullTrainingConfigError` for an empty, mutable, or
  malformed revision, so the config cannot be used until it is pinned;
- `pinned_revision_from_outcome()` raises a message containing `BLOCKER` when a validated
  manifest carries no immutable revision;
- the full-training notebook reads the pin from the validated artifact at runtime and asserts
  `backbone.config._commit_hash == PINNED_MODEL_REVISION`.

Recorded in both manifests: model id, requested revision, pinned resolved revision, tokenizer
revision (same pin), and VnCoreNLP resource hashes.

## 6. Smoke-checkpoint policy

```text
smoke checkpoint             = pipeline integrity artifact (ONE optimizer step)
full-training initialization = approved pretrained ViHealthBERT base revision
```

The spec selects the pretrained ViHealthBERT backbone for the mention role (§17, registry role
`mention_span_type`); nothing in it or in any tracked policy designates a one-step smoke
artifact as an initialization. Enforcement:

- the config requires `initialize_from: pretrained_base` and rejects anything else;
- `validate_resume_checkpoint()` rejects any payload whose `mode` is `SMOKE_ONLY`;
- the full-training notebook never names `s1_mention_smoke_model.pt` (enforced by test);
- the full-training manifest records `initialized_from_smoke_checkpoint: false`.

## 7. Full S1 training design

`notebooks/MedNorm_S1_Mention_Full_Training.ipynb` reuses the validated components instead of
re-implementing them: the two-pass bootstrap with the fingerprint marker and forced restart,
the NumPy `RandomState` + dummy `AdamW` ABI preflight, the scoped S1 dependency-health gate,
`verify_governed_corpus()`, VnCoreNLP production segmentation (asserted, never degraded),
`verify_tokenizer_equivalence()`, and `encode_mention_example_slow()`.

Order: bootstrap → restart → ABI preflight → GPU/Drive → **smoke-artifact gate + revision
pin** → **full-training guard** → corpus gate → segmentation → tokenizer → encoding →
DataLoaders → model/optimizer/scheduler → resume → training loop → manifest.

**Full-training guard.** `CONFIRM_FULL_TRAINING` must equal the tracked
`runtime.confirmation_phrase` (`RUN FULL S1 TRAINING`); it defaults to empty, so an accidental
`Run all` raises `SystemExit` before model acquisition, the real optimizer, and any backward
pass (ordering enforced by test).

`src/mednorm_vi/training/s1_full_training.py` holds the pure logic: config loading and
validation, schedule arithmetic, the metrics accumulator, checkpoint/resume contracts, output
paths, and the manifest builder.

## 8. Hyperparameters and rationale

Derived from the spec, the **measured** governed corpus, the 135M backbone, and Colab limits —
not scaled from the smoke limits.

| Setting | Value | Derivation |
| --- | --- | --- |
| `max_sequence_length` | 256 | p99 train example is 329 characters; 256 is also the hard ceiling for ViHealthBERT-Word (PhoBERT: 258 position embeddings) |
| `per_device_batch_size` | 16 | 135M encoder at 256 tokens with AMP fits a T4/L4 comfortably |
| `gradient_accumulation_steps` | 2 | effective batch 32, the standard encoder fine-tuning batch |
| `num_epochs` | 4 | 23,799 supervised examples / 32 = 744 steps per epoch → 2,976 steps; inside the usual 3–5 epoch band |
| `learning_rate` | 3e-5 | middle of the 2e-5…5e-5 BERT-base band |
| `head_learning_rate` | 1e-4 | the classifier head is randomly initialized |
| `weight_decay` | 0.01, excluding bias/LayerNorm | standard AdamW practice |
| `warmup_ratio` | 0.06 | RoBERTa convention; PhoBERT/ViHealthBERT-Word are RoBERTa-based |
| `max_grad_norm` | 1.0 | bounded gradients |
| `mixed_precision` | auto | bf16 when the GPU supports it, else fp16 with `GradScaler` |
| `loss` | focal (γ=2.0, α=0.25) | spec §15 sanctions "Span BCE/focal"; only 8.26% of supervised training characters fall inside an entity, and spec §1 puts entity recall at the head of the optimization chain. Plain BCE — what the smoke exercised — stays selectable |
| `decision_threshold` | 0.5 | training-time selection only; calibrated metric-aware thresholds are L8/S6 work (spec §13) |
| `seed` | 20260723 | repository convention |

**Measured corpus facts driving the design** (computed locally from the governed splits):

| Split | Examples | Supervised | With entities | Entities |
| --- | --- | --- | --- | --- |
| train | 33,826 | **23,799** | 6,313 | 11,720 |
| validation | 1,045 | 1,045 | 1,045 | 1,991 |
| internal_test | 1,045 | 1,045 | 1,045 | 2,046 |

`phoner_covid19` (10,027 train examples) declares `boundary: false` / `entity_type: false` in
the governed annotation-coverage manifest, so its `label_mask` is entirely zero and it
produces **no gradient at all**. `data.filter_unsupervised_examples: true` scopes the sampler
to the 23,799 supervised examples, removing ~30% of otherwise wasted GPU time. **The governed
corpus files are not modified** — only which rows the loader draws.

Only DIAGNOSIS, SYMPTOM, and MEDICATION occur in this corpus; the head keeps all five spec
entity types, and the macro average is computed over the types actually observed so absent
classes do not silently deflate it.

**Validation metrics.** Per-type token precision/recall/F1 (spec §18.1 error analysis) plus
micro/macro aggregates, and span-level micro P/R/F1 over contiguous typed token runs.
Padding and unsupervised positions are excluded through `label_mask`. A wrong type produces
both a false positive and a false negative, matching spec §1's double penalty.

## 9. Output, checkpoint, and resume contract

Output directory `.../artifacts/s1_mention_full_training_v1` — **distinct from the smoke
artifact**, which the config rejects if they are equal.

```text
checkpoints/latest.pt        resumable state, written every epoch
checkpoints/best.pt          best validation checkpoint
logs/training_history.jsonl  per-log-interval train records + per-epoch validation records
resolved_config.json         the exact resolved hyperparameters
validation_metrics.json      final validation metrics
training_manifest.json       full-training manifest
```

No Hugging Face base-model cache is written here (`HF_HOME` stays under `model_cache/`).

Best-checkpoint criterion: `validation_span_micro_f1`, strictly increasing.

Resume contract — `validate_resume_checkpoint()` rejects a checkpoint that: has
`mode == SMOKE_ONLY`; has an unknown mode; is missing any of `mode`, `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `global_step`, `best_metric`,
`entity_type_order`, `seed`, `pinned_model_revision`, `config_sha256`; was trained on a
different pinned revision; or carries a different label space.

The manifest identifies repository SHA, architecture spec version 1.1, governed corpus
hashes, model/tokenizer ids and pins, dependency contract and scoped health, segmentation and
alignment contracts, hyperparameters, effective batch size, schedule, completed epochs and
optimizer steps, validation metrics, best-checkpoint criterion, checkpoint paths and SHA-256
values, `run_completed`, `interrupted_reason`, and `safe_to_resume`.

OOM handling: `torch.cuda.OutOfMemoryError` is caught, the resumable checkpoint is noted, and
ordered recovery guidance is printed (halve batch and double accumulation → shorten sequences
→ larger GPU), then re-raised. `KeyboardInterrupt` records an honest interrupted manifest.

## 10. Tests and validation

New: `tests/unit/test_s1_artifact_validation.py` (50) and `tests/unit/test_s1_full_training.py`
(52). Extended: `tests/unit/test_notebook_placeholders.py` (+11) and
`tests/unit/test_notebooks.py` (inventory 10 → 12).

Covered: valid artifact passes; every readiness field false/missing fails (parametrized over
20 fields); governed-corpus and leakage mismatches fail; all failures reported, not just the
first; checkpoint hash must match the manifest **and** the expected hash; hash recomputed from
bytes (tamper detection); missing checkpoint; **global `pip_check_passed == false` with a
healthy scoped closure still validates**, while a blocking closure conflict fails; only 40-hex
revisions are immutable; a mutable revision fails validation and is never invented; base-model
cache inside the artifact fails; unreadable/malformed manifests raise.

Full training: the tracked config refuses to load unpinned; mutable/malformed revisions
rejected; the pin reaches config, tokenizer and backbone; smoke checkpoint rejected as both
initialization and resume source; output dir must differ from the smoke artifact; latest/best
paths distinct and never under the smoke directory; resume contract parametrized over every
required key; revision and label-space drift rejected; schedule arithmetic (744 × 4 = 2,976,
warmup 178); unsupervised filter; hyperparameter bounds and rejection of implausible values;
config hash determinism; the full-training corpus gate equals the smoke gate; metrics
(perfect, wrong-type double penalty, boundary miss, masking, macro over observed types, batch
accumulation); best-metric criterion; manifest completeness and resumability.

Notebook integrity (static, no execution): the validation notebook reads the real manifest and
checkpoint, treats global `pip check` as diagnostic, and never installs/restarts/trains; the
full-training notebook has an explicit guard that precedes model acquisition, the real
optimizer, and any backward pass, requires a validated smoke artifact first, uses the pinned
revision for tokenizer and backbone, never names the smoke checkpoint, writes to a separate
output directory, preserves every validated gate (bootstrap decision, single restart,
`RandomState(42)`, dummy AdamW, ABI verdict, scoped health, corpus gate, equivalence, slow
tokenizer, VnCoreNLP), has a real training loop, and runs no inference or packaging.

```text
env PYTHONPATH=src python3 -m pytest -q          -> 895 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 224 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
notebook code-cell AST parse                      -> 3 S1 notebooks, 0 syntax errors
```

No local training, no local GPU, no downloads, no NumPy/Torch modification.

## 11. Exact changed files

**Added (8):**
```text
configs/training/s1_mention_full_training.yaml                 (full-training config; revision pin required)
docs/audits/0025-s1-smoke-artifact-validation-and-full-training-prep.md
notebooks/MedNorm_S1_Mention_Full_Training.ipynb               (guarded GPU full-training run)
notebooks/MedNorm_S1_Smoke_Artifact_Validation.ipynb           (read-only real-artifact validator)
src/mednorm_vi/training/s1_artifact_validation.py              (smoke manifest + checkpoint SHA validation)
src/mednorm_vi/training/s1_full_training.py                    (config, schedule, metrics, checkpoint/resume, manifest)
tests/unit/test_s1_artifact_validation.py                      (50 tests)
tests/unit/test_s1_full_training.py                            (52 tests)
```

**Modified (4):**
```text
docs/audits/README.md                               (0025 index entry)
docs/notebooks/notebook_execution_integrity.md      (validation + full-training notebook status)
tests/unit/test_notebook_placeholders.py            (+11 validation/full-training assertions)
tests/unit/test_notebooks.py                        (inventory 10 -> 12, later notebooks documented by their own audit)
```

Counted precisely: **8 added, 4 modified — 12 changed tracked files.**

## 12. Remaining risks and blockers

1. **BLOCKER — no pinned revision yet.** The immutable commit hash is unknown until the
   validation notebook runs on the real Drive artifact. Full training cannot start; the config
   raises rather than falling back to `main`.
2. **Artifact validation is PENDING.** No claim is made that the smoke artifact is valid.
3. **Full training has not run.** No accuracy, loss curve, or checkpoint may be claimed.
4. Span-level micro F1 over token-index runs is a *proxy* for the organizer's character-exact
   entity scoring; exact scoring belongs to L4 and the mandatory local evaluator (§18.1).
5. Only 3 of 5 entity types occur in the governed corpus; TEST_NAME/TEST_RESULT get no
   supervision from this data.
6. The boundary-offset head and the W2NER/MRC ensemble members of spec §15 are **not** trained
   here; this milestone trains the ViHealthBERT span+type head only. Deliberately deferred, not
   silently omitted.
7. Encoding is not cached: a resumed session re-encodes (a few minutes) rather than storing a
   large derived artifact on Drive.
8. Colab session limits may interrupt a ~40-minute run; `latest.pt` makes that recoverable.

## 13. Honest status

`IMPLEMENTED_UNEXECUTED`. The validator and the full-training notebook/config are implemented,
unit-tested, and statically verified, but **neither has been executed**. The status may only
become `SMOKE_ARTIFACT_VALIDATED_FULL_TRAINING_UNEXECUTED` after the real Drive artifact has
actually been read and passed every condition in `MedNorm_S1_Smoke_Artifact_Validation.ipynb`.
Full S1 training has **not** run.
