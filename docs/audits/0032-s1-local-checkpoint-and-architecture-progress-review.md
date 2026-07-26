# Audit 0032 — S1 Local Checkpoint Custody, Held-Out Evaluation, and Architecture Progress Review

- **Date:** 2026-07-27
- **Author:** Claude (AI agent), for human review
- **Change type:** Local checkpoint custody and ignore hygiene, one shared checkpoint-path
  contract, a best-checkpoint-only validator, dual-mode (local/Colab) internal-test evaluation,
  and a whole-repository progress review. **No commit. No push. No training, no retraining, no
  backward pass, no optimizer state. No organizer inference. No `output.zip`. No Git LFS.
  Governed corpus files unmodified. Audit 0031 not overwritten.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — read in full, **unmodified** (`git status`
  clean; SHA-256 `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`; last
  touched by commit `4be29de`, 2026-07-11).
- **Status:** `BEST_CHECKPOINT_VALIDATED_AND_EVALUATED — FULL_ARTIFACT_UNVALIDATED`.

## 1. Exact initial Git state

```text
pwd     /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch  main   (## main...origin/main, in sync)
HEAD    36c520d fix: protect S1 astral characters and finalize preflight
        9e01cbd fix: add full-corpus S1 alignment preflight
        cbda043 fix: support Vietnamese tone-placement alignment
        aa9ef73 fix: unify S1 character alignment and segmentation

git status --short
 M docs/audits/README.md
 M src/mednorm_vi/training/cli.py
 M src/mednorm_vi/training/s1_full_training.py
 M tests/unit/test_notebook_placeholders.py
 M tests/unit/test_notebooks.py
?? docs/audits/0031-...md
?? notebooks/MedNorm_S1_Mention_InternalTest_Evaluation.ipynb
?? tests/unit/test_s1_full_training_validation.py

git diff --check          clean
git ls-files -- '*.pt' '*.pth' '*.ckpt' '*.safetensors'    (empty)
git check-ignore -v best.pt      .gitignore:16:*.pt  best.pt
find ... weight files            ./best.pt
```

## 2. Audit 0031 state

**UNCOMMITTED — untracked.** `docs/audits/0031-...md` appears under `??`, together with two
other new files and five modified tracked files. Audit 0031 has **not** been overwritten,
edited, or merged into this one; both 0031 and 0032 are pending review together. Audit 0030
(`36c520d`) is committed and untouched.

## 3. Checkpoint custody

```text
source path        ./best.pt (repository root, operator-placed)
pre-move SHA-256   a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c   MATCH
size               1,615,513,303 bytes
operation          mv (moved, never copied or duplicated)
final path         checkpoint/s1_mention_full_training_v1/best.pt
post-move SHA-256  a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c   UNCHANGED
root best.pt       confirmed absent
copies on disk     exactly one
```

The digest was verified **before** moving; had it differed, nothing would have been moved or
deleted. mtime is unchanged (`2026-07-27 00:53:41`) after the move, after validation, and after
inference.

## 4. `.gitignore` and weight hygiene

Two repository-root rules added; every pre-existing rule preserved, including the global `*.pt`
and the `!models/.gitkeep` / `!**/README.md` fixture exceptions:

```gitignore
# --- Model weights / checkpoints / adapters ---
/checkpoint/
/best.pt

models/**
```

Proof:

```text
git check-ignore -v checkpoint/.../best.pt   .gitignore:15:/checkpoint/
git check-ignore -v best.pt                  .gitignore:23:*.pt        (rule still armed)
git ls-files -- checkpoint/ best.pt          (empty)
git ls-files | grep -E '\.(pt|pth|ckpt|safetensors)$'   (none)
git status --short            checkpoint/ absent
git status --short --ignored  !! checkpoint/
```

A history scan (`git log --all --diff-filter=A --name-only`) shows **no weight file has ever
been added** to this repository. No blocker. No Git LFS. No weight bytes were archived,
compressed, embedded or serialized into any tracked file.

## 5. Shared checkpoint-path contract

One resolver in `s1_full_training.py`, used by both the CLI and the notebook, so local and Colab
cannot drift:

| Order | Condition | Path |
| --- | --- | --- |
| 1 | `MEDNORM_S1_BEST_CHECKPOINT` non-empty | that path |
| 2 | Colab | `/content/drive/MyDrive/MedNorm-VI/artifacts/s1_mention_full_training_v1/checkpoints/best.pt` |
| 3 | local | `<repo>/checkpoint/s1_mention_full_training_v1/best.pt` |

`CheckpointLocation` reports `path`, `environment` and `exists`; `.require()` raises naming the
exact missing path plus how to fix it for that environment. `pathlib` throughout. Nothing
rewrites the checkpoint, creates optimizer state, trains, or calls backward.

## 6. Validators: full artifact preserved, best-checkpoint added

**Audit 0031's `validate_full_training_artifact()` is unchanged and unweakened.** It still
requires all six artifact files, and a missing `training_manifest.json`, `latest.pt` or
`training_history.jsonl` still fails.

`validate_best_checkpoint_only()` is a **separate, narrower** path for local development. It
checks only what a lone `best.pt` can prove and **always** reports `full_artifact_validated:
false`, unconditionally — even when every other check fails — so a local result can never be
mistaken for the full-artifact result.

### Executed result (real 1.6 GB checkpoint, CPU, read-only)

```text
best_checkpoint_validated  true
full_artifact_validated    false          <- explicit, always
failed_condition_count     0
checkpoint_sha256          a64cc173...1017c
epoch                      4
global_step                2976
pinned_model_revision      f89e80b461e86f9cfc1c84019bd819830c24b6c5
environment                local
scope                      best_checkpoint_only
```

This **independently confirms** the operator-reported epoch and optimizer-step counts from the
checkpoint bytes, plus `mode == FULL_TRAINING`, all required resume keys, the label space, and
`model_state_dict` presence. It does **not** confirm the reported validation metric, the
manifest, `latest.pt`, or the training history — none of which exist locally.

## 7. Held-out internal-test evaluation (executed)

All local prerequisites were verified present before running: `internal_test.jsonl` (1,045
rows), all three governed manifests, cached tokenizer files at the pinned revision, VnCoreNLP
resources, and Python dependencies. Base-model **weights** were absent — but the checkpoint
carries the complete 201-entry state dict, so the architecture was rebuilt from the cached
`config.json` via `AutoModel.from_config()` and then loaded with `strict=True`. **No weights
were downloaded.** Torch 2.13.0+cpu was installed with explicit operator approval.

Governed corpus gate re-verified unchanged (`corpus_manifest_sha256 a3fd365d…fb8e3`,
`train_sha256 892dc22d…2a4a`, counts 35916/33826/1045/1045).

```text
split                        internal_test (held out; never used for selection)
evaluated examples           1045
alignment preflight          1045/1045 aligned, 0 unalignable, counters reconciled
supervised tokens            28,605

span micro F1                0.746182      <- FIRST held-out number
span precision / recall      0.7705 / 0.7233
token micro / macro F1       0.8454 / 0.8234

per type (token P/R/F1)
  DIAGNOSIS    0.9180 / 0.8672 / 0.8919
  MEDICATION   0.9691 / 0.8135 / 0.8845
  SYMPTOM      0.7820 / 0.6234 / 0.6938
  TEST_NAME    no supervision in the governed corpus (undefined, not poor)
  TEST_RESULT  no supervision in the governed corpus (undefined, not poor)
```

For context only: the operator-reported **validation** figure was 0.7194. The held-out number
is not lower, which is unusual but explainable — validation and internal_test are different
1,045-example samples from the same two sources, so sampling variance dominates at this size.
No conclusion is drawn from the comparison.

Forward passes only: `model.eval()`, `torch.no_grad()`, every parameter `requires_grad_(False)`,
no optimizer, no scheduler, no backward. The checkpoint SHA-256 and mtime were re-verified
**after** inference and are unchanged. Report written to the git-ignored
`reports/s1_internal_test_eval/internal_test_evaluation.json`, recording
`full_artifact_validated: false`.

## 8. Dual-mode notebook

`MedNorm_S1_Mention_InternalTest_Evaluation.ipynb` now supports both runtimes without
duplication. `EXECUTION_MODE` is derived from `google.colab` presence.

| | Colab | Local |
| --- | --- | --- |
| Gate | `validate_full_training_artifact()` (Audit 0031, unchanged) | `validate_best_checkpoint_only()` |
| Checkpoint | Drive artifact | ignored local `checkpoint/` |
| Backbone | pretrained weights | config-only + strict state dict |
| Output | `artifacts/s1_mention_internal_test_eval_v1/` | git-ignored `reports/s1_internal_test_eval/` |
| Records | `full_artifact_validated: true` | `full_artifact_validated: false` |

Both modes: forward-only, never write into `checkpoint/` or the training artifact, no organizer
inference, no `output.zip`, no hardcoded digest or revision.

## 9. L1–L9 progress matrix

Evidence: module inventory, configs, tests, registry, and artifacts actually on disk.

| Layer | Status | Implementation found | Executed evidence | Next blocker |
| --- | --- | --- | --- | --- |
| L1 Document Intelligence | `IMPLEMENTED_UNEXECUTED` | `document_intelligence/` 16 files, 2,109 lines; `position/` 641 lines | unit tests only | end-to-end run on organizer input |
| L2 Case Router | `IMPLEMENTED_UNEXECUTED` | `case_router/` 7 files, 743 lines + `configs/case_router` | unit tests only | routing accuracy never measured |
| L3 Mention Factory | `PARTIALLY_IMPLEMENTED` | `mention_factory/` 20 files, 1,710 lines; grammar + lab parsers | **S1 neural span/type trained and evaluated** (§7) | W2NER/MRC/GLiNER members absent |
| L4 Boundary & Type Resolver | `SCAFFOLD_ONLY` | `boundary_type/` 1 file, 36 lines; `resolution/` 649 lines | none | boundary-offset head not built or trained |
| L5 Assertion | `PARTIALLY_IMPLEMENTED` | `specialists/assertion/`; `assertion_hydra/` is **empty (0 files)** | deterministic rules tested | S2 neural cue/scope/relation untrained |
| L5 ICD-10 linker | `PARTIALLY_IMPLEMENTED` | `kb/` 32 files, 2,230 lines; `linking/`; `indices/icd10_vi` present | KB snapshot intake audited | no dense/BM25 index built; S3 untrained |
| L5 RxNorm linker | `PARTIALLY_IMPLEMENTED` | RxNorm full + prescribable snapshots present; `specialists/rxnorm` | KB intake audited (0014) | same as ICD |
| L6 Evidence Graph | `SCAFFOLD_ONLY` | `evidence_graph/` 2 files, 100 lines | none | edges/global features not implemented |
| L7 Confidence Cascade | `SCAFFOLD_ONLY` | `confidence_cascade/` 2 files, 59 lines | none | no LLM critic/adjudicator wired |
| L8 Metric Decoder | `SCAFFOLD_ONLY` | `metric_decoder/` 2 files, 53 lines | none | expected-Jaccard/WER decoding absent |
| L9 Validator & Packaging | `IMPLEMENTED_UNEXECUTED` | `validator/` 9 files, 1,370 lines; `organizer_policy/` | unit tests only | never run to a real `output.zip` |

## 10. S0–S6 progress matrix

| Stage | Status | Implementation | Executed evidence | Checkpoint | Next blocker |
| --- | --- | --- | --- | --- | --- |
| S0 domain adaptation | `SCAFFOLD_ONLY` | `MedNorm_S0_DomainAdaptation.ipynb` is a DESIGN_DRAFT with placeholders | none | none | notebook is not runnable |
| **S1 mention** | **`IMPLEMENTED_AND_EXECUTED`** | full training path, alignment backend, preflight, validators | full-corpus preflight 24,844/24,844; smoke-v5; training reported 4 epochs / 2,976 steps; **best.pt validated and evaluated held-out (§6, §7)** | `best.pt` local, SHA verified | full artifact (manifest/latest/history) still unvalidated |
| S2 assertion | `SCAFFOLD_ONLY` | `MedNorm_S2_Assertion.ipynb` DESIGN_DRAFT | none | none | no assertion corpus or trained head |
| S3 retrieval | `SCAFFOLD_ONLY` | `MedNorm_S3_Retrieval.ipynb` DESIGN_DRAFT | none | none | no embeddings/index built |
| S4 reranking | `SCAFFOLD_ONLY` | `MedNorm_S4_Reranker.ipynb` DESIGN_DRAFT | none | none | depends on S3 |
| S5 Qwen LoRA | `NOT_STARTED` | notebook placeholder only | none | none | 4B/1.7B weights not acquired |
| S6 calibration | `NOT_STARTED` | notebook placeholder only | none | none | needs out-of-fold predictions |

Registry: 6 roles, full-profile base parameters **8,335,616,512** (within 9B ✓), but **5 of 6
roles still carry `checkpoint_hash: missing`** — only the S1 mention role has real weights.

**Organizer readiness is NOT claimed.** One of nine layers has a trained model; L4, L6, L7 and
L8 are scaffolds; no `output.zip` has ever been produced.

## 11. Critical inconsistencies fixed

1. A 1.6 GB checkpoint sitting at the repository root, one `git add -A` away from a catastrophic
   commit — now moved into an explicitly ignored directory.
2. No repository-root ignore rule for the operator's chosen `checkpoint/` convention — added.
3. Checkpoint path resolution was hardcoded to the Drive artifact, so the notebook could not run
   locally — replaced with one shared resolver.
4. The evaluation notebook assumed the complete Drive artifact and would have failed locally, or
   worse, tempted a weakening of the full-artifact validator — a separate narrower validator was
   added instead, and the full one is untouched.
5. Local model reconstruction required downloading base weights — avoided via config-only
   construction plus the strict state-dict load.

Not fixed (deliberately out of scope): the S0/S2–S6 notebooks remain DESIGN_DRAFTs, and the
empty `assertion_hydra/` package is left as-is. No S2–S6 training was started and S1 was not
retrained.

## 12. Tests and static checks

```text
env PYTHONPATH=src python3 -m pytest -q        -> 1120 passed, 1 skipped (pyarrow reader)
ruff check .                                    -> All checks passed!
env PYTHONPATH=src python3 -m mypy              -> Success: no issues found in 224 source files
env PYTHONPATH=src python3 -m compileall -q src -> clean
git diff --check                                -> clean
notebook code cells parsed                      -> 0 new syntax errors
governed corpus hashes                          -> unchanged
architecture PDF                                -> no diff
checkpoint SHA-256 after all operations         -> unchanged
```

`tests/unit/test_s1_full_training_validation.py` grew to 71 tests, adding the resolver contract
(env override wins, Colab path, local path, empty override ignored, `require()` error content)
and the best-checkpoint-only validator (validates; `full_artifact_validated` false even when
other checks fail; epoch/step/revision/SHA mismatches; payload defects including `SMOKE_ONLY`,
wrong label space, missing `model_state_dict`, mutable revision; omitted payload never silently
passes; missing file; and that validation does not modify the checkpoint).

## 13. Changed tracked files

**Added (4):**
```text
docs/audits/0031-s1-full-training-run-validation-and-internal-test-evaluation.md   (Audit 0031, still pending)
docs/audits/0032-s1-local-checkpoint-and-architecture-progress-review.md
notebooks/MedNorm_S1_Mention_InternalTest_Evaluation.ipynb
tests/unit/test_s1_full_training_validation.py
```

**Modified (6):**
```text
.gitignore                                       (/checkpoint/ and /best.pt)
docs/audits/README.md                            (0031 + 0032 index entries)
src/mednorm_vi/training/cli.py                   (validate-full-training, validate-s1-best-checkpoint)
src/mednorm_vi/training/s1_full_training.py      (artifact validator, resolver, best-checkpoint validator)
tests/unit/test_notebook_placeholders.py         (evaluation-notebook + dual-mode assertions)
tests/unit/test_notebooks.py                     (notebook inventory 12 -> 13)
```

Audit 0031's files are included because it is still uncommitted; its content is unchanged.

## 14. Ignored / untracked local assets

```text
checkpoint/s1_mention_full_training_v1/best.pt   1.6 GB   ignored (/checkpoint/)
reports/s1_internal_test_eval/                            ignored (reports/**)
reports/s1_best_checkpoint/                               ignored (reports/**)
reports/local_alignment_preflight/                        ignored (reports/**)
.venv/                                            ~2 GB   ignored (.venv/)
~/.cache/huggingface/  ~/.cache/mednorm-vi/                outside the repository
```

## 15. Remaining blockers

1. **The full training artifact is still unvalidated.** `manifest`, `latest.pt`, history,
   `resolved_config.json` and `validation_metrics.json` exist only on Drive. Audit 0031's
   validator must run there. The reported validation metric 0.7194 remains operator-reported.
2. Held-out span micro F1 **0.746** is a token-index proxy, not the organizer's character-exact
   entity metric with assertions and candidate Jaccard.
3. TEST_NAME and TEST_RESULT have no supervision in the governed corpus.
4. Eight of nine layers and six of seven stages remain scaffold, unexecuted or not started.
5. Five of six registry roles still have `checkpoint_hash: missing`.

## 16. Honest status

`BEST_CHECKPOINT_VALIDATED_AND_EVALUATED — FULL_ARTIFACT_UNVALIDATED`. The local best checkpoint
is validated from its own bytes and has produced a first held-out internal-test result. Full
artifact validation, S2–S6, L4/L6/L7/L8 and organizer packaging all remain outstanding. No
training, retraining, organizer inference or `output.zip` occurred in this milestone.
