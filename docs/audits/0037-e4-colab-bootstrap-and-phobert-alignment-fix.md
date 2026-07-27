# Audit 0037 - E4 Colab Bootstrap and PhoBERT Alignment Fix

Date: 2026-07-27

## 1. Objective and Scope

This append-only bug-fix audit records the repository fix for a real Colab E4
smoke attempt on:

`notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`

The milestone fixes repository implementation defects exposed by Colab. It does
not edit `docs/MedNorm-VI_Architecture.pdf`, Audit 0035, or Audit 0036. It does
not train locally, run internal_test, run organizer inference, create
`output.zip`, or commit/push changes.

Status vocabulary:

- `IMPLEMENTED`: repository source/notebook/test change exists.
- `PREFLIGHT_EXECUTED`: local unit/static checks executed.
- `READY_FOR_COLAB_SMOKE`: operator can retry the E4 smoke path in Colab.
- `SMOKE_EXECUTED`: not reached in this audit.
- `FULLY_TRAINED`: not reached in this audit.
- `UNAVAILABLE_UNTRAINED`: no valid trained E4 artifact is present.

Final status for this bug fix:

- E4 Colab bootstrap fix: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`,
  `READY_FOR_COLAB_SMOKE`.
- E4 PhoBERT W2NER training artifact: `UNAVAILABLE_UNTRAINED`.
- E4 smoke execution: not claimed.

## 2. Initial Git and Audit State

Precondition commands were inspected before new work began:

```text
pwd
/mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI

git branch --show-current
main

git status -sb
## main...origin/main

git status --short
<clean>

git log --oneline -8
03fc55f feat: complete phase2 Colab training readiness
c6107a4 feat: add phase2 mention ensemble and learned l4 v2 contracts
23065f9 feat: implement L3 span lattice and L4 resolver ablation
a00af3c feat: add exact character-offset S1 mention evaluator
15d0b07 feat: validate S1 checkpoint and add local held-out evaluation
36c520d fix: protect S1 astral characters and finalize preflight
9e01cbd fix: add full-corpus S1 alignment preflight
cbda043 fix: support Vietnamese tone-placement alignment

git diff --check
<clean>
```

Audit 0036 was already committed at `03fc55f`, so it was left immutable and this
new append-only Audit 0037 was created.

Architecture PDF read status: `docs/MedNorm-VI_Architecture.pdf` was read in
full before modifying code. Audits 0035 and 0036 were read and preserved
unchanged.

## 3. Observed Real Colab Failures

The operator reported these real failures in order:

1. `ModuleNotFoundError: mednorm_vi`.
   Root cause: the E4 notebook imported project modules before cloning/updating
   the repository and before adding `<repo>/src` to `sys.path`/`PYTHONPATH`.

2. Governed corpus path mismatch.
   Root cause: the notebook expected stale file names under
   `/content/drive/MyDrive/MedNorm-VI/data/processed/`, while the authoritative
   governed split identity is by SHA-256:
   train `892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a`;
   validation `ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103`.

3. Empty model/tokenizer revisions at the revision gate.
   Root cause: the gate ran before the notebook resolved immutable Hugging Face
   revisions or accepted operator-provided 40-hex revisions.

4. `KeyError: 'offset_mapping'` during real smoke training after loading
   `vinai/phobert-large`.
   Root cause: the notebook assumed a fast tokenizer offset mapping. The
   official PhoBERT path may resolve to a slow `PhobertTokenizer`, where
   `return_offsets_mapping` is unavailable.

The reported unused pretrained MLM-head keys when loading only the base encoder
are not themselves a blocker, provided only expected MLM-head keys are ignored.

## 4. Architecture Contracts Applied

The fix applies the architecture PDF offset and L3/L4 contracts:

- `original_text[start:end] == entity_text` remains the governing invariant.
- Offsets are end-exclusive and reconstructed from original text coordinates.
- W2NER labels are word-level relation grids, not tokenizer-offset grids.
- Repeated text and repeated whitespace/newlines are mapped monotonically by
  original coordinates.
- Decomposed Unicode is preserved; no normalized copy is used for offsets.
- Invalid, incomplete, or truncated alignments fail loudly.
- The E4 expert remains an L3 proposal source and does not emit final organizer
  JSON.

## 5. Alignment Solution

Implemented in:

- `src/mednorm_vi/training/phase2/e4_w2ner_training.py`

The corrected E4 path is:

1. Original text with absolute offsets.
2. Existing production VnCoreNLP/PhoBERT alignment helpers:
   `resolve_segmented_text`, `segmented_text_to_words`, and
   `map_segmented_words`.
3. Segmenter word list with original character spans.
4. W2NER grid words stored as exact original slices.
5. Tokenizer surface stored separately as segmenter `model_text`, which may
   contain underscores.
6. Slow PhoBERT path tokenizes each segmented word without special tokens,
   builds deterministic subtoken-to-word indices, adds special/padding tokens
   with `None` word ids, and never requests `offset_mapping`.
7. Fast tokenizer path is allowed only when `tokenizer.is_fast is True`; it
   explicitly requests `return_offsets_mapping=True`, maps offsets through the
   governed segmented word intervals, and removes `offset_mapping` from encoder
   inputs.
8. W2NER decodes word spans back to exact original character offsets.

Truncation policy: E4 W2NER training refuses silent truncation. If the encoded
PhoBERT sequence exceeds `max_length`, the conversion raises before producing a
partially valid gold relation grid.

## 6. Notebook Bootstrap Fix

Updated notebook:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`

Corrected Run-all order:

1. Mount Google Drive.
2. Clone/update the repository.
3. Check out the requested ref and record the 40-hex commit.
4. Install Colab-only runtime dependencies.
5. Change to repo root.
6. Add `<repo>/src` to `sys.path` and `PYTHONPATH`.
7. Verify `mednorm_vi` imports from the clone.
8. Resolve governed train/validation files by authoritative SHA-256, not stale
   aliases.
9. Resolve immutable model/tokenizer revisions in an explicit bootstrap cell.
10. Gate on 40-hex revisions after resolution.
11. Acquire the tokenizer at the resolved revision and report class/`is_fast`.
12. Run VnCoreNLP alignment/grid/tokenizer preflight.
13. Acquire the base model inside the smoke/full training function using the
    resolved revision and `use_safetensors=True`.
14. Validate the encoder load report so only expected MLM-head keys are ignored.
15. Save, reload, hash, manifest, and read-only validate the artifact generated
    by a real smoke/full run.

Drive artifact directories are created only after Drive is mounted. Model caches
remain outside the artifact directory.

## 7. Model Loading and Download Custody

The notebook now:

- records `HF_TOKEN` presence without printing or hardcoding the token;
- resolves revisions via Hugging Face metadata or operator-provided 40-hex
  environment variables;
- avoids mutable `main`/`latest` revision in training manifests;
- requests safetensors for the base encoder to avoid unnecessary duplicate
  `.bin` and `.safetensors` acquisition where the official checkpoint supports
  safetensors;
- treats expected MLM-head keys as ignored load noise;
- fails on unexpected missing encoder keys or unexpected non-MLM keys;
- never expects a W2NER head from the pretrained base.

## 8. Regression Tests Added

Added:

- `tests/unit/test_e4_phobert_alignment_fix.py`

Coverage:

- official/slow PhoBERT-like tokenizer contract without `offset_mapping`;
- deterministic per-word subtoken mapping;
- word split into multiple BPE pieces;
- special and padding token mapping to no word;
- exact original-offset reconstruction;
- VnCoreNLP segmented words containing underscores;
- repeated whitespace/newlines;
- decomposed Unicode;
- truncation failure before partial W2NER grid targets;
- fast tokenizer branch requesting and removing `offset_mapping`;
- `offset_mapping` absent from encoder inputs;
- corpus resolution by authoritative hash rather than stale file names;
- revision gate ordering in the notebook;
- repository import ordering before project-module use;
- expected-only MLM-head load mismatch validation.

## 9. Local Verification

Executed locally without training:

```text
env PYTHONPATH=src python -m pytest tests/unit/test_e4_phobert_alignment_fix.py -q
9 passed in 0.36s

env PYTHONPATH=src python -m pytest -q
1331 passed, 1 skipped in 116.50s (0:01:56)

ruff check .
All checks passed!

env PYTHONPATH=src python -m mypy
Success: no issues found in 256 source files

env PYTHONPATH=src python -m compileall -q src
<passed>

git diff --check
<passed>
```

Every modified E4 notebook code cell was parsed with `compile(...)`.

The single skipped pytest is the existing local `pyarrow` skip in
`tests/unit/test_vietmed_adapter.py`; it is unrelated to E4.

## 10. Unchanged Protected Files

Verified no diff for:

- `docs/MedNorm-VI_Architecture.pdf`
- `docs/audits/0035-multi-expert-mention-ensemble-and-learned-l4-v2.md`
- `docs/audits/0036-phase2-training-readiness-and-artifact-validation.md`

No checkpoint, corpus, report, model weight, cache, artifact, ZIP, or organizer
output was generated or tracked by this local fix.

## 11. Changed Files

Tracked/added files in this bug-fix milestone:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `src/mednorm_vi/training/phase2/e4_w2ner_training.py`
- `tests/unit/test_e4_phobert_alignment_fix.py`
- `docs/audits/0037-e4-colab-bootstrap-and-phobert-alignment-fix.md`

## 12. Remaining Colab Work

The operator still must rerun the E4 notebook in Colab to produce real smoke
artifacts:

- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/checkpoints/best.pt`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/checkpoints/latest.pt`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/logs/training_history.jsonl`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/resolved_config.json`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/validation_metrics.json`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/training_manifest.json`

No `SMOKE_EXECUTED`, `ARTIFACT_VALIDATED`, or `FULLY_TRAINED` claim is made in
this audit because no new Colab artifact exists in this repository session.

## 13. Safe-to-Commit Verdict

Safe to commit after review. Do not include runtime directories or model
weights in `git add`.

Suggested commands:

```bash
git add \
  notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb \
  src/mednorm_vi/training/phase2/e4_w2ner_training.py \
  tests/unit/test_e4_phobert_alignment_fix.py \
  docs/audits/0037-e4-colab-bootstrap-and-phobert-alignment-fix.md

git commit -m "fix: repair E4 Colab bootstrap and PhoBERT alignment"
git push origin main
```
