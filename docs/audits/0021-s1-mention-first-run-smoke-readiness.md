# Audit 0021 - S1 Mention First-Run Smoke Readiness

- **Date:** 2026-07-25
- **Author:** Codex (AI agent), for human review
- **Change type:** Colab notebook construction, smoke-contract helpers, static/behavioral
  tests, and S1 handoff documentation. **No commit. No push. No local training, model
  weight/tokenizer download, local GPU workload, full S1 training, organizer inference, or
  packaging artifact generation.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 was read in full before notebook
  changes. The architecture PDF was not modified, moved, regenerated, or replaced.

## 1. Objective and scope

Audit 0020 is committed at:

```text
2c6dbb1 feat: integrate VietMed into governed training corpus
```

The governed corpus data gate allows S1 smoke. This milestone prepares one executable
first-run smoke notebook:

```text
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
```

The notebook is ready for a fresh Colab Pro GPU run, but it has not been executed on Colab
yet. Its status is therefore **IMPLEMENTED_UNEXECUTED**, not smoke-verified.

## 2. Preflight and raw notebook discovery

Initial local state:

```text
git status --short -> clean
git diff --check   -> clean
```

The old S1 smoke notebook was a design draft. Raw code-cell inspection found:

```text
PROJECT_ROOT = '/content/drive/MyDrive/mednorm-vi'   # editable placeholder path
# !pip -q install ...                                # commented required install
EXPECTED_COMMIT = '<fill: reviewed repo HEAD>'       # placeholder
MODEL_REVISION = '<pin model revision>'              # placeholder
# AutoTokenizer.from_pretrained(...)                 # commented model acquisition
print('smoke would run', ...)                        # fake execution statement
```

No model or tokenizer was downloaded during discovery.

## 3. Notebook configuration

Fresh Colab defaults now match the required contract:

```python
DRIVE_ROOT = Path("/content/drive/MyDrive/MedNorm-VI")
REPO_DIR = Path("/content/MedNorm-VI")
REPO_URL = "https://github.com/vquclinh/MedNorm-VI.git"
REPO_REF = "main"
CORPUS_DIR = DRIVE_ROOT / "data/derived/training_corpora/mednorm_vi_training_v1"
OUTPUT_DIR = DRIVE_ROOT / "artifacts/s1_mention_first_run_smoke"
MODEL_CACHE_DIR = DRIVE_ROOT / "model_cache/huggingface"
```

The notebook allows environment overrides for local static harnesses, but the fresh Colab
defaults above are the intended user path. The Git clone is temporary under `/content`.
The corpus, model cache, smoke checkpoint, and returned manifest persist on Drive.

Remaining user-editable/runtime variables:

```text
Colab runtime type    GPU
REPO_REF              main by default; change only for a reviewed branch/ref
Drive corpus files    must already exist under CORPUS_DIR
```

No user edit is required for repository URL, Drive root, corpus path, output path, or model
source in the normal fresh Colab run.

## 4. Corpus gate before model acquisition

The notebook clones the real repository, resolves a 40-character commit SHA, verifies
`src/mednorm_vi`, imports tracked code, then validates the Drive corpus **before** installing
model dependencies or calling `from_pretrained`.

Required corpus facts are governed by:

```text
configs/training/s1_mention_first_run_smoke.yaml
```

Expected values:

```text
canonical examples          35,916
train examples              33,826
validation examples          1,045
internal_test examples       1,045
corpus manifest SHA-256     a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3
train split SHA-256         892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a
vietmed_status              included_from_artifacts
cross_split_family_leakage  0
eval MAP_APPROXIMATE        0
```

`src/mednorm_vi/training/s1_mention_smoke.py` verifies manifest hashes, split hash,
manifest counts, and actual JSONL line counts. Missing, partial, stale, or hash-mismatched
Drive corpora fail before model acquisition.

## 5. Model acquisition and budget behavior

Model/tokenizer acquisition occurs only in Colab after the corpus gate passes.

Tracked smoke config:

```text
registry model id       vihealthbert_span_type
approved model source   Hugging Face model hub
HF model id             demdecuong/vihealthbert-base-word
requested revision      main
cache root              /content/drive/MyDrive/MedNorm-VI/model_cache/huggingface
```

The notebook loads the tracked model registry, selects `vihealthbert_span_type`, records
registry parameter counts, validates the full profile remains within 9B, then records the
runtime-resolved model revision after `AutoModel.from_pretrained`.

Registry counts:

```text
vihealthbert_span_type base parameters     135,000,000
vihealthbert_span_type adapter/head budget   2,000,000
full profile base parameters             8,335,616,512
full profile adapter parameters             73,000,000
full profile total parameters             8,408,616,512
full profile within 9B                    true
```

No external inference API is used.

## 6. Smoke limits

The smoke is explicitly bounded by `configs/training/s1_mention_first_run_smoke.yaml`:

```text
max_train_examples       8
max_validation_examples  4
max_train_batches        2
max_validation_batches   1
max_optimizer_steps      1
batch_size               2
max_sequence_length      160
learning_rate            0.00002
```

The notebook prints these limits before execution. It runs one forward pass, finite
token-level span/type loss, backward pass, and optimizer step. It does not run an epoch and
does not contain a full-training branch.

## 7. Loss masks and VietMed policy

The helper module builds per-example masks from the governed
`annotation_coverage.json`. VietMed `MAP_APPROXIMATE` entities are checked in the tiny train
subset and must remain:

```text
target_type     MEDICATION
mapping_status  MAP_APPROXIMATE
loss mask       span=true, entity_type=true,
                assertions=false, icd_candidates=false, rxnorm_candidates=false
```

Validation/internal_test remain MAP_EXACT-only by the corpus gate. No organizer input is
used as labeled data.

## 8. Output artifacts expected from Colab

The notebook writes only under:

```text
/content/drive/MyDrive/MedNorm-VI/artifacts/s1_mention_first_run_smoke/
```

Expected returned artifacts:

```text
checkpoint/s1_mention_smoke_model.pt
training_manifest.json
```

The training manifest records:

```text
runtime/device report
repository URL/ref/resolved SHA
corpus counts and hashes
model registry id, HF model id, requested/resolved model revision
registry parameter counts and actual loaded base-parameter count
trainable parameter count
batch shapes
finite train loss
optimizer-step confirmation
tiny validation smoke metrics
checkpoint path and SHA-256
SMOKE_ONLY status and smoke_only_not_full_training=true
```

Do not return Hugging Face base-model cache files. Do not run organizer inference or
packaging from this notebook.

## 9. Notebook integrity results

Raw S1 notebook scan after implementation:

```text
placeholder tokens                    0
commented required !pip/!git commands 0
fake "smoke would run" statements      0
TODO/FIXME/pass/NotImplemented code    0
saved notebook outputs                 0
local-machine model paths              0
full-training auto-enable              0
```

Targeted tests:

```text
env PYTHONPATH=src python3 -m pytest -q \
  tests/unit/test_s1_mention_smoke.py \
  tests/unit/test_notebook_placeholders.py \
  tests/unit/test_notebooks.py

51 passed
```

## 10. Files created

```text
configs/training/s1_mention_first_run_smoke.yaml
docs/audits/0021-s1-mention-first-run-smoke-readiness.md
src/mednorm_vi/training/s1_mention_smoke.py
tests/unit/test_s1_mention_smoke.py
```

## 11. Files modified

```text
docs/audits/README.md
docs/notebooks/notebook_execution_integrity.md
docs/training/colab_execution_policy.md
docs/training/first_colab_run.md
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
src/mednorm_vi/training/__init__.py
tests/unit/test_notebook_placeholders.py
tests/unit/test_notebooks.py
```

## 12. Validation commands

```text
env PYTHONPATH=src python3 -m pytest -q -> 653 passed, 1 skipped (pyarrow not installed locally)
ruff check . -> All checks passed
env PYTHONPATH=src python3 -m mypy -> Success, 220 source files
env PYTHONPATH=src python3 -m compileall -q src -> clean
git diff --check -> clean
env PYTHONPATH=src python3 -m mednorm_vi.model_registry.cli --profile full ->
  base_parameters=8,335,616,512; adapter_parameters=73,000,000;
  total_parameters=8,408,616,512; within_9b=true
env PYTHONPATH=src python3 -m mednorm_vi.training.cli plan ->
  plan_id=train-plan-037b3bdbe3b9eddb; S1 requires_gpu=true
```

No local model/tokenizer acquisition occurred. `--require-local-paths` was not needed for
this notebook-construction milestone because the smoke is expected to create a new returned
checkpoint on Drive.

## 13. Deferred items and residual risks

- The notebook has not yet run on Colab Pro GPU. Status remains IMPLEMENTED_UNEXECUTED.
- The requested HF model revision is `main`; the actual immutable model revision is recorded
  in `training_manifest.json` after Colab acquisition.
- TEST_NAME and TEST_RESULT still have no governed real supervision.
- This smoke checkpoint is not a trained model and must not be treated as full S1 output.
- Full S1 training remains blocked until the smoke return is validated and the full S1
  notebook receives its own implementation milestone.

## 14. Confirmations

- No models, tokenizers, datasets, or KB indices were downloaded locally.
- No local GPU workload, training, fine-tuning, organizer inference, or packaging artifact
  generation was run.
- No raw clinical text appears in this audit.
- The architecture PDF was read in full and not modified.

## 15. S1 readiness verdict

The notebook is **safe for a fresh Colab Pro GPU smoke run after human review**.
It is **not safe to describe as executed or smoke-verified** until the user returns a
successful `training_manifest.json` and checkpoint SHA-256 from Drive.

Final Git status:

```text
 M docs/audits/README.md
 M docs/notebooks/notebook_execution_integrity.md
 M docs/training/colab_execution_policy.md
 M docs/training/first_colab_run.md
 M notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
 M src/mednorm_vi/training/__init__.py
 M tests/unit/test_notebook_placeholders.py
 M tests/unit/test_notebooks.py
?? configs/training/
?? docs/audits/0021-s1-mention-first-run-smoke-readiness.md
?? src/mednorm_vi/training/s1_mention_smoke.py
?? tests/unit/test_s1_mention_smoke.py
```

Ignored generated artifacts remain ignored:

```text
!! data/derived/training_corpora/
.gitignore:24:data/** data/derived/training_corpora/mednorm_vi_training_v1/splits/train.jsonl
.gitignore:24:data/** data/derived/training_corpora/vietmed_ner_v1/canonical_examples/vietmed_ner_examples.jsonl
```

Safe to commit after human review: **yes**. Do not commit automatically.
