# First Colab Runs - Order and Gating (Audits 0017, 0020, 0021)

Two Colab actions, in order. Neither is full training. **Do not run full S1
training** until the gating conditions at the bottom are all satisfied.

## FIRST Colab action — VietMed preprocessing (CPU; data prep only)

```text
notebooks/MedNorm_Data_VietMed_Preprocess.ipynb
```

- **Runtime:** standard Colab **CPU** (GPU unnecessary). Pinned `pyarrow==17.0.0`.
- **Status:** completed for Audit 0020. The returned artifact is placed at
  `data/derived/training_corpora/vietmed_ner_v1/` and remains git-ignored.
- **Why first:** VietMed-NER is governance-training-allowed but its Parquet read step
  needs `pyarrow` in Colab CPU. The notebook converts it to canonical half-open JSONL
  while excluding audio and emitting aggregate manifest + hashes only.
- **Audit 0020 exception:** the returned generated manifest did not contain
  `run_mode`; the human reviewer accepted it as REAL using equivalent provenance.
  Future returned artifacts must contain `run_mode: REAL`.
- **Local result:** `build-governed-corpus` now reports
  `vietmed_status: included_from_artifacts`, 35,916 examples, 15,757 entities,
  cross-split family leakage 0, and eval MAP_APPROXIMATE entities 0.

## SECOND Colab action - S1 mention smoke

```text
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb   (status: IMPLEMENTED_UNEXECUTED)
notebooks/MedNorm_S1_MentionExtraction.ipynb        (status: DESIGN_DRAFT)
```

Audit 0020 clears the **data/corpus gate** for the S1 smoke milestone: VietMed is
included, duplicate-family leakage is 0, eval approximate entities are 0, and the
double rebuild is byte-identical. Audit 0021 implements the first-run smoke notebook but
does not execute it locally. The notebook remains `IMPLEMENTED_UNEXECUTED` until the user
runs it on Colab Pro and returns the artifacts.

Fresh Colab defaults:

```python
DRIVE_ROOT = Path("/content/drive/MyDrive/MedNorm-VI")
REPO_DIR = Path("/content/MedNorm-VI")
REPO_URL = "https://github.com/vquclinh/MedNorm-VI.git"
REPO_REF = "main"
CORPUS_DIR = DRIVE_ROOT / "data/derived/training_corpora/mednorm_vi_training_v1"
OUTPUT_DIR = DRIVE_ROOT / "artifacts/s1_mention_first_run_smoke"
MODEL_CACHE_DIR = DRIVE_ROOT / "model_cache/huggingface"
```

The Git clone is temporary under `/content`. The governed corpus, Hugging Face cache,
smoke checkpoint, logs, and returned manifest persist on Drive. The notebook fails before
model acquisition unless the Drive corpus has:

```text
canonical examples          35,916
train examples              33,826
validation examples          1,045
internal_test examples       1,045
corpus manifest SHA-256     a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3
train split SHA-256         892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a
```

Smoke bounds from `configs/training/s1_mention_first_run_smoke.yaml`:

```text
max_train_examples       8
max_validation_examples  4
max_train_batches        2
max_validation_batches   1
max_optimizer_steps      1
batch_size               2
max_sequence_length      160
```

- **Runtime:** Colab Pro GPU. T4 is sufficient for the smoke path; L4/A100 are also fine.
- **Primary backbone:** ViHealthBERT via registry id `vihealthbert_span_type` and Hugging
  Face source `demdecuong/vihealthbert-base-word`, cached under Drive. The notebook records
  the resolved model revision after acquisition.
- **Scope:** one bounded forward/backward/optimizer step plus one tiny validation inference
  batch. **No** leaderboard/full-model claim; **no** automatic transition to full training.
- **Return:** `OUTPUT_DIR`, containing `checkpoint/s1_mention_smoke_model.pt` and
  `training_manifest.json`. Do not return base-model cache files.

## Do NOT recommend full S1 training until ALL of these hold

1. S1 smoke notebook is behaviorally verified from a successful Colab return;
2. near-duplicate-safe splits remain rebuilt (duplicate-family grouping; 0 family leakage);
3. TEST_NAME and TEST_RESULT supervision exists (synthetic/lexicon — currently absent);
4. medication coverage is further improved beyond approximate VietMed spans;
5. approximate-mapping loss masks are validated;
6. the smoke checkpoint is returned and integration-tested.

## Local validation after any return

```bash
env PYTHONPATH=src python3 -m mednorm_vi.data_engine.cli build-governed-corpus
env PYTHONPATH=src python3 -m mednorm_vi.model_registry.cli --profile full --require-local-paths
env PYTHONPATH=src python3 -m pytest -q tests/unit/test_budget.py tests/unit/test_training_readiness.py
env PYTHONPATH=src python3 -m mednorm_vi.phase1c_foundation.cli doctor
```

The smoke checkpoint is not a trained model. Full-mode inference continues to fail fast
until all required checkpoints exist.
