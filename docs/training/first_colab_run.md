# First Colab Runs - Order and Gating (Audits 0017, 0020)

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
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb   (status: DESIGN_DRAFT)
notebooks/MedNorm_S1_MentionExtraction.ipynb        (status: DESIGN_DRAFT)
```

Audit 0020 clears the **data/corpus gate** for the S1 smoke milestone: VietMed is
included, duplicate-family leakage is 0, eval approximate entities are 0, and the
double rebuild is byte-identical. Do not run full training, and do not run S1 smoke
inside the local repository task. The S1 notebooks still need their own execution
integrity work before any broad "complete workflow" claim. See
`docs/notebooks/notebook_execution_integrity.md`.

Intended design once repaired (unchanged):

- **Runtime:** GPU (T4 sufficient for smoke; ~4–8 GB VRAM; < 2 GB Drive; ~10–20 min).
- **Primary backbone:** ViHealthBERT (~0.135B) — smallest capable specialist. Does not
  change the architecture.
- **Scope:** ≤ 3 batches through the training path. **No** leaderboard/full-model claim;
  **no** automatic transition to full training (`CONFIRM_FULL="YES"` required).
- **Return:** smoke checkpoint + manifest to `models/checkpoints/full_v1/mention/vihealthbert/`;
  run the local validation commands below.

## Do NOT recommend full S1 training until ALL of these hold

1. S1 smoke notebook execution integrity is repaired and behaviorally verified;
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
