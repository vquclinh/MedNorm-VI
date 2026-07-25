# First Colab Runs — Order and Gating (Audit 0017)

Two Colab actions, in order. Neither is full training. **Do not run full S1 training**
until the gating conditions at the bottom are all satisfied.

## FIRST Colab action — VietMed preprocessing (CPU; data prep only)

```text
notebooks/MedNorm_Data_VietMed_Preprocess.ipynb
```

- **Runtime:** standard Colab **CPU** (GPU unnecessary). Pinned `pyarrow==17.0.0`.
- **Why first:** VietMed-NER is governance-training-allowed but its Parquet adapter is
  not runnable in the offline repo runtime, so it is **excluded from governed corpus v1**
  (`GOVERNANCE_TRAINING_ALLOWED` + `TECHNICALLY_PENDING_PREPROCESSING` +
  `NOT_INCLUDED_IN_GOVERNED_CORPUS_V1`). This notebook converts it to canonical
  half-open JSONL (excluding audio), emitting aggregate manifest + hashes only.
- **Return:** copy `data/derived/training_corpora/vietmed_ner_v1/` back (git-ignored),
  then rebuild locally: `python -m mednorm_vi.data_engine.cli build-governed-corpus`
  and re-run leakage/offset validation.

## SECOND Colab action — S1 mention smoke (BLOCKED: notebooks are DESIGN_DRAFT)

```text
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb   (status: DESIGN_DRAFT)
notebooks/MedNorm_S1_MentionExtraction.ipynb        (status: DESIGN_DRAFT)
```

> **Not runnable yet (Audit 0019).** Both S1 notebooks still contain placeholder cells
> (`<fill: …>`, commented `# !pip install`, commented model loading, describe-only prints),
> so a fresh Colab runtime cannot execute them end-to-end. They must first be repaired the
> same way the VietMed notebook was — real executable checkout via
> `mednorm_vi.reproducibility.repository_checkout`, real pinned install, real training call,
> real artifact export — and then behaviorally verified. See
> `docs/notebooks/notebook_execution_integrity.md`.

Intended design once repaired (unchanged):

- **Runtime:** GPU (T4 sufficient for smoke; ~4–8 GB VRAM; < 2 GB Drive; ~10–20 min).
- **Primary backbone:** ViHealthBERT (~0.135B) — smallest capable specialist. Does not
  change the architecture.
- **Scope:** ≤ 3 batches through the training path. **No** leaderboard/full-model claim;
  **no** automatic transition to full training (`CONFIRM_FULL="YES"` required).
- **Return:** smoke checkpoint + manifest to `models/checkpoints/full_v1/mention/vihealthbert/`;
  run the local validation commands below.

## Do NOT recommend full S1 training until ALL of these hold

1. VietMed inclusion decision is finalized (preprocess returned + corpus rebuilt, or
   explicitly deferred);
2. near-duplicate-safe splits are rebuilt (duplicate-family grouping; 0 family leakage);
3. TEST_NAME and TEST_RESULT supervision exists (synthetic/lexicon — currently absent);
4. medication coverage is improved (currently thin: 905 spans);
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
