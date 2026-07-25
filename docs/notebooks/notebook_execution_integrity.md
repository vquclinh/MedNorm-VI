# Notebook Execution Integrity

Honest, evidence-based status of every tracked notebook. **File names, section
headings, and keyword presence are not proof.** A notebook is only promoted when
behavioral execution evidence exists.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| `DESIGN_DRAFT` | Headings + configuration, but placeholder or describe-only operations. Not runnable end-to-end. |
| `IMPLEMENTED_UNEXECUTED` | Executable orchestration exists, but it has never been behaviorally run/verified. |
| `SYNTHETIC_SMOKE_VERIFIED` | The repository's test suite **executes** the notebook on synthetic input and verifies real artifacts/hashes. |
| `REAL_DATA_EXECUTED` | The user ran it on real data in Colab. |
| `ARTIFACTS_VERIFIED` | Real-data artifacts were returned and verified in the repository. |
| `RETAINED_INFERENCE_NOTEBOOK` | Kept from an earlier milestone; not part of the current training path. |
| `PLACEHOLDER_FOUND` | Contains placeholders in required executable cells. |

## Current inventory (10 notebooks)

| Notebook | Status | Evidence / gap |
| --- | --- | --- |
| `MedNorm_Data_VietMed_Preprocess.ipynb` | **SYNTHETIC_SMOKE_VERIFIED + ARTIFACTS_VERIFIED** | Executed cell-by-cell by `tests/unit/test_notebook_execution.py` in `SYNTHETIC_SMOKE` mode against a local `file://` checkout: real clone -> resolved 40-hex SHA -> adapter import -> 7 real artifact files -> manifest hash re-verified -> audio-absence asserted. Audit 0020 verified the returned real artifact at `data/derived/training_corpora/vietmed_ner_v1/` using human-approved REAL-equivalent provenance. The original generated manifest did not contain `run_mode`; future manifests must contain `run_mode: REAL`. |
| `MedNorm_S0_DomainAdaptation.ipynb` | `DESIGN_DRAFT` | Code cells contain `<fill:` placeholders, a commented `# !pip install`, commented model-loading calls, and describe-only prints. Not runnable. |
| `MedNorm_S1_MentionExtraction.ipynb` | `DESIGN_DRAFT` | Same placeholder pattern as S0. |
| `MedNorm_S1_Mention_FirstRun_Smoke.ipynb` | `IMPLEMENTED_UNEXECUTED` | Audit 0021 replaced the design draft with a bounded Colab Pro GPU smoke wrapper: real Git clone, resolved commit SHA, Drive corpus gate, tracked smoke config, registry-selected ViHealthBERT source, Drive model cache, one optimizer step, tiny validation inference, checkpoint save/reload, and `training_manifest.json`. Local tests validate structure and pure helpers without model downloads. **Audit 0022** replaced the invalid fast-tokenizer assumption: ViHealthBERT-Word uses the slow `PhobertTokenizer` (`is_fast=False`), so character spans now come from the tracked `phobert_alignment` backend (word->char mapping + BPE subtoken alignment), with a segmentation contract and an alignment preflight before model download. The first Colab GPU run failed at the old assertion. **Audit 0023** then fixed a NumPy C-ABI failure: the notebook had three separate pip transactions and imported torch in the same cell as an install with no kernel restart, so `torch.optim.AdamW` died at `numpy.dtype size changed ... Expected 96 ... got 88`. It now uses a two-phase bootstrap (one constrained install -> forced kernel restart -> NumPy/AdamW ABI preflight before any acquisition) driven by a tracked dependency contract. **Running it requires TWO passes** (Run all, restart, Run all again). It still has not been run on Colab, so it remains IMPLEMENTED_UNEXECUTED. |
| `MedNorm_S2_Assertion.ipynb` | `DESIGN_DRAFT` | Same placeholder pattern as S0. |
| `MedNorm_S3_Retrieval.ipynb` | `DESIGN_DRAFT` | Same placeholder pattern as S0. |
| `MedNorm_S4_Reranker.ipynb` | `DESIGN_DRAFT` | Same placeholder pattern as S0. |
| `MedNorm_S5_QwenLoRA.ipynb` | `DESIGN_DRAFT` | Same placeholder pattern as S0. |
| `MedNorm_S6_Calibration.ipynb` | `DESIGN_DRAFT` | Same placeholder pattern as S0. |
| `MedNorm_Full_Offline_Inference_and_Packaging.ipynb` | `RETAINED_INFERENCE_NOTEBOOK` | Retained from an earlier milestone; no placeholders detected, but not behaviorally verified. |

**No S0–S6 notebook is a "complete workflow."** The S1 first-run smoke notebook is
implemented but awaits a real Colab execution result; the other S0–S6 notebooks still
need focused repair milestones. Full training remains blocked until smoke artifacts are
returned and validated.

## What made the earlier claim wrong

Audit 0018 described the VietMed notebook as ready while its committed cells still had:

```text
# !git clone <repo-url> {REPO_DIR}      # commented -> no checkout ever happened
EXPECTED_COMMIT = '<fill: reviewed repo HEAD>'
# !pip -q install {PYARROW_PIN}         # commented -> pyarrow never installed
```

A fresh Colab runtime therefore had no `/content/MedNorm-VI/src/mednorm_vi` and failed
with `ModuleNotFoundError: No module named 'mednorm_vi'`. The tests of that milestone were
**static**: they asserted section headings and keyword presence, which a commented command
still satisfies. Audit 0019 replaces that with behavioral execution plus placeholder rules
that specifically reject commented-out required commands.

## Guards now in place

- `tests/unit/test_notebook_execution.py` — executes the VietMed notebook's code cells in a
  fresh namespace, verifies artifacts, hashes, resolved SHA, and audio exclusion; asserts
  REAL mode fails fast without Parquet and never falls back to synthetic rows.
- `tests/unit/test_notebook_placeholders.py` — rejects `<...>`/`TODO`/`REPLACE_ME`
  placeholders and **commented-out `!pip`/`!git` commands** in executable notebooks; asserts
  the VietMed checkout helper and pinned install are actually invoked; asserts the S1 smoke
  notebook uses the required Colab/Drive paths, verifies the corpus before model acquisition,
  remains strictly bounded, and writes manifest/checkpoint evidence fields.
- `tests/unit/test_repository_checkout.py` — the checkout helper against local `file://`
  repositories (placeholder rejection, resolved SHA, layout verification, failure modes).
- `tests/unit/test_vietmed_adapter.py` - requires future generated manifests to record
  `run_mode`, rejects missing/invalid values by default, rejects `SYNTHETIC_SMOKE`
  artifacts for governed-corpus inclusion, and allows the Audit 0020 missing-field artifact
  only through exact human-approved REAL-equivalent provenance.
- `tests/unit/test_s1_mention_smoke.py` - validates the S1 smoke config, corpus gate,
  deterministic subset selection, VietMed approximate-label loss masks, token-label encoding,
  and batch collation using synthetic fixtures plus the ignored governed corpus when present.
