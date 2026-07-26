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

## Current inventory (12 notebooks)

| Notebook | Status | Evidence / gap |
| --- | --- | --- |
| `MedNorm_Data_VietMed_Preprocess.ipynb` | **SYNTHETIC_SMOKE_VERIFIED + ARTIFACTS_VERIFIED** | Executed cell-by-cell by `tests/unit/test_notebook_execution.py` in `SYNTHETIC_SMOKE` mode against a local `file://` checkout: real clone -> resolved 40-hex SHA -> adapter import -> 7 real artifact files -> manifest hash re-verified -> audio-absence asserted. Audit 0020 verified the returned real artifact at `data/derived/training_corpora/vietmed_ner_v1/` using human-approved REAL-equivalent provenance. The original generated manifest did not contain `run_mode`; future manifests must contain `run_mode: REAL`. |
| `MedNorm_S0_DomainAdaptation.ipynb` | `DESIGN_DRAFT` | Code cells contain `<fill:` placeholders, a commented `# !pip install`, commented model-loading calls, and describe-only prints. Not runnable. |
| `MedNorm_S1_MentionExtraction.ipynb` | `DESIGN_DRAFT` | Same placeholder pattern as S0. |
| `MedNorm_S1_Mention_FirstRun_Smoke.ipynb` | `IMPLEMENTED_UNEXECUTED` | Audit 0021 replaced the design draft with a bounded Colab Pro GPU smoke wrapper: real Git clone, resolved commit SHA, Drive corpus gate, tracked smoke config, registry-selected ViHealthBERT source, Drive model cache, one optimizer step, tiny validation inference, checkpoint save/reload, and `training_manifest.json`. Local tests validate structure and pure helpers without model downloads. **Audit 0022** replaced the invalid fast-tokenizer assumption: ViHealthBERT-Word uses the slow `PhobertTokenizer` (`is_fast=False`), so character spans now come from the tracked `phobert_alignment` backend (word->char mapping + BPE subtoken alignment), with a segmentation contract and an alignment preflight before model download. The first Colab GPU run failed at the old assertion. **Audit 0023** then fixed a NumPy C-ABI failure: the notebook had three separate pip transactions and imported torch in the same cell as an install with no kernel restart, so `torch.optim.AdamW` died at `numpy.dtype size changed ... Expected 96 ... got 88`. It now uses a two-phase bootstrap (one constrained install -> forced kernel restart -> NumPy/AdamW ABI preflight before any acquisition) driven by a tracked dependency contract. **Audit 0024** scopes dependency health to the S1 import closure: global `pip check` output is preserved in full, unrelated IPython/Jedi and Gradio/Hugging Face Hub conflicts are non-blocking diagnostics, and real S1 import/conflict/NumPy/AdamW failures still block before training. **Running it requires TWO passes** (Run all, restart, Run all again). **The user reports a successful end-to-end fresh-Colab GPU run** (`status = SMOKE_ONLY`, `checkpoint_reload_succeeded = true`, checkpoint SHA-256 `7310f69a...92213`). That report is **not** treated as verification: Audit 0025 added a validator that must read the real Drive manifest and recompute the checkpoint hash. Until that validator has actually run, this notebook stays IMPLEMENTED_UNEXECUTED. **Audit 0026** then fixed the one blocker the real manifest exposed: several governed sources (ViMQ ~93%, PhoNER-COVID19 ~99%) ship ALREADY word-segmented, so `_` is a literal character of `original_text`, and `map_segmented_words()` rejected it as an illegal separator - which would also have discarded ~34% of supervised full-training examples. The alignment preflight now covers train **and** validation with reconciling counters, records privacy-safe failure diagnostics (source, split, hashed example id, stage, reason code, exception type - never text), separates unexpected failures from tracked governed exclusions, and masks segmenter boundary merges instead of discarding the whole example. **Audit 0027** then resolved the two failures the real v2 rerun exposed, both implementation defects rather than bad data: RDRSegmenter re-segmenting already-segmented text emits the join character as a standalone token (shredding the very words ViHealthBERT-Word expects), and it pulls spaced punctuation compounds together so no emitted syllable exists verbatim in the source. Word mapping is now character-level and monotonic - separators (whitespace and the join character) may move freely, but a real character may never change, and any divergence fails with a structural, text-free `SEGMENTER_ALTERED_CHARACTERS` diagnostic. A single segmentation policy covers both source kinds: already-segmented text is used verbatim, everything else goes through VnCoreNLP. **The corrected rerun writes to `s1_mention_first_run_smoke_v3`**; v1 and v2 remain immutable evidence, and the validator accepts v3 through its existing runtime path plus an operator-supplied SHA-256, with no source edit. **Audit 0028** closed the last alignment blocker: RDRSegmenter normalizes Vietnamese tone-mark placement (in an oa/oe/uy cluster the tone may sit on either vowel), which the character aligner read as corruption. Proven from the corpus itself - the two sources that ARE segmenter output carry 4,700 traditional clusters and zero alternative ones, while raw text runs ~23% alternative - and it predicted the v3 outcome exactly (1 alternative cluster in the one failing example, 0 in all eleven that aligned). A tracked 15-pair equivalence table is consulted only after a character comparison fails; each side is exactly two code points, so offsets stay exact and every other difference still fails. A dropped-character check was added for characters the segmenter removes from an example's tail. **The corrected rerun writes to `s1_mention_first_run_smoke_v4`**; v1-v3 remain immutable evidence. **Audit 0029** replaced the fixed pair table with a general orthographic-equivalence rule and added the real acceptance gate. **Audit 0030 corrects one Audit 0029 explanation:** in the observed decomposed-source failures, the real RDRSegmenter did not compose the source sequences; the old aligner composed the original side to NFC and compared that against still-decomposed segmenter output. The direction-agnostic canonical-equivalence rule remains valid. Audit 0030 also protects non-BMP characters across the JVM boundary with one BMP private-use sentinel per astral code point and rejects lost, duplicated, reordered, corrupted, or preexisting sentinels. The authoritative local real-component preflight now passes with real VnCoreNLP and the real slow tokenizer: 24,844/24,844 aligned, zero unalignable examples, zero governed exclusions, zero equivalence failures. **The corrected rerun target remains `s1_mention_first_run_smoke_v5`**; v1-v4 remain immutable evidence, and smoke-v5 still has not run. |
| `MedNorm_S1_Smoke_Artifact_Validation.ipynb` (Audit 0027) | `IMPLEMENTED_UNEXECUTED` | Defaults to the v3 artifact; warns when pointed at the historical v1 or v2. |
| `MedNorm_S1_Smoke_Artifact_Validation.ipynb` | `IMPLEMENTED_UNEXECUTED` | Audit 0025. Read-only Colab entry point that validates the **real** S1 smoke artifact on Drive: it reads the actual `training_manifest.json`, recomputes the checkpoint SHA-256 from the bytes on disk, and requires it to match both the manifest and the observed run hash. Every readiness condition is checked against a recorded value (corpus hashes and counts, zero leakage, zero approximate eval mappings, PhobertTokenizer/`is_fast=False`, alignment backend, zero equivalence failures, production VnCoreNLP with non-empty resource hashes, scoped S1 dependency closure with no blocking conflicts, NumPy ABI preflight, finite loss, backward, AdamW step, validation, `full_training_readiness`). A failed **global** `pip check` is preserved as a diagnostic and never invalidates the artifact. On failure it prints every failed condition and refuses full-training readiness. It also prints the immutable revision full training must pin. No install, no restart, no GPU, no `from_pretrained`, no training (enforced by test). Not yet executed. |
| `MedNorm_S1_Mention_Full_Training.ipynb` | `IMPLEMENTED_UNEXECUTED` | Audit 0025. Real guarded Colab GPU full-training run for the S1 mention span+type head. Reuses the validated components rather than duplicating them: two-pass fingerprint bootstrap with one forced restart, NumPy `RandomState` + dummy AdamW ABI preflight, scoped S1 dependency health, governed corpus gate, production VnCoreNLP segmentation (asserted), tokenizer equivalence preflight, and slow-tokenizer character alignment. Training is authorized only by a **validated** smoke artifact and is pinned to the **immutable** model revision read from it; `main` is rejected. An explicit `CONFIRM_FULL_TRAINING` guard (default empty) raises `SystemExit` before model acquisition, the real optimizer, and any backward pass. Writes to a separate Drive directory (`s1_mention_full_training_v1`) with `latest`/`best` checkpoints, JSONL history, resolved config, validation metrics, and a full-training manifest; resume rejects the SMOKE_ONLY checkpoint. Not yet executed - and it cannot run until the revision is pinned. |
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
