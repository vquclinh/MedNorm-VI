# Audit 0019 — Colab Notebook Execution Integrity and VietMed End-to-End Preprocessing Validation

- **Date:** 2026-07-23
- **Author:** Claude (AI agent), for human review
- **Change type:** Correctness milestone for notebook **execution integrity**. **No commit.
  No push. No training, no model/tokenizer downloads, no GPU, no external API. The real
  restricted VietMed Parquet was NOT processed locally.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (Data Engine §14, offset contract,
  offline/fail-fast, reproducibility). No architecture change.

**Operating principle:** file names, headings, audit prose, and keyword presence are not
proof. Only executable code, behavioral tests, execution from a clean environment, real
artifact files, verified hashes, and fail-fast behavior count.

## 1. Audit 0018 commit

Audit 0018 is committed and pushed as **`6817feb9a48c69573176ebd9093cd4f00d9b48d2`**
(`fix: implement VietMed preprocessing workflow`), on top of Audit 0017 (`0704aa6`). It is
**immutable and unmodified** by this milestone; its incorrect notebook-readiness claim is
retained as historical evidence of why Audit 0019 exists.

## 2. Exact defective cells in the committed notebook

Verified by reading the raw JSON cell sources of the committed
`notebooks/MedNorm_Data_VietMed_Preprocess.ipynb` (not rendered headings):

```text
code cell  5 : # !git clone <repo-url> {REPO_DIR}   # clone into Colab temp storage
code cell  6 : EXPECTED_COMMIT = '<fill: reviewed repo HEAD>'
code cell  4 : # !pip -q install {PYARROW_PIN}
```

All three are **commented or placeholder** — nothing executes. A fresh Colab runtime
therefore never creates `/content/MedNorm-VI/src/mednorm_vi`, and the adapter import fails
with `ModuleNotFoundError: No module named 'mednorm_vi'` (exactly as the user reported).
pyarrow is likewise never installed.

## 3. Root cause: why the previous tests missed it

Audit 0018's notebook tests were **static keyword/section checks**
(`tests/unit/test_notebooks.py`, `test_vietmed_adapter.py`): they asserted that strings such
as section titles, `vm.write_artifacts(`, and `pyarrow==` were *present in the file*. A
commented-out command still contains those strings, so the assertions passed on a notebook
that could not run. There was no execution, no artifact, and no hash produced by any test.
The adapter itself was genuinely implemented and tested — only the **notebook execution
path** was fake.

## 4. Placeholder findings across all tracked notebooks

Raw-JSON scan of code cells (before this milestone's fix):

| Notebook | angle `<…>` | `<fill:` | commented `!pip`/`!git` | describe-only prints |
| --- | ---: | ---: | ---: | ---: |
| `MedNorm_Data_VietMed_Preprocess.ipynb` | 2 | 1 | 2 | 0 |
| `MedNorm_S0_DomainAdaptation.ipynb` | 5 | 4 | 1 | 2 |
| `MedNorm_S1_MentionExtraction.ipynb` | 5 | 4 | 1 | 2 |
| `MedNorm_S1_Mention_FirstRun_Smoke.ipynb` | 4 | 3 | 1 | 5 |
| `MedNorm_S2_Assertion.ipynb` | 5 | 4 | 1 | 2 |
| `MedNorm_S3_Retrieval.ipynb` | 5 | 4 | 1 | 2 |
| `MedNorm_S4_Reranker.ipynb` | 5 | 4 | 1 | 2 |
| `MedNorm_S5_QwenLoRA.ipynb` | 5 | 4 | 1 | 2 |
| `MedNorm_S6_Calibration.ipynb` | 4 | 4 | 1 | 2 |
| `MedNorm_Full_Offline_Inference_and_Packaging.ipynb` | 0 | 0 | 0 | 0 |

After the fix, the VietMed notebook has **0** placeholders, **0** commented required
commands, and **0** describe-only prints. (Three `<…>` matches remain only inside pyarrow
**type strings** such as `list<element: string>` — data, not placeholders.)

## 5. Notebook status inventory (10 notebooks)

Full detail in `docs/notebooks/notebook_execution_integrity.md`.

| Notebook | Status |
| --- | --- |
| `MedNorm_Data_VietMed_Preprocess.ipynb` | **SYNTHETIC_SMOKE_VERIFIED** |
| `MedNorm_S0_DomainAdaptation.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_S1_MentionExtraction.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_S1_Mention_FirstRun_Smoke.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_S2_Assertion.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_S3_Retrieval.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_S4_Reranker.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_S5_QwenLoRA.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_S6_Calibration.ipynb` | `DESIGN_DRAFT` |
| `MedNorm_Full_Offline_Inference_and_Packaging.ipynb` | `RETAINED_INFERENCE_NOTEBOOK` |

Audit 0017's description of S0–S6 as "complete workflows" is **superseded**: they are design
drafts. `docs/training/first_colab_run.md` now marks the S1 notebooks as not runnable, and
`docs/training/colab_execution_policy.md` states that readiness requires execution evidence.

## 6. Repository-checkout helper

New tracked module **`src/mednorm_vi/reproducibility/repository_checkout.py`**:

```python
checkout_repository(repo_url, repo_dir, repo_ref="main", *, clean_existing=True) -> CheckoutResult
```

- rejects empty and `<…>`/`REPLACE_ME`/`YOUR_` placeholder URL and ref values;
- removes an existing temp checkout only when `clean_existing=True` (else raises);
- real `git clone` → `git fetch <ref>` (falls back to a full fetch for bare SHAs) →
  `git checkout --detach`;
- returns the **resolved 40-hex commit SHA**;
- verifies `src/mednorm_vi` and `src/mednorm_vi/data_engine/vietmed_ner.py` exist;
- never continues silently after a git failure; uses `subprocess` **argument arrays** (no
  shell interpolation); records command diagnostics without credentials.

## 7–8. Repository URL/ref policy and resolved provenance

The notebook uses the real repository:

```text
REPO_URL = "https://github.com/vquclinh/MedNorm-VI.git"   (env-overridable for tests)
REPO_REF = "main"                                          (user may pin a commit SHA)
REPO_DIR = /content/MedNorm-VI                             (Colab temp)
DRIVE_ROOT = /content/drive/MyDrive/MedNorm-VI             (persistent)
```

When `REPO_REF="main"`, the notebook records the **resolved** HEAD SHA — never the word
`main` — into every artifact manifest (`repo_commit`). No `<fill: …>` remains.

## 9. Synthetic execution mode

`RUN_MODE` ∈ {`REAL`, `SYNTHETIC_SMOKE`}, default **`REAL`**. `SYNTHETIC_SMOKE` uses two tiny
synthetic decoded rows, touches no restricted data, needs no pyarrow, writes only into a
temporary directory, and prints an explicit banner that it is **not** real VietMed
preprocessing. `REAL` mode asserts the Parquet directory and files exist and **fails fast**;
there is no fallback path from REAL to synthetic (test-enforced: the REAL branch contains no
synthetic rows).

## 10. Behavioral execution test design

`nbformat`/`nbclient` are **not** repository dependencies and are unavailable locally, so
rather than add a heavy test dependency this milestone uses an equivalent deterministic
harness: `tests/unit/test_notebook_execution.py` loads the **tracked notebook**, extracts its
real code cells, and `exec`s them **in order in a fresh namespace**, with configuration
injected purely through environment variables (temporary `DRIVE_ROOT`, `file://` local git
URL, `HEAD` ref, temporary `REPO_DIR`, `RUN_MODE=SYNTHETIC_SMOKE`). No network, no models, no
real data. It verifies adapter import, artifact creation, manifest hashes, exact filenames,
audio absence, recorded resolved SHA, and REAL-mode fail-fast.

## 11–12. Actual synthetic artifacts and hashes produced by the test run

Real files written by executing the notebook (SYNTHETIC_SMOKE, resolved commit
`6817feb9a48c69573176ebd9093cd4f00d9b48d2`):

```text
 bytes  sha256(first 32)                  path
  1026  226e9d42b4516fc25887caa1fd480bbc  canonical_examples/vietmed_ner_examples.jsonl
   999  74fe2ef0596aa2e813be7a37260a8c69  manifests/vietmed_ner_preprocessing_manifest.json
    51  723acac51d3ba407d2077dee56c654a8  manifests/vietmed_ner_source_hashes.json
   147  7e2cd3f9bd357263a205ce826e5b6d5e  quality/vietmed_ner_label_inventory.json
    34  b9fd35c086baa2cdcef71aaf90c791bc  quality/vietmed_ner_offset_validation.json
   137  02ed614f095afd7bc569f287237d1135  quality/vietmed_ner_repair_exclusion_summary.json
    69  df60790291e2d943eade142bfb75e127  quality/vietmed_ner_split_summary.json

examples 2 · entities 1 · offset_invalid 0 · human_review_required 0
resolved_columns {'word_col': 'words', 'tag_col': 'labels'}
examples_jsonl_sha256 226e9d42b4516fc25887caa1fd480bbcb535e5d563ec4fdddb150c4eb699b5fe
repo_commit           6817feb9a48c69573176ebd9093cd4f00d9b48d2
```

These are **synthetic test artifacts**, not VietMed data, and must never be returned as real
preprocessing output.

## 13. Real-data execution still pending

Status remains **not** `REAL_DATA_EXECUTED` and **not** `ARTIFACTS_VERIFIED`. No real VietMed
artifacts exist. VietMed stays `NOT_INCLUDED_IN_GOVERNED_CORPUS_V1`; the governed corpus
rebuild in this milestone still reports `vietmed_status: not_included_pending`
(26,649 examples, cross-split family leakage 0 — unchanged).

## 14. Exact files changed

Added (6):
```text
src/mednorm_vi/reproducibility/__init__.py
src/mednorm_vi/reproducibility/repository_checkout.py
docs/notebooks/notebook_execution_integrity.md
docs/audits/0019-colab-notebook-execution-integrity.md
tests/unit/test_notebook_execution.py
tests/unit/test_notebook_placeholders.py
tests/unit/test_repository_checkout.py
```
(seven paths; `docs/notebooks/` and `src/mednorm_vi/reproducibility/` are new directories)

Modified (5):
```text
notebooks/MedNorm_Data_VietMed_Preprocess.ipynb   (real executable workflow; RUN_MODE; no placeholders)
tests/unit/test_notebooks.py                      (accept env-overridable DRIVE_ROOT default)
docs/training/first_colab_run.md                  (S1 notebooks marked DESIGN_DRAFT / not runnable)
docs/training/colab_execution_policy.md           (evidence-based readiness section)
docs/audits/README.md                             (0019 index entry)
```

Unchanged (immutable): `docs/audits/0018-*.md`, all earlier audits, the Architecture PDF.

## 15. Tests and static checks

```text
env PYTHONPATH=src python3 -m pytest -q          -> 625 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 219 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
build-governed-corpus                             -> vietmed_status not_included_pending; leakage 0
model_registry budget (full)                      -> within_9b true (8,408,616,512)
phase1c doctor                                    -> green; network access: none
```
New tests: 8 behavioral notebook-execution, 9 checkout-helper, 22 static placeholder/status.

## 16. Final Git state

HEAD `6817feb` (unchanged). Working tree = the 5 modified + 7 added paths above. No raw
dataset payload, credentials, or generated corpus is tracked; the Architecture PDF is
untouched.

## 17. Readiness for a Colab CPU rerun

The VietMed notebook is ready to rerun on a **fresh Colab CPU runtime**:

1. Open `notebooks/MedNorm_Data_VietMed_Preprocess.ipynb` (CPU runtime; no GPU).
2. Leave `RUN_MODE = "REAL"`. Edit only `DRIVE_ROOT` (and optionally pin `REPO_REF` to a
   commit SHA). Ensure the VietMed Parquet is at
   `DRIVE_ROOT/data/external/public_ner/vietmed_ner/data/`.
3. Run all cells top to bottom. The notebook installs pinned `pyarrow==17.0.0`, performs a
   **real** clone/fetch/checkout, prints the resolved commit SHA, imports the adapter from
   that checkout, resolves the schema columns, converts rows, asserts
   `offset_invalid == 0` and `human_review_required == 0`, re-runs conversion for
   determinism, writes the artifact tree, reloads it and verifies hashes, asserts audio
   absence, and prints the real files with sizes and hashes.
4. Copy `DRIVE_ROOT/data/derived/training_corpora/vietmed_ner_v1/` into the repository at
   `data/derived/training_corpora/vietmed_ner_v1/` (git-ignored).
5. Rebuild locally: `env PYTHONPATH=src python3 -m mednorm_vi.data_engine.cli
   build-governed-corpus` → expect `vietmed_status: included_from_artifacts`.

Only after step 4/5 may VietMed be recorded as `REAL_DATA_EXECUTED` / `ARTIFACTS_VERIFIED`.

## Safe to commit

Audit 0019 is internally consistent and **safe to commit** after human review. It does not
claim real VietMed artifacts exist, does not claim `REAL_DATA_EXECUTED`, and does not modify
the immutable Audit 0018.
