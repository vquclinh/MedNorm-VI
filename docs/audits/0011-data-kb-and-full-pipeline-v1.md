# Audit 0011 — Data/KB Preparation and Full Pipeline v1

- **Date:** 2026-07-17
- **Author:** Codex (AI agent), for human review
- **Change type:** Consolidated implementation milestone. No commit performed.
- **Spec reference:** `docs/MedNorm-VI_Architecture.pdf` v1.1, especially L1-L9,
  Data Engine, S0-S6, 9B model budget, frozen-KB provenance, offline inference,
  and fail-fast validation constraints.

## 1. Objective and scope

Implement the first coherent code-level Full Pipeline v1 surface for MedNorm-VI:

- derive a local normalized Vietnamese ICD-10 snapshot from the official TT06-2026 PDF text layer;
- build local ICD and RxNorm search indexes under ignored `indices/`;
- implement Data Engine, training workflow contracts, model registry/budget checks, Round-2 comparison;
- implement one shared L1-L9 inference path with deterministic, specialist, and full modes;
- add Colab Pro notebooks for S0-S6 training and offline inference;
- document the result without claiming trained checkpoints, competition readiness, or legal approval.

Raw medical resources were not moved, deleted, downloaded, OCRed, rewritten, or committed. No model
training, API use, prediction submission, or `output.zip` for organizer data was produced.

## 2. Initial Git state

```text
pwd: /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch: main
latest commit: c5a3f82 feat: register local medical knowledge resources
git status --short: clean
git diff --check: clean
```

## 3. Architecture implementation matrix

| Architecture item | Status after Audit 0011 |
| --- | --- |
| L1 Document Intelligence | Existing implementation retained; used by shared inference. |
| L2 Case Router | Existing route contracts retained; used by shared inference. |
| L3 Mention Factory | Deterministic medication/lab experts retained; neural adapter contracts added for ViHealthBERT, PhoBERT-W2NER, XLM-R MRC-NER, GLiNER, and anchored Qwen proposer. Neural predictions await checkpoints/training. |
| L4 Boundary & Type Resolver | Existing resolver extended to all five organizer-facing types when proposal evidence exists. Learned ranker/classifier training is planned, not trained. |
| L5 Assertion Hydra | Deterministic cue/scope fallback implemented; neural/calibrated assertion models await training. |
| L5 ICD-10 Super Linker | Local exact/accent-insensitive/n-gram/sparse retrieval implemented over generated ICD index; dense/reranker/calibration await training. |
| L5 RxNorm Super Linker | Local exact/n-gram/sparse/structured retrieval implemented over prescribable RxNorm index; Full RxNorm compatibility retained by contracts. |
| L6 Clinical Evidence Graph | Implemented immutable evidence nodes/edges and graph hash. |
| L7 Confidence Cascade | Deterministic fallback cascade and missing-checkpoint gates implemented; critic/adjudicator awaits Qwen LoRA. |
| L8 Metric-aware Set Decoder | Expected-Jaccard style deterministic set decoder implemented. |
| L9 Validation/Packaging | Organizer serializer and deterministic `output.zip` packaging implemented through existing validator. |
| Data Engine | Canonical schema, offset validation, adapters, folds, leakage checks, weak labels, review queues, synthetic fixtures, teacher contracts, build hashes implemented. |
| S0-S6 training | Dry-run training plan and CLI implemented; no large training run. |
| Safe/Lean/No-LLM profiles | Model registry validates shared-backbone budgets. |
| Offline inference | Full mode fails on missing checkpoints before writing predictions. |
| Round-2 upgrade support | Descriptor comparison CLI implemented. |

## 4. ICD conversion result

Source:

```text
data/external/icd10_vi/tt06-2026-official/06-byt-kem.pdf
```

The PDF has an embedded text layer and was converted with `pdftotext -layout`; OCR was not used.
The converter preserves source PDFs byte-for-byte and writes ignored derived artifacts under:

```text
data/derived/icd10_vi/tt06-2026/
```

Derived ICD snapshot inspection:

```text
ICD-10 snapshot: icd10-vi-local-c6b9568cf35b413a
categories      : 2011
chapters        : 24
concepts        : 15308
subcodes        : 13297
with_aliases    : 0
with_english    : 0
validation      : OK (0 warning(s))
```

ICD conversion artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `icd10_vi_normalized.csv` | 2,386,902 | `7d975de0f8b9bd16ded36e8abd29a3f5b4c4c1864183ec7593943a8fd59dc7d4` |
| `icd10_vi_aliases.csv` | 28 | `669b52c20b2416c34083bd36a71a3aa2c9ac6f99ddd2844eeb0682132298c833` |
| `icd10_vi_hierarchy.csv` | 344,209 | `9e26cabad4aff6c5f7717ba31f938e6895ac36cc8ace4d26bd75baf777e32062` |
| `conversion_manifest.yaml` | 638 | `9bef6a0e883bfe94a9e3c3b8e086ae96dd1b6defe04f9e7d2051d6574f9ba006` |
| `conversion_report.json` | 916 | `cd1f71b60881ee48175885412bb5293bd6f4cebff28faef99f797c4ab6f0c242` |

ICD limitations:

- labels are reconstructed from embedded PDF text, not from OCR or human review;
- aliases and English labels are not populated by this source conversion;
- organizer-exact compatibility remains unresolved;
- legal/reuse status remains `REVIEW_REQUIRED`.

## 5. KB indexes

ICD index:

```text
path: indices/icd10_vi/tt06-2026/index.json
file sha256: 94a1378b0f355afc6da23bb0da4d1e015dbbd4f51a1f2c0508f06e7b1727a768
bytes: 10138449
source_snapshot_id: icd10-vi-local-c6b9568cf35b413a
source_hash: 7d975de0f8b9bd16ded36e8abd29a3f5b4c4c1864183ec7593943a8fd59dc7d4
deterministic_index_hash: 4202a320e8be297852fcb96c1ec65f0614731285b36260bf957a4694a9cffdd5
records/concepts: 15308 / 15308
```

RxNorm index:

```text
path: indices/rxnorm/prescribable-2026-07-06/index.json
file sha256: 4e02ecbfc61e586e041a424177fa682f048618e8e0431e8204f6d2afc2692aec
bytes: 103374362
source_snapshot_id: rxnorm-local-75a100c0b70b67d0
source_hash: ecab2350f2ea03f0125041eb4fade4b642523b65cc649c0fd2d54136f033f335
deterministic_index_hash: ecaf487b86e27f0575ddba1904147fce9f214d148887e1519025ea34d1b16b10
records/concepts: 245401 / 82429
relations_seen: 2563978
attributes_seen: 3340763
```

Generated indexes are ignored by `.gitignore`; inference uses index lookups rather than linear scans.

## 6. RxNorm status

Existing RxNorm snapshot inspection was rerun:

```text
RxNorm snapshot: rxnorm-local-75a100c0b70b67d0
atoms                 : 245401
attributes            : 3340763
concepts              : 82429
distinct_sab          : 3
distinct_tty          : 26
relations             : 2563978
semantic_types        : 0
suppressed_concepts   : 0
validation            : OK (0 warning(s))
```

`RXNSTY.RRF` is absent from this Current Prescribable package, so semantic types remain `0`.
RxNorm Full 2026 remains missing pending UMLS approval. The prescribable package is not treated as
equivalent to the full monthly snapshot.

## 7. Manifests and governance

Updated tracked manifests:

- `data/manifests/icd10-vi-tt06-2026-official.yaml`
- `data/manifests/rxnorm-prescribable-2026-07-06.yaml`

Validation:

```text
icd10-vi-tt06-2026-official: OK (1 warning)
  warning: license status REVIEW_REQUIRED - not usable until reviewed

rxnorm-prescribable-2026-07-06: OK (2 warnings)
  warning: archive SHA-256 unknown because deletion preceded manifest completion
  warning: license status REVIEW_REQUIRED - not usable until reviewed
```

Both manifests remain `REVIEW_REQUIRED`. This is not a failed technical intake; it means human
license/reuse review is still required before governed use.

## 8. Pipeline and training surface

Tracked additions include:

- `src/mednorm_vi/kb/icd10/conversion/`
- `src/mednorm_vi/kb/indexing/`
- `src/mednorm_vi/data_engine/`
- `src/mednorm_vi/mention_factory/adapters.py`
- `src/mednorm_vi/linking/`
- `src/mednorm_vi/evidence_graph/graph.py`
- `src/mednorm_vi/confidence_cascade/cascade.py`
- `src/mednorm_vi/metric_decoder/decoder.py`
- `src/mednorm_vi/inference/`
- `src/mednorm_vi/training/`
- `src/mednorm_vi/model_registry/`
- `src/mednorm_vi/round2/`
- `configs/pipeline/full_v1.yaml`
- `configs/model_registry/models_v1.yaml`
- `configs/resources/icd10_vi_tt06_derived_columns.yaml`

Colab notebooks:

- `notebooks/MedNorm_DataEngine_and_Folds.ipynb`
- `notebooks/MedNorm_Mention_Assertion_Training.ipynb`
- `notebooks/MedNorm_ICD_RxNorm_Retrieval_Training.ipynb`
- `notebooks/MedNorm_Reranker_and_Resolver_Training.ipynb`
- `notebooks/MedNorm_Qwen_LoRA_and_Calibration.ipynb`
- `notebooks/MedNorm_Full_Offline_Inference_and_Packaging.ipynb`

Training plan output:

```text
plan_id: train-plan-037b3bdbe3b9eddb
seed: 20260717
mixed_precision: bf16
stages: S0 data_engine_and_folds; S1 mention_span_type_models; S2 assertion_models;
        S3 icd_rxnorm_retrieval; S4 reranker_resolver; S5 qwen_lora_critic_adjudicator;
        S6 calibration_and_packaging
```

No training was run.

## 9. Model budget

Profile budget checks:

```text
deterministic: total_parameters=0, within_9b=true
specialist   : total_parameters=137000000, within_9b=true
Safe         : total_parameters=410000000, within_9b=true
Lean         : total_parameters=0, within_9b=true
No-LLM       : total_parameters=0, within_9b=true
full         : total_parameters=8408616512, within_9b=true
```

`full --require-local-paths` reports missing local checkpoints, as expected. Shared backbones are not
double-counted.

## 10. Inference modes

The CLI entry point is implemented:

```text
python -m mednorm_vi.inference.cli run \
  --input-dir data/organizer_test/input \
  --output-zip outputs/output.zip \
  --config configs/pipeline/full_v1.yaml \
  --mode full
```

Observed full-mode readiness failure:

```text
pipeline not ready for mode=full:
missing_checkpoint:models/checkpoints/full_v1/mention/vihealthbert
missing_checkpoint:models/checkpoints/full_v1/mention/phobert_w2ner
missing_checkpoint:models/checkpoints/full_v1/mention/xlmr_mrc
missing_checkpoint:models/checkpoints/full_v1/mention/gliner
missing_checkpoint:models/checkpoints/full_v1/assertion/assertion_hydra
missing_checkpoint:models/checkpoints/full_v1/retrieval/icd_dense
missing_checkpoint:models/checkpoints/full_v1/retrieval/rxnorm_dense
missing_checkpoint:models/checkpoints/full_v1/reranker/cross_encoder
missing_checkpoint:models/checkpoints/full_v1/qwen_lora
missing_checkpoint:models/checkpoints/full_v1/calibration/full_v1
```

This is the intended behavior: no fake neural predictions are returned.

## 11. Doctor output

```text
MedNorm-VI Phase 1C-A — doctor
organizer registry     : 13 facts, 20/20 open policies, 36 hypotheses
position policies      : 6 registered (default raw-codepoint-half-open)
resource manifests     : 2 tracked, 2 templates
public NER manifests   : 3
RxNorm Prescribable    : available
RxNorm Full 2026       : MISSING / awaiting UMLS approval
ICD-10 VI source PDFs  : available
ICD-10 VI table        : available
RxNorm snapshot        : available
resolver               : ready
network access         : none
determinism hash       : 407070af11cf1d1e962ebe157c257311345405a5ce4c35005d53fd8ec2318bb2
```

Local untracked public-NER payload directories were observed under `data/external/public_ner/`. They
were not downloaded, integrated, converted, or reviewed in this milestone, and `.gitignore` now keeps
those raw payloads ignored. The repository still treats public NER datasets as not governed/usable.

## 12. Tests and static checks

Commands run:

```text
env PYTHONPATH=src python3 -m pytest -q tests/unit/test_full_pipeline_v1.py
6 passed in 0.45s

env PYTHONPATH=src python3 -m pytest -q tests/unit/test_phase1c_doctor.py
4 passed in 0.37s

env PYTHONPATH=src python3 -m pytest -q
470 passed in 15.51s

ruff check .
All checks passed!

env PYTHONPATH=src python3 -m mypy
Success: no issues found in 212 source files
```

The ignored deterministic preparation report is:

```text
reports/full_pipeline_v1/preparation_report.json
sha256: 6000f70d92a028fa031b86c209a54df9ef5f6fe829731d550ae46715908ca259
```

## 13. Repository hygiene

Confirmed during this audit:

- `docs/MedNorm-VI_Architecture.pdf` has no diff.
- No root duplicates of `06-byt.pdf`, `06-byt-kem.pdf`, or `RxNorm_full_prescribe_07062026.zip` exist.
- Raw ICD/RxNorm resources remain ignored.
- Derived ICD artifacts, KB indexes, reports, output zips, checkpoints, and public-NER raw payloads remain ignored.
- No `CLAUDE.md`, `AGENTS.md`, or tracked `.claude/` content was created.
- No network access was used by Codex.
- No training was run.
- No KB index is tracked.
- No organizer predictions or `output.zip` were generated for submission.

## 14. Known limitations

- RxNorm ZIP was already deleted before manifest completion; local ZIP MD5 and ZIP SHA-256 remain unavailable.
- RxNorm Full remains unavailable pending UMLS approval.
- RxNorm Prescribable is not equivalent to Full RxNorm.
- ICD conversion is text-layer based and not human-verified against the organizer hidden snapshot.
- ICD aliases and English labels are not populated by the official PDF conversion.
- Both medical-resource manifests remain `REVIEW_REQUIRED`.
- Public Vietnamese medical NER corpora are not governed/usable in this repository yet.
- No trained checkpoints exist.
- Dense retrieval, neural rerankers, learned resolver/classifier heads, Qwen LoRA critic/adjudicator, and calibration models await training.

## 15. Remaining manual actions

1. Complete human license/reuse review for ICD, RxNorm, and any public NER corpora.
2. Obtain RxNorm Full after UMLS approval and register it without changing linker contracts.
3. Review the derived ICD snapshot against official expectations and organizer policy hypotheses.
4. Build reviewed train/dev/test folds from lawful gold data.
5. Run the six Colab notebooks in order, exporting checkpoint manifests and hashes.
6. Populate `models/checkpoints/full_v1/` with reviewed checkpoint artifacts.
7. Re-run full offline inference only after resources and checkpoints are ready.

## 16. Readiness for commit

The resource-intake and Full Pipeline v1 code milestone is technically ready for human review and
commit. It is not trained, not legally approved for governed resource use, and not competition-ready.

Recommended next milestone: human review of Audit 0011, then Colab S0/S1 data-fold and mention/assertion
training dry run with checkpoint manifests.
