# Private-Rebuild Data Plan (Audit 0017)

If BTC requests source code, data, model weights, and a README for a private rebuild,
this document classifies every resource and states the rebuild gates. **Restricted raw
data is never promised as redistributable.** Only code/config/manifests are tracked;
raw payloads, derived corpora, indexes, and weights are git-ignored and handled per the
classification below.

## Classification legend

- **MAY_BE_BUNDLED** — small, unrestricted, safe to include in the source package.
- **MUST_BE_REDOWNLOADED_BY_LICENSED_USER** — the licensed user re-obtains it (we do not
  redistribute).
- **MAY_BE_PROVIDED_PRIVATELY_TO_BTC_SUBJECT_TO_LICENSE** — could be shared privately with
  the organizer only if the source license/attestation permits; otherwise not.
- **GENERATED_FROM_TRACKED_CODE** — reproducible from tracked code/config; not shipped.
- **ORGANIZER_PROVIDED** — BTC already has it.
- **REVIEW_REQUIRED_BEFORE_SUBMISSION** — needs human/legal check before any sharing.

## Resource classification

| Resource | Classification | Rebuild gate / command |
| --- | --- | --- |
| Organizer input (`data/organizer_test/`) | ORGANIZER_PROVIDED | BTC provides; never redistributed by us. |
| ICD-10 source PDFs (`data/external/icd10_vi/…`) | MUST_BE_REDOWNLOADED_BY_LICENSED_USER / REVIEW_REQUIRED_BEFORE_SUBMISSION | MoH TT06-2026 source; license `REVIEW_REQUIRED`. |
| ICD derived snapshot (`data/derived/icd10_vi/…`) | GENERATED_FROM_TRACKED_CODE | `kb.icd10.conversion.cli` from the source PDFs (deterministic, no OCR). |
| RxNorm Prescribable (`…/prescribable-2026-07-06/`) | MUST_BE_REDOWNLOADED_BY_LICENSED_USER | NLM no-login download; `REVIEW_REQUIRED`. |
| RxNorm Full (`…/full-2026-07-06/`) | MUST_BE_REDOWNLOADED_BY_LICENSED_USER | UMLS-licensed (`REDISTRIBUTION_RESTRICTED`); user re-downloads with their UMLS license. |
| RxNorm/ICD indexes (`indices/…`) | GENERATED_FROM_TRACKED_CODE | `kb.indexing.cli build-rxnorm/build-icd`. |
| PhoNER_COVID19 / ViMedNER / ViMQ / VietMed-NER (`data/external/public_ner/…`) | MUST_BE_REDOWNLOADED_BY_LICENSED_USER + REVIEW_REQUIRED_BEFORE_SUBMISSION | Source licenses `REVIEW_REQUIRED`; internal training is USER_ATTESTED_PERMISSION only; **raw redistribution prohibited**. Re-obtain from the recorded `source_revision`. |
| Governed corpus (`data/derived/training_corpora/mednorm_vi_training_v1/`) | GENERATED_FROM_TRACKED_CODE | `data_engine.cli build-governed-corpus` after the public datasets are re-obtained. Derived from restricted data → same redistribution restriction as its sources. |
| VietMed derived (`…/vietmed_ner_v1/`) | GENERATED_FROM_TRACKED_CODE | Colab `MedNorm_Data_VietMed_Preprocess.ipynb` (pyarrow), then rebuild. |
| Synthetic data | GENERATED_FROM_TRACKED_CODE / MAY_BE_BUNDLED (tiny fixtures) | `data_engine.synthetic`; large synthetic corpus via a Colab/Drive step. |
| Model base weights | MUST_BE_REDOWNLOADED_BY_LICENSED_USER | Pinned revisions in the S0–S6 notebooks; downloaded in Colab under each base-model license. |
| Adapters / LoRA | MAY_BE_PROVIDED_PRIVATELY_TO_BTC_SUBJECT_TO_LICENSE / REVIEW_REQUIRED_BEFORE_SUBMISSION | Trained artifacts; check base-model license before sharing. |
| Checkpoints / optimizer states | REVIEW_REQUIRED_BEFORE_SUBMISSION | Returned from Colab; weights git-ignored; manifests tracked. |
| Code / configs / manifests / mappings / notebooks / docs / tests | MAY_BE_BUNDLED | The tracked source package. |

## Rebuild order (deterministic where possible)

1. Re-obtain restricted raw data (ICD PDFs, RxNorm, four public datasets) under the
   appropriate licenses/attestations; verify recorded `source_revision` / payload hashes.
2. `kb.icd10.conversion.cli` + `kb.indexing.cli` → ICD/RxNorm derived + indexes.
3. `data_engine.cli build-governed-corpus` → governed corpus v1 (verify split hashes,
   `cross_split_family_leakage == 0`, `offset_invalid == 0`).
4. Optional Colab: VietMed preprocess → rebuild corpus to include VietMed.
5. Colab S0–S6 (pinned base-model revisions) → checkpoints + manifests → return to
   `models/checkpoints/full_v1/…`.
6. Full-mode inference (fails fast until required checkpoints exist).

## Non-promises

We do not promise redistribution of RxNorm, ICD source PDFs, or the four public datasets;
those are re-obtained by the licensed user. Derived artifacts inherit their sources'
restrictions. UMLS/UTS credentials and any permission correspondence are never stored in
the repository.
