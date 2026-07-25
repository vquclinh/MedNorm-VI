# Governed Training Corpus v1 (Audits 0017, 0020)

Deterministic, reproducible training corpus built from the governance-approved
public Vietnamese medical NER datasets. Raw payloads and derived JSONL remain
git-ignored; this document + the tracked builder/config regenerate them.

## Build command (local, CPU-only, offline)

```bash
env PYTHONPATH=src python3 -m mednorm_vi.data_engine.cli build-governed-corpus \
  --base-dir data/external/public_ner \
  --out-dir  data/derived/training_corpora/mednorm_vi_training_v1 \
  --seed 20260723
```

## Sources and governance

Training eligibility is **USER_ATTESTED_PERMISSION** (author permission attested by
the user; evidence stored externally, not in the repo). Raw redistribution remains
prohibited/restricted; only internal training/development/ablation/domain-adaptation
is permitted. See `data/manifests/*-public-ner.yaml` (`training_use` block) and
`configs/resources/label_mappings/public_ner/*_v1.yaml`.

## Current v1 result (seed 20260723, Audit 0020)

| Metric | Value |
| --- | --- |
| canonical examples | 35,916 |
| typed entities | 15,757 |
| offset-invalid entities | **0** (`text[start:end] == entity.text`) |
| DIAGNOSIS / SYMPTOM / MEDICATION | 8,987 / 3,751 / 3,019 |
| TEST_NAME / TEST_RESULT | **0 / 0** (no source supervision) |
| splits train / validation / internal_test | 33,826 / 1,045 / 1,045 |
| split entities train / validation / internal_test | 11,720 / 1,991 / 2,046 |
| cross-split family leakage | **0** |
| eval MAP_APPROXIMATE entities | **0** |
| canonical corpus SHA-256 | `9e8060775d97713485c964df9165f640050a0085f385b2a9ff216c75f9d81588` |
| train / validation / internal_test SHA-256 | `892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a` / `ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103` / `e23acde0d87154d1750d25d1e47c92c86e457e42f2f97cb985661253dea7e135` |

Concrete contributors: ViMedNER (`ten_benh→DIAGNOSIS`, `trieu_chung_benh→SYMPTOM`;
12,738 entities), ViMQ (`drug→MEDICATION`; 905), and VietMed-NER
(`DRUGCHEMICAL→MEDICATION`; 2,114). PhoNER_COVID19 contributes domain text but
**0 concrete entities** (its only clinical label is a merged symptom+disease
category held for review and is never split into two targets).

VietMed-NER is included from the verified returned artifact at
`data/derived/training_corpora/vietmed_ner_v1/`. Audit 0020 records a one-time
human-approved exception: the returned generated manifest did not persist
`run_mode`, but the reviewer accepted REAL-equivalent provenance from concrete
Parquet source hashes, full row/entity counts, repository commit, resolved schema,
matching output hash, zero offset errors, zero human-review rows, and audio
exclusion. Future VietMed manifests must contain `run_mode: REAL`; the builder
rejects `SYNTHETIC_SMOKE` artifacts for governed-corpus inclusion.

## Layout (ignored)

```text
data/derived/training_corpora/mednorm_vi_training_v1/
  canonical_examples/public_ner_examples.jsonl
  splits/{train,validation,internal_test}.jsonl
  manifests/{corpus_manifest,split_manifest,source_contributions,annotation_coverage}.json
  quality/{leakage_summary,offset_validation,label_distribution}.json
```

Offsets are half-open `[start, end)`. The internal-test split is named
`internal_test` (never `test`) to avoid confusion with organizer test data. The
organizer's 100 input documents are **never** used as labeled train/dev/test.

## Annotation coverage and masked losses

No governed source provides **assertions** or **ICD/RxNorm candidates**; those
objectives are masked out per example (missing ≠ negative) — see
`src/mednorm_vi/data_engine/annotation_coverage.py`. Span/type are supervised only
for the concrete typed sources.

## Previous pre-VietMed baseline

Before Audit 0020, the governed corpus had 26,649 examples and 13,643 typed
entities. The final Audit 0017 split-policy-v2 state immediately before VietMed
was train=24,559, validation=1,045, and internal_test=1,045, with 905 MEDICATION
entities. Audit 0020 added all 9,267 VietMed examples to train only; validation
and internal_test counts and hashes remained unchanged.

## Known gaps (feed the synthetic/lexicon roadmap)

- **TEST_NAME / TEST_RESULT**: no supervision -> require synthetic laboratory
  generation + lexicon coverage (organizer input is lab-heavy).
- **MEDICATION**: improved to 3,019 spans, but VietMed contributes approximate
  labels only; structured medication-list/RxNorm supervision is still needed.
- **Assertions / candidates**: no gold -> synthetic + weak rules + ontology aliases.
- **VietMed-NER**: included as train-only approximate medication supervision.
  Validation and internal-test remain MAP_EXACT-only.
