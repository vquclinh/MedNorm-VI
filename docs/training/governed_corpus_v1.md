# Governed Training Corpus v1 (Audit 0017)

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

## v1 result (seed 20260723)

| Metric | Value |
| --- | --- |
| canonical examples | 26,649 |
| typed entities | 13,643 |
| offset-invalid entities | **0** (`text[start:end] == entity.text`) |
| DIAGNOSIS / SYMPTOM / MEDICATION | 8,987 / 3,751 / 905 |
| TEST_NAME / TEST_RESULT | **0 / 0** (no source supervision) |
| splits train / validation / internal_test | 18,657 / 3,997 / 3,995 |
| cross-split group leakage | **0** |

Concrete contributors: ViMedNER (`ten_benh→DIAGNOSIS`, `trieu_chung_benh→SYMPTOM`;
12,738 entities), ViMQ (`drug→MEDICATION`; 905). PhoNER_COVID19 contributes domain
text but **0 concrete entities** (its only clinical label is a merged
symptom+disease category held for review — never split into two targets).

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

## Known gaps (feed the synthetic/lexicon roadmap)

- **TEST_NAME / TEST_RESULT**: no supervision → require synthetic laboratory
  generation + lexicon coverage (organizer input is lab-heavy).
- **MEDICATION**: thin (905) → synthetic medication lists + RxNorm alias
  self-supervision.
- **Assertions / candidates**: no gold → synthetic + weak rules + ontology aliases.
- **VietMed-NER**: Parquet source; its adapter requires `pyarrow`. It is **not** in
  the v1 text build; add it via a documented pyarrow preprocessing step (or Colab),
  contributing `DRUGCHEMICAL→MEDICATION` (approximate) once converted.
