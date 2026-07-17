# Audit 0012 — Public Vietnamese Medical Dataset Intake and Governance

- **Date:** 2026-07-17
- **Author:** Codex (AI agent), for human review
- **Change type:** Consolidated public-dataset intake/governance milestone. No commit performed.
- **Spec reference:** `docs/MedNorm-VI_Architecture.pdf` v1.1, especially Data Engine,
  S0 data governance, public-corpus controls, leakage prevention, offline/no-network
  constraints, and the rule that no external corpus may be used before provenance and legal
  review are explicit.

## 1. Objective and scope

Audit 0012 verifies four already-local public Vietnamese medical NER datasets without
downloading, updating, training on, converting into final folds, or modifying raw payloads:

- `data/external/public_ner/phoner_covid19`
- `data/external/public_ner/vimedner`
- `data/external/public_ner/vimq`
- `data/external/public_ner/vietmed_ner`

The milestone adds concrete public-NER governance manifests, label-mapping configs, Data Engine
inspection/adaptation helpers, ignored deterministic aggregate reports, tests, and this audit.
It does not make any dataset legally usable for training.

## 2. Initial Git state

```text
pwd: /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch: main
latest commit: ce061e4 feat: implement full pipeline v1 foundation
git status --short: clean
git diff --check: clean
```

The workspace was clean at the start of this milestone. No network access was used.

## 3. Architecture relevance

This work supports the architecture's Data Engine and S0-S2 preparation path:

- records public-corpus provenance before use;
- separates raw ignored payloads from tracked metadata;
- blocks REVIEW_REQUIRED datasets from training builds;
- preserves exact source split identity and source-revision metadata;
- creates reviewable source-label to MedNorm-VI mapping configs;
- records leakage aggregates without copying raw examples into tracked files.

No L1-L9 inference behavior, KB linker, checkpoint, model registry, or prediction packaging behavior
was changed.

## 4. Dataset inventory

| Dataset | Local root | Revision/snapshot | Files | Bytes | Tree SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| PhoNER_COVID19 | `data/external/public_ner/phoner_covid19` | `9af190aa76adcfad0a2d14fab9bcab2b252c9c2e` | 43 | 21,923,658 | `0c75f4ed7742b918be3527140841894aa434a901f3aeac08ee36ec292ba407b4` |
| ViMedNER | `data/external/public_ner/vimedner` | `ce3deaa13837b5964c1fc8a83098d82418482a41` | 43 | 7,452,185 | `7ba9ad178b28f80805b2651369ae9fc8d9b1dc49a08dc3c8f1e217f54be10dc3` |
| ViMQ | `data/external/public_ner/vimq` | `b35341cd506f012a4d868674210c532d5a771063` | 48 | 3,583,659 | `d42cf7660adeb38941182bca072f93955c2cd47b5a2c8199dc93e56a40ea61e9` |
| VietMed-NER | `data/external/public_ner/vietmed_ner` | `e3d0393c733858402a7c04228f45d351d2ce6d8f` | 18 | 893,515,036 | `8c073f68b03a05738374b614c3e5314d9ac4a33fae94e866c16b1c9ab13f57aa` |

The first three roots are nested git checkouts and are not parent-repo submodules. VietMed-NER is
a Hugging Face-style snapshot with local cache metadata. No symlinks or Git LFS pointer files were
found. VietMed-NER contains five zero-byte Hugging Face `.lock` files; `lsof` found no open handles,
so they were treated as stale cache markers and left untouched.

## 5. Completeness and schema

PhoNER_COVID19:

- train/dev/test present for both `word` and `syllable` variants;
- CONLL and JSONL records are count- and content-consistent for all six split/variant pairs;
- BIO validation found 0 malformed sequences;
- label inventory hash: `1ee009e6d5a2116ce057c2b0f457fcea154baa04943b982c7f9ec693ca321994`.

ViMedNER:

- `train.txt`, `dev.txt`, `test.txt`, `labels.txt`, `ViMedNER.txt`, and `README.docx` present;
- `.DS_Store` files are present and recorded as non-data filesystem artifacts;
- BIO validation found 1 malformed transition in `train.txt`, 0 in dev/test;
- label inventory hash: `6123432e9cfbabb87c703913d8bebcbdc8f844e29452e06970b1bda128cc32ce`.

ViMQ:

- `train.json`, `dev.json`, `test.json`, and `entity_set.txt` present;
- JSON arrays parse successfully;
- entity annotations use inclusive token-index span triples, not BIO tags;
- label inventory hash: `f8edcd0465f0e9354d67b5ea5bd9263aa56014cadbfd2f8504681386fb90d9f9`.

VietMed-NER:

- train, validation, and test Parquet files are present;
- all three files have valid `PAR1` headers/footers;
- row counts: train 4,616; validation 1,154; test 3,497;
- schema is consistent across splits: `words`, `tags`, `labels`, `text`, `audio`, `duration`;
- `pyarrow`/`pandas` are not installed locally, so row-level label vocabulary and text leakage were
  not read. No label vocabulary was fabricated.

## 6. Governance and label mappings

All four datasets remain `REVIEW_REQUIRED` and are not eligible for MedNorm-VI training builds.
`REVIEW_REQUIRED` is a legal/governance gate, not a file-integrity failure.

| Dataset | License/reuse conclusion from local evidence | Training eligibility |
| --- | --- | --- |
| PhoNER_COVID19 | Local README states research/educational use only, no redistribution, and citation requirement; human review still required. | Not eligible |
| ViMedNER | Dataset license not identified locally; code utility Apache header does not establish dataset license. | Not eligible |
| ViMQ | Dataset license not identified locally. | Not eligible |
| VietMed-NER | Local dataset card requests citation but does not state license/reuse terms. | Not eligible |

Tracked mapping configs:

- `configs/resources/label_mappings/public_ner/phoner_covid19.yaml`
- `configs/resources/label_mappings/public_ner/vimedner.yaml`
- `configs/resources/label_mappings/public_ner/vimq.yaml`
- `configs/resources/label_mappings/public_ner/vietmed_ner.yaml`

Mappings distinguish exact, partial, ambiguous, auxiliary-only, unmappable, and
human-review-required source labels. Disease/symptom/procedure/treatment labels were not silently
collapsed into MedNorm-VI labels.

## 7. Leakage analysis

Aggregate-only leakage report:

- text-accessible examples checked: 36,676;
- PhoNER_COVID19 exact duplicate groups: 104; cross-split or cross-variant groups: 85;
- ViMedNER exact duplicate groups: 2; cross-split groups: 2;
- ViMQ exact duplicate groups: 1; cross-split groups: 1;
- exact cross-dataset duplicate groups among text-accessible datasets: 0;
- deterministic near-duplicate candidate groups by first/last normalized-token key: 2,973.

VietMed-NER row-level leakage was not run because no Parquet row reader is installed. Footer/schema
integrity was verified independently.

## 8. Manifests and reports

Tracked manifests:

- `data/manifests/phoner-covid19-public-ner.yaml`
- `data/manifests/vimedner-public-ner.yaml`
- `data/manifests/vimq-public-ner.yaml`
- `data/manifests/vietmed-ner-public-ner.yaml`

Manifest validation:

```text
phoner_covid19: OK (1 warning: REVIEW_REQUIRED not usable until reviewed)
vimedner: OK (1 warning: REVIEW_REQUIRED not usable until reviewed)
vimq: OK (1 warning: REVIEW_REQUIRED not usable until reviewed)
vietmed_ner: OK (1 warning: REVIEW_REQUIRED not usable until reviewed)
```

Ignored deterministic reports:

| Report | SHA-256 |
| --- | --- |
| `reports/public_dataset_intake/inventory.json` | `5b40f401b2b8d2cf389e9d786ba9995bf189df71c21abb28c5a06771499ddcbd` |
| `reports/public_dataset_intake/schema_summary.json` | `a952ebf12f3d38f15d705152a6f715fc78f4dc474387d6318d7433f471eea759` |
| `reports/public_dataset_intake/label_mapping.json` | `a18c4ff9d0c50c56752e28c4d2638cc95a573c21f3c0bbd12086640f3420b52c` |
| `reports/public_dataset_intake/leakage_summary.json` | `207afe0bd7ef5bd187705b180b5a01d8673dd29b790a30d6c22ba908f3548037` |
| `reports/public_dataset_intake/governance_summary.md` | `84c1d2cb80d8a6778333d49f84c6a4048e9a42790bf06193e698d9be048bbfe1` |

Reports contain aggregate metadata only and no raw examples.

## 9. Implementation changes

Code:

- extended `src/mednorm_vi/resources/ner.py` with local public-NER intake metadata validation,
  label-mapping review states, tree/hash validation, and stricter NER governance warnings;
- added `src/mednorm_vi/data_engine/public_ner.py` for deterministic tree hashing, source revision
  parsing, CONLL/JSONL/ViMQ readers, Parquet footer validation, canonical adapters, duplicate
  aggregates, and REVIEW_REQUIRED training gates;
- exported the training gate from `src/mednorm_vi/data_engine/__init__.py`.

Tests:

- added `tests/unit/test_public_ner_intake.py` using synthetic fixtures only.

Repository hygiene:

- removed `data/external/public_ner/README.md` from the Git index only so
  `git ls-files data/external/public_ner` is empty. The local file remains on disk and ignored.

## 10. Ignore verification

Representative checks:

```text
data/external/public_ner/phoner_covid19/data/word/train_word.conll -> .gitignore:42:data/external/public_ner/**
data/external/public_ner/vimedner/data/train.txt -> .gitignore:42:data/external/public_ner/**
data/external/public_ner/vimq/data/train.json -> .gitignore:42:data/external/public_ner/**
data/external/public_ner/vietmed_ner/data/train-00000-of-00001.parquet -> .gitignore:42:data/external/public_ner/**
data/external/public_ner/vietmed_ner/.cache/huggingface -> .gitignore:42:data/external/public_ner/**
data/external/public_ner/phoner_covid19/.git/HEAD -> .gitignore:42:data/external/public_ner/**
reports/public_dataset_intake/inventory.json -> .gitignore:45:reports/**
```

`git ls-files data/external/public_ner` now returns no paths. No raw dataset file, nested Git object,
Hugging Face cache file, Parquet payload, or generated report is tracked.

## 11. Tests and static checks

```text
env PYTHONPATH=src python3 -m pytest -q tests/unit/test_public_ner_intake.py tests/unit/test_resource_governance.py
22 passed in 0.28s

env PYTHONPATH=src python3 -m pytest -q
479 passed in 15.76s

ruff check .
All checks passed!

env PYTHONPATH=src python3 -m mypy
Success: no issues found in 213 source files

git diff --check
clean
```

## 12. Known limitations

- No dataset is approved for training yet.
- VietMed-NER label vocabulary and row-level leakage need a reviewed Parquet row reader such as
  `pyarrow`; only footer/schema metadata was inspected here.
- ViMedNER has one malformed BIO transition in the train split.
- PhoNER_COVID19 has explicit no-redistribution wording and ambiguous `SYMPTOM_AND_DISEASE` labels.
- ViMQ uses token-index span triples and sentence-level labels, not BIO or original character offsets.
- Near-duplicate candidates are aggregate deterministic candidates and require human review before
  any fold decisions.

## 13. Current status

Current `git status --short` at audit-writing time:

```text
D  data/external/public_ner/README.md
 M src/mednorm_vi/data_engine/__init__.py
 M src/mednorm_vi/resources/ner.py
?? configs/resources/label_mappings/
?? data/manifests/phoner-covid19-public-ner.yaml
?? data/manifests/vietmed-ner-public-ner.yaml
?? data/manifests/vimedner-public-ner.yaml
?? data/manifests/vimq-public-ner.yaml
?? src/mednorm_vi/data_engine/public_ner.py
?? tests/unit/test_public_ner_intake.py
```

The audit file and audit index update are additional tracked changes after this section was drafted.

## 14. Remaining manual actions

- Human legal/reuse review for all four datasets.
- Confirm required citations and attribution wording for each dataset.
- Decide whether PhoNER_COVID19's research/education-only and no-redistribution terms permit the
  intended internal MedNorm-VI use.
- Install or provide an approved Parquet row reader if VietMed-NER row-level label analysis is
  required.
- Resolve label mappings for ambiguous/partial labels before any training.
- Build final public-data folds only after governance approval and leakage review.

## 15. Readiness and next milestone

Audit 0012 is ready for human review. The public-dataset intake metadata is technically complete
for the three text-readable datasets and metadata-complete for VietMed-NER, but none of the four
datasets is training-eligible. The recommended next milestone is human license/reuse review plus
dataset-label adjudication; only then should MedNorm-VI build governed public training folds.
