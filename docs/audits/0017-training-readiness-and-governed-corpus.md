# Audit 0017 — Training Readiness, Governed Corpus, and Colab Workflows

- **Date:** 2026-07-23
- **Author:** Claude (AI agent), for human review
- **Change type:** Training-readiness + governed-corpus milestone. **No commit. No push.**
  **No model trained/fine-tuned locally. No weights/tokenizers downloaded locally.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (L1–L9, Data Engine §14, S0–S6 §15,
  9B budget §17, offset/fail-fast, reproducibility). Also read Audits 0001–0016,
  `docs/task_contracts/btc_turn2_vong1_contract_review.md`, all Data Engine code,
  manifests, mappings, notebooks, registry/budget configs, `.gitignore`.

Model acquisition and all GPU/neural work (model/tokenizer download, S0–S6 training,
embeddings, reranking, LoRA, neural pseudo-labeling, GPU inference) is **Colab-only**.
**VietMed Parquet preprocessing is also Colab-only by project policy** — but it is
**deterministic CPU data preparation and does not require a GPU** (it is not a GPU/neural
workload; it is kept off the local machine only because pyarrow is not a repo runtime
dependency). Local work here is deterministic and offline: governance, corpus conversion,
offset validation, label mapping, duplicate-family leakage analysis, split generation,
annotation-coverage/masked-loss contracts, synthetic fixtures, weak-supervision
contracts, tests, notebooks, docs.

## 1. Initial Git state

```text
branch : main   HEAD : 3aa44fc  chore: standardize organizer data layout and contract
Audit 0016 committed; the changes in this audit are the only uncommitted work.
```

## 2. Architecture traceability

No architecture change; no base model added. L1–L9 preserved. The governed corpus and
adapters feed **S1** mention extraction and support **L4 Boundary & Type Resolver**
development. **S2** assertions and **S3** linkers remain data-gated; **S4** is reranking,
while **S5/S6** require upstream checkpoints and out-of-fold predictions. (Note: L4 is the
Boundary & Type Resolver architecture layer; S4 is the reranking training stage — distinct.)
Offset contract, 9B budget, offline/fail-fast preserved.

## 3. BTC contract & end-exclusive offsets

Output contract per `btc_turn2_vong1_contract_review.md`: one JSON per TXT; five Vietnamese
types; assertions on symptom/diagnosis/medication; `CHẨN_ĐOÁN→ICD-10`, `THUỐC→RxNorm`.
Position is half-open `[start, end)` end-exclusive — **CONFIRMED** by the organizer example
(`amlodipine 10 mg po daily`, 25 chars → `[58, 83]`, `text[58:83] == span`), matching
`POSITION_IS_END_EXCLUSIVE = True`. Canonical offsets use the same `text[start:end] == text`
invariant (0 violations in the built corpus).

## 4. Reproducibility / private-rebuild

Every training-ready artifact records source IDs/revisions, license/governance, mapping
version, split-policy version, content/config hashes, and seed. See §5 and
`docs/reproducibility/private_rebuild_data_plan.md`. Restricted raw data and weights stay
out of the tracked package; the corpus is regenerable from tracked code/config.

## 5. Resource inventory

Public NER (phoner_covid19, vimedner, vimq, vietmed_ner), organizer active input + archives,
ICD-10 source+derived, RxNorm Prescribable + Full + indexes — all present and unchanged
(doctor green; RxNorm Full/Prescribable + ICD intact).

## 6. Governance / license separation

The **source license fact is preserved and never rewritten**. Training permission is a
separate block.

| Field | Value (all four datasets) |
| --- | --- |
| `license.status` (source) | **REVIEW_REQUIRED** (unchanged source fact) |
| manifest validation | **OK with exactly 1 WARNING** each: `ner.not_yet_usable` (source license review-required) — this is deliberate and accurate, not an error |
| technical readability | readable (3 text datasets; VietMed via pyarrow) |
| `training_use.status` | USER_ATTESTED_PERMISSION (`permission_attested_by_user: true`, `evidence_storage: external_not_tracked`, `internal_training_allowed: true`, `development_evaluation_allowed: true`, `raw_redistribution_allowed: false`) |
| redistribution eligibility | raw redistribution **prohibited/restricted** |

`NerDatasetManifest.training_use` is a real field; `require_training_allowed` gates on it
(not on the source license). No email/identity/credential/correspondence is stored.

## 7. Dataset inclusion matrix

| Dataset | Source license | training_use | Readable | Adapter complete | In corpus v1 | Types contributed | Objectives | In val/internal_test | Pending |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PhoNER_COVID19 | REVIEW_REQUIRED | ATTESTED | yes | yes | **yes** | **none** | **domain adaptation text only** | no | — |
| ViMedNER | REVIEW_REQUIRED | ATTESTED | yes | yes | **yes** | DIAGNOSIS, SYMPTOM | span+type | yes (MAP_EXACT) | — |
| ViMQ | REVIEW_REQUIRED | ATTESTED | yes | yes | **yes** | MEDICATION | span+type | yes (MAP_EXACT) | — |
| VietMed-NER | REVIEW_REQUIRED | ATTESTED | raw Parquet present; readable with pyarrow (Colab) | **artifacts not yet returned (pending)** | **no** | (DRUGCHEMICAL→MEDICATION, approx) | — | no | Colab CPU preprocess |

VietMed-NER — distinct states, do not conflate:

- **raw source present:** yes (`data/external/public_ner/vietmed_ner/` Parquet).
- **readable with pyarrow:** yes, but only in Colab (pyarrow is not a repo runtime dependency).
- **canonical adapter artifacts available:** **no** — the preprocessing notebook has not been
  run and its derived JSONL/manifest have not been returned to the repo.
- **included in governed corpus v1:** **no**.

Net: **governance training allowed · raw Parquet present · readable with pyarrow in Colab ·
canonical preprocessing artifacts not yet returned · adapter inclusion pending · NOT included
in governed corpus v1.**

## 8. Source-label mappings

Versioned `configs/resources/label_mappings/public_ner/{phoner_covid19,vimedner,vimq,vietmed_ner}_v1.yaml`
with per-label `MAP_EXACT | MAP_APPROXIMATE | CONTEXT_ONLY | DROP | REVIEW_REQUIRED`,
rationale, and training-permission flags. Merged categories (`SYMPTOM_AND_DISEASE`,
`DISEASESYMTOM`) are **REVIEW_REQUIRED** — never split into two targets. Concrete typed
maps (type-exact; boundary review flagged separately): `ten_benh→CHẨN_ĐOÁN`,
`trieu_chung_benh→TRIỆU_CHỨNG` (ViMedNER, MAP_EXACT); `drug→THUỐC` (ViMQ, MAP_EXACT);
`DRUGCHEMICAL→THUỐC` (VietMed, MAP_APPROXIMATE, pyarrow-gated).

## 9. Adapters & repairs

Deterministic adapters (`bio_examples_to_documents`, `vimq_records_to_documents`)
reconstruct half-open offsets and enforce `text[start:end] == entity.text`. Repairs:
ViMedNER orphan `I-` tags (no preceding `B-`) are ignored → no annotation (deterministic,
provenance-safe); ViMQ inclusive token-index spans → character half-open + validated.
VietMed Parquet adapter deferred to the Colab CPU notebook.

## 10. Duplicate-family algorithm

`duplicate_families()` unions three kinds via union-find, with stable hash-only family IDs:
1. **exact-byte** duplicates;
2. **normalized** duplicates (NFC + casefold + accent-fold + punctuation→space +
   whitespace-collapse);
3. **near-duplicates** — 64-bit SimHash over character 4-grams with a **four-band LSH**
   candidate step. Bands are bits 0–15 / 16–31 / 32–47 / 48–63. Any pair with Hamming
   distance ≤ 3 differs in ≤ 3 bit positions, which touch at most 3 of the 4 bands, so it
   shares ≥ 1 identical band and is generated as a candidate (**complete recall for
   Hamming ≤ 3**). Each unique candidate is then verified by the full 64-bit Hamming
   distance and accepted only if ≤ 3.

Adversarial tests prove: Hamming-1 in every band is caught; Hamming-3 across three bands is
caught; Hamming-4 (one bit per band) is a candidate but **rejected** by the final threshold;
identical hashes pair; family assignment is byte-identical across runs.

## 11. Leakage limitations

Reported separately and honestly: **0 cross-split leakage among exact duplicates, 0 among
normalized duplicates, and 0 among detected near-duplicate (SimHash Hamming ≤ 3) families.**
Splits assign **whole families atomically**, so no family can cross splits by construction
(cross-split family violations = 0 before and after). This does **not** claim absolute
absence of all possible semantic/paraphrase leakage — heuristic SimHash near-duplicate
detection does not capture deep semantic templating; that remains a residual limitation.

## 12. Final governed corpus (seed 20260723, split-policy-v2-duplicate-family)

```text
canonical examples : 26,649     typed entities : 13,643     offset-invalid : 0
target types  : DIAGNOSIS 8,987   SYMPTOM 3,751   MEDICATION 905   TEST_NAME 0   TEST_RESULT 0
duplicate families : 26,489   exact groups : 26,641   normalized groups : 26,571
near-duplicate candidate pairs (Hamming<=3, verified) : 214
examples_jsonl sha256      : 92ea02ac2dbb8eee775de8a9d637eab78cd04f4be2f3626d44b687f5aa9d0a4a
corpus_manifest.json sha256: ddadd20568104fac8af4aebe70cf51f0fa37764476434c9b235321348764483c
```
Deterministic: two rebuilds produced **byte-identical** split files.

## 13. Split counts and hashes

```text
split           examples   entities(D/S/M)          sha256
train           24,559     6338 / 2649 / 619         2a8539e581cbfa027b34e1c0dcd9a883a96ebeb4759db1dcd2116a9b3a26f3a4
validation       1,045     1302 / 533 / 156          ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103
internal_test    1,045     1347 / 569 / 130          e23acde0d87154d1750d25d1e47c92c86e457e42f2f97cb985661253dea7e135
cross-split FAMILY leakage : 0     eval MAP_APPROXIMATE entities : 0
```
Validation/internal_test contain **only MAP_EXACT** entities; no REVIEW_REQUIRED/approximate/
repaired/synthetic examples, and no organizer document is used as labeled data. The internal
split is named `internal_test` (never `test`).

## 14. Per-source / per-type / mapping-quality distributions

```text
source contributions : phoner_covid19 10,027 docs / 0 entities ;
                       vimedner 7,622 docs / 12,738 entities ; vimq 9,000 docs / 905 entities
per-split source     : train {phoner 10027, vimedner 5796, vimq 8736} ;
                       validation {vimedner 906, vimq 139} ; internal_test {vimedner 920, vimq 125}
per-split mapping     : train {MAP_EXACT 9606, MAP_APPROXIMATE 0} ;
                       validation {MAP_EXACT 1991, MAP_APPROXIMATE 0} ;
                       internal_test {MAP_EXACT 2046, MAP_APPROXIMATE 0}
```

## 15. Annotation coverage & loss masks

`annotation_coverage.py`: per-source coverage → per-example loss masks. No governed source
provides assertions or ICD/RxNorm candidates, so those objectives are masked out (missing ≠
negative; unannotated types not treated as true negatives). Tests enforce masking.

## 16. Synthetic fixtures

`synthetic.py::synthetic_fixture_examples()` — tiny deterministic, offset-validated fixtures
covering the priority gaps: TEST_NAME + TEST_RESULT (value-only, value+unit, comparison,
qualitative), medication list boundary, and negation/family/historical assertions. Only tiny
fixtures are tracked; the large synthetic corpus is a Colab/Drive step; synthetic never
enters validation/internal_test (the governed builder does not read them).

## 17. Weak-supervision contracts

`weak_labeling.py`: `LabelingFunctionVote` (label/span/confidence/evidence, `ABSTAIN`),
deterministic `aggregate_votes` (majority + conflict flag + confidence threshold + abstention),
`is_valid_weak_label`. No neural pseudo-labeling / label-model training locally.

## 18. Organizer-domain gap

`reports/training_readiness/organizer_training_domain_gap.json` (aggregate; no raw text): the
active organizer input is medication/lab-heavy (137 med lines, 140 lab lines, 36 qualitative
results). Top gap = **TEST_NAME/TEST_RESULT** (0 governed supervision), then thin **MEDICATION**.
Recommends synthetic laboratory + medication generators and lexicon coverage. Thresholds NOT
tuned against organizer input.

## 19. Notebook inventory (10 notebooks; from the filesystem)

| Path | Role | Env | GPU | Complete workflow | Smoke | Full default | Confirm gate | Model download | Resume | Manifest export | Return-to-repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `MedNorm_Data_VietMed_Preprocess.ipynb` | VietMed Parquet→JSONL data prep | Colab | no (CPU) | yes | n/a | n/a | n/a | none (pyarrow only) | n/a | yes | yes |
| `MedNorm_S0_DomainAdaptation.ipynb` | S0 continued pretraining | Colab | yes | yes | yes | OFF | CONFIRM_FULL | Colab-only, pinned | yes | yes | yes |
| `MedNorm_S1_MentionExtraction.ipynb` | S1 mention specialists | Colab | yes | yes | yes | OFF | CONFIRM_FULL | Colab-only, pinned | yes | yes | yes |
| `MedNorm_S2_Assertion.ipynb` | S2 assertion heads | Colab | yes | yes | yes | OFF | CONFIRM_FULL | Colab-only, pinned | yes | yes | yes |
| `MedNorm_S3_Retrieval.ipynb` | S3 ICD/RxNorm embeddings | Colab | yes | yes | yes | OFF | CONFIRM_FULL | Colab-only, pinned | yes | yes | yes |
| `MedNorm_S4_Reranker.ipynb` | S4 cross-encoder reranker | Colab | yes | yes | yes | OFF | CONFIRM_FULL | Colab-only, pinned | yes | yes | yes |
| `MedNorm_S5_QwenLoRA.ipynb` | S5 Qwen LoRA critic/adjudicator | Colab | yes | yes | yes | OFF | CONFIRM_FULL | Colab-only, pinned | yes | yes | yes |
| `MedNorm_S6_Calibration.ipynb` | S6 OOF calibration/stacking | Colab | optional | yes | yes | OFF | CONFIRM_FULL | Colab-only, pinned | yes | yes | yes |
| `MedNorm_S1_Mention_FirstRun_Smoke.ipynb` | Focused S1 smoke (≤3 batches) | Colab | yes | yes | yes | OFF | — | Colab-only, pinned | n/a | yes | yes |
| `MedNorm_Full_Offline_Inference_and_Packaging.ipynb` | Retained L9 inference/packaging | local/Colab | no | (retained) | — | — | — | — | — | — | — |

Arithmetic: **1 VietMed prep + 7 stage (S0–S6) + 1 S1 smoke + 1 retained inference/packaging
= 10.** All valid JSON; 0 secrets; 0 personal paths; single editable `PROJECT_ROOT`; pinned
model revisions; Colab-only downloads; full mode OFF by default; export manifests+hashes;
resume. Five superseded thin scaffolds were removed (see §26). A test compares this inventory
to the filesystem and to this audit.

## 20. Colab-only execution policy

`docs/training/colab_execution_policy.md` — local-allowed vs Colab-required, stop-and-hand-off
rule, return-to-repo contract. No training/fine-tuning/downloads locally.

## 21. First & second Colab actions

- **First:** `notebooks/MedNorm_Data_VietMed_Preprocess.ipynb` — CPU; pinned pyarrow inside
  Colab; configurable Drive paths; emits canonical JSONL + manifest/counts/label-inventory/
  hashes; excludes audio; return-artifact instructions.
- **Second:** `notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb` — GPU smoke only (≤3
  batches; no full-training claim; no auto full run; checkpoint+manifest returned).

**Full S1 is NOT the next automatic step.** Do not recommend it until: VietMed artifacts are
returned and inclusion decided; the governed corpus is rebuilt; TEST_NAME/TEST_RESULT
supervision is adequate; medication coverage improves; approximate-mapping masks are
validated; and the smoke checkpoint is locally integrated and validated.

## 22. Checkpoint contracts

`configs/model_registry/checkpoint_manifest.template.yaml` (role, base/tokenizer revision,
param counts, dataset/split/config hashes, seed, metrics, artifact hashes, license,
redistribution, expected path). No empty weight files. Full mode still fails fast on missing
checkpoints (not weakened).

## 23. 9B budget

From config/manifests only (no weights downloaded): deterministic 0, specialist 137M, Safe
410M, **full 8,408,616,512 (`within_9b = true`)**. `full --require-local-paths` reports
missing checkpoints.

## 24. Module readiness matrix (conservative)

| Module | Status |
| --- | --- |
| L3 Mention Factory | READY_FOR_SMOKE_TRAINING (DIAGNOSIS/SYMPTOM/MEDICATION); **NOT_READY_FOR_FULL_FIVE_TYPE_TRAINING** |
| L4 Boundary & Type Resolver | READY_FOR_SMOKE_TRAINING; type-risk review required |
| L5A Assertion Hydra | BLOCKED_BY_MISSING_LABELS (synthetic assertions for smoke only) |
| L5B ICD-10 linker | NEEDS_ONTOLOGY_SELF_SUPERVISION_AND_COLAB_TRAINING |
| L5C RxNorm linker | NEEDS_ONTOLOGY_SELF_SUPERVISION_AND_COLAB_TRAINING |
| L6 Evidence Graph | READY_RULE_BASED |
| L7 Confidence Cascade | BLOCKED_BY_MISSING_CHECKPOINTS |
| L8 Metric Decoder | READY_DETERMINISTIC |

## 25. Tests & static checks

```text
env PYTHONPATH=src python3 -m pytest -q          -> 562 passed
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 216 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
build-governed-corpus x2                          -> byte-identical (deterministic)
validate-resource-manifest (4 NER)                -> OK, 1 warning each (source-license review-required)
model_registry budget / doctor                    -> full within 9B; doctor green
```
Deterministic checks: governance separation; concrete mappings; duplicate-family clustering;
four-band SimHash adversarial recall (Hamming 1/3 accepted, 4 rejected, identical, stable);
split family leakage 0; annotation masks; offset invariant; synthetic fixtures; weak-vote
aggregation; notebook JSON/section/safety; notebook-inventory/audit consistency; checkpoint
template; parameter budget; Phase 1C doctor.

## 26. Exact tracked changes

Modified (13):
```text
data/manifests/phoner-covid19-public-ner.yaml
data/manifests/vietmed-ner-public-ner.yaml
data/manifests/vimedner-public-ner.yaml
data/manifests/vimq-public-ner.yaml
docs/audits/README.md
src/mednorm_vi/data_engine/__init__.py
src/mednorm_vi/data_engine/cli.py
src/mednorm_vi/data_engine/public_ner.py
src/mednorm_vi/data_engine/synthetic.py
src/mednorm_vi/data_engine/weak_labeling.py
src/mednorm_vi/resources/ner.py
tests/unit/test_organizer_output_contract.py
tests/unit/test_public_ner_intake.py
```
Added (24):
```text
configs/model_registry/checkpoint_manifest.template.yaml
configs/resources/label_mappings/public_ner/phoner_covid19_v1.yaml
configs/resources/label_mappings/public_ner/vietmed_ner_v1.yaml
configs/resources/label_mappings/public_ner/vimedner_v1.yaml
configs/resources/label_mappings/public_ner/vimq_v1.yaml
docs/audits/0017-training-readiness-and-governed-corpus.md
docs/reproducibility/private_rebuild_data_plan.md
docs/training/colab_execution_policy.md
docs/training/first_colab_run.md
docs/training/governed_corpus_v1.md
notebooks/MedNorm_Data_VietMed_Preprocess.ipynb
notebooks/MedNorm_S0_DomainAdaptation.ipynb
notebooks/MedNorm_S1_MentionExtraction.ipynb
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
notebooks/MedNorm_S2_Assertion.ipynb
notebooks/MedNorm_S3_Retrieval.ipynb
notebooks/MedNorm_S4_Reranker.ipynb
notebooks/MedNorm_S5_QwenLoRA.ipynb
notebooks/MedNorm_S6_Calibration.ipynb
src/mednorm_vi/data_engine/annotation_coverage.py
src/mednorm_vi/data_engine/corpus_build.py
tests/unit/test_notebooks.py
tests/unit/test_synthetic_weak.py
tests/unit/test_training_readiness.py
```
(These 24 files are the complete `git status --short` `??` set, with the
`docs/reproducibility/` and `docs/training/` directories expanded to their individual
files. This audit file is item 6 in the list.)

Deleted (5, superseded thin scaffolds, via git rm):
```text
notebooks/MedNorm_DataEngine_and_Folds.ipynb
notebooks/MedNorm_ICD_RxNorm_Retrieval_Training.ipynb
notebooks/MedNorm_Mention_Assertion_Training.ipynb
notebooks/MedNorm_Qwen_LoRA_and_Calibration.ipynb
notebooks/MedNorm_Reranker_and_Resolver_Training.ipynb
```

## 27. Ignored generated artifacts

```text
data/derived/training_corpora/mednorm_vi_training_v1/**   (canonical_examples, splits, manifests, quality)
reports/training_readiness/**
```
All git-ignored; 0 tracked. No raw dataset payload, organizer text, corpus JSONL, or weights
is tracked.

## 28. Remaining blockers

- TEST_NAME/TEST_RESULT + assertion gold: only synthetic (smoke) so far.
- ICD/RxNorm supervised candidates: absent → ontology-alias self-supervision (S3, Colab).
- VietMed adapter: needs pyarrow → Colab CPU preprocess.
- All neural checkpoints: Colab S0–S6; full mode fails fast by design.

## 29. Final Git state

HEAD `3aa44fc` (unchanged). Working tree carries only the Audit 0017 changes in §26; the
architecture PDF is unchanged; no `CLAUDE.md`/`AGENTS.md`/`.claude`; no credentials.

## 30. Safe-to-commit decision

Audit 0017 is internally consistent (single authoritative values throughout), genuinely
complete for the local-only scope, and **safe to commit** as one milestone after human
review. No model was trained; no smoke notebook was run. First Colab action:
`MedNorm_Data_VietMed_Preprocess.ipynb`.
