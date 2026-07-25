# Audit 0020 - VietMed-NER Real Artifact Intake and Governed Corpus Rebuild

- **Date:** 2026-07-25
- **Author:** Codex (AI agent), for human review
- **Change type:** Completed vertical-slice intake, governed-corpus rebuild, and split
  verification. **No commit. No push. No model training, model/tokenizer download, GPU
  workload, external API, organizer inference, S1 smoke run, or `output.zip`.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 was read in full before artifact
  decisions. The architecture PDF was not modified, moved, regenerated, or replaced.

## 1. Objective and scope

This milestone completes the Audit 0020 VietMed-NER artifact intake:

- validate the returned Colab-generated artifact without printing clinical text;
- record the human-approved resolution of the initial review-required findings;
- place the artifact at the canonical ignored path;
- harden future manifests so newly generated artifacts must persist `run_mode`;
- rebuild the governed corpus with VietMed included;
- verify deterministic rebuilds, duplicate-family leakage, and train-only approximate
  mapping policy.

## 2. Audit number decision and preflight

Audit 0019 is committed and tracked at HEAD:

```text
3a065d4 review and fix bug in notebooks
```

Audit 0020 was already present and uncommitted as the fail-fast record. This milestone
continues **Audit 0020** and does not create Audit 0021. Preflight state before continuing:

```text
 M docs/audits/README.md
?? docs/audits/0020-vietmed-real-artifact-intake-and-corpus-rebuild.md
?? vietmed_ner_v1/
```

`git diff --check` was clean. No unrelated work was present.

## 3. Initial fail-fast discovery preserved

Read-only discovery originally found two review-required conditions:

1. The incoming root was nested as `vietmed_ner_v1/vietmed_ner_v1/...`.
2. The generated preprocessing manifest lacked an explicit `run_mode` field.

No artifact was moved during that initial discovery. Hashes, counts, JSONL streaming
validation, and audio-exclusion evidence were recorded before any placement.

## 4. Human resolution and REAL-equivalent acceptance

The human reviewer resolved both findings:

- the outer `./vietmed_ner_v1/` directory was only a transport/copy wrapper;
- the inner `./vietmed_ner_v1/vietmed_ner_v1/` directory was the intended artifact root;
- the missing `run_mode` field is a metadata omission in the older notebook writer, not
  evidence of synthetic execution;
- this current artifact is accepted as REAL based on equivalent provenance proof;
- the exception applies only to this artifact;
- future artifacts must contain `run_mode: REAL`.

The original generated manifest was **not** edited post hoc and still does not contain
`run_mode`. Future generated manifests now record `run_mode`.

Equivalent REAL evidence:

```text
source Parquet files        3 concrete filenames with SHA-256 hashes
canonical examples          9,267
mapped entities             2,114
repo_commit                 3a065d459237da52f1cd733b2c84048eb879a96f
creation_notebook           notebooks/MedNorm_Data_VietMed_Preprocess.ipynb
resolved columns            words + BIO-string labels
adapter_version             vietmed-adapter-v1
canonical JSONL hash        matches manifest
offset_invalid              0
human_review_required       0
audio_excluded              true
synthetic-only provenance   absent
audio/binary payloads       absent
```

## 5. Incoming and canonical artifact tree

Human-approved source before placement:

```text
./vietmed_ner_v1/vietmed_ner_v1/
```

Canonical destination after placement:

```text
data/derived/training_corpora/vietmed_ner_v1/
```

Final artifact tree:

```text
data/derived/training_corpora/vietmed_ner_v1/
  canonical_examples/
    vietmed_ner_examples.jsonl
  manifests/
    vietmed_ner_preprocessing_manifest.json
    vietmed_ner_source_hashes.json
  quality/
    vietmed_ner_label_inventory.json
    vietmed_ner_offset_validation.json
    vietmed_ner_repair_exclusion_summary.json
    vietmed_ner_split_summary.json
```

Regular files: 7. Symlinks: 0. Hidden/temp/wrapper files: 0. Raw Parquet: 0. Audio files:
0. Unexpected binary files: 0.

## 6. Artifact hashes preserved

| Relative path | Bytes | MD5 | SHA-256 |
| --- | ---: | --- | --- |
| `canonical_examples/vietmed_ner_examples.jsonl` | 5,213,791 | `6b9f2e3d5fa31c1f432cd7356675f8d6` | `5bec7ac41dbaed165ced95c617d3d98f974652a77f160431c48b96adc48dc0fd` |
| `manifests/vietmed_ner_preprocessing_manifest.json` | 1,285 | `33468fbd23c4fa5417327127ccd96153` | `210ac58af74e2147a2b76e9764fbba05d9af306e2e172959d604940fdccaaba4` |
| `manifests/vietmed_ner_source_hashes.json` | 312 | `60123b6440e45069ccbb085d40ea1f84` | `19a68251500420e74f3278dd5aa77a9f69e9f14fe67f6f3c886766ce31b8883b` |
| `quality/vietmed_ner_label_inventory.json` | 150 | `d1750c48862eb3c80b61a3391d571f91` | `78319a6499dfd538276b7929e0e914dbd3355084e5a51c30d27182123f9a8e70` |
| `quality/vietmed_ner_offset_validation.json` | 37 | `405a69d16bff87c910db3230bfbece2b` | `de3a5f9f7e5bc49d4323aed3dab47dfe6ae2a6b0076cd9f98c21efa5d4832a10` |
| `quality/vietmed_ner_repair_exclusion_summary.json` | 143 | `83e58bc5cbbd3ca7b78cfee01850f19f` | `755d1a37754f59fd7c27359089ec5ab671555cb592b6f664d122cb60fa8ddd22` |
| `quality/vietmed_ner_split_summary.json` | 93 | `301183182999a846b3570d2c9391cdd8` | `461569f996e4f2de583254a925dbec08192d18efd6256297ca484ff63b3e25ce` |

No duplicate file content was found. The post-move hashes match the original read-only
discovery hashes exactly.

Source Parquet hashes recorded in the artifact:

```text
test-00000-of-00001.parquet        fd9c5e5778ded8d01512e1acf19c4e75a788c861d43299683e16d13a2b05228a
train-00000-of-00001.parquet       5d04692de75257c5392ad7a1b4cc4fbdd078f4b454719dbae46b89d284452a3c
validation-00000-of-00001.parquet  bf1caebca2d49d6f2fb91bfb7297d32b083be87a31fb64cadf05b6eadcbd80ed
```

## 7. Post-move artifact validation

The canonical artifact validates under the explicit Audit 0020 legacy approval path:

```text
effective run_mode              REAL
manifest has run_mode field     false
schema_version                  canonical_example_v1
adapter_version                 vietmed-adapter-v1
mapping hash matches tracked    true
JSONL SHA-256 matches manifest  true
examples                        9,267
entities                        2,114
source splits                   train=4,616; validation=1,154; test=3,497
original labels                 DRUGCHEMICAL=2,114
target types                    MEDICATION=2,114
mapping statuses                MAP_APPROXIMATE=2,114
invalid JSONL lines             0
duplicate example IDs           0
empty example IDs               0
empty text records              0
offset-invalid                  0
human_review_required           0
repaired                        3
excluded                        0
unsupported types/statuses      0
DRUGCHEMICAL mapping violations 0
fabricated assertions/candidates 0
```

Audio/privacy scan:

```text
audio keys                      0
audio-like serialized content   0
binary-like payloads            0
private-path strings            0
```

VietMed artifact loader:

```text
load_vietmed_artifacts(..., allow_human_approved_legacy_real=True, require_real=True)
examples=9,267 entities=2,114
```

## 8. Placement and git-ignore safety

Destination pre-state: **ABSENT**.

Move performed:

```text
mkdir -p data/derived/training_corpora
mv vietmed_ner_v1/vietmed_ner_v1 data/derived/training_corpora/vietmed_ner_v1
rmdir vietmed_ner_v1
```

Post-move checks:

```text
./vietmed_ner_v1/                                      absent
data/derived/training_corpora/.vietmed_intake_staging/ absent
git ls-files vietmed_ner_v1                            no files
git ls-files data/derived/training_corpora/vietmed_ner_v1 no files
```

Representative ignore checks:

```text
.gitignore:24:data/** data/derived/training_corpora/vietmed_ner_v1/canonical_examples/vietmed_ner_examples.jsonl
.gitignore:24:data/** data/derived/training_corpora/vietmed_ner_v1/manifests/vietmed_ner_preprocessing_manifest.json
```

No generated JSONL, generated manifests, quality reports, raw Parquet, or audio payloads
were staged.

## 9. Future run-mode contract hardening

Tracked runtime contract changes:

- `write_artifacts()` now requires a keyword-only `run_mode` argument.
- Valid values are exactly `REAL` and `SYNTHETIC_SMOKE`.
- Missing/invalid run mode fails for newly generated artifacts.
- The VietMed notebook passes `run_mode=RUN_MODE` into `write_artifacts()`.
- The loader validates run mode and can require REAL for governed-corpus inclusion.
- The governed-corpus builder rejects `SYNTHETIC_SMOKE` artifacts.
- The Audit 0020 missing-field artifact is accepted only through a tracked exact-match,
  opt-in `allow_human_approved_legacy_real=True` path bound to
  `configs/resources/legacy_intake/vietmed_ner_audit0020_real_equivalent.json`.
- That approval record fixes the canonical JSONL SHA-256, preprocessing-manifest SHA-256,
  source-hashes JSON SHA-256, adapter version, mapping hash, repository commit, source
  Parquet hashes, example count, and entity count.

Tests cover REAL, SYNTHETIC_SMOKE, missing, invalid, builder-level smoke rejection, exact
legacy acceptance, changed JSONL hash rejection, changed manifest hash rejection, changed
source-hashes JSON rejection, changed source Parquet hash rejection, changed repository
commit rejection, and random missing-`run_mode` rejection.

## 10. Governed corpus rebuild

Command:

```bash
env PYTHONPATH=src python3 -m mednorm_vi.data_engine.cli build-governed-corpus
```

Result:

```text
vietmed_status                 included_from_artifacts
total examples                 35,916
total entities                 15,757
offset_invalid                 0
cross_split_family_leakage     0
eval_map_approximate_entities  0
duplicate families             35,753
near-duplicate candidates      217
```

Source contributions:

| Source | Examples | Entities |
| --- | ---: | ---: |
| `phoner_covid19` | 10,027 | 0 |
| `vimedner` | 7,622 | 12,738 |
| `vimq` | 9,000 | 905 |
| `vietmed_ner` | 9,267 | 2,114 |

Target-type distribution:

```text
DIAGNOSIS   8,987
SYMPTOM     3,751
MEDICATION  3,019
```

Before/after corpus counts:

| Metric | Before VietMed | After Audit 0020 |
| --- | ---: | ---: |
| examples | 26,649 | 35,916 |
| entities | 13,643 | 15,757 |
| MEDICATION entities | 905 | 3,019 |
| train examples | 24,559 | 33,826 |
| validation examples | 1,045 | 1,045 |
| internal_test examples | 1,045 | 1,045 |

Audit 0020 did not redistribute evaluation splits. Validation and internal_test were
already 1,045 examples each in the final Audit 0017 split-policy-v2 state immediately
before VietMed. All 9,267 VietMed examples were added to train only, increasing train
from 24,559 to 33,826. The validation and internal_test output hashes remained unchanged.

## 11. Split counts and hashes

| Split | Examples | Entities | SHA-256 |
| --- | ---: | ---: | --- |
| train | 33,826 | 11,720 | `892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a` |
| validation | 1,045 | 1,991 | `ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103` |
| internal_test | 1,045 | 2,046 | `e23acde0d87154d1750d25d1e47c92c86e457e42f2f97cb985661253dea7e135` |

Additional output hashes:

```text
canonical examples JSONL      9e8060775d97713485c964df9165f640050a0085f385b2a9ff216c75f9d81588
corpus manifest               a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3
split manifest                ceb962b54c8d74eb89a016571a350987d1bfd226d883f871ad057b9997c53778
source contributions          9eb82426b24d3f71c61b59674e88f015085ba5797e7c0c210ac39f4131ab1fb8
annotation coverage           b33614a1bf1f2216530a85acfbf61ac66bc9de0e8ccecb59d7382b46067f59f9
leakage summary               aada8b15fc89b838533ee7650b9bfffc5cfb54c6762d0eb2468b4382a148b558
offset validation             17f581b7f49f160288ca709b2993643d02781fb95068f2bd1727894509d2193f
label distribution            3e5ed4ccf98281712e17700f65aa2979f92a46f4c595687fc56ef6f97a6dc61c
```

## 12. Deterministic double rebuild

Two fresh builds under `/tmp` produced byte-identical outputs for:

```text
canonical_examples/public_ner_examples.jsonl
manifests/corpus_manifest.json
manifests/split_manifest.json
manifests/source_contributions.json
manifests/annotation_coverage.json
quality/leakage_summary.json
quality/offset_validation.json
quality/label_distribution.json
splits/train.jsonl
splits/validation.jsonl
splits/internal_test.jsonl
```

All file hashes matched the final default corpus output. The Python return objects differed
only by `out_dir` because the temporary directories were different; the persisted artifacts
were byte-identical.

## 13. VietMed train-only policy

Streaming split validation produced:

```text
VietMed examples by split       train=9,267; validation=0; internal_test=0
VietMed entities by split       train=2,114; validation=0; internal_test=0
MAP_APPROXIMATE by split        train=2,114; validation=0; internal_test=0
non-MAP_EXACT eval entities     validation=0; internal_test=0
wrong VietMed mappings          0
organizer/synthetic source rows 0
```

Every VietMed mapped entity remains:

```text
source_label    DRUGCHEMICAL
target_type     MEDICATION
mapping_status  MAP_APPROXIMATE
```

VietMed loss masks are span/type only:

```text
span=true, entity_type=true, assertions=false, icd_candidates=false, rxnorm_candidates=false
```

## 14. Validation commands

```text
focused VietMed adapter tests -> 37 passed, 1 skipped (pyarrow not installed locally)
env PYTHONPATH=src python3 -m pytest -q -> 638 passed, 1 skipped (pyarrow not installed locally)
ruff check . -> All checks passed
env PYTHONPATH=src python3 -m mypy -> Success, 219 source files
env PYTHONPATH=src python3 -m compileall -q src -> clean
git diff --check -> clean
public-NER manifest validators -> phoner_covid19, vimedner, vimq, vietmed_ner OK;
one existing license-status warning per manifest
corpus governance sanity -> included, split-policy-v2, leakage 0, eval approximate 0, offset invalid 0
model budget validation -> full profile within_9b=true, base_parameters=8,335,616,512;
adapter_parameters=73,000,000; total_parameters=8,408,616,512
phase1c doctor -> green, network access none
```

Additional budget note:

```text
env PYTHONPATH=src python3 -m mednorm_vi.model_registry.cli --profile full --require-local-paths
```

returned within-9B but exit code 2 because the restricted local model checkpoints are absent.
No checkpoint download was attempted; this is expected before S1 smoke/full training artifacts
exist and is not a parameter-budget failure.

Full-profile parameter breakdown:

| Model | Base parameters | Adapter parameters |
| --- | ---: | ---: |
| `deterministic_experts` | 0 | 0 |
| `vihealthbert_span_type` | 135,000,000 | 2,000,000 |
| `phobert_w2ner` | 135,000,000 | 3,000,000 |
| `xlmr_mrc_ner` | 270,000,000 | 3,000,000 |
| `gliner_small` | 180,000,000 | 1,000,000 |
| `qwen2_5_7b_lora` | 7,615,616,512 | 64,000,000 |

Budget/profile hashes:

```text
configs/model_registry/models_v1.yaml   6d5a59c11fa53b5b12296fdf6ab5516cf909e6836387278012e0ef67ba2a0939
configs/parameter_budget.yaml           4bb986ae3d68ed81fa0628865834d6f37580547e7a8e6065ba3c3c65b399a5df
```

`git diff -- configs/model_registry configs/parameter_budget.yaml` produced no output, so
Audit 0020 made no model-registry or parameter-budget config changes. Audit 0017's
8,408,616,512 value is the full total including adapters. The stale Audit 0020 value
8,335,616,512 was only the base-parameter subtotal and is corrected here.

## 15. Files created

```text
docs/audits/0020-vietmed-real-artifact-intake-and-corpus-rebuild.md
configs/resources/legacy_intake/vietmed_ner_audit0020_real_equivalent.json
```

## 16. Files modified

Source:

```text
src/mednorm_vi/data_engine/__init__.py
src/mednorm_vi/data_engine/corpus_build.py
src/mednorm_vi/data_engine/vietmed_ner.py
```

Tests:

```text
tests/unit/test_notebook_execution.py
tests/unit/test_notebook_placeholders.py
tests/unit/test_notebooks.py
tests/unit/test_vietmed_adapter.py
```

Notebook:

```text
notebooks/MedNorm_Data_VietMed_Preprocess.ipynb
```

Documentation:

```text
docs/audits/README.md
docs/training/governed_corpus_v1.md
docs/training/first_colab_run.md
docs/notebooks/notebook_execution_integrity.md
```

Ignored payloads/outputs:

```text
data/derived/training_corpora/vietmed_ner_v1/
data/derived/training_corpora/mednorm_vi_training_v1/
```

## 17. Design decisions and invariants

- The current missing-`run_mode` artifact is accepted only by human-approved exact
  provenance, preserving original bytes.
- New artifacts must explicitly record `run_mode`.
- Governed-corpus inclusion requires REAL mode, or this one exact legacy REAL-equivalent
  artifact.
- `SYNTHETIC_SMOKE` artifacts may be used for notebook/adapter tests but are rejected for
  governed-corpus inclusion.
- VietMed `MAP_APPROXIMATE` entities are train-only; validation/internal_test remain
  MAP_EXACT-only.
- Organizer input and synthetic fixtures are not labeled training/evaluation sources.

## 18. Deferred items and residual risks

- TEST_NAME and TEST_RESULT still have no governed real supervision.
- VietMed medication labels are approximate and improve train coverage, not evaluation gold.
- Full local checkpoint availability remains intentionally absent; model weights were not
  downloaded.
- Future Colab preprocessing artifacts must include `run_mode: REAL`; missing `run_mode`
  should not be accepted again without a new explicit human audit decision.

## 19. Confirmations

- No models, neural tokenizers, datasets, or KB indices were downloaded.
- No GPU workload, model training, fine-tuning, organizer inference, S1 smoke, or
  `output.zip` generation was run.
- No raw clinical text or audio payload appears in this audit.
- The architecture PDF was read in full and not modified.

## 20. S1 smoke readiness

S1 smoke is **ALLOWED by the Audit 0020 data/corpus gate**:

```text
REAL-equivalent provenance accepted      yes
canonical placement complete             yes
seven artifact files validated           yes
original hashes preserved                yes
offset_invalid                           0
human_review_required                    0
vietmed_status                           included_from_artifacts
cross_split_family_leakage               0
eval_map_approximate_entities            0
VietMed mapped entities train-only       yes
deterministic double rebuild             byte-identical
tests/static checks                      pass
```

S1 smoke was not run in this task. The next Colab notebook remains:

```text
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
```

## 21. Git status snapshot

Expected final status before human commit:

```text
 M docs/audits/README.md
 M docs/notebooks/notebook_execution_integrity.md
 M docs/training/first_colab_run.md
 M docs/training/governed_corpus_v1.md
 M notebooks/MedNorm_Data_VietMed_Preprocess.ipynb
 M src/mednorm_vi/data_engine/__init__.py
 M src/mednorm_vi/data_engine/corpus_build.py
 M src/mednorm_vi/data_engine/vietmed_ner.py
 M tests/unit/test_notebook_execution.py
 M tests/unit/test_notebook_placeholders.py
 M tests/unit/test_notebooks.py
 M tests/unit/test_vietmed_adapter.py
?? configs/resources/legacy_intake/
?? docs/audits/0020-vietmed-real-artifact-intake-and-corpus-rebuild.md
```

Safe to commit after human review: **yes**. Do not commit automatically.
