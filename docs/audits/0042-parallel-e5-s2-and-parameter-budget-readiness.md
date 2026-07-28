# Audit 0042 - Parallel E5, S2, and Parameter-Budget Readiness

Date: 2026-07-28

## 1. Objective and Scope

Milestone:

```text
Parallel Post-E4 Readiness:
E5 XLM-R MRC
+ S2 Assertion
+ Candidate/Deployment Parameter Registry
```

The real E4 full-training run remains external to this repository work. This
milestone deliberately does **not** validate any real E4 full artifact, read E4
full metrics, capture E4 checkpoint hashes, generate E4 proposals, train learned
L4 v2, access `internal_test`, run organizer inference, train locally, or create
`output.zip`.

Status vocabulary:

- `IMPLEMENTED`: source/config/notebook contract exists.
- `PREFLIGHT_EXECUTED`: deterministic local contract tests executed.
- `READY_FOR_COLAB_SMOKE`: operator-facing Colab path/config exists, gated.
- `DATA_BLOCKED`: code exists but the governed corpus lacks required
  supervision.
- `NOT_TRAINED`: no smoke/full training success is claimed.
- `BLOCKED_ON_E4_FULL_ARTIFACT`: work must wait for the E4 full artifact gate.

## 2. Initial Git and Audit State

At milestone start:

```text
branch                         main
git status -sb                 ## main...origin/main
git status --porcelain         <clean>
HEAD                           4a55682 feat: add E4 training progress heartbeats and ETA
Audit 0041 committed           yes, at 4a55682
```

The stop condition did not trigger: Audit 0041 was already committed and the
working tree was clean before this milestone began.

Architecture and audit inputs read before implementation:

- `docs/MedNorm-VI_Architecture.pdf` in full; SHA-256
  `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`.
- `docs/audits/0035-multi-expert-mention-ensemble-and-learned-l4-v2.md`.
- `docs/audits/0036-phase2-training-readiness-and-artifact-validation.md`.
- `docs/audits/0040-e4-colab-io-and-memory-robustness.md`.
- `docs/audits/0041-e4-training-progress-observability.md`.

## 3. E4 Protection

No active E4 implementation or active runtime contract was modified.

Git-level protected-path check was empty for:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `configs/training/phase2_e4_phobert_w2ner_colab.yaml`
- `src/mednorm_vi/training/phase2/e4_runtime_io.py`
- `src/mednorm_vi/training/phase2/e4_progress.py`
- `src/mednorm_vi/training/phase2/e4_w2ner_training.py`
- `src/mednorm_vi/training/phase2/e4_alignment_diagnostic.py`
- `src/mednorm_vi/mention_factory/w2ner.py`
- `src/mednorm_vi/training/phase2/artifacts.py`
- `src/mednorm_vi/evaluation/exact_mention.py`
- `docs/MedNorm-VI_Architecture.pdf`
- governed corpus paths under `data/`

Pinned hashes at final verification:

```text
85040fbb824521582d5d04d7c45af293db7927b8a746a4d5b52558b32ce92813  notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb
054d376570eafe611cd1a2a35c3e24aaa30390a8a48f775a838d47b6500942ac  configs/training/phase2_e4_phobert_w2ner_colab.yaml
ad74031f9a909eb6d40eddeed1bb3f0eeb9183f015e3024d7d8c425df3b43e77  src/mednorm_vi/training/phase2/e4_runtime_io.py
2d770fa9c2c5551d4183d9d671c06d7965cda1ef104410de72d57adc29e95474  src/mednorm_vi/training/phase2/e4_progress.py
fc44befd49bd9a56a4efad49fff65ef17686bf9272689828f850e1d87cff48dd  src/mednorm_vi/training/phase2/e4_w2ner_training.py
0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b  docs/MedNorm-VI_Architecture.pdf
```

No Google Drive E4 artifact directory was touched. A local `local-artifacts/`
directory existed in the working tree during finalization; it is now ignored so a
future broad `git add` cannot stage local copies of generated Colab artifacts.
This milestone did not validate or read that artifact as an E4 result. The nine
post-E4 tasks in section 8 remain blocked.

## 4. Architecture and 9B Policy

The full architecture is intentionally preserved:

- all seven planned candidate training stages remain planned;
- candidates are not removed merely because the final deployment must fit 9B;
- model quality and complementarity will be measured after training;
- validation ablations and sparse leaderboard evidence will determine the final
  deployed subset;
- no external model API is permitted;
- final self-hosted inference must load no more than 9,000,000,000 parameters.

This milestone records two distinct inventories:

- **candidate inventory**: every model trained or evaluated during research. It
  may exceed 9B and is never gated.
- **deployment inventory**: only models loaded together by one final inference
  pipeline. The 9B gate applies here and fails closed.

The Architecture PDF states the organizer base-model interpretation where
adapters/heads are outside the base-model budget. This milestone applies the
stricter project rule requested by the operator: the gated deployment total is
the total parameters loaded by the process, adapters included. A LoRA deployment
therefore counts the base model plus adapters. Passing the stricter gate
necessarily satisfies the PDF base-only budget.

## 5. E5 XLM-R MRC Readiness

Status:

```text
IMPLEMENTED
PREFLIGHT_EXECUTED
READY_FOR_COLAB_SMOKE
NOT_TRAINED
```

Implemented/verified in:

- `src/mednorm_vi/training/phase2/e5_readiness.py`
- existing `src/mednorm_vi/training/phase2/e5_mrc_training.py`
- existing `src/mednorm_vi/mention_factory/mrc.py`
- `configs/training/phase2_e5_xlmr_mrc_ner_colab.yaml`
- `notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb`
- `tests/unit/test_parallel_post_e4_readiness.py`

Readiness coverage:

- governed train/validation loading by authoritative hashes;
- strict `internal_test` rejection;
- exact original character-offset preservation through MRC examples;
- versioned MRC query construction and query hash;
- context-only answer start/end targets;
- type-aware decoding and exact-span validation;
- deterministic example ordering;
- pinned revision policy for model and tokenizer in Colab;
- smoke/full configuration;
- bf16 / fp16 + GradScaler / CPU fp32 mixed-precision policy;
- real gradient accumulation and derived optimizer-step accounting;
- gradient clipping through tracked `max_grad_norm`;
- checkpoint/resume custody fields;
- governed validation-only best-checkpoint criterion;
- local-first artifact-persistence contract through Phase-2 infrastructure;
- existing E5 artifact validator;
- progress heartbeat/ETA settings aligned with Audit 0041.

Measured supervised type report:

```text
train:
  DIAGNOSIS   6,338
  MEDICATION  2,733
  SYMPTOM     2,649
  TEST_NAME       0
  TEST_RESULT     0

validation:
  DIAGNOSIS   1,302
  MEDICATION    156
  SYMPTOM       533
  TEST_NAME       0
  TEST_RESULT     0
```

The code supports all five architecture types, and queries are built for all
five. The training manifest/config reports only `DIAGNOSIS`, `MEDICATION`, and
`SYMPTOM` as supervised in the governed split. No supervised `TEST_NAME` or
`TEST_RESULT` laboratory labels are fabricated.

No XLM-R weights were downloaded and no E5 training was run locally.

## 6. S2 Assertion Readiness

Status:

```text
IMPLEMENTED
PREFLIGHT_EXECUTED
READY_FOR_COLAB_SMOKE
DATA_BLOCKED_ON_TRUSTED_ASSERTION_SUPERVISION
NOT_TRAINED
```

Implemented/verified in:

- `src/mednorm_vi/training/phase2/s2_assertion_training.py`
- `configs/training/phase2_s2_assertion_colab.yaml`
- `notebooks/MedNorm_S2_Assertion.ipynb`
- `tests/unit/test_parallel_post_e4_readiness.py`

Readiness coverage:

- mention-conditioned assertion examples;
- assertion eligibility restricted to `MEDICATION`, `DIAGNOSIS`, `SYMPTOM`;
- unsupported `TEST_NAME` and `TEST_RESULT` targets excluded by recorded policy;
- tri-state supervision: `supervised_true`, `supervised_false`, `unknown`;
- missing supervision is masked, never converted to false;
- trusted and weak/synthetic provenance channels are separated;
- source document ID, mention span, entity type, source dataset, and leakage group
  preserved;
- document-level train/validation leakage guard;
- deterministic class and label ordering;
- class/source coverage and balance reporting;
- exact multi-label metric over supervised positions only;
- strict `internal_test` rejection;
- smoke/full Colab config and notebook;
- S2 checkpoint payload schema and read-only artifact validator;
- progress/ETA config fields aligned with Audit 0041;
- no uncontrolled heuristic labels reported as gold.

Measured governed assertion supervision:

```text
train rows                  33,826
train eligible mentions     11,720
train supervised mentions        0
train trusted examples           0

validation rows              1,045
validation eligible mentions 1,991
validation supervised mentions   0
validation trusted examples      0

coverage manifest: assertions=false for phoner_covid19, vimedner, vimq, vietmed_ner
```

Every eligible governed mention is `unknown` for all three assertion dimensions.
`assert_trainable()` raises on this corpus. This is intentional: training S2 on
the current governed corpus would require fabricating assertion labels. No such
fabrication is implemented or claimed.

The S2 head is buildable and counted programmatically:

```text
ViHealthBERT shared base       134,998,272
S2 task heads                        9,228
```

The head is four 3-label linear heads over the 768-hidden shared backbone: cue,
scope, relation, and final assertion logits.

No S2 model was trained locally.

## 7. Parameter Registry and Deployment Gate

Implemented in:

- `configs/models/candidate_model_registry.yaml`
- `configs/models/deployment_budget_template.yaml`
- `src/mednorm_vi/governance/parameter_budget.py`
- `scripts/report_parameter_budget.py`
- `tests/unit/test_parallel_post_e4_readiness.py`

Candidate records cover:

- E3 S1 baseline;
- E4 PhoBERT-W2NER;
- E5 XLM-R MRC;
- L4 v2 resolver;
- S2 assertion model;
- S3 retrieval model(s);
- S4 reranker model(s);
- S5 Qwen LoRA;
- S6 calibration parameters;
- E6 GLiNER and E7 Qwen proposer as architecture-preserving candidate entries.

Concrete verified counts:

```text
e3_vihealthbert_span_type     134,998,272  counted_from_config
l4_learned_resolver_v2             6,542  counted_from_config
s2_assertion_head             134,998,272  shared ViHealthBERT base
s2_assertion_head adapter           9,228  counted from build_s2_assertion_head()
```

Unverified/planned entries:

```text
e4_phobert_w2ner                 IMPLEMENTED, unverified count
e5_xlmr_mrc_ner                  IMPLEMENTED, unverified count
s3_retrieval_dense               PLANNED
s3_retrieval_ontology_encoder    PLANNED
s4_reranker                      PLANNED
s5_qwen_lora_critic              PLANNED
s6_calibration_meta_model        PLANNED
e6_gliner_open_type              PLANNED
e7_qwen_proposer                 PLANNED
```

No PLANNED entry carries a guessed model ID or guessed parameter count. The PDF's
approximately 8.852B planning total is not hardcoded as fact.

Current candidate inventory summary:

```text
component_count                 12
components_with_counts           3
verified_components              3
candidate_total_parameters       270,012,314
candidate_total_exceeds_9b       false
gate_applies_to_candidates       false
```

Current deployment template:

```text
selected components:
  e3_vihealthbert_span_type
  l4_learned_resolver_v2

total loaded (gated)        135,004,814
remaining margin          8,864,995,186
within budget                      true
```

Tests demonstrate:

- candidate total may exceed 9B without failure;
- deployment total <= 9B passes;
- deployment total > 9B fails;
- unknown selected count fails closed;
- shared base loaded once is counted once;
- independent checkpoints are both counted;
- LoRA counts base plus adapter;
- calibration parameters are included even if small.

The gate is enforced only against a selected deployment manifest, never the
complete candidate-training inventory.

## 8. Post-E4 Blocked Tasks

Implemented in:

- `src/mednorm_vi/governance/post_e4_gates.py`
- tests in `tests/unit/test_parallel_post_e4_readiness.py`

Exact post-E4 gate:

```text
E4 full_training_complete
artifact_validator_ok = true
persistent best.pt and latest.pt verified
validation metrics available
internal_test_accessed = false
```

Until that gate opens, all nine tasks remain:

```text
BLOCKED_ON_E4_FULL_ARTIFACT
```

Blocked tasks:

1. validate the real E4 full artifact;
2. capture best/latest checkpoint hashes;
3. read final E4 validation metrics;
4. compare E3 versus E4;
5. evaluate E3 + E4 proposal union;
6. generate frozen E4 proposals;
7. train learned L4 v2;
8. access `internal_test`;
9. submit leaderboard predictions.

Frozen E4 proposals for L4 v2 are explicitly refused until the gate opens.

## 9. Leaderboard and Selection Policy

Documented in:

- `docs/decisions/0001-candidate-training-and-deployment-selection-policy.md`

Policy:

1. train all planned candidate components;
2. validate each component independently;
3. perform base/add-one/remove-one/complementary ensemble ablations;
4. compute score contribution, parameter contribution, latency, and memory;
5. shortlist only meaningful configurations;
6. use leaderboard submissions sparingly;
7. select a final self-hosted configuration with total loaded parameters <= 9B
   and no external API.

A component omitted from final deployment remains a researched candidate and is
recorded as excluded by ablation, not deleted from architecture history.

## 10. Hygiene

No prohibited files were added:

```text
tracked *.pt/*.pth/*.ckpt/*.safetensors/*.bin/*.zip        0
tracked artifacts|weights|caches|checkpoint|.claude paths  0
CLAUDE.md / AGENTS.md tracked                              0
output.zip                                                 absent
```

`.gitignore` now also excludes `local-artifacts/` so operator-placed local copies
of Colab outputs cannot be staged accidentally. This is a hygiene rule only; it
does not validate, move, inspect, or rewrite any E4 artifact.

## 11. Tests and Static Checks

Focused checks executed during the milestone:

```text
env PYTHONPATH=src python -m pytest -q tests/unit/test_parallel_post_e4_readiness.py
64 passed

env PYTHONPATH=src python -m pytest -q \
  tests/unit/test_notebooks.py::test_stage_notebook_has_required_sections \
  tests/unit/test_parallel_post_e4_readiness.py
71 passed

ruff check focused changed Python paths
All checks passed!
```

Final required checks after Audit 0042 was written:

```text
env PYTHONPATH=src python -m pytest -q
1596 passed, 1 skipped in 134.87s
SKIPPED: tests/unit/test_vietmed_adapter.py:399 because pyarrow is not installed locally

ruff check .
All checks passed!

ruff check notebooks
All checks passed!

env PYTHONPATH=src python -m mypy
Success: no issues found in 264 source files

env PYTHONPATH=src python -m compileall -q src
<passed>

git diff --check
<passed>
```

## 12. Changed Files

Added:

- `configs/models/candidate_model_registry.yaml`
- `configs/models/deployment_budget_template.yaml`
- `configs/training/phase2_s2_assertion_colab.yaml`
- `docs/audits/0042-parallel-e5-s2-and-parameter-budget-readiness.md`
- `docs/decisions/0001-candidate-training-and-deployment-selection-policy.md`
- `scripts/report_parameter_budget.py`
- `src/mednorm_vi/governance/__init__.py`
- `src/mednorm_vi/governance/parameter_budget.py`
- `src/mednorm_vi/governance/post_e4_gates.py`
- `src/mednorm_vi/training/phase2/e5_readiness.py`
- `src/mednorm_vi/training/phase2/s2_assertion_training.py`
- `tests/unit/test_parallel_post_e4_readiness.py`

Modified:

- `.gitignore`
- `configs/training/phase2_e5_xlmr_mrc_ner_colab.yaml`
- `docs/audits/README.md`
- `notebooks/MedNorm_S2_Assertion.ipynb`

No active E4 path was modified.

## 13. Non-execution Guarantees

This milestone did not:

- train locally;
- download model weights;
- run `internal_test`;
- run organizer inference;
- create `output.zip`;
- commit or push;
- validate a real E4 full artifact;
- read final E4 validation metrics;
- generate frozen E4 proposals.

## 14. Remaining Colab Work

E5:

- resolve immutable XLM-R model/tokenizer revisions in Colab;
- execute bounded smoke only;
- validate smoke artifact;
- only then authorize full training later if smoke passes.

S2:

- acquire or approve a controlled assertion-supervised corpus;
- keep trusted and weak/synthetic supervision separate;
- rerun governed S2 preflight;
- execute smoke only after trusted supervision exists;
- validate S2 artifact with `validate_s2_artifact`.

E4/post-E4:

- wait for the real E4 full run to finish;
- validate the full artifact;
- record best/latest checkpoint hashes and validation metrics;
- only then generate frozen E4 proposals for learned L4 v2.

## 15. Safe-to-Commit Verdict

Safe to commit after review. Required tests and static checks pass, no active E4
path changed, no governed corpus file changed, and no model/checkpoint/cache/ZIP
artifact is tracked. Do not use `git add -A`; add only the listed files. Do not
add ignored runtime artifacts, weights, checkpoints, caches, generated outputs,
or `local-artifacts/`.
