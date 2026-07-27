# Audit 0036 - Phase 2 Training Readiness and Artifact Validation

Date: 2026-07-27

## 1. Objective and Scope

Milestone:

`Phase 2 Training Readiness Closure + Colab Execution Package + Artifact Validators + Frozen Validation Ablation Path`

This audit records the closure work that turns Audit 0035's Phase-2 L3/L4
implementation contracts into operator-runnable Colab execution packages and
read-only validators.

This milestone does not train E4, E5, or learned L4 v2 locally. It does not
start S2 assertion, S3 retrieval, S4 reranking, S5 Qwen LoRA, or S6
calibration. It does not run internal_test, organizer inference, or packaging.

Status vocabulary used here:

- `IMPLEMENTED`: source/notebook/config contract exists.
- `PREFLIGHT_EXECUTED`: deterministic local unit/static checks executed.
- `READY_FOR_COLAB_SMOKE`: an operator can run the smoke path in Colab.
- `SMOKE_EXECUTED`: real smoke output exists and was validated.
- `READY_FOR_FULL_TRAINING`: full path is gated and runnable after smoke/artifact checks.
- `FULLY_TRAINED`: real full-training output exists and was validated.
- `ARTIFACT_VALIDATED`: read-only validator passed on a real artifact.
- `VALIDATION_EVALUATED`: governed validation ablation was executed on frozen artifacts.
- `UNAVAILABLE_UNTRAINED`: valid trained checkpoint/artifact is not present.

Execution states not reached in this milestone are not claimed.

## 2. Initial Git and Audit State

Precondition commands were run before new work began:

```text
pwd
/mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI

git branch --show-current
main

git status -sb
## main...origin/main

git status --short
<clean>

git log --oneline -8
c6107a4 feat: add phase2 mention ensemble and learned l4 v2 contracts
23065f9 feat: implement L3 span lattice and L4 resolver ablation
a00af3c feat: add exact character-offset S1 mention evaluator
15d0b07 feat: validate S1 checkpoint and add local held-out evaluation
36c520d fix: protect S1 astral characters and finalize preflight
9e01cbd fix: add full-corpus S1 alignment preflight
cbda043 fix: support Vietnamese tone-placement alignment
aa9ef73 fix: unify S1 character alignment and segmentation

git diff --check
<clean>
```

Audit 0035 and its Phase-2 implementation were already committed at `c6107a4`.
The milestone precondition passed; no Audit 0035 work was mixed into this
milestone.

Architecture PDF read status: `docs/MedNorm-VI_Architecture.pdf` was read in
full before code changes. Audits 0033, 0034, and 0035 were read in full, and
the preceding audit history was reviewed for corpus, checkpoint, notebook,
artifact, budget, and held-out evaluation constraints.

## 3. Architecture Sections Applied

Applied architecture contracts:

- Section 4 offset invariant: every proposal/mention/artifact surface preserves
  exact end-exclusive original offsets.
- Section 5 case routing and C7 duplicate/overlap policy: proposal datasets keep
  coordinate identity and group leakage controls.
- Section 6 L3 expert ensemble: E4/E5/E6/E7 remain proposal sources, never final
  organizer JSON.
- Section 7 learned L4 resolver: boundary action, type action, wrong-type risk,
  abstention, IoU auxiliary signal, WER boundary surrogate interface, and exact
  boundary action constraints are explicit.
- Sections 15 and 16 training/data flow: governed train only for training,
  governed validation for model selection and ablation, internal_test frozen.
- Section 17 parameter budget: Phase-2 validation profiles validate shared
  backbones, immutable revisions, checkpoint hashes, and local paths.
- Sections 18 and 19 ablation and module contracts: unavailable experts are
  reported as `UNAVAILABLE_UNTRAINED`.

## 4. Notebook Completeness Before Changes

The three Audit 0035 notebooks were inspected before modification:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb`
- `notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb`

Finding: they were honest implementation scaffolds. They contained Drive paths,
governed hash gates, smoke/full boolean guards, and simple manifest writes, but
they did not contain a complete epoch training loop, six-file artifact custody,
checkpoint reload validation against a reusable validator, frozen proposal
generation, validation-only ablation, or internal_test freeze gate.

## 5. E4 PhoBERT W2NER Training Pipeline

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, `READY_FOR_COLAB_SMOKE`,
`READY_FOR_FULL_TRAINING`, `UNAVAILABLE_UNTRAINED`.

Implemented source:

- `src/mednorm_vi/training/phase2/e4_w2ner_training.py`

Implemented behavior:

- governed W2NER data-to-loss-to-decode contract;
- padded relation-grid mask contract;
- pure numeric relation cross-entropy helper for local contract tests;
- deterministic argmax decode back through the existing W2NER decoder;
- smoke/full incompatibility rejection;
- resolved config and six-file manifest helpers;
- checkpoint payload helper with schema fields used by the artifact validator.

Notebook:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`

Artifact paths:

- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_full_v1`

Full authorization string:

```text
I_AUTHORIZE_E4_FULL_TRAINING
```

No PhoBERT training or model acquisition was executed locally.

## 6. E5 XLM-R MRC-NER Training Pipeline

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, `READY_FOR_COLAB_SMOKE`,
`READY_FOR_FULL_TRAINING`, `UNAVAILABLE_UNTRAINED`.

Implemented source:

- `src/mednorm_vi/training/phase2/e5_mrc_training.py`

Implemented behavior:

- five versioned organizer queries via `mrc-type-queries-v1`;
- query hash manifest field;
- negative query example enforcement;
- query/context token boundary checks;
- context-only start/end labels;
- pure numeric start/end loss helper;
- deterministic decode through the existing MRC pairing constraints;
- smoke/full incompatibility rejection;
- checkpoint and manifest helpers.

Notebook:

- `notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb`

Artifact paths:

- `/content/drive/MyDrive/MedNorm-VI/artifacts/e5_xlmr_mrc_smoke_v1`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e5_xlmr_mrc_full_v1`

Full authorization string:

```text
I_AUTHORIZE_E5_FULL_TRAINING
```

No XLM-R training or model acquisition was executed locally.

## 7. Frozen Proposal Generation

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, `READY_FOR_COLAB_SMOKE`.

Implemented source:

- `src/mednorm_vi/training/phase2/proposal_generation.py`

Implemented behavior:

- train/validation-only proposal dataset construction;
- internal_test split rejection;
- exact-offset validation via the existing `SpanLattice`;
- privacy-safe source group IDs;
- deterministic proposal JSONL hashing;
- per-source provenance, model revision, config hash, checkpoint hash, route,
  section, boundary alternatives, grammar/laboratory features, and expert
  agreement fields;
- unavailable expert reporting as `UNAVAILABLE_UNTRAINED`;
- manifest loader and JSONL writer.

Configured outputs:

- `/content/drive/MyDrive/MedNorm-VI/artifacts/phase2_frozen_proposals_train_v1`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/phase2_frozen_proposals_validation_v1`

No frozen governed proposal dataset was generated locally.

## 8. Learned L4 v2 Training Pipeline

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, `READY_FOR_COLAB_SMOKE`,
`READY_FOR_FULL_TRAINING`, `UNAVAILABLE_UNTRAINED`.

Implemented source:

- `src/mednorm_vi/training/phase2/l4_training.py`

Implemented behavior:

- compact tracked MLP architecture contract;
- stable feature order;
- boundary action output for `KEEP`, `SELECT_EXISTING`, `OFFSET`, `DROP`;
- type action output for five organizer types plus `DROP`;
- wrong-type risk, IoU auxiliary score, and bounded offset outputs;
- explicit loss terms: boundary action, type, wrong-type, token IoU auxiliary,
  WER boundary surrogate, and configured loss weights;
- exact-offset boundary output validation;
- checkpoint and six-file manifest helpers.

Notebook:

- `notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb`

Artifact paths:

- `/content/drive/MyDrive/MedNorm-VI/artifacts/l4_learned_resolver_v2_smoke_v1`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/l4_learned_resolver_v2_full_v1`

Full authorization string:

```text
I_AUTHORIZE_L4_V2_FULL_TRAINING
```

No learned L4 smoke/full training was executed locally.

## 9. Artifact Validators

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`.

Implemented source:

- `src/mednorm_vi/training/phase2/artifacts.py`
- `src/mednorm_vi/training/phase2/cli.py`

Validated artifact contract:

- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- `logs/training_history.jsonl`
- `resolved_config.json`
- `validation_metrics.json`
- `training_manifest.json`

Validator checks:

- stage/expert identity;
- mode and completion state;
- epoch and optimizer-step accounting;
- immutable model/tokenizer revision requirements;
- E5 query revision and query hash;
- full-training initialization source;
- config hash agreement;
- metric-file agreement;
- history length and schema;
- recomputed best/latest checkpoint hashes;
- checkpoint payload schema;
- label/action spaces;
- parameter count;
- data/corpus hashes;
- no internal_test access;
- no embedded base-model cache;
- best/latest checkpoint identity rejection unless explicitly justified.

Reusable CLI commands:

```text
env PYTHONPATH=src python -m mednorm_vi.training.phase2.cli validate-artifact --kind e4 --artifact-dir ARTIFACT_DIR --mode full
env PYTHONPATH=src python -m mednorm_vi.training.phase2.cli validate-artifact --kind e5 --artifact-dir ARTIFACT_DIR --mode full
env PYTHONPATH=src python -m mednorm_vi.training.phase2.cli validate-artifact --kind l4 --artifact-dir ARTIFACT_DIR --mode full
env PYTHONPATH=src python -m mednorm_vi.training.phase2.cli validate-proposals --manifest PROPOSAL_MANIFEST --proposals-jsonl PROPOSALS_JSONL
env PYTHONPATH=src python -m mednorm_vi.training.phase2.cli gate-internal-test --profile-manifest PROFILE_MANIFEST --authorization I_AUTHORIZE_ONE_SHOT_INTERNAL_TEST_PHASE2
```

No real full-training artifact exists yet, so no full artifact is
`ARTIFACT_VALIDATED`.

## 10. Validation-only Ablation Path

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`, not `VALIDATION_EVALUATED`.

Implemented source/config/notebook:

- `src/mednorm_vi/training/phase2/validation_ablation.py`
- `src/mednorm_vi/evaluation/l3_l4_ablation_v2.py`
- `configs/evaluation/l3_l4_ablation_v2.yaml`
- `configs/evaluation/phase2_validation_ablation.yaml`
- `notebooks/MedNorm_Phase2_Validation_Ablation.ipynb`

Supported arms:

- `E3_only`
- `E3_plus_E4_phobert_w2ner`
- `E3_plus_E5_xlmr_mrc`
- `E3_plus_E4_plus_E5`
- `all_available_l3_experts`
- `deterministic_l4_v1`
- `learned_l4_v2`

The ablation path uses `src/mednorm_vi/evaluation/exact_mention.py` unchanged.
It rejects internal_test examples and reports unavailable experts as
`UNAVAILABLE_UNTRAINED`.

Validation ablation authorization string:

```text
I_AUTHORIZE_PHASE2_VALIDATION_ABLATION
```

No governed validation ablation was executed in this local milestone because
validated E4/E5/L4 full artifacts do not exist yet.

## 11. Internal_test Freeze Gate

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`.

Implemented source:

- `src/mednorm_vi/training/phase2/internal_test_gate.py`

The gate requires:

- selected E4/E5/L4 artifacts validate;
- feature flags and thresholds frozen;
- config hashes recorded;
- checkpoint hashes recorded;
- validation ablation hash recorded;
- model revisions immutable;
- exact operator authorization string supplied.

Future one-shot authorization string:

```text
I_AUTHORIZE_ONE_SHOT_INTERNAL_TEST_PHASE2
```

This milestone did not run internal_test.

## 12. Feature Flags and Defaults

The current production-development defaults are preserved:

- E1 medication grammar: enabled;
- E2 laboratory parser: enabled;
- E3 ViHealthBERT: enabled;
- E4 PhoBERT W2NER: disabled;
- E5 XLM-R MRC-NER: disabled;
- E6 GLiNER: disabled;
- E7 Qwen proposer: disabled;
- deterministic L4 v1: disabled;
- learned L4 v2: disabled.

No new trained expert or learned resolver is enabled by code existence alone.

## 13. Parameter Budget

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`.

Implemented source:

- `src/mednorm_vi/training/phase2/budget.py`

The existing model registry remains unchanged for planned future models. The
new Phase-2 validation-profile budget helper validates only models selected for
a frozen validation profile, rejecting:

- base parameters above 9B;
- missing or mutable model revision;
- missing checkpoint hash for enabled models;
- missing local path when required;
- silent network acquisition paths;
- duplicate shared-backbone counting.

Local unit tests cover the pass and rejection paths.

## 14. Colab Notebooks

Status for all Phase-2 notebooks: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`,
`READY_FOR_COLAB_SMOKE`.

Tracked notebooks:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb`
- `notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb`
- `notebooks/MedNorm_Phase2_Validation_Ablation.ipynb`

Each notebook records intended environment, Drive paths, smoke/full distinction,
authorization strings, artifact directories, resume behavior, expected validator
JSON, and no organizer inference or `output.zip` generation.

## 15. Files Created

- `configs/evaluation/phase2_validation_ablation.yaml`
- `configs/training/phase2_e4_phobert_w2ner_colab.yaml`
- `configs/training/phase2_e5_xlmr_mrc_ner_colab.yaml`
- `configs/training/phase2_frozen_proposal_generation.yaml`
- `configs/training/phase2_l4_learned_resolver_v2_colab.yaml`
- `docs/audits/0036-phase2-training-readiness-and-artifact-validation.md`
- `notebooks/MedNorm_Phase2_Validation_Ablation.ipynb`
- `src/mednorm_vi/training/phase2/__init__.py`
- `src/mednorm_vi/training/phase2/artifacts.py`
- `src/mednorm_vi/training/phase2/budget.py`
- `src/mednorm_vi/training/phase2/cli.py`
- `src/mednorm_vi/training/phase2/common.py`
- `src/mednorm_vi/training/phase2/e4_w2ner_training.py`
- `src/mednorm_vi/training/phase2/e5_mrc_training.py`
- `src/mednorm_vi/training/phase2/internal_test_gate.py`
- `src/mednorm_vi/training/phase2/l4_training.py`
- `src/mednorm_vi/training/phase2/proposal_generation.py`
- `src/mednorm_vi/training/phase2/validation_ablation.py`
- `tests/unit/test_phase2_training_readiness.py`

## 16. Files Modified

- `.gitignore`
- `configs/evaluation/l3_l4_ablation_v2.yaml`
- `docs/notebooks/notebook_execution_integrity.md`
- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb`
- `notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb`
- `src/mednorm_vi/evaluation/l3_l4_ablation_v2.py`
- `tests/unit/test_notebooks.py`

No committed audit 0034 or 0035 file was modified.

## 17. Ignored Runtime Artifacts

`.gitignore` now explicitly excludes:

- `artifacts/**`
- `weights/**`
- `caches/**`
- `*.ckpt`

Existing exclusions for `checkpoint/`, `reports/`, `data/`, `models/`,
`*.pt`, `*.pth`, `*.safetensors`, `*.bin`, and `*.zip` remain in force.

No checkpoint, report, corpus, model weight, ZIP, or cache artifact was created
or staged by this milestone.

## 18. Local Preflight Evidence

Required checks executed:

```text
env PYTHONPATH=src python -m pytest -q
1322 passed, 1 skipped in 118.58s
SKIPPED: tests/unit/test_vietmed_adapter.py:399 because pyarrow is not installed locally

ruff check .
All checks passed!

env PYTHONPATH=src python -m mypy
Success: no issues found in 256 source files

env PYTHONPATH=src python -m compileall -q src
<no output; success>

git diff --check
<no output; success>
```

Additional local checks:

```text
env PYTHONPATH=src python -m pytest -q tests/unit/test_phase2_flags_registry_notebooks.py tests/unit/test_notebooks.py tests/unit/test_w2ner_expert.py tests/unit/test_mrc_expert.py tests/unit/test_l4_learned_v2.py tests/unit/test_l3_multi_expert_phase2.py tests/unit/test_gliner_adapter.py tests/unit/test_phase2_training_readiness.py
67 passed

ruff check notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb notebooks/MedNorm_Phase2_Validation_Ablation.ipynb
All checks passed!

env PYTHONPATH=src python -m mednorm_vi.model_registry.cli --profile Safe-8.85B
base_parameters=8852000000, adapter_parameters=10000000, within_9b=true
```

Modified notebook code cells parse:

```text
all modified notebook code cells parse
```

Protected files unchanged by Git diff/status:

- `docs/MedNorm-VI_Architecture.pdf`
- `docs/audits/0034-l3-span-lattice-l4-resolver-and-ablation.md`
- `docs/audits/0035-multi-expert-mention-ensemble-and-learned-l4-v2.md`
- `src/mednorm_vi/evaluation/exact_mention.py`

Governed corpus hashes recomputed unchanged:

```text
9e8060775d97713485c964df9165f640050a0085f385b2a9ff216c75f9d81588  canonical_examples/public_ner_examples.jsonl
892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a  splits/train.jsonl
ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103  splits/validation.jsonl
e23acde0d87154d1750d25d1e47c92c86e457e42f2f97cb985661253dea7e135  splits/internal_test.jsonl
a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3  manifests/corpus_manifest.json
```

Existing S1 checkpoint unchanged:

```text
a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c  checkpoint/s1_mention_full_training_v1/best.pt
mtime: 2026-07-27 00:53:41.047814400 +0700
```

Weight/output checks:

- `git ls-files | rg '\.(pt|pth|ckpt|safetensors|bin|zip)$'` produced no tracked weight/archive files.
- `git check-ignore -v` confirmed `artifacts/`, `weights/`, `caches/`,
  `reports/`, `checkpoint/`, `data/`, `models/`, and `output.zip` are ignored.
- `test ! -e output.zip` passed.
- `git status --short -- reports checkpoint artifacts weights caches output.zip`
  produced no output.

## 19. Non-execution Guarantees

This milestone did not:

- train locally;
- run backward on governed full datasets locally;
- download large model weights;
- call external APIs;
- mutate the existing S1 checkpoint;
- run internal_test;
- run organizer inference;
- create `output.zip`;
- generate governed proposal datasets or reports in Git.

## 20. Remaining Colab Blockers

The following remain outside local execution:

- E4 immutable PhoBERT model/tokenizer revision resolution in Colab;
- E4 smoke/full training and artifact validation;
- E5 immutable XLM-R model/tokenizer revision resolution in Colab;
- E5 smoke/full training and artifact validation;
- frozen train/validation proposal dataset generation from frozen experts;
- learned L4 v2 smoke/full training and artifact validation;
- governed validation-only ablation with real validated artifacts;
- profile freeze and future internal_test gate authorization.

## 21. Safe-to-Commit Verdict

Safe to commit after human review. Required tests and static checks passed, no
protected architecture/audit/evaluator file changed, no checkpoint/report/data/
model/cache/ZIP artifact is staged, and the default feature flags remain
unchanged.

Final `git status --short`:

```text
 M .gitignore
 M configs/evaluation/l3_l4_ablation_v2.yaml
 M docs/audits/README.md
 M docs/notebooks/notebook_execution_integrity.md
 M notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb
 M notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb
 M notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb
 M src/mednorm_vi/evaluation/l3_l4_ablation_v2.py
 M tests/unit/test_notebooks.py
?? configs/evaluation/phase2_validation_ablation.yaml
?? configs/training/phase2_e4_phobert_w2ner_colab.yaml
?? configs/training/phase2_e5_xlmr_mrc_ner_colab.yaml
?? configs/training/phase2_frozen_proposal_generation.yaml
?? configs/training/phase2_l4_learned_resolver_v2_colab.yaml
?? docs/audits/0036-phase2-training-readiness-and-artifact-validation.md
?? notebooks/MedNorm_Phase2_Validation_Ablation.ipynb
?? src/mednorm_vi/training/phase2/
?? tests/unit/test_phase2_training_readiness.py
```
