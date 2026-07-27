# Audit 0035 — Multi-Expert Mention Ensemble and Learned L4 v2

Date: 2026-07-27

## Status Vocabulary

- `IMPLEMENTED`: tracked code/config/notebook contract exists.
- `PREFLIGHT_EXECUTED`: deterministic local contract or conversion preflight was run by tests.
- `SMOKE_EXECUTED`: bounded model/shape smoke with execution evidence.
- `FULLY_TRAINED`: full training/fine-tuning completed with manifest and checkpoint.
- `EVALUATED`: governed evaluation completed with frozen checkpoint/config.
- `UNAVAILABLE_UNTRAINED`: component has no valid local trained checkpoint and is not scored.

## Initial State

- `pwd`: `/mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI`
- branch: `main`
- initial status: `## main...origin/main`, clean
- initial short status: clean
- recent HEAD at milestone start: `23065f9 feat: implement L3 span lattice and L4 resolver ablation`
- preflight `git diff --check`: clean
- Audit 0034 was committed before this milestone began; no Audit 0034 files were pending.
- No commit or push was performed by this milestone.

## Architecture Sections Applied

- §4 span contracts: all new expert proposals validate exact `original_text[start:end] == text`, half-open offsets, and original-coordinate identity.
- §6 Mention Factory: E4 PhoBERT W2NER, E5 XLM-R MRC-NER, E6 GLiNER, and E7 Qwen3-1.7B proposer now have L3 contracts that feed the unified `SpanLattice`.
- §7 Boundary & Type Resolver: learned L4 v2 target construction and read-only checkpoint runtime implement boundary action and type decision responsibilities while preserving deterministic L4 v1.
- §15 leakage protocol: learned L4 v2 examples reject `internal_test` and include grouped train/validation splitting.
- §17 parameter budget: registry rows for Phase-2 base models use the Safe-8.85B planned stack and shared-backbone accounting.
- §18 ablation protocol: Phase-2 ablation arms report untrained experts as `UNAVAILABLE_UNTRAINED`.
- §19 core contracts: no expert emits organizer JSON; all outputs remain proposal or hypothesis contracts.

## E4 PhoBERT W2NER

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`.

Implemented in `src/mednorm_vi/mention_factory/w2ner.py`:

- governed `EntitySpan` adapter and exact word/original-offset validation;
- whitespace word scan and subword-to-word alignment validation;
- W2NER label vocabulary with `NONE`, `NNW`, and `THW:<TYPE>` relation labels;
- square relation-grid construction and padded-pair masking;
- contiguous/overlapping span decoding from relation grids back to original offsets;
- conversion into `ExpertSpanProposal` with E4 provenance, model revision, config hash, and checkpoint hash fields;
- lazy relation-grid model-head factory, with no torch import during normal repository import;
- checkpoint metadata schema and disabled-by-default checkpoint validation.

No discontinuous entity support is claimed.

## E5 XLM-R MRC-NER

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`.

Implemented in `src/mednorm_vi/mention_factory/mrc.py`:

- five versioned type queries for MEDICATION, DIAGNOSIS, SYMPTOM, TEST_NAME, and TEST_RESULT;
- governed conversion into query/context examples;
- query and context masks, with start/end labels only allowed on context tokens;
- exact start/end pairing with max span length and optional overlap suppression;
- conversion into `ExpertSpanProposal` with E5 provenance and checkpoint metadata;
- lazy MRC span-head factory and disabled-by-default checkpoint validation.

Queries are evidence only; they do not authorize text absent from input.

## E6 GLiNER Adapter

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`.

Implemented in `src/mednorm_vi/mention_factory/gliner.py`:

- configurable local model path with no automatic download;
- five tracked label descriptions;
- strict raw-span parsing;
- exact substring and offset validation;
- rejection of unsupported labels, malformed offsets, and non-reversible spans;
- per-label confidence features;
- batch and single-document adapter methods around an injected local backend;
- disabled-by-default local checkpoint validation.

No GLiNER performance is claimed.

## E7 Qwen3-1.7B Proposer

Status: `IMPLEMENTED` as interface only; `UNAVAILABLE_UNTRAINED`.

Implemented in `src/mednorm_vi/mention_factory/qwen_proposer.py`:

- stable E7 config and status interface;
- disabled/missing-checkpoint status reporting;
- proposal execution raises `QwenProposerUnavailableError`;
- no download, training, execution, or fabricated prediction path.

The older anchored helper in `src/mednorm_vi/mention_factory/adapters.py` is retained only for pre-Phase-2 unit fixtures and is no longer the production E7 path.

## Learned L4 v2

Status: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`; `UNAVAILABLE_UNTRAINED` as a model.

Implemented in `src/mednorm_vi/resolution/learned_v2.py`:

- deterministic training-example builder from lattice proposals and governed gold mentions;
- leakage-safe split restriction to train/validation;
- grouped train/validation construction by source/document group;
- feature vectors covering expert logits/agreement, boundaries, section/route evidence, grammar completeness, laboratory structure, normalized/original span representations, nearby delimiters/tokens, ontology placeholders, overlap structure, candidate length, and boundary deltas;
- target definitions for `KEEP`, `SELECT_EXISTING`, bounded `OFFSET`, and `DROP`;
- type target over five organizer types plus `DROP`;
- wrong-type cost and explicit DIAGNOSIS/SYMPTOM confusion flag;
- token/span IoU auxiliary target and WER-aware boundary surrogate field;
- read-only JSON checkpoint schema and runtime;
- calibration thresholds for keep score, abstention margin, and wrong-type risk.

No learned L4 v2 checkpoint was trained locally.

## Data, Split, and Leakage Controls

- Trainable components use governed train data for training and governed validation for model selection.
- `internal_test` is rejected by learned L4 training-example construction.
- Phase-2 ablation planning reports unavailable checkpoints rather than producing zero-prediction metric rows.
- Grouped splitting keeps one source/document group entirely in train or validation.
- Run manifests require stage, expert, config hash, data/corpus hashes, model revision, seed, Git commit, checkpoint SHA-256, parameter count, split identities, and `internal_test_accessed`.

## Feature Flags and Profiles

Updated `configs/pipeline/full_v1.yaml` and `src/mednorm_vi/inference/config.py` with explicit flags:

- `enable_e1_medication_grammar: true`
- `enable_e2_laboratory_parser: true`
- `enable_e3_vihealthbert: true`
- `enable_e4_phobert_w2ner: false`
- `enable_e5_xlmr_mrc: false`
- `enable_e6_gliner: false`
- `enable_e7_qwen_proposer: false`
- `enable_l4_deterministic_v1: false`
- `enable_l4_learned_v2: false`

The measured default remains E3 ViHealthBERT enabled and deterministic L4 v1 disabled.

## Parameter Budget

- `configs/parameter_budget.yaml` remains the confirmed Safe-8.85B manifest with 8,852,000,000 planned base parameters.
- `configs/model_registry/models_v1.yaml` was updated to version 2 with PhoBERT-large, XLM-R-large, GLiNER-medium-v2.1, Qwen3-1.7B proposer, Qwen3-4B adjudicator, Qwen3 embedding/reranker, BGE-M3, SapBERT, and shared ViHealthBERT rows.
- Adapters/heads are recorded separately and excluded from the base-parameter cap.
- E3 points to the validated Audit 0033 local checkpoint hash.
- New trainable experts carry `UNAVAILABLE_UNTRAINED` checkpoint status until real checkpoints exist.

## Colab Notebooks

Added:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb`
- `notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb`

Each notebook writes to Drive, validates governed corpus hashes before model acquisition, separates smoke/full outputs, rejects incompatible smoke resume for full training, writes immutable manifests, validates checkpoints after save/reload, and requires an explicit full-training authorization string.

## Tests and Static Checks

Executed locally:

- `env PYTHONPATH=src python -m pytest -q` — `1312 passed, 1 skipped` (`tests/unit/test_vietmed_adapter.py:399`, local `pyarrow` not installed).
- `ruff check .` — passed.
- `env PYTHONPATH=src python -m mypy` — passed after adding a type-only local PyYAML stub path under `typings/`.
- `env PYTHONPATH=src python -m compileall -q src` — passed.
- `git diff --check` — passed.
- Notebook code-cell parse — 157 code cells compiled after eliding notebook-only magic/shell lines; raw Python compile is not appropriate for retained notebooks with `%cd` magic, and IPython is not installed locally.

Custody checks:

- `docs/MedNorm-VI_Architecture.pdf` diff: unchanged.
- `docs/audits/0034-l3-span-lattice-l4-resolver-and-ablation.md` diff: unchanged.
- `src/mednorm_vi/evaluation/exact_mention.py` diff: unchanged.
- governed corpus hashes under `data/derived/training_corpora/mednorm_vi_training_v1/`:
  - canonical examples: `9e8060775d97713485c964df9165f640050a0085f385b2a9ff216c75f9d81588`
  - train: `892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a`
  - validation: `ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103`
  - internal_test: `e23acde0d87154d1750d25d1e47c92c86e457e42f2f97cb985661253dea7e135`
  - corpus manifest: `a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3`
  - governed exclusions config: `a949ce2101132482e7b409cbb59eaf2578f35cfe9ed2855a54967e6475efe8cf`
- S1 checkpoint SHA-256: `a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c`.
- S1 checkpoint stat at verification: mtime `2026-07-27 00:53:41.047814400 +0700`, size `1615513303`; no milestone command wrote the checkpoint.
- no tracked `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`, or `output.zip` files.
- `.gitignore` covers `checkpoint/`, `models/**`, `reports/**`, `*.zip`, `*.bin`, and `*.safetensors`.
- no repository `output.zip` exists.
- no organizer inference was run. Required pytest includes isolated temp-dir packaging tests from older milestones; no organizer input/full inference or repository output archive was produced.

## Changed Tracked Files

Changed files at verification time:

- `configs/evaluation/l3_l4_ablation_v2.yaml`
- `configs/mention_factory/gliner_v1.yaml`
- `configs/mention_factory/phobert_w2ner_v1.yaml`
- `configs/mention_factory/qwen_proposer_v1.yaml`
- `configs/mention_factory/xlmr_mrc_ner_v1.yaml`
- `configs/model_registry/models_v1.yaml`
- `configs/pipeline/full_v1.yaml`
- `configs/resolution/learned_l4_v2.yaml`
- `docs/audits/0035-multi-expert-mention-ensemble-and-learned-l4-v2.md`
- `docs/notebooks/notebook_execution_integrity.md`
- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb`
- `notebooks/MedNorm_L4_Learned_Resolver_v2_Training.ipynb`
- `pyproject.toml`
- `src/mednorm_vi/evaluation/l3_l4_ablation_v2.py`
- `src/mednorm_vi/inference/config.py`
- `src/mednorm_vi/lattice/__init__.py`
- `src/mednorm_vi/lattice/builder.py`
- `src/mednorm_vi/lattice/models.py`
- `src/mednorm_vi/mention_factory/adapters.py`
- `src/mednorm_vi/mention_factory/gliner.py`
- `src/mednorm_vi/mention_factory/mrc.py`
- `src/mednorm_vi/mention_factory/qwen_proposer.py`
- `src/mednorm_vi/mention_factory/w2ner.py`
- `src/mednorm_vi/model_registry/registry.py`
- `src/mednorm_vi/resolution/learned_v2.py`
- `src/mednorm_vi/training/manifests.py`
- `tests/unit/test_gliner_adapter.py`
- `tests/unit/test_l3_multi_expert_phase2.py`
- `tests/unit/test_l4_learned_v2.py`
- `tests/unit/test_mrc_expert.py`
- `tests/unit/test_notebooks.py`
- `tests/unit/test_phase2_flags_registry_notebooks.py`
- `tests/unit/test_w2ner_expert.py`
- `typings/yaml/__init__.pyi`

## Immutable Evidence Preserved

- `docs/MedNorm-VI_Architecture.pdf` unchanged.
- Audit 0034 unchanged.
- `src/mednorm_vi/evaluation/exact_mention.py` unchanged.
- No files under `checkpoint/`, `reports/`, `data/`, `models/`, `weights/`, or `caches/` were intentionally modified.
- No model weights were added to Git.
- No organizer inference was run.
- No `output.zip` was generated.

## Remaining Blockers

- E4 requires full Colab training and a pinned PhoBERT-large revision.
- E5 requires full Colab training and a pinned XLM-R-large revision.
- E6 requires a real local GLiNER checkpoint and evaluation only; this milestone did not download or train GLiNER.
- E7 remains interface-only until the later Confidence Cascade/LLM milestone.
- Learned L4 v2 requires leakage-safe training from governed train/validation and validation-selected thresholds before any internal_test access.

## Safe-to-Commit Verdict

Safe to commit after review. Do not add ignored runtime artifacts or weights.
