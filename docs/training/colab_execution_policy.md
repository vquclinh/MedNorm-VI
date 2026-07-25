# Colab Execution Policy (Audit 0017)

MedNorm-VI trains on Google Colab Pro. This policy is authoritative for where each
kind of work runs. It exists so that no GPU/neural workload runs on the local
machine and so every artifact remains reproducible for the organizer's private
rebuild.

## Local machine — ALLOWED (deterministic, offline, CPU-only)

- repository audit and hygiene;
- data governance and manifest validation;
- deterministic dataset conversion / adapters;
- offset validation;
- label mapping;
- leakage analysis and grouped split generation;
- lightweight corpus statistics;
- small synthetic fixtures;
- schema/tests/static checks;
- notebook construction and JSON validation;
- training **configuration**, checkpoint **contracts**, reproducibility manifests;
- documentation.

## Colab — REQUIRED (anything GPU-benefiting)

- pretrained model / tokenizer downloads;
- model conversion;
- S0 continued pretraining;
- S1 mention-model training;
- S2 assertion-head training;
- S3 ICD/RxNorm retriever (embedding) training;
- S4 reranker (cross-encoder) training;
- S5 Qwen LoRA critic/adjudicator;
- large neural pseudo-labeling / embedding generation;
- checkpoint export; GPU benchmarks.

**No training or fine-tuning may run on the local machine.** No external/hosted
inference APIs. No paid inference APIs.

## Stop-and-hand-off rule

Any command that would download model weights or begin neural training must STOP
locally and emit, for the user to run in Colab:

1. the exact notebook to open;
2. required Colab runtime type + expected GPU;
3. Google Drive paths (project / data / model-cache / checkpoint roots);
4. estimated disk / RAM / VRAM;
5. expected duration;
6. the commands the user runs;
7. the expected generated artifacts;
8. how to return artifacts to the repository (`models/checkpoints/...` + manifest);
9. the local validation commands to run after return.

## Return-to-repo contract

Checkpoints return under `models/checkpoints/full_v1/<role>/...` with a checkpoint
manifest (see `configs/model_registry/checkpoint_manifest.template.yaml`). Weights
and optimizer states remain **git-ignored**; only manifests/metrics/hashes are
tracked (and restricted base-model weights are never committed). The pipeline
continues to **fail fast** when a required checkpoint is absent — this is not
weakened to make smoke tests pass.
