# CLAUDE.md — MedNorm-VI repository rules

Guidance for AI coding agents (and humans) working in this repository. These
rules are binding. `AGENTS.md` mirrors this file.

## Source of truth

- **Read `MedNorm-VI_Architecture.pdf` (the English PDF) before any
  architecture-sensitive change.** It is the authoritative full specification.
- `docs/architecture/` (ARCHITECTURE_DIGEST, LAYER_CONTRACTS, CASE_ROUTING) is a
  faithful summary — if it conflicts with the PDF, the PDF wins; fix the digest.
- The Vietnamese PDF is a local translation, **git-ignored on purpose**. Never
  add, rename, delete, or commit it, and never modify the English PDF.

## Non-negotiable invariants

1. **Preserve exact original offsets.** `original_text` is immutable and is the
   ONLY source of emitted `text`/`position`. For every entity,
   `original_text[start:end] == entity.text`, with `[start, end)` end-exclusive.
   Normalization runs on a **copy** with a preserved alignment; **normalized
   coordinates never become output coordinates.** An LLM must never compute
   offsets freely.
2. **Never use external inference APIs.** All models are self-hosted. No network
   calls to model providers anywhere in the pipeline.
3. **Enforce the total 9B base-model budget.** The sum of `base_params` over
   enabled base models must be ≤ 9,000,000,000. Adapter/LoRA/head params are
   **excluded** (a small excess, ~9.1B, caused only by adapters is accepted).
   Keep the manifest in `configs/parameter_budget.yaml` accurate; verify with
   `validator.budget`.
4. **Track base models, adapters, data, and generated artifacts separately.**
   Base weights, adapters, raw/private data, KB snapshots, generated indices,
   and prediction outputs each have their own root and are git-ignored.
5. **Never commit** secrets, private competition data, model weights, generated
   indices, or prediction outputs. Only code, configs, docs, tests, and small
   tracked fixtures belong in git.
6. **Do not silently simplify or bypass a layer.** The nine layers (L1–L9) are
   all mandatory even when a module is hard. Mark unfinished work with explicit
   `TODO(Lx)` and `NotImplementedError`; do not fake task-solving logic.
7. **All new modules need tests and documented contracts.** Add unit tests and
   update `docs/architecture/LAYER_CONTRACTS.md` when a layer's interface changes.

## Change discipline

- **Every meaningful change must produce a numbered audit Markdown file** under
  `audits/` (e.g. `audits/0002-<slug>.md`). Use the next available number; never
  overwrite an existing audit. Include what changed, why, tests run, and results.
- Record architecture-sensitive decisions as numbered ADRs in `docs/decisions/`.
- Do not auto-commit; leave changes for human review unless explicitly asked.

## Engineering conventions

- Python ≥ 3.10, full type hints; keep modules small with explicit
  responsibilities. Schemas are contracts (dataclasses), not implementations.
- Avoid premature model-framework dependencies (torch/transformers/faiss are
  added per-layer as that layer is implemented, not before).
- Do not download any model or dataset, and do not build ICD/RxNorm indices,
  during unrelated work.
- Run `pytest`, `ruff check .`, and `mypy` before finishing a change.
- Fail-fast: on any offset/schema/KB violation, stop and write an audit log;
  never silently repair output.

## Open questions

Several matching/packaging details depend on organizer data and must not be
guessed — see `docs/decisions/0001-open-questions-pending-organizer-data.md`.
