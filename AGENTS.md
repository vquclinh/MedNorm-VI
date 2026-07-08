# AGENTS.md — MedNorm-VI

This file mirrors [`CLAUDE.md`](CLAUDE.md); both define the binding rules for AI
agents and humans in this repository. Read `CLAUDE.md` for full detail. The core
rules, restated:

## Source of truth
- Read `MedNorm-VI_Architecture.pdf` (English) before any architecture-sensitive
  change; it is authoritative. `docs/architecture/` summarizes it.
- The Vietnamese PDF is git-ignored on purpose — never add, rename, delete, or
  commit it. Never modify the English PDF.

## Non-negotiable invariants
1. **Preserve exact original offsets:** `original_text[start:end] ==
   entity.text`, `[start, end)` end-exclusive. Normalization on a copy only;
   normalized coordinates never become output coordinates. LLMs never compute
   offsets freely.
2. **No external inference APIs** — all models self-hosted.
3. **Enforce the 9B base-model budget** (adapters excluded; ~9.1B tolerated).
   Keep `configs/parameter_budget.yaml` accurate; verify with `validator.budget`.
4. **Track base models, adapters, data, and generated artifacts separately.**
5. **Never commit** secrets, private data, model weights, generated indices, or
   prediction outputs.
6. **Do not silently simplify or bypass a layer** (L1–L9 all mandatory). Mark
   future work with `TODO(Lx)` / `NotImplementedError`; never fake solving logic.
7. **All new modules need tests and documented contracts.**

## Change discipline
- **Every meaningful change adds a numbered audit file** under `audits/` (next
  available number; never overwrite). Record ADRs in `docs/decisions/`.
- Run `pytest`, `ruff check .`, and `mypy`. Do not auto-commit; leave for review.
- Fail-fast on offset/schema/KB violations; never silently repair output.

Open, organizer-dependent questions:
`docs/decisions/0001-open-questions-pending-organizer-data.md`.
