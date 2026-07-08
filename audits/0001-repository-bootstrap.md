# Audit 0001 — Repository Bootstrap

- **Date:** 2026-07-08
- **Author:** Claude (AI agent), for review by the MedNorm-VI team
- **Change type:** Repository foundation (Phase 0 — contracts, validators, docs)
- **Spec reference:** `MedNorm-VI_Architecture.pdf` (Super Architecture
  Specification v1.1, 08 July 2026), especially sections 3–4, 5, 16, 17, 19, 20.

## 1. Objective and scope

Bootstrap the MedNorm-VI repository into a clean, extensible foundation that
mirrors the nine-layer architecture and lets each layer be implemented
incrementally. Scope is **bootstrap only**: typed contracts, the deterministic
validation foundation (Phase 0), configuration, documentation, and per-layer
module stubs. **No models, datasets, or ICD/RxNorm indices were downloaded or
implemented.** No placeholder logic pretends to solve the task; unimplemented
layers raise `NotImplementedError` with `TODO(Lx)` markers.

## 2. Files created

**Root / project docs**
- `pyproject.toml` — packaging, dev extras (pytest/ruff/mypy), tool config.
- `CLAUDE.md`, `AGENTS.md` — binding repository rules (mirrored).
- `README.md` — rewritten (see §3).

**configs/**
- `base.yaml` — task contract, label sets, offset policy, determinism, KB
  placeholders.
- `parameter_budget.yaml` — 9B manifest + Safe-8.85B default stack + variants.
- `routes.yaml` — declarative, versioned C1–C7 placeholders + routing signals.

**docs/**
- `architecture/ARCHITECTURE_DIGEST.md`
- `architecture/LAYER_CONTRACTS.md`
- `architecture/CASE_ROUTING.md`
- `decisions/0001-open-questions-pending-organizer-data.md`

**src/mednorm_vi/** (package + `py.typed`)
- `schemas/`: `constants.py`, `spans.py`, `document.py`, `routing.py`,
  `hypotheses.py`, `evidence.py`, `prediction.py`, `__init__.py`.
- Layer stubs (interfaces + `NotImplementedError`): `document_intelligence/`,
  `case_router/`, `mention_factory/`, `boundary_type/`,
  `specialists/{,assertion,icd,rxnorm}/`, `evidence_graph/`,
  `confidence_cascade/`, `metric_decoder/`.
- `validator/` (implemented): `results.py`, `_coerce.py`, `offsets.py`,
  `entities.py`, `duplicates.py`, `budget.py`, `submission.py`, `__init__.py`.

**tests/**
- `unit/`: `test_schemas.py`, `test_offsets.py`, `test_entities.py`,
  `test_duplicates.py`, `test_budget.py`, `test_normalization_offsets.py`,
  `test_submission.py`.
- `integration/test_document_validation.py`.
- `fixtures/sample_document.json` (synthetic, offset-verified).

**Artifact roots (README-only, git-ignored contents)**
- `data/README.md`, `models/README.md`, `indices/README.md`,
  `outputs/README.md`, `scripts/README.md`.

## 3. Files modified

- `.gitignore` — extended to ignore model weights, raw/private data, KB
  snapshots, generated indices, prediction outputs, secrets, and Python/OS
  caches, while keeping each artifact root's `README.md`. **The Vietnamese-PDF
  ignore line was preserved.**
- `README.md` — replaced the 2-line stub with the required content (problem, five
  entity types, assertion labels, ICD-10/RxNorm linking, nine layers, seven case
  families, 9B constraint, bootstrap status, install/test commands, note that the
  English PDF is authoritative).

No source-tracked PDF was modified.

## 4. Architecture decisions encoded

- **Nine layers (L1–L9)** each get their own package with an input/output
  contract; none is collapsed or skipped. `docs/architecture/LAYER_CONTRACTS.md`
  documents input/output/responsibility/must-not/provenance/offset rules per
  layer.
- **Multi-label routing** for C1–C7 is explicit in schemas (`RouteDecision`
  holds many `RouteTag`s) and in `configs/routes.yaml` (`multi_label: true`).
- **Five entity types** and **three assertion labels** are frozen in
  `schemas/constants.py`; ontology-by-type forbids cross-linking (ICD↔MEDICATION,
  RxNorm↔DIAGNOSIS).
- **Coordinate triplet** (absolute/local/normalized) is a first-class contract
  (`SpanCoordinates`, `OffsetAlignment`); output position always derives from
  **absolute** coordinates.
- Model choices are **configurable**, not hard-coded in code — only the manifest
  fixes the spec's Safe-8.85B stack, with named variants documented.

## 5. Parameter-budget handling

- `configs/parameter_budget.yaml` encodes `max_total_base_params =
  9,000,000,000`, `adapters_excluded: true`, a soft target of 8.85B, a per-model
  registry (name/role/base_params/adapter_params/enabled/notes), and the four
  named variants.
- `mednorm_vi.validator.budget` sums **only** enabled `base_params` (adapters
  never added), fails when the base sum exceeds 9B, warns past the soft target,
  and validates required model fields.
- Shipped manifest base total = **8,852,000,000** (≤ 9B), matching the spec's
  Safe Submission Stack.

## 6. Offset and schema invariants added

- `validator.offsets`: enforces `original_text[start:end] == entity.text`,
  end-exclusive `[start, end)` (length must equal `len(text)`), non-empty text,
  in-bounds, non-negative start; code-point indexing (matches spec alignment).
- `validator.entities`: required-field schema, allowed entity type, allowed &
  non-duplicated assertion labels, candidate-ontology-vs-type consistency.
- `validator.duplicates`: identity key `(text, type, start, end)` — same text at
  different positions stays distinct; exact duplicates are flagged
  (`validate_no_duplicates`) or deterministically merged (`deduplicate`,
  first-occurrence wins).
- `validator.submission`: STUB checking the `1.json`..`100.json` file set, UTF-8,
  and JSON well-formedness (packaging rules unconfirmed — see ADR 0001).

## 7. Tests and validation commands run

```
python3 -m pytest -q
ruff check .
python3 -m mypy
```

Plus git ignore verification via `git check-ignore` and a validator smoke test.

## 8. Exact results of those validations

- **pytest:** `45 passed in 0.13s` (7 unit files + 1 integration file).
- **ruff:** `All checks passed!`
- **mypy (strict):** `Success: no issues found in 28 source files`.
- **Budget:** `total_base_params == 8,852,000,000`, budget validation `ok` (1
  advisory WARNING: 8.852B > 8.850B soft target, within the hard 9B cap).
- **git check-ignore:** Vietnamese PDF → ignored (`.gitignore:8`); English PDF →
  not ignored (exit 1); `data/**`, `models/**`, `indices/**`, `outputs/**`,
  `output/1.json`, `.env` → ignored; all four artifact `README.md` files → kept.

## 9. Items intentionally not implemented

- All nine layers' logic (L1–L9) except the deterministic L9 validator
  foundation — stubs raise `NotImplementedError` with `TODO(Lx)`.
- No model/adapter weights, no dataset, no ICD/RxNorm indices, no LLM.
- No heavy dependencies (torch/transformers/faiss) added.
- Submission packaging (ZIP layout, file-name convention, ordering) is a stub;
  KB-membership checking of candidate codes deferred (needs frozen KB snapshot).
- Local metric evaluator (WER / Jaccard) not implemented — next milestone.

## 10. Risks, assumptions, and open questions

- **PDF filename differs from the task prompt.** The prompt referenced
  `MedNorm-VI_Super_Architecture_English_pdfLaTeX.pdf`; the actual authoritative
  file at the repo root is **`MedNorm-VI_Architecture.pdf`**. Content matches the
  described spec; docs reference the real filename.
- **Five entity types** are `MEDICATION, DIAGNOSIS, SYMPTOM, TEST_NAME,
  TEST_RESULT`. The PDF says "five entity categories" and these five appear
  throughout (sections 4.2, 7.2); they are not enumerated in a single explicit
  list, so this is a (well-supported) inference.
- **Submission file naming** assumed `{i}.json` from the prompt's
  `output/1.json`..`output/100.json`; padding/prefix/casing unconfirmed.
- Organizer-dependent decisions (matching by position vs text+type, WER
  tokenization, ICD-10/RxNorm versions, candidate-list limits, nested entities,
  future relation field, rebuild limits) are unresolved — see ADR 0001; nothing
  hard-codes a guess.
- Dev tools (pytest/ruff/mypy) and `types-PyYAML`-free yaml import were installed
  into the environment for validation; they are declared as dev extras, not
  runtime deps.

## 11. Vietnamese PDF confirmation

`MedNorm-VI_Architecture_Vietnamese.pdf` remains **ignored and untracked**:
`git check-ignore -v` matches `.gitignore:8`, and it does not appear in
`git status --short`. It was not added, renamed, deleted, or modified. The
authoritative English PDF was not modified.

## 12. `git status --short`

```
 M .gitignore
 M README.md
?? AGENTS.md
?? CLAUDE.md
?? configs/
?? data/
?? docs/
?? indices/
?? models/
?? outputs/
?? pyproject.toml
?? scripts/
?? src/
?? tests/
```

(Bootstrap changes are left **uncommitted** for human review, as required.)

## 13. Recommended next implementation milestone

**Phase 0 completion — Local metric evaluator + audit/replay format.** Implement
a faithful clone of the organizer metric (text WER 30%, assertion Jaccard 30%,
candidate Jaccard 40%, wrong-type double penalty) plus the per-sample HTML/JSON
audit export. The roadmap forbids reversing this order: the evaluator, offset
logic, and validator must be complete before any heavy model, so model changes
can be judged against the true metric. Then proceed to L1 Document Intelligence
(reversible normalization + `OffsetAlignment` construction) as the first
model-adjacent layer.
