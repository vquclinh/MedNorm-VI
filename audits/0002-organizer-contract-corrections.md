# Audit 0002 — Organizer contract corrections (pre-commit pass)

- **Date:** 2026-07-08
- **Author:** Claude (AI agent), for review by the MedNorm-VI team
- **Change type:** Contract correction on the (still-uncommitted) bootstrap from
  Audit 0001. No new architecture milestone; no models/datasets/indices.
- **Spec reference:** `MedNorm-VI_Architecture.pdf` v1.1, plus organizer-
  confirmed contracts (Vietnamese labels, per-type fields, submission layout,
  offset convention).

## 1. Objective

Correct organizer-facing contracts before the first commit: Vietnamese entity
labels with per-type output fields, deterministic submission (directory + ZIP)
validation for the confirmed layout, an offset-convention golden test, and a
parameter-budget config cleanup (planned vs installed; sensible soft target).
Update documentation and tests accordingly. No L1–L8 model logic was added.

## 2. Files modified and created

**Created**
- `src/mednorm_vi/validator/organizer.py` — organizer-facing entity/document
  validation (Vietnamese labels, per-type fields, candidate ontology syntax).
- `tests/unit/test_organizer.py` — label mapping, serialization, field policy,
  English-rejection, UTF-8 round-trip, candidate syntax.
- `tests/integration/test_submission_smoke.py` — build valid `output.zip`,
  validate, corrupt, verify failure; UTF-8-through-packaging.
- `audits/0002-organizer-contract-corrections.md` — this file.

**Modified**
- `src/mednorm_vi/schemas/constants.py` — added `ORGANIZER_LABEL_BY_TYPE`,
  `TYPE_BY_ORGANIZER_LABEL`, `ORGANIZER_LABELS`, `ORGANIZER_FIELDS_BY_TYPE`;
  confirmed offset-convention comment.
- `src/mednorm_vi/schemas/prediction.py` — `to_internal_dict` (was
  `to_output_dict`, kept as alias) + new `to_submission_dict` (organizer shape).
- `src/mednorm_vi/schemas/__init__.py` — export the new constants.
- `src/mednorm_vi/validator/submission.py` — replaced the stub with
  `validate_output_directory`, `validate_submission_zip`, `write_submission_zip`.
- `src/mednorm_vi/validator/budget.py` — sum the profile status field
  (`in_profile`, configurable via `budget.count_status`); added
  `installed_models`; required fields now include `installed`.
- `src/mednorm_vi/validator/__init__.py` — export organizer + new submission API.
- `configs/parameter_budget.yaml` — `in_profile`/`installed` per model, soft
  target 8.9B, `count_status`, `active_profile`; version bumped to 2.
- `configs/base.yaml` — organizer label map, per-type field policy, confirmed
  offset convention, confirmed submission block.
- `README.md`, `docs/architecture/LAYER_CONTRACTS.md`,
  `docs/architecture/ARCHITECTURE_DIGEST.md`,
  `docs/decisions/0001-open-questions-pending-organizer-data.md`,
  `outputs/README.md` — documentation updates.
- `tests/unit/test_budget.py`, `tests/unit/test_submission.py`,
  `tests/unit/test_offsets.py` — updated for the new contracts + golden test.

## 3. Organizer contracts corrected

1. **Vietnamese entity labels** with an explicit, tested internal↔organizer
   mapping; submission validator rejects English `type` values. JSON is UTF-8
   with `ensure_ascii=false`.
2. **Per-entity-type output fields** enforced by both the serializer (emits only
   allowed fields) and the organizer validator (rejects missing/unsupported).
3. **Confirmed submission layout** (`output.zip` → `output/` → `1.json`..`100.json`)
   with deterministic directory and ZIP validation.
4. **Confirmed end-exclusive offset convention** with a published golden test.
5. **Parameter-budget** profile cleanup (planned vs installed; 8.9B soft target).

## 4. Exact organizer label mapping

| Internal enum | Organizer label |
| ------------- | --------------- |
| `MEDICATION`  | `THUỐC` |
| `DIAGNOSIS`   | `CHẨN_ĐOÁN` |
| `SYMPTOM`     | `TRIỆU_CHỨNG` |
| `TEST_NAME`   | `TÊN_XÉT_NGHIỆM` |
| `TEST_RESULT` | `KẾT_QUẢ_XÉT_NGHIỆM` |

## 5. Per-type field policy

| Organizer type | Emitted fields (in order) |
| -------------- | ------------------------- |
| `THUỐC`, `CHẨN_ĐOÁN` | text, type, candidates, assertions, position |
| `TRIỆU_CHỨNG` | text, type, assertions, position |
| `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM` | text, type, position |

Validated: symptoms cannot carry candidates; test names/results cannot carry
candidates or assertions; no unsupported field reaches final JSON; MEDICATION
candidates must be numeric RxCUIs; DIAGNOSIS candidates are ICD-10 strings.

## 6. Submission ZIP validation behavior

- Exactly one top-level `output/` directory; `1.json`..`100.json`, no missing
  IDs, no extra prediction JSON (extra JSON fails as ERROR).
- Each file UTF-8; root value must be a JSON list; every element passes
  organizer-facing entity validation.
- Unsafe/unexpected paths (absolute, `..`, backslash, drive, deep nesting) fail;
  wrong top-level directory fails; non-prediction metadata files are reported as
  WARNINGs (allowed, not forbidden).
- `write_submission_zip` packages deterministically (fixed timestamps, numeric
  order) — byte-reproducible (asserted in tests). Both directory and ZIP entry
  points provided.

## 7. Offset golden test

`tests/unit/test_offsets.py::test_published_offset_golden_example` uses the
published example: text `amlodipine 10 mg po daily`, position `[58, 83]`,
length 25 (`end - start == len(text)`), asserting
`original_text[58:83] == text` with explicit Unicode code-point indexing.

## 8. Parameter-budget config change

- Per model: `enabled` → `in_profile: true` + `installed: false` (bootstrap:
  nothing installed). Budget sums the `count_status` field (default
  `in_profile`).
- `soft_target_base_params`: 8,850,000,000 → **8,900,000,000**, so the planned
  8,852,000,000 stack no longer triggers a spurious advisory. Hard cap unchanged
  at 9,000,000,000. Config `version: 2`.

## 9. Tests and exact results

```
python3 -m pytest -q     ->  75 passed in 0.25s
ruff check .             ->  All checks passed!
python3 -m mypy          ->  Success: no issues found in 29 source files
```

Manual end-to-end smoke (serialize → 100 files → `output.zip`):
- `to_submission_dict` →
  `{"text": "paracetamol", "type": "THUỐC", "candidates": ["1049640"], "assertions": ["isHistorical"], "position": [0, 11]}`
- English `type` rejected: `True`
- valid zip `ok`: `True` (0 errors)
- corrupt zip (one file removed) `ok`: `False`, first error `submission.file_missing`

All 12 test scenarios requested in the task are covered (label serialization,
English rejection, UTF-8 round-trip, per-type allowed/forbidden fields, `[58,83]`
golden, 100-file pass, missing-file fail, extra-JSON fail, wrong top-level ZIP
dir fail, invalid JSON root fail, adapters excluded, no soft-target advisory).

## 10. Remaining genuine open questions

Unchanged from ADR 0001 §"Still undecided": entity-matching mode; WER
tokenization; ICD-10/RxNorm versions; candidate list limit/ordering; nested
entities; future relation field; rebuild time/memory limits; active-only vs
suppressed concepts. KB-membership validation stays deferred until frozen KB
releases exist.

## 11. PDF confirmation

Both PDFs are **unmodified** (`git status --short -- '*.pdf'` is empty). The
Vietnamese PDF remains **ignored and untracked**
(`git check-ignore -v` → `.gitignore:8`). The English PDF was not modified.

## 12. `git status --short`

```
 M .gitignore
 M README.md
?? AGENTS.md
?? CLAUDE.md
?? audits/
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

(The entire bootstrap + these corrections remain **uncommitted**; nothing has
been committed yet, so new files show as untracked directories.)

## 13. Readiness for the first bootstrap commit

**Ready.** The organizer contracts flagged for correction are now encoded and
tested; static checks (pytest/ruff/mypy) are green; both PDFs are untouched and
the Vietnamese PDF stays ignored. Recommended commit contents: everything in the
`git status` above (code, configs, docs, tests, audits) — excluding the ignored
artifact roots' contents. Commit is left to a human reviewer (not performed
automatically).

## 14. Recommended next milestone

Proceed with **Phase 0 completion — the local metric evaluator** (text WER 30%,
assertion Jaccard 30%, candidate Jaccard 40%, wrong-type double penalty) plus the
per-sample audit/replay export, operating over the organizer-facing serialization
now in place. Only then begin L1 Document Intelligence (reversible normalization
+ `OffsetAlignment` construction).
