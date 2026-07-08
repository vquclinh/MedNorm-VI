# outputs/

Root for **prediction outputs and submission packages** (the 100 JSON files and
the final ZIP). Everything under this directory except this README is
git-ignored — **never commit predictions**.

- **Confirmed layout:** `output.zip` → one top-level `output/` directory →
  `1.json` … `100.json`.
- Validated by `mednorm_vi.validator.validate_output_directory` (unpacked) and
  `validate_submission_zip` (packaged), plus organizer-facing entity validation
  (`validate_organizer_document`): Vietnamese `type` labels, per-type fields,
  UTF-8 with `ensure_ascii=false`, list roots, no extra/unsafe paths.
- Fail-fast: if any entity violates `original_text[start:end] == text`, or a code
  is absent from the frozen KB snapshot, stop that file and write an audit log —
  never silently repair output.

KB-membership checking of candidate codes remains deferred until frozen KB
snapshots exist — see `docs/decisions/0001`.
