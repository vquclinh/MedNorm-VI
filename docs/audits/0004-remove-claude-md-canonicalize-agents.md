# Audit 0004 — Remove CLAUDE.md; make AGENTS.md the canonical rules file

- **Date:** 2026-07-11
- **Author:** Claude (AI agent), for human review
- **Change type:** Repository housekeeping / governance. No code, model, or
  architecture changes.

## 1. Objective and scope

At the repository owner's request, remove `CLAUDE.md` from the project and ensure
nothing CLAUDE-related (`CLAUDE.md`, `.claude/`, etc.) can be committed to GitHub.
The binding repository rules are preserved by promoting `AGENTS.md` to the sole
canonical rules file.

## 2. Files created

- `docs/audits/0004-remove-claude-md-canonicalize-agents.md` (this file).

## 3. Files modified / removed

- **Removed (staged deletion):** `CLAUDE.md` (`git rm -f`).
- `AGENTS.md` — rewrote the header so it is standalone ("canonical, binding rules
  file"); folded in the "Engineering conventions" items that previously only
  lived in `CLAUDE.md` (Python ≥ 3.10 + type hints; no premature model
  frameworks; no model/dataset/index downloads during unrelated work). No rule
  content was lost.
- `README.md` — "See also `CLAUDE.md` / `AGENTS.md`" → "See also `AGENTS.md`".
- `docs/architecture/LAYER_CONTRACTS.md` — "(see `CLAUDE.md`)" → "(see `AGENTS.md`)".
- `.gitignore` — added a "Claude / AI-assistant local files" block:
  `CLAUDE.md`, `CLAUDE.local.md`, `**/CLAUDE.md`, `.claude/`, `.claude.json`.

## 4. Rules preservation

All binding rules (source of truth, non-negotiable invariants, change discipline,
engineering conventions) now live in `AGENTS.md`. Deleting `CLAUDE.md` removed no
governance content. Historical audits (0001–0003) still mention `CLAUDE.md` as a
record of past state; those are not rewritten (audit history is append-only).

## 5. Tests and validation commands run

```
python3 -m pytest -q   ->  159 passed
ruff check .           ->  All checks passed!
python3 -m mypy        ->  Success: no issues found in 59 source files
git diff --check       ->  clean
```

Ignore behavior verified: a stray `CLAUDE.md` and `.claude/settings.json` are both
matched by `git check-ignore`.

## 6. Items intentionally deferred

- No scrubbing of historical `CLAUDE.md` mentions inside prior audit records
  (would falsify history). Can be revisited if the owner explicitly wants it.

## 7. Risks and open questions

- Low risk: `CLAUDE.md` is removed from tracking; the deletion is **staged** and
  takes effect on the owner's next commit. Until committed, GitHub still contains
  the previously committed `CLAUDE.md`.

## 8. No models/datasets/KB indices downloaded

Confirmed — this change touches only docs/config/ignore files.

## 9. PDF confirmation

Both architecture PDFs are **unmodified**; the Vietnamese PDF remains
ignored/untracked. Neither PDF was touched.

## 10. `git status --short`

```
 M .gitignore
 M AGENTS.md
D  CLAUDE.md
 M README.md
 D audits/0001-repository-bootstrap.md
 D audits/0002-organizer-contract-corrections.md
 M data/README.md
 M docs/architecture/LAYER_CONTRACTS.md
 M docs/decisions/0001-open-questions-pending-organizer-data.md
?? configs/annotation/  configs/evaluation/  configs/experiments/
?? docs/annotation/  docs/audits/  docs/evaluation/  docs/experiments/
?? experiments/  reports/
?? src/mednorm_vi/annotation/  src/mednorm_vi/evaluation/  src/mednorm_vi/experiments/
?? tests/fixtures/annotation/  tests/fixtures/evaluation/  tests/fixtures/experiments/
?? tests/integration/test_evaluator_cli.py  …_full_synthetic_evaluation.py  …_leaderboard_workflow.py
?? tests/unit/test_aggregation.py …_annotation_agreement.py …_annotation_validation.py
?? tests/unit/test_diagnostics.py …_experiment_registry.py …_jaccard.py …_matching.py …_replay.py …_wer.py
```
(`D  CLAUDE.md` = staged deletion. Everything else remains uncommitted, left for
human review.)

## 11. Recommended next milestone

Unchanged from Audit 0003 — **Phase 1: L1 Document Intelligence**.
