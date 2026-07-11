# Audit 0005 — Remove AGENTS.md from repo; move architecture PDFs into docs/

- **Date:** 2026-07-11
- **Author:** Claude (AI agent), for human review
- **Change type:** Repository housekeeping / file relocation. No code, model, or
  architecture-behavior changes.
- **Supersedes:** Audit 0004's decision to make `AGENTS.md` the canonical
  in-repo rules file. The owner does not want AI-assistant rule files tracked at
  all; the rules now live locally under `.claude/` (git-ignored).

## 1. Objective and scope

At the owner's request: (a) remove `AGENTS.md` from the repository, keeping the
rules locally under the git-ignored `.claude/`; (b) move both architecture PDFs
from the repo root into `docs/`.

## 2. Files created

- `.claude/AGENTS.md` — local, **git-ignored** copy of the repository rules
  (preserved so agents/humans still have them locally; not committed).
- `docs/audits/0005-remove-agents-md-move-pdfs-to-docs.md` (this file).

## 3. Files removed / moved / modified

- **Removed (staged deletion):** `AGENTS.md` (`git rm -f`). A local copy remains
  at `.claude/AGENTS.md` (ignored).
- **Moved (tracked rename, content unchanged):**
  `MedNorm-VI_Architecture.pdf` → `docs/MedNorm-VI_Architecture.pdf`
  (`git mv`; shows as `R` 100% similarity).
- **Moved (untracked, stays ignored):**
  `MedNorm-VI_Architecture_Vietnamese.pdf` → `docs/…` (bare-filename ignore
  pattern matches it in any directory, so it remains untracked).
- **Reference updates** (English PDF path `→ docs/…`):
  `README.md`, `configs/base.yaml`, `src/mednorm_vi/__init__.py`,
  `src/mednorm_vi/schemas/constants.py`,
  `docs/architecture/{ARCHITECTURE_DIGEST,CASE_ROUTING,LAYER_CONTRACTS}.md`.
- **`AGENTS.md` reference removal:** `README.md` (now points to
  `docs/architecture/` + `docs/audits/`); `LAYER_CONTRACTS.md` (dropped the file
  pointer).
- **`configs/base.yaml`:** `paths.audits: audits` → `docs/audits` (matches the
  Audit-0003 relocation).
- **`.gitignore`:** the "AI-assistant local files" block now also ignores
  `AGENTS.md` and `**/AGENTS.md` (alongside the CLAUDE entries and `.claude/`),
  so neither file can be re-committed; the rules survive only under ignored
  `.claude/`.

## 4. Rules preservation

No governance content is lost: `.claude/AGENTS.md` holds the full ruleset
locally. Because it is git-ignored, it is intentionally not part of the tracked
repository, per the owner's decision.

## 5. Tests and validation commands run

```
python3 -m pytest -q   ->  159 passed
ruff check .           ->  All checks passed!
python3 -m mypy        ->  Success: no issues found in 59 source files
git diff --check       ->  clean
```

Ignore/tracking verified via `git check-ignore`:
- root `AGENTS.md` → ignored; `.claude/AGENTS.md` → ignored (local-only);
- `docs/MedNorm-VI_Architecture.pdf` → tracked; `docs/MedNorm-VI_Architecture_Vietnamese.pdf` → ignored.

## 6. Items intentionally deferred

- Historical audits (0001–0004) still mention `CLAUDE.md`/`AGENTS.md` as records
  of past state; not rewritten (audit history is append-only).

## 7. Risks and open questions

- The English PDF move is a pure rename (content identical); deletions/renames
  are **staged** and take effect on the owner's next commit.
- The Vietnamese PDF was relocated locally; it remains untracked/ignored as
  before.

## 8. No models/datasets/KB indices downloaded

Confirmed — only file moves, docs, config, and ignore edits.

## 9. PDF confirmation

Neither PDF's **content** was modified. The English PDF was relocated via
`git mv` (rename only); the Vietnamese PDF was moved on disk and remains
ignored/untracked.

## 10. `git status --short`

```
 M .gitignore
D  AGENTS.md
D  CLAUDE.md
 M README.md
 D audits/0001-repository-bootstrap.md
 D audits/0002-organizer-contract-corrections.md
 M configs/base.yaml
 M data/README.md
R  MedNorm-VI_Architecture.pdf -> docs/MedNorm-VI_Architecture.pdf
 M docs/architecture/ARCHITECTURE_DIGEST.md
 M docs/architecture/CASE_ROUTING.md
 M docs/architecture/LAYER_CONTRACTS.md
 M docs/decisions/0001-open-questions-pending-organizer-data.md
 M src/mednorm_vi/__init__.py
 M src/mednorm_vi/schemas/constants.py
?? configs/annotation/  configs/evaluation/  configs/experiments/
?? docs/annotation/  docs/audits/  docs/evaluation/  docs/experiments/
?? experiments/  reports/
?? src/mednorm_vi/annotation/  src/mednorm_vi/evaluation/  src/mednorm_vi/experiments/
?? tests/fixtures/annotation/  tests/fixtures/evaluation/  tests/fixtures/experiments/
?? tests/integration/test_evaluator_cli.py  …_full_synthetic_evaluation.py  …_leaderboard_workflow.py
?? tests/unit/test_aggregation.py …_annotation_agreement.py …_annotation_validation.py
?? tests/unit/test_diagnostics.py …_experiment_registry.py …_jaccard.py …_matching.py …_replay.py …_wer.py
```
(`D CLAUDE.md`, `D AGENTS.md` = staged deletions; `R …pdf` = tracked rename.
`.claude/AGENTS.md` and the moved Vietnamese PDF are ignored and do not appear.
Everything remains uncommitted, left for human review.)

## 11. Recommended next milestone

Unchanged — **Phase 1: L1 Document Intelligence**.
