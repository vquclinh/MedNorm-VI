# docs/audits/

Numbered audit records for MedNorm-VI. **Every meaningful change must create the
next numbered audit Markdown file under `docs/audits/`** (this directory is the
canonical location; the former root-level `audits/` is retired).

## Numbering

- Files are named `NNNN-<slug>.md` with a **sequential four-digit** number:
  `0001-...`, `0002-...`, `0003-...`, and so on.
- Always use the **next available** number.
- **Never overwrite or renumber** an existing audit. History is append-only; to
  revise a finding, add a new audit that supersedes it.

## Required audit sections

Each audit should record, at minimum:

1. Objective and scope.
2. Files created.
3. Files modified.
4. Design/architecture decisions encoded.
5. Contracts/invariants added or changed.
6. Tests and validation commands run, with exact results.
7. Items intentionally deferred.
8. Risks, assumptions, and open questions.
9. Confirmation that no models/datasets/KB indices were downloaded (when relevant).
10. Confirmation that the architecture PDFs were not modified and the Vietnamese
    PDF remains ignored/untracked.
11. `git status --short` snapshot.
12. Recommended next milestone.

(Individual audits may add task-specific sections; the list above is the floor.)

## Purpose & handling

- Audits are **human-review artifacts** — a durable, reviewable trail of what
  changed and why. They are not code and carry no runtime behavior.
- **Agents must not commit automatically.** Leave audits and their accompanying
  changes staged/unstaged for a human to review and commit.
- Only the numbered audit `.md` files are tracked here; run artifacts
  (`*.log`, `*.json`) under this directory are git-ignored.

## Index

- `0001-repository-bootstrap.md`
- `0002-organizer-contract-corrections.md`
- `0003-phase-0_5-validation-annotation-evaluation-experiments.md`
- `0004-remove-claude-md-canonicalize-agents.md`
- `0005-remove-agents-md-move-pdfs-to-docs.md`
- `0006-phase-1a-l1-document-intelligence.md`
- `0007-phase-1b-deterministic-medical-specialists.md`
- `0008-phase-1c-a-policy-resource-kb-foundation.md`
