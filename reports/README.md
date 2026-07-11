# reports/

Output directory for **PROVISIONAL LOCAL EVALUATOR** runs. Generated report
folders are **git-ignored** (only this README and `.gitkeep` are tracked).

Each run's `--report-dir` contains:

- `summary.json` — final + component scores, policies, per-type/per-case, counts.
- `per_document.json` — per-document scores.
- `entity_pairs.jsonl` — every matched pair with WER/assertion/candidate detail.
- `unmatched_entities.jsonl` — missing (GT) and spurious (prediction) slots.
- `diagnostics.jsonl` — structured error categories for analysis.
- `config_snapshot.json` — raw + resolved evaluator config.
- `replay_manifest.json` — input/config/report hashes, env, Git state, timestamp.
- `report.html` — self-contained (no network) human report.

The data reports are deterministic; only `replay_manifest.json` and the HTML
banner carry a timestamp. Reports scored on **team-owned labeled data only** —
the organizer competition test set has no ground truth.
