# experiments/

Local, diff-friendly **experiment registry** for connecting leaderboard results
to fully recorded local runs. See
[`docs/experiments/LEADERBOARD_EXPERIMENT_TRACKING.md`](../docs/experiments/LEADERBOARD_EXPERIMENT_TRACKING.md).

- `registry/EXP-NNNN.json` — one sorted-key JSON record per experiment. These
  **are tracked** in Git (small, diffable metadata: versions, hashes, local +
  leaderboard scores, narrative).
- Generated predictions, `output.zip`, and reports are **not** stored here — they
  stay git-ignored under `outputs/` and `reports/`.

The tracker is **local only**: it never scrapes the leaderboard or calls any
external service. Leaderboard scores are entered manually and never replace local
error analysis. Local gold / silver / synthetic scores are kept separate from the
leaderboard score.

```bash
python -m mednorm_vi.experiments.cli create --title "baseline deterministic"
python -m mednorm_vi.experiments.cli attach-output --experiment EXP-0001 --zip output.zip
python -m mednorm_vi.experiments.cli record-local --experiment EXP-0001 --kind gold --score 0.83
python -m mednorm_vi.experiments.cli record-leaderboard --experiment EXP-0001 --score 0.71234
python -m mednorm_vi.experiments.cli compare EXP-0001 EXP-0002
```
