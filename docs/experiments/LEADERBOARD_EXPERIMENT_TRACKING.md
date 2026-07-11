# Leaderboard experiment tracking

A **local**, diff-friendly experiment registry. It never scrapes the leaderboard
or calls any external service — leaderboard results are entered **manually** by
the team, and never replace detailed local error analysis.

## Why

The organizer leaderboard returns a single number per submission and no ground
truth. To learn anything, we must connect each leaderboard number to a fully
recorded local experiment: code/config/model/data versions, output hashes, and
**local** provisional scores (kept separate by trust level).

## What each experiment records

Identity & provenance: `experiment_id`, title, description, creation time, Git
commit + dirty-tree flag, config path + hash, model profile, model/adapter/data/
KB versions, code version, seed, thresholds, decoding settings.

Outputs: output directory, `output.zip` hash, per-prediction-file hashes.

Scores (kept separate):
- **local gold**, **local silver**, **local synthetic** (from the provisional
  evaluator);
- **leaderboard** score + submission id + timestamp + component scores (manual).

Narrative: change summary, hypothesis, observed result, conclusion, parent
experiment id, tags, notes.

Records are stored as sorted-key JSON under `experiments/registry/EXP-NNNN.json`
so Git diffs are clean. Generated predictions/outputs stay git-ignored.

## CLI

```bash
python -m mednorm_vi.experiments.cli create --title "baseline deterministic"
python -m mednorm_vi.experiments.cli attach-output --experiment EXP-0001 --zip output.zip
python -m mednorm_vi.experiments.cli record-local --experiment EXP-0001 --kind gold --score 0.83
python -m mednorm_vi.experiments.cli record-leaderboard --experiment EXP-0001 \
    --score 0.71234 --submission-id optional-id
python -m mednorm_vi.experiments.cli show --experiment EXP-0001
python -m mednorm_vi.experiments.cli compare EXP-0001 EXP-0002
```

- Experiment ids are allocated deterministically (`EXP-0001`, `EXP-0002`, …).
- `create` refuses to overwrite an existing id.
- `attach-output` hashes the ZIP and each prediction file; the hash changes when
  the output changes.
- `record-local` keeps gold/silver/synthetic scores distinct.
- `record-leaderboard` stores a **manually** supplied number.

## Discipline

- The leaderboard score is one bit of signal; **local diagnostics drive fixes**.
- Always record the hypothesis before a run and the conclusion after.
- Never enter organizer test data as labeled; never auto-fetch leaderboard scores.
