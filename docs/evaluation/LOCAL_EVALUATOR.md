# PROVISIONAL LOCAL EVALUATOR

> **The organizer does NOT provide ground-truth labels for the competition test
> set.** The public workflow is: organizer test inputs → MedNorm-VI predictions
> → `output.zip` → leaderboard submission → leaderboard score. There is **no
> hidden test label file** to evaluate against, and none may ever be created,
> reconstructed, inferred, or fabricated.

This is a **provisional** local evaluator — an engineering aid, **not** an
official evaluator clone. It approximates the published metric so we can compare
models and analyze errors on **team-owned labeled data**.

## Intended datasets (labeled provenance only)

- **GOLD** — human-reviewed, adjudicated team ground truth (model selection).
- **SILVER** — weak/pseudo/LLM labels (lower trust; flagged in reports).
- **SYNTHETIC** — controlled generated data (unit/stress tests).
- **ORGANIZER_PUBLISHED_EXAMPLE** — examples where the organizer published both
  input and gold output.
- **EXTERNAL_PERMITTED** — public data allowed by competition rules.

`ORGANIZER_TEST` is **never** a labeled provenance. Organizer test inputs live in
`data/organizer_test/` (input only) and are namespaced apart from labeled data.
A dataset manifest that declares `provenance: ORGANIZER_TEST` with
`labeled: true` is rejected.

## CLI

```bash
python -m mednorm_vi.evaluation.cli \
  --ground-truth data/dev_gold/example \
  --predictions outputs/example \
  --config configs/evaluation/provisional_v1.yaml \
  --report-dir reports/example
```

Behavior: prints the `PROVISIONAL LOCAL EVALUATOR` banner; resolves and validates
ground-truth provenance (from `manifest.yaml` or `--provenance`); validates
predictions strictly as organizer submission entities; **fails fast** (nonzero
exit) on malformed inputs or organizer-test-as-labels; permits low-quality but
structurally valid predictions.

## Outputs (in `--report-dir`)

`summary.json`, `per_document.json`, `entity_pairs.jsonl`,
`unmatched_entities.jsonl`, `diagnostics.jsonl`, `config_snapshot.json`,
`replay_manifest.json`, and a self-contained `report.html`. All JSON is UTF-8
(`ensure_ascii=false`). The data reports are deterministic; only the replay
manifest and the HTML banner carry a timestamp.

## Interpretation

- The **final provisional score** is `0.3·text + 0.3·assertions + 0.4·candidates`
  under the configured (provisional) matching, tokenization, and aggregation.
- Read it as a **relative** signal for model comparison and error analysis, not
  as a leaderboard prediction.
- Use the diagnostics and per-type / per-case breakdowns to drive fixes.

## Limitations

- Entity matching, WER tokenization, clipping, and aggregation are **provisional**
  (organizer implementation unpublished). See `METRIC_ASSUMPTIONS.md`.
- KB-membership of candidate codes is **not** checked (no frozen KB yet).
- Never treat this score as equivalent to the organizer's private evaluator.
