# Gold / Silver / Synthetic data policy

Defines the trust levels for team-owned labeled data and how each may be used.
Pseudo-labels are **never** described as ground truth.

> **Competition reality.** The organizer does not release ground truth for the
> competition test set. No trust level here refers to organizer test labels,
> because none exist. `ORGANIZER_TEST` is input-only and can never be `labeled`.

## GOLD

- Human-reviewed and guideline-compliant.
- Offset-verified (`original_text[start:end] == text`).
- Adjudicated where annotators disagreed.
- **Suitable for model selection.** Only **ADJUDICATED** (or explicitly approved
  GOLD) records count as final gold by default.

## SILVER

- Weakly labeled: rule-generated, model-generated, or LLM-generated.
- **Not fully human-verified.**
- Useful for **training** and coverage, but **not** trusted as the only evaluation
  source. The evaluator emits a `silver_label_warning` when scoring SILVER data.

## SYNTHETIC

- Generated from controlled templates or ontology records.
- Labels are **known by construction**.
- Useful for **unit tests and controlled stress tests**.
- May **not** represent the real clinical-language distribution.

## EXTERNAL_PERMITTED

- Public data explicitly allowed by competition rules, with a recorded license /
  permission note in the dataset manifest.

## Namespacing & manifests

Datasets are separated by trust level under `data/`:

```
data/organizer_test/      # INPUT ONLY — never labeled
data/dev_gold/            # GOLD
data/dev_silver/          # SILVER
data/synthetic/           # SYNTHETIC
data/external_permitted/  # EXTERNAL_PERMITTED
```

Every dataset carries a manifest (see `experiments`/`annotation` schemas) with
`provenance` and `labeled`. A manifest that includes organizer test inputs
**must** set `labeled: false`. `provenance: ORGANIZER_TEST` with `labeled: true`
is rejected by `mednorm_vi.annotation.validation`.

## Usage matrix

| Trust      | Train | Model selection | Unit/stress tests | Leaderboard precheck |
| ---------- | :---: | :-------------: | :---------------: | :------------------: |
| GOLD       |  ✅   |       ✅        |        ✅         |         ✅           |
| SILVER     |  ✅   |       ⚠️        |        ⚠️         |         ⚠️           |
| SYNTHETIC  |  ✅   |       ❌        |        ✅         |         ⚠️           |

Leaderboard score is never a replacement for local error analysis.
