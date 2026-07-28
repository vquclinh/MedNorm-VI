# Candidate Training and Deployment Selection Policy

Status: ACTIVE (recorded by Audit 0042)

## Principle

The complete architecture in `docs/MedNorm-VI_Architecture.pdf` is preserved. All
seven planned candidate training stages may be implemented and trained. **No
candidate is removed prematurely because the eventual deployment budget is 9B.**

Two inventories are kept distinct:

| Inventory | Contents | 9B gate |
| --- | --- | --- |
| **candidate** | every model trained or evaluated during research | **not applied** |
| **deployment** | only the models one final inference pipeline loads together | **enforced** |

The candidate total may exceed 9B. Spec §17's limit constrains what is submitted,
not what is researched.

## Selection procedure

1. Train all planned candidate components.
2. Validate each component independently on the governed validation split.
3. Run validation ablations:
   - base pipeline;
   - add one component;
   - remove one component;
   - selected complementary ensembles.
4. For each configuration compute: score contribution, parameter contribution,
   inference latency, memory footprint.
5. Shortlist only meaningful configurations — not every combination.
6. Use leaderboard submissions **sparingly**, to avoid overfitting the public
   leaderboard (spec §21 lists synthetic/validation drift as a named risk).
7. Select the final configuration satisfying all of:
   - total deployment parameters <= 9,000,000,000;
   - no external model API;
   - reproducible self-hosted inference.

## Accounting conventions

Reported side by side; the gate uses the conservative figure.

- `total_loaded_parameters` (**gated**) — everything the inference process loads,
  adapters included. A LoRA deployment counts base + adapters.
- `base_only_parameters` — spec §17's convention, which states heads/adapters are
  outside the organizer's base-model budget.

Because the conservative total is always >= the base-only total, passing the
conservative gate necessarily satisfies §17.

Shared weights loaded once are counted once (spec §15.1). Independently loaded
checkpoints are counted in full. A selected component without a **verified
programmatic** count fails closed — published estimates, including spec §17's
approximate ~8.852B table, are planning figures and are never treated as fact.

## Exclusion is recorded, not deleted

A component omitted from the final deployment remains a researched candidate. It
is marked `EXCLUDED_BY_ABLATION` in `configs/models/candidate_model_registry.yaml`
with the evidence that excluded it. It is **never** deleted from the registry or
from the architecture document.
