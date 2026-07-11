# data/

Root for all datasets and knowledge-base snapshots. **Everything under this
directory except READMEs/.gitkeep is git-ignored.** Never commit raw or private
competition data, organizer gold, or KB snapshots.

## ⚠️ Competition reality — the organizer test set has NO ground truth

The public workflow is: **organizer test inputs → predictions → `output.zip` →
leaderboard → score**. There are **no** hidden test labels. Never create,
reconstruct, infer, leak, or fabricate organizer-test labels, and never call
organizer-test predictions "locally evaluated". Organizer test **inputs** and
internal **labeled** data use **separate namespaces** (below).

## Trust-level namespaces (populated locally, not committed)

```
data/
├── organizer_test/       # INPUT ONLY — organizer competition inputs; never labeled
├── dev_gold/             # GOLD: human-reviewed, adjudicated team ground truth
├── dev_silver/           # SILVER: weak/pseudo/LLM labels (lower trust)
├── synthetic/            # SYNTHETIC: controlled generated examples
├── external_permitted/   # EXTERNAL_PERMITTED: public data allowed by the rules
└── kb/
    ├── icd10/            # frozen WHO ICD-10 snapshot (exact organizer version + hash)
    └── rxnorm/           # frozen NLM RxNorm release (exact organizer release + hash)
```

Each labeled dataset carries a `manifest.yaml` with `provenance` and `labeled`.
A manifest that includes organizer test inputs **must** set `labeled: false`; a
manifest declaring `provenance: ORGANIZER_TEST` with `labeled: true` is rejected
by `mednorm_vi.annotation.validation`. See
[`docs/annotation/GOLD_SILVER_POLICY.md`](../docs/annotation/GOLD_SILVER_POLICY.md).
Only the provisional local evaluator's labeled provenances (GOLD/SILVER/
SYNTHETIC/ORGANIZER_PUBLISHED_EXAMPLE/EXTERNAL_PERMITTED) may be evaluated.

## Rules (see the architecture spec, sections 4, 14, 15.2, 21)

- **Freeze KB snapshots** to the exact organizer-provided version and record the
  hash in `configs/base.yaml` (`knowledge_bases`). The ICD-10 version and RxNorm
  release are currently **UNKNOWN** — do not guess (see `docs/decisions/0001`).
- **Preserve exact offsets** in every derived/synthetic example:
  `original_text[start:end] == entity.text`.
- **No data directly related to the private test** may enter public corpora or
  synthetic generation.
- Group-split by template/source/document so synthetic variants cannot cross CV
  folds.

Small, synthetic, non-sensitive fixtures used by tests live under
`tests/fixtures/`, not here.
