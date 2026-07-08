# data/

Root for all datasets and knowledge-base snapshots. **Everything under this
directory except this README is git-ignored.** Never commit raw or private
competition data, organizer gold, or KB snapshots.

## Expected layout (populated locally, not committed)

```
data/
├── raw/            # organizer-provided input documents (private)
├── gold/           # manual/organizer gold annotations (private)
├── kb/
│   ├── icd10/      # frozen WHO ICD-10 snapshot (exact organizer version + hash)
│   └── rxnorm/     # frozen NLM RxNorm release (exact organizer release + hash)
├── synthetic/      # controlled synthetic data (must preserve exact spans)
├── weak/           # weak-labeling / teacher-consensus outputs
└── folds/          # group-split CV folds (by template/source/document)
```

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
