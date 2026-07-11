# MedNorm-VI

**Clinical Information Extraction & Medical Knowledge Retrieval for Vietnamese
clinical text** — Viettel AI Race 2026 (*Ontological Reasoning in Medical
Knowledge Retrieval*).

MedNorm-VI is a multi-expert system that converts free-form Vietnamese clinical
notes into structured, offset-preserving entity hypotheses with assertions and
ICD-10 / RxNorm candidate codes.

> **Status: research-foundation stage (Phase 0 + Phase 0.5).** This repository
> contains the typed data contracts, the deterministic validation foundation,
> per-layer module stubs, and — new in Phase 0.5 — a **PROVISIONAL LOCAL
> EVALUATOR**, an **annotation/data-quality** foundation, and **leaderboard
> experiment tracking**. **No models, indices, or datasets are downloaded or
> implemented yet.** See `docs/` and the audit trail in `docs/audits/`.

> ### ⚠️ Competition reality
> The organizer does **not** release ground truth for the competition test set.
> The workflow is: **organizer test inputs → predictions → `output.zip` →
> leaderboard → score**. The local evaluator scores **team-owned labeled data
> only** (gold / silver / synthetic / permitted / organizer-published examples).
> Organizer test inputs are input-only and never labeled.

## Intended order of operations

```
internal annotation
  → local provisional evaluation
  → controlled experiment (tracked)
  → organizer test inference
  → output.zip
  → leaderboard submission
  → manual score recording
```

## What MedNorm-VI solves

Given a clinical document, produce a set of entity hypotheses:

```
H = (start, end, text, type, assertions, candidates)
```

emitted as JSON per document (`text`, `type`, `assertions`, `candidates`,
`position`), where `position` is a half-open `[start, end)` interval into the
original text. The mandatory invariant is:

```
original_text[start:end] == entity.text
```

Scoring (qualification round, 100 documents → 100 JSON files): **candidates 40%,
text 30% (1 − WER), assertions 30%**. A **wrong type is double-penalized**.

Offsets are **confirmed** Python-style, end-exclusive character offsets over
Unicode code points.

## Output entity types (five)

Internal English enums map to **confirmed organizer-facing Vietnamese labels**
that appear in submission JSON:

| Internal enum | Organizer label (JSON `type`) |
| ------------- | ----------------------------- |
| `MEDICATION`  | `THUỐC`                       |
| `DIAGNOSIS`   | `CHẨN_ĐOÁN`                   |
| `SYMPTOM`     | `TRIỆU_CHỨNG`                 |
| `TEST_NAME`   | `TÊN_XÉT_NGHIỆM`             |
| `TEST_RESULT` | `KẾT_QUẢ_XÉT_NGHIỆM`        |

JSON is UTF-8 with `ensure_ascii=false` (Vietnamese characters preserved). The
submission validator rejects English `type` values.

### Per-type output fields (confirmed)

| Organizer type | Fields emitted |
| -------------- | -------------- |
| `THUỐC`, `CHẨN_ĐOÁN` | `text, type, candidates, assertions, position` |
| `TRIỆU_CHỨNG`  | `text, type, assertions, position` |
| `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM` | `text, type, position` |

Symptoms carry no candidates; test names/results carry neither candidates nor
assertions. The serializer emits only the fields allowed for each type.

## Assertion labels (multi-label)

`isNegated` · `isHistorical` · `isFamily`

## Concept linking (candidates)

- **DIAGNOSIS → ICD-10 codes** (WHO ICD-10, frozen to the organizer version).
- **MEDICATION → RxCUIs** (NLM RxNorm, frozen release; SCD/SBD term types).
- Cross-linking is forbidden (no ICD on MEDICATION, no RxNorm on DIAGNOSIS).
- LLMs may only **select** from retrieved candidates — never recall codes from
  memory.

## Nine-layer architecture

| Layer | Name                     | Output                  |
| ----- | ------------------------ | ----------------------- |
| L1    | Document Intelligence    | DocumentGraph           |
| L2    | Case Router (multi-label)| Route tags + priors     |
| L3    | Mention Factory          | Span lattice            |
| L4    | Boundary & Type Resolver | Typed hypotheses        |
| L5    | Specialists (Assertion / ICD / RxNorm) | Evidence bundles |
| L6    | Evidence Graph           | Clinical graph          |
| L7    | Confidence Cascade       | Adjudicated hypotheses  |
| L8    | Metric Decoder           | Optimal prediction set  |
| L9    | Validator                | 100 JSON files + ZIP    |

Details in [`docs/architecture/LAYER_CONTRACTS.md`](docs/architecture/LAYER_CONTRACTS.md).

## Seven routed case families

The **multi-label** Case Router (a segment may carry several at once, e.g.
C1+C4+C6):

- **C1** Medication-list · **C2** Lab semi-structured · **C3** Clinical narrative
- **C4** Assertion-heavy · **C5** Abbreviation/noise · **C6** Ambiguous linking
- **C7** Overlap/duplicates

Details in [`docs/architecture/CASE_ROUTING.md`](docs/architecture/CASE_ROUTING.md)
and declarative config in [`configs/routes.yaml`](configs/routes.yaml).

## Parameter budget (9B base)

The **total parameters across all base models must not exceed 9B**. Adapter /
LoRA / head parameters are **excluded** from that budget, so the physical total
may sit slightly above 9B (~9.1B) and that is accepted. The default planned
"Safe Submission Stack" is ~8.85B base. Model membership in a plan (`in_profile`)
is tracked separately from whether weights are actually downloaded (`installed`);
at the bootstrap stage **nothing is installed**. All models are **self-hosted**;
**no external inference APIs** are used.
See [`configs/parameter_budget.yaml`](configs/parameter_budget.yaml).

## Submission layout (confirmed)

```
output.zip
└── output/
    ├── 1.json
    ├── 2.json
    ├── ...
    └── 100.json
```

Validated by `mednorm_vi.validator.validate_output_directory` (unpacked) and
`validate_submission_zip` (packaged): exactly `1.json`..`100.json`, one top-level
`output/` directory, each file UTF-8 with a JSON list root, and every element
passing organizer-facing entity validation.

## Repository layout

```
configs/    base.yaml, parameter_budget.yaml, routes.yaml
docs/       architecture/ (digest, layer contracts, case routing) + decisions/ (ADRs)
src/mednorm_vi/
            schemas/                typed data contracts
            document_intelligence/  L1   case_router/         L2
            mention_factory/        L3   boundary_type/       L4
            specialists/{assertion,icd,rxnorm}/  L5
            evidence_graph/         L6   confidence_cascade/  L7
            metric_decoder/         L8   validator/           L9 (implemented)
tests/      unit/ integration/ fixtures/
data/ models/ indices/ outputs/ reports/    git-ignored artifact roots (see each README)
docs/audits/  numbered audit records of every meaningful change (see docs/audits/README.md)
```

## Installation

```bash
python -m pip install -e ".[dev]"      # editable install + dev tools
```

Requires Python ≥ 3.10. The bootstrap depends only on `PyYAML` (+ `pytest`,
`ruff`, `mypy` for development). Heavy model frameworks are intentionally not
added yet.

## Phase 0.5 research tooling

Provisional local evaluator (team-owned labeled data only):

```bash
python -m mednorm_vi.evaluation.cli \
  --ground-truth data/dev_gold/example \
  --predictions outputs/example \
  --config configs/evaluation/provisional_v1.yaml \
  --report-dir reports/example
```

Annotation / manifest validation:

```bash
python -m mednorm_vi.annotation.validation \
  --dataset data/dev_gold/example --manifest path/to/manifest.yaml
```

Experiment + leaderboard tracking (local only; scores entered manually):

```bash
python -m mednorm_vi.experiments.cli create --title "baseline deterministic"
python -m mednorm_vi.experiments.cli attach-output --experiment EXP-0001 --zip output.zip
python -m mednorm_vi.experiments.cli record-local --experiment EXP-0001 --kind gold --score 0.83
python -m mednorm_vi.experiments.cli record-leaderboard --experiment EXP-0001 --score 0.71234
python -m mednorm_vi.experiments.cli compare EXP-0001 EXP-0002
```

Docs: [`docs/evaluation/`](docs/evaluation/), [`docs/annotation/`](docs/annotation/),
[`docs/experiments/`](docs/experiments/). Metric confirmed-vs-provisional split:
[`docs/evaluation/METRIC_ASSUMPTIONS.md`](docs/evaluation/METRIC_ASSUMPTIONS.md).

## Test & static checks

```bash
pytest                    # unit + integration tests
ruff check .              # lint
mypy                      # strict type check
```

## Authoritative specification

The complete, authoritative design is the English PDF under `docs/`:
**`docs/MedNorm-VI_Architecture.pdf`** (Super Architecture Specification, v1.1). Read
it before any architecture-sensitive change. A local Vietnamese translation may
exist but is intentionally **untracked** (git-ignored) and must not be committed.

Engineering contracts are documented under [`docs/architecture/`](docs/architecture/);
the change log lives in [`docs/audits/`](docs/audits/).
