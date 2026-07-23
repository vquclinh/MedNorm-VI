# MedNorm-VI Repository Layout

Canonical map of the repository as verified in Audit 0013 (2026-07-18). The
authoritative design source remains `docs/MedNorm-VI_Architecture.pdf`; this file
only documents where things live and what is tracked vs ignored.

## Tracked roots (code, config, docs, small fixtures)

| Path | Contents | Tracked? |
| --- | --- | --- |
| `src/mednorm_vi/` | All Python packages (L1–L9, Data Engine, KB, inference, training, registry). `snake_case` modules, `PascalCase` classes. | Yes |
| `tests/` | Unit + integration tests and small synthetic fixtures (`tests/fixtures/`). | Yes |
| `configs/` | Pipeline, model-registry, parameter-budget, route, resource, and label-mapping configs. | Yes |
| `schemas/` | Human-readable resource-manifest schema docs. | Yes |
| `docs/` | Architecture digest, layer contracts, per-module docs, decisions, and `docs/audits/`. | Yes |
| `docs/MedNorm-VI_Architecture.pdf` | **Protected** primary source of truth. Never modified. | Yes |
| `docs/MedNorm-VI_Architecture_Vietnamese.pdf` | Vietnamese copy. | **Ignored** by policy |
| `notebooks/` | Six Colab scaffolding notebooks (S0–S6 + offline inference). | Yes |
| `scripts/` | Helper scripts (currently `README.md`). | Yes |
| `data/manifests/*.yaml` | Resource/KB/NER governance manifests (**metadata only**; raw payloads stay ignored). | Yes |
| `experiments/registry/` | `EXP-*.json` experiment records. | Yes (records only) |
| `pyproject.toml`, `.gitignore`, `README.md` | Project metadata. | Yes |

## Raw external data roots (ignored payloads, described by manifests)

| Path | Contents | Tracked? |
| --- | --- | --- |
| `data/external/icd10_vi/tt06-2026-official/` | Official ICD-10 source PDFs (`06-byt.pdf`, `06-byt-kem.pdf`). **Protected byte-for-byte.** | Ignored |
| `data/external/rxnorm/prescribable-2026-07-06/` | RxNorm Current Prescribable RRF package (`RXNCONSO/RXNREL/RXNSAT.RRF`; **no `RXNSTY.RRF`**; **no ZIP**). `REVIEW_REQUIRED`. | Ignored |
| `data/external/rxnorm/full-2026-07-06/` | RxNorm Full Monthly Release: `archive/` (verified ZIP, preserved) + `raw/` (Full `rrf/` incl. `RXNSTY.RRF`, historical files; bundled `prescribe/rrf/` subset). UMLS-licensed, `REDISTRIBUTION_RESTRICTED`. Added in Audit 0014. | Ignored |
| `data/external/public_ner/{phoner_covid19,vimedner,vimq,vietmed_ner}/` | Four public Vietnamese medical NER datasets. All `REVIEW_REQUIRED` (not training-eligible). | Ignored |
| `data/organizer_test/input/` | **Active** organizer input documents (**input only, never labeled**). Stable path used by all commands; currently holds snapshot `btc-input-turn2-vong1` (Audit 0015). | Ignored |
| `data/organizer_test/snapshots/<id>/input/` | Versioned organizer input snapshots for rollback (`btc-input-turn2-vong1` active; `btc-input-prior-…` preserved). Descriptor: `configs/organizer/active_input.yaml`. | Ignored |
| `data/organizer_test/archives/*.zip` | Original organizer input ZIP archives (provenance). | Ignored |
| `data/external_permitted/`, `data/dev_gold/`, `data/dev_silver/`, `data/synthetic/` | Reserved data roots (`.gitkeep` placeholders). | Ignored |

## Derived / generated roots (ignored, reproducible)

| Path | Contents |
| --- | --- |
| `data/derived/icd10_vi/tt06-2026/` | Deterministic normalized ICD snapshot (CSV + hierarchy + aliases + conversion manifest/report) from the PDF text layer (no OCR). |
| `indices/icd10_vi/`, `indices/rxnorm/` | Generated local retrieval indexes (`index.json`). |
| `reports/` | Generated review/eval reports, incl. `reports/repository_review/` (Audit 0013 artifacts). |
| `outputs/` | Prediction outputs / submission packages (`output.zip`). |
| `models/` | Base models, adapters, checkpoints (`models/checkpoints/full_v1/…` populated by training). |

## Naming rules (enforced)

- Python packages/modules `snake_case`; classes `PascalCase`.
- Dataset directories lowercase `snake_case`.
- Dated snapshots use ISO `YYYY-MM-DD` (e.g. `prescribable-2026-07-06`) or a
  stable regulation id (`tt06-2026`).
- Audit files `NNNN-kebab-case.md` under `docs/audits/`.
- No ambiguous tokens (`new`, `old`, `copy`, `temp`, `final2`, `latest2`).
- Raw / derived / index / report / checkpoint / output roots stay separated.

Known naming inconsistency (deferred to human decision): the route-contract
vocabulary in `configs/routes.yaml` / `docs/architecture/CASE_ROUTING.md` uses the
spec layer names `boundary_type`, `specialists.icd`, `specialists.rxnorm`, while
the runtime pipeline implements those layers under `mednorm_vi.resolution`,
`mednorm_vi.linking.icd10`, `mednorm_vi.linking.rxnorm`. The former are
CONTRACT_ONLY bootstrap stubs. See `reports/repository_review/naming_review.json`.

## Reproducible rebuild commands

```bash
# Derived ICD snapshot from the protected source PDFs (deterministic, no OCR):
env PYTHONPATH=src python3 -m mednorm_vi.kb.icd10.conversion.cli ...

# Local retrieval indexes:
env PYTHONPATH=src python3 -m mednorm_vi.kb.indexing.cli ...

# Readiness / budget / manifest validation:
env PYTHONPATH=src python3 -m mednorm_vi.phase1c_foundation.cli doctor
env PYTHONPATH=src python3 -m mednorm_vi.model_registry.cli --profile full --require-local-paths
env PYTHONPATH=src python3 -m mednorm_vi.phase1c_foundation.cli validate-resource-manifest --manifest data/manifests/<id>.yaml [--ner]
```

## Protected files (never modify/delete)

- `docs/MedNorm-VI_Architecture.pdf`
- Audit history under `docs/audits/`
- ICD source PDFs; RxNorm RRF payloads
- The only copy of any resource; raw dataset payloads
- Tracked manifests; current canonical indexes; any checkpoint files
