# Audit 0013 — Repository-Wide Architecture, Data, Naming, and Hygiene Review

- **Date:** 2026-07-18
- **Author:** Claude (AI agent), for human review
- **Change type:** Consolidated review + safe cleanup milestone. **No commit performed. No push.**
- **Spec reference:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (read in full: L1–L9, Data
  Engine §14, S0–S6 §15, 9B budget §17, offline/fail-fast §16, repo contracts §19).

This audit reviews the entire repository against the architecture PDF, verifies the
implemented layers, reviews all local data/KB resources, standardizes/validates naming,
removes objectively-safe caches and VCS/OS artifacts, repairs manifest fingerprints, and
completes the VietMed-NER row-level review that Audit 0012 had to defer. No models were
trained, no submission was generated, no network dataset/KB was (re)downloaded, and the
architecture PDF was not modified.

## 1. Initial Git state

```text
pwd    : /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch : main
HEAD   : 61fe5458410779ee1ca6077674a44b72b236e9a4  feat: govern public medical NER datasets
status : clean
diff --check : clean
tracked files : 427
```

## 2. Recovery point (destructive-change safety)

Working tree was clean, so a local recovery branch was created (not switched, not pushed,
did not overwrite an existing name):

```text
backup/pre-audit-0013 -> 61fe5458410779ee1ca6077674a44b72b236e9a4
```

Recovery SHA: **`61fe5458410779ee1ca6077674a44b72b236e9a4`**.

## 3. Repository size and file counts (before cleanup)

```text
files (excl. parent .git) : 1018
directories               : 248
total bytes               : 1,595,281,585  (~1.5 GiB, dominated by 893 MB VietMed-NER parquet)
tracked files             : 427   tracked bytes : 2,652,822
ignored bytes             : 1,592,628,763
nested .git dirs          : 3 (phoner_covid19, vimedner, vimq)
.DS_Store                 : 2   stale zero-byte .lock : 5   python/tool caches : 295 files
duplicate groups          : 26 (all inside nested .git hook samples / intentional test fixtures)
```

Reports (ignored) written under `reports/repository_review/`:
`repository_inventory.json`, `duplicate_files.json`, `import_graph.json`,
`data_catalog.json`, `vietmed_ner_rows.json`, `architecture_traceability.json`,
`naming_review.json`, `path_reference_inventory.json`, `deletion_ledger.json`,
`final_cleanup_summary.json`.

## 4. Architecture traceability matrix

Verified via an AST import graph, `src/mednorm_vi/inference/pipeline.py` wiring, CLI
execution, `pytest` (479 passed), and the budget CLI. Full detail in
`reports/repository_review/architecture_traceability.json`.

| Layer | Package (runtime) | Status |
| --- | --- | --- |
| L1 Document Intelligence | `document_intelligence` | IMPLEMENTED_AND_TESTED |
| L2 Case Router | `case_router` + `configs/routes.yaml` (via `deterministic_baseline`) | IMPLEMENTED_AND_TESTED |
| L3 Mention Factory | `mention_factory` (medication/lab) + `adapters.py` | PARTIAL (deterministic tested; neural adapters AWAITING_CHECKPOINT) |
| L4 Boundary & Type Resolver | `resolution` | IMPLEMENTED_AND_TESTED (learned ranker awaits training) |
| L5A Assertion Hydra | `specialists.assertion` | PARTIAL (deterministic tested; neural head AWAITING_CHECKPOINT) |
| L5B ICD-10 Super Linker | `linking.icd10` + `kb.indexing` | PARTIAL (sparse/exact tested; dense/reranker/calibration AWAITING_CHECKPOINT) |
| L5C RxNorm Super Linker | `linking.rxnorm` + `kb.indexing` | PARTIAL (sparse/structured tested; dense AWAITING_CHECKPOINT; RXNSTY AWAITING_RESOURCE) |
| L6 Clinical Evidence Graph | `evidence_graph` | IMPLEMENTED_AND_TESTED |
| L7 Confidence Cascade | `confidence_cascade` | PARTIAL (deterministic gates tested; Qwen critic/adjudicator AWAITING_CHECKPOINT) |
| L8 Metric-aware Set Decoder | `metric_decoder` | IMPLEMENTED_AND_TESTED |
| L9 Validation & Packaging | `validator` + `inference.serialization/packaging` | IMPLEMENTED_AND_TESTED |
| Data Engine (§14) | `data_engine` | PARTIAL (intake/folds/leakage tested; no governed corpora built) |
| S0–S6 Training (§15) | `training` + 6 notebooks | CONTRACT_ONLY (dry-run plan; notebooks are thin scaffolds; no training) |
| Model registry / 9B (§17) | `model_registry` | IMPLEMENTED_AND_TESTED |
| Round-2 comparison | `round2` | IMPLEMENTED (CLI-driven) |

Budget CLI (verified today): deterministic=0, specialist=137,000,000, Safe=410,000,000,
**full=8,408,616,512 (`within_9b=true`)**. `full --require-local-paths` reports missing
checkpoints and does not fabricate predictions (fail-fast preserved).

## 5. Contract-only / stub / partial components discovered

- `mednorm_vi.boundary_type` — L4 bootstrap **CONTRACT_ONLY** stub (`raise
  NotImplementedError`). Real L4 is `mednorm_vi.resolution`. Referenced by name in
  `configs/routes.yaml` and `docs/architecture/CASE_ROUTING.md` → **REVIEW_REQUIRED, not
  deleted.**
- `mednorm_vi.specialists.icd`, `mednorm_vi.specialists.rxnorm` — **CONTRACT_ONLY** stubs.
  Real linkers are `mednorm_vi.linking.icd10` / `mednorm_vi.linking.rxnorm` + `kb`.
  Referenced in `routes.yaml`/`CASE_ROUTING.md` → **REVIEW_REQUIRED, not deleted.**
- `mednorm_vi.specialists.assertion` — **active** (`resolve_assertions` is runtime-imported);
  its compatibility wrapper raises `NotImplementedError` only when a trained head is
  required (legitimate fail-fast).
- `mednorm_vi.assertion_hydra` — present but not runtime-imported (superseded by
  `specialists.assertion`); role uncertain → not deleted.
- `mention_factory.adapters`, `data_engine.{weak_labeling,synthetic,teacher,review}`,
  `training`, `round2`, `model_registry` — not runtime-imported by the inference pipeline;
  they are CLI/notebook/test-driven surfaces, IMPLEMENTED_AWAITING_CHECKPOINT or
  intentionally deferred, **not dead code.**

`NotImplementedError` count in `src`: 2 (both legitimate fail-fast). TODO/FIXME/XXX/HACK
comments in `src`: 0.

## 6. Source-code / package review

No tracked Python module was deleted. Reference analysis (AST import graph +
config/doc/test/notebook grep) showed every "unreferenced-in-`src`" module is either an
entrypoint (CLI), a test/notebook target, or a spec-named contract stub referenced by
config/docs. No circular imports, hard-coded absolute paths, secrets, or network calls
were found in runtime code. Static checks (below) are clean.

## 7. Naming policy and migrations

Policy documented in `docs/repository-layout.md` and `naming_review.json`. Findings: no
ambiguous names, no spaces/uppercase module paths, snapshot dirs compliant (RxNorm ISO
date; ICD regulation id). **Zero renames performed** — the repository already conforms.
One real inconsistency (spec/route names vs implemented package names) is a design
decision with wide blast radius and is **deferred to human decision**, not refactored.

## 8. Complete data catalog

Payload vs checkout fingerprints computed for all datasets
(`data_catalog.json`). All four public-NER manifest `tree_hash_sha256` values were
**checkout-tree hashes that folded in `.git`/`.cache`**; they reproduced exactly, and a
canonical **payload** tree hash (excluding `.git`, `.cache`, `.DS_Store`, `*.lock`) was
computed and added to each manifest without overwriting the previous field (§8 of task).

### ICD-10 (`data/external/icd10_vi/tt06-2026-official/`)
Two source PDFs, SHA-256 match the manifest byte-for-byte (`06-byt.pdf` 598,417 B;
`06-byt-kem.pdf` 21,988,472 B) — **preserved, no OCR**. Derived snapshot present under
`data/derived/icd10_vi/tt06-2026/`. Note: the ICD manifest `extracted_snapshot.tree_hash`
(`ad13c9db…`) was produced by a **different** tree-hasher than the public-NER one, so it
does not reproduce with the public-NER algorithm even though the two file bytes/counts
match; recorded as a hashing-convention difference, not a data discrepancy. License
`REVIEW_REQUIRED`.

### RxNorm (`data/external/rxnorm/prescribable-2026-07-06/`)
16 files, tree hash reproduces manifest exactly; payload == checkout (no incidental files).
`RXNCONSO.RRF`, `RXNREL.RRF`, `RXNSAT.RRF` present; **`RXNSTY.RRF` absent** (semantic types
= 0); **no ZIP present** (deleted before manifest completion). Full RxNorm still awaiting
UMLS approval. License `REVIEW_REQUIRED`.

### Four public NER datasets
Re-reviewed from the filesystem. Checkout hashes reproduce all four manifests exactly.

| Dataset | Payload files/bytes | Canonical payload hash | Governance |
| --- | --- | --- | --- |
| phoner_covid19 | 14 / 18,111,809 | `eeae88fd…a9a4d9` | REVIEW_REQUIRED |
| vimedner | 12 / 5,946,942 | `abce130a…57ff6a` | REVIEW_REQUIRED |
| vimq | 19 / 3,228,352 | `26090803…48cf8090` | REVIEW_REQUIRED |
| vietmed_ner | 5 / 893,513,070 | `c4d07b72…01f6a6` | REVIEW_REQUIRED |

### Organizer data
`data/organizer_test/input/` present and ignored (input only, never labeled). Not copied
into any tracked file. No organizer predictions or `output.zip` were produced.

## 9. VietMed-NER row-level review (completes Audit 0012)

Performed with an **isolated review-only** venv at
`/tmp/mednorm-repository-review-venv` (pyarrow 25.0.0), outside the repo and **not** added
to runtime deps. Aggregate-only; no raw examples stored.

```text
rows        : train 4616, validation 1154, test 3497  (total 9267 — matches 0012 footer)
schema      : words[str], tags[int64], labels[str,BIO], text[str], audio[struct], duration[double]
integrity   : words/tags length mismatches 0; empty texts 0; missing word/tag 0
labels      : 37 BIO tags = O + B-/I- for 18 entity types
              AGE DATETIME DIAGNOSTICS DISEASESYMTOM DRUGCHEMICAL FOODDRINK GENDER LOCATION
              MEDDEVICETECHNIQUE OCCUPATION ORGAN ORGANIZATION PERSONALCARE PREVENTIVEMED
              SURGERY TRANSPORTATION TREATMENT UNITCALIBRATOR
label_inventory_hash : f1f48484fbe3addcfd79f341b9b0e4996fa035cda67ce2117081449be9c7ee06
leakage (exact text) : train↔val 2, train↔test 0, val↔test 0; duplicate text rows: test 1
```

The VietMed manifest was updated with these supported values (`label_inventory_hash`,
`original_labels`, `original_label_scheme`, `row_counts`, `row_level_integrity`,
`cross_split_exact_text_leakage`) and a first-pass reviewable `label_mappings` block.
Governance unchanged (`REVIEW_REQUIRED`).

## 10. Naming/migration table

None. No file or directory was renamed or moved. See `path_reference_inventory.json`.

## 11. Manifest / config / reference updates

Modified (tracked): four public-NER manifests
(`phoner-covid19`, `vimedner`, `vimq`, `vietmed-ner`) — added
`tree_hash_scope`, `canonical_payload_tree_hash_sha256`, `payload_file_count`,
`payload_total_bytes`, `payload_excludes`; VietMed additionally received the row-level
fields above. All six manifests re-validated OK (only pre-existing REVIEW_REQUIRED /
archive-absent warnings). No config path references broke (no renames).

## 12–13. Files deleted, bytes removed, duplicates/caches/locks (SAFE_DELETE only)

Ledger written **before** deletion (`deletion_ledger.json`). Every target was verified
untracked; nothing tracked was removed.

```text
category      files   bytes
tool_cache       35   15,443,103   (.mypy_cache/.pytest_cache/.ruff_cache — reproducible)
nested_git       87    5,658,055   (3 dataset .git dirs — provenance in manifests)
python_cache    268    2,292,753   (__pycache__/*.pyc — reproducible)
os_artifact       2       14,344   (.DS_Store)
hf_cache          8        1,966   (VietMed HF download metadata)
stale_lock        5            0   (zero-byte HF .lock, no process)
TOTAL           405   23,410,221   (~22.3 MiB reclaimed)
```

## 14. Nested Git / cache handling

The three dataset `.git` dirs and the VietMed `.cache/huggingface` were removed only after
confirming all conditions: (a) `source_revision` recorded in the tracked manifest; (b)
canonical **payload** tree hash now recorded; (c) not submodules/gitlinks (independent
shallow checkouts, confirmed in Audit 0012); (d) no future local git op required; (e)
payload files untouched. **Verified after deletion:** all four payload tree hashes match
their manifest `canonical_payload_tree_hash_sha256` byte-for-byte.

## 15. Tracked/ignored verification

`git ls-files data/external/public_ner` → empty. No raw payload, nested `.git`, HF cache,
Parquet, RRF, index, checkpoint, output zip, or generated report is tracked. No secrets.
Architecture PDF has no diff. No `CLAUDE.md`/`AGENTS.md`/tracked `.claude/` created.
`.gitignore` reviewed: correctly ignores caches, model weights, raw data, indices,
reports, outputs, and re-includes only tracked manifests/templates — no changes needed.

## 16. Notebook review

All six notebooks are valid JSON, contain no secrets and no hardcoded local absolute
paths, and use a configurable Google-Drive mount. Each is a **thin scaffold** (2 code
cells: Drive mount + one package CLI call), consistent with the CONTRACT_ONLY S0–S6 status
— they do not fabricate training. One expected-input path `data/manual_gold/text`
(DataEngine notebook) is not present locally; it matches `data_engine.adapters.load_manual_gold`
and is a Colab-populated input, not a broken reference.

## 17. Tests, static checks, deterministic checks

```text
env PYTHONPATH=src python3 -m pytest -q            -> 479 passed (before AND after cleanup)
ruff check .                                        -> All checks passed!
env PYTHONPATH=src python3 -m mypy                  -> Success: no issues found in 213 source files
env PYTHONPATH=src python3 -m compileall -q src     -> clean
git diff --check                                    -> clean
phase1c doctor                                      -> green (network access: none)
model_registry budget (det/spec/Safe/full)         -> within_9b=true; full fail-fast on missing checkpoints
validate-resource-manifest (all 6)                  -> OK (only REVIEW_REQUIRED / archive-absent warnings)
```

Deterministic check: all four dataset payload tree hashes recomputed post-cleanup and
matched their recorded canonical hashes exactly. No organizer predictions or `output.zip`
were generated.

## 18. Exact repository state after cleanup

```text
branch : main   HEAD : 61fe545 (unchanged)   recovery : backup/pre-audit-0013 -> 61fe545
tracked files : 427 (unchanged; only 4 modified, 0 added, 0 deleted tracked)
git status --short:
  M data/manifests/phoner-covid19-public-ner.yaml
  M data/manifests/vietmed-ner-public-ner.yaml
  M data/manifests/vimedner-public-ner.yaml
  M data/manifests/vimq-public-ner.yaml
untracked additions: docs/repository-layout.md, docs/audits/0013-*.md
ignored additions  : reports/repository_review/*.json
```

## 19. Remaining missing checkpoints / resource / governance blockers

- **Checkpoints (all absent, by design):** mention (ViHealthBERT, PhoBERT-W2NER, XLM-R-MRC,
  GLiNER), assertion head, ICD/RxNorm dense retrievers, cross-encoder reranker, Qwen LoRA
  critic/adjudicator, calibration. Full mode fails fast until populated.
- **Resource:** RxNorm Full (and `RXNSTY.RRF`) awaiting UMLS approval; ICD English labels /
  aliases not populated by the PDF conversion; organizer-exact ICD/RxNorm versions unresolved.
- **Governance:** ICD, RxNorm, and all four public NER datasets remain `REVIEW_REQUIRED` —
  not training-eligible until human license/reuse review.

## 20. Final classification matrix

| Classification | Components |
| --- | --- |
| Implemented & executable today | L1, L2, L4, L6, L8, L9, model registry/budget, deterministic profile, ICD/RxNorm sparse retrieval, Data Engine intake, validators/doctor |
| Implemented, awaiting checkpoint | L3 neural adapters, L5A assertion head, L5B/L5C dense+reranker+calibration, L7 Qwen critic/adjudicator |
| Implemented, awaiting reviewed data | governed training folds (all 4 public NER + ICD + RxNorm are REVIEW_REQUIRED) |
| Awaiting resource | RxNorm Full / RXNSTY (UMLS approval) |
| Contract only | S0–S6 training runs (notebooks are scaffolds), `boundary_type`, `specialists.icd`, `specialists.rxnorm` |
| Partial | L3, L5A, L5B, L5C, L7, Data Engine |
| Blocked | none technically; governance gates all corpus training |
| Intentionally deferred | package↔spec rename decision; ICD hash-convention unification |
| Removed as obsolete | none (no tracked module removed); only untracked caches/VCS/OS metadata deleted |

**Answers to the required questions:**
- *Full Pipeline v1 genuinely implemented, or interfaces only?* Genuinely implemented as a
  wired L1→L9 pipeline that runs deterministically and fails fast in full mode; neural
  components are real code awaiting checkpoints (not empty interfaces).
- *Which components run today?* See "executable today" row.
- *Which require checkpoints?* See "awaiting checkpoint" row.
- *Which datasets are technically readable?* All four public NER (VietMed now row-verified);
  ICD PDFs + derived snapshot; RxNorm RRFs (no RXNSTY).
- *Which are blocked by governance?* All six resources (REVIEW_REQUIRED).
- *Which generated resources are valid?* ICD derived snapshot + ICD/RxNorm indexes (per
  0011 hashes); VietMed row report; all Audit 0013 reports.
- *Which directories renamed?* None.
- *Which files deleted?* 405 untracked cache/VCS/OS files (22.3 MiB); no tracked file.
- *Bytes reclaimed?* 23,410,221.
- *Working tree ready for human review/commit?* Yes (see §21).

## 21. Readiness

Audit 0013 is **ready for human review**. It is a review + hygiene milestone: no models
trained, no submission produced, no competition-readiness claimed. The only tracked changes
are four additive manifest edits (payload fingerprints + VietMed row-level metadata) plus
this audit and `docs/repository-layout.md`. It is **safe to commit** after human review;
the agent did not commit or push.

**Recommended next action (after BTC publishes the upgraded task/data):** re-fetch the
organizer's updated task spec and data, re-confirm ICD/RxNorm versions and the five
entity/assertion/candidate contracts against the new rulebook, complete human license/reuse
review to lift REVIEW_REQUIRED, then run S0/S1 folds + mention/assertion training in Colab
to produce the first checkpoint manifests. Decide the `resolution/linking` ↔
`boundary_type/specialists` naming direction before wider refactors.
