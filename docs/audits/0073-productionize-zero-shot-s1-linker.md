# Audit 0073: Productionize the Frozen Zero-Shot S1 Semantic Linker

Milestone: 4N - wire S1 into canonical inference for one controlled public submission
Date: 2026-08-01
Revision: 2 - corrected to route BOTH ontologies (see §0)
Verdict: `SEMANTIC_S1_WIRED_PUBLIC_INFERENCE_BLOCKED_ON_ARTIFACTS`

No clinical text appears in this audit. **`training_performed = false`.**

## 0. Correction in Revision 2

Revision 1 of this audit implemented and claimed:

> "The ICD branch may route to the semantic linker; the RxNorm branch is untouched and cannot
> be routed away."

**That was wrong.** The frozen Audit-0072 S1 is a complete two-ontology system, and its
2,000-query selection benchmark was approximately **197 ICD10 and 1,803 RXNORM** - the
document corpus is 15,308 ICD against 211,949 RxNorm, and the deterministic ontology-sorted
slice inherits that ratio. The frozen metrics (R@1 0.5115, R@10 0.7440, MRR 0.5985) are
therefore roughly **90% RxNorm-driven**. An ICD-only production wiring would reproduce neither
the selected system nor the evidence that selected it.

Revision 2 routes both ontologies through frozen S1. The incorrect claim is removed, and the
test that enforced it (`test_semantic_backend_never_routes_rxnorm`) is deleted and replaced.

## 1. The Scored Baseline's Actual Mode (§11)

Determined from evidence, not from the name of a config key. The scored 11.9188 run's own
manifest records:

```
mode        : specialist
experts ran : ['E3_vihealthbert_span_type']
readiness   : {'status': 'READY', 'errors': [], 'fail_closed': False}
```

**The production path is `--mode specialist`, not `full`.** `full_requires_checkpoints` names
components that are disabled or unavailable, so `full` would change far more than the linker
and would not reproduce the scored path. The ablation therefore **preserves `specialist` mode
and swaps only the linker backend** — and that backend, frozen S1, covers **both ICD and
RxNorm**.

## 2. Exact Delta From Baseline

`configs/pipeline/semantic_s1_0073.yaml` differs from `full_v1.yaml` in **exactly two config
fields**, asserted by test rather than by inspection:

| Field | Baseline | Ablation |
| --- | --- | --- |
| `linker_backend` | `lexical_v3` (default) | `semantic_s1_0072` |
| `semantic_linker` | `{}` | S1 settings block |

`icd_index` stays **competition-v3** — the v3 lexical precision anchor is retained, not
replaced by v4.1. v4.1 enters only as an additional coverage source inside the hybrid union.

Unchanged and verified: E1, E2, E3, L4, assertion logic, entity spans, entity types, decoding
semantics, output schema, dotted ICD serialization. **RxNorm linking does change** — it is
part of frozen S1 — but only its linker; no structured-RxNorm work is introduced.

### Baseline vs frozen S1, per ontology

| | `lexical_v3` (baseline, 11.9188) | `semantic_s1_0072` (frozen S1) |
| --- | --- | --- |
| **ICD** | competition-v3 lexical linker, H0 | v3 Top-20 + v4.1 Top-20 + dense Top-50 → union → Qwen3-Reranker-4B → **H2** |
| **RxNorm** | structured RxNorm linker | lexical Top-20 + dense Top-50 → union → Qwen3-Reranker-4B → **no hierarchy** |
| **NULL** | n/a | shadow only |

H2 is an ICD within-family rule and is **never** applied to RxNorm: `run_hybrid` gates it on
`ontology == "ICD10"`, so the separation is structural, and a test asserts
`hierarchy_applied is False` on the RxNorm path.
`configs/pipeline/full_v1.yaml` is **byte-for-byte unmodified** (`git diff --quiet` clean) and
remains the rollback.

## 3. Selected 0072 Artifacts and Revisions

| Component | Full immutable revision | Parameters |
| --- | --- | ---: |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` | 4,021,774,336 |
| `Qwen/Qwen3-Reranker-4B` | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | 4,021,784,576 |
| ViHealthBERT E3 (unchanged) | `f89e80b461e86f9cfc1c84019bd819830c24b6c5` | 135,002,117 |
| **Total deployed** | | **8,178,561,029** |

0072 results package SHA-256 `c23135508c73b2d80cf14dab9c6993effe18ef3fca54d1d4b0449806d5465b39`.
S2 rejected. Revisions are recorded in `model-manifest.json` on every run and asserted by test
to be 40-character hex.

## 4. Parameter Budget (§5)

8,178,561,029 < 9,000,000,000, headroom 821,438,971. Asserted in
`test_deployed_parameters_stay_under_nine_billion` and cross-checked against the shipped
`artifacts/semantic/0072/deployed_parameter_manifest.json`. No learned component was added
beyond the two S1 models.

## 5. Architecture of the Change

`src/mednorm_vi/linking/semantic/runtime.py` is a `LinkerResult`-shaped wrapper that **reuses**
the 0072 components — `build_pool`, `run_hybrid`, `Qwen3Reranker`, `SemanticQuery` are
imported, never reimplemented.

```
ICD10   mention → v3 Top-20 ∪ v4.1 Top-20 ∪ dense Top-50
                → dedup against the governed KB (nothing outside it survives)
                → Qwen3-Reranker-4B (yes/no logit protocol)
                → H2 within-family arbitration
                → NULL shadow → LinkerResult, dotless codes

RXNORM  mention → lexical Top-20 ∪ dense Top-50
                → dedup against the governed RxNorm KB
                → Qwen3-Reranker-4B
                → (no hierarchy)
                → NULL shadow → LinkerResult
```

Dense retrieval runs **within one ontology's slice** of the frozen 227,257-vector index, so an
ICD mention can never retrieve an RxCUI or vice versa. Document row order is the contract
between `documents.jsonl` and the cached vectors, and a count mismatch fails closed.

`test_production_wrapper_matches_the_0072_hybrid_for_both_ontologies` drives the same inputs
through `build_pool` / `run_hybrid` directly and asserts identical output, proving the
production wrapper does not implement a different algorithm.

## 6. Readiness Contract (§6)

`evaluate_readiness` refuses the run when the backend is `semantic_s1_0072` and anything is
missing. Observed output with no models present:

```
BLOCKED - readiness failed closed:
  - semantic_linker_not_ready:semantic_linker.model_root is unset; set it in the profile,
    pass --semantic-model-root, or export MEDNORM_SEMANTIC_MODEL_ROOT
  - semantic_linker_not_ready:dense index not found: artifacts/semantic/0072/cache/
    S1_doc_vectors.pt. Build it on a GPU host first; inference never encodes 227k documents
    on the fly.
```

There is **no silent fallback to lexical**. A run that claimed semantic but quietly used v3
would make the ablation a lie, so it stops instead. `null_mode` other than `shadow` is also
refused, as this milestone forbids activating the gate.

Model root precedence, with no user-specific path as a repository default:
`--semantic-model-root` > profile `model_root` > `$MEDNORM_SEMANTIC_MODEL_ROOT` > unset → fail.

## 7. Tests (§8)

`tests/unit/test_productionized_semantic_linker_0073.py` — **31 tests, no weights, no CUDA,
no network**:

backend selectable · baseline still resolves to lexical · exactly-two-field delta · unknown
backend refused (5 values) · baseline YAML declares no backend · fails closed on missing
models · every missing piece named · baseline readiness unaffected · non-shadow null refused ·
env override honoured · no user-specific default path · **governed concept ids only** ·
**codes dotless internally** · **shadow NULL does not alter the emitted list** · deterministic
ordering under ties · backend and snapshot recorded · manifest records exact 40-hex revisions ·
parameter budget < 9B · no network symbols in the inference path · script does not submit ·
shipped manifest agreement.

Replacing the deleted `test_semantic_backend_never_routes_rxnorm`:

* `test_semantic_backend_routes_both_ontologies` — ICD **and** RxNorm route through frozen S1
* `test_lexical_backend_leaves_both_ontology_linkers_unchanged`
* `test_semantic_rxnorm_is_governed_by_the_frozen_rxnorm_kb` — snapshot + ontology evidence
* `test_no_out_of_kb_rxcui_can_be_emitted` — an out-of-KB RxCUI is dropped, not emitted
* `test_h2_is_never_applied_to_rxnorm`
* `test_production_wrapper_matches_the_0072_hybrid_for_both_ontologies` — parity regression

### Authoritative Revision-2 counts

| Suite | Result |
| --- | ---: |
| `test_productionized_semantic_linker_0073.py` | **31 passed** |
| Semantic + hierarchy suites (0069-0073) | **135 passed** |
| Targeted `-k "config or pipeline or inference"` | **97 passed** (2,108 deselected) |

These supersede every earlier count in this document.

## 8. Validation

| Gate | Result |
| --- | --- |
| Targeted 0073 tests | **31 passed** |
| Semantic suites 0069-0073 | **135 passed** |
| Targeted config/pipeline/inference | **97 passed** |
| `ruff check .` | clean |
| `ruff format --check` (changed files) | clean |
| `mypy src/mednorm_vi` | no issues, 304 source files |
| `compileall src tests scripts` | clean |
| `git diff --check` | clean |
| Mock smoke test (no models) | baseline `READY`, `semantic_linker_runtime → None`; ablation `NOT_READY` |
| Tracked diff scope | only `inference/config.py` (+35) and `inference/pipeline.py` (+50/-1) |
| Deletions | none |
| Weights / private pack tracked | none |
| Protected artifacts | scored ZIP, ICD v3, E3 checkpoint all unchanged |

`mypy` caught a real defect during implementation: the query encoder was cached via
`object.__setattr__` on a non-frozen dataclass, which type-checks as an undeclared attribute.
It is now a declared field with plain assignment.

## 9. Commands

**Baseline (rollback, reproduces 11.9188):**

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  .venv/bin/python -m mednorm_vi.inference.cli run \
    --input-dir data/organizer_test/input \
    --output-zip runs/<name>/output.zip \
    --config configs/pipeline/full_v1.yaml \
    --mode specialist \
    --run-manifest runs/<name>/run-manifest.json
```

**Semantic S1 ablation (Colab A100/H100) — canonical command.** All three overrides are
required; each flag was verified against the runner's argparse interface:

```bash
PYTHONPATH=src python scripts/run_public_inference_0073.py \
  --input-dir data/organizer_test/input \
  --run-dir runs/semantic_s1_0073_final_dotted \
  --semantic-model-root  /content/drive/MyDrive/MedNorm-VI/0072/models \
  --semantic-doc-vectors /content/drive/MyDrive/MedNorm-VI/0072/cache/S1_doc_vectors.pt \
  --semantic-documents   /content/drive/MyDrive/MedNorm-VI/0072/artifacts/documents.jsonl
```

This runs `specialist` mode with frozen S1 on **both** ontologies.

### Persistent Drive artifacts (§6)

The 36 KB `0072-results.tgz` does **not** contain the dense vectors. The heavy artifacts live
on Drive and are reused in place — nothing is copied or re-encoded:

| Artifact | Path |
| --- | --- |
| Embedding weights | `<0072>/models/emb-4b/` |
| Reranker weights | `<0072>/models/rr-4b/` |
| Frozen dense index | `<0072>/cache/S1_doc_vectors.pt` |
| Document corpus | `<0072>/artifacts/documents.jsonl` (+ `manifest.json`) |

Supplied by explicit CLI overrides — `--semantic-model-root`, `--semantic-doc-vectors`,
`--semantic-documents` — which take precedence over the profile and `$MEDNORM_SEMANTIC_MODEL_ROOT`.
No user-specific path is a repository default.

### Preflight before any model byte loads (§7)

`preflight()` refuses the run unless: `emb-4b/config.json` and `rr-4b/config.json` exist,
`documents.jsonl` and `S1_doc_vectors.pt` exist, the corpus has exactly **227,257** rows with
both ontologies present, the dense index first dimension is **227,257** and equals the corpus
row count, and the 0071 `manifest.json` agrees with the corpus size. Any mismatch is fatal —
a cache built over a different corpus would map every vector to the wrong concept, silently.

The script then runs the canonical CLI, applies the governed `icd_dotted` derivation, validates
exactly 100 files with schema and offset checks, packages `output.zip`, prints its SHA-256, and
**never submits**.

## 10. Offline Evidence — Stated Plainly

The frozen 0072 result is **mixed**, and the mix matters for a candidate-set metric:

| Metric | Lexical baseline | S1 | Delta |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.5455 | 0.5115 | **−0.0340** |
| Recall@10 | 0.5795 | 0.7440 | **+0.1645** |
| MRR | 0.5559 | 0.5985 | +0.0426 |

S1 wins decisively on Recall@10 and MRR and **loses on Recall@1**. Audit 0070 is the standing
precedent: it improved coverage, changed 150 public ICD top-1 decisions, and J_candidates
*fell* 2.8646 → 2.6459. This submission is a genuine experiment, not a safe upgrade, and the
Recall@1 regression is the specific reason it might repeat that outcome.

## 11. Changed Files

Modified (tracked): `src/mednorm_vi/inference/config.py`,
`src/mednorm_vi/inference/pipeline.py`.

New (tracked): `src/mednorm_vi/linking/semantic/runtime.py`,
`configs/pipeline/semantic_s1_0073.yaml`, `scripts/run_public_inference_0073.py`,
`tests/unit/test_productionized_semantic_linker_0073.py`,
`docs/audits/0073-productionize-zero-shot-s1-linker.md`.

Unchanged: `configs/pipeline/full_v1.yaml`, all indices, both checkpoints, the scored ZIP, E4
(still retired), E5/E6/E7 (still false).

## 12. Rollback

Use `--config configs/pipeline/full_v1.yaml --mode specialist`. Nothing else needs reverting:
the baseline profile declares no `linker_backend`, so it resolves to `lexical_v3` by default
and the semantic code is never constructed.

## 13. Remaining Risks

1. **Recall@1 regresses 3.4pp offline.** The most likely failure mode is a J_candidates loss.
2. **Benchmark queries were ontology surface forms, not clinical mentions.** Real mentions
   carry context and noise the 0072 numbers never exercised.
3. **NULL stays shadow**, so the 43/54 false-emission defect is unchanged by this run.
4. **The dense index must match `documents.jsonl`** — now enforced by preflight (227,257 rows,
   matching first dimension, manifest agreement) rather than trusted.
5. **RxNorm is now also semantic.** The baseline's structured RxNorm linker is replaced on this
   path, which is what frozen S1 requires — but it means the submission changes RxNorm
   candidates too, and RxNorm dominates the 0072 evidence.
6. **Runtime is unmeasured.** Reranking ~40-70 candidates per diagnosis mention across 100
   documents on a 4B cross-encoder has no local timing evidence.

## 14. Verdict

The frozen S1 linker is wired into the canonical production path for **both ontologies**, as
the selection evidence requires, as a clean single-variable ablation (linker backend only). It
fails closed when its artifacts are absent, keeps every non-linker behaviour identical, and
stays under the 9B budget.

**Public inference is BLOCKED** until the **complete** frozen artifact set is present on the
run host — by design, not by defect. All five are required, and preflight refuses the run if
any is missing or inconsistent:

| # | Artifact | Path under `<0072>` |
| --- | --- | --- |
| 1 | Embedding weights | `models/emb-4b/` |
| 2 | Reranker weights | `models/rr-4b/` |
| 3 | Document corpus | `artifacts/documents.jsonl` |
| 4 | Corpus manifest | `artifacts/manifest.json` |
| 5 | Frozen dense index | `cache/S1_doc_vectors.pt` |

A partial set does not degrade to lexical; it stops.

**No training or fine-tuning was performed.**
