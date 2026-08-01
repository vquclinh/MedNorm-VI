# Audit 0074: Structured RxNorm Recovery

Milestone: 4O - recover structured drug attributes and exploit them in the RxNorm S1 path
Date: 2026-08-02
Verdict: `STRUCTURED_RXNORM_ACCEPTED_OFFLINE_COMPLETE_SYSTEM_AB_REQUIRED`

No clinical text appears in this audit. **No model was trained, fine-tuned or downloaded.**

## 1. New Scored Baseline (Audit 0073)

Recorded in the governed registry as `EXP-0001`
(`experiments/registry/EXP-0001.json`, `record-leaderboard`, manually entered). The scored ZIP
was **not** modified.

| | Score | WER | J_assertion | J_candidates |
| --- | ---: | ---: | ---: | ---: |
| **0073 frozen S1 (new best)** | **12.4757** | 82.7090 | 18.6187 | **4.2570** |
| 0069/0062 lexical (previous) | 11.9188 | 82.7090 | 18.6187 | 2.8646 |

Submission SHA-256 `0995c7c7a070460a28d76f1dfdd24b3cc0e7d986e90173a583eadf214f551a5c`,
`num_scored = 100`. WER and J_assertion are byte-identical, so the entire +0.5569 is
J_candidates (+48.6% relative) — the single-variable ablation makes that attribution safe.

Worth recording plainly: Audit 0073 flagged S1's offline Recall@1 regression (0.5115 vs
0.5455) as the specific reason the submission might repeat the Audit-0070 outcome. It did not.
Recall@10 (+16.5pp) and MRR dominated. The caution was reasonable on the evidence available;
the evidence was incomplete, not the reasoning.

## 2. A Reader Defect Found First (§ prerequisite)

Before any recovery was possible, `kb/rxnorm/rrf_reader.py` had **ATV and SAB transposed**:

```python
_SAT = {"RXCUI": 0, "ATN": 8, "ATV": 9, "SAB": 10, "SUPPRESS": 11}   # wrong
```

The official RXNSAT layout is `RXCUI|LUI|SUI|RXAUI|STYPE|CODE|ATUI|SATUI|ATN|SAB|ATV|SUPPRESS|CVF`,
and the governed 2026-07-06 file agrees — column 9 holds `RXNORM`/`MTHSPL` (a source
vocabulary) and column 10 holds the value. Read swapped, **every attribute value was a source
abbreviation and every source abbreviation was a value**, so no `SAB == "RXNORM"` filter could
ever match and structured recovery was silently impossible.

Corrected to `{"ATN": 8, "SAB": 9, "ATV": 10}`; the eight existing RxNorm reader/snapshot tests
still pass, and a new test pins the column contract.

## 3. Recovered Fields and Provenance (§1, §2)

Source: `data/external/rxnorm/full-2026-07-06/archive/RxNorm_full_07062026.zip`, streamed
without extraction. Provenance per concept records release, RRF file and the attribute name or
relation. **No external source; nothing inferred.**

**RXNSAT attributes (SAB = RXNORM), governed rows:**

| Attribute | Governed |
| --- | ---: |
| `RXN_AVAILABLE_STRENGTH` | 52,318 |
| `RXTERM_FORM` | 49,187 |
| `RXN_HUMAN_DRUG` | 23,879 |
| `RXN_STRENGTH` | 23,418 |
| `RXN_BOSS_STRENGTH_{NUM,DENOM}_{VALUE,UNIT}` | 14,920 each |
| `RXN_QUANTITY` | 7,066 |

**RXNREL relations (SAB = RXNORM), both endpoints governed:**

| Relation | Governed pairs |
| --- | ---: |
| `isa` / `inverse_isa` | 206,815 |
| `has_ingredient` | 134,452 |
| `tradename_of` | 104,677 |
| `consists_of` | 104,454 |
| `has_dose_form` | 95,730 |
| `has_precise_ingredient` | 11,706 |

Deliberately **not** recovered: `NDC`, `SPL_SET_ID`, `ATC_LEVEL` (external identifiers, not
drug structure) and `included_in`/`includes` (0 governed pairs).

## 4. Coverage Across Governed Searchable Concepts (§3)

211,949 concepts processed; 211,949 rows written.

| Field | Concepts | Coverage |
| --- | ---: | ---: |
| **any structure** | **178,340** | **84.14%** |
| ingredient | 116,950 | 55.18% |
| dose form (relation) | 95,730 | 45.17% |
| brand | 85,668 | 40.42% |
| parsed strength | 59,636 | 28.14% |
| RxTerm form | 49,187 | 23.21% |
| available strength | 36,218 | 17.09% |
| precise ingredient | 9,744 | 4.60% |

Strength normalization is deterministic and conservative: `500 MG` = `500mg`, but `500 MG` and
`0.5 G` are **not** unified. Unit conversion is a clinical judgement the source does not make,
and a wrong equivalence would manufacture false exact matches.

## 5. Where the Structure Is Used (§4)

The S1 architecture is unchanged — lexical Top-20 + dense Top-50 → governed union →
Qwen3-Reranker-4B. Two additions, RxNorm only:

* **Document text** (`kb/rxnorm/structured.semantic_text`) — labelled fields for ingredient,
  strength, dose form, brand, quantity, so retrieval and reranking see *what* a token is.
* **Compatibility evidence** (`linking/semantic/rxnorm_structured`) — a deterministic verdict
  per facet, ranked lexicographically:

  `exact` › `supported` › `unsupported` › `contradicted`

The asymmetry is the design. A mention stating `500 mg` against a candidate recorded as
`250 mg` is **contradicted** — the candidate is wrong, not merely unsupported. A candidate with
no recorded strength is **unsupported**: silence in the source is not disagreement. A mention
stating no strength contradicts nothing, so bare-ingredient mentions are a strict no-op.

Ordering is lexicographic, never additive, so no amount of lexical similarity can promote a
contradicted candidate. **ICD is untouched** and a test asserts the module never references it.

## 6. A/B Ablation (§6)

Ontology-native throughout. Queries restate facts the governed source already records, in the
surface form a Vietnamese note writes them. No public-leaderboard signal, no clinical text.

### The first benchmark was invalid and was rebuilt

Queries were initially built verbatim from each concept's own fields. Lexical retrieval solved
that at R@1 0.957 with hard-negative rates of 0.09% and 1.5% — it measured string copying, not
drug discrimination, and every subset was identical. Rebuilt with four disjoint variants:
`strength_compact` (`amoxicillin 500mg`), `vi_form` (`amoxicillin viên nén`, no strength),
`strength_vi_form`, and `ingredient_only`. 1,472 cases, ≥3 competing siblings per family.

**Overall (A = frozen 0073 RxNorm S1, B = A + structured):**

| | R@1 | R@2 | R@5 | R@10 | MRR | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **A** | 0.3240 | 0.3757 | 0.6026 | 0.6168 | 0.4162 | 0.6182 |
| **B** | **0.3274** | 0.3757 | **0.6039** | **0.6175** | **0.4185** | 0.6182 |

**Hard negatives — the headline result:**

| Subset | A | B | Change |
| --- | ---: | ---: | ---: |
| same ingredient, **wrong strength** | 2.80% (19/678) | **1.33% (9/678)** | **−52.5% relative** |
| same ingredient, **wrong form** | 10.31% (23/223) | **4.93% (11/223)** | **−52.2% relative** |

**High-value subsets:**

| Subset | A R@1 | B R@1 | A MRR | B MRR | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact strength stated | 0.9556 | **0.9662** | 0.9743 | **0.9796** | 473 |
| form stated | 0.3586 | **0.3632** | 0.4364 | **0.4401** | 647 |

**By variant:**

| Variant | A R@1 | B R@1 | n |
| --- | ---: | ---: | ---: |
| `strength_vi_form` | 0.9539 | **0.9677** | 217 |
| `strength_compact` | 0.9570 | **0.9648** | 256 |
| `vi_form` | 0.0581 | 0.0581 (MRR +0.0021) | 430 |
| `ingredient_only` | 0.0000 | 0.0000 (exact no-op) | 569 |

Ranking fixes at top-1: **6**. Regressions: **1**. Candidate sets preserved exactly.
NULL remains **shadow**: no candidate is removed by this milestone.

**Brand subset is not informative and is reported as such** — the flag was true for all 1,472
cases, so it does not separate. 37.5% of SCD/SBD concepts carry a recovered brand; a
discriminating brand benchmark needs brand-name *queries*, which this milestone did not build.

## 7. Failure Taxonomy

| Class | Observation |
| --- | --- |
| bare-ingredient ambiguity | `ingredient_only` R@1 0.000, coverage 0.45 — the mention states nothing to arbitrate; structure correctly does nothing |
| form-only mentions | `vi_form` R@1 0.058 — lexical rarely surfaces the right SCD; structure reorders within Top-20 but cannot add what retrieval missed |
| retrieval ceiling | overall coverage 0.618 caps every R@k; ~38% of golds are absent from lexical Top-20 entirely |
| strength contradiction | halved, the intended effect |
| form contradiction | halved |
| single top-1 regression | 1 case where a `supported`-form sibling displaced an `unsupported` gold |
| unrecorded structure | 15.86% of governed concepts have no recovered structure and are unaffected by design |

## 8. Hard Gate (§ hard gate)

> Do not productionize B unless it improves the structured/high-value subsets without a
> material overall RxNorm regression.

| Criterion | Result |
| --- | --- |
| Structured/high-value subsets improve | **PASS** — exact-strength R@1 +1.06pp, both hard-negative rates halved |
| No material overall regression | **PASS** — R@1 +0.34pp, MRR +0.23pp, coverage identical |
| Candidate sets preserved | **PASS** |
| NULL unchanged (shadow) | **PASS** |
| ICD unchanged | **PASS** |

The gate passes **for the stages measured**. It does not authorize a public submission, because
the dense and reranker stages were not measured — see §9.

## 9. What Is NOT Measured

The A/B above covers lexical retrieval and structured reordering, both runnable locally. The
two stages that need a GPU were **not** executed:

* **dense retrieval over the new structured document text** — requires re-encoding the RxNorm
  documents with Qwen3-Embedding-4B, since changing document text invalidates the frozen
  `S1_doc_vectors.pt` for the 211,949 RxNorm rows;
* **reranking with structured pair text** — requires Qwen3-Reranker-4B.

Reporting the §6 table as a complete-system result would repeat exactly the Audit-0071 error
this project already corrected once. It is a retrieval-stage result and is labelled as such in
`ab_report.json` itself.

**Consequence for the frozen index:** activating structured document text means the ICD half of
`S1_doc_vectors.pt` stays valid but the RxNorm half must be rebuilt. That is a Colab job, not a
local one.

## 10. Files Changed

Three distinct categories. The distinction matters because `git diff` reports only the first:
a new file is untracked until the owner stages it, so it never appears in a diff no matter how
substantial it is.

**A. Modified — already tracked, appears in `git diff` (1 file):**

| File | Change |
| --- | --- |
| `src/mednorm_vi/kb/rxnorm/rrf_reader.py` | ATV/SAB transposition fix, +6/−1 |

**B. New — untracked, intended for tracking, will NOT appear in `git diff` (7 files):**

`src/mednorm_vi/kb/rxnorm/structured.py` ·
`src/mednorm_vi/linking/semantic/rxnorm_structured.py` ·
`scripts/build_rxnorm_structured_0074.py` ·
`scripts/evaluate_rxnorm_structured_0074.py` ·
`tests/unit/test_rxnorm_structured_0074.py` ·
`docs/audits/0074-structured-rxnorm-recovery.md` ·
`experiments/registry/EXP-0001.json`

**C. New — untracked and gitignored, never to be tracked (build output):**

`artifacts/rxnorm_structured/0074/` — `structured.jsonl`, `manifest.json`, `ab_report.json`.

**Deletions: none.** No model weights and no private clinical material are in any category.

**Not changed:** E1, E2, E3, L4, assertions, spans, types, ICD behaviour, NULL policy,
candidate-set decoding, output serialization, `full_v1.yaml`, `semantic_s1_0073.yaml`, every
index, both checkpoints, and the 0073 scored artifact.

## 11. Validation

| Gate | Result |
| --- | --- |
| `test_rxnorm_structured_0074.py` | **27 passed** |
| Regression (`rxnorm or semantic or linker or config or pipeline or inference`) | **288 passed** |
| `ruff check .` | clean |
| `ruff format --check` (changed files) | clean |
| `mypy src/mednorm_vi` | no issues, 306 source files |
| `compileall src tests scripts` | clean |
| `git diff --check` | clean |
| Modified already-tracked files (§10 A) | 1 — `rrf_reader.py` (+6/−1) |
| New files awaiting staging (§10 B) | 7 |
| Deletions | none |
| Scored ZIP / indices / checkpoints | unchanged |

## 12. Remaining Risks

1. **The complete-system A/B is unmeasured.** Retrieval-stage gains need not survive reranking.
2. **Rebuilding the RxNorm dense vectors is required** to use structured document text, and a
   partially rebuilt index is a silent correctness hazard — the runtime's row-count check
   catches size mismatches but not a half-refreshed corpus.
3. **Retrieval coverage 0.618 is the ceiling.** Structure reorders; it cannot recall.
4. **Form cue table is small** (24 Vietnamese/English pairs) and hand-listed. It is governed by
   review, not generated.
5. **The brand facet is unproven** — recovered for 40% of concepts but never benchmarked.
6. **1 top-1 regression** out of 1,472; small but real.

## 13. Verdict and Next Step

**`STRUCTURED_RXNORM_ACCEPTED_OFFLINE_COMPLETE_SYSTEM_AB_REQUIRED`** — **NOT_READY** for public
submission.

The structure is recovered with provenance, covers 84% of governed concepts, and halves both
hard-negative rates on the subsets this milestone targeted. That is a real, measured
improvement to the RxNorm path. It is not yet evidence that the *system* improves, and the
project's own history (0070 rejected, 0071 corrected, 0073 accepted) is that only
complete-system evidence should reach the leaderboard.

**Next step:** a Colab job that (a) rebuilds the RxNorm half of the dense index over the
structured document text, (b) re-runs the frozen S1 pipeline A/B with the real reranker on the
same 1,472 cases plus the 0072 benchmark slice, and (c) reports complete-system RxNorm
Recall@1/2/5/10, MRR and the hard-negative subsets. Only if that passes does a second public
submission become justified; 0073 remains the scored baseline and rollback until then.
