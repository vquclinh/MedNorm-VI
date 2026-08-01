# Audit 0072: Complete Zero-Shot Hybrid Semantic Linker — Pre-GPU

Milestone: 4M - real reranking, complete-system evaluation, frozen selection
Date: 2026-08-01
Verdict: `ZERO_SHOT_COLAB_READY`

No clinical text appears in this audit. **No model was executed, downloaded or trained.**

## 1. What Audit 0071 Shipped, and Why It Could Not Go to Colab

The Audit-0071 handoff would have produced a confident but hollow answer:

| Defect | Consequence |
| --- | --- |
| `evaluate_semantic_linker_0071.py` scored **dense retrieval only** | no complete-system number existed |
| Qwen3 reranking was never implemented | the discriminator - the part Audit 0070 proved decisive - was absent |
| Cell 9 selected S1/S2 on dense Recall@10 / MRR | the architecture would have been chosen on the wrong metric |
| `train_semantic_linker_0071.py` wrote a plan and stopped | a "training" cell that trains nothing |

Audit 0070 already established the governing fact: expanding the candidate pool **lowered**
J_candidates (2.8646 → 2.6459). Selecting an embedding model by how many candidates it
retrieves optimizes precisely the quantity that hurt the score. This milestone fixes that
before any GPU time is spent.

## 2. Preserved Baseline (§1)

Current public best is unchanged and remains production and rollback:

| | Score | WER | J_assertion | J_candidates |
| --- | ---: | ---: | ---: | ---: |
| **Best (control)** | **11.9188** | 82.7090 | 18.6187 | 2.8646 |
| Audit-0070 (rejected) | 11.8313 | 82.7090 | 18.6187 | 2.6459 |

No pipeline config was modified, v4.1 was **not** activated globally, and every Audit
0068-0071 artifact is retained. Protected digests re-verified unchanged: scored ZIP
`eae8a348…4c05`, ICD v3 `d72ed17f…5fdd`, ICD v4.1 `d1803be3…ea09c`, E3 `524ece1e…dde3a`.

## 3. Real Reranker Backend (§2)

`src/mednorm_vi/linking/semantic/reranker.py`.

Qwen3-Reranker is **not** an embedding model and its score is not a cosine similarity. It is a
causal LM asked a yes/no question; the score is `softmax([logit_no, logit_yes])[yes]` at the
final position, using the official framing verbatim:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct
provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: … <Query>: … <Document>: …<|im_end|>
<|im_start|>assistant
<think>

</think>
```

| Requirement | Implementation |
| --- | --- |
| batched inference | `--rerank-batch-size`, chunked with `torch.no_grad()` |
| BF16 on CUDA | `torch.bfloat16` on cuda, float32 elsewhere |
| deterministic ordering | sort by `(-score, rank_before, concept_id)` - total order, no sampling |
| bounded token length | `MAX_PAIR_TOKENS = 1024`, truncation on |
| OOM-safe messages | both load and score paths name the exact flag to lower |
| explicit revision | **full 40-char sha required**; abbreviated shas and branch names raise |
| unload between systems | `unload()` drops model and tokenizer, then `empty_cache()` |
| recorded per candidate | raw score, normalized score, `rank_before`, `rank_after`, sources, tier |

`RerankerBackend` is a Protocol and `MockReranker` is a weight-free implementation, so every
test runs without CUDA. Substituting embedding similarity for a reranker score is now a type-
level distinction rather than an easy mistake.

## 4. Hybrid Candidate Flow (§3, §4)

`src/mednorm_vi/linking/semantic/hybrid.py`.

```
ICD:    v3 Top-20  ∪  v4.1 Top-20  ∪  dense Top-50
RxNorm: lexical Top-20            ∪  dense Top-50
   → dedup by governed identity
   → REAL cross-encoder rerank
   → H2 within-family arbitration
   → null gate (shadow | conservative)
```

Two structural invariants, both tested:

* **No model may generate a code.** `build_pool` looks every candidate up in the governed
  frozen KB and drops anything absent. A dense hit naming an unknown concept cannot survive.
* **v3 is an anchor, not a veto.** v3 always contributes, because Audit 0070 proved replacing
  it costs J_candidates - but nothing encodes "v3 wins". The reranker may overturn it, and the
  evaluator counts `v3_top1_kept`, `v3_top1_overturned`, `reranker_fixes`,
  `reranker_regressions`.

Per-candidate provenance is retained: `from_v3` / `from_v41` / `from_dense`, plus each
source's rank, the dense score, the lexical score and the Audit-0069 evidence tier. RxNorm
documents carry RxCUI, name, TTY and aliases only - Audit 0071 confirmed ingredient, strength,
dose form and brand are **absent from the governed index**, and they are not fabricated.

## 5. Hierarchy Position (§5)

H2 runs **after** semantic reranking. It is a tie-break over judgement already made, not a
substitute for it; running it first would let a lexical family rule reorder candidates the
reranker had not yet seen. Containment is unchanged and tested: a family keeps the best
position any member held, and **no family may move past another**. The test asserts both
halves - an unrelated concept scored highest by the reranker stays at rank 1 while H2 still
corrects the parent/child order inside the family below it.

## 6. Null Gate Policy for the First Run (§6)

Audit 0071 measured the lexical-only gate honestly: false emissions 43/54 → 9/54, but R@1
11/15 → 8/15 and R@10 15/15 → 9/15. Refusal without a discriminator trades one error for
another.

Two modes now exist:

| Mode | Behaviour |
| --- | --- |
| `shadow` | evaluates the gate, records every feature and the decision, **changes no output** |
| `conservative` | applies refusal; Audit-0069 tiers A-C stay protected and are never refused |

**The first zero-shot run uses `shadow`.** Architecture selection therefore depends only on
valid-concept ranking quality, and the run still returns reranker-aware NULL evidence
(`S1_null_shadow.jsonl`, `S2_null_shadow.jsonl`) for later calibration **locally**, against the
private 69-record pack, which is never uploaded. No leaderboard signal is used.

## 7. Complete-System Evaluator (§7)

`scripts/run_0072_zero_shot_colab.py` reports three clearly separated stages:

| Block | Meaning |
| --- | --- |
| `stage_dense` | dense retrieval Recall@1/2/5/10 - **a retrieval stage, not a result** |
| `stage_union` | v3 ∪ v4.1 ∪ dense Recall@1/2/5/10 |
| `final_reranked` | Recall@1/2/5/10 + MRR **after** rerank + H2 - the only system result |

Plus `by_ontology` (ICD10 and RxNorm separately) and diagnostics: `v3_top1_kept`,
`v3_top1_overturned`, `reranker_fixes`, `reranker_regressions`,
`dense_added_correct_concepts`, `hierarchy_changes`, mean/max pool size, `runtime_seconds`,
`peak_vram_gib`. Every result file carries an explicit warning that dense and union numbers
are not complete-system results.

## 8. Frozen Selection Rule (§8)

Selection reads `final_reranked` only. The old dense-based Cell-9 logic is gone.

1. final Recall@10
2. final Recall@1
3. final MRR
4. fewer `reranker_regressions` (clean lexical-anchor regressions)
5. deterministic name tie-break

Required: `final Recall@10 >= lexical baseline`, computed in the same run over the same
queries. Preferred and reported separately: final Recall@1 > baseline, final MRR > baseline.
A system cannot be selected because its embedding retrieved more candidates.

## 9. Full Immutable Revisions (§9)

Resolved from the Hugging Face API and used everywhere - configs, download commands,
manifests, orchestrator. `Qwen3Reranker` raises on anything shorter than 40 hex characters.

| Model | Full revision | Params | License |
| --- | --- | ---: | --- |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` | 4,021,774,336 | apache-2.0 |
| `Qwen/Qwen3-Reranker-4B` | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | 4,021,784,576 | apache-2.0 |
| `Qwen/Qwen3-Embedding-8B` | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | 7,567,295,488 | apache-2.0 |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 595,776,512 | apache-2.0 |

## 10. Parameter Budget (§16)

| System | E3 | Embedding | Reranker | **Total** | < 9B | Headroom |
| --- | ---: | ---: | ---: | ---: | :---: | ---: |
| **S1** | 135,002,117 | 4,021,774,336 | 4,021,784,576 | **8,178,561,029** | ✅ | 821,438,971 |
| **S2** | 135,002,117 | 7,567,295,488 | 595,776,512 | **8,298,074,117** | ✅ | 701,925,883 |

`artifacts/semantic/0072/deployed_parameter_manifest.json`, fail-closed. No learned model was
added beyond the two already planned.

## 11. Sequential Execution Strategy (§10)

Per system: load embedding → encode documents (cached to
`artifacts/semantic/0072/cache/{system}_doc_vectors.pt`) → encode queries → **delete the
embedding model and `empty_cache()`** → load reranker → rerank → `unload()` → next system.
No runtime holds all four models. `--stage all` skips any system whose `*_complete.json`
already exists, so a disconnect never costs a 227k-document re-encode.

## 12. No Training (§11)

`train_semantic_linker_0071.py` is **not invoked** anywhere in the 0072 runbook or
orchestrator, and a test asserts the orchestrator contains no reference to it. The selection
output records `"training_performed": false`. Training becomes a later milestone once zero-shot
evidence exists.

## 13. Public Inference Preparation (§12) — and a correction

The code path exists and is **not** run automatically. It stays out of the runbook because the
offline gate has not passed, which §12 requires first.

**Correcting the Audit-0071 handoff:** I stated that public inference "cannot and must not" run
on Colab. The "must not" was wrong - no competition rule established in this repository forbids
processing organizer input on Colab, and the owner's A100/H100 access is legitimate for heavy
inference. The real constraint is narrower and practical: `data/organizer_test/input` is
gitignored, so it is not in a fresh clone and would have to be uploaded deliberately. The
runbook now says exactly that. `data/private_linker_gold/` remains local, unconditionally.

## 14. Tests (§15)

`tests/unit/test_complete_semantic_linker_0072.py` - **28 tests, no weights downloaded**:
reranker prompt protocol and instruction wording, pair construction, abbreviated/branch
revision refusal, 40-hex revision check, rerank reordering with both ranks recorded,
deterministic tie ordering, mock determinism, three-source union, dedup across sources,
**no arbitrary-code generation**, v3 anchor always pooled, reranker overturning v3, v3 kept
when the reranker agrees, H2-after-rerank with family containment, H2 skipped for RxNorm,
shadow mode changing nothing, conservative mode refusing, conservative mode still protecting
tiers A-C, unknown null mode refused, empty pool, orchestrator pins full revisions, selection
reads `final_reranked`, orchestrator does not train, dense/union/final separately labelled,
parameter budget, shipped manifest agreement.

Combined semantic + hierarchy + tier suites: **104 passed**.

## 15. Validation (§17)

| Gate | Result |
| --- | --- |
| Targeted semantic tests | 104 passed |
| `ruff check .` | clean |
| `ruff format --check` (changed files) | clean |
| `mypy src/mednorm_vi` | no issues, 303 source files |
| `compileall src tests scripts` | clean |
| `git diff --check` | clean |

Full `pytest tests/` was not re-run: no existing module was modified, only new files added.
Stated as a scope decision, not claimed as a pass.

## 16. Changed-File Inventory

New (tracked): `src/mednorm_vi/linking/semantic/{reranker,hybrid}.py`,
`scripts/run_0072_zero_shot_colab.py`,
`tests/unit/test_complete_semantic_linker_0072.py`,
`docs/colab/0072-zero-shot-semantic-linker.md`,
`docs/audits/0072-complete-zero-shot-semantic-linker.md`.

New (untracked, ignored): `artifacts/semantic/0072/deployed_parameter_manifest.json`.

Modified: **none.** No existing module, config or index changed.

## 17. Remaining Risks

1. **Reranking 2,000 queries × ~40-70 candidates is the run's cost centre** - roughly 80-140k
   cross-encoder pairs per system. The 4B reranker on an A100 is the long pole; if it proves
   too slow, reduce `--dense-topk` before reducing query count, since pool size drives cost
   linearly while query count drives statistical power.
2. **The evaluation queries are ontology surface forms, not clinical mentions.** They carry no
   context, so the context-aware half of the query template is exercised only structurally.
   The clinical-mention measurement still requires the private pack, locally.
3. **Shadow-mode NULL means the first run cannot improve the false-emission rate** - it only
   collects the evidence needed to calibrate it. That is deliberate.
4. **Both systems sit within ~700-820M of the 9B cap.** No further learned component fits.
5. **Zero-shot quality is genuinely unknown.** Nothing here predicts it; Audit 0070 is a
   standing reminder that a larger pool can lower the score.

## 18. Verdict

**`ZERO_SHOT_COLAB_READY`**

The reranker is implemented against the official protocol, the hybrid union preserves the v3
anchor while letting semantics overturn it, H2 runs after reranking with containment intact,
NULL is shadow-only so it cannot distort selection, the evaluator separates retrieval stages
from the complete system, and selection is frozen on final reranked metrics. Full immutable
revisions are pinned everywhere and both architectures remain under 9B.

**No model is accepted.** Nothing about zero-shot quality is claimed - that is what the GPU run
is for.
