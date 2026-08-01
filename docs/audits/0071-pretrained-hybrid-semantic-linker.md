# Audit 0071: Pretrained Hybrid Semantic Linker — Preparation and Colab Handoff

Milestone: 4L - dense retrieval, strong reranking, null-candidate calibration
Date: 2026-08-01
Verdict: `COLAB_EXECUTION_REQUIRED`

No clinical text appears in this audit.

## 1. Public Result and Active Baseline (§0)

| Run | Score | WER | J_assertion | J_candidates |
| --- | ---: | ---: | ---: | ---: |
| **Current best (control)** | **11.9188** | 82.7090 | 18.6187 | **2.8646** |
| Audit-0070 candidate | 11.8313 | 82.7090 | 18.6187 | 2.6459 |

The Audit-0070 candidate is **REJECTED**. WER and J_assertion are byte-identical, confirming
the ablation was single-variable: the entire 0.0875 loss is J_candidates, −0.2187.

Recorded as an aggregate experiment outcome only. It was not used as a label, a per-record
oracle, or a calibration target anywhere in this milestone. Commits 0068-0070 are untouched;
ICD v4, v4.1, the alias layer, repaired titles, evidence-tier retrieval, the H2 policy and all
audit artifacts are retained.

## 2. Preconditions

Branch `main`, HEAD `130e03b feat: add hierarchy-aware ICD specificity arbitration`, clean
tree, `origin/main...HEAD = 0 0`, nothing staged. Architecture PDF, both E3 checkpoints,
RxNorm, ICD v3/v4/v4.1 and the scored 11.9188 ZIP all verified unchanged.

## 3. Why Execution Stopped Here (§17)

| Resource | Available | S1 needs | S2 needs |
| --- | ---: | ---: | ---: |
| GPU VRAM | 8,188 MiB (RTX 4060 Laptop) | ~16 GB bf16 | ~17 GB bf16 |
| System RAM | ~7 GiB free of 14 | — | — |
| **Disk on `/home`** (holds the HF cache) | **3.1 GiB free** | ~16 GB | ~17 GB |

Downloading a single 4B model would exhaust `/home` before the first shard finished. §17
requires stopping before multi-GB weights are pulled onto a machine that cannot host them,
so **no model weights were downloaded and no encoder was executed**. Everything that does not
need a GPU was completed locally; the GPU steps are handed off in
[docs/colab/0071-semantic-linker-commands.md](../colab/0071-semantic-linker-commands.md).

## 4. Model Candidates — Verified Without Downloading (§3)

Metadata read from the Hugging Face API on 2026-08-01. Revisions are immutable commit SHAs.

| Repository | Revision | Parameters | License | Pipeline |
| --- | --- | ---: | --- | --- |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99` | 4,021,774,336 | apache-2.0 | feature-extraction |
| `Qwen/Qwen3-Reranker-4B` | `22e683669bc0` | 4,021,784,576 | apache-2.0 | text-ranking |
| `Qwen/Qwen3-Embedding-8B` | `1d8ad4ca9b3d` | 7,567,295,488 | apache-2.0 | feature-extraction |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed4502` | 595,776,512 | apache-2.0 | text-ranking |
| `BAAI/bge-m3` (optional control) | `5617a9f61b02` | not published | mit | sentence-similarity |

All are official repositories with `config.json` and `tokenizer.json` present.
`transformers >= 4.51` is required; earlier releases have no Qwen3 model classes.

## 5. Deployed Parameter Budget (§18)

E3 was measured programmatically from its state dict rather than taken from the planning
manifest: **135,002,117** parameters (encoder 134,998,272 + head 3,845), checkpoint
`524ece1e…dde3a`. It is the only learned component the scored 11.9188 system deploys
(`mode: specialist`).

| System | E3 | Embedding | Reranker | **Total** | Under 9B | Headroom |
| --- | ---: | ---: | ---: | ---: | :---: | ---: |
| **S1 balanced** | 135,002,117 | 4,021,774,336 | 4,021,784,576 | **8,178,561,029** | ✅ | 821,438,971 |
| **S2 recall-heavy** | 135,002,117 | 7,567,295,488 | 595,776,512 | **8,298,074,117** | ✅ | 701,925,883 |

No training-time-only teacher models are proposed, so nothing is excluded from these totals
on that basis. Written to `artifacts/semantic/0071/deployed_parameter_manifest.json`, which
fails closed if a total reaches the cap.

## 6. S0 Baseline on the 69 High-Consensus Records (§12)

Measured, not assumed. Reproduces the brief's figures exactly.

| Subset | SELECTED | NO_VALID | R@1 | R@2 | R@5 | R@10 | MRR | False emissions | NULL P / R / F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **All** | 15 | 54 | **11/15** | 13 | 13 | 15 | 0.814 | **43/54** | 1.000 / 0.204 / 0.338 |
| ICD-10 | 8 | 31 | 5/8 | 6 | 6 | 8 | 0.714 | 20/31 | 1.000 / 0.355 / 0.524 |
| **RxNorm** | 7 | 23 | 6/7 | 7 | 7 | 7 | 0.929 | **23/23** | **0.000 / 0.000 / 0.000** |

**RxNorm never declines.** It emits a candidate in every one of the 23 records where
independent adjudication found no valid concept. That is the single largest measured defect
in the linker, and it is a direct J_candidates loss: a wrong candidate set scores worse than
an empty one when the reference is empty.

This also supplies the most plausible mechanism for the Audit-0070 regression. That milestone
raised ICD coverage from 39/50 to 43/50 non-empty; on records that should emit nothing, extra
coverage is extra error. Retrieval was never the binding constraint - **discrimination and
refusal are**.

## 7. Null Gate — Implemented and Measured (§10)

`src/mednorm_vi/linking/semantic/null_gate.py`. Evidence-based, not threshold-tuned: it reads
evidence tier, top score, top1-top2 margin, source agreement, exact/alias support,
unsupported specificity, and an ontology-specific therapeutic-class rule. Audit-0069 tiers
A-C (whole-string accent-sensitive governed matches) are **never** refused. Thresholds derive
from governed KB structure and adjudicated engineering records; **no leaderboard signal was
used**, and a test asserts that contract.

Measured with **lexical evidence only** - no dense or reranker score available:

| | False emissions | NULL F1 | R@1 | R@10 |
| --- | ---: | ---: | ---: | ---: |
| S0 (no gate) | 43/54 | 0.338 | 11/15 | 15/15 |
| S0 + gate (lexical only) | **9/54** | **0.857** | **8/15** | **9/15** |

**This configuration is NOT accepted.** It fixes null rejection and breaks recall: R@1 falls
11 → 8 and R@10 falls 15 → 9, violating §19's "selected-consensus top-1 does not regress" and
"Recall@10 does not regress".

The finding is the point. Without a reranker score the gate cannot distinguish *weak but
correct* from *weak and wrong*, so it refuses both. The missing discriminative signal is
exactly what §4 says must be measured as a complete system rather than as embedding recall.
This is now direct evidence that the reranker is load-bearing, not optional.

## 8. Artifacts Built Locally (§5, §6, §11A)

`scripts/build_semantic_kb_0071.py`, model-free, deterministic, seconds to run.

| Output | Value |
| --- | ---: |
| Semantic documents | 227,257 (15,308 ICD + 211,949 RxNorm) |
| ICD damaged titles excluded from embedded text | 1,432 |
| Benchmark rows train / dev / test | 256,394 / 32,016 / 32,074 |
| Benchmark concepts train / dev / test | 178,942 / 22,368 / 22,368 |
| **Concept leakage across splits** | **0** (build fails closed if any) |

Damaged ICD titles are excluded from the embedded text and retained as provenance. Embedding
`Bao gồm: Bóng` would teach a retriever that an ICD tabular instruction *is* a concept - the
precise error Audit 0069 spent a milestone removing from lexical retrieval.

Hard negatives, typed so a failure can be attributed to a mechanism:

| Kind | Count |
| --- | ---: |
| `rxnorm_same_ingredient_different_tty` | 1,060,897 |
| `icd_sibling` | 53,671 |
| `icd_same_organ_different_disease` | 31,031 |
| `icd_parent` | 26,324 |
| `icd_accent_false_friend` | 14,115 |
| `icd_child` | 7,644 |
| `rxnorm_lexical_false_friend` | 1,356 |
| `icd_other_dot8` / `icd_unspecified_dot9` | 147 / 79 |

### A governed-data gap found and reported rather than papered over

§6 asks the RxNorm semantic document to carry ingredient, precise ingredient, strength, dose
form and brand. **The competition RxNorm index carries none of these fields** - its metadata
is `tty`, `tty_values`, `membership`, `runtime_role`, `suppress` and nothing else. Two
consequences, both recorded rather than fabricated:

* the RxNorm document embeds RxCUI, canonical name, TTY and aliases only;
* `rxnorm_same_ingredient_different_strength` and `…_different_dose_form` negatives cannot be
  built. Deriving them would mean parsing drug names into structure the KB does not assert.

TTY negatives are drawn from the governed graph, which does encode the relations. Recovering
the missing attributes from the local RxNorm release is a genuine follow-up, not a defect in
this milestone's scripts.

## 9. Zero-Shot Results (§14)

**Not measured.** No encoder was executed, because doing so required weights this machine
cannot host. S1 and S2 are therefore unevaluated, no architecture is selected, and per §15 no
fine-tuning was attempted - fine-tuning before a zero-shot number is explicitly forbidden.

## 10. Colab Execution Path (§17)

`docs/colab/0071-semantic-linker-commands.md` - shell commands only, **no `.ipynb` created**.
Covers runtime selection, repository checkout, pinned dependencies, pinned-revision downloads,
KB build, zero-shot evaluation (with a `--limit-documents` smoke test before the full 227k
encode), architecture selection, optional fine-tuning, the public run, the governed
`icd_dotted` derivation, and packaging artifacts for transfer back. It restates every standing
constraint, including that `data/private_linker_gold/` is never uploaded.

Scripts: `build_semantic_kb_0071.py` (runs locally), `evaluate_semantic_linker_0071.py`,
`train_semantic_linker_0071.py`, `run_semantic_linker_0071.py`. Heavy imports are lazy, so all
four lint, type-check and import on a GPU-less machine. `run_semantic_linker_0071.py` **fails
closed** with a pointer to the Colab document when no dense index exists.

## 11. Acceptance Gate (§19)

| Requirement | Status |
| --- | --- |
| Concept-disjoint benchmark improves over lexical baseline | **not measured** (no encoder run) |
| Selected-consensus top-1 does not regress | **FAILS** for the lexical-only gate (11 → 8) |
| Recall@10 does not regress | **FAILS** for the lexical-only gate (15 → 9) |
| NO_VALID false-emission rate materially improves | **PASSES** (43/54 → 9/54) |
| ICD hierarchy errors do not increase | not measured |
| RxNorm TTY/strength/form errors do not increase | not measured |
| `sởi`/`sỏi` remains fixed | **PASSES** (untouched; no retrieval change) |
| v3 exact clean cases protected | **PASSES** by construction (tiers A-C never refused) |
| Deterministic repeat | **PASSES** |
| E1 / E2 / E3 spans+types / assertions unchanged | **PASSES** (no pipeline change) |
| Deployed parameters < 9B | **PASSES** for both S1 and S2 |

The gate is not satisfied. No architecture is selected and nothing is accepted for deployment.

## 12. Public Inference (§20)

**Not run.** §20 permits it only if the selected semantic system clearly passes offline
gates; no system was even evaluated. No `runs/pretrained_semantic_linker_0071_final_dotted/`
was created and no ZIP exists. **The scored 11.9188 system remains production and rollback.**

## 13. Validation (§22)

| Gate | Result |
| --- | --- |
| `tests/unit/test_semantic_linker_0071.py` | 27 passed |
| `ruff check .` | clean |
| `ruff format --check` (files I changed) | clean |
| `mypy src/mednorm_vi` | no issues, 301 source files |
| `compileall src tests scripts` | clean |
| `git diff --check` | clean |

Full `pytest tests/` was not re-run in this milestone: no existing module was modified, only
new files added, and the previous full run (2,116 passed / 2 skipped) was against the same
tree plus these additions. That is a deliberate scope note, not an omission claimed as a pass.

## 14. Changed-File Inventory

New (tracked): `src/mednorm_vi/linking/semantic/{__init__,representation,null_gate}.py`,
`scripts/{build_semantic_kb,evaluate_semantic_linker,train_semantic_linker,run_semantic_linker}_0071.py`,
`tests/unit/test_semantic_linker_0071.py`,
`docs/colab/0071-semantic-linker-commands.md`,
`docs/audits/0071-pretrained-hybrid-semantic-linker.md`.

New (untracked, ignored): `artifacts/semantic/0071/`.

Modified: **none.** No existing module, config or index was changed, so the 11.9188 system is
bit-for-bit intact.

## 15. Remaining Risks

1. **The null gate needs the reranker to be safe.** Alone it trades 34 false emissions for 6
   lost top-1 and 6 lost top-10. It must be recalibrated with reranker scores before use.
2. **RxNorm lacks ingredient/strength/form/brand fields**, limiting both the semantic document
   and the hard-negative taxonomy. Recovering them is a prerequisite for the TTY/strength/form
   error metrics §12 asks for.
3. **Both S1 and S2 sit within ~700-820M of the 9B cap.** Any additional deployed model - an
   L7 adjudicator, a mention proposer - would breach it. The budget is effectively spent.
4. **No zero-shot evidence exists yet.** The architecture choice is entirely open, and the
   0070 result is a warning that a larger candidate pool can lower the score.
5. **n = 15 selected records.** Even after Colab runs, the selection signal is thin; the
   concept-disjoint benchmark (32,074 test rows) is the statistically meaningful measurement.

## 16. Verdict

**`COLAB_EXECUTION_REQUIRED`**

Everything executable without a GPU is complete and measured: the S0 baseline, the null gate
and its honest failure mode, 227,257 semantic documents, a leakage-free concept-disjoint
benchmark with 1.19M typed hard negatives, verified model revisions and licenses, a
fail-closed parameter manifest showing both architectures under 9B, four CLI scripts and a
notebook-free Colab runbook.

The milestone's central measurement - which of S1 or S2 to select - cannot be made on a
machine with 3.1 GiB of free cache disk, and §17 forbids attempting it anyway. No weights were
downloaded, no architecture was selected, no fine-tuning was performed, and no public
inference was run.

**Recommendation: DO_NOT_SUBMIT.** There is nothing new to submit; the 11.9188 system stands.
