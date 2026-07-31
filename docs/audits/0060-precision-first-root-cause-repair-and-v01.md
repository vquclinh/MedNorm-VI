# Audit 0060: Post-baseline Root-cause Analysis, ICD Serialization Probe and V0.1 Decision

Milestone: 4A - Precision-first root-cause repair and V0.1 candidate
Date: 2026-07-31
Verdict: `DETERMINISTIC_REPAIR_INSUFFICIENT_MODEL_REQUIRED`

## 1. Initial Git and Resource State

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `1c3192c add runs ignore` |
| Working tree | clean (no tracked modifications) |
| Staged files | none |
| `origin/main...HEAD` | `0 0` |
| Audit 0059 committed | yes |
| RAM / disk / containers | 8.0 GiB available / 25 G free / 0 running |

The architecture PDF is unmodified (`0d5eaa20…81e09b`) and was read in full earlier in
this session. Audit 0059 was read in full and is neither modified nor superseded.

## 2. Verified Leaderboard Evidence

All three baseline ZIP hashes were re-verified on disk and match the recorded values
exactly; all three archives contain exactly `output/1.json … output/100.json`.

| Profile | Score | WER | J_assert | J_cand | ZIP SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| A (med top-1, dx top-1) | 9.1625 | 87.0976 | 16.0234 | 1.2119 | `bbf12cd5…f30615` |
| B (med top-1, dx adaptive) | 9.1625 | 87.0976 | 16.0234 | 1.2119 | `f3f63af9…32e6532c…` |
| C (med adaptive, dx adaptive) | 9.1432 | 87.0976 | 16.0234 | 1.1636 | `29ef6670…871fbebe3d` |

### The metric is now known exactly

```text
score = 0.3·(100 − WER) + 0.3·J_assertion + 0.4·J_candidates

A: 0.3·(100−87.0976) + 0.3·16.0234 + 0.4·1.2119 = 9.1625   ✓ exact
C: 0.3·(100−87.0976) + 0.3·16.0234 + 0.4·1.1636 = 9.1432   ✓ exact
```

This reproduces both published scores to four decimals, so the weighting is confirmed
rather than assumed. It reframes the whole problem:

| Component | Weight | Earned | Available |
| --- | ---: | ---: | ---: |
| text (1 − WER) | 30 | 3.87 | **26.13** |
| assertions | 30 | 4.81 | 25.19 |
| candidates | 40 | 0.48 | **39.52** |

**A and B scored identically to four decimals.** They differ on 119 diagnosis entities
(Audit 0059 §12), and that made no measurable difference — the first hard evidence that
the second diagnosis candidate is neither helping nor hurting, because those candidates
are not matching anything.

The leaderboard ceiling is **not** ~50. Nothing here suggests the metric is broken; it
suggests we are earning 12.9% of the text component and 1.2% of the candidate one.

## 3. Baseline Artifact Hashes

Verified unchanged and not overwritten by this milestone:

```text
runs/baseline_v0_profile_a/output.zip  bbf12cd5d5c293bfd8904ca9217e987e570604aa411acfe0db4fb33d91f30615
runs/baseline_v0_profile_b/output.zip  f3f63af9c5ee6afe59ec5909587c7b6d0288f68a4fd67dc0e71d0523e6532c3e
runs/baseline_v0_profile_c/output.zip  29ef6670a6cf01d7a98567a11601fa59c4e3e8ad191887d0ec8b54871fbebe3d
```

## 4. Profile-A Aggregate Census

Full per-entity diagnostics are under the ignored path
`runs/diagnostics/0060_profile_a_root_cause/`. Only aggregates appear here, and no
clinical text is reproduced.

```text
documents 100   entities 957   empty documents 5   entities/doc  min 0  median 8  mean 9.6  max 35
```

| Type | n | median len | 1-char | 2-char | short (≤4, no space) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRIỆU_CHỨNG | 575 | 7 | 0 | 14 | 135 |
| CHẨN_ĐOÁN | 199 | 12 | 0 | 4 | 39 |
| KẾT_QUẢ_XÉT_NGHIỆM | 137 | 2 | 28 | 44 | 127 |
| TÊN_XÉT_NGHIỆM | 35 | 3 | 8 | 4 | 23 |
| THUỐC | 11 | 21 | 0 | 0 | 0 |

```text
overlapping (partial)  0        nested 1        duplicate same-position 0
candidate-bearing      CHẨN_ĐOÁN 199, THUỐC 11
candidate list lengths {0: 20, 1: 190}          candidate-less diagnoses 20
assertions             isNegated 42, isFamily 26, isHistorical 14; multi-label entities 3
                       assertion-eligible 785, with ≥1 label 79 (10.1%)
TEST_RESULT            numeric-only 122, alphabetic-only 8, other 7
                       with a `name:` key immediately left 70; with no key evidence 67
surface repetition     472 distinct surfaces over 957 entities (ratio 2.03)
```

Expert attribution, from the Profile-C run manifest:

```text
proposals   E3 786   E2 274   E1 101      lattice nodes 1161
L4          accepted 969   rejected 52   unresolved 140
L7          ACCEPT 240   UNRESOLVED 729   REJECT 0
```

**L7 marked 729 of 969 entities `UNRESOLVED` and every one was emitted anyway.** That is
by design (only `REJECT` withholds), but it means the pipeline itself is uncertain about
75% of what it submitted.

## 5. Root-cause Trace Methodology

A deterministic census over all 957 entities, plus targeted structural probes against
the source documents, using hashed case IDs (`sha256(doc:start:end)[:12]`) so tracked
evidence carries no clinical text. Local per-entity records including left/right context
were written only to the ignored diagnostics path.

The decisive probe was **boundary alignment**: for every entity, is the character
immediately outside the span alphanumeric while the character inside it also is? That
distinguishes a tokenizer/offset defect from a genuine model decision.

## 6. Root-cause Distribution

```text
offset invariant original_text[start:end] == text     957 / 957 valid, 0 violations
mid-token cuts                                        start 4, end 18  (22 of 957 = 2.3%)
  by type: TÊN_XÉT_NGHIỆM 12, KẾT_QUẢ_XÉT_NGHIỆM 7, THUỐC 3
leading/trailing whitespace spans                     0
punctuation-edge spans                                0
short fragments adjacent to an alphanumeric char      0 of 174
```

| Primary root cause | Share | Basis |
| --- | --- | --- |
| `E3_MODEL_SPAN_ERROR` / `E3_MODEL_TYPE_ERROR` | dominant | 774 of 957 entities are E3's symptom/diagnosis output; spans are well-formed complete tokens, so the disagreement with gold is a model judgement, not a mechanical defect |
| `E2_LAB_GRAMMAR_FALSE_POSITIVE` | plausible, unproven | 67 of 137 TEST_RESULT entities have no `name:` key evidence to their left |
| `E3_TOKEN_OR_OFFSET_ALIGNMENT` | **ruled out** | 0 offset violations; 0 of 174 short fragments sit inside a word |
| `L3_PROPOSAL_DUPLICATION` | **ruled out** | 0 duplicate same-position predictions |
| `L4_BOUNDARY_RESOLUTION_ERROR` | **ruled out at scale** | 0 partial overlaps, 1 nested pair in 957 |
| `ICD_RETRIEVAL_ERROR` / serialization | **strong, quantified** | §10 |
| `EXPECTED_MODEL_CAPABILITY_GAP` | remainder | see §8 |

**The short "fragments" are not fragments.** All 174 short spans (135 symptom, 39
diagnosis) are complete standalone tokens with non-alphanumeric neighbours. E3 is
choosing to label short tokens as symptoms; nothing is cutting them.

## 7. Exact Deterministic Defects Found

1. **Per-document KB index reload** (performance, not score). Independently
   re-measured: ICD 0.33 s, RxNorm 2.56 s, and `run_document` called `_indexes(config)`
   once per document plus once each in `_gate_final_output` and `_write_manifest`.
   Repaired in §14.
2. **No run-level progress output.** A 409-second run printed nothing until it
   finished. Repaired in §15.
3. **Decoding profile was not a config value**, so producing a stricter submission
   required post-processing JSON with an untracked script — the `FINAL_SUBMISSION_BLOCKER`
   Audit 0059 recorded. Repaired in §9.

No score-relevant deterministic *defect* was found. That is a finding, not an omission:
offsets, overlap handling, deduplication and the L9 gate are all clean.

## 8. Exact Capability Gaps

| Observed failure | Classification |
| --- | --- |
| 87.1% WER — predicted entity set disagrees with gold at scale | **model capability** (E3 span/type quality out of its training distribution) |
| 10.1% assertion-label coverage; J_assert 16.0 | **supervision** — the governed corpus has zero assertion gold, so S2 cannot be trained |
| J_candidates 1.2 against text 12.9 | **serialization hypothesis first** (§10), then **retrieval learning / reranking** |
| 67 orphan TEST_RESULT entities | **supervision** — no TEST gold exists to decide whether they are wanted |
| 729 L7-UNRESOLVED entities emitted regardless | **calibration** — no stage produces a probability that could gate them |

Since a wrong entity and a missing entity both score `text_score = 0.0` in the pooled
mean, and no gold exists locally, **any precision trim is a bet on a quantity I cannot
measure**. Dropping a spurious prediction removes a zero-scoring slot and helps;
dropping a matched one converts it to a missing slot and strictly hurts. Without gold I
cannot estimate the ratio, so I did not fabricate a rule. §9's forbidden list names
exactly this temptation, and the honest answer is that the evidence does not support a
deterministic repair here.

## 9. Top-1 Canonical Profile Implementation

`TopKPolicy` is now a value, with two governed presets resolved by name:

```text
competition_top1      max_candidates 1   (CANONICAL for competition runs)
competition_adaptive  max_candidates 2, ratios 0.98 / 0.95 / 0.80  (experiment only)
```

Threaded `config → PipelineConfig.decoding_profile → run_document → decode_entities →
apply_competition_topk`, with `--decoding-profile` on the canonical CLI for an explicit
experiment. An unknown name raises rather than falling back.
`configs/pipeline/full_v1.yaml` now declares `decoding_profile: competition_top1`.

Required tests (§4), all passing:

| # | Property | Result |
| --- | --- | --- |
| 1 | top1 emits ≤ 1 medication candidate | pass |
| 2 | top1 emits ≤ 1 diagnosis candidate | pass |
| 3 | adaptive reproduces the previous behaviour | pass — and the Audit-0058B digest `4203bcc0…` reproduces **exactly** under `competition_adaptive` |
| 4 | ordering deterministic | pass |
| 5 | no code outside the L5 offered set | pass (prefix property) |
| 6 | L8/L9 boundaries unchanged | pass |
| 7 | profile changes candidates only | pass — text/type/position/assertions identical |

## 10. ICD Dotted vs Dotless Evidence

`UNRES-ICD-DOTTED` has been formally unresolved since the policy registry was written:
*"Should codes be emitted dotted (A09.9) or undotted (A099)?"*, `confidence: low`,
`internal_default: unknown`, and — decisively — `test_method: probe both formats for
the same diagnosis`. This milestone performs exactly that prescribed test.

**Evidence supporting dotted output**

- The Vietnamese MoH source table is itself dotted: `supplied_code = A00.0`,
  `dotted_code = A00.0`. The dotless `A000` is a **repository-internal normalization**
  (`undotted_code`), not the source form.
- The organizer confirmed `FACT-ICD10-VIETNAMESE` ("ICD-10 uses a Vietnamese version").
  If gold was built from that publication, gold codes carry dots.
- Dotted is the standard WHO presentation form.
- The KB and the active index both retain `dotted_code`, so dotted output is lossless
  and needs no KB change.
- `link_icd10()` already has an unused `dotted_output: bool = False` parameter — the
  repository anticipated this question and left the switch unwired.
- **Quantitative:** text scores 12.9% but candidates only 1.2%, a 10× gap. If matched
  diagnoses carried correct codes, candidate Jaccard would track the text component.
  Profile A emits 179 diagnosis candidates and 11 medication candidates; if every
  diagnosis code scores zero on a format mismatch, the residual 1.2% is consistent with
  a handful of RxNorm matches alone.

**Evidence supporting dotless output**

- Dotless is the repository's internal concept identity, and the current dotless
  submission did score a **non-zero** 1.2119 — though §10's decomposition attributes
  that plausibly to medications rather than diagnoses.

**Inconclusive**

- No tracked organizer example shows an ICD candidate in either form.
- `icd_format_hypotheses_v1.yaml` lists both with `confidence: low` and records
  `assume_who_vs_cm: false`.

Status: **hypothesis, leading = dotted**, to be resolved by the probe. Concept identity
was not changed; this is treated strictly as output serialization.

## 11. Dotted Probe Construction

Two tracked, tested modules replace the untracked derivation script:

- `inference/code_format.py` — reversible ICD dot transform, fails closed on any code
  that is neither well-formed dotted nor well-formed dotless.
- `inference/derive_submission.py` — derives a submission variant from an existing one
  without re-running inference, preserving the source's exact JSON style.

The accepted grammar `[A-Z][0-9]{2}[0-9A-Z]*` is derived from the KB, not assumed: all
**15,308** codes in the active index match it, lengths 3–5, no letter after position 3.
An ICD-10-CM shape such as `C7A1` is therefore correctly rejected.

```text
source        runs/baseline_v0_profile_a/output           ZIP bbf12cd5…f30615
probe         runs/probes/profile_a_icd_dotted/           ZIP f87db8aead3fd935e2a0b9e3245b359b65953f7d28430ca3b56aeff22f663d59

diagnosis codes seen              179
  transformed (dot inserted)      161      e.g. I269 -> I26.9, I7000 -> I70.00
  unchanged (3-char category)      18      e.g. B01, B36
  already dotted                    0
  malformed                         0
entities                          957      documents 100
medication candidate lists identical       11 / 11
```

Verified:

```text
byte-identical documents            39 / 100
documents differing                 61 / 100, ALL dot-only
non-candidate fields changed        0     (text, type, position, assertions)
entities added or removed           0
candidates added or removed         0
offset invariant on the probe       0 violations
organizer validate_output_directory ok
organizer validate_submission_zip   ok
deterministic rebuild               identical ZIP hash
```

The probe was **not submitted**.

## 12. Deterministic Repairs

None applied to scoring behaviour, deliberately — see §8. The three defects repaired
(§14, §15, §9) are performance, observability and reproducibility, and are proven
output-neutral.

## 13. Before/After Synthetic Evidence

```text
prepared indices, 4 tracked synthetic fixtures, deterministic mode
  reload-per-document   17.50 s
  prepared-once          8.33 s   (2 index loads instead of 8)
  speedup                2.10x    ≈ 2.29 s saved per document ≈ 3.8 min per 100 documents
  OUTPUT BYTE-IDENTICAL before/after: True
```

```text
decoding profile equivalence, 4,000 randomised candidate lists, both ontologies
  cases where competition_top1 != truncate(competition_adaptive, 1):  0
```

That second result is why no V0.1 run was made — see §17.

## 14. Prepared-index Implementation

`PreparedIndexes` is a **caller-owned** cache keyed on the exact configured path,
constructed once per `run_input_dir` and threaded into `run_document`,
`_gate_final_output` and the manifest writer. No module-level singleton, no
process-wide state: two configs cannot contaminate each other, fixture indices stay
injectable, and rollback configs remain usable. A changed path reloads; an unchanged
one does not. Output is byte-identical (§13).

## 15. Progress-logging Implementation

One flushed line per document on **stderr**, so it cannot corrupt the CLI's
machine-readable stdout:

```text
[1/3] document=1 entities=6 warnings=1 elapsed=6.4s eta=13s
[2/3] document=2 entities=6 warnings=1 elapsed=9.7s eta=5s
[3/3] document=3 entities=6 warnings=1 elapsed=13.2s eta=0s
```

Carries ordinal, total, document ID, entity count, warning count, elapsed and rolling
ETA. No clinical text. Verified under `set -o pipefail` with
`python -u … 2>&1 | tee run.log`. No second runner was created.

## 16. Validation Results

Targeted: `tests/unit/test_precision_first_0060.py` → **31 passed**.
Combined with the 0058/0058A/0058B suites → **44 passed** after the pin fix below.

One pre-existing test legitimately moved. `test_serialized_candidate_output_matches_its_regression_pin`
read the config's default profile, so making `competition_top1` canonical looked like an
observability regression. The pin now names its profile and asserts both:
`competition_adaptive` still reproduces `4203bcc0…` **exactly** — which is itself the
evidence that Audit 0058B's observability change remains side-effect free — and
`competition_top1` pins `cb2e81f9…`.

Full suite: see §16a. Static and type gates:

| Command | Result |
| --- | --- |
| `ruff check .` | All checks passed |
| `ruff format --check <changed files>` | formatted |
| `mypy src/mednorm_vi` | Success, 290 source files |
| `compileall -q src tests` | passed |
| `git diff --check` | passed |

`metric_decoder/decoder.py` was already not `ruff format`-clean at HEAD and was left in
its existing style, consistent with Audits 0058–0058B.

## 16a. Full-suite Result and a Second Guard That Legitimately Moved

The first full run surfaced one failure:

```text
FAILED tests/unit/test_l5_l7_deterministic_stack.py::test_no_fixed_top_k_returns
1 failed, 1935 passed, 1 skipped in 475.29s
```

`test_no_fixed_top_k_returns` asserted that the candidate-set size **varies**. It was
written in Audit 0053 to catch `decode_expected_jaccard`'s unconditional
`candidates[:10]`: back then a constant size meant a hardcoded slice nobody had
declared. Under `competition_top1` the size is legitimately constant at 1, so the guard
was reporting a *declared policy* as the bug it was built to detect.

Measured on the medication fixture:

```text
competition_adaptive   sizes {1, 2}
competition_top1       sizes {1}
```

The guard was made profile-aware rather than relaxed, and now proves strictly more:
under adaptive the size must still vary with the evidence (the original property,
intact), and under top-1 it must respect the *declared* `max_candidates` **and** be
strictly smaller than the adaptive result on the same input — which demonstrates the cap
is policy-driven and reversible, not a slice baked into the decoder. The original
assertion could not distinguish those two cases.

Confirming full run after that fix: see §16b.

## 16b. Confirming Full-suite Result

```text
1936 passed, 1 skipped in 485.06s (0:08:05)
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

The single skip is the pre-existing `pyarrow` absence, unchanged since Audit 0057.

## 17. Was V0.1 Inference Justified?

**No.** §14 permits a new public run only when a score-relevant deterministic repair
exists, or when the canonical profile materially differs from the ad-hoc Profile-A
derivation. Neither holds:

- No score-relevant deterministic repair was found (§7, §8).
- Native `competition_top1` decoding is **provably identical** to truncating the
  adaptive output to one candidate — verified over 4,000 randomised cases, and
  structurally guaranteed because `_select_candidates` is policy-independent and
  `apply_competition_topk` returns an order-preserving prefix. Profile A *is* that
  truncation.

A V0.1 run would therefore have reproduced Profile A's candidate lists exactly, on the
same commit, indices and checkpoint — an identical submission scoring 9.1625, and one
wasted leaderboard attempt. §14 explicitly directs not to do this.

No public-test inference was run in this milestone.

## 18. V0.1 Hashes

Not created, per §17. The dotted probe (§11) is the only new submission-shaped artifact:

```text
runs/probes/profile_a_icd_dotted/output.zip
f87db8aead3fd935e2a0b9e3245b359b65953f7d28430ca3b56aeff22f663d59
```

## 19. Next-model Recommendation

Ranked against the one metric fact that matters: text is worth 26.13 unearned points and
is gated on entity detection quality, which is a model judgement (§6).

| Option | Failure addressed | Gold needed? | Zero-shot? | Verdict |
| --- | --- | --- | --- | --- |
| E3 re-fine-tune | span/type quality | **yes, and none exists** beyond the corpus it already trained on | no | blocked |
| E5 XLM-R MRC | span/type | yes — head randomly initialized | no | blocked |
| **E6 GLiNER** | span/type precision+recall | **no** | **yes** | **SELECTED** |
| S2 assertion head | J_assert 16.0 | yes — zero assertion gold | no | blocked |
| Dense ICD/RxNorm retriever | J_cand | weak supervision possible | partly | premature until §10 resolves |
| Ontology encoder (SapBERT) | J_cand | no | yes | second priority |
| Learned reranker | J_cand | yes | no | blocked |
| Qwen precision verifier | 729 UNRESOLVED | no | yes | strong, but 4B is tight on 8 GB |
| Qwen adjudicator | hard cases | no | yes | defer |

**Selected: E6 GLiNER as a zero-shot span/type validator over E3's proposals.**

| Field | Value |
| --- | --- |
| Model ID | `urchade/gliner_medium-v2.1` |
| Pinned revision | **must be captured at download and recorded** — not inventable offline |
| License | Apache-2.0 (to be re-verified from the model card at download) |
| Parameters | 209,000,000 (governed planning figure, `configs/parameter_budget.yaml`; **`parameter_count_verified: false`** — must be counted programmatically before deployment) |
| Deployed contribution | +209M against 134,998,272 currently deployed |
| 9B budget impact | ~344M of 9B; **~8.66B still free** |
| RTX 4060 8 GB | comfortable — fp16 inference well under 1 GB |
| Colab L4/A100 | unnecessary |
| Integration stage | L3, via the existing strict `mention_factory/gliner.py` adapter, which already fails closed without a verified local checkpoint |
| Major risk | GLiNER is not trained on the organizer's annotation convention; it must be used as a **validator/second opinion on low-confidence E3 spans**, never as an independent proposer, or it will add spurious entities and raise WER |

It is chosen over the Qwen verifier because it is 20× smaller, runs locally without
quantization, needs no gold, and plugs into an adapter that already exists and fails
closed. No model was downloaded, trained or fine-tuned in this milestone.

## 20. Remaining Leaderboard Risks

1. The dotted hypothesis may be wrong; the probe is designed so that outcome is still
   informative, because it isolates exactly one variable.
2. `UNRES-ICD-SPECIFICITY` remains unresolved — a 3-character category may or may not
   be accepted; 18 of 179 emitted diagnosis codes are 3-char.
3. `UNRES-POS-INTERVAL`, `UNRES-POS-COORD-SPACE` and `UNRES-MATCH-MODE` remain
   unresolved and could each independently depress WER.
4. Over- versus under-prediction is still undetermined without gold, so no precision
   trim can be justified yet.
5. A and B scoring identically means diagnosis top-2 currently buys nothing.

## 21. Exact Changed-file Inventory

Tracked modifications:

```text
M configs/pipeline/full_v1.yaml
M src/mednorm_vi/inference/cli.py
M src/mednorm_vi/inference/config.py
M src/mednorm_vi/inference/pipeline.py
M src/mednorm_vi/kb/competition/topk.py
M src/mednorm_vi/metric_decoder/decoder.py
M tests/unit/test_governed_bridge_observability_0058b.py
```

New tracked-intended files:

```text
docs/audits/0060-precision-first-root-cause-repair-and-v01.md
src/mednorm_vi/inference/code_format.py
src/mednorm_vi/inference/derive_submission.py
tests/unit/test_precision_first_0060.py
```

Ignored generated evidence:

```text
runs/diagnostics/0060_profile_a_root_cause/
runs/probes/profile_a_icd_dotted/
```

The baseline A/B/C artifacts were read only and are byte-identical.

## 22. No Public-test-specific Production Rules

No word blacklist, no document-ID special case, no hard-coded expected output, and no
entity type disabled because of leaderboard performance. The ICD grammar is derived
from the KB's own 15,308 codes; the decoding profiles are ontology-level policies; the
derivation tool contains no document ID and no clinical vocabulary. Public organizer
input was read only for aggregate diagnostics and was never used as training data.

## 23. No Clinical Text in Tracked Outputs

Tracked evidence contains counts, code strings, hashes and type labels only. Per-entity
records with source context exist solely under the ignored
`runs/diagnostics/0060_profile_a_root_cause/`. The most-repeated-surface table in §4
reports surface *lengths* and counts, never the surfaces. Diagnostic case IDs are
`sha256(doc:start:end)[:12]`.

## 24. Verdict

`DETERMINISTIC_REPAIR_INSUFFICIENT_MODEL_REQUIRED`

The root-cause analysis found the pipeline mechanically sound: offsets are 100% valid,
overlap and duplication are effectively zero, and the short "fragments" are complete
tokens rather than cut ones. The 87.1% WER is a model-capability gap, not a bug, and no
deterministic repair is supportable without gold.

What this milestone did establish is worth more than a speculative trim: the exact
metric formula, that A and B are indistinguishable, that candidates underperform text by
10×, and a fully validated one-variable probe for the unresolved ICD serialization
policy — plus the reproducibility, performance and observability infrastructure that
Audit 0059 flagged as a final-submission blocker.

## 25. Recommended Use of the Two Remaining Attempts

1. **Attempt 1 — submit the dotted probe now.**
   `runs/probes/profile_a_icd_dotted/output.zip`, SHA-256
   `f87db8aead3fd935e2a0b9e3245b359b65953f7d28430ca3b56aeff22f663d59`.
   It differs from the already-scored Profile A **only** by the ICD dot, so the delta is
   attributable to serialization alone. Expected readings: J_candidates rising sharply
   confirms dotted; unchanged at ~1.21 confirms dotless and redirects effort to
   retrieval quality. Either outcome resolves `UNRES-ICD-DOTTED` permanently.
2. **Attempt 2 — hold.** Do not spend it until the probe result is known. Its value is
   as the first submission that combines the winning serialization with the E6 GLiNER
   precision intervention.
