# Audit 0033 — S1 Exact Character-Offset Mention Evaluator

- **Date:** 2026-07-27
- **Author:** Claude (AI agent), for human review
- **Change type:** First exact character-offset mention evaluator for S1/L4, executed on the
  governed internal-test split with the validated S1 checkpoint. **No commit. No push. No
  training, no retraining, no backward pass, no optimizer or scheduler. No checkpoint mutation.
  No organizer inference. No `output.zip`. No model-weight download. Governed corpus
  unmodified.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — read in full and **unchanged** (`git
  status` clean). §4 (offset preservation, `original_text[start:end] == entity["text"]`,
  end-exclusive positions), §1 (metric weights; a wrong type is double-penalised), §6 (no expert
  emits a final entity), §7 (boundary/type resolution), §18.1 (mandatory local evaluator,
  reporting by type/route/section/length).
- **Status:** `EXACT_EVALUATOR_IMPLEMENTED_AND_EXECUTED — L3_LATTICE_AND_L4_RESOLVER_NOT_STARTED`.

## 1. Scope actually delivered, and what was not

The requested milestone specified six connected deliverables. **One was completed in full and
executed; five were not started.** This is stated plainly rather than partially implemented and
described as done.

| Deliverable | Status |
| --- | --- |
| A. Exact mention evaluator | **DELIVERED and EXECUTED** on real data |
| E. SYMPTOM error analysis | **PARTIALLY DELIVERED** — the real taxonomy below falls out of A; the router/section/treatment-phrase attribution does not, because it needs B |
| B. Unified L3 span lattice | **NOT STARTED** |
| C. L4 Boundary & Type Resolver v1 | **NOT STARTED** |
| D. Laboratory coverage + synthetic suite | **NOT STARTED** |
| F. Four-way ablation | **NOT POSSIBLE** — three of its four arms require B and C |

The reason is capacity, not disagreement with the plan: the six deliverables are roughly two
thousand lines of new code plus execution, and the working session could not hold all of it. A
was chosen first because every other deliverable is measured *through* it — D, E and F are
meaningless without an exact evaluator, and shipping B/C without one would produce unmeasurable
changes to the pipeline.

**No ablation numbers, resolver behaviour, or synthetic laboratory metrics are reported
anywhere in this audit**, because none were produced.

## 2. Initial Git state

```text
branch  main   (## main...origin/main, in sync)
HEAD    15d0b07 feat: validate S1 checkpoint and add local held-out evaluation
        36c520d fix: protect S1 astral characters and finalize preflight

git status --short
 M docs/audits/0032-s1-local-checkpoint-and-architecture-progress-review.md
 M docs/notebooks/notebook_execution_integrity.md
 M tests/unit/test_notebook_placeholders.py
```

Audits 0030, 0031 and 0032 are committed; 0032 carries uncommitted edits from the previous
milestone recording the authoritative full-artifact validation. Nothing in this milestone
overwrites a committed audit. The next available number is **0033**.

## 3. Checkpoint and corpus evidence verified before use

```text
checkpoint  checkpoint/s1_mention_full_training_v1/best.pt   (git-ignored)
SHA-256     a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c   verified
mode FULL_TRAINING | epoch 4 | global_step 2976
pinned revision f89e80b461e86f9cfc1c84019bd819830c24b6c5
governed corpus  corpus_manifest_sha256 a3fd365d…fb8e3  PASS (counts and hashes unchanged)
```

The checkpoint gate ran before inference; SHA-256 and mtime were re-verified afterwards and are
unchanged. Inference was forward-only on CPU: `model.eval()`, `torch.no_grad()`, every parameter
`requires_grad_(False)`, no optimizer, no scheduler, no backward. The backbone was rebuilt from
the cached `config.json` and the checkpoint's own complete state dict, so **no base weights were
downloaded**.

## 4. The evaluator

`src/mednorm_vi/evaluation/exact_mention.py`.

Mentions are compared on exact `(start, end, type)` over `original_text`, and both gold and
predicted mentions are validated against the §4 invariant before anything is counted — an
invariant violation raises rather than being repaired. Type must be one of the five organizer
types; the span must be in range and end-exclusive.

**No fuzzy matching.** A prediction whose text merely resembles a gold mention scores as a false
positive *and* a false negative. It is additionally *categorised* as a boundary error so the
failure is explainable, but it never earns partial credit.

Error taxonomy: `exact_match`, `missed`, `spurious`, `wrong_type`, `left_boundary`,
`right_boundary`, `both_boundary`, `duplicate_overlap`.

**Wrong type is double-penalised exactly as spec §1 requires**: a false positive charged to the
*predicted* type and a false negative charged to the *gold* type.

Groupings: entity type, route, section, source, entity-length bucket, and
deterministic/neural/hybrid provenance. Outputs: machine-readable JSON plus a compact Markdown
report, both carrying the resolved config hash. Diagnostics use a salted-free SHA-256 handle
(`privacy_safe_example_id`) and never contain clinical text — asserted by test.

The report states in its own payload that it **does not reproduce the complete organizer
score**: assertions (Jaccard, 30%) and ICD/RxNorm candidate sets (Jaccard, 40%) are not
integrated, so this is a mention-level component only.

## 5. Executed result — governed internal_test, S1 neural predictions

Token-level predictions were decoded back to character spans through the tracked alignment
backend (segmented words → subtokens → original offsets), then contiguous positive runs per type
were emitted as mentions.

```text
examples          1045      (all supervised internal-test rows; 0 skipped)
gold mentions     2046
predicted         1854

EXACT precision   0.7470
EXACT recall      0.6769
EXACT F1          0.7103
```

| Category | Count |
| --- | --- |
| exact_match | 1385 |
| missed | 385 |
| spurious | 193 |
| right_boundary | 140 |
| left_boundary | 87 |
| both_boundary | 26 |
| wrong_type | 23 |
| duplicate_overlap | 0 |

Per-type exact F1: DIAGNOSIS **0.7621**, MEDICATION **0.7448**, SYMPTOM **0.5793**,
TEST_NAME **0.0**, TEST_RESULT **0.0**.

**The headline correction.** The previously reported held-out figure of **0.746182** is a
token-index span proxy. Measured on exact character offsets the same checkpoint scores
**0.7103** — 0.0359 lower. The exact number is the one that relates to the organizer's
convention, and it should supersede the proxy in status reporting. Neither is the complete
organizer score.

TEST_NAME and TEST_RESULT score 0.0 because the governed corpus contains **no supervision** for
them. That is an absent-data fact, not a model failure.

## 6. SYMPTOM error taxonomy (real, from the executed run)

Gold-side SYMPTOM failures, 255 in total:

| Category | Count | Share |
| --- | --- | --- |
| missed (complete) | 140 | 54.9% |
| right_boundary (too long/short on the right) | 49 | 19.2% |
| left_boundary | 37 | 14.5% |
| wrong_type | 15 | 5.9% |
| both_boundary | 14 | 5.5% |

Plus 93 spurious SYMPTOM predictions with no gold counterpart.

Wrong-type confusions are almost entirely the DIAGNOSIS/SYMPTOM boundary the spec flags as
double-penalised: gold SYMPTOM predicted DIAGNOSIS **15**, gold DIAGNOSIS predicted SYMPTOM
**8**.

The finer categories the milestone asked for — treatment-purpose phrase, section/router error,
overlap competition, deterministic-evidence-absent — are **not** reported, because attributing
them requires the L2 route tags and L3 lattice provenance that deliverable B would supply.

## 7. The quantified L4 opportunity

Boundary errors total **253 of 2046 gold mentions (12.4%)**: DIAGNOSIS 135, SYMPTOM 100,
MEDICATION 18. Every one is currently scored as a full miss *and* a full false positive.

This is the concrete, measured case for the L4 Boundary & Type Resolver: a resolver that fixed
even half of them would move exact F1 materially, and it is now measurable rather than assumed.
No claim is made that a resolver will achieve this — only that the opportunity is real and
quantified.

## 8. Tests and static checks

`tests/unit/test_exact_mention_evaluator.py` (29): the §4 invariant and end-exclusive offsets;
out-of-range and unsupported types rejected; decomposed Unicode; repeated spaces and newlines;
exact match; wrong type charged as FP on predicted **and** FN on gold; left/right/both boundary
categories that never earn credit; a near miss never silently accepted; missed vs spurious
separated; duplicate/overlap; repeated identical text at different offsets kept separate (no
text-only deduplication) in both directions; adjacent spans; grouping by provenance, length,
source, route and section; the report asserting it is not the complete organizer score;
diagnostics carrying hashed ids and no clinical text; deterministic config hash.

```text
env PYTHONPATH=src python3 -m pytest -q          -> 1150 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 225 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
governed corpus hashes                            -> unchanged
architecture PDF                                  -> no diff
checkpoint SHA-256 and mtime                      -> unchanged after inference
model weights tracked                             -> 0
generated reports                                 -> ignored
```

No notebook was modified in this milestone, so no notebook cells needed re-parsing.

## 9. Exact changed tracked files

**Added (3):**
```text
docs/audits/0033-s1-exact-mention-evaluator.md
src/mednorm_vi/evaluation/exact_mention.py
tests/unit/test_exact_mention_evaluator.py
```

**Modified (1):**
```text
docs/audits/README.md                              (0033 index entry)
```

Carried over uncommitted from the previous milestone (unchanged here):
`docs/audits/0032-…md`, `docs/notebooks/notebook_execution_integrity.md`,
`tests/unit/test_notebook_placeholders.py`.

## 10. Ignored reports and local assets

```text
reports/s1_exact_mention/internal_test_exact_mention.json   ignored (reports/**)
reports/s1_exact_mention/internal_test_exact_mention.md     ignored (reports/**)
reports/s1_internal_test_eval/, reports/s1_best_checkpoint/ ignored
checkpoint/s1_mention_full_training_v1/best.pt   1.6 GB     ignored (/checkpoint/)
.venv/                                                      ignored
```

## 11. Limitations and remaining blockers

1. **B, C, D and F were not started**, and F cannot be produced without B and C.
2. The exact evaluator measures mention span and type only — **not** the organizer score.
3. TEST_NAME and TEST_RESULT have no governed supervision; their 0.0 is undefined, not poor.
4. SYMPTOM remains the weakest supervised type (exact F1 0.5793), dominated by complete misses.
5. The SYMPTOM taxonomy lacks route/section/treatment-phrase attribution pending B.
6. S2–S6 remain untrained; L4, L6, L7, L8 remain scaffolds.
7. No organizer inference and no `output.zip`.

## 12. Honest status

`EXACT_EVALUATOR_IMPLEMENTED_AND_EXECUTED — L3_LATTICE_AND_L4_RESOLVER_NOT_STARTED`.

The exact character-offset mention evaluator exists, is unit-tested, and has been executed on
the governed internal-test split with the validated S1 checkpoint, producing exact F1 **0.7103**
and a measured 12.4% boundary-error rate that quantifies the L4 opportunity. The unified L3
lattice, the L4 resolver v1, the laboratory synthetic suite and the four-way ablation remain to
be built. No training, retraining, checkpoint mutation, organizer inference or `output.zip`
occurred.
