# Audit 0048 - E4 Retirement and the ZS0 Zero-Shot Submission Baseline

Date: 2026-07-28

## 1. Objective and Scope

Two owner decisions, implemented together: E4 PhoBERT-W2NER is retired from the
active stack, and the next deliverable is a pure zero-shot/pretrained baseline
(ZS0) that can produce a valid organizer `output.zip`.

**Nothing was run on Colab in this milestone.** ZS0-A/B/C have not been executed,
no arm has been selected, and no `output.zip` exists. What this milestone
delivers is the implementation, the gates and the notebook that produce those
results — reported honestly as pending in §5, §6 and §14.

No training ran locally. `internal_test` was never opened. Organizer inference
did not run. No assistant control file was created. The architecture PDF is
unchanged.

### Status

| Subject | Status |
| --- | --- |
| E4 | `RETIRED_FROM_ACTIVE_STACK` |
| E4 Stage 3 / Stage 4 | `PERMANENTLY_BLOCKED` |
| ZS0 implementation | `IMPLEMENTED_NOT_RUN` |
| ZS0 arm selection | **pending** — Stage 3 has not executed |
| `output.zip` | **does not exist** |
| Leaderboard score | **no claim** — that belongs to the organizer |

## 2. E4 Retirement

### 2.1 The evidence it rests on

The completed Stage-2 ablation, recorded exactly in
`governance/e4_retirement.STAGE2_FINAL_RESULT`:

```text
examples                    12
gold mentions               22
all required types present  yes
epochs per recipe           200 / 200
optimizer steps per recipe  600 / 600
warmup fully served         yes
peak learning rates         reached
save/reload reproduction    PASSED for every recipe

reference_ce         best exact F1  0.3448
group_balanced_ce    best exact F1  0.7333
hard_negative_ce     best exact F1  0.3704

gate                 exact F1 >= 0.95
any recipe passed    NO
selected_recipe      null
Stage 3 / Stage 4    blocked
```

This is a **valid completed run**, not a runtime failure. Audit 0047's diagnosis
was correct — `group_balanced_ce` more than doubled the baseline, confirming that
background dominance was the binding constraint — and it still fell well short of
memorizing twelve examples.

### 2.2 What retirement does, and does not do

**Withdrawn:** active and default feature flags; active inference registries; the
deployment parameter ledger; any expectation that an E4 checkpoint exists;
Stages 3 and 4.

**Preserved:** `src/mednorm_vi/training/phase2/e4/`, its tests, its notebook, its
config, and Audits 0043-0047. A test asserts every preserved path still exists.

Concretely:

```text
configs/pipeline/full_v1.yaml      enable_e4_phobert_w2ner: false (already), and
                                   mention/phobert_w2ner REMOVED from
                                   full_requires_checkpoints — a retired expert
                                   has no checkpoint and none is expected
configs/model_registry/models_v1.yaml   status: RETIRED_FROM_ACTIVE_STACK
configs/models/candidate_model_registry.yaml
                                   status: RETIRED_FROM_ACTIVE_STACK,
                                   retired_by_audit "0048",
                                   best_measured_tiny_exact_f1 0.7333,
                                   in_active_deployment: false
configs/models/zs0_parameter_ledger.yaml   listed under `excluded`, never counted
```

`assert_stage_not_forbidden` raises for `subset_smoke` and `full_training` and
permits `tiny_recipe_ablation` — the stage that produced the evidence stays
runnable as history.

The architecture PDF describes a candidate **super**-architecture; the active
runtime stack is a validated subset of it. Retiring E4 changes the subset, not
the document.

## 3. Active ZS0 Architecture

```text
ZS0-A   E1 medication grammar + E2 laboratory parser
ZS0-B   ZS0-A + pretrained GLiNER
ZS0-C   ZS0-B + constrained Qwen proposer on uncertain segments

resolution   conservative deterministic L4 (zs0-conservative-resolver-v1)
assertions   deterministic cue+scope first; Qwen only where uncertain
linking      alias -> lexical -> dense -> optional reranker; Qwen SELECTS ONLY
packaging    validated, hashed, exactly 100 root-level JSON files
```

**Excluded and enforced by name** (`FORBIDDEN_COMPONENTS`, each with a test):

* **E3** — fine-tuned in this project. It is the best mention expert the project
  has (exact F1 0.7103, Audit 0033), and it is excluded precisely because ZS0
  exists to measure what a *pretrained-only* stack achieves.
* **E4** — retired.
* **E5** — its MRC task head is randomly initialized. An untrained head is not
  zero-shot; it is noise behind a confident interface.
* **learned L4 v2** — no trained checkpoint. Deterministic L4 v1 is also off:
  Audit 0034 measured it *below* the E3-only baseline (0.7039 against 0.7103).

Also enforced: no backward pass, optimizer or scheduler; no external API; no
`internal_test`.

## 4. Model Revisions and Parameter Ledger

`configs/models/zs0_parameter_ledger.yaml` ships every pretrained component with
`parameter_count: null` and `count_verified: false`, so `build_ledger` **fails
closed** until Stage 1 counts each checkpoint programmatically on Colab.

```text
component          model                          active  concurrent  counted
e1_medication      deterministic                  yes     no          0 (verified)
e2_laboratory      deterministic                  yes     no          0 (verified)
e6_gliner          urchade/gliner_medium-v2.1     yes     yes         PENDING
e7_qwen_cascade    Qwen/Qwen3-1.7B                yes     yes         PENDING
dense_embedder     BAAI/bge-m3                    yes     no          PENDING
reranker           Qwen/Qwen3-Reranker-0.6B       no      no          PENDING
```

Two rules the ledger enforces, both from Audit 0042:

* a LoRA deployment counts the **base plus** its adapters, never the adapter
  alone;
* a shared backbone is counted **once** — one Qwen instance serves hard mention
  proposal, assertion adjudication and candidate selection, which is why the
  smallest supported model is preferred and Qwen3-4B is **not** added merely
  because the super-architecture lists it.

**Revisions are not yet pinned.** They are resolved and recorded by Stage 1
before any inference; the ledger ships them empty rather than guessed.

## 5. ZS0-A / B / C Comparison

**Not yet run.** Stage 3 executes each arm exactly once on the governed
validation split and Stage 4 prints the single comparison table: exact
precision/recall/F1, per-type metrics, wrong-type count, assertion metrics where
labels exist, ICD/RxNorm candidate diagnostics where labels exist, malformed
proposal count, offset violations, runtime, peak VRAM and active parameter total.

No numbers are reported here because none have been measured.

## 6. Selected Submission Arm

**None.** Selection requires Stage 3 results.

The rule is fixed in code (`submission.SELECTION_RULE`) so the choice is
mechanical when the numbers arrive:

> Among arms with **zero offset violations**, take the highest exact F1. Break a
> tie within 0.005 F1 by fewest wrong-type errors, then fewest malformed
> proposals, then smallest active parameter total, then the simpler arm
> (A < B < C).

An arm with any offset violation is inadmissible at any score: a violated offset
invariant means a malformed submission, and no F1 compensates for that. The 0.005
tolerance exists so a negligible F1 difference does not buy a larger model.

## 7. Schema, Offset and Ontology Validation

Every mention path is held to spec §4: `original_text[start:end] == text`,
end-exclusive.

```text
GLiNER      returns offsets; they are VERIFIED against the original text and the
            span is rejected outright if they disagree
Qwen        returns STRINGS ONLY. Never trusted with offsets. Every mention must
            be a literal substring of the segment it was shown; position is
            resolved by deterministic exact matching
ambiguity   a repeated substring is REJECTED unless an anchor identifies exactly
            one occurrence — guessing which "sốt" was meant would fabricate
            evidence
types       a mention outside the five BTC types is rejected
codes       Qwen may only SELECT from the retrieved candidate set; a code it
            returns that was not offered is dropped even if it exists in the
            snapshot, and every emitted code is re-checked against the locked
            snapshot before it leaves the linker
fallback    exact alias hit wins outright; else calibrated top-k; else EMPTY.
            Empty beats invented.
```

Rejections are counted by reason code and never carry clinical text.

Package validation checks: exact filenames `1.json`..`100.json`, valid JSON,
allowed organizer labels only, end-exclusive positions, text equality, assertion
fields only where allowed, candidate codes present in the locked snapshot, no
duplicate records, deterministic ordering, and a ZIP containing only those 100
files at its root.

## 8. Tests and Static Checks

`tests/unit/test_zs0_baseline.py` — 80 tests. They prove: E4 is disabled in every
pipeline profile and no profile requires its checkpoint; enabling it raises;
Stages 3 and 4 can never run; E4 is absent from the ledger; the Stage-2 result is
recorded exactly; E4 source and audits are preserved; ZS0 cannot load E3, E4 or a
random E5 head; no arm contains a trained expert; ZS0 refuses training and
external APIs; the ZS0 package never trains and never opens the frozen split;
GLiNER spans preserve exact substrings and offsets; unmappable labels and
out-of-range spans are rejected; Qwen may only return literal substrings; a
repeated substring fails closed without an anchor; unsupported types and
malformed output are rejected; the rejection ledger holds no clinical text; the
resolver preserves provenance, keeps identical text at different positions as two
mentions, is deterministic and sorted, and abstains on thin evidence; assertions
are constrained to the three labels and stay empty when uncertain; Qwen cannot
introduce a code outside the candidate set; every emitted code exists in the
snapshot; cross-ontology linking is refused; the ledger fails closed above 9B and
on any unverified active component; a LoRA counts base plus adapters; a shared
backbone counts once; arm selection excludes offset violations and does not buy a
larger model for a negligible gain; organizer inference requires the exact
authorization string and each of the eight preconditions; packaging requires
exactly 100 root-level JSON files; and no assistant control file is tracked.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q            1758 passed, 1 skipped
tests/unit/test_zs0_baseline.py                             80 passed
ruff check .                                                All checks passed
ruff check notebooks                                        All checks passed
env PYTHONPATH=src .venv/bin/python -m mypy                 Success: no issues found
                                                            in 280 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src    clean
git diff --check                                            clean
```

### 8.1 Two supporting changes

`governance/parameter_budget.py` gained `RETIRED_FROM_ACTIVE_STACK` as a valid
registry status. It is deliberately distinct from `EXCLUDED_BY_ABLATION`:
exclusion is a measurement outcome, retirement is an owner decision recorded on
top of one. Both are listed in the new `NON_DEPLOYABLE_STATUSES`.

`tests/unit/test_notebooks.py` registers the ZS0 notebook and its audit — the
mechanism that file uses for every notebook added after Audit 0017. No assertion
was weakened.

## 9. Limitations

* **ZS0 has not been run.** No arm result, no selection, no `output.zip`. Every
  number in §5 and §6 is pending.
* Model revisions are unpinned in the tracked ledger. Stage 1 pins and counts
  them; until then the budget gate fails closed by design.
* The GLiNER label mapping is ours, not the organizer's. It is recorded in
  `GLINER_LABEL_TO_TYPE` and is a judgement that Stage 4's per-type metrics will
  test.
* The assertion cue lexicons are small and high-precision by choice. They will
  miss cues; the fallback is an empty assertion set, which is the safe error.
* No assertion or linking metric can be computed where the governed corpus has no
  labels — it has **zero** assertion supervision (Audit 0042).
* The resolver thresholds are fixed conservative defaults, not a search. With no
  trained resolver, searching them on the validation split would fit the metric
  rather than the problem.
* ZS0 is a *baseline*. Its purpose is a valid submission and a floor to measure
  against, not a competitive score.

## 10. Exact Fresh-Colab Commands

```text
0.  New Colab notebook -> Runtime -> Change runtime type -> T4 (or better) GPU.
    Open notebooks/MedNorm_ZS0_Baseline_Submission.ipynb from the updated repo.

1.  Runtime -> Run all.
    Stage 0 installs pinned dependencies and prints the environment and GPU.
    Stage 1 verifies configs, pins revisions, hashes ontology snapshots and
    builds the parameter ledger. Stage 2 runs the contract smoke tests.
    Stages 3-5 run the three arms once, print the comparison table and select
    the arm. Stage 6 stops at the organizer gate: every flag ships False.

2.  Read the Stage-4 table and the Stage-5 selection before going further.
    If Stage 5 selects nothing, STOP — no submission may be built.

3.  Only then, for the authorized run, set in the Stage-0b cell:
        RUN_ORGANIZER_INFERENCE = True
        CONFIRM_ORGANIZER_INFERENCE = "I_AUTHORIZE_ZS0_ORGANIZER_INFERENCE_AND_PACKAGE"
    and place the 100 organizer .txt files at
        /content/drive/MyDrive/MedNorm-VI/organizer/round1_input/
    Runtime -> Run all.

4.  Stage 6 re-checks all eight preconditions before reading a document.
    Stage 7 validates every output, packages output.zip, and prints its SHA-256
    together with the per-file hashes and the run manifest.

5.  Return with zs0_run_manifest.json. Upload output.zip yourself.
```

## 11. Exact Organizer Inference Settings

```text
RUN_ORGANIZER_INFERENCE       = True
CONFIRM_ORGANIZER_INFERENCE   = "I_AUTHORIZE_ZS0_ORGANIZER_INFERENCE_AND_PACKAGE"

preconditions, all required
  selected ZS0 arm is non-null
  no E3 / E4 / E5 active
  no training executed
  parameter budget passed (active total <= 9,000,000,000, every count verified)
  every local schema/offset/ontology gate passed
  all 100 organizer TXT files discovered exactly once
  no external API configured
  deterministic seed and run manifest recorded
```

## 12. `output.zip` Path and SHA-256

```text
path      /content/zs0_output/../output.zip   (written beside the output directory)
sha256    NOT YET PRODUCED
```

The authorized run has not happened. Stage 7 prints the digest and writes it into
`zs0_run_manifest.json`; it will be recorded in the next audit.

**No leaderboard score is claimed.** That belongs to the organizer, after the
human uploads `output.zip`.

## 13. Changed Files

```text
A  docs/audits/0048-e4-retirement-and-zs0-zero-shot-baseline.md
M  docs/audits/README.md
A  src/mednorm_vi/governance/e4_retirement.py
A  src/mednorm_vi/zs0/__init__.py
A  src/mednorm_vi/zs0/assertions.py
A  src/mednorm_vi/zs0/ledger.py
A  src/mednorm_vi/zs0/linking.py
A  src/mednorm_vi/zs0/profile.py
A  src/mednorm_vi/zs0/proposals.py
A  src/mednorm_vi/zs0/resolver.py
A  src/mednorm_vi/zs0/submission.py
A  configs/pipeline/zs0_baseline.yaml
A  configs/resolution/zs0_conservative_v1.yaml
A  configs/models/zs0_parameter_ledger.yaml
A  notebooks/MedNorm_ZS0_Baseline_Submission.ipynb
A  tests/unit/test_zs0_baseline.py
M  configs/pipeline/full_v1.yaml
M  src/mednorm_vi/governance/parameter_budget.py
M  tests/unit/test_notebooks.py
M  configs/model_registry/models_v1.yaml
M  configs/models/candidate_model_registry.yaml
```

## 14. Safe-to-Commit Verdict

Safe to commit after review. The architecture PDF and Audits 0043-0047 are
byte-identical; E4 source, tests, notebook and config are preserved; no assistant
control file exists; no artifact, checkpoint or cache is staged; no training ran;
`internal_test` was not opened; organizer inference did not run; no Git history
was rewritten.

```bash
git add \
  docs/audits/0048-e4-retirement-and-zs0-zero-shot-baseline.md \
  docs/audits/README.md \
  src/mednorm_vi/governance/e4_retirement.py \
  src/mednorm_vi/zs0/ \
  configs/pipeline/zs0_baseline.yaml \
  configs/pipeline/full_v1.yaml \
  configs/resolution/zs0_conservative_v1.yaml \
  configs/models/zs0_parameter_ledger.yaml \
  configs/models/candidate_model_registry.yaml \
  configs/model_registry/models_v1.yaml \
  notebooks/MedNorm_ZS0_Baseline_Submission.ipynb \
  tests/unit/test_zs0_baseline.py

git commit -m "feat: retire E4 and add the ZS0 zero-shot submission baseline"
git push origin main
```
