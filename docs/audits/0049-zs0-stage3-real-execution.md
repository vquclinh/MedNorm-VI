# Audit 0049 - ZS0 Stage 3 Real Execution

Date: 2026-07-28

## 1. Objective and Scope

A fresh `Runtime -> Run all` with organizer inference disabled failed at Stage 4:

```text
SystemExit: Stage 3 produced no arm results
```

That is a real implementation bug, and this audit fixes it by implementing what
was missing rather than by relaxing the guard — the guard was correct.

No training ran. `internal_test` was never opened. Organizer inference did not
run. No `output.zip` was created. E4 remains retired; no E3, no random E5 head,
no external API. The architecture PDF is unchanged.

**No ZS0 metrics are claimed.** Neural inference has not been re-run on Colab.

### Status

| Subject | Status |
| --- | --- |
| Root cause | **PROVEN** — Stage 3 was a placeholder |
| Stage 3 | `IMPLEMENTED` — real E1/E2, GLiNER and Qwen paths |
| Stage-4 guard | **unchanged** — it was right |
| ZS0 metrics | **none** — not yet re-run |

## 2. Proven Root Cause

Stage 3 as it shipped, in full after the corpus resolution:

```python
# Load the pretrained models here (GLiNER, Qwen, embedder), count their
# parameters into COUNTED, then run each arm once over the validation split
# ...
ARM_RESULTS = []   # -> list[ArmResult]
print(json.dumps({"stage": "stage3_ready",
                  "arms": [a.name for a in all_arms()],
                  "note": "populate ARM_RESULTS with one ArmResult per arm"},
                 indent=2, sort_keys=True))
```

**Stage 3 was a pure placeholder.** It resolved the validation split, bound
`ARM_RESULTS` to an empty list, and printed a note instructing the reader to
populate it. It loaded no model, ran no arm and appended nothing. Stage 4's
`if not ARM_RESULTS: raise SystemExit(...)` then fired on an empty list — exactly
as designed.

### 2.1 Every alternative cause, checked and excluded

Measured against the shipped cell, not assumed:

```text
Stage 3 references RUN_ORGANIZER_INFERENCE   False  -> not organizer-gated
Stage 3 contains try/ except                 False  -> nothing swallowed
Stage 3 contains any `if` gate               False  -> not flag-gated
Stage 3 calls .append(                       False  -> nothing to append
`ARM_RESULTS =` assignments in the notebook       1  -> never reset after filling
Stage 3 loads a model                        False  -> the only "model" text was
                                                       the comment saying to
```

So it was not a false flag, not the organizer flag, not a swallowed exception and
not a silent skip. **The validation runner was never called because it did not
exist.** No such function was anywhere in `src/mednorm_vi/zs0/`.

Audit 0048 delivered ZS0's *contracts* — offset verification, substring
resolution, candidate constraints, the ledger, the gates — and every one of them
is tested and correct. What it never delivered was the code that calls a model.
The audit's own §5 said the arms had not run; what it did not say, because I did
not notice, was that they *could not*.

## 3. Implementation Added

### 3.1 `src/mednorm_vi/zs0/backends.py` — the real inference paths

`GLiNERBackend` and `QwenBackend`, both self-hosted, both importing their heavy
dependency lazily inside `load()` so the module still imports on a laptop with
neither `gliner` nor a GPU. That is what lets the contract tests run locally
while the models run on Colab.

```text
GLiNERBackend   gliner.GLiNER.from_pretrained -> eval() -> predict_entities
                under torch.no_grad(); prompted with five labels that map into
                the five BTC types; offsets returned are NOT trusted — every
                span goes through convert_gliner_spans and is rejected if it
                does not reproduce the original text
QwenBackend     AutoModelForCausalLM + AutoTokenizer -> eval() -> generate
                under torch.no_grad(), do_sample=False (greedy, reproducible);
                one instance serves every ZS0 Qwen duty
```

`build_mention_prompt` states the rules the parser then enforces: verbatim
substrings only, the five types only, an `anchor` when a mention repeats, and
`{"mentions": []}` when unsure. `extract_json_object` recovers the first balanced
object from a fenced or prefixed completion — that is recovery, not repair:
anything malformed still fails the schema and substring checks with a reason
code.

Neither backend trains. A test asserts no `backward`, `AdamW`, `optim` or
`zero_grad` token appears in either, and that `torch.no_grad()`, `.eval()` and
`do_sample=False` do.

### 3.2 `src/mednorm_vi/zs0/runner.py` — the missing runner

```text
Document        one governed document: id, text, gold spans
ArmSources      the callables one arm uses; a missing one is REFUSED, not skipped
run_arm         executes one arm over every document -> exactly one ArmResult
run_all_arms    every arm in order, or a loud failure
assert_stage3_complete   the postcondition, checked where results are produced
```

Two behaviours matter more than the metrics:

* **an arm that predicts nothing still returns an `ArmResult`** with zero
  metrics. Zero predictions is a measurement; a missing result is a bug, and the
  two must never look the same;
* **a failing arm raises `ArmExecutionError`** naming the arm and the exact
  prerequisite. There is deliberately no `try/except` around an arm in
  `run_all_arms`: catching one and continuing would leave Stage 4 a short list —
  which is precisely how the placeholder presented itself.

Metrics are real: exact span-and-type precision/recall/F1 against gold, per-type
counts, wrong-type count (right span, wrong type), malformed-proposal count from
the rejection ledger, offset violations, runtime and peak VRAM.

### 3.3 Stage 3, rewritten

```text
E1 + E2   load_l1_config -> analyze_document -> run_phase1b, the project's real
          deterministic path; each proposal is re-checked against the document
          text before it is converted
GLiNER    GLiNERBackend(...).load(); parameter_count() feeds COUNTED
Qwen      QwenBackend(...).load(); parameter_count() feeds COUNTED
ledger    rebuilt with the counts just measured and enforced at 9B
arms      ARM_RESULTS = run_all_arms(all_arms(), DOCUMENTS, ARM_SOURCES, ...)
          assert_stage3_complete(ARM_RESULTS, all_arms())
```

`ARM_RESULTS` is now assigned exactly once, from the runner — a test asserts the
only assignment in the notebook is `ARM_RESULTS = run_all_arms(`.

The tracked ledger still ships `parameter_count: null`, so the gate fails closed
until Stage 3 counts the checkpoints it actually loaded.

## 4. What Was Not Changed

The Stage-4 guard is untouched. It correctly refused to build a comparison table
from nothing, and weakening it would have converted a missing implementation into
a silent empty report.

## 5. Constraints Held

```text
E4 retired                     yes - untouched, still RETIRED_FROM_ACTIVE_STACK
E3                             absent - refused by name in FORBIDDEN_COMPONENTS
random E5 head                 absent - refused by name
training / backward / optimizer NONE - asserted by token scan over the package,
                               the runner, the backends and Stage 3
internal_test                  never opened
external API                   none
organizer inference            did not run
output.zip                     not created
active model total             <= 9B, enforced by build_ledger(enforce=True)
```

## 6. Tests and Static Checks

`tests/unit/test_zs0_baseline.py` grew from 80 to 105 tests. The new ones prove:
Stage 3 produces exactly three real arm results and Stage 4 receives all three;
an arm with zero predictions still returns a result, and so does every arm when
all of them predict nothing; a missing backend fails loudly and names the arm and
prerequisite; an arm exception propagates and is never swallowed; a failing arm
stops the whole run rather than shortening it; an arm with no sources and a run
with no documents are both refused; the Stage-3 postcondition catches a short
result set; metrics are real and move with the predictions; a right span with the
wrong type is counted as wrong-type; malformed model output is counted rather
than crashed on; Stage 3 references no organizer flag and no
`assert_organizer_inference_allowed`; Stage 3 is no longer a placeholder and
contains no `try/except` around the arm run; `ARM_RESULTS` is assigned exactly
once from the runner; the backends import without their heavy dependencies and
refuse to predict or count when unloaded; the prompt forbids invention and names
the types; a fenced completion is recovered without its content being repaired;
the backends never train; and no forbidden expert or training path is reachable
from the runner or from Stage 3.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q            1783 passed, 1 skipped
tests/unit/test_zs0_baseline.py                             105 passed
ruff check .                                                All checks passed
ruff check notebooks                                        All checks passed
env PYTHONPATH=src .venv/bin/python -m mypy                 Success: no issues found
                                                            in 282 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src    clean
git diff --check                                            clean
```

## 7. Changed Files

```text
A  docs/audits/0049-zs0-stage3-real-execution.md
M  docs/audits/README.md
A  src/mednorm_vi/zs0/backends.py
A  src/mednorm_vi/zs0/runner.py
M  src/mednorm_vi/zs0/__init__.py
M  notebooks/MedNorm_ZS0_Baseline_Submission.ipynb
M  tests/unit/test_zs0_baseline.py
M  pyproject.toml                (mypy: gliner is a Colab-only import)
```

## 8. Limitations

* **ZS0 has still not been run.** No arm result, no selection, no `output.zip`.
  Every number remains pending.
* The Qwen cascade currently routes a whole document to Qwen when the cheaper
  experts produced nothing at all (`uncertain_when_fewer_than = 1`). That is the
  simplest robust routing rule, not a tuned one; the Stage-4 table will show
  whether ZS0-C earns its parameters.
* `GLINER_PROMPT_LABELS` and the label mapping are our judgement, not the
  organizer's. Stage 4's per-type metrics are what test them.
* The GLiNER threshold (0.35) is a conventional default, fixed once and
  configurable; no search was performed.
* Stage 3 writes each document to a scratch file because `analyze_document` takes
  a path. That is a real dependency of the existing L1 contract, not a shortcut.

## 9. Exact Fresh-Colab Instructions

```text
0.  New Colab notebook -> Runtime -> Change runtime type -> T4 (or better) GPU.
    Open notebooks/MedNorm_ZS0_Baseline_Submission.ipynb from the updated repo.
    Leave every organizer flag at its committed value:
        RUN_ORGANIZER_INFERENCE = False
        CONFIRM_ORGANIZER_INFERENCE = ""

1.  Runtime -> Run all.

    Stage 0   installs gliner / transformers / sentence-transformers; prints
              the environment and GPU.
    Stage 1   verifies configs, asserts E4 is retired, hashes them, and reports
              the ledger as PENDING (it fails closed until Stage 3 counts).
    Stage 2   runs the contract smoke tests; it raises if any fails.
    Stage 3   loads GLiNER and Qwen, COUNTS both, rebuilds and ENFORCES the
              ledger, then runs ZS0-A, ZS0-B and ZS0-C once over the governed
              validation split. Expect:
                  "stage3_complete", arm_results: 3,
                  arms: ["ZS0-A", "ZS0-B", "ZS0-C"]
    Stage 4   prints the comparison table.
    Stage 5   selects an arm deterministically.
    Stage 6   STOPS: organizer inference is disabled.

2.  If Stage 3 raises, read the message. It names the arm and the exact
    prerequisite — for example "ZS0 arm ZS0-B could not run: the pretrained
    GLiNER backend is not loaded". Fix that prerequisite; do not rerun blindly.

3.  Return with the Stage-4 table and the Stage-5 selection. Organizer inference
    remains a separate, explicitly authorized step.
```

## 10. Safe-to-Commit Verdict

Safe to commit after review. The architecture PDF and Audits 0043-0048 are
byte-identical; the Stage-4 guard is unchanged; E4 stays retired; no E3, no
random E5 head, no training, no `internal_test`, no external API, no organizer
inference and no `output.zip`; no assistant control file exists; nothing
artifact-shaped is staged; no Git history was rewritten.

```bash
git add \
  docs/audits/0049-zs0-stage3-real-execution.md \
  docs/audits/README.md \
  src/mednorm_vi/zs0/backends.py \
  src/mednorm_vi/zs0/runner.py \
  src/mednorm_vi/zs0/__init__.py \
  notebooks/MedNorm_ZS0_Baseline_Submission.ipynb \
  tests/unit/test_zs0_baseline.py \
  pyproject.toml

git commit -m "fix: implement real ZS0 Stage-3 arm execution"
git push origin main
```
