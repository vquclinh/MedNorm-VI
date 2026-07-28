# Audit 0050 - ZS0 Stage-3 Import and Public-API Repair

Date: 2026-07-28

## 1. Objective and Scope

A fresh Colab `Runtime -> Run all` on the Audit-0049 notebook failed immediately
at Stage 3:

```text
ImportError: cannot import name 'load_l1_config'
from 'mednorm_vi.document_intelligence'
```

That is a real integration bug. This audit proves the symbol's location, applies
the smallest correct fix, and — because the first real run exposed one bad
import — audits **every** Stage-3 import and call signature in one pass rather
than one Colab run at a time.

That sweep found **three further defects** behind the reported one. All four are
fixed and locally proven.

No training ran. `internal_test` was never opened. Organizer inference did not
run. No `output.zip` was created. No Stage-3 or Stage-4 guard was weakened. E4
remains retired. The architecture PDF is unchanged.

**No ZS0 metrics are claimed.** The repaired workflow has not completed on Colab.

## 2. The Symbol, Proven

```text
grep -rn "def load_l1_config" src/          NO DEFINITION FOUND
```

`load_l1_config` **does not exist anywhere in the source tree**. It is a *local
alias* that every internal caller gives to the real function:

```text
src/mednorm_vi/inference/pipeline.py:12
src/mednorm_vi/corpus_analysis/pipeline.py:16
src/mednorm_vi/deterministic_baseline/cli.py:19
src/mednorm_vi/phase1c_foundation/cli.py:163
    from ..document_intelligence.builder import load_config as load_l1_config
```

The real function:

```text
defined   src/mednorm_vi/document_intelligence/builder.py:271
          def load_config(path: str | Path) -> tuple[L1Config, SectionLexicon]
exported  src/mednorm_vi/document_intelligence/__init__.py — imported at line 22
          and listed in __all__
```

Verified at runtime:

```text
document_intelligence exports load_config   True
load_config in __all__                      True
document_intelligence has load_l1_config    False
load_config.__module__                      mednorm_vi.document_intelligence.builder
```

`analyze_document` is defined in the package `__init__` itself and is likewise in
`__all__`, which is why it imported cleanly and `load_l1_config` did not.

### 2.1 Exact cause

I wrote the Stage-3 import block from how `inference/pipeline.py` *uses* these
names, not from what the packages *export*. That file aliases `load_config` to
`load_l1_config` for local readability; I copied the alias and imported it as
though it were the public name. The package was never at fault.

## 3. Public-API Decision

**Option B-shaped, from the public package: import the real exported name.**

Option A — re-exporting `load_l1_config` from `document_intelligence.__init__` —
was rejected. `load_config` is *already* the public API: it is in `__all__` and
has been imported by four internal callers for many audits. Adding a second
public name for the same function would create exactly the duplication the
milestone forbids, and would enshrine a private alias as an API.

So the fix imports `load_config` by its real name and drops the alias in the
notebook entirely — there is no naming conflict there to justify one.

## 4. All Stage-3 Imports Audited

The block was extracted from the notebook and executed in a **clean subprocess**
with `PYTHONPATH=src`. That surfaced a second failure the reported error had been
masking, and the field/label audit that followed surfaced two more.

```text
name                             stated module                        verdict
run_phase1b                      deterministic_baseline.pipeline      OK (moved)
analyze_document                 document_intelligence                OK
load_l1_config                   document_intelligence                BUG 1
SpanProposal                     mention_factory.models               removed
Phase1BConfig                    phase1c_foundation                   BUG 2
E4_GOVERNED_VALIDATION_SHA256    training.phase2.e4.contracts         OK
resolve_governed_split_by_sha256 training.phase2.e4.contracts         OK
EXPERT_LABORATORY_PARSER         lattice.models                       OK
EXPERT_MEDICATION_GRAMMAR        lattice.models                       OK
ExpertSpanProposal               lattice.models                       OK
GLiNERBackend                    zs0.backends                         OK
QwenBackend                      zs0.backends                         OK
build_mention_prompt             zs0.backends                         OK
parse_completion                 zs0.backends                         OK
ArmSources                       zs0.runner                           OK
Document                         zs0.runner                           OK
assert_stage3_complete           zs0.runner                           OK
run_all_arms                     zs0.runner                           OK
```

**BUG 1** — `load_l1_config` -> `load_config`, from the package.

**BUG 2** — `Phase1BConfig` is **not** in `mednorm_vi.phase1c_foundation`. It is
defined at `deterministic_baseline/models.py:18` and exported from
`mednorm_vi.deterministic_baseline.__all__`, alongside `run_phase1b`. Both are
now imported from there in one statement, which is also how
`phase1c_foundation/cli.py:161` does it.

**`SpanProposal` removed** — it was imported with `# noqa: F401` and never used;
only `ExpertSpanProposal` is. Three different classes in this repository are
named `SpanProposal`, so an unused import of it was a genuine liability.

The repaired block now executes clean: **PASS**.

## 5. All Stage-3 Call Signatures Audited

Checked with `inspect.signature` against the real implementations. Two more
defects, neither of which would have appeared until a document was processed on
a GPU.

```text
load_config(path)                -> tuple[L1Config, SectionLexicon]     OK
analyze_document(path, *, config, lexicon)   keyword-only               OK
Phase1BConfig.load(router, medication, laboratory)   3 positional       OK
run_phase1b(graph, config)                                             OK
GLiNERBackend(...).load() -> self                                      OK
QwenBackend(...).load() -> self                                        OK
run_all_arms(arms, documents, sources, *, active_parameters_by_arm)    OK
assert_stage3_complete(results, arms)                                  OK
```

**BUG 3 — wrong field names on the proposal.** Stage 3 read
`proposal.entity_type` and `proposal.score`. `SpanProposal` has neither:

```text
SpanProposal fields: proposal_id, document_id, start, end, text,
                     proposed_types, source_specialist, ..., local_score, ...
```

It carries **`proposed_types`** (a tuple) and **`local_score`**. Stage 3 would
have raised `AttributeError` on the first document. Fixed to read the real
fields, splitting the score across a multi-type tuple.

The expert is now chosen from **`source_specialist`** (`"medication"` /
`"laboratory"`), which the specialist itself sets, rather than inferred from the
entity type — inferring it was a guess that happened to work only for
unambiguous types.

**BUG 4 — organizer labels versus internal enums.** `proposed_types` carries the
organizer-facing Vietnamese labels; `ExpertSpanProposal` requires the internal
English enums, and refuses anything else:

```text
LatticeError: unsupported proposed type 'THUỐC'
```

Stage 3 now maps through `TYPE_BY_ORGANIZER_LABEL`, accepting either form and
dropping a proposal whose type maps to neither.

### 5.1 End-to-end proof, locally

The repaired deterministic path was run against a real document — no pretrained
model loaded:

```text
input     "Thuốc: Paracetamol 500mg uống 2 viên/ngày .
           Xét nghiệm: Glucose 6.2 mmol/L ."
E1/E2 raw proposals                 9
converted to ExpertSpanProposal     9
resolved and emitted                6
example    7:18  E1_medication_grammar  {'MEDICATION': 0.45}  'Paracetamol'
           7:29  E1_medication_grammar  {'MEDICATION': 1.0}   'Paracetamol 500mg uống'
```

Every converted proposal satisfies `text[start:end] == proposal.text`, and every
type is an internal enum.

## 6. Tests and Static Checks

`tests/unit/test_zs0_baseline.py` grew from 105 to 114 tests. The new ones
extract the Stage-3 import block from the notebook and **execute it in a clean
subprocess against the real installed package** — a clean interpreter on purpose,
since importing in-process would pass on names another test already imported.
They also parse the block and resolve every imported name against its stated
module; pin the two names that were wrong so they cannot regress; check each call
signature against the real implementation, including that `analyze_document`'s
keyword-only parameters really are passed as keywords; assert Stage 3 reads
`proposed_types`/`local_score`/`source_specialist` and never `entity_type`/
`score`; assert the organizer-label map is used, with a test that the lattice
genuinely refuses `"THUỐC"` so the map is shown to be necessary rather than
decorative; run the deterministic path end to end on a real document; and assert
no pretrained model is loaded locally.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q            1792 passed, 1 skipped
tests/unit/test_zs0_baseline.py                             114 passed
Stage-3 import block, clean subprocess                      PASS
ruff check .                                                All checks passed
ruff check notebooks                                        All checks passed
env PYTHONPATH=src .venv/bin/python -m mypy                 Success: no issues found
                                                            in 282 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src    clean
git diff --check                                            clean
```

## 7. Constraints Held

```text
E4 retired                      yes - untouched
E3                              absent
random E5 head                  absent
training / backward / optimizer NONE
internal_test                   never opened
external API                    none
organizer inference             did not run
output.zip                      not created
Stage-3 / Stage-4 guards        unchanged
architecture PDF                unchanged
```

## 8. Changed Files

```text
A  docs/audits/0050-zs0-stage3-import-and-public-api-repair.md
M  docs/audits/README.md
M  notebooks/MedNorm_ZS0_Baseline_Submission.ipynb
M  tests/unit/test_zs0_baseline.py
```

No source module changed. All four defects were in the notebook's use of the
packages, not in the packages themselves.

## 9. Limitations

* **ZS0 has still not completed a run.** No arm result, no selection, no
  `output.zip`.
* This audit proves the imports resolve, the signatures match and the
  deterministic path runs. It cannot prove the GLiNER and Qwen paths run,
  because neither model can be loaded here — their first real exercise is the
  next Colab run.
* The local end-to-end test covers a short synthetic medication line. It shows
  the conversion is correct in shape; it says nothing about quality.
* The score split across a multi-type `proposed_types` tuple is a convention
  chosen here, not a measured one. Where E1/E2 propose a single type — the
  common case — it is exactly the specialist's own score.

## 10. Exact Fresh-Colab Instructions

```text
0.  New Colab notebook -> Runtime -> Change runtime type -> T4 (or better) GPU.
    Open notebooks/MedNorm_ZS0_Baseline_Submission.ipynb from the updated repo.
    Leave the organizer flags at their committed values:
        RUN_ORGANIZER_INFERENCE = False
        CONFIRM_ORGANIZER_INFERENCE = ""

1.  Runtime -> Run all.

    Stage 0-2  install, verify, smoke-test (unchanged).
    Stage 3    now imports cleanly, loads GLiNER and Qwen, COUNTS both,
               enforces the 9B ledger, and runs all three arms. Expect:
                   "stage3_complete", arm_results: 3,
                   arms: ["ZS0-A", "ZS0-B", "ZS0-C"]
    Stage 4    prints the comparison table.
    Stage 5    selects an arm deterministically.
    Stage 6    STOPS: organizer inference is disabled.

2.  If Stage 3 raises now, it will be from model loading or inference, not
    imports — the import block and every call signature are verified locally.
    The message names the arm and the prerequisite; fix that rather than
    rerunning unchanged.

3.  Return with the Stage-4 table and the Stage-5 selection.
```

## 11. Safe-to-Commit Verdict

Safe to commit after review. The architecture PDF and Audits 0043-0049 are
byte-identical; no source module changed; no guard was weakened; E4 stays
retired; no training, `internal_test`, external API, organizer inference or
`output.zip`; no assistant control file exists; nothing artifact-shaped is
staged; no Git history was rewritten.

```bash
git add \
  docs/audits/0050-zs0-stage3-import-and-public-api-repair.md \
  docs/audits/README.md \
  notebooks/MedNorm_ZS0_Baseline_Submission.ipynb \
  tests/unit/test_zs0_baseline.py

git commit -m "fix: repair ZS0 Stage-3 imports, field names and label mapping"
git push origin main
```
