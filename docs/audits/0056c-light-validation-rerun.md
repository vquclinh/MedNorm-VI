# Audit 0056c - light validation rerun (Milestone 2D-B.1)

**This is a 2D-B.1 audit, not a 2D-C/container audit.** It records the
final-gate test-contract correction, one targeted pass, one full pytest rerun, and
the requested follow-on checks. Audit 0056b remains unchanged as the historical
record of the failed first 2D-B attempt.

- **Scope:** Milestone 2D-B.1 - final-gate test-contract correction and one light
  validation rerun
- **Date:** 2026-07-30
- **Input state:** current uncommitted 2D-A tree plus Audits 0056a and 0056b
- **Verdict:** `NOT_SAFE_FOR_2D_C`

## 1. Initial Git and working-tree state

Read-only prevalidation:

```text
git branch --show-current                         main
git log -1 --oneline                              bbc3d02 feat: close the framework and verify the offline container
git diff --cached --name-status                   (empty)
git rev-list --left-right --count origin/main...HEAD   0    0
```

The starting working tree contained the reviewed 2D-A inventory plus
`docs/audits/0056b-light-validation-pass.md`. No file was staged, committed,
pushed, reset, cleaned, checked out or stashed.

## 2. Protected hashes

Verified with streaming `sha256sum` only:

```text
0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b  docs/MedNorm-VI_Architecture.pdf
a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c  checkpoint/s1_mention_full_training_v1/best.pt
```

Both matched the expected protected digests. The architecture PDF was read in
full statically with `pdftotext`. The E3 checkpoint was not deserialized.

## 3. Memory-safety observations

Thread limits were used for pytest and mypy:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
PYTORCH_NUM_THREADS=1
```

Observed memory:

```text
initial prevalidation              6.6 GiB available, 2.3 GiB swap used
before targeted tests              6.0 GiB available, 2.2 GiB swap used
before full pytest rerun           6.9 GiB available, 2.2 GiB swap used
after full pytest rerun            6.5 GiB available, 3.5 GiB swap used
second post-pytest stability check 5.9 GiB available, 3.5 GiB swap used
before mypy                        6.0 GiB available, 3.5 GiB swap used
final memory check                 6.2 GiB available, 3.4 GiB swap used
lowest available RAM observed      5.9 GiB
largest observed swap delta        +1.3 GiB during full pytest
```

The available-RAM floor stayed above the 4 GiB start threshold and the 3 GiB stop
threshold. Swap increased during the full pytest pass, then stayed stable on the
next check before follow-on static/type commands.

No notebook was executed, no notebook kernel was started, no E3/model/tokenizer was
loaded, `torch.load` was not called, CUDA was not initialized, no validation corpus
or model inference ran, `internal_test` contents were not accessed, no Docker build
or container run occurred, and no external download was attempted.

## 4. Cause of the first 2D-B failure

Audit 0056b failed because two older Audit-0053 L8/L9 tests encoded the pre-0056a
internal call graph:

```text
tests/unit/test_route_gating_and_l8_l9.py::test_packaging_stops_when_an_injected_code_is_outside_the_snapshot
tests/unit/test_route_gating_and_l8_l9.py::test_the_canonical_packaging_path_actually_invokes_the_membership_gate
```

The first expected `KbMembershipViolation` from the canonical packaging boundary.
The real 0056a boundary now aggregates final L9 checks and raises
`FinalValidationError`.

The second monkeypatched `mednorm_vi.inference.pipeline.validate_document_candidates`.
That symbol is no longer imported or exported by `inference.pipeline`; membership
validation is called inside `mednorm_vi.validator.final_gate`.

## 5. Owner final-gate contract decision

Owner decision for the canonical final packaging boundary:

```text
validator.final_gate -> FinalValidationError
```

No compatibility alias was added to `inference.pipeline`. The lower-level
`KbMembershipViolation` remains valid only for direct unit-level use of the KB
membership validator.

## 6. Exact test corrections

Changed `tests/unit/test_route_gating_and_l8_l9.py`:

- `test_packaging_stops_when_an_injected_code_is_outside_the_snapshot` now expects
  `FinalValidationError` and asserts that `kb.candidate_not_in_snapshot` is present
  in the structured violations.
- The test still proves that the forged candidate is rejected, the control run can
  package successfully before injection, and a failed run leaves no `output/` or
  `output.zip`.
- `test_the_canonical_packaging_path_actually_invokes_the_membership_gate` now
  patches `mednorm_vi.validator.final_gate.validate_document_candidates`, records
  document-order calls, injects a structured `kb.injected_membership_probe` issue
  for document 2, and asserts that it propagates through `FinalValidationError`.

No production compatibility code was added for either test.

## 7. Parent-symlink verification and correction

The 0056a claim that paths reached through symlinked parents were refused was false.
`assert_safe_output_directory()` called `Path.resolve()` but only rejected
`root.is_symlink()`, so `symlink_parent/output` could be created through the symlink.

Changed `src/mednorm_vi/inference/packaging.py`:

- added `_symlink_component(path)`, which checks each path component without
  resolving it;
- `assert_safe_output_directory()` now refuses any symlink component before deletion
  or writing.

Added to `tests/unit/test_static_review_corrections_0056a.py`:

```text
test_writing_refuses_output_reached_through_a_symlinked_parent
```

The test constructs:

```text
real_parent/
symlink_parent -> real_parent/
symlink_parent/output
```

and proves the run is refused before `real_parent/output` is created.

## 8. Stale KB docstring correction

Changed `src/mednorm_vi/validator/kb_membership.py`:

- removed the stale statement that L9 had always enforced
  `original_text[start:end] == text`;
- documented that serialized offset/text validation is owned by
  `mednorm_vi.validator.final_gate`;
- clarified that `KbMembershipViolation` is for direct membership-only callers,
  while canonical packaging raises `FinalValidationError`.

## 9. Remaining unsafe-deserialization inventory

Static text scan only:

```text
rg -n "torch\.load\(|weights_only\s*=\s*False" src tests notebooks configs docs
```

Classification:

```text
src/mednorm_vi/mention_factory/neural/runtime.py:330
  active inference, but canonical E3 path only; calls torch.load with
  weights_only=policy.weights_only after assert_load_is_permitted.
  Not a blocker under the trusted E3 policy.

src/mednorm_vi/mention_factory/neural/checkpoint_loading.py:120
  policy-only trusted legacy exception; weights_only=False is returned only for
  TRUSTED_LEGACY_E3_SHA256. Not a torch.load call.

src/mednorm_vi/training/cli.py:185
  artifact-validation-only / training CLI; --load-checkpoints path uses
  torch.load(..., weights_only=False). Deferred to a pre-training security milestone.

src/mednorm_vi/training/cli.py:233
  artifact-validation-only / training CLI; best-checkpoint validation loads payload
  unless --skip-payload is used. Deferred to a pre-training security milestone.

src/mednorm_vi/training/phase2/artifacts.py:225
  artifact-validation-only; _load_checkpoint_payload uses torch.load(path,
  map_location="cpu") before JSON fallback. Deferred to a pre-training security
  milestone.

notebooks/MedNorm_S1_Mention_InternalTest_Evaluation.ipynb:565
  historical/non-executed notebook source for internal-test evaluation; not active
  canonical inference.

notebooks/MedNorm_S1_Mention_Full_Training.ipynb:1146
  training-only notebook source; not active canonical inference.

notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb:1257
  historical/non-executed notebook smoke source; not active canonical inference.

tests/unit/test_static_review_corrections_0056a.py:731,761,769
  test-only strings/assertions.

docs/audits/* and docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
  historical/documentation references.
```

No active inference occurrence outside the trusted E3 policy was found.

## 10. Targeted test commands and results

Before targeted tests:

```text
Mem available: 6.0 GiB
Swap used:     2.2 GiB
```

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_route_gating_and_l8_l9.py::test_packaging_stops_when_an_injected_code_is_outside_the_snapshot tests/unit/test_route_gating_and_l8_l9.py::test_the_canonical_packaging_path_actually_invokes_the_membership_gate tests/unit/test_static_review_corrections_0056a.py::test_writing_refuses_output_reached_through_a_symlinked_parent -q
```

Result:

```text
3 passed in 22.27s
```

## 11. One-time full pytest rerun

Before full pytest:

```text
Mem available: 6.9 GiB
Swap used:     2.2 GiB
```

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Result:

```text
1806 passed, 1 skipped in 311.65s (0:05:11)
```

## 12. Skipped tests and reasons

Pytest reported one skip:

```text
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

No deselected tests were reported.

## 13. Ruff result

Command:

```text
ruff check .
```

Result:

```text
All checks passed!
```

## 14. Ruff-format result

Command:

```text
ruff format --check .
```

Result:

```text
FAILED - 261 files would be reformatted, 133 files already formatted
```

No formatting command was run. The reported files span notebooks, scripts, source,
integration tests and unit tests, including files changed by 2D-A/2D-B.1. This is a
validation failure.

## 15. Mypy result

Before mypy:

```text
Mem available: 6.0 GiB
Swap used:     3.5 GiB
```

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m mypy src/mednorm_vi
```

Result:

```text
FAILED - 8 errors in 2 files
```

Errors:

```text
src/mednorm_vi/resolution/learned_dispatch.py:338:
  Incompatible return value type (got "CheckpointLearnedL4Backend", expected
  "LearnedL4Backend"); protocol member contract_version expected settable variable,
  got read-only attribute.

src/mednorm_vi/inference/config.py:407:
  Incompatible types in assignment (expression has type "Path", variable has type
  "str").

src/mednorm_vi/inference/config.py:408:
  "str" has no attribute "exists".

src/mednorm_vi/inference/config.py:410:
  "str" has no attribute "is_dir".

src/mednorm_vi/inference/config.py:411:
  "str" has no attribute "iterdir".

src/mednorm_vi/inference/config.py:416:
  "str" has no attribute "is_file".

src/mednorm_vi/inference/config.py:416:
  "str" has no attribute "stat".

src/mednorm_vi/inference/config.py:496:
  Incompatible types in assignment (expression has type "ExpertSpec | None",
  variable has type "ExpertSpec").
```

This is a validation failure.

## 16. Compileall result

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m compileall -q src tests
```

Result:

```text
PASSED
```

## 17. Git diff check

Command:

```text
git diff --check
```

Result:

```text
PASSED
```

## 18. Lightweight static scans

No executable E4 notebook cells:

```text
e4_notebook_offenders=[]
```

Old L4 source/config resurrection:

```text
No active resurrection of mednorm_vi.resolution.resolver or
configs/resolution/resolver_v1.yaml was found.
```

The scan did find expected references to the current `resolution.resolver_v1`, plus
tests/comments documenting refusal of the retired `resolution/resolver.py` and
deleted `configs/resolution/resolver_v1.yaml`.

Deleted adapter importers:

```text
mention_factory_adapters_importers=[]
```

Assistant-specific files:

```text
git ls-files CLAUDE.md AGENTS.md .claude .claudeignore .codex .agents
  (empty)

find . -maxdepth 2 ...
  ./.agents
  ./.codex
```

The `.agents` and `.codex` directories are read-only environment-local directories;
they are not tracked and are not reported by `git status`.

Tracked model weights or submission ZIPs:

```text
git ls-files '*.pt' '*.pth' '*.ckpt' '*.safetensors' '*.bin' '*.zip'
  (empty)
```

Canonical runner:

```text
src/mednorm_vi/inference/pipeline.py:188:def run_document(
src/mednorm_vi/inference/pipeline.py:373:def run_input_dir(
```

Canonical final L9 gate:

```text
src/mednorm_vi/validator/final_gate.py:59:class FinalValidationError(RuntimeError):
src/mednorm_vi/validator/final_gate.py:309:def gate_final_documents(
src/mednorm_vi/inference/pipeline.py:333:def _gate_final_output(
```

No staged files:

```text
git diff --cached --name-status
  (empty)
```

Docker digest-pinning remains 2D-C:

```text
Dockerfile:17:# NOT YET DIGEST-PINNED...
Dockerfile:89:FROM python:3.14.5-slim-bookworm
```

## 19. Exact changed-file inventory

Correction files touched in 2D-B.1:

```text
src/mednorm_vi/inference/packaging.py
src/mednorm_vi/validator/kb_membership.py
tests/unit/test_route_gating_and_l8_l9.py
tests/unit/test_static_review_corrections_0056a.py
docs/audits/0056c-light-validation-rerun.md
```

Full working-tree inventory after this audit:

```text
 M Dockerfile
 M configs/pipeline/full_v1.yaml
 M docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
 M notebooks/MedNorm_Phase2_Validation_Ablation.ipynb
 M notebooks/MedNorm_S0_DomainAdaptation.ipynb
 M src/mednorm_vi/inference/cli.py
 M src/mednorm_vi/inference/config.py
 M src/mednorm_vi/inference/packaging.py
 M src/mednorm_vi/inference/pipeline.py
 D src/mednorm_vi/mention_factory/adapters.py
 M src/mednorm_vi/mention_factory/laboratory/synthetic.py
 M src/mednorm_vi/mention_factory/neural/runtime.py
 M src/mednorm_vi/resolution/canonical.py
 M src/mednorm_vi/validator/__init__.py
 M src/mednorm_vi/validator/kb_membership.py
 M src/mednorm_vi/validator/submission.py
 M tests/unit/test_framework_closure.py
 M tests/unit/test_full_pipeline_v1.py
 M tests/unit/test_phase2_flags_registry_notebooks.py
 M tests/unit/test_route_gating_and_l8_l9.py
?? docs/audits/0056a-static-independent-review-corrections.md
?? docs/audits/0056b-light-validation-pass.md
?? docs/audits/0056c-light-validation-rerun.md
?? src/mednorm_vi/mention_factory/expert_spec.py
?? src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
?? src/mednorm_vi/resolution/learned_dispatch.py
?? src/mednorm_vi/validator/final_gate.py
?? tests/unit/test_static_review_corrections_0056a.py
```

Tracked diff stat before this audit:

```text
20 files changed, 920 insertions(+), 541 deletions(-)
```

Nothing was staged.

## 20. Unresolved items for 2D-C or later

Blocks 2D-C readiness:

- `ruff format --check .` fails globally: 261 files would be reformatted.
- `mypy src/mednorm_vi` fails with 8 errors in 2 files.

Still 2D-C scope after the above validation failures are corrected:

- Docker base image must be digest-pinned.
- One Docker build and offline smoke must be run under the authorized 2D-C scope.
- Real E3 checkpoint runtime compatibility under the hardened load policy remains
  unverified.
- Container-level final L9 gate behavior still needs smoke verification.

Later/pre-training security scope:

- training/artifact-validation `torch.load` sites outside canonical inference need a
  hardened policy before they become part of trusted training or artifact intake.

Still data/architecture-open:

- `full_requires_checkpoints` still names `assertion/assertion_hydra`, while the
  future learned artifact is S2.
- Much of the nested `profiles:` block in `configs/pipeline/full_v1.yaml` remains
  unread by runtime configuration.
- E3 checkpoint role resolution remains indirect through the E3 adapter.
- Deterministic output ordering is enforced although organizer relevance remains
  unconfirmed.
- No packaging-stress suite exists.
- Code-bearing and assertion gold labels remain absent.

## 21. Verdict

```text
NOT_SAFE_FOR_2D_C
```

Rationale: the targeted fixes pass and the full pytest rerun is green, but the
required follow-on validation did not pass. `ruff format --check .` and
`mypy src/mednorm_vi` are red, so 2D-B.1 is not a complete light-validation pass.

No staging, commit, push, reset, clean, checkout or stash was performed.
