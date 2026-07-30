# Audit 0056d - type correction and final light validation (Milestone 2D-B.2)

**This is a 2D-B.2 audit, not a 2D-C/container audit.** It records the type
correction, changed-file-only format gate, one final full pytest pass, and final
static/type validation over the current uncommitted 2D-A/2D-B working tree.

- **Scope:** Milestone 2D-B.2 - type correction, changed-file format gate, final
  light validation
- **Date:** 2026-07-30
- **Input state:** current uncommitted tree containing 2D-A, 2D-B.1 and Audits
  0056a-0056c
- **Verdict:** `SAFE_FOR_2D_C_WITH_NOTES`

## 1. Initial Git state

Read-only prevalidation:

```text
git branch --show-current                         main
git log -1 --oneline                              bbc3d02 feat: close the framework and verify the offline container
git diff --cached --name-status                   (empty)
git rev-list --left-right --count origin/main...HEAD   0    0
```

The working tree contained the reviewed 2D-A and 2D-B.1 changes plus Audits
0056a-0056c. No unexplained file was present. No file was staged, committed,
pushed, reset, cleaned, checked out, restored or stashed.

## 2. Protected hashes

Verified by streaming `sha256sum` only:

```text
0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b  docs/MedNorm-VI_Architecture.pdf
a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c  checkpoint/s1_mention_full_training_v1/best.pt
```

Both matched the expected protected digests. The architecture PDF was read in full
statically with `pdftotext`. The E3 checkpoint was not deserialized.

## 3. Memory and swap observations

Thread limits were used for mypy and pytest:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
PYTORCH_NUM_THREADS=1
```

Observed memory:

```text
before targeted mypy        6.0 GiB available, 3.0 GiB swap used
before final full pytest    5.9 GiB available, 2.7 GiB swap used
before package-wide mypy    6.6 GiB available, 3.7 GiB swap used
final memory check          6.6 GiB available, 3.7 GiB swap used
lowest available RAM        5.9 GiB
largest observed swap delta +1.0 GiB from the pre-pytest reading
```

Available RAM stayed above the 4 GiB start threshold and 3 GiB stop threshold.
No notebook was executed, no kernel was started, no E3/model/tokenizer was loaded,
`torch.load` was not called, CUDA was not initialized, no corpus/model inference
ran, `internal_test` contents were not accessed, Docker was not built, and no
container was started.

## 4. Why repository-wide format is not the incremental gate

Audit 0056c recorded:

```text
ruff format --check .
FAILED - 261 files would be reformatted, 133 files already formatted
```

Owner decision: this is repository-wide formatting debt, not valid evidence that
the 2D-A/2D-B correction itself is incorrectly formatted. Formatting hundreds of
unrelated files would create noisy metadata churn outside the milestone scope.

The accepted 2D-B.2 gate is changed-Python only. The repository-wide formatter debt
is recorded as separate non-blocking technical debt for a dedicated formatting
commit.

## 5. Changed-Python formatting policy

Changed Python files were constructed from tracked modified/added/copied/renamed
Python files plus untracked Python files:

```bash
mapfile -t CHANGED_PY < <(
  {
    git diff --name-only --diff-filter=ACMR -- '*.py'
    git ls-files --others --exclude-standard -- '*.py'
  } | sort -u
)
```

List printed:

```text
src/mednorm_vi/inference/cli.py
src/mednorm_vi/inference/config.py
src/mednorm_vi/inference/packaging.py
src/mednorm_vi/inference/pipeline.py
src/mednorm_vi/mention_factory/expert_spec.py
src/mednorm_vi/mention_factory/laboratory/synthetic.py
src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
src/mednorm_vi/mention_factory/neural/runtime.py
src/mednorm_vi/resolution/canonical.py
src/mednorm_vi/resolution/learned_dispatch.py
src/mednorm_vi/validator/__init__.py
src/mednorm_vi/validator/final_gate.py
src/mednorm_vi/validator/kb_membership.py
src/mednorm_vi/validator/submission.py
tests/unit/test_framework_closure.py
tests/unit/test_full_pipeline_v1.py
tests/unit/test_l4_learned_v2.py
tests/unit/test_phase2_flags_registry_notebooks.py
tests/unit/test_route_gating_and_l8_l9.py
tests/unit/test_static_review_corrections_0056a.py
```

Only these 20 files were passed to `ruff format`.

## 6. Original mypy errors

Audit 0056c recorded eight errors:

```text
src/mednorm_vi/resolution/learned_dispatch.py:338:
  CheckpointLearnedL4Backend incompatible with LearnedL4Backend because
  contract_version was a settable Protocol variable but implemented by an immutable
  backend.

src/mednorm_vi/inference/config.py:407:
  Incompatible types in assignment: Path assigned to a local inferred as str.

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
  ExpertSpec | None assigned to a local previously inferred as ExpertSpec.
```

## 7. Exact type corrections

Changed `src/mednorm_vi/resolution/learned_dispatch.py`:

- `LearnedL4Backend.contract_version` is now a read-only `@property`.
- This matches the intended immutable backend contract and preserves structural
  typing and fail-closed dispatch.

Changed `src/mednorm_vi/inference/config.py`:

- renamed the expert readiness path detail to `detail_path`;
- introduced `artifact_path: Path` for role artifact checks;
- renamed the full-mode artifact spec local to `artifact_spec`;
- introduced `candidate_spec = EXPERT_SPEC_BY_FLAG.get(flag)` for optional lookup
  and narrowed it before reading `implementation`.

Changed `tests/unit/test_l4_learned_v2.py`:

- added `test_checkpoint_backend_satisfies_the_learned_l4_protocol`, using the
  existing tiny JSON checkpoint helper;
- the test proves the concrete checkpoint backend satisfies `LearnedL4Backend` and
  advertises the expected immutable contract version without model loading.

No broad `type: ignore`, `cast(Any, ...)`, `Any` weakening, `noqa`, or compatibility
wrapper was added.

## 8. Targeted mypy result

Before command:

```text
Mem available: 6.0 GiB
Swap used:     3.0 GiB
```

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m mypy src/mednorm_vi/resolution/learned_dispatch.py src/mednorm_vi/inference/config.py src/mednorm_vi/mention_factory/expert_spec.py
```

Result:

```text
Success: no issues found in 3 source files
```

## 9. Targeted test result

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_l4_learned_v2.py tests/unit/test_static_review_corrections_0056a.py tests/unit/test_route_gating_and_l8_l9.py::test_packaging_stops_when_an_injected_code_is_outside_the_snapshot tests/unit/test_route_gating_and_l8_l9.py::test_the_canonical_packaging_path_actually_invokes_the_membership_gate -q
```

Result:

```text
64 passed in 20.87s
```

Coverage: learned-L4 dispatch and backend Protocol, expert readiness, fake/empty
artifact directories, specialist readiness, output-directory symlink safety, and
final L9 behavior.

## 10. Changed-file formatting result

Initial changed-file check:

```text
ruff format --check "${CHANGED_PY[@]}"
20 files would be reformatted
```

Scoped formatting:

```text
ruff format "${CHANGED_PY[@]}"
20 files reformatted
```

Recheck:

```text
ruff format --check "${CHANGED_PY[@]}"
20 files already formatted
```

No Python file outside `CHANGED_PY` was formatted. The post-format diff was
inspected around the type-correction hunks; semantics remained the intended
Protocol/property, path-variable, optional-narrowing and targeted-test changes.

## 11. One-time full pytest result

Before command:

```text
Mem available: 5.9 GiB
Swap used:     2.7 GiB
```

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Result:

```text
1807 passed, 1 skipped in 315.75s (0:05:15)
```

## 12. Skipped tests and reason

```text
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

No tests were deselected.

## 13. Repository-wide ruff result

Command:

```text
ruff check .
```

Result:

```text
All checks passed!
```

## 14. Package-wide mypy result

Before command:

```text
Mem available: 6.6 GiB
Swap used:     3.7 GiB
```

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m mypy src/mednorm_vi
```

Result:

```text
Success: no issues found in 278 source files
```

## 15. Compileall result

Command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m compileall -q src tests
```

Result:

```text
PASSED
```

## 16. Git diff check

Command:

```text
git diff --check
```

Result:

```text
PASSED
```

## 17. Lightweight static scans

No executable E4 notebook cells:

```text
e4_notebook_offenders=[]
```

Old L4 resurrection:

```text
test ! -e src/mednorm_vi/resolution/resolver.py    PASSED
test ! -e configs/resolution/resolver_v1.yaml      PASSED
```

The text scan finds expected references to current `resolution.resolver_v1` and
tests/comments documenting the retired `resolution/resolver.py` /
`configs/resolution/resolver_v1.yaml`; no deleted module/config resurrection was
found.

Deleted adapter importers:

```text
mention_factory_adapters_importers=[]
```

No staged files:

```text
git diff --cached --name-status
  (empty)
```

Assistant-specific tracked files:

```text
git ls-files CLAUDE.md AGENTS.md .claude .claudeignore .codex .agents
  (empty)
```

Tracked model weights or submission ZIP:

```text
git ls-files '*.pt' '*.pth' '*.ckpt' '*.safetensors' '*.bin' '*.zip'
  (empty)
```

Docker digest pinning remains explicitly deferred:

```text
Dockerfile:17:# NOT YET DIGEST-PINNED...
Dockerfile:89:FROM python:3.14.5-slim-bookworm
ACTIVE_RUNTIME_MANIFEST.md records that Milestone 2D-C will pin the base image.
```

## 18. Complete changed-file inventory

Tracked modified/deleted files before this audit:

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
 M tests/unit/test_l4_learned_v2.py
 M tests/unit/test_phase2_flags_registry_notebooks.py
 M tests/unit/test_route_gating_and_l8_l9.py
```

Untracked files before this audit:

```text
docs/audits/0056a-static-independent-review-corrections.md
docs/audits/0056b-light-validation-pass.md
docs/audits/0056c-light-validation-rerun.md
src/mednorm_vi/mention_factory/expert_spec.py
src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
src/mednorm_vi/resolution/learned_dispatch.py
src/mednorm_vi/validator/final_gate.py
tests/unit/test_static_review_corrections_0056a.py
```

After this audit is created, add:

```text
docs/audits/0056d-type-correction-and-final-light-validation.md
```

Tracked diff stat before this audit:

```text
21 files changed, 1740 insertions(+), 990 deletions(-)
```

## 19. Remaining 2D-C work

- Pin the Docker base image by digest.
- Perform the single authorized Docker build.
- Run offline container smoke under `--network=none`.
- Verify real E3 checkpoint runtime compatibility under the hardened loading
  policy.
- Verify final L9 offset/text and offered-set failures inside the container.

These are 2D-C tasks and were not started here.

## 20. Repository-wide formatting debt

Repository-wide `ruff format --check .` remains known technical debt:

```text
261 files would be reformatted, 133 files already formatted
```

This is non-blocking for 2D-C under the owner decision because the 2D-B.2
changed-file format gate is green. It should be handled later as a dedicated,
repository-wide formatting commit.

## 21. Verdict

```text
SAFE_FOR_2D_C_WITH_NOTES
```

Rationale: full pytest, repository-wide lint, changed-file formatting, package-wide
mypy, compileall and `git diff --check` all pass. Remaining notes are exactly the
intended 2D-C container/runtime work plus separate repository-wide formatting debt.

No staging, commit, push, reset, clean, checkout, restore or stash was performed.
