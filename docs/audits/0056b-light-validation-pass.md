# Audit 0056b - light validation pass (Milestone 2D-B)

**This is a 2D-B audit, not the final 0056 closure audit.** It records one
light validation attempt over the uncommitted Milestone 2D-A working tree. No
source correction was made in this milestone.

- **Scope:** Milestone 2D-B - one full pytest pass plus follow-on checks only if
  pytest is clean
- **Date:** 2026-07-30
- **Input state:** uncommitted Milestone 2D-A working tree from Audit 0056a
- **Verdict:** `NOT_SAFE_FOR_2D_C`

## 1. Initial Git and working-tree state

Prevalidation commands were run read-only:

```text
git branch --show-current                         main
git log -1 --oneline                              bbc3d02 feat: close the framework and verify the offline container
git rev-list --left-right --count origin/main...HEAD   0    0
git diff --cached --name-status                   (empty)
```

No file was staged. The working tree contained the reviewed 2D-A inventory:

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
 M src/mednorm_vi/validator/submission.py
 M tests/unit/test_framework_closure.py
 M tests/unit/test_full_pipeline_v1.py
 M tests/unit/test_phase2_flags_registry_notebooks.py
?? docs/audits/0056a-static-independent-review-corrections.md
?? src/mednorm_vi/mention_factory/expert_spec.py
?? src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
?? src/mednorm_vi/resolution/learned_dispatch.py
?? src/mednorm_vi/validator/final_gate.py
?? tests/unit/test_static_review_corrections_0056a.py
```

Tracked diff stat at prevalidation:

```text
18 files changed, 865 insertions(+), 522 deletions(-)
```

## 2. Protected artifact hashes

Verified with streaming `sha256sum` only:

```text
0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b  docs/MedNorm-VI_Architecture.pdf
a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c  checkpoint/s1_mention_full_training_v1/best.pt
```

Both matched the expected protected digests. The PDF was read statically in full
with `pdftotext`; the checkpoint was not opened with `torch.load` and was not
deserialized.

## 3. Memory-safety compliance

Thread limits were set on the pytest command:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
PYTORCH_NUM_THREADS=1
```

Observed memory:

```text
before prevalidation/context reads   7.1 GiB available, 1.4 GiB swap used
before pytest                        7.1 GiB available, 1.4 GiB swap used
after pytest                         7.7 GiB available, 2.6 GiB swap used
after static diagnosis status check  7.5 GiB available, 2.6 GiB swap used
lowest available RAM observed        7.1 GiB
```

The available-RAM floor stayed above the 4 GiB start threshold and the 3 GiB stop
threshold. Swap usage increased from 1.4 GiB to 2.6 GiB by the post-pytest check,
so no further heavy validation commands were started after the pytest failure.

No notebook was executed, no Jupyter/kernel process was started, no model was
loaded, `torch.load` was not called, no tokenizer/Transformer/CUDA path was
initialized, no validation corpus was run, no Docker build ran, and no container
was started.

## 4. Pytest command and result

Exactly one full-suite pytest command was run, sequentially:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Result:

```text
2 failed, 1803 passed, 1 skipped in 309.23s (0:05:09)
```

Failing tests:

```text
tests/unit/test_route_gating_and_l8_l9.py::test_packaging_stops_when_an_injected_code_is_outside_the_snapshot
tests/unit/test_route_gating_and_l8_l9.py::test_the_canonical_packaging_path_actually_invokes_the_membership_gate
```

## 5. Skipped tests and reasons

Pytest reported one skip:

```text
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

No tests were reported as deselected by the full-suite command.

## 6. Ruff result

Not run.

Reason: the one full pytest pass failed. Per the 2D-B instructions, the validation
sequence stopped after pytest and no follow-on static/type command was started.

## 7. Ruff format check result

Not run for the same reason.

## 8. Mypy result

Not run for the same reason.

## 9. Compileall result

Not run for the same reason.

## 10. Git diff check

Not run for the same reason.

## 11. Lightweight static scans

The planned lightweight static-scan section was not run as a validation phase
because pytest failed and the sequence stopped.

Static diagnosis of the two failures was performed by reading source/tests only:

- `tests/unit/test_route_gating_and_l8_l9.py` still expects
  `KbMembershipViolation` from the canonical packaging path.
- `src/mednorm_vi/inference/pipeline.py` now calls `_gate_final_output`, which
  constructs `FinalDocument` values and delegates to
  `validator.final_gate.gate_final_documents`.
- `src/mednorm_vi/validator/final_gate.py` aggregates organizer, offset,
  duplicate-candidate, order, snapshot, and offered-set issues, then raises
  `FinalValidationError` on any error.
- The injected forged RxCUI test did prove the new final gate stops the run before
  writing; the failure is that the older test catches the pre-0056a exception type.
- `tests/unit/test_route_gating_and_l8_l9.py` also monkeypatches
  `mednorm_vi.inference.pipeline.validate_document_candidates`, but that symbol is
  no longer exported or imported by `pipeline.py`; the membership validator is now
  called inside `validator.final_gate`.

This is a full-suite compatibility regression in the tests around the 0056a final
gate consolidation. It is not evidence of a notebook execution, model load, or
Docker/container path.

## 12. Review of test quality

The failing tests are useful: they caught that the older 0053 route-gating/L9 tests
were not migrated to the new 0056a final-gate API.

The first failure is primarily an expected-exception mismatch:

```text
expected: KbMembershipViolation
actual:   FinalValidationError containing repeated kb.candidate_not_in_snapshot entries
```

The second failure is a stale monkeypatch target:

```text
expected symbol: mednorm_vi.inference.pipeline.validate_document_candidates
actual path:     mednorm_vi.validator.final_gate.validate_document_candidates
```

Because pytest failed, the run cannot be used as evidence that all 2D-A changes are
fully regression-clean. However, the failure report also confirms that the final L9
gate is reachable in the canonical path and prevents apparently successful
packaging on a forged candidate.

The skip reason is explicit and environment-related (`pyarrow` absent). No
model-dependent skip was newly observed in the pytest summary.

## 13. Exact changed-file inventory

After this audit is created, the 2D-B inventory is the 2D-A inventory plus this new
untracked audit file.

Tracked modified/deleted files:

```text
M  Dockerfile
M  configs/pipeline/full_v1.yaml
M  docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
M  notebooks/MedNorm_Phase2_Validation_Ablation.ipynb
M  notebooks/MedNorm_S0_DomainAdaptation.ipynb
M  src/mednorm_vi/inference/cli.py
M  src/mednorm_vi/inference/config.py
M  src/mednorm_vi/inference/packaging.py
M  src/mednorm_vi/inference/pipeline.py
D  src/mednorm_vi/mention_factory/adapters.py
M  src/mednorm_vi/mention_factory/laboratory/synthetic.py
M  src/mednorm_vi/mention_factory/neural/runtime.py
M  src/mednorm_vi/resolution/canonical.py
M  src/mednorm_vi/validator/__init__.py
M  src/mednorm_vi/validator/submission.py
M  tests/unit/test_framework_closure.py
M  tests/unit/test_full_pipeline_v1.py
M  tests/unit/test_phase2_flags_registry_notebooks.py
```

Untracked files:

```text
docs/audits/0056a-static-independent-review-corrections.md
docs/audits/0056b-light-validation-pass.md
src/mednorm_vi/mention_factory/expert_spec.py
src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
src/mednorm_vi/resolution/learned_dispatch.py
src/mednorm_vi/validator/final_gate.py
tests/unit/test_static_review_corrections_0056a.py
```

Nothing was staged.

## 14. Unresolved items deferred to 2D-C

These items remain 2D-C scope, but 2D-C should not begin until the 2D-B pytest
regression is corrected and the light validation pass is rerun successfully.

- Docker base image remains a mutable tag and must be digest-pinned in 2D-C.
- Real E3 checkpoint runtime compatibility under the new checkpoint-loading policy
  remains unverified; that requires an authorized model-load/container smoke.
- `full_requires_checkpoints` still names `assertion/assertion_hydra`, while the
  future learned artifact is S2.
- Much of the nested `profiles:` block in `configs/pipeline/full_v1.yaml` remains
  unread except for retired-E4 scanning and documentation alignment.
- E3 checkpoint role resolution remains indirect through the E3 adapter.
- Deterministic output ordering is enforced although organizer relevance remains
  unconfirmed.
- No packaging-stress suite exists.
- Code-bearing and assertion gold labels remain absent.

Additional notes from the preceding independent static review:

- `assert_safe_output_directory()` does not appear to reject parent-directory symlink
  components even though the audit text claims "a path reached through one" is
  refused.
- Unsafe `torch.load`/`weights_only=False` occurrences remain in active training and
  artifact-validation source outside the canonical E3 inference path.
- `validator/kb_membership.py` still has a docstring sentence saying L9 has always
  enforced the offset/text half, which contradicts the 0056a final-gate rationale.

## 15. Verdict

```text
NOT_SAFE_FOR_2D_C
```

Rationale: the required single full pytest pass failed. The failure is localized to
older 0053 route-gating/L9 tests that were not migrated to the new 0056a final-gate
exception/API shape, but a red full suite means Milestone 2D-B did not pass.

## 16. Recommended 2D-C scope

Do not begin 2D-C yet.

Recommended next scope before 2D-C:

1. Authorize a correction turn for the two failing
   `tests/unit/test_route_gating_and_l8_l9.py` tests.
2. Decide whether the correct public contract is:
   `FinalValidationError` from the canonical packaging path, plus direct
   membership-gate tests through `validator.final_gate`; or a compatibility alias
   in `inference.pipeline`.
3. Rerun exactly one light 2D-B validation pass after the correction.

Only after a green 2D-B pass should 2D-C proceed with:

1. Docker base digest pinning.
2. One authorized Docker build.
3. Offline container smoke under `--network=none`.
4. Real E3 checkpoint compatibility verification under the hardened load policy.
5. Container-level verification that final L9 offset/text and offered-set failures
   stop packaging.
