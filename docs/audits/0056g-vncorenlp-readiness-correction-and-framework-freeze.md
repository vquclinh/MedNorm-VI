# Audit 0056g - VnCoreNLP readiness correction and framework freeze

**This is the final framework correction audit.** It resolves the one framework-freeze
blocker identified by Audit 0056f: an explicitly configured VnCoreNLP path could
report specialist readiness READY even though runtime would fail before E3
construction because `jnius_config` was unavailable.

- **Scope:** VnCoreNLP readiness honesty, documentation alignment,
  `.dockerignore` build-context hygiene, one full validation pass, one final Docker
  build and one readiness-only offline container smoke.
- **Date:** 2026-07-30
- **Input state:** current uncommitted 2D-A/2D-B/2D-C tree plus Audits 0056a-0056f.
- **Verdict:** `FREEZE_FRAMEWORK_RUNTIME_WITH_NOTES`

No KB repair, annotation, model selection, model download, training, fine-tuning,
organizer inference, submission generation, notebook execution or E3 forward pass
was performed.

## 1. Initial Git and protected-artifact state

Read-only preconditions before edits:

```text
git branch --show-current                         main
git log -1 --oneline                              bbc3d02 feat: close the framework and verify the offline container
git diff --cached --name-status                   (empty)
git rev-list --left-right --count origin/main...HEAD   0    0
```

The working tree contained the expected uncommitted 2D-A/B/C source and audit
changes, including Audit 0056f. No unexplained file was present. Nothing was
staged, committed, pushed, reset, cleaned, checked out, restored or stashed.

Protected artifacts, verified by streaming `sha256sum` only:

```text
docs/MedNorm-VI_Architecture.pdf
  0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b

checkpoint/s1_mention_full_training_v1/best.pt
  a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
```

The PDF was read in full statically with `pdftotext`. The E3 checkpoint was not
deserialized on the host.

## 2. Owner VnCoreNLP policy

Owner decision implemented here:

- VnCoreNLP is optional for the shipped active E3 runtime profile.
- `configs/pipeline/full_v1.yaml` does not configure
  `expert_settings.e3_vncorenlp_dir`.
- When the field is absent or empty, E3 uses the verified default /
  pre-segmented-source path and no VnCoreNLP dependency is required.
- When `e3_vncorenlp_dir` is explicitly configured, VnCoreNLP becomes mandatory for
  that run. Readiness must validate the configured capability, and runtime must not
  silently fall back to `segmenter=None`.

This does not change E3 weights, tokenizer behavior, checkpoint policy, default
specialist behavior, or the Audit-0056e default E3 smoke contract.

## 3. Reproduced conditional readiness bug

Before this correction, source inspection showed:

- `E3VietHealthBertExpert.readiness()` checked only that `e3_vncorenlp_dir` was a
  directory and that some `*.jar` existed.
- `E3VietHealthBertExpert.prepare()` called `build_segmenter()` before
  `load_expert()`.
- `build_segmenter()` imported `py_vncorenlp`; Audit 0056e recorded that this path
  failed with `ModuleNotFoundError: No module named 'jnius_config'`.
- Therefore a configured fake/non-empty VnCoreNLP directory could be READY during
  readiness and fail before E3 construction at runtime.

That is a conditional readiness bug because readiness cannot report READY for a
configured component that cannot execute.

## 4. Canonical capability probe

Added to `src/mednorm_vi/mention_factory/neural/runtime.py`:

```text
VnCoreNLPCapability
probe_vncorenlp_capability()
```

The probe is typed, lightweight and model-free. For an explicitly configured path
it checks:

- path exists;
- path is a directory;
- at least one non-empty `*.jar` exists;
- `java` is available;
- `jnius_config` is importable;
- `py_vncorenlp` is importable.

Named blockers include:

```text
e3_vncorenlp_not_ready:missing_path
e3_vncorenlp_not_ready:not_directory
e3_vncorenlp_not_ready:missing_jar
e3_vncorenlp_not_ready:empty_jar
e3_vncorenlp_not_ready:missing_java_runtime
e3_vncorenlp_not_ready:missing_jnius_config
e3_vncorenlp_not_ready:missing_py_vncorenlp
e3_vncorenlp_not_ready:runtime_import_failed
e3_vncorenlp_not_ready:runtime_construction_failed:<ExceptionType>
```

The broad import/construction guards preserve the exact exception type in the
blocker detail; no exception is swallowed into READY.

## 5. Readiness/runtime shared behavior

`E3VietHealthBertExpert.readiness()` now calls
`probe_vncorenlp_capability()` only when `vncorenlp_dir` is non-empty.

`build_segmenter()` calls the same probe before constructing VnCoreNLP. If the
probe fails, it raises the same named blocker before `load_expert()` can start. If
construction itself fails after the lightweight probe passes, runtime raises
`e3_vncorenlp_not_ready:runtime_construction_failed:<ExceptionType>`.

Required behavior now holds:

```text
e3_vncorenlp_dir absent/empty          no probe, no dependency, default path unchanged
e3_vncorenlp_dir configured and valid  readiness may be READY, runtime constructs segmenter
e3_vncorenlp_dir configured invalid    NOT_READY; runtime fails closed; no fallback
```

## 6. Default path preservation

The default active profile still leaves `expert_settings.e3_vncorenlp_dir` unset.
`resolve_segmented_text(..., segmenter=None)` returns the original text with
`pre_segmented_source`, which is the verified active-path behavior from Audit
0056e.

Targeted tests and the readiness container both confirmed default specialist
readiness remains READY with the valid E3 checkpoint mounted.

## 7. Documentation alignment

Updated active policy text in:

```text
Dockerfile
configs/pipeline/full_v1.yaml
docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
requirements-image.lock
src/mednorm_vi/mention_factory/experts/e3_vihealthbert.py
src/mednorm_vi/mention_factory/neural/runtime.py
```

The active wording now states:

- VnCoreNLP is optional in the shipped active profile;
- absence means the verified default / pre-segmented-source path;
- explicit configuration makes VnCoreNLP mandatory for that run;
- an unusable configured path is NOT_READY and never silently falls back;
- historical training use of VnCoreNLP does not make it mandatory for current
  shipped runtime unless configured.

## 8. Dockerfile/RESOLVER/image-size wording fixes

Also corrected:

- Dockerfile example attribution from the stale Audit-0055 sentence to a generic
  Audit-0056e/0056g verified contract.
- `docs/resolution/RESOLVER.md` now names the active
  `configs/resolution/boundary_type_resolver_v1.yaml`, not deleted
  `configs/resolution/resolver_v1.yaml`.
- Docker image size wording now distinguishes Docker inspect/API size from
  `docker images` table size.
- `ACTIVE_RUNTIME_MANIFEST.md` corrected the stale L4 enabled line to the shipped
  profile state: deterministic route selected, learned route disabled.

## 9. `.dockerignore` and context hygiene

Created project-root `.dockerignore`. It excludes local/private/heavy material,
including:

```text
.git, .venv, caches, checkpoint, checkpoints, models, data, indices, notebooks,
internal_test, organizer_test, governed, private, output, output.zip,
*.pt, *.pth, *.ckpt, *.safetensors, *.bin, *.zip, temporary smoke paths,
editor/OS cache files
```

It does not exclude required build inputs:

```text
Dockerfile
requirements-image.lock
pyproject.toml
src/
configs/
schemas/
```

Build evidence:

```text
.dockerignore transfer size 594B
build-context transfer size 66.87kB
```

The final image history copied only:

```text
requirements-image.lock
pyproject.toml
src/
configs/
schemas/
```

No checkpoint, models, data, indices, notebooks, governed corpus, model weights or
submission artifact was copied into the image.

## 10. Targeted tests

Initial targeted pass exposed one stale assertion in
`tests/unit/test_e3_canonical_integration.py`: it expected the old human phrase
instead of the new structured blocker. The test was corrected to assert:

```text
e3_vncorenlp_not_ready:missing_jar
```

Final targeted command:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_vncorenlp_readiness_0056g.py tests/unit/test_e3_canonical_integration.py::test_a_vncorenlp_directory_without_a_jar_is_refused tests/unit/test_static_review_corrections_0056a.py::test_specialist_stays_ready_with_the_real_e3_asset tests/unit/test_static_review_corrections_0056a.py::test_no_executable_notebook_cell_references_active_e4 tests/unit/test_framework_closure.py::test_full_mode_still_fails_closed tests/unit/test_framework_closure.py::test_the_image_is_offline_by_construction tests/unit/test_framework_closure.py::test_the_image_copies_no_weights_or_restricted_data tests/unit/test_framework_closure.py::test_the_image_documents_read_only_asset_mounts -q
```

Result:

```text
18 passed in 0.57s
```

Coverage:

- default profile with no VnCoreNLP path remains specialist READY;
- configured directory without JAR is NOT_READY;
- fake/non-empty JAR with missing `jnius_config` is NOT_READY;
- missing `py_vncorenlp` is NOT_READY;
- import and construction failures are named;
- valid mocked capability may become READY;
- invalid configured VnCoreNLP cannot reach E3 construction or fallback;
- `.dockerignore` excludes protected/heavy paths and preserves build inputs;
- E4 retirement remains intact.

## 11. Full pytest result

One full suite pass was run exactly once after targeted tests and static/type gates:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Result:

```text
1818 passed, 1 skipped in 315.71s (0:05:15)
```

Skipped test:

```text
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

No deselected tests were reported.

## 12. Static/type results

Changed-Python list contained 23 files. The changed-file formatter gate was:

```text
ruff format --check "${CHANGED_PY[@]}"
```

Initial result:

```text
3 files would be reformatted, 20 files already formatted
```

Scoped formatting was run only on `CHANGED_PY`, then rechecked:

```text
23 files already formatted
```

Other gates:

```text
ruff check .                                      All checks passed!
env PYTHONPATH=src .venv/bin/python -m mypy src/mednorm_vi
                                                  Success: no issues found in 278 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src tests
                                                  PASSED
git diff --check                                  PASSED
```

Repository-wide `ruff format --check .` was not run, per owner decision. The known
repository-wide formatting debt remains separate technical debt.

## 13. Final Docker build result

Pre-build resource snapshot:

```text
Mem available 7.8 GiB
Swap used     2.7 GiB
docker ps     no running containers
```

Exactly one final build was run:

```text
docker build --pull --progress=plain -t mednorm-vi:framework-frozen .
```

Result:

```text
start time                2026-07-30T19:49:15+07:00
end time                  2026-07-30T19:54:02+07:00
exit code                 0
base                      python:3.14.5-slim-bookworm@sha256:ec58d916f9e24a6035cab2bdf07f6206c4cc092a16613c60597534711332d9d6
build-context transfer    66.87kB
```

The build downloaded locked runtime Python packages, including torch CPU wheels. It
did not download pretrained models or data.

Final image metadata:

```text
image id                  sha256:5cf3d2bf03220c3845fc564f0525f98fac30cd4dfeae070d90da67b2859a4e54
Docker inspect/API size   428,802,683 bytes
docker images table size  1.94GB
OS/Arch                   linux/amd64
created                   2026-07-30T19:52:53.655600086+07:00
user                      mednorm
workdir                   /app
entrypoint                ["python","-m","mednorm_vi.inference.cli"]
cmd                       ["--help"]
Python                    3.14.5
torch                     2.13.0+cpu
torch CUDA                None
```

## 14. Lightweight default/configured readiness smoke

Pre-container resource snapshot:

```text
Mem available 7.7 GiB
Swap used     2.7 GiB
docker ps     no running containers
```

One readiness-only container was run:

```text
docker run --rm --network=none --memory=2g --memory-swap=3g --cpus=2 --pids-limit=128 ...
```

Mounted read-only:

```text
/app/checkpoint
/app/indices
/vncore  (temporary fake VnCoreNLP directory with one non-empty JAR)
```

No validation corpus, internal_test, organizer input or output submission path was
mounted. `/app/indices` was mounted because the active `full_v1.yaml` readiness
contract checks `icd_index` and `rxnorm_index`; without the read-only KB indices,
default specialist readiness would correctly be NOT_READY for missing KB assets.

Container output:

```text
uid_gid 10001 10001
torch_loaded_before_readiness False
default_specialist READY []
configured_vncore_specialist NOT_READY ['missing_required_checkpoint:mention/vihealthbert:e3_vncorenlp_not_ready:missing_jnius_config (jnius_config)']
direct_invalid_e3 False e3_vncorenlp_not_ready:missing_jnius_config jnius_config loaded False
torch_loaded_after_negative_readiness False
full NOT_READY 11
network_blocked OSError 101
python 3.14.5
torch 2.13.0+cpu None
```

This proves:

- default specialist readiness is READY;
- explicitly configured unusable VnCoreNLP is NOT_READY;
- the blocker names the missing VnCoreNLP dependency;
- negative readiness did not import torch, deserialize the checkpoint, construct
  E3, or fall back to `segmenter=None`;
- full mode remains truthfully NOT_READY;
- runtime UID/GID is non-root;
- network is unavailable.

No E3 forward pass or document inference ran in this container.

## 15. Evidence no model was loaded

The negative readiness smoke printed:

```text
torch_loaded_before_readiness False
direct_invalid_e3 ... loaded False
torch_loaded_after_negative_readiness False
```

The configured invalid path failed inside E3 readiness before `prepare()` /
`load_expert()`. The container imported torch only after the assertions, solely to
print the runtime torch version. No `torch.load` call or checkpoint deserialization
occurred during the negative readiness probe.

## 16. Offline/non-root evidence

The container ran with `--network=none`. External socket creation failed:

```text
network_blocked OSError 101
```

Runtime identity:

```text
uid/gid 10001/10001
Docker Config.User mednorm
```

The image environment keeps:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/models/hf
```

## 17. Memory and swap observations

Observed resource checkpoints:

```text
before targeted tests          6.7 GiB available, 2.3 GiB swap used
before targeted rerun          6.8 GiB available, 2.1 GiB swap used
before mypy                    6.8 GiB available, 2.1 GiB swap used
before full pytest             6.8 GiB available, 2.1 GiB swap used
before Docker build            7.8 GiB available, 2.7 GiB swap used
before readiness container     7.7 GiB available, 2.7 GiB swap used
after readiness container      7.7 GiB available, 2.7 GiB swap used
lowest available RAM observed  6.7 GiB
largest observed swap delta    +0.6 GiB from pre-pytest to later Docker checks
```

No command started below 4 GiB available RAM. The 3 GiB stop threshold was never
approached, and no rapid swap growth was observed.

## 18. Cleanup evidence

Temporary fake VnCoreNLP resources were created only under:

```text
/tmp/mednorm-056g
```

They were removed after the readiness container exited. Final Docker status:

```text
docker ps   no running containers
```

The final verified image `mednorm-vi:framework-frozen` was intentionally kept for
owner review.

## 19. Lightweight static scans

Static scans after validation:

```text
e4_notebook_offenders=[]
old_l4_source_exists=False
old_l4_config_exists=False
adapter_file_exists=False
active mention_factory.adapters importers=[]
tracked assistant-specific files=[]
tracked model weights or submission ZIP=[]
git diff --cached --name-status  (empty)
```

The only `mention_factory.adapters` references outside audits are tests/comments
that assert the deleted module remains absent.

## 20. Exact changed-file inventory

Tracked modified/deleted files after this audit is created:

```text
M  Dockerfile
M  configs/pipeline/full_v1.yaml
M  docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
M  docs/resolution/RESOLVER.md
M  notebooks/MedNorm_Phase2_Validation_Ablation.ipynb
M  notebooks/MedNorm_S0_DomainAdaptation.ipynb
M  requirements-image.lock
M  src/mednorm_vi/inference/cli.py
M  src/mednorm_vi/inference/config.py
M  src/mednorm_vi/inference/packaging.py
M  src/mednorm_vi/inference/pipeline.py
D  src/mednorm_vi/mention_factory/adapters.py
M  src/mednorm_vi/mention_factory/experts/e3_vihealthbert.py
M  src/mednorm_vi/mention_factory/laboratory/synthetic.py
M  src/mednorm_vi/mention_factory/neural/runtime.py
M  src/mednorm_vi/resolution/canonical.py
M  src/mednorm_vi/validator/__init__.py
M  src/mednorm_vi/validator/kb_membership.py
M  src/mednorm_vi/validator/submission.py
M  tests/unit/test_e3_canonical_integration.py
M  tests/unit/test_framework_closure.py
M  tests/unit/test_full_pipeline_v1.py
M  tests/unit/test_l4_learned_v2.py
M  tests/unit/test_phase2_flags_registry_notebooks.py
M  tests/unit/test_route_gating_and_l8_l9.py
```

Untracked files after this audit is created:

```text
.dockerignore
docs/audits/0056a-static-independent-review-corrections.md
docs/audits/0056b-light-validation-pass.md
docs/audits/0056c-light-validation-rerun.md
docs/audits/0056d-type-correction-and-final-light-validation.md
docs/audits/0056e-isolated-container-and-runtime-verification.md
docs/audits/0056f-post-2dc-independent-framework-freeze-review.md
docs/audits/0056g-vncorenlp-readiness-correction-and-framework-freeze.md
src/mednorm_vi/mention_factory/expert_spec.py
src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
src/mednorm_vi/resolution/learned_dispatch.py
src/mednorm_vi/validator/final_gate.py
tests/unit/test_static_review_corrections_0056a.py
tests/unit/test_vncorenlp_readiness_0056g.py
```

Tracked diff stat before this audit:

```text
25 files changed, 2000 insertions(+), 1062 deletions(-)
```

Nothing is staged.

## 21. Remaining non-blocking notes

- `full` mode remains truthfully NOT_READY for E5/E6/E7 and future
  checkpoint/model roles.
- `full_requires_checkpoints` still names `assertion/assertion_hydra`; rename before
  model-phase artifact contracts harden.
- Much of the nested `profiles:` block in `full_v1.yaml` remains documentary rather
  than consumed.
- E3 checkpoint role resolution remains indirect through the E3 adapter.
- Training/artifact-validation unsafe-deserialization sites remain deferred to the
  pre-training security milestone.
- Repository-wide formatter debt remains separate from this milestone.
- Code-bearing ICD/RxNorm gold and assertion gold remain absent; candidate and
  assertion quality remain unmeasurable.
- No organizer inference or submission readiness is claimed.

None of these permits wrong organizer output, silent fallback, dishonest readiness,
an untrained/random model path, or a canonical-runner redesign.

## 22. Acceptance criteria

| Criterion | Result |
| --- | --- |
| Default E3 specialist readiness remains READY | PASS |
| Explicit unusable VnCoreNLP is NOT_READY | PASS |
| Configured VnCoreNLP never silently falls back | PASS |
| Negative readiness does not load E3/checkpoint | PASS |
| Source/config/docs aligned | PASS |
| `.dockerignore` protects local assets | PASS |
| Targeted tests pass | PASS |
| One full pytest pass green | PASS |
| Ruff, changed-file format, mypy, compileall, diff-check pass | PASS |
| One final Docker build succeeds | PASS |
| One lightweight offline readiness container succeeds | PASS |
| Runtime remains non-root | PASS |
| No notebook executed | PASS |
| No model/data downloaded | PASS |
| No training occurred | PASS |
| `internal_test` untouched | PASS |
| Nothing staged, committed or pushed | PASS |
| Audit 0056g contains actual evidence | PASS |

## 23. Verdict

```text
FREEZE_FRAMEWORK_RUNTIME_WITH_NOTES
```

The only Audit-0056f framework-freeze blocker is corrected. The remaining notes are
data/model/security-pretraining items or documented cleanup; they do not block
freezing the framework/runtime boundary.

## 24. Recommendation for owner commit

The owner can commit the uncommitted 2D-A/2D-B/2D-C/0056g working tree after their
review. The next phase after that owner commit should be KB repair/freeze and
creation of human-adjudicated code-bearing/assertion gold. Do not begin model
download, training, fine-tuning, organizer inference or submission generation until
the owner has frozen this framework/runtime state.

No Git staging command, commit command or push command is included here.
