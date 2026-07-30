# Audit 0056f - post-2D-C independent framework/runtime freeze review

**This is an independent post-2D-C review.** It verifies the current uncommitted
2D-A/2D-B/2D-C working tree against the architecture PDF, the approved E4 retirement
decision, the active runtime manifest, Audits 0056a-0056e, Dockerfile,
`requirements-image.lock`, `configs/pipeline/full_v1.yaml` and the current source.

- **Scope:** final independent framework/runtime freeze review after Milestone 2D-C
- **Date:** 2026-07-30
- **Verdict:** `CORRECTIONS_REQUIRED_BEFORE_FREEZE`

The verdict is not a rejection of the 2D-C default smoke evidence. The default
deterministic and default specialist/E3 container paths are supported by the audit
record and source wiring. The blocker is narrower: when `e3_vncorenlp_dir` is
explicitly configured, specialist readiness can report READY for a runtime path that
then fails before E3 construction because `py_vncorenlp` imports missing
`jnius_config`. That is a conditional readiness bug, and the project rule says
readiness must not be dishonest.

## 1. Git and protected-artifact state

Read-only state at review start:

```text
git branch --show-current                         main
git log -1 --oneline                              bbc3d02 feat: close the framework and verify the offline container
git diff --cached --name-status                   (empty)
git rev-list --left-right --count origin/main...HEAD   0    0
```

Working tree before this audit contained the expected uncommitted 2D-A/2D-B/2D-C
source and audit changes. No unexplained file was found.

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
M  src/mednorm_vi/validator/kb_membership.py
M  src/mednorm_vi/validator/submission.py
M  tests/unit/test_framework_closure.py
M  tests/unit/test_full_pipeline_v1.py
M  tests/unit/test_l4_learned_v2.py
M  tests/unit/test_phase2_flags_registry_notebooks.py
M  tests/unit/test_route_gating_and_l8_l9.py
```

Untracked files before this audit:

```text
docs/audits/0056a-static-independent-review-corrections.md
docs/audits/0056b-light-validation-pass.md
docs/audits/0056c-light-validation-rerun.md
docs/audits/0056d-type-correction-and-final-light-validation.md
docs/audits/0056e-isolated-container-and-runtime-verification.md
src/mednorm_vi/mention_factory/expert_spec.py
src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
src/mednorm_vi/resolution/learned_dispatch.py
src/mednorm_vi/validator/final_gate.py
tests/unit/test_static_review_corrections_0056a.py
```

Tracked diff stat before this audit:

```text
21 files changed, 1797 insertions(+), 999 deletions(-)
```

Protected hashes, verified by streaming `sha256sum` only:

```text
docs/MedNorm-VI_Architecture.pdf
  0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b

checkpoint/s1_mention_full_training_v1/best.pt
  a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
```

The architecture PDF was read in full statically with `pdftotext`. The E3 checkpoint
was not deserialized.

## 2. Architecture and E4 interpretation

The PDF remains the primary source of truth: L1-L9 preserve original offsets, no
expert emits a final entity, LLMs may only choose from offered options, L8/L9 must be
deterministic and fail-fast, and final output must satisfy:

```text
original_text[start:end] == text
```

The owner-approved architecture deviation is also applied: E4 PhoBERT/W2NER is
permanently retired. Its absence is not a defect. Legitimate E3
`training/phobert_alignment.py` remains because ViHealthBERT-Word uses the slow
PhoBERT tokenizer family.

The intended active/future expert set is:

```text
E1 medication grammar
E2 laboratory parser
E3 ViHealthBERT
E5 XLM-R MRC
E6 GLiNER
E7 constrained Qwen proposer
```

## 3. Audit-0056e claim verification

| 0056e claim | Independent disposition |
| --- | --- |
| Base digest is official Docker Hub linux/amd64 | Supported by the recorded `docker buildx imagetools inspect` evidence and by the pinned Dockerfile reference. This review did not re-run Docker/network resolution. |
| Dockerfile and manifest agree on tag/index/platform digest/image/version | Supported. Dockerfile, ACTIVE_RUNTIME_MANIFEST and Audit 0056e all record tag `python:3.14.5-slim-bookworm`, index `sha256:a9bee155...9ccb`, linux/amd64 manifest `sha256:ec58d916...d9d6`, image `sha256:47d630a...9ba6`, Python 3.14.5 and torch `2.13.0+cpu`. |
| Exactly one Docker build recorded | Supported by Audit 0056e. Static review cannot prove shell history, but no contrary source or audit evidence was found. |
| Image excludes checkpoints/models/data/indices/notebooks/submission artifacts | Supported for the image by Dockerfile COPY instructions: only `requirements-image.lock`, `pyproject.toml`, `src/`, `configs/` and `schemas/` are copied. No `.dockerignore` exists, so the build context itself may still include ignored local assets; that is not the same as image content but should be cleaned up later. |
| Runtime user non-root | Supported by Dockerfile `USER mednorm` and 0056e UID/GID evidence. |
| Canonical entrypoint | Supported: `ENTRYPOINT ["python", "-m", "mednorm_vi.inference.cli"]`; CLI calls `run_input_dir`. |
| Runtime download disabled | Supported: Dockerfile sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME=/models/hf`; E3 and LLM loaders use local-only paths. |
| Deterministic mode completed offline | Supported by 0056e smoke evidence and canonical CLI wiring. |
| Final serialized L9 executed | Supported by `inference.pipeline._gate_final_output` calling `validator.final_gate.gate_final_documents` before writing. |
| Offset/text mismatch failed closed | Supported by 0056e container evidence and `final_gate` source. |
| Candidate-not-offered failed closed | Supported by 0056e container evidence and `kb_membership` offered-set logic. |
| E3 completed one successful specialist smoke | Supported by 0056e evidence for the default active E3 path. |
| Two pre-model E3 attempts failed before checkpoint/model construction | Supported by source ordering and 0056e evidence: configured VnCoreNLP fails in `build_segmenter`; explicit `/models/hf` cache-dir fails at local tokenizer/config lookup before checkpoint deserialization/model construction. |
| Trusted legacy E3 policy exact digest only | Supported by `checkpoint_loading.TRUSTED_LEGACY_E3_SHA256` and `assert_load_is_permitted`. |
| Unknown/unverified checkpoints cannot reach `weights_only=False` | Supported by `assert_load_is_permitted`: no digest raises; unknown digest returns `WEIGHTS_ONLY` with `weights_only=True`; only the exact trusted digest returns false. |
| E3 checkpoint hash/size/mtime unchanged | Supported by 0056e before/after evidence and current SHA-256. |
| No notebook/training/internal_test/organizer inference | Supported by 0056e evidence and current static state. No notebook was executed during this review. |
| Temporary artifacts removed/no container remained running | Supported by 0056e cleanup evidence and the absence of generated smoke artifacts in current Git status. This review did not run a container. |
| Nothing staged/committed/pushed | Supported by current `git diff --cached --name-status` and unchanged HEAD/origin counts. |

## 4. Docker digest and metadata consistency

Consistent values across Dockerfile, ACTIVE_RUNTIME_MANIFEST and Audit 0056e:

```text
tag:             python:3.14.5-slim-bookworm
tag/index:       sha256:a9bee15510a364124aa24692899d269835683b883de42f7ebec8c293cf679ccb
linux/amd64:     sha256:ec58d916f9e24a6035cab2bdf07f6206c4cc092a16613c60597534711332d9d6
image id:        sha256:47d630a570d8f5d5194becc3ffce45f770f7c63fc119324d33186e4151b59ba6
Python:          3.14.5
torch:           2.13.0+cpu
runtime user:    mednorm
entrypoint:      python -m mednorm_vi.inference.cli
```

Non-blocking wording issue: Dockerfile comments still say "The command Audit 0055
actually ran" immediately above the 2D-C `mednorm-vi:2d-c-verified` example. That
sentence should say Audit 0056e or be rewritten as a generic verified example. The
runtime instruction itself is not affected.

## 5. Image-size interpretation

Audit 0056e records:

```text
docker inspect size: 431,104,710 bytes
docker image table: 1.95 GB
```

These can be consistent if they come from different Docker accounting views: the
API/inspect field and the user-facing image table can report different layer/accounting
quantities. The active manifest labels the table size as "table size", and Audit
0056e labels the inspect value separately, so this is not a contradiction.

Recommended wording cleanup: explicitly say "Docker inspect/API size" and
"`docker images` table size" wherever both appear, so readers do not compare them as
the same metric.

## 6. L1-L9 final architecture map

| Layer | Canonical owner | Active deterministic implementation | Learned/future seam | Checkpoint readiness/fail-closed behavior |
| --- | --- | --- | --- | --- |
| L1 Document Intelligence | `document_intelligence`, `position` | Builds `DocumentGraph`, sections, lines, tables, offset-preserving views | none planned | deterministic, no checkpoint |
| L2 Case Router | `case_router` | Multi-label route tags/priors with required positive evidence | no active learned router | deterministic, no checkpoint |
| L3 Mention Factory | `mention_factory`, `lattice` | E1/E2 deterministic plus registered E3 when specialist mode enables it | E5/E6/E7 typed slots in `expert_spec.py` | E5/E6/E7 enabled-but-unregistered are NOT_READY; E3 lazy-loads only when enabled |
| L4 Boundary/Type | `resolution.canonical.resolve_lattice_to_hypotheses` | Deterministic `resolver_v1` adapted to `EntityHypothesis` | `resolution.learned_dispatch.LearnedL4Backend` | exactly one L4 route; learned route raises, never falls back |
| L5 Assertion/Linking | `specialists/assertion`, `linking/icd10.py`, `linking/rxnorm.py` | deterministic assertion cues, ICD lexical/hierarchy, RxNorm structured graph | S2/S3/S4 future checkpoints | full mode NOT_READY for missing future artifacts |
| L6 Evidence Graph | `evidence_graph` | typed evidence graph and consistency report | future global optimization | no checkpoint |
| L7 Confidence Cascade | `confidence_cascade` | deterministic fallback plus locked-option contract | future Qwen critic/adjudicator | no model stage reachable without weights |
| L8 Decoder | `metric_decoder.decode_entities` | deterministic evidence-ranked decoder | calibrated expected-Jaccard/WER path raises unavailable | no silent calibrated substitute |
| L9 Validator/Packager | `validator.final_gate`, `inference.packaging` | final serialized gate, output dir guard, deterministic ZIP | none | raises before writing output on final validation errors |

Freeze invariants that hold statically:

- one canonical runner: `inference.pipeline.run_input_dir`;
- one canonical L4 public entry point: `resolution.canonical.resolve_lattice_to_hypotheses`;
- one final serialized L9 gate on the canonical packaging path;
- E5 random head remains unreachable;
- E6/E7 remain unavailable rather than pretending to run;
- E4 is absent from active registries/config flags and executable notebook cells;
- future E5/E6/E7 can be added through the registry/spec without runner redesign.

Known lower-level helper: `validator.submission.write_submission_zip` still exists for
layout tests and direct helper use. It is not called by the canonical inference CLI.
The canonical packaging path is `inference.pipeline -> _gate_final_output ->
write_output_directory -> package_output_zip`.

## 7. VnCoreNLP disposition

Findings:

- The active shipped `full_v1.yaml` does not set `expert_settings.e3_vncorenlp_dir`.
- `E3VietHealthBertExpert.prepare()` calls `build_segmenter()` only when
  `vncorenlp_dir` is non-empty.
- Therefore VnCoreNLP is optional in the current active runtime configuration.
- The default active E3 path was sufficient for the 2D-C default specialist smoke:
  E3 loaded, produced proposals, reached L9, and validated offsets.
- If `e3_vncorenlp_dir` is explicitly configured, readiness checks only directory
  existence and a `*.jar`. It does not verify that `py_vncorenlp` can import its
  required `jnius_config` dependency. Audit 0056e records `specialist_readiness READY
  []` before that configured path fails with `ModuleNotFoundError`.

Disposition:

```text
conditional readiness bug
framework-freeze blocker until corrected or the configured VnCoreNLP path is made
explicitly unsupported/not-ready
```

This does not invalidate the default E3 smoke, but it does violate the project rule
that readiness cannot report READY for an enabled/configured component that cannot
execute.

Related documentation ambiguity:

- `Dockerfile` and ACTIVE_RUNTIME_MANIFEST now describe VnCoreNLP as optional.
- `requirements-image.lock` still says E3 "refuses to fall back to whitespace
  segmentation".
- historical training documentation says production S1 used VnCoreNLP.
- `resolve_segmented_text(..., segmenter=None)` returns the original text as
  `pre_segmented_source`.

The owner should decide the intended E3 runtime segmentation policy and align code,
readiness and active docs to it.

## 8. Config-contract disposition

| Item | Disposition |
| --- | --- |
| `full_requires_checkpoints` names `assertion/assertion_hydra` | Truthful known limitation for full mode, later config-contract cleanup. It does not affect `specialist`, and `full` remains NOT_READY. Rename before model-phase artifact contracts harden. |
| Nested `profiles:` block mostly unread | Truthful known limitation/later cleanup. Top-level `feature_flags` drive runtime; nested flags are scanned for retired E4 but otherwise documentary. Not a freeze blocker while documented. |
| Indirect E3 checkpoint role resolution | Acceptable current contract: role `mention/vihealthbert` is satisfied through the E3 adapter and explicit `e3_checkpoint_path`. Not a blocker, but should be made less surprising before broader model-profile work. |

## 9. Unsafe-deserialization disposition

Static scan:

```text
src/mednorm_vi/mention_factory/neural/runtime.py
  active E3 inference; calls torch.load with weights_only=policy.weights_only after
  assert_load_is_permitted.

src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
  policy-only trusted legacy exception; not a torch.load call.

src/mednorm_vi/training/cli.py
  training/artifact-validation CLI; torch.load(..., weights_only=False).

src/mednorm_vi/training/phase2/artifacts.py
  artifact-validation path; torch.load(path, map_location="cpu").

notebooks/MedNorm_S1_*.ipynb
  non-executed notebook training/evaluation source.
```

No active inference `torch.load(..., weights_only=False)` exists outside the exact
trusted E3 policy. Remaining training/artifact-validation occurrences do not block
framework runtime freeze, but they must be fixed before training starts or before
artifact intake is trusted.

## 10. L9 container-evidence review

The 2D-C deterministic smoke exercised the canonical CLI path and therefore the
canonical final gate before packaging. The negative smokes directly called
`validator.final_gate`, not a duplicate validator.

This is acceptable because:

- `inference.pipeline._gate_final_output` imports and calls
  `gate_final_documents` from the same `validator.final_gate` module;
- 2D-B tests behaviorally prove `run_input_dir` propagates `FinalValidationError`
  and leaves no output directory or ZIP;
- 0056e container evidence proves the same final gate raises
  `final.text_offset_mismatch` and `kb.candidate_not_offered` inside the image.

No evidence suggests an alternate canonical packaging path bypassing final L9.

## 11. E4 retirement review

Static notebook JSON scan over code cells found no active E4 offenders:

```text
[]
```

Active source/config disposition:

- `AVAILABLE_EXPERTS` contains E1, E2, E3, E5, E6, E7; E4 is only in
  `RETIRED_EXPERTS`.
- `ExpertRegistration` refuses retired expert ids.
- `PipelineConfig.load` refuses the retired E4 feature flag and checkpoint role.
- `full_v1.yaml` has no E4 flag.
- Dockerfile has no E4 runtime variable.
- current documentation references E4 as retired/historical only.
- `training/phobert_alignment.py` remains valid E3 infrastructure.

No E4 resurrection was found.

## 12. Active-documentation consistency

Runtime-critical documentation is mostly consistent:

- Dockerfile and ACTIVE_RUNTIME_MANIFEST agree on the pinned base and runtime image
  evidence.
- ACTIVE_RUNTIME_MANIFEST accurately avoids organizer/submission readiness claims.
- Audit 0056e records the default E3 smoke and the failed configured VnCoreNLP path.

Non-blocking inconsistencies/cleanup:

1. Dockerfile comment line above the 2D-C run example still says "Audit 0055 actually
   ran" while showing the 2D-C image tag and limits. Correct to Audit 0056e or make it
   a generic example.
2. `docs/resolution/RESOLVER.md` still names deleted
   `configs/resolution/resolver_v1.yaml` as the configurable boundary-policy source.
   The active source uses `configs/resolution/boundary_type_resolver_v1.yaml`.
3. VnCoreNLP policy text is inconsistent across Dockerfile/manifest,
   `requirements-image.lock`, training history and runtime default behavior.

Item 3 is part of the freeze blocker because it is tied to readiness honesty.

## 13. Blocker versus non-blocker table

| Item | Classification | Blocks freeze? |
| --- | --- | --- |
| Explicit `e3_vncorenlp_dir` can be readiness-READY but prepare-failing due missing `jnius_config` | Conditional readiness bug | YES |
| VnCoreNLP default/required policy ambiguous across active docs and runtime | Contract ambiguity tied to readiness | YES, until owner policy is aligned |
| Dockerfile example says Audit 0055 for a 2D-C command | Documentation cleanup | NO |
| `docs/resolution/RESOLVER.md` mentions deleted resolver config | Documentation cleanup | NO |
| Docker inspect size vs image table size | Accounting wording cleanup | NO |
| `full_requires_checkpoints` role name `assertion/assertion_hydra` | Known full-mode config cleanup | NO before framework freeze; fix before model phase |
| Nested `profiles:` mostly unread | Known config cleanup | NO while documented |
| Indirect E3 checkpoint role | Known contract quirk | NO |
| Training/artifact-validation unsafe deserialization | Pre-training security work | NO for inference freeze; YES before training/artifact intake |
| Missing code-bearing/assertion gold | Data phase work | NO |
| Missing E5/E6/E7 checkpoints | Future model work | NO |
| No packaging-stress suite | Later test hardening | NO |
| `.dockerignore` absent; build context may include ignored local assets | Build-context hygiene | NO for image-content claim; recommended before repeated builds |

## 14. Final freeze verdict

```text
CORRECTIONS_REQUIRED_BEFORE_FREEZE
```

Reason: a note is acceptable only if it does not make readiness dishonest. The
configured VnCoreNLP path currently does make readiness dishonest. A default active
E3 smoke passed, but a configured optional runtime dependency reported READY and then
failed before model construction.

## 15. Exact corrections required

Required before freeze:

1. Decide and document whether VnCoreNLP is required or optional for active E3
   inference.
2. If optional, make `e3_vncorenlp_dir` readiness fail closed when the configured
   path cannot actually be imported/constructed, including the `jnius_config`
   dependency. Do not initialize heavy model/CUDA paths in readiness.
3. If required, make the default active profile require it and fix the image
   dependency so configured VnCoreNLP readiness and runtime both pass.
4. Align Dockerfile, ACTIVE_RUNTIME_MANIFEST, `requirements-image.lock` and E3
   runtime docstrings with that owner decision.

Recommended non-blocking cleanup before owner commit if convenient:

- correct the Dockerfile example sentence from Audit 0055 to Audit 0056e;
- update `docs/resolution/RESOLVER.md` to name
  `configs/resolution/boundary_type_resolver_v1.yaml`;
- add `.dockerignore` before repeated container builds so local checkpoints, data,
  indices and notebooks are not sent in the Docker build context.

## 16. Exact next phase after owner commit

After the readiness/documentation correction is made, rerun a narrow correction
validation:

- targeted static/unit test for configured `e3_vncorenlp_dir` readiness;
- changed-file formatting gate;
- one light validation pass equivalent to 2D-B for the touched files;
- if the owner wants runtime proof, one 2D-C-style configured VnCoreNLP container
  smoke under the same resource limits.

Only after a freeze verdict should the project move to the next architecture-first
phase: KB repair/freeze and creation of human-adjudicated code-bearing/assertion
gold. Do not start model downloads, training, fine-tuning, organizer inference or
submission generation before that owner commit.

## 17. Heavy-workload confirmation

This independent review did not execute notebooks, start Jupyter, load E3, call
`torch.load`, run model inference, run training/fine-tuning, access `internal_test`,
build Docker, run containers, run the full pytest suite, run mypy, download models or
download data.

Only static/read-only commands and one append-only audit creation were performed.
The final memory check showed:

```text
Mem available: 7.2 GiB
Swap used:     2.5 GiB
```

## 18. Git-operation confirmation

No file was staged. No commit or push was performed. No reset, clean, checkout,
restore or stash was performed.

After this audit is created, the only new file from this review is:

```text
docs/audits/0056f-post-2dc-independent-framework-freeze-review.md
```
