# Audit 0056e - isolated container and runtime verification (Milestone 2D-C)

**This is the 2D-C container/runtime audit.** It records digest pinning, the one
authorized Docker build, offline container smokes, E3 runtime policy verification,
cleanup evidence and remaining gaps. It is not a labels, KB-rebuild, model-download,
training, fine-tuning, organizer-inference or submission-readiness audit.

- **Scope:** Milestone 2D-C - isolated container and runtime verification
- **Date:** 2026-07-30
- **Input state:** current uncommitted working tree reviewed by Audit 0056d
- **Verdict:** `FRAMEWORK_RUNTIME_VERIFIED`

## 1. Initial Git state

Read-only prevalidation:

```text
git branch --show-current                         main
git log -1 --oneline                              bbc3d02 feat: close the framework and verify the offline container
git diff --cached --name-status                   (empty)
git rev-list --left-right --count origin/main...HEAD   0    0
```

The working tree matched the Audit-0056d inventory before 2D-C edits. Nothing was
staged, committed, pushed, reset, cleaned, checked out, restored or stashed.

## 2. Protected artifact hashes

Verified by streaming hash only:

```text
docs/MedNorm-VI_Architecture.pdf
  0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b

checkpoint/s1_mention_full_training_v1/best.pt
  a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
  size  1,615,513,303 bytes
  mtime 2026-07-27 00:53:41.047814400 +0700
```

The PDF was read in full statically with `pdftotext`. The checkpoint was not
deserialized on the host.

## 3. Memory and swap safety observations

All Docker builds and container runs were preceded by:

```text
free -h
docker ps
ps aux --sort=-%mem | head -15
```

All smokes were sequential, with no parallel containers or workers. The container
runs used:

```text
--network=none
--memory=4g
--memory-swap=5g
--cpus=2
--pids-limit=256
```

The cleanup container used a smaller 512 MiB limit because it only removed temporary
files under `/tmp/mednorm-2dc`.

Observed available RAM stayed between 6.3 GiB and 7.1 GiB. Lowest available RAM was
6.3 GiB before the Docker build. Swap was 3.0 GiB at the first check, peaked at
3.1 GiB, and ended at 2.7 GiB. No stop threshold was reached.

## 4. Base-image digest resolution

Docker:

```text
Client/Server Version: 29.5.3
Server OS/Arch: linux/amd64
docker info Architecture: x86_64
SELinux: enabled
```

Local image inspect before pull/build:

```text
python:3.14.5-slim-bookworm
RepoDigest: python@sha256:a9bee15510a364124aa24692899d269835683b883de42f7ebec8c293cf679ccb
Descriptor media type: application/vnd.oci.image.index.v1+json
Architecture: amd64
```

Official Docker Hub manifest inspection:

```text
docker.io/library/python:3.14.5-slim-bookworm
index digest: sha256:a9bee15510a364124aa24692899d269835683b883de42f7ebec8c293cf679ccb
media type:   application/vnd.oci.image.index.v1+json

linux/amd64 child manifest:
  sha256:ec58d916f9e24a6035cab2bdf07f6206c4cc092a16613c60597534711332d9d6
  media type: application/vnd.oci.image.manifest.v1+json
  source:     https://github.com/docker-library/python.git#078b07840dfee55993c57dada1e5cf99ebd16dce:3.14/slim-bookworm
  base:       debian:bookworm-slim
```

No registry other than official Docker Hub was used. A separate `docker pull` was not
needed because the image was already present and manifest inspection resolved the
official digest; the authorized build later used `--pull`.

## 5. Dockerfile digest pin

`Dockerfile` now uses:

```dockerfile
FROM python:3.14.5-slim-bookworm@sha256:ec58d916f9e24a6035cab2bdf07f6206c4cc092a16613c60597534711332d9d6
```

This is the official Docker Hub platform-specific `linux/amd64` manifest digest. The
tag-level OCI index digest is recorded separately in the Dockerfile and active
runtime manifest.

## 6. Docker build result

Exactly one Docker build was run:

```text
docker build --pull --progress=plain -t mednorm-vi:2d-c-verified .
```

Result:

```text
start time 2026-07-30T18:31:22+07:00
end time   2026-07-30T18:31:50+07:00
exit code  0
base       docker.io/library/python:3.14.5-slim-bookworm@sha256:ec58d916f9e24a6035cab2bdf07f6206c4cc092a16613c60597534711332d9d6
```

Build output resolved the pinned base and completed. The dependency and apt layers
were cached; no pretrained model or data was downloaded by the build.

No second build was run. After the successful smokes, only documentation comments
and the append-only audit were updated.

## 7. Final image metadata

`docker image inspect mednorm-vi:2d-c-verified`:

```text
image id       sha256:47d630a570d8f5d5194becc3ffce45f770f7c63fc119324d33186e4151b59ba6
repo digest    mednorm-vi@sha256:47d630a570d8f5d5194becc3ffce45f770f7c63fc119324d33186e4151b59ba6
created        2026-07-30T18:31:38.132491556+07:00
OS/Arch        linux/amd64
size inspect   431,104,710 bytes
image table    1.95 GB
user           mednorm
workdir        /app
entrypoint     ["python", "-m", "mednorm_vi.inference.cli"]
cmd            ["--help"]
```

Runtime versions from the container:

```text
Python 3.14.5
torch 2.13.0+cpu
torch.version.cuda None
```

The image is CPU-only for torch.

## 8. Image-content audit

Project-owned layers copy only:

```text
requirements-image.lock
pyproject.toml
src/
configs/
schemas/
```

Concise image history showed those COPY layers and no COPY of `checkpoint/`,
`models/`, `data/`, `indices/`, `notebooks/`, `output.zip`, governed corpus,
restricted KB payloads, model weights or submission ZIP.

The lightweight container also confirmed:

```text
/app/data      absent
/app/notebooks absent
/app/output.zip absent
```

`/app/checkpoint` and `/app/indices` are runtime mounts, not baked image content.

## 9. Mount-contract verification

The verified runtime paths agree with `configs/pipeline/full_v1.yaml`, Dockerfile
examples and the canonical runner:

```text
/app/indices     read-only KB index mount used by icd_index/rxnorm_index
/app/checkpoint  read-only E3 checkpoint mount
/models/hf       read-only Hugging Face cache via HF_HOME=/models/hf
/input           read-only fixture input
/output          only writable workflow mount
```

This host has SELinux enabled, so read-only mounts used `:ro,Z` and writable output
mounts used `:Z`. Read-only mounts were not weakened to make tests pass.

One deterministic attempt failed before packaging because the temporary host output
directory was not writable by container UID 10001. The host directory mode was
corrected outside the repository and the deterministic smoke was rerun. This did not
change read-only input/model/KB mounts.

## 10. Lightweight CLI/readiness check

One offline container verified:

```text
cli_help_first_line usage: python -m mednorm_vi.inference.cli [-h] {run} ...
uid/gid             10001/10001
workdir             /app
HF_HUB_OFFLINE      1
TRANSFORMERS_OFFLINE 1
HF_HOME             /models/hf
thread env          1 1 1 false 1
```

Readiness:

```text
deterministic READY
specialist    READY
full          NOT_READY
```

`full` blockers named E5, E6, E7 and the remaining future model/checkpoint roles,
including `mention/xlmr_mrc`, `mention/gliner`,
`mention/qwen3_1_7b_proposer`, `resolution/learned_l4_v2`,
`assertion/assertion_hydra`, dense retrieval/reranker/Qwen/calibration roles, and
the full-mode artifact path for E3.

Network/download evidence:

```text
external socket: OSError [Errno 101] Network is unreachable
HF lookup:       LocalEntryNotFoundError
```

## 11. Deterministic offline smoke

Fixture:

```text
tests/fixtures/phase1b/synthetic_medication_list.txt
```

It was copied outside the repo as `/tmp/mednorm-2dc/deterministic-input/1.txt`.
A temporary config outside the repo set only `expected_documents: 1`.

The canonical CLI was run twice inside one offline container:

```text
python -m mednorm_vi.inference.cli run --input-dir /input --output-zip /output/run1/output.zip --config /config/deterministic_1doc.yaml --mode deterministic --run-manifest /output/run1/run-manifest.json
python -m mednorm_vi.inference.cli run --input-dir /input --output-zip /output/run2/output.zip --config /config/deterministic_1doc.yaml --mode deterministic --run-manifest /output/run2/run-manifest.json
```

Both CLI runs succeeded:

```text
processed_documents: 1
```

Post-run artifact validation:

```text
deterministic_json_sha256 95952592404076edeadad36caf3645d5528d3b6508fe942ae6e90bd695ce00cf
deterministic_zip_sha256  c601e270c6823f5b2d04f2de0b4bb2198fe249bca1ec8f1ed481ea5e115e48b1
deterministic_entities    6
deterministic_candidates  21
```

The two JSON payloads were byte-identical, the two ZIP files were byte-identical, the
manifest `output_zip_sha256` matched the ZIP hash, final L9 reported no issues,
entity order was non-decreasing by `(start, end)`, and every emitted entity satisfied
`original_text[start:end] == text`.

## 12. Final L9 offset/text negative smoke

Inside the container, a tiny synthetic final-gate payload deliberately violated:

```text
original_text[start:end] == text
```

Expected result occurred:

```text
FinalValidationError
1.json:final.text_offset_mismatch
```

No `/output/output` directory and no `/output/output.zip` were created.

## 13. Final L9 offered-set negative smoke

Inside the container, a valid RxNorm code was selected from the mounted frozen
snapshot and emitted for a medication entity while the offered set for that entity
was empty.

Expected result occurred:

```text
FinalValidationError
1.json:kb.candidate_not_offered
```

No successful package was created.

## 14. Specialist/E3 offline smoke

Fixture:

```text
tests/fixtures/phase1b/synthetic_medical_document.txt
```

It was copied outside the repo as `/tmp/mednorm-2dc/specialist-input/1.txt`.

Two pre-model environmental attempts are recorded because neither deserialized the
checkpoint or constructed E3:

1. Setting `e3_vncorenlp_dir=/models/vncorenlp` failed before model load:

   ```text
   ModuleNotFoundError: No module named 'jnius_config'
   ```

   This exposes a dependency gap in the optional configured VnCoreNLP path.

2. Setting `e3_model_cache_dir=/models/hf` failed before model load:

   ```text
   LocalEntryNotFoundError / OSError: cached files not found, outgoing traffic disabled
   ```

   The Dockerfile contract is `HF_HOME=/models/hf`, whose hub cache is
   `/models/hf/hub`; passing `/models/hf` as `cache_dir` bypassed that contract.

The final E3 smoke used the active cache contract and set only:

```yaml
expert_settings:
  e3_checkpoint_path: /app/checkpoint/s1_mention_full_training_v1/best.pt
  e3_batch_size: 1
  e3_max_sequence_length: 128
```

Result:

```text
specialist_readiness READY []
runtime_policy_call TRUSTED_LEGACY_PICKLE False True
e3_record E3_vihealthbert_span_type 2 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
specialist_entities 17
specialist_manifest_zip_sha256 01e01c0ac698509b7c232488134f6509472a912c7f669bb4597c34ff2ce5ffea
specialist_smoke_ok
```

The run reached L9, L9 reported no issues, E3 appeared in expert run records, and
every emitted entity satisfied exact offset/text equality.

## 15. Trusted-legacy checkpoint-policy verification

Inside the runtime image:

```text
preload_policy TRUSTED_LEGACY_PICKLE False True
runtime_policy_call TRUSTED_LEGACY_PICKLE False True
unverified_policy_blocked CheckpointTrustError
unknown_policy WEIGHTS_ONLY True
```

This proves:

- the accepted E3 digest receives the named trusted-legacy policy;
- the actual E3 load path calls `assert_load_is_permitted`;
- an unverified checkpoint cannot reach a load policy;
- an unknown digest is restricted to `weights_only=True`.

## 16. E3 before/after hash, size and mtime

Before E3 smoke:

```text
sha256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
size   1,615,513,303 bytes
mtime  2026-07-27 00:53:41.047814400 +0700
```

After E3 smoke:

```text
sha256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
size   1,615,513,303 bytes
mtime  2026-07-27 00:53:41.047814400 +0700
```

The checkpoint was byte-identical and unchanged.

## 17. Offline/no-download evidence

Every runtime smoke used `--network=none`. External socket probes failed with:

```text
OSError [Errno 101] Network is unreachable
```

The image environment sets:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/models/hf
```

The failed cache-path attempt is itself evidence that runtime download did not occur:
Transformers raised a local-cache/offline error instead of fetching from the network.

No pretrained model or data was downloaded during 2D-C.

## 18. Non-root/read-only/writable-mount evidence

Runtime identity:

```text
uid 10001
gid 10001
Docker Config.User mednorm
```

Checkpoint read-only evidence:

```text
checkpoint_write_blocked_errno 30
```

Errno 30 is `EROFS`. The output mount was the only writable workflow mount. The
temporary output directory needed host-side permission adjustment for UID 10001; the
read-only input, checkpoint, cache and KB mounts remained read-only.

## 19. Deterministic packaging evidence

The deterministic fixture was packaged twice in one offline container. Evidence:

```text
JSON SHA-256 run1 == run2
ZIP SHA-256  run1 == run2
manifest output_zip_sha256 == actual ZIP SHA-256
ZIP namelist == ["output/1.json"]
```

This proves deterministic packaging reproducibility for the lightweight fixture.
It does not claim neural bitwise determinism for E3; E3 was intentionally run only
once after the pre-model environmental corrections.

## 20. Cleanup evidence

Temporary smoke artifacts were created only under `/tmp/mednorm-2dc`.

Host cleanup first failed because container-created files were owned by UID 10001.
A minimal root cleanup container, mounted only on `/tmp/mednorm-2dc`, removed those
files. The empty directory was then removed from the host.

Final cleanup checks:

```text
docker ps                         no running containers
git diff --cached --name-status   (empty)
free -h final                     6.7 GiB available, 2.7 GiB swap used
```

No generated smoke artifact entered repository Git status.

## 21. Exact changed-file inventory

Tracked modified/deleted files after 2D-C source/doc updates:

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

Untracked files after this audit is created:

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

Nothing is staged.

## 22. Remaining KB/data/model gaps

- VnCoreNLP configured path is not runtime-ready in this image: `py_vncorenlp`
  imports missing `jnius_config`. The active successful E3 smoke left VnCoreNLP
  unset; fix the dependency before requiring `e3_vncorenlp_dir`.
- `full` mode remains truthfully NOT_READY for E5, E6, E7 and other future
  checkpoint/model roles.
- `full_requires_checkpoints` still names `assertion/assertion_hydra`, while the
  future learned artifact is S2.
- Much of the nested `profiles:` block in `configs/pipeline/full_v1.yaml` remains
  unread by runtime configuration.
- E3 checkpoint role resolution remains indirect through the E3 adapter.
- Deterministic output ordering is enforced although organizer relevance remains
  unconfirmed.
- No packaging-stress suite exists.
- Code-bearing ICD/RxNorm gold and assertion gold remain absent, so candidate and
  assertion quality remain unmeasurable.
- Training/artifact-validation `torch.load` sites outside canonical inference remain
  deferred to a pre-training security milestone.
- Repository-wide formatter debt remains separate from 2D-C.

## 23. Acceptance-criteria table

| Criterion | Result |
| --- | --- |
| Docker base pinned to verified official digest | PASS |
| Exactly one authorized build succeeds | PASS |
| Image metadata and content truthful | PASS |
| Container runs as non-root | PASS |
| Runtime network disabled | PASS |
| Runtime downloads cannot occur | PASS |
| Deterministic mode completes offline | PASS |
| Final serialized L9 executes in container | PASS |
| Offset/text mismatch fails closed in container | PASS |
| Candidate-not-offered fails closed in container | PASS |
| Specialist/E3 completes one offline fixture smoke | PASS |
| E3 uses hardened checkpoint-loading policy | PASS |
| E3 checkpoint remains byte-identical and read-only | PASS |
| No notebook executed | PASS |
| No model/data downloaded | PASS |
| No training/fine-tuning | PASS |
| `internal_test` not accessed | PASS |
| Full mode remains truthfully NOT_READY | PASS |
| Temporary artifacts cleaned | PASS |
| No file staged | PASS |
| No commit | PASS |
| No push | PASS |
| Audit 0056e contains real evidence | PASS |

## 24. Verdict

```text
FRAMEWORK_RUNTIME_VERIFIED
```

This verdict applies to the current active framework/runtime boundary. It does not
claim organizer readiness, submission readiness, candidate quality, assertion quality
or a final deployment profile.

The VnCoreNLP configured-path dependency gap is real and recorded, but the active E3
specialist smoke completed offline without that optional setting.

## 25. Recommendation for independent post-2D-C review

Run an independent read-only/static review of Audits 0056a-0056e and the current
working tree, with special attention to:

- Dockerfile digest/source-state documentation after the post-build comment updates;
- VnCoreNLP dependency policy and whether `e3_vncorenlp_dir` should be mandatory;
- full-mode checkpoint role naming, especially `assertion/assertion_hydra`;
- remaining unsafe deserialization in training/artifact-validation paths;
- whether the 2D-C evidence is sufficient to freeze the framework/runtime boundary
  before starting KB repair, labels, model selection or training.

No staging, commit, push, reset, clean, checkout, restore or stash was performed.
