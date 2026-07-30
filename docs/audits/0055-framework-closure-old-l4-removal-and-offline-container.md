# Audit 0055 — Framework Closure: Old-L4 Removal and the Offline Container

- **Date:** 2026-07-30
- **Author:** Claude (AI agent), for human review
- **Change type:** Framework closure. The obsolete non-canonical L4 is migrated and
  **deleted**; the offline container is **built and smoke-tested**. **No commit. No push.**
- **Status:**

```text
FRAMEWORK_COMPLETE
DETERMINISTIC_ARCHITECTURE_COMPLETE
PRETRAINED_READY
ENVIRONMENT_REPRODUCIBLE
```

  Every acceptance criterion in the Milestone 2C list is met and evidenced — §18 is the
  table. **This is a statement about the framework, not about organizer readiness.**
  Candidate quality is still `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`, eight of nine
  layers have no trained checkpoint, and no organizer inference has ever run. §16 and
  §17 are the honest limits; §21 is what comes next.
- **Constraints held:** no model downloaded; no training or fine-tuning; no calibration
  fitted; `internal_test` never opened; no organizer inference; **no organizer
  `output.zip`**; no thresholds optimized; no assistant control file created or tracked;
  the architecture PDF unmodified.
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1, SHA-256
  `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b` — read in full,
  **unchanged**.

---

## 1. Initial Git state

```text
git branch --show-current   main
git log -1 --oneline        3740722 feat: complete deterministic L5-L7 plumbing and add
                                    run reproducibility   (committed by the owner)
origin/main                 3740722a23b0a60cfee0813fe9f502bc3a32c196 — in sync
git status --short -uall    (empty — clean)
tracked files               624
docker version              client 29.5.3 / server 29.5.3
docker pull python:3.14.5-slim-bookworm
                            Status: Image is up to date
                            digest sha256:a9bee15510a364124aa24692899d269835683b883de42f7ebec8c293cf679ccb
```

**Both Milestone 2C preconditions were met at this attempt, and neither was met at the
previous one.** The immediately preceding attempt stopped at §1 and reported two
blockers: the tree held 24 uncommitted Audit-0054 paths, and Docker Hub returned `401`
from stale credentials in `~/.docker/config.json`. The owner cleared both. Audit 0054 is
committed in `3740722` (verified by `git log -1 --name-only`, which lists
`docs/audits/0054-…md`, `requirements-image.lock`,
`src/mednorm_vi/confidence_cascade/escalation.py`,
`src/mednorm_vi/evidence_graph/consistency.py`, `src/mednorm_vi/inference/manifest.py`,
`src/mednorm_vi/linking/{icd10_hierarchy,rxnorm_graph,structured_medication}.py`).

Assistant control files, filesystem **and** index: `CLAUDE.md` absent/absent,
`AGENTS.md` absent/absent, `.claude/` absent/absent.

No unexplained artifact. The git-ignored entries under `reports/` and
`docs/MedNorm-VI_Architecture_Vietnamese.pdf` are pre-existing generated output and an
operator-supplied translation; neither was created here.

### Baseline suite, recorded before editing

```text
1 failed, 1728 passed, 1 skipped in 323.70s
FAILED tests/unit/test_old_l4_characterization.py::
       test_the_known_importers_of_the_old_resolver_are_the_documented_ones
```

**That failure is a latent defect in a test this agent wrote in Audit 0054, and it is
reported rather than smoothed over.** The test used `git grep` to enumerate importers of
the obsolete resolver. `git grep` searches only **tracked** files, so while
`test_old_l4_characterization.py` was untracked the test could not see its own import
and passed; once the owner committed it in `3740722`, it found itself and failed. The
committed baseline is therefore 1728 passed / 1 failed, not the 1729 passed measured
while the file was untracked. The module is deleted in §6, which removes the test along
with the defect.

## 2. Architecture PDF and E3 checkpoint integrity

```text
PDF   sha256 0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b
E3    sha256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
      size   1,615,513,303 bytes
      mtime  2026-07-27 00:53:41.047814400 +0700
```

Both unchanged from Audits 0051-0054, and **re-verified after the container mounted the
checkpoint read-only** (§11): identical digest, identical mtime, and
`verify_checkpoint_unchanged()` passed inside the container.

## 3. Old-L4 behaviour inventory

`src/mednorm_vi/resolution/resolver.py`, 121 lines, sha256 `49d25775b9b1643a…5eb809d`,
compared in full against `canonical.py`, `resolver_v1.py`, `boundary.py`, `typing.py`,
`overlap.py`.

**Unique to the retired resolver — a configurable per-type boundary policy:**

```text
medication_boundary    full | name_only | name_strength | name_strength_route
test_result_boundary   value_only | value_unit (a.k.a. value_with_unit)
abstain_on_conflict    true | false
```

**Not unique** — the canonical path already had these, with more evidence: proposal
grouping (`features.group_key`), type assignment, same-type overlap resolution,
`has_result` retention, scoring, punctuation trimming and unit/dosage attachment (both
already in the *shared* `boundary.py`), per-type tie-breaking, warning/provenance.

**A limitation, deliberately not preserved:** the retired resolver forced every type
outside `{THUỐC, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM}` to `unresolved` with a warning.
That was a capability gap, not a behaviour — the canonical L4 resolves all five
organizer types, which is how E3's DIAGNOSIS and SYMPTOM spans reach L5 at all. Dropping
it is recorded here as an improvement rather than migrated.

Live importers at the start:

```text
src/mednorm_vi/resolution/__init__.py        re-exported ResolverConfig, resolve
src/mednorm_vi/phase1c_foundation/cli.py     ran it in `resolve-phase1b-debug-output`
src/mednorm_vi/phase1c_foundation/doctor.py  validated its config loads
tests/unit/test_resolution.py                11 behavioural tests
configs/resolution/resolver_v1.yaml          its config file (3 keys, nothing else read it)
```

## 4. Migration mapping

| Retired | Canonical destination | Notes |
| --- | --- | --- |
| `medication_boundary` | `boundary.group_preference.medication` | the mechanism existed but was **inert** — see below |
| `test_result_boundary` | `boundary.group_preference.test_result` | both spellings of `value_unit` accepted |
| `abstain_on_conflict` | `overlap.abstain_on_conflict` | new field; abstention stays distinct from rejection |
| the ladder itself | `boundary.preference_rank` / `preference_note` | **one** shared function |
| `configs/resolution/resolver_v1.yaml` | `configs/resolution/boundary_type_resolver_v1.yaml` | old file deleted |
| `PipelineConfig.resolver_config` | removed; `l4_config` is the only L4 config field | a profile declaring the old key is **refused** |

**The mechanism existed and did not work.** `BoundaryPolicy.group_preference` was already
in the canonical config, and its own comment claimed it "mirrors
configs/resolution/resolver_v1.yaml so the deterministic foundation and v1 cannot
silently disagree". Two things were wrong with that:

1. **The selection was binary, not a ladder.** `_select_within_boundary_groups` sorted by
   `0 if kind == preferred else 1`, so it could only express "exact match or not". With
   `medication_boundary: name_strength` on a group offering only `name_only` and `full`,
   everything tied at rank 1 and utility decided — which could pick the **wider** span,
   the opposite of the retired resolver's documented step-down.
2. **A mirror is what drifts.** Two ladders described as mirroring each other are two
   ladders. There is now one function, and a test asserts `preference_rank` is defined in
   exactly one file.

The ladder, stated once: the exact configured kind if the group offers it; otherwise the
**widest kind at or below** the policy's width (a narrower span is a safe under-read);
otherwise the **narrowest above** it. Test-result: the configured kind, else `value_only`
(the conservative reading while the organizer convention is unresolved), else anything.

Every rung is recorded in provenance — `EntityHypothesis.boundary_evidence.policy` now
carries e.g. `policy=name_only:exact` or
`policy=name_strength:fallback_widest_at_or_below_target:kind=name_only`, via the new
`ResolutionDecision.boundary_policy` field. The reason **code** stayed stable
(`boundary_alternative_not_selected`); an earlier revision appended the note to the code
and broke `test_resolver_v1.py`, which was right to fail — a machine-readable code should
not carry free text.

Requirements held, each with its evidence:

```text
one canonical public L4 entry point     resolution.canonical.resolve_lattice_to_hypotheses
policies operate on SpanLattice         _boundary_groups reads SourceEvidence.boundary_group_id
medication components survive           test_relation_is_optional_evidence + §14 control
TEST_NAME-TEST_RESULT pairs survive     has_result_pair_group_ids == ("pg1",) asserted
unit/dosage attachment stays exact      test_unit_and_dosage_attachment_stays_exact
punctuation trimming deterministic      trim_span unchanged; §14 control identical
abstention distinct from rejection      test_abstention_is_distinct_from_rejection
unsupported types stay unresolved       abstention path unchanged; §3 records the widening
every policy decision recorded          boundary_evidence.policy, asserted in two tests
learned L4 v2 uses the same lattice     test_the_learned_l4_v2_stays_disabled_and_fail_closed
learned L4 disabled without checkpoint  enable_l4_learned_v2 false, asserted
```

### A defect the migration exposed

**Boundary expansion was silently undoing the boundary policy.** With
`medication_boundary: name_only`, the ladder correctly selected `amlodipine` (0,10) and
then `expand_to_competitor` adopted the group's own `full` sibling (0,25) — it shares the
left edge, adds 15 ≤ 40 characters, and carries enough grammar completeness. The policy
had **no observable effect on the output**.

That is why the equivalence harness first reported 38 disagreements out of 84 while the
decision records agreed: the shaped coordinates had been expanded past the selection.
`_competitors` now excludes siblings of the node's own boundary group. Expansion exists
to adopt a boundary *another expert* proposed; a node's own group is the alternative
ladder the policy has already chosen from.

## 5. Equivalence proof

Run **before** deletion, with both implementations live, over all four tracked fixtures ×
3 medication policies × 2 test-result policies:

```text
boundary-group decisions compared : 84
agreements                        : 84
disagreements                     : 0
```

**Claim scope, stated precisely.** What is proven is that both implementations select the
**same boundary alternative within each boundary group** — exactly what was migrated.
Accepted-set equality is **not** claimed and would be false: the canonical L4 also applies
type utilities, abstention, trim/expand shaping and a near-complete overlap competition,
none of which the retired resolver had. The comparison therefore reads the canonical
**decision records** (which carry `original_start`/`original_end` and the group verdict)
rather than the shaped output coordinates, and it only compares groups with more than one
alternative, because a policy cannot matter where there is no choice.

The shared-ladder ranks, which both paths consumed:

```text
medication policy=full           {name_only:4, name_strength:3, name_strength_route:2, full:0}
medication policy=name_only      {name_only:0, name_strength:101, name_strength_route:102, full:103}
medication policy=name_strength  {name_only:2, name_strength:0, name_strength_route:102, full:103}
test_result policy=value_only    {value_only:0, value_unit:100}
test_result policy=value_unit    {value_only:1, value_unit:0}
```

After deletion the equivalence harness cannot run, so its evidence lives here and the
behaviour is held by the migrated policy tests in `tests/unit/test_resolution.py` (16
tests, including one per policy) and the ladder tests in
`tests/unit/test_framework_closure.py`.

## 6. Old-resolver deletion and import proof

Deleted:

```text
src/mednorm_vi/resolution/resolver.py            121 lines, sha256 49d25775…5eb809d
configs/resolution/resolver_v1.yaml              3 keys, all migrated (§4)
tests/unit/test_old_l4_characterization.py       288 lines — its 13 behavioural
                                                 assertions are migrated into
                                                 test_resolution.py and
                                                 test_framework_closure.py
```

Proofs:

```text
$ python -c "import mednorm_vi.resolution.resolver"
ModuleNotFoundError: No module named 'mednorm_vi.resolution.resolver'

git grep for  resolution\.resolver | from \.resolver import | import resolver
  -> only prose in docs/audits/*, ACTIVE_RUNTIME_MANIFEST.md, and four modules'
     docstrings recording the deletion. No import.

git grep -- '*.yaml' '*.yml' for medication_boundary | test_result_boundary | resolver_config
  -> no matches.

AST scan of inference/pipeline.py and phase1c_foundation/cli.py
  -> both import `resolve_lattice_to_hypotheses`; neither imports `resolve`.

mednorm_vi.resolution has no attribute `resolve`, `ResolverConfig` or `resolver`,
and none appears in __all__.

whole-package import sweep: 267 modules, 0 failures (host) / 267, 0 (container)
```

**No compatibility alias and no deprecated shim.** `resolution/__init__.py` exports the
canonical entry point and nothing that competes with it;
`test_no_compatibility_alias_or_shim_exists` asserts it.

**Stale keys are refused, never ignored:**

```text
PipelineConfig.load(profile with resolver_config:)  -> ValueError naming l4_config
phase1c_foundation.cli --resolver-config            -> parser.error naming --l4-config
```

## 7. Canonical CLI and doctor migration

**`phase1c_foundation/cli.py`** — `resolve-phase1b-debug-output` now runs
L1 → L2/L3 → `build_span_lattice` → `resolve_lattice_to_hypotheses`, i.e. the same two
calls the canonical runner makes, and stops at L4. It is **not** a second inference
runner: no L5-L9, no linking, no packaging, no `output.zip`, and its help text says so
and points at `python -m mednorm_vi.inference.cli run`. Live output:

```text
MedNorm-VI Phase 1C — canonical L4 debug view (deterministic; not finals)
l4 entry point  : resolution.canonical.resolve_lattice_to_hypotheses
l4 version      : l4-canonical-lattice-v1
l4 config       : configs/resolution/boundary_type_resolver_v1.yaml
l4 config sha256: 46203c915833e07a1a6424b2268bc04e8a396e4f76e97370c4c9760effe540b6
boundary policy : {'medication': 'full', 'test_result': 'value_only'}
abstain on tie  : False
lattice nodes   : 19
hypotheses      : 19   accepted 6   rejected 13   unresolved 0
validation      : OK
```

**`phase1c_foundation/doctor.py`** — validates the **canonical** config, and not merely
that it loads: it checks that both migrated boundary policies are actually declared, and
reports a missing one as a missing local resource. A config that loaded but declared no
policy would silently disable the ladder.

```text
canonical L4          : ready (resolution.canonical.resolve_lattice_to_hypotheses)
  config              : configs/resolution/boundary_type_resolver_v1.yaml
  config sha256       : 46203c915833e07a1a6424b2268bc04e8a396e4f76e97370c4c9760effe540b6
  boundary policy     : {'medication': 'full', 'test_result': 'value_only'}
  abstain on tie      : False
```

`tests/unit/test_resolution.py` was rewritten: all 11 behavioural assertions now run
through the canonical L4 over a real lattice, plus 5 new ones for the migrated policies.
Making them pass surfaced two fixture-fidelity problems worth recording, because both
were the tests being unfaithful rather than the code being wrong: synthetic medication
proposals carried the laboratory route `C2` (so the MEDICATION route prior never applied),
and they carried no `ComponentSpan` records (so `grammar_completeness` was 0 and every
node scored 0.200 against a 0.30 abstention floor and abstained). Both now match what
Phase 1B actually supplies.

## 8. Docker dependency audit

```text
Dockerfile                base python:3.14.5-slim-bookworm, JRE only, offline env,
                          non-root uid 10001, one ENTRYPOINT, no weight/data COPY
requirements.lock         the full development environment, incl. torch==2.13.0+cu126
                          and the verification tools (pytest/ruff/mypy)
requirements-image.lock   the inference image's set: same upstream torch version from
                          the CPU index, no dev tools
```

Audited and corrected here:

- **The image lock was wrong, and the build proved it.** An earlier revision (written in
  Audit 0054) enumerated the transitive closure **from memory**. `fsspec==2026.5.0` does
  not exist on PyPI, and the build failed at the pip step. Every pin is now read from
  `importlib.metadata` in the environment that produced the measurements —
  `fsspec==2026.6.0`, `networkx==3.6.1`, `certifi==2026.7.22`,
  `charset-normalizer==3.4.9`, `idna==3.18`, `urllib3==2.7.0`,
  `typing_extensions==4.16.0`. **No pin was loosened to make the build pass**; the wrong
  versions were replaced with the real ones.
- **Two dead environment variables removed.** `MEDNORM_E3_CHECKPOINT=/models/e3/best.pt`
  and `MEDNORM_VNCORENLP_DIR=/models/vncorenlp` were declared by Audit 0053 and **nothing
  in the source ever read them** (`grep` over `src/` returns nothing), and they pointed at
  paths the active profile does not use. A configuration variable no code reads is worse
  than none: it tells an operator they have configured something when they have not.
- **Both package indexes are named explicitly** in the `RUN`, so the build cannot inherit
  an ambient pip configuration and silently resolve a different torch build. The build log
  records the wheel it actually took.

## 9. Actual build result

```text
$ docker build --pull -t mednorm-vi:framework-closed .
EXIT=0   DURATION=164s

base image        python:3.14.5-slim-bookworm
base digest       sha256:a9bee15510a364124aa24692899d269835683b883de42f7ebec8c293cf679ccb
final image id    c65344f6e040
final image size  1.95 GB
torch profile     CPU  (the default; a GPU image is the same Dockerfile with
                        --build-arg TORCH_INDEX_URL=…/cu126
                        --build-arg TORCH_SPEC=torch==2.13.0+cu126)
torch wheel        torch-2.13.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl
                   from https://download.pytorch.org/whl/cpu
in-build check     python -c "import torch; print(torch.__version__)" -> 2.13.0+cpu
import smoke       mednorm_vi + inference.pipeline + validator.kb_membership
                   + evidence_graph.consistency + confidence_cascade.escalation
                   + inference.manifest   -> ok
dependency install SUCCESS on the second attempt; the first failed on the invented
                   fsspec pin (§8) — recorded rather than quietly fixed
```

**Deviations from the local development environment, all deliberate and recorded:**

| Item | Local | Image | Why |
| --- | --- | --- | --- |
| torch variant | `2.13.0+cu126` | `2.13.0+cpu` | same upstream version; every recorded E3 validation run in Audits 0052-0055 was a CPU forward pass, and the CUDA runtime would add ~2.5 GB the inference path never touches |
| verification tools | pytest, ruff, mypy present | absent | the inference image runs inference; `requirements.lock` remains the file a verification image installs |
| model weights | `checkpoint/` on disk | **not in the image** | mounted read-only at run time |
| KB indices | `indices/` on disk | **not in the image** | mounted read-only; RxNorm is UMLS-licensed |
| governed corpus | `data/` on disk | **not in the image** | licence-restricted |

Nothing forbidden is baked in: every `COPY` line is `pyproject.toml`, `src/`, `configs/`,
`schemas/`, and a test asserts no `COPY` touches `checkpoint`, `models/`, `data/`,
`indices/` or `notebooks`.

## 10. Offline deterministic smoke

All eleven required checks, `--network=none` throughout:

```text
 1. canonical CLI help                 OK  (usage: python -m mednorm_vi.inference.cli)
 2. package import sweep               267 modules, 0 failures
 3. deterministic readiness            deterministic=READY specialist=READY full=NOT_READY
 4. full mode fail-closed              RuntimeError raised
 5. no external API / network          socket.create_connection -> OSError
                                       HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
 6. no runtime model download          snapshot_download -> LocalEntryNotFoundError
 7. non-root runtime user              uid=10001 gid=10001 (mednorm)
 8. read-only model/KB/input mounts    /app/checkpoint EROFS  /app/indices EROFS
                                       /input EROFS
 9. writable output mount only         /output write+unlink OK
10. no organizer inference             only tracked fixtures used
11. no organizer output.zip            /output empty after both runs
```

Bounded tracked-fixture `run_document` smoke, deterministic mode, inside the container:

```text
1.txt  experts=[]  ents=6   l4={accepted 6, rejected 13, unresolved 0, total 19}
       edges={supports 6, has_assertion 6, has_candidate 93, modified_by 20, in_section 6}
       consistency={SUPPORTED 30, UNRESOLVED 12, NOT_APPLICABLE 6}  fatal=0
       L7={UNRESOLVED 6}
2.txt  experts=[]  ents=15  l4={accepted 15, rejected 2, unresolved 1, total 18}
       edges={supports 15, has_assertion 15, has_result 7, modified_by 14, in_section 15}
       consistency={SUPPORTED 44, CONTRADICTED 1, NOT_APPLICABLE 30}  fatal=0
       L7={ACCEPT 14, UNRESOLVED 1}
every emitted offset re-verified: original_text[start:end] == text
```

Identical to the host results, which is the point of the exercise.

### Two mount-contract defects that only a real run could find

Audit 0054 authored the mount contract and could not build the image. Running it broke
the contract twice:

**(a) The documented mount points did not match the paths the active profile resolves.**
`configs/pipeline/full_v1.yaml` declares `indices/...` and E3 defaults to
`checkpoint/...`, both **relative**, and `WORKDIR` is `/app`. So they mount at
`/app/indices` and `/app/checkpoint` — not the `/kb/indices` and `/models/e3` Audit 0053
documented. With the old paths, `deterministic` mode reported:

```text
NOT_READY ('missing_icd_index:indices/icd10_vi/tt06-2026/index.json',
           'missing_rxnorm_index:indices/rxnorm/prescribable-2026-07-06/index.json')
```

Fail-closed and correct, but for a reason that looks like a missing asset rather than a
wrong contract. The Dockerfile now documents the paths that work, and notes that a
profile pointing at `/kb` and `/models` is equally valid provided the two agree.

**(b) On an SELinux-enforcing host every bind mount needs a relabel suffix.** `/output`
was unwritable at mode `0777` and `/input` unreadable, because the denial is a label
mismatch and has nothing to do with mode bits:

```text
host: getenforce -> Enforcing ; docker SecurityOptions include name=selinux
container: /output is drwxrwxrwx 1000:1000, tmpfs rw,seclabel  -> touch: Permission denied
with :ro,Z on assets and :Z on output               -> all mounts behave correctly
```

The Dockerfile's example run command now carries `:ro,Z` / `:Z` and says to drop `,Z` on
a host without SELinux.

## 11. Offline specialist / E3 smoke

Checkpoint, HF cache and VnCoreNLP mounted **read-only**; `--network=none`:

```text
checkpoint dir writable?              read-only (errno EROFS)
E3 readiness                          True
E3 loaded OFFLINE                     revision f89e80b461e86f9cfc1c84019bd819830c24b6c5
1.txt  experts=['E3_vihealthbert_span_type']  ents=6   l4={accepted 6, rejected 13}
2.txt  experts=['E3_vihealthbert_span_type']  ents=15  l4={accepted 15, rejected 2, unresolved 1}
every emitted offset re-verified      original_text[start:end] == text
verify_checkpoint_unchanged()         PASS (inside the container)

checkpoint BEFORE  sha256 a64cc173…1017c   mtime 2026-07-27 00:53:41.047814400 +0700
checkpoint AFTER   sha256 a64cc173…1017c   mtime 2026-07-27 00:53:41.047814400 +0700
```

E3 loads from the mount with no network and no download; specialist mode runs E1/E2/E3;
the container cannot write into the checkpoint directory; offsets remain exact; the
checkpoint is byte-identical with an unchanged mtime. **The checkpoint is not baked into
the image** — a test asserts no `COPY` line references it, and
`docker inspect` shows no checkpoint layer.

## 12. Final L1–L9 framework matrix

See `docs/architecture/ACTIVE_RUNTIME_MANIFEST.md` §0a for the full table. Summary:

| Layer | Canonical path | Integrated | Determ. complete | Pretrained adapter | Checkpoint | Duplicate impl. |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | `document_intelligence/` | YES | YES | n/a | n/a | none |
| L2 | `case_router/` | YES | YES | n/a | n/a | none |
| L3 | `mention_factory/registry.py` + `lattice/` | YES | YES | E5/E6/E7 typed, fail-closed | **E3 only** | none |
| L4 | `resolution/canonical.py` | YES | YES | `learned_v2` typed, disabled | none | **none — deleted here** |
| L5 assertion | `specialists/assertion/` | YES | YES | S2 typed | none | none |
| L5 ICD | `linking/icd10*.py` | YES | YES | S3/S4 typed | none | none |
| L5 RxNorm | `linking/rxnorm*.py` + `structured_medication.py` | YES | YES | S3/S4 typed | none | none |
| L6 | `evidence_graph/graph.py` + `consistency.py` | YES | YES | n/a | none | none |
| L7 | `confidence_cascade/` | YES | YES | S5 typed, no backend | none | none |
| L8 | `metric_decoder/decoder.py` | YES | YES | S6 typed, fails closed | none | none |
| L9 | `validator/` + `inference/packaging.py` | YES | YES | n/a | none | none |

Every closure condition verified:

```text
one canonical runner                             YES  inference/pipeline.py
one canonical L4                                 YES  resolution.canonical
no obsolete resolver                             YES  ModuleNotFoundError (§6)
every deterministic L1-L9 path integrated        YES  run_document end to end (§10, §11)
every learned/pretrained slot fail-closed typed  YES  full mode NOT_READY, 12 blockers
container build succeeds                         YES  §9
offline deterministic smoke succeeds             YES  §10
offline specialist/E3 smoke succeeds             YES  §11
no runtime download is possible                  YES  LocalEntryNotFoundError
no model/restricted data in Git or image         YES  COPY audit + git ls-files
```

## 13. Tests and static checks

New: `tests/unit/test_framework_closure.py` — 336 lines, **29 tests** in five sections:
(A) the obsolete L4 is gone — file, config, import, no alias/shim, no tracked reference,
one L4 entry point by AST; (B) stale keys refused, not ignored — profile, dataclass
field, CLI flag; (C) the ladder is one shared implementation — per-policy ranking,
step-down-before-step-up, `value_with_unit` aliasing, unconfigured policy, provenance
note, single definition site; (D) canonical config carries the migrated policies, learned
v2 disabled and fail-closed, full mode fail-closed; (E) container contract — one
entrypoint, non-root, offline env, no weight/data `COPY`, documented read-only mounts,
every lock line `==`-pinned, no weights tracked in Git.

Rewritten: `tests/unit/test_resolution.py` — 16 tests, all through the canonical L4.
Deleted: `tests/unit/test_old_l4_characterization.py` — its assertions are migrated.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q
    1750 passed, 1 skipped in 346.38s
    (baseline 1728 passed / 1 failed / 1 skipped; the failure is fixed by deleting the
     module that contained it, and +22 net tests were added)
ruff check .                    All checks passed!
ruff check notebooks            All checks passed!
env PYTHONPATH=src .venv/bin/python -m mypy
    Success: no issues found in 276 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src     clean
git diff --check                                             clean
whole-package import sweep      267 modules, 0 failures (host and container)
```

Defects found and fixed during the work, reported rather than smoothed over:

```text
1. group_preference was INERT: binary match instead of a ladder (§4).
2. Boundary EXPANSION silently undid the boundary policy (§4).
3. My own Audit-0054 importer test used `git grep`, which cannot see untracked
   files, so it passed while untracked and failed once committed (§1).
4. requirements-image.lock listed transitive pins from memory; the build failed on
   the non-existent fsspec==2026.5.0 (§8).
5. The documented container mount points did not match the active profile (§10a).
6. Bind mounts need SELinux relabelling on this host (§10b).
7. Two MEDNORM_* env vars that nothing read (§8).
8. Appending the policy note to a stable reason code broke test_resolver_v1 —
   the note belongs in its own field (§4).
9. Two test-fixture fidelity bugs: wrong route, missing ComponentSpans (§7).
```

## 14. Bounded validation control

Same 200 rows, same digest-resolved split, `internal_test` never opened. The purpose is
narrow: prove the L4 migration did **not** change span/type behaviour.

```text
split  data/derived/training_corpora/mednorm_vi_training_v1/splits/validation.jsonl
       sha256 ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103
       bounded to the first 200 of 1,045 rows

arm                       P        R       F1     TP    FP    FN
E1+E2 (deterministic) 0.0000   0.0000   0.0000     0     5   406
E3 only               0.6220   0.5025   0.5559   204   124   202
E1+E2 + E3            0.6126   0.5025   0.5521   204   130   202

per type (merged arm)  DIAGNOSIS   P 0.6600 R 0.4783 F1 0.5546 TP 132 FP 68 FN 144
                       SYMPTOM     P 0.5625 R 0.5538 F1 0.5581 TP  72 FP 56 FN  58
                       TEST_NAME   1 FP     TEST_RESULT 4 FP

route eligibility      E1 elig 230 skip 5 | E2 elig 6 skip 229
C2 suppressed          7 nodes
lattice / L4           334 nodes, 0 merges; accepted 333 / rejected 0 / unresolved 1
offset violations      0
assertions             with labels 13, all three 0, uncertain 35
ICD decisions          DROP_NO_LEXICAL_SUPPORT 11,609  DROP_UNSUPPORTED_SPECIFICITY 3,328
                       KEEP_SPECIFIC_SUPPORTED 478  KEEP_BROADER_FALLBACK 193
                       KEEP_SIBLING_COMPETITION 290  KEEP_EXACT_NAME 10
                       KEEP_LEXICAL 1,128  DROP_BUDGET 113
runtime                35.7 s for 200 docs x 3 arms (0.179 s/doc)   peak RSS 3.16 GiB
E3 checkpoint          verified UNCHANGED
```

**Every figure is identical to Audits 0053 and 0054 to four decimal places, including
each per-type count and every ICD decision-reason count.** That is the control: this
milestone touched L4 selection and the container, so span/type numbers must not move, and
they did not.

**No organizer score is claimed. No thresholds were optimized.** Candidate quality
remains `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`.

## 15. Exact Git changed-file inventory

`git diff HEAD --stat`, **23 modified + 3 deleted** (26 paths, 768 insertions,
678 deletions):

```text
 Dockerfile                                        |  92 ++++++--
 configs/pipeline/full_v1.yaml                     |   1 -
 configs/resolution/boundary_type_resolver_v1.yaml |  29 ++-
 configs/resolution/resolver_v1.yaml               |  15 ---   (DELETED)
 docs/architecture/ACTIVE_RUNTIME_MANIFEST.md      | 130 +++++++----
 docs/audits/0053-route-gating-honest-l8-and-l9-kb-membership.md | 13 +-
 docs/audits/README.md                             |   1 +
 requirements-image.lock                           |  19 +-
 src/mednorm_vi/inference/config.py                |  17 +-
 src/mednorm_vi/inference/manifest.py              |   2 +-
 src/mednorm_vi/inference/pipeline.py              |  15 +-
 src/mednorm_vi/phase1c_foundation/cli.py          |  71 +++++-
 src/mednorm_vi/phase1c_foundation/doctor.py       |  52 +++-
 src/mednorm_vi/resolution/__init__.py             |  45 +++-
 src/mednorm_vi/resolution/boundary.py             | 127 +++++++---
 src/mednorm_vi/resolution/canonical.py            |  17 +-
 src/mednorm_vi/resolution/config_v1.py            |   6 +
 src/mednorm_vi/resolution/overlap.py              |  21 +-
 src/mednorm_vi/resolution/resolver.py             | 121 ---------   (DELETED)
 src/mednorm_vi/resolution/resolver_v1.py          |  83 +++++--
 tests/unit/test_canonical_l3_l4_spine.py          |   2 +-
 tests/unit/test_full_pipeline_v1.py               |   1 -
 tests/unit/test_old_l4_characterization.py        | 288 -----------   (DELETED)
 tests/unit/test_phase1c_doctor.py                 |  11 +-
 tests/unit/test_resolution.py                     | 279 ++++++++++++++-----
 tests/unit/test_rxnorm_full_intake.py             |   4 +-
 26 files changed, 768 insertions(+), 678 deletions(-)
```

`git ls-files --others --exclude-standard`, **2 added** (untracked files never appear in
`git diff --stat`, which is why the counts come from two commands):

```text
   336  tests/unit/test_framework_closure.py
  <this audit>  docs/audits/0055-framework-closure-old-l4-removal-and-offline-container.md
```

**Total: 23 modified + 3 deleted + 2 added = 28 paths.** `ACTIVE_RUNTIME_MANIFEST.md` and
`docs/audits/README.md` are counted in the modified list because they are modified; Audit
0052 §13 was corrected once for omitting exactly that kind of entry.

`docs/audits/0053-…md` is in this list, and it is the **one exception to the
append-only rule** in the repository's history. The Milestone 2C brief said "Do not edit
Audits 0001-0054"; this edit was made afterwards, at the owner's explicit instruction, to
strip an assistant co-author trailer from that audit's §21 commit template. GitHub derives
its contributor list from commit authors **and** co-author trailers, and the owner does not
want an AI assistant listed among the repository's contributors. The edit removes an
instruction that would have re-introduced the trailer if the template were re-run, adds a
note recording why, and changes no finding, measurement, count or verdict. The same turn
deleted the local tag `backup/pre-uncommit-0052`, which was the last reference to
`9469059` — the single commit that ever carried the trailer, and one that has not been an
ancestor of `main` since Audit 0052 was re-committed as `ce18755`.

The three deletions are **already staged**, because `git rm` stages a removal as it
performs it. Step 2 of §20 is therefore a no-op that costs nothing and keeps the command
list complete and copy-pasteable.

No model weights, governed data, indices or generated artifacts are in the change set —
verified by extension pattern (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, `.zip`) and
by the Dockerfile `COPY` test.

## 16. Remaining pretrained / download gaps

| Gap | State |
| --- | --- |
| E5 XLM-R MRC-NER | contract + trainer exist; **task head randomly initialized**; must never run untrained |
| E6 GLiNER | **no local weights**; strict adapter fails closed |
| E7 Qwen proposer | **no local weights**; loader fails closed |
| S3 dense retrieval | no embedder weights; retrieval is lexical + graph only |
| S4 cross-encoder reranker | none; depends on S3 |
| S5 critic / adjudicator | no local weights. The L7 contract is complete, so a backend is a drop-in — **but there is no backend**, and `ESCALATE` is therefore never returned |
| S6 calibration | none — still what blocks real spec §13 decoding |
| learned L4 v2 | implemented, uses the same `SpanLattice` contract, **disabled**: no checkpoint |

Every loader is lazy and `local_files_only=True`. **No model was downloaded**, on the
host or in the container, and the container cannot download one (§10 check 6). `full`
mode fails closed with 12 named blockers.

## 17. Remaining training / data gaps

| Stage | State | Blocker |
| --- | --- | --- |
| S0 domain adaptation | `SCAFFOLD_ONLY` | notebook is a design draft |
| **S1 mention** | **EXECUTED** | complete — E3 is the only trained model |
| S2 assertion | `IMPLEMENTED_NOT_RUN` | **zero assertion supervision** |
| S3 retrieval | `SCAFFOLD_ONLY` | no embedder; **zero ontology codes** to train or evaluate against |
| S4 reranking | `SCAFFOLD_ONLY` | depends on S3 |
| S5 critic LoRA | `NOT_STARTED` | no local weights |
| S6 calibration | `NOT_STARTED` | needs out-of-fold predictions no stage produces |

Data gaps, now the only thing between this framework and a measurable system:

```text
ICD-10 / RxNorm gold codes    ZERO — 40% of the metric is unmeasurable
assertion supervision         ZERO — 30% is held by regression tests only
route gold                    ZERO — routing accuracy has never been measured
INN <-> RxNorm crosswalk      ABSENT — a 12-entry REQUIRES_CLINICAL_REVIEW stopgap
ICD canonical_name quality    some records are truncated PDF fragments
KB graph relation labels      DISCARDED at index-build time in both ontologies
```

**No training or fine-tuning ran. No calibration was fitted. No thresholds were searched.**

## 18. Acceptance-criteria table

```text
MILESTONE 2C ACCEPTANCE CRITERIA                                   STATUS
old L4 policies are migrated                                       MET    §4
equivalence tests pass                                             MET    §5  84/84
resolution/resolver.py is deleted                                  MET    §6
the old module is unimportable                                     MET    §6  ModuleNotFoundError
no source/config/test/notebook references it                       MET    §6
Phase1C CLI and doctor use canonical L4                            MET    §7
exactly one L4 public entry point remains                          MET    §6  AST test
learned L4 v2 remains disabled and fail-closed                     MET    §13
Docker image builds successfully                                   MET    §9
deterministic offline container smoke passes                       MET    §10
specialist/E3 offline container smoke passes                       MET    §11
E3 checkpoint remains byte-identical and read-only                 MET    §2, §11
no runtime network access or model download occurs                 MET    §10 checks 5,6
no weights or restricted KB data enter Git or the image            MET    §9, §13
deterministic and specialist modes remain operational              MET    §10, §11
full mode remains fail-closed                                      MET    §10 check 4
full tests and static checks pass                                  MET    §13
Audit 0055 contains real evidence                                  MET    §5, §9-§14
no commit or push occurred                                         MET
```

**All nineteen criteria are met**, which is what permits the four status lines in the
header. They describe the framework; §16 and §17 describe what the framework still lacks.

## 19. Safe-to-commit verdict

```text
VERDICT: SAFE_TO_COMMIT
```

Safe because every claim is measured, every defect found along the way is recorded
(§13), and the only behavioural change outside the migration — expansion no longer
overruling the boundary policy — is a fix whose effect on span/type output is proven nil
by the §14 control.

Safety checks: working tree contains only the paths in §15; the three deletions are
intended and proven complete; the architecture PDF unmodified (§2); the E3 checkpoint
byte-identical before and after the container mounted it (§2, §11); no assistant control
file present or tracked (§1); no weights, governed data or generated artifacts staged;
`git diff --check` clean; 1750 tests pass; ruff, mypy strict, compileall and the import
sweep clean.

## 20. Exact staging and commit commands

To be run by the repository owner. **Nothing below was executed by this audit.**
`git add -A`, `git add .` and `git commit -a` are deliberately not used.

```bash
cd /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI

# 1. Re-verify the preflight
env PYTHONPATH=src .venv/bin/python -m pytest -q
ruff check . && ruff check notebooks
env PYTHONPATH=src .venv/bin/python -m mypy
env PYTHONPATH=src .venv/bin/python -m compileall -q src
git diff --check
python -c "import mednorm_vi.resolution.resolver" 2>&1 | tail -1   # must be ModuleNotFoundError

# 2. Stage the three deletions
git add configs/resolution/resolver_v1.yaml
git add src/mednorm_vi/resolution/resolver.py
git add tests/unit/test_old_l4_characterization.py

# 3. Stage the modified files
git add Dockerfile
git add configs/pipeline/full_v1.yaml
git add configs/resolution/boundary_type_resolver_v1.yaml
git add docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
git add docs/audits/0053-route-gating-honest-l8-and-l9-kb-membership.md
git add requirements-image.lock
git add src/mednorm_vi/inference/config.py
git add src/mednorm_vi/inference/manifest.py
git add src/mednorm_vi/inference/pipeline.py
git add src/mednorm_vi/phase1c_foundation/cli.py
git add src/mednorm_vi/phase1c_foundation/doctor.py
git add src/mednorm_vi/resolution/__init__.py
git add src/mednorm_vi/resolution/boundary.py
git add src/mednorm_vi/resolution/canonical.py
git add src/mednorm_vi/resolution/config_v1.py
git add src/mednorm_vi/resolution/overlap.py
git add src/mednorm_vi/resolution/resolver_v1.py
git add tests/unit/test_canonical_l3_l4_spine.py
git add tests/unit/test_full_pipeline_v1.py
git add tests/unit/test_phase1c_doctor.py
git add tests/unit/test_resolution.py
git add tests/unit/test_rxnorm_full_intake.py

# 4. Stage the new files
git add tests/unit/test_framework_closure.py
git add docs/audits/0055-framework-closure-old-l4-removal-and-offline-container.md
git add docs/audits/README.md

# 5. Confirm what is staged before committing
git status --porcelain
git diff --cached --stat

# 6. One commit
git commit -F - <<'MSG'
feat: close the framework — remove the obsolete L4 and verify the offline container

Framework closure. One canonical runner, one canonical L4, no duplicate active
implementation, every deterministic L1-L9 path integrated, every learned slot
typed and fail-closed, and an offline container that builds and passes a real
smoke. This says nothing about organizer readiness: candidate quality is still
unmeasurable and only L3 has a trained checkpoint.

L4 - the retired Phase-1C-A resolver had exactly one unique behaviour, a
configurable per-type boundary policy. It is migrated into the canonical lattice
L4 and both paths ran through ONE shared ladder before the old module was
deleted: exact configured kind, else the widest kind at or below it, else the
narrowest above it. Equivalence proven on 84 of 84 boundary-group decisions
across 3 medication x 2 test-result policies on all four tracked fixtures, 0
disagreements. Every rung is now recorded on the hypothesis's
boundary_evidence.policy.

Two defects the migration exposed. group_preference already existed in the
canonical config and was INERT: it tested "is this the preferred kind?" as a
binary, so a policy naming an absent kind fell through to utility and could pick
the wider span. And boundary EXPANSION was silently undoing the policy — with
name_only the ladder chose `amlodipine` and expand_to_competitor then adopted
the group's own `full` sibling, so the policy had no observable effect at all.
Expansion now excludes siblings of the node's own boundary group.

Deleted: resolution/resolver.py, configs/resolution/resolver_v1.yaml and the
characterization module whose assertions were migrated. The module is
unimportable, no tracked file references it, no alias or shim was left, and
stale keys are REFUSED rather than ignored — PipelineConfig.load raises on
`resolver_config` and the Phase-1C CLI errors on --resolver-config, each naming
the replacement. The Phase-1C debug CLI and the doctor now use the canonical L4;
the CLI stops at L4 and says so, so no second inference runner exists.

Container - built and smoke-tested offline for the first time, which found four
things authoring alone could not. The image lock listed transitive pins from
memory and the build failed on a non-existent fsspec version; every pin is now
read from importlib.metadata. The documented mount points did not match the
paths the active profile resolves. Bind mounts need SELinux relabelling on this
host. And two MEDNORM_* env vars were dead. All 11 offline checks pass under
--network=none, plus deterministic and specialist/E3 run_document smokes with
the checkpoint mounted read-only; it stays byte-identical.

Control: span/type F1 unchanged at 0.5559 / 0.5521 on the same 200 governed
validation rows, every per-type count identical to Audits 0053 and 0054.

Evidence: Audit 0055. 1750 passed / 1 skipped; ruff, mypy strict, compileall and
the import sweep clean. No model downloaded, no training, no calibration, no
internal_test, no organizer inference, no output.zip.
MSG

# 7. Verify, do not push
git log -1 --stat
git status
```

The `docs/audits/README.md` line to add before staging it:

```text
- `0055-framework-closure-old-l4-removal-and-offline-container.md`
```

## 21. Next data/model milestone

**The framework is done; everything left is data or training.** In dependency order:

1. **Produce the code-bearing gold set** specified in
   `evaluation.code_linking.REQUIRED_ANNOTATION_ARTIFACT`: governed validation split
   only, every DIAGNOSIS and MEDICATION mention in the sampled documents, at least two
   independent clinical annotators, an explicit adjudication step, agreement reported
   before use, split identity by SHA-256. **40% of the organizer metric is unmeasurable
   without it**, and every linking decision built in Audit 0054 is unvalidatable until it
   exists. Nothing else should be tuned first.
2. **Build an assertion corpus.** 30% of the metric, currently held by regression tests
   against constructed cases. It also unblocks S2.
3. **A governed INN ↔ RxNorm crosswalk at KB-intake time**, retiring the 12-entry
   `REQUIRES_CLINICAL_REVIEW` bridge, and **rebuild both KB indexes keeping relation
   labels** so §4/§5's careful inferences become assertions. Same intake pass: fix the
   truncated ICD `canonical_name` extraction.
4. **Calibration (S6)** once (1) exists — it is what turns L8's honest deterministic
   decoder into spec §13, and it needs out-of-fold predictions.
5. **Train the remaining L3 experts** (E5's head, E6/E7 weights) and re-run the ablation
   to see whether a wider lattice beats E3 alone. Deterministic L4 still measures below
   the E3-only baseline (0.5521 vs 0.5559), so the L4 boundary-offset head belongs here.
6. **L7 model stages last**, per spec §20's ordering. The contract, entry conditions,
   locked option sets and refusal path are complete, so a critic/adjudicator is a
   drop-in — but it should not be attached to a system whose candidate quality nobody can
   measure.

---

**Audit 0055 ends. No commit. No push. `FRAMEWORK_COMPLETE` /
`DETERMINISTIC_ARCHITECTURE_COMPLETE` / `PRETRAINED_READY` / `ENVIRONMENT_REPRODUCIBLE`
— the framework, not the system. The next milestone is data, not code.**
