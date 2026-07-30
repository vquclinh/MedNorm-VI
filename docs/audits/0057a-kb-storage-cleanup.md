# Audit 0057A: KB Storage Hygiene and Superseded Build Cleanup

Milestone: 3A.1 - KB storage hygiene and superseded build cleanup
Date: 2026-07-30
Verdict: `KB_STORAGE_CLEAN`

## 1. Initial Git State

Required read-only state checks were run before cleanup:

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `b12633e feat: freeze the verified framework and container runtime` |
| `origin/main...HEAD` | `0 0` |
| Staged files | none |
| Working tree | Milestone-3A Audit 0057 changes present and uncommitted |

`git status --short --untracked-files=all` before cleanup:

```text
 M docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
 M docs/kb/icd10/SNAPSHOT.md
 M docs/kb/rxnorm/SNAPSHOT.md
 M src/mednorm_vi/linking/rxnorm.py
 M src/mednorm_vi/linking/structured_medication.py
 M tests/unit/test_l5_l7_deterministic_stack.py
?? docs/audits/0057-kb-audit-repair-and-freeze.md
?? docs/kb/KB_FREEZE_0057.md
?? schemas/kb/candidate-representation-schema-v1.json
?? schemas/kb/governed-synonym-schema-v1.json
?? schemas/kb/icd10-kb-schema-v1.json
?? schemas/kb/inn-rxnorm-crosswalk-schema-v1.json
?? schemas/kb/kb-provenance-schema-v1.json
?? schemas/kb/kb-snapshot-manifest-schema-v1.json
?? schemas/kb/rxnorm-kb-schema-v1.json
?? src/mednorm_vi/kb/freeze.py
?? tests/unit/test_kb_freeze_0057.py
```

Audit 0057 and its source/schema/test/doc changes are therefore currently
uncommitted. No files were staged.

Protected artifacts were verified by streaming SHA-256 only:

| Artifact | Expected SHA-256 | Observed |
| --- | --- | --- |
| `docs/MedNorm-VI_Architecture.pdf` | `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b` | match |
| `checkpoint/s1_mention_full_training_v1/best.pt` | `a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c` | match |

The E3 checkpoint was not deserialized.

## 2. Required Safety Preamble

The required pre-traversal checks were run:

| Command | Result |
| --- | --- |
| `free -h` | Mem 14Gi total, 6.3Gi used, 3.7Gi free, 8.6Gi available; swap 8.0Gi total, 0B used |
| `df -h .` before cleanup | 100G size, 82G used, 19G available, 82% use |
| `docker ps` | no running containers |

`docker ps` required read-only socket access outside the sandbox. No Docker build,
run, prune, image inspection, volume operation or storage modification occurred.
Docker storage is deliberately out of scope for this audit.

## 3. Git Size Versus Local Data Size

| Area | Pre-cleanup size |
| --- | ---: |
| `.git` | `6.2M` (`5,043,572` apparent bytes, measured after cleanup and unchanged by ignored-data deletion) |
| `git count-objects -vH` | count `532`, loose size `2.82 MiB`, in-pack `1621`, packs `2`, pack size `2.70 MiB`, garbage `0 bytes` |
| Tracked working-tree files, excluding `internal_test/**` | `6,001,040` bytes (`5.724MiB`) |
| Full local project | `22G` (`23,078,714,912` bytes, `21.494GiB`) |
| `data` | `13G` (`13,827,004,379` bytes computed from post-cleanup bytes plus exact deleted bytes) |
| `data/external` | `3.3G` (`3,538,890,577` bytes, unchanged) |
| `data/derived` | `9.6G` (`10,287,585,115` bytes computed from post-cleanup bytes plus exact deleted bytes) |
| `data/derived/kb_freeze` | `9.6G` (`10,232,557,390` bytes, `9.530GiB`) |
| `indices` | `660M` (`691,731,363` bytes, unchanged) |
| `checkpoint` | `1.6G` (`1,615,513,303` bytes, unchanged) |
| `models` | `4.0K` (`718` bytes, unchanged) |

Interpretation:

- Git clone/history size is `.git` plus tracked working-tree content, roughly
  megabytes, not gigabytes.
- Tracked working-tree size is about 6.0 MB and is not the storage pressure.
- Ignored local KB/source/model data accounts for the large local footprint.
- Generated freeze outputs under `data/derived/kb_freeze` were the cleanup target.
- Docker image/container/volume storage was not measured or touched and is not in
  the reclaimed-space result.

## 4. Pre-cleanup KB-freeze Inventory

Immediate KB-freeze build directories before cleanup:

| Path | Size |
| --- | ---: |
| `data/derived/kb_freeze/0057` | `3.2G` (`3,410,858,417` bytes, `3.177GiB`) |
| `data/derived/kb_freeze/0057_candidate_v2` | `3.2G` (`3,410,849,497` bytes, `3.177GiB`) |
| `data/derived/kb_freeze/0057_final` | `3.2G` (`3,410,849,476` bytes, `3.177GiB`) |

Largest repeated files under the KB-freeze area were the generated RxNorm
review/provenance/relation CSVs:

| Relative file pattern | Per-build size |
| --- | ---: |
| `rxnorm_review_queue_v1.csv` | about `1.7G` |
| `rxnorm_provenance_v1.csv` | about `728M` |
| `rxnorm_relations_v1.csv` | about `661M` |
| `rxnorm_synonyms_v1.csv` | about `82M` |
| `rxnorm_concepts_v1.csv` | about `34M` |
| `icd10_vi_tt06_2026/` directory | about `17M` |

No symlinks were found under `data/derived/kb_freeze`.

## 5. Retention Policy

Retained unless future source evidence proves a path is mislabeled:

| Category | Paths |
| --- | --- |
| Raw/authorized sources | `data/external/icd10_vi/`; `data/external/rxnorm/prescribable-2026-07-06/`; `data/external/rxnorm/full-2026-07-06/` |
| Current legacy/normalized sources | `data/derived/icd10_vi/tt06-2026/` |
| Current active runtime indices | `indices/icd10_vi/tt06-2026/`; `indices/rxnorm/prescribable-2026-07-06/`; `indices/rxnorm/full-2026-07-06/` |
| Current 3A candidate freeze | `data/derived/kb_freeze/0057_candidate_v2/` |
| Governance and reproducibility | `data/manifests/`; `schemas/kb/`; `docs/kb/`; `docs/audits/0057-kb-audit-repair-and-freeze.md`; `src/mednorm_vi/kb/`; `tests/unit/test_kb_freeze_0057.py` |
| Protected model/architecture | `checkpoint/`; `docs/MedNorm-VI_Architecture.pdf` |

The active runtime KB was not changed. The 0057 candidate snapshot was not
activated. ICD and RxNorm were not rebuilt. Full RxNorm source was retained for
Milestone 3B relation endpoint closure.

## 6. Candidate Deletion Classification

Tracked files were searched for exact path, snapshot ID and manifest references,
excluding `internal_test/**`. Untracked pending Audit-0057 docs were also checked
for path references. Only `0057_candidate_v2` is named by retained docs.

| Path | Size | Directory metadata | Manifest presence | Snapshot ID | Validation status | Tracked references | Superseded by retained artifact | Recommendation | Classification | Reason |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `data/derived/kb_freeze/0057` | `3,410,858,417` bytes | birth unavailable; mtime `2026-07-30 21:26:29.368922300 +0700`; manifest timestamps ICD `2026-07-30T14:26:03.271870+00:00`, RxNorm `2026-07-30T14:29:56.076222+00:00` | ICD and RxNorm manifests present | ICD `icd10-vi-tt06-2026-kb-v1-candidate-52c4de3abfd25dc4`; RxNorm `rxnorm-prescribable-2026-07-06-kb-v1-candidate-d7181e39bd4053e7` | `CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED` | none for exact path | yes, `0057_candidate_v2` | delete | `FAILED_OR_INCOMPLETE_BUILD` | Earlier generated output with RxNorm crosswalk/review counts and hashes differing from Audit 0057: 59 review-required crosswalk rows versus the documented 46. No approved review decisions; generated from retained local source manifests and builder path. |
| `data/derived/kb_freeze/0057_candidate_v2` | `3,410,849,497` bytes | birth unavailable; mtime `2026-07-30 21:48:41.198128700 +0700`; manifest timestamps ICD `2026-07-30T14:45:00.659879+00:00`, RxNorm `2026-07-30T14:48:41.195524+00:00` | ICD and RxNorm manifests present | ICD `icd10-vi-tt06-2026-kb-v1-candidate-52c4de3abfd25dc4`; RxNorm `rxnorm-prescribable-2026-07-06-kb-v1-candidate-d7181e39bd4053e7` | `CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED` | exact path in `docs/kb/icd10/SNAPSHOT.md` and `docs/kb/rxnorm/SNAPSHOT.md`; snapshot IDs also in `docs/architecture/ACTIVE_RUNTIME_MANIFEST.md` | no | retain | `KEEP_CURRENT_CANDIDATE` | This is the candidate path named by Audit 0057 and the KB snapshot docs. It remains review-required and not active. |
| `data/derived/kb_freeze/0057_final` | `3,410,849,476` bytes | birth unavailable; mtime `2026-07-30 21:38:45.777564200 +0700`; manifest timestamps ICD `2026-07-30T14:34:46.730194+00:00`, RxNorm `2026-07-30T14:38:45.775589+00:00` | ICD and RxNorm manifests present | ICD `icd10-vi-tt06-2026-kb-v1-candidate-52c4de3abfd25dc4`; RxNorm `rxnorm-prescribable-2026-07-06-kb-v1-candidate-d7181e39bd4053e7` | `CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED` | none for exact path | yes, `0057_candidate_v2` | delete | `SUPERSEDED_REPRODUCIBLE_BUILD` | Complete generated candidate-style output, but no retained doc/source/config/test path references it. Audit 0057 and snapshot docs designate `0057_candidate_v2`; no approved human review decisions exist only here. |

No directory was classified as `AMBIGUOUS_DO_NOT_DELETE`.

## 7. Deleted Paths

The following exact paths were deleted, with path, size, classification and reason
printed before each deletion:

| Deleted path | Size | Classification |
| --- | ---: | --- |
| `data/derived/kb_freeze/0057` | `3,410,858,417` bytes (`3.177GiB`) | `FAILED_OR_INCOMPLETE_BUILD` |
| `data/derived/kb_freeze/0057_final` | `3,410,849,476` bytes (`3.177GiB`) | `SUPERSEDED_REPRODUCIBLE_BUILD` |

No broad deletion glob, `find -delete`, `git clean`, Docker prune, reset, restore,
stash, checkout, staging, commit or push was used.

## 8. Retained Paths

Retained KB/storage paths:

```text
data/external/icd10_vi/
data/external/rxnorm/prescribable-2026-07-06/
data/external/rxnorm/full-2026-07-06/
data/derived/icd10_vi/tt06-2026/
indices/icd10_vi/tt06-2026/
indices/rxnorm/prescribable-2026-07-06/
indices/rxnorm/full-2026-07-06/
data/derived/kb_freeze/0057_candidate_v2/
data/manifests/
schemas/kb/
docs/kb/
docs/audits/0057-kb-audit-repair-and-freeze.md
src/mednorm_vi/kb/
tests/unit/test_kb_freeze_0057.py
checkpoint/
docs/MedNorm-VI_Architecture.pdf
```

Ambiguous paths not deleted: none. Ambiguous footprint: `0` bytes.

## 9. Space Reclaimed

| Metric | Before | After | Reclaimed |
| --- | ---: | ---: | ---: |
| Full local project | `23,078,714,912` bytes (`21.494GiB`, `du -sh`: `22G`) | `16,257,007,019` bytes (`15.141GiB`, `du -sh`: `16G`) | `6,821,707,893` bytes (`6.354GiB`) |
| `data/derived/kb_freeze` | `10,232,557,390` bytes (`9.530GiB`, `du -sh`: `9.6G`) | `3,410,849,497` bytes (`3.177GiB`, `du -sh`: `3.2G`) | `6,821,707,893` bytes (`6.354GiB`) |
| Retained candidate-v2 footprint | n/a | `3,410,849,497` bytes (`3.177GiB`) | n/a |

Filesystem free space went from `19G` available to `26G` available on the repo
filesystem. This is safe for Milestone 3B review/relation-closure work if it
continues to use staged outputs and no out-of-scope model/Docker/download work.

## 10. Retained Candidate-v2 Verification

`data/derived/kb_freeze/0057_candidate_v2/` still exists.

ICD expected artifacts exist and are nonzero where expected:

| Artifact | Manifest rows | Manifest hash comparison to Audit 0057 | File size |
| --- | ---: | --- | ---: |
| `icd10_concepts_v1.csv` | 15,308 | match | `5,543,383` bytes |
| `icd10_hierarchy_edges_v1.csv` | 12,968 | match | `3,015,755` bytes |
| `icd10_aliases_v1.csv` | 0 | match | `206` bytes |
| `icd10_provenance_v1.csv` | 15,308 | match | `4,414,521` bytes |
| `icd10_review_queue_v1.csv` | 17,123 | match | `4,445,257` bytes |
| `kb_snapshot_manifest_v1.json` | n/a | snapshot ID match | `3,559` bytes |

RxNorm expected artifacts exist and are nonzero:

| Artifact | Manifest rows | Manifest hash comparison to Audit 0057 | File size |
| --- | ---: | --- | ---: |
| `rxnorm_concepts_v1.csv` | 82,429 | match | `34,798,850` bytes |
| `rxnorm_synonyms_v1.csv` | 245,401 | match | `85,718,537` bytes |
| `rxnorm_relations_v1.csv` | 2,563,978 | match | `692,695,627` bytes |
| `rxnorm_provenance_v1.csv` | 2,809,379 | match | `762,571,928` bytes |
| `inn_rxnorm_crosswalk_v1.csv` | 46 | match | `16,540` bytes |
| `rxnorm_review_queue_v1.csv` | 6,031,588 | match | `1,817,618,160` bytes |
| `kb_snapshot_manifest_v1.json` | n/a | snapshot ID match | `4,151` bytes |

Bounded CSV-header/data-line sampling succeeded for all retained ICD and RxNorm
CSV artifacts. No retained manifest points to the deleted `0057/` or
`0057_final/` paths.

## 11. Active Runtime-index Hash Verification

The active runtime indices still exist and match Audit 0057:

| Runtime index | Expected SHA-256 | Observed |
| --- | --- | --- |
| `indices/icd10_vi/tt06-2026/index.json` | `94a1378b0f355afc6da23bb0da4d1e015dbbd4f51a1f2c0508f06e7b1727a768` | match |
| `indices/rxnorm/prescribable-2026-07-06/index.json` | `4e02ecbfc61e586e041a424177fa682f048618e8e0431e8204f6d2afc2692aec` | match |
| `indices/rxnorm/full-2026-07-06/index.json` | `9ec49f060683d5af9078c6df58fa95792cc3ca841866ad8de1c743684084c80d` | match |

Runtime configuration was not changed.

## 12. Post-cleanup Disk State

Required post-cleanup commands:

| Command | Result |
| --- | --- |
| `df -h .` | 100G size, 75G used, 26G available, 75% use |
| `du -sh .` | `16G` |
| `du -sh data/derived/kb_freeze` | `3.2G` |
| `du -sh data/derived/kb_freeze/* 2>/dev/null` | `3.2G data/derived/kb_freeze/0057_candidate_v2` |
| `du -sh data` | `6.6G` |
| `du -sh indices` | `660M` |

Exact post-cleanup byte measurements:

| Path | Bytes | GiB/MiB |
| --- | ---: | ---: |
| `.` | `16,257,007,019` | `15.141GiB` |
| `data` | `7,005,296,486` | `6.525GiB` |
| `data/derived` | `3,465,877,222` | `3.228GiB` |
| `data/derived/kb_freeze` | `3,410,849,497` | `3.177GiB` |
| `indices` | `691,731,363` | `659.687MiB` |

## 13. Lightweight Validation

Commands run:

```text
git status --short --untracked-files=all
git diff --check
```

Result:

- `git status` still shows only the pre-existing Milestone-3A uncommitted tracked
  and untracked changes; ignored generated deletions are not Git changes.
- `git diff --check` passed with no output.
- No full pytest suite was run.
- No pytest was selected for candidate-v2 locatability because the available
  `test_kb_freeze_0057.py` exercises builders on temporary fixtures rather than
  validating the real ignored `0057_candidate_v2` path. Manual manifest/path
  validation above covers this storage-cleanup milestone without rebuilding KBs.

## 14. Safety Confirmation

Not performed:

- notebook execution or Jupyter startup;
- model or tokenizer load;
- `torch.load`;
- CUDA initialization;
- training, fine-tuning or inference;
- `internal_test` content access;
- Docker build/run/prune/storage modification;
- external data download or internet access;
- ICD or RxNorm rebuild;
- human adjudication or annotation;
- Git staging, commit, push, reset, clean, checkout, restore or stash.

## 15. Changed-file Inventory

Pre-existing Milestone-3A uncommitted tracked modifications:

```text
M docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
M docs/kb/icd10/SNAPSHOT.md
M docs/kb/rxnorm/SNAPSHOT.md
M src/mednorm_vi/linking/rxnorm.py
M src/mednorm_vi/linking/structured_medication.py
M tests/unit/test_l5_l7_deterministic_stack.py
```

Pre-existing Milestone-3A untracked files:

```text
docs/audits/0057-kb-audit-repair-and-freeze.md
docs/kb/KB_FREEZE_0057.md
schemas/kb/candidate-representation-schema-v1.json
schemas/kb/governed-synonym-schema-v1.json
schemas/kb/icd10-kb-schema-v1.json
schemas/kb/inn-rxnorm-crosswalk-schema-v1.json
schemas/kb/kb-provenance-schema-v1.json
schemas/kb/kb-snapshot-manifest-schema-v1.json
schemas/kb/rxnorm-kb-schema-v1.json
src/mednorm_vi/kb/freeze.py
tests/unit/test_kb_freeze_0057.py
```

New file from this cleanup milestone:

```text
docs/audits/0057a-kb-storage-cleanup.md
```

Ignored generated data deletions:

```text
data/derived/kb_freeze/0057/
data/derived/kb_freeze/0057_final/
```

Nothing was staged, committed or pushed.

## 16. Verdict

`KB_STORAGE_CLEAN`

The KB-freeze storage area now retains only the documented current 3A candidate
snapshot, `0057_candidate_v2`, plus the protected raw sources, active runtime
indices, governance files, schemas, source code, tests, checkpoint and
architecture PDF. No ambiguous KB-freeze directory remains.
