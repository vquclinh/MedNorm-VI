# Audit 0010 — Local Medical Resource Intake

- **Date:** 2026-07-17
- **Author:** Codex (AI agent), for human review
- **Change type:** Consolidated local-resource intake record plus narrow resource
  governance/detection support. No raw medical resource modification; no network;
  no extraction/download/conversion; no training; no index/linker/prediction.
- **Spec reference:** `docs/MedNorm-VI_Architecture.pdf` v1.1, especially L5
  ICD/RxNorm specialist requirements, frozen-KB provenance, offline inference,
  reproducibility, and data/model hygiene constraints.

## 1. Objective and scope

Document the completed local intake of:

- Vietnamese ICD-10 source documents:
  - `06-byt.pdf`
  - `06-byt-kem.pdf`
- RxNorm Current Prescribable Content, release `2026-07-06`.

The scope was limited to inspection, recovery, verification, organization status,
manifesting, ignored local reporting, and readiness checks. It did not redo the
implementation work, mutate raw resources, download the missing ZIP, convert ICD
PDFs, OCR text, build KB indexes, implement linking, train models, run inference,
or create `output.zip`.

This audit intentionally contains metadata only. It does not reproduce ICD PDF
text, RxNorm rows, organizer input text, secrets, or private absolute paths.

## 2. Git state before the intake

At the start of the local-resource intake operation:

```text
pwd: /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch: main
latest commit: d43a72b feat: add public corpus analysis
git status --short: clean
git diff --check: clean
```

No tracked modifications were present before the intake began.

## 3. Why this operation was needed

The previous local-resource intake attempt had been interrupted by server-side
529/5xx failures. The user then manually moved and possibly extracted resources,
so the repository could not assume any step had completed safely. The operation
was needed to:

- discover the actual filesystem state;
- prove no root duplicates or data-loss conflicts remained;
- verify source artifacts and extracted RxNorm content independently;
- register metadata without committing raw resource payloads;
- represent the deleted RxNorm archive honestly rather than fabricating hashes;
- update resource-governance and doctor detection so the repository can
  distinguish source artifacts, normalized snapshots, and legal usability.

## 4. Architecture components supported

These resources support the architecture's L5 specialist layer:

- **ICD-10 Super Linker** for future `DIAGNOSIS -> ICD-10` candidate evidence.
- **RxNorm Super Linker** for future `MEDICATION -> RxCUI` candidate evidence.

They also support the Phase 1C resource-governance foundation from Audit 0008:
tracked manifests, local-only KB inspection, snapshot provenance, and offline
deterministic validation. No L5 linking implementation was added in this intake.

## 5. Initial filesystem state detected

Detected state: **E**.

- The ICD PDFs were already under
  `data/external/icd10_vi/tt06-2026-official/`.
- `RxNorm_full_prescribe_07062026.zip` was not present anywhere in the repository
  tree.
- RxNorm Current Prescribable Content was already extracted under
  `data/external/rxnorm/prescribable-2026-07-06/`.
- The actual RRF root was nested at
  `data/external/rxnorm/prescribable-2026-07-06/raw/rrf/`.
- No `.tmp` extraction directories, unsafe symlinks, broken symlinks, or
  unreadable files were found in the resource trees.

## 6. Root duplicate confirmation

Final root duplicate check found no repository-root copies of:

- `06-byt.pdf`
- `06-byt-kem.pdf`
- `RxNorm_full_prescribe_07062026.zip`

No duplicate deletion or move was needed during this Codex intake pass.

## 7. Final ICD resource locations

The final ICD source-artifact locations are:

- `data/external/icd10_vi/tt06-2026-official/06-byt.pdf`
- `data/external/icd10_vi/tt06-2026-official/06-byt-kem.pdf`

Both are ignored raw external resources; they are not tracked.

## 8. ICD source-document integrity

`06-byt.pdf`:

- bytes: `598417`
- pages: `5`
- MD5: `8093f034eba61cfcc6be033517bc05c3`
- SHA-256:
  `1216d8a2f980c56c5c300a7ab00e8cbb063d0ff0b0cfb404cf195eae6dbc5c92`
- MIME: `application/pdf`
- signature: `%PDF-1.3`
- `pdfinfo`: readable
- `qpdf --check`: OK; no syntax or stream encoding errors found

`06-byt-kem.pdf`:

- bytes: `21988472`
- pages: `1271`
- MD5: `f019955133d62e939b20e9ad4b42524a`
- SHA-256:
  `8639f5eeb77b571363dc841923095895d2498748f9bc6620f50710a6da9159e2`
- MIME: `application/pdf`
- signature: `%PDF-1.3`
- `pdfinfo`: readable
- `qpdf --check`: operation succeeded with warnings; qpdf reported offset-0
  object warnings that it describes as a common error handled correctly by qpdf
  and most applications

The PDFs were preserved byte-for-byte. No OCR, text extraction, CSV conversion,
metadata rewrite, optimization, or normalization was performed.

## 9. ICD status

ICD status is intentionally limited:

- official-source candidate available;
- conversion not started;
- normalized ICD table not prepared;
- organizer-exact compatibility unresolved.

The PDFs are not claimed to be the organizer's hidden ICD snapshot. A later human
or pipeline step must decide whether and how to convert them into a normalized
table and compare that table against organizer expectations.

## 10. Final RxNorm snapshot locations

RxNorm namespace root:

```text
data/external/rxnorm/prescribable-2026-07-06
```

Actual RRF root used by the repository loader:

```text
data/external/rxnorm/prescribable-2026-07-06/raw/rrf
```

Core RRF path:

```text
data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNCONSO.RRF
```

## 11. RxNorm package identity

The local resource is **RxNorm Current Prescribable Content** with release date:

```text
2026-07-06
```

The local package README identifies the archive as
`RxNorm_full_prescribe_07062026.zip` and describes this Current Prescribable
Content release as created from the July 06, 2026 Full Release version of RxNorm.
This prescribable package is not treated as equivalent to the full monthly
RxNorm release.

## 12. Archive-history limitation

The archive-history limitation is explicit:

- ZIP was already deleted before manifest completion.
- Expected NLM-published MD5 is known:
  `767678e3b5b1d6fe358b61c21659f3ef`.
- Locally calculated archive MD5 is unavailable.
- Local archive MD5 verification status is `unknown`.
- Archive SHA-256 is unavailable.
- No archive MD5 or SHA-256 value was fabricated.

Repository-local manifests and reports were searched for prior archive hash
evidence; none was found.

## 13. Core RxNorm files found

Core file inventory:

- `raw/rrf/RXNCONSO.RRF`: found, `30506659` bytes, pipe-delimited with trailing
  pipe
- `raw/rrf/RXNREL.RRF`: found, `196795423` bytes, pipe-delimited with trailing
  pipe
- `raw/rrf/RXNSAT.RRF`: found, `280646563` bytes, pipe-delimited with trailing
  pipe
- `raw/rrf/RXNSTY.RRF`: absent

## 14. RXNSTY.RRF absence

`RXNSTY.RRF` is absent from this local package. The local package inventory lists
`RXNCONSO.RRF`, `RXNREL.RRF`, and `RXNSAT.RRF`, but not `RXNSTY.RRF`. Therefore
the semantic-type count is `0`.

This is recorded as a package characteristic, not as snapshot corruption.

## 15. Extracted file count and size

The extracted RxNorm tree contains:

- extracted files: `16`
- extracted total bytes: `507963140`
- symlinks: `0`

## 16. Extracted-tree SHA-256

The deterministic extracted-tree hash is:

```text
fae9a40088f4b4082bfb7d8798e3e5720ee800c0fa11db44f81345eba2c71d4d
```

It was computed from sorted entries containing:

```text
relative_path<TAB>file_sha256
```

## 17. Content-derived snapshot ID

The RxNorm content-derived snapshot ID is:

```text
rxnorm-local-75a100c0b70b67d0
```

## 18. RxNorm loader and validator results

Command:

```text
env PYTHONPATH=src python3 -m mednorm_vi.phase1c_foundation.cli inspect-rxnorm-snapshot --dir data/external/rxnorm/prescribable-2026-07-06
```

Output:

```text
RxNorm snapshot: rxnorm-local-75a100c0b70b67d0 (data/external/rxnorm/prescribable-2026-07-06)
  atoms                 : 245401
  attributes            : 3340763
  concepts              : 82429
  distinct_sab          : 3
  distinct_tty          : 26
  relations             : 2563978
  semantic_types        : 0
  suppressed_concepts   : 0
  validation            : OK (0 warning(s))
```

## 19. Resource manifest design changes

Resource-governance schema support was extended narrowly:

- `ChecksumRecord` now accepts optional `md5` while continuing to require
  SHA-256.
- `ArchiveRecord` records original archive filename, package type, expected
  published MD5, local MD5 if known, verification status, archive SHA-256 if
  known, archive presence, and whether archive deletion preceded manifest
  completion.
- `ExtractedSnapshotRecord` records local extraction path, deterministic tree
  hash, file count, total bytes, content snapshot ID, validation status, and core
  relative paths.
- The field reference in `schemas/resources/RESOURCE_MANIFEST.md` was updated.

This is a general-purpose schema extension for historical archive provenance and
verified extracted trees. It does not weaken the normal rule that a present
archive requires local MD5 and SHA-256.

## 20. Honest unavailable archive-hash representation

Validation now explicitly permits an unknown absent-archive SHA-256 only when the
manifest records:

- `archive_present: false`
- `archive_deleted_before_manifest_completion: true`

That state produces a warning, not an error. A present archive still requires
local MD5 and archive SHA-256. A known local archive MD5 must match the expected
published MD5 when both are present.

## 21. Nested RRF discovery changes

`mednorm_vi.kb.rxnorm.discover_rrf` was updated to:

- preserve direct-child discovery for existing fixtures and callers;
- recursively search nested directories when no direct `RXNCONSO.RRF` exists;
- choose the directory with `RXNCONSO.RRF` and the most known RRF companions;
- expose the discovered RRF root in `RrfFileSet.root`.

This allows the existing RxNorm loader and CLI to validate the actual local
layout under `raw/rrf/` without moving extracted files.

## 22. Phase 1C doctor changes

The Phase 1C doctor now distinguishes:

- RxNorm Prescribable snapshot: available when `RXNCONSO.RRF` is discovered;
- RxNorm Full 2026: missing / awaiting UMLS approval;
- ICD source PDFs: available when both source PDFs are present;
- ICD normalized table/snapshot: not prepared / missing when no CSV exists.

This avoids forcing every resource state into a single green/missing bucket.

## 23. Resource status distinctions

Resource status after intake:

- RxNorm Current Prescribable Content extracted snapshot: available and
  structurally valid.
- RxNorm Full 2026: unavailable, awaiting UMLS approval.
- ICD official source PDFs: available source artifacts.
- ICD normalized table/snapshot: not prepared.
- Both resource manifests: `REVIEW_REQUIRED`.
- Both resources: not legally/reuse usable until human review changes the
  supported license status.

`REVIEW_REQUIRED` is not a failed intake. It separates file integrity from
license/reuse approval.

## 24. Manifest paths and validation results

Created tracked manifests:

- `data/manifests/rxnorm-prescribable-2026-07-06.yaml`
- `data/manifests/icd10-vi-tt06-2026-official.yaml`

RxNorm manifest validation:

```text
resource manifest: rxnorm-prescribable-2026-07-06 (data/manifests/rxnorm-prescribable-2026-07-06.yaml)
  warning: resource.archive_sha256_unknown: rxnorm-prescribable-2026-07-06: archive absent; SHA-256 unknown because deletion preceded manifest completion
  warning: resource.not_yet_usable: rxnorm-prescribable-2026-07-06: license status REVIEW_REQUIRED — not usable until reviewed
OK (2 warning(s))
```

ICD manifest validation:

```text
resource manifest: icd10-vi-tt06-2026-official (data/manifests/icd10-vi-tt06-2026-official.yaml)
  warning: resource.not_yet_usable: icd10-vi-tt06-2026-official: license status REVIEW_REQUIRED — not usable until reviewed
OK (1 warning(s))
```

## 25. Local deterministic report paths and hashes

Ignored local reports:

- `reports/resource_intake/intake.json`
- `reports/resource_intake/intake.md`

Report hashes:

```text
418f77b69c5179b30cd8d5f5b1abb876db521807b44722a2e32e656ccd4e2a44  reports/resource_intake/intake.json
a0daf854daf4464350b8dfbc237857286caba92a3ddcb8f5eb0b4a8a2a7e95f7  reports/resource_intake/intake.md
```

The reports contain aggregate metadata only. They are ignored by `reports/**`
and are not tracked.

## 26. Git-ignore verification

Representative ignore checks:

```text
.gitignore:24:data/**	data/external/icd10_vi/tt06-2026-official/06-byt.pdf
.gitignore:24:data/**	data/external/icd10_vi/tt06-2026-official/06-byt-kem.pdf
.gitignore:24:data/**	data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNCONSO.RRF
.gitignore:24:data/**	data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNREL.RRF
.gitignore:42:reports/**	reports/resource_intake/intake.json
.gitignore:42:reports/**	reports/resource_intake/intake.md
```

`git check-ignore -v docs/MedNorm-VI_Architecture.pdf` returned no match, as
expected.

## 27. Root cleanup verification

Final root cleanup verification found no root-level copies of:

- `06-byt.pdf`
- `06-byt-kem.pdf`
- `RxNorm_full_prescribe_07062026.zip`

Repository-local search also found no `RxNorm_full_prescribe_07062026.zip`.

## 28. No raw resource or report tracked

`git status --short --ignored` reports the resource/report directories as ignored:

```text
!! data/external/icd10_vi/tt06-2026-official/
!! data/external/rxnorm/prescribable-2026-07-06/
!! reports/resource_intake/
```

Only metadata manifests under `data/manifests/*.yaml` are tracked candidates.

## 29. Architecture PDFs unchanged

The architecture PDFs were not modified, moved, regenerated, optimized, or
rewritten. `git status --short -- '*.pdf'` is empty.

## 30. Assistant-specific files untracked

No assistant-specific file is tracked. `git ls-files CLAUDE.md AGENTS.md .claude`
returned no tracked paths. `.claude/AGENTS.md` remains ignored local guidance.

## 31. Tests and exact outputs

Targeted resource/KB/doctor tests:

```text
env PYTHONPATH=src python3 -m pytest -q tests/unit/test_resource_governance.py tests/unit/test_kb_rxnorm.py tests/unit/test_kb_icd10.py tests/unit/test_phase1c_doctor.py
.............................                                            [100%]
29 passed in 0.51s
```

Full test suite:

```text
env PYTHONPATH=src python3 -m pytest -q
........................................................................ [ 15%]
........................................................................ [ 31%]
........................................................................ [ 46%]
........................................................................ [ 62%]
........................................................................ [ 77%]
........................................................................ [ 93%]
................................                                         [100%]
464 passed in 15.10s
```

`PYTHONPATH=src` is required in the current shell because the package is not
installed in the active environment.

## 32. Static-check outputs

```text
ruff check .
All checks passed!
```

```text
env PYTHONPATH=src python3 -m mypy
Success: no issues found in 161 source files
```

```text
git diff --check
```

No output; clean.

## 33. Current `git status --short`

After creating this audit and updating the audit index, status is:

```text
 M docs/audits/README.md
 M schemas/resources/RESOURCE_MANIFEST.md
 M src/mednorm_vi/kb/rxnorm/discovery.py
 M src/mednorm_vi/phase1c_foundation/doctor.py
 M src/mednorm_vi/resources/__init__.py
 M src/mednorm_vi/resources/manifest.py
 M src/mednorm_vi/resources/models.py
 M src/mednorm_vi/resources/validation.py
 M tests/unit/test_kb_rxnorm.py
 M tests/unit/test_phase1c_doctor.py
 M tests/unit/test_resource_governance.py
?? data/manifests/icd10-vi-tt06-2026-official.yaml
?? data/manifests/rxnorm-prescribable-2026-07-06.yaml
?? docs/audits/0010-local-medical-resource-intake.md
```

## 34. Known limitations

- RxNorm ZIP was deleted before manifest completion.
- Locally calculated ZIP MD5 and ZIP SHA-256 are unavailable.
- Only the published expected MD5 is known.
- Extracted content is independently protected by the extracted-tree hash and
  content-derived snapshot ID.
- RxNorm Full remains unavailable pending UMLS approval.
- RxNorm Prescribable is not equivalent to the full monthly snapshot.
- ICD PDFs have not been converted into a normalized table.
- The ICD source may not exactly match the organizer's hidden snapshot.
- Both resource manifests remain `REVIEW_REQUIRED` for human license/reuse
  review.
- No public Vietnamese medical NER dataset has been acquired yet.
- No KB index or linker has been built.

## 35. Remaining manual actions

- Obtain RxNorm Full after UMLS approval, if needed for future Full-vs-
  Prescribable comparison.
- Convert the ICD PDFs later into a reviewed normalized ICD table/snapshot.
- Download and review public Vietnamese medical NER datasets later.
- Complete human license/reuse and redistribution reviews for both resources.
- Decide whether the prescribable RxNorm package is sufficient for experiments or
  whether the full monthly snapshot is required first.

## 36. Is the resource intake ready for commit?

**Yes, pending human review.** The raw resources are ignored; the tracked changes
are metadata, detection/schema support, tests, and this audit. File integrity and
snapshot validation are complete for the local artifacts that are actually
present. License/reuse approval remains intentionally unresolved and is not a
commit blocker for metadata intake.

## 37. Recommended next milestone

**Phase 1C-B resource review and conversion planning:** complete human
license/reuse review; decide whether to acquire RxNorm Full for comparison; plan
the ICD PDF-to-normalized-table conversion as a separate audited task; keep
linker/index construction out of scope until reviewed normalized resources are
available.

## 38. Explicit non-actions

Confirmed:

- no network access;
- no deleted ZIP download;
- no model training;
- no ICD OCR or conversion;
- no RxNorm/ICD index build;
- no linker implementation;
- no public NER dataset acquisition;
- no Full Pipeline v1 work;
- no prediction generation;
- no `output.zip` generation;
- no automatic commit.
