# Audit 0014 — RxNorm Full 2026 Intake, Extraction, Comparison, and Integration

- **Date:** 2026-07-22
- **Author:** Claude (AI agent), for human review
- **Change type:** Resource intake + integration milestone. **No commit performed. No push.**
- **Spec reference:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (§10 RxNorm Super Linker,
  §16 offline/fail-fast, §19 repo contracts, frozen-KB provenance). Also read:
  `README.md`, `docs/repository-layout.md`, Audits 0001–0013, all RxNorm loaders /
  index builders / linker / doctor, `.gitignore`.

The user placed the official RxNorm Full Monthly Release ZIP at the repository root.
This audit verified it, moved it to the canonical ignored namespace, extracted and
promoted it, inspected it, compared it against the existing Prescribable snapshot,
built the Full index, and integrated both snapshots into the manifests, doctor, and
pipeline config. No model was trained, no organizer inference or `output.zip` was
produced, no network was accessed, and no UMLS credentials were stored.

## 1. Initial Git state

```text
pwd    : /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch : main
HEAD   : b0fe7629aa08caec54d922f95b720975a5688a46  chore: audit and clean repository resources
status : clean (working tree); root ZIP present as an allowed untracked file
tracked files : 429
```

Audit 0013 is confirmed **committed** at HEAD `b0fe762` (`docs/audits/0013-*.md` and
`docs/repository-layout.md` are tracked). No unrelated uncommitted changes existed.

## 2. Root archive discovery

Exactly one candidate at the repository root, no root extraction directory:

```text
./RxNorm_full_07062026.zip   259,313,098 bytes   untracked (ignored by *.zip)
```

## 3. Checksums

```text
expected published MD5 : 33acdc0176af35808f91b3fc74ff2bb4
locally calculated MD5 : 33acdc0176af35808f91b3fc74ff2bb4   ✅ MATCH
archive SHA-256        : 53523ee9f1fcd7ee426698edf566aedebe548a6ec8cc372c41271fc5b28e784c
```

## 4. Archive safety validation

PK zip signature; 44 members; 1,828,225,005 uncompressed bytes; `testzip` OK; and
**zero** issues across: absolute paths, `../` traversal, encrypted members, symlinks,
backslash names, duplicate members, zero-byte RRF. (`archive_inspection.json`.)

## 5. Canonical archive migration

Moved after all checks passed; destination had no conflicting archive; checksums
re-verified at the destination (MD5/SHA-256 unchanged); root ZIP confirmed gone. The
ZIP is **preserved permanently** as provenance (not deleted after extraction).

```text
data/external/rxnorm/full-2026-07-06/archive/RxNorm_full_07062026.zip
```

## 6–9. Fresh extraction, staging validation, promotion, before/after hashes

Extracted to `…/full-2026-07-06/.staging-extraction/` (staging was absent
beforehand). Staging validation: **COMPLETE_AND_VALID** — 44/44 archive members,
no size mismatches, no partials/symlinks/zero-byte RRF, member-to-file
correspondence exact. Staging tree hash `f77a33c352817f36…` was recorded, then the
tree was promoted to `raw/` and the hash **recomputed identical** (`f77a33c3…`,
44 files, 1,828,225,005 bytes). Staging removed; no root extraction remains.

Git-ignore verified: ZIP (`*.zip`), raw RRF (`data/**`), index (`indices/**`), and
reports (`reports/**`) are all ignored; `git ls-files` under the Full data root and
Full index root are both empty.

## 10–11. Final canonical layout & RRF inventory

```text
data/external/rxnorm/full-2026-07-06/
  archive/RxNorm_full_07062026.zip           (259,313,098 B, MD5 verified)
  raw/
    Readme_Full_07062026.txt
    rrf/            <- canonical Full RRF root (discover_rrf selects it: 4 core files incl RXNSTY)
    prescribe/rrf/  <- bundled Prescribable subset (byte-identical sizes to the standalone snapshot)
    scripts/{mysql,oracle}/
```

Full `rrf/` inventory (bytes / md5 / sha256 in the manifest):

| File | Bytes |
| --- | ---: |
| RXNCONSO.RRF | 131,620,308 |
| RXNREL.RRF | 527,774,099 |
| RXNSAT.RRF | 556,998,743 |
| RXNSTY.RRF | 20,362,133 |
| RXNSAB.RRF | 10,192 |
| RXNDOC.RRF | 219,293 |
| RXNCUI.RRF | 1,745,274 |
| RXNATOMARCHIVE.RRF | 81,488,158 |
| RXNCUICHANGES.RRF | 14,136 |

Discovery required **no code change**: `discover_rrf` ranks candidate roots by RRF
companion count, so it selects the Full `rrf/` (conso+rel+sat+sty) over
`prescribe/rrf/` (conso+rel+sat).

## 12. Full snapshot statistics (streaming, aggregate only)

```text
snapshot_id : rxnorm-full-local-f77a33c352817f36
tree/payload hash : f77a33c352817f366bd62227d3ff47d75199968a4d46d8404288f0d774648141
atoms                : 1,202,603      concepts            : 412,701
attributes           : 7,687,120      relations           : 7,423,180
semantic-type rows   : 488,937        distinct TUI        : 57
suppressed atoms     : 394,623        distinct SAB        : 13   distinct TTY : 45
historical/remap     : RXNATOMARCHIVE, RXNCUICHANGES, RXNCUI, RXNSAB, RXNDOC present
```

## 13. Full vs Prescribable comparison

Full is a **strict superset** (no SAB/TTY exists only in Prescribable).
(`full_vs_prescribable.json`.)

| Metric | Full | Prescribable | Δ |
| --- | ---: | ---: | ---: |
| atoms | 1,202,603 | 245,401 | +957,202 |
| concepts | 412,701 | 82,429 | +330,272 |
| attributes | 7,687,120 | 3,340,763 | +4,346,357 |
| relations | 7,423,180 | 2,563,978 | +4,859,202 |
| semantic-type rows | 488,937 | 0 | +488,937 |
| distinct TUI | 57 | 0 | +57 |
| suppressed atoms | 394,623 | 0 | +394,623 |
| distinct SAB | 13 | 3 | +10 |
| distinct TTY | 45 | 26 | +19 |
| ingredients (IN) | 37,506 | 5,820 | +31,686 |
| clinical drugs (SCD) | 40,027 | 12,055 | +27,972 |
| branded drugs (SBD) | 24,060 | 8,083 | +15,977 |
| brand names (BN) | 29,834 | 4,139 | +25,695 |
| dose forms (DF/DFG) | 983 | 164 | +819 |

Full-only SABs: `ATC, CVX, DRUGBANK, GS, MMSL, MMX, NDDF, SNOMEDCT_US, USP, VANDF`.

**Full-only capabilities relevant to MedNorm-VI:** semantic-type filtering (RXNSTY),
external-vocabulary aliases (broader surface-form recall), brand/generic resolution,
historical/remapped RxCUIs (RXNATOMARCHIVE/RXNCUICHANGES/RXNCUI), and legacy/obsolete
medication mentions (suppressed atoms).

**Documented trade-offs (not a claim that Full is superior):** higher recall but more
candidate noise (394k suppressed/obsolete atoms → needs current-prescribable
preference or suppress filtering); larger index (~578 MB vs ~103 MB); higher memory
(build peaked ~6.7 GB); slower startup/runtime; UMLS-licensed (redistribution
restricted) vs the no-login Prescribable download. The 9B model-parameter budget is
unaffected by KB data size.

## 14–15. Semantic types & historical/remap support

RXNSTY present → 488,937 semantic-type rows across 57 TUIs (Prescribable had none).
Historical/remap files (RXNATOMARCHIVE, RXNCUICHANGES, RXNCUI) are available for
future retired/changed-RxCUI handling.

## 16. Manifest & governance

Created `data/manifests/rxnorm-full-2026-07-06.yaml` (validates **OK, 0 warnings**).
Governance: `license.status: REDISTRIBUTION_RESTRICTED` (a *usable* status meaning
usable for internal/local use, redistribution restricted); `review_status: reviewed`.
Minimum necessary governance fact recorded: **UMLS license confirmed by the user;
licensed local/internal project use; redistribution restricted; credentials not
stored.** No email, UTS username/password, API key, or token is present (enforced by a
test). Full is **not** marked public-domain or freely redistributable. The Prescribable
manifest was left unchanged.

## 17. Full index & provenance

```text
path                  : indices/rxnorm/full-2026-07-06/index.json  (578,218,016 B)
index_id              : rxnorm-index-52b761dc64c99ad7
deterministic_index_hash : 52b761dc64c99ad71c57736f7922557266d504d9e4c8273ecbfc542143c0051d
file sha256           : 9ec49f060683d5af9078c6df58fa95792cc3ca841866ad8de1c743684084c80d
source_snapshot_id    : rxnorm-full-local-f77a33c352817f36
source_hash (payload) : f77a33c352817f366bd62227d3ff47d75199968a4d46d8404288f0d774648141
records / concepts    : 1,202,603 / 412,701   relations_seen 7,423,180  attributes_seen 7,687,120
build                 : peak RSS ~6.68 GB, ~2:00 wall
```

**Determinism:** built twice; both runs produced the identical `deterministic_index_hash`
and a **byte-identical** `index.json`. The Prescribable index
(`indices/rxnorm/prescribable-2026-07-06/index.json`, sha `4e02ecbf…`) was **not**
touched; both indexes coexist. Both load and search successfully (`paracetamol` → hits
in each).

## 18–20. Loader / doctor / config changes

- **doctor** (`src/mednorm_vi/phase1c_foundation/doctor.py`): now discovers Prescribable
  and Full **separately** by namespaced subdirectory (`prescribable-*`, `full-*`), each
  reporting availability, root, missing files, RXNSTY/semantic-type availability, and
  index availability, plus an `active` snapshot field. Full is only recognized inside a
  `full-*` subdirectory (never confused with a Prescribable checkout). Prescribable keeps
  a legacy whole-directory fallback for old layouts/synthetic tests. Fail-fast for missing
  RXNCONSO and missing indexes preserved.
- **loaders/validators**: no change required — `discover_rrf` and `build_rxnorm_index`
  already handle the Full layout; the package internal structure was preserved (not
  flattened).
- **config**: `configs/pipeline/full_v1.yaml` gained a documented `rxnorm_snapshots`
  registry and `rxnorm_selection` policy; the active `rxnorm_index` was **kept at the
  conservative Prescribable default**. New `configs/linking/rxnorm_snapshots_v1.yaml`
  documents four modes — `prescribable_only`, `full_only`, `full_prefer_prescribable`,
  `ablation_full_vs_prescribable` — with `current_prescribable_preference: true`. The
  linker was **not** hard-coded to any policy; switching snapshots is a config-only change.

Recommended configurable policy (until BTC clarifies candidate/membership rules): retrieve
from Full, **boost current-prescribable members**, retain Prescribable-only fallback for
ablation. Default stays Prescribable.

## 21–22. Tests, static checks, deterministic checks

```text
env PYTHONPATH=src python3 -m pytest -q          -> 487 passed  (479 + 8 new)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 213 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
validate-resource-manifest (all 7)               -> OK (full: 0 warnings)
phase1c doctor                                    -> Prescribable available [index]; Full available (UMLS) [RXNSTY] [index]
Full index built twice                            -> byte-identical (deterministic)
both indexes load + search                        -> OK
```

New `tests/unit/test_rxnorm_full_intake.py` (8 tests, synthetic fixtures only):
manifest usable/restricted, archive checksum fields, no-credentials pattern scan,
Full-has-sem-types / Prescribable-does-not, doctor two-snapshot coexistence + index
detection, Full-absent case, and selection-config completeness/conservatism.

## 23. Ignored / tracked verification

`git ls-files` under `data/external/rxnorm/full-2026-07-06` and
`indices/rxnorm/full-2026-07-06` are both **empty**. No ZIP, RRF, index, staging, or
report is tracked. Tracked changes are only code/config/manifest/test/docs.

## 24. Protected files

Architecture PDF unchanged; Prescribable raw (16 files) and index (`4e02ecbf…`)
unchanged; canonical Full ZIP preserved; no `CLAUDE.md`/`AGENTS.md`/tracked `.claude/`;
no `output.zip`; no organizer prediction; no training.

## 25. Remaining limitations

- The Full index build (single-process, in-memory) peaks ~6.7 GB RAM; a streaming/sharded
  builder would be needed for much larger releases or low-memory Colab tiers.
- `current_prescribable_preference` is declared in config but the linker does not yet
  consume a prescribable-membership signal (Full index has no per-record "prescribable"
  flag yet); wiring this boost is a follow-up once BTC clarifies candidate policy.
- Organizer-exact RxNorm release/version compatibility remains to be confirmed after the
  BTC task update. Full is UMLS-licensed (redistribution restricted).

## 26. Exact tracked changed files

```text
 M configs/pipeline/full_v1.yaml
 M src/mednorm_vi/phase1c_foundation/doctor.py
?? configs/linking/rxnorm_snapshots_v1.yaml
?? data/manifests/rxnorm-full-2026-07-06.yaml
?? tests/unit/test_rxnorm_full_intake.py
```
(Ignored additions: `data/external/rxnorm/full-2026-07-06/**`,
`indices/rxnorm/full-2026-07-06/index.json`, `reports/rxnorm_full_intake/**`.)

## 27. Final Git status

Branch `main`, HEAD `b0fe762` (unchanged). 429 tracked files (2 modified, 3 new to add,
0 deleted). Working tree otherwise clean.

## 28–29. Readiness

Audit 0014 is **ready for human review** and **safe to commit** after review. No model
was trained; no claim is made that BTC requires Full RxNorm, that Full is always more
accurate than Prescribable, or that the resource is freely redistributable. The agent did
not commit or push.

**Recommended next milestone:** after BTC publishes the upgraded task/data, confirm the
organizer-exact RxNorm version and candidate/membership policy, then decide the active
snapshot and wire the `current_prescribable_preference` boost into the linker; optionally
add a streaming index builder if memory limits require it.
