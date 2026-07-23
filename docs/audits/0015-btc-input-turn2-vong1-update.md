# Audit 0015 — BTC Updated Input (turn2 / vong1) Intake, Diff, and Activation

- **Date:** 2026-07-22
- **Author:** Claude (AI agent), for human review
- **Change type:** Organizer-input update + activation milestone. **No commit. No push.**
- **Spec reference:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (§4 offset preservation,
  §14 Data Engine, input-only organizer data policy). Also read `README.md`,
  `docs/repository-layout.md`, Audits 0001–0014, Audit 0009, organizer configs,
  `corpus_analysis`, `round2` tooling, `.gitignore`, and current Git state.

The user placed a revised organizer input archive at the repository root. This audit
verified it, compared it against the currently active input, found it **materially
different (CONTENT_CHANGED)**, and safely activated it while preserving the old dataset
as a rollback snapshot. No model was trained, no submission/`output.zip` was produced,
no network was used, no labels/ground truth were inferred, and the architecture PDF was
not modified.

## 1. Initial branch and HEAD

```text
branch : main
HEAD   : b0fe7629aa08caec54d922f95b720975a5688a46  chore: audit and clean repository resources
```

## 2. Pre-existing uncommitted Audit 0014 changes (PRESERVED, untouched)

Audit 0014 was intentionally uncommitted. Classified **PRE_EXISTING_AUDIT_0014** and
preserved exactly:

```text
 M configs/pipeline/full_v1.yaml
 M docs/audits/README.md                     (0014 index line)
 M docs/repository-layout.md                 (0014 RxNorm Full row)
 M src/mednorm_vi/phase1c_foundation/doctor.py
?? configs/linking/rxnorm_snapshots_v1.yaml
?? data/manifests/rxnorm-full-2026-07-06.yaml
?? docs/audits/0014-rxnorm-full-intake-and-integration.md
?? tests/unit/test_rxnorm_full_intake.py
```
(Plus ignored 0014 artifacts under `data/external/rxnorm/full-2026-07-06/**`,
`indices/rxnorm/full-2026-07-06/**`, `reports/rxnorm_full_intake/**`.) The RxNorm Full
integration remains intact (doctor still reports Full available [RXNSTY] [index];
Prescribable index unchanged `4e02ecbf…`).

## 3. Root new-ZIP discovery

```text
./input_turn2_vong1.zip   134,462 bytes   untracked (ignored by *.zip)
```
Neutral resource ID adopted: `btc-input-turn2-vong1` (organizer's exact "turn2 / vong1"
naming unverified; "vong1"≈"vòng 1"/Round 1, not asserted as BTC Round 2).

## 4. Old active-input discovery

Active input: `data/organizer_test/input/` — 100 flat `.txt` files, consumed by
`corpus_analysis` (default `--input-dir data/organizer_test/input`), the single canonical
consumer. An old archive `data/organizer_test/input.zip` (85,874 B, uncompressed 170,948
= the active dataset) was also present.

## 5. Archive safety validation (new ZIP)

`official_checksum: unavailable` (BTC published none) → relied on local SHA-256 + integrity.

```text
MD5    : 59a776041d004d625e1a9ad22d49f817
SHA-256: 989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545
PK zip signature ✓   101 members (100 .txt under input/ + 1 dir)   testzip OK
safety : 0 absolute / 0 traversal / 0 encrypted / 0 symlink / 0 backslash / 0 dup / 0 zero-byte
```
Extraction to ignored staging `data/organizer_test/.staging/input-turn2-vong1/` →
**COMPLETE_AND_VALID** (input root inside the ZIP is `input/`; UTF-8, no BOM, LF).

## 6–8. Snapshot identities & hashes

```text
OLD active : 100 files, 170,948 B, tree 56244aa3e772dd67bcdd80a5e02b50bb4c1b12818edd1b69adce6582377af23e
NEW staging: 100 files, 265,147 B, tree d7ace1838a3433ba40ca3a9df5a8fc3a86be5a2a7ca989c8361e2175dd58620d
new archive: md5 59a7760…  sha256 989d824…
old archive: md5 4148360…  (data/organizer_test/input.zip)
```

## 9. Exact file-level comparison

Same 100 filenames (`1.txt`…`100.txt`); **0 added, 0 removed, 0 renamed**; 100 common.
`round2.classify_input_change` (real data): 0 byte-identical, **100 genuinely changed**.

## 10. Text-level comparison

Per common file: byte-identical 0, decoded-text-identical 0, newline-identical 0,
BOM-only 0, whitespace-only 0, **genuinely changed 100**. Content-identical documents
old∩new: **0 of 100**. So every document's clinical text was rewritten/expanded, not
merely repackaged.

## 11. Corpus-level comparison

| Metric | Old | New | Δ |
| --- | ---: | ---: | ---: |
| documents | 100 | 100 | 0 |
| characters | 132,336 | 203,817 | +71,481 (+54%) |
| tokens (analysis) | 30,470 | 51,111 | +20,641 |
| sentences | 970 | 1,925 | +985 |
| lines | 2,964 | 3,028 | +64 |
| empty / duplicate docs | 0 / 0 | 0 / 0 | — |

## 12. L1 / route comparison

Both corpora are fully L1-valid (offset alignment intact; no failures; corpus-analysis
completed with determinism hashes). The new corpus is more **medication/narrative-heavy**:

```text
med_lines 64→137   med_section_cue_lines 84→187   med_dose_form 18→72   med_strength 48→74
medication_layout docs 30→57   lab_numeric_with_unit 22→50   sentences/doc mean 9.7→19.25
list_items 1913→1478   numbered_items docs 98→77   (less list-dominant, more prose)
max_line_length mean 163→516
```

## 13. Final change classification

**Primary: `CONTENT_CHANGED`.** Secondary: none — the file contract is unchanged (100 flat
`.txt`, identical names), so **not** SCHEMA_OR_LAYOUT_CHANGED; not EXACTLY_IDENTICAL / not
REPACKAGED_ONLY / not INVALID.

## 14. Whether activation occurred

**Yes** — CONTENT_CHANGED warranted replacing the active input, after all validation passed.

## 15–17. Preserved old, new snapshot, active path (all verified byte-for-byte)

```text
old preserved : data/organizer_test/snapshots/btc-input-prior-56244aa3e772dd67/input
                100 files, 170,948 B, tree 56244aa3…  (verified == prior active)
new snapshot  : data/organizer_test/snapshots/btc-input-turn2-vong1/input
                100 files, 265,147 B, tree d7ace183…
ACTIVE path   : data/organizer_test/input   -> tree d7ace183…  (== new snapshot, verified)
canonical archives: data/organizer_test/archives/input_turn2_vong1.zip  (+ old input.zip moved here)
```
The old dataset was **never destroyed** — "replacement" applied to the active dataset only.
Re-running corpus-analysis from the ACTIVE path reproduced the new snapshot's determinism
hash `3a81c4b2…`.

## 18. Root cleanup

Root ZIP relocated to `archives/`; old `input.zip` relocated to `archives/`; staging
removed; no root ZIP or root extraction remains (`cleanup_ledger.json`).

## 19. Manifest / config / code changes (NEW_AUDIT_0015)

- **`configs/organizer/active_input.yaml`** — canonical active-input descriptor (single
  registry reference: active snapshot id/path/hash, archive checksums, prior snapshot,
  classification, corpus counts; **no raw text**). Deliberately **not** under
  `data/manifests/` because that directory is auto-validated by the external-resource
  governance validator, which would misclassify input-only organizer data.
- **`src/mednorm_vi/round2/input_update.py`** (+ `round2/__init__.py` exports) — deterministic
  `fingerprint`, `classify_input_change` (the 5 categories), and `unsafe_zip_members` safety
  helper. Extends the existing Round-2 comparison tooling; no parallel schema.
- **`tests/unit/test_btc_input_update.py`** — 10 synthetic-fixture tests.
- Docs: `docs/audits/README.md` (0015 index line) and `docs/repository-layout.md`
  (organizer snapshots/archives rows) — append-only, non-overlapping with the 0014 lines.
- **No** change to the active command path (`data/organizer_test/input/` is stable), the
  architecture, thresholds, or model settings.

## 20. Git-ignore verification

`git ls-files data/organizer_test` → empty. New ZIP, old ZIP, active input, both snapshots,
and reports are all ignored (`data/**`, `reports/**`, `*.zip`). No organizer TXT, archive,
snapshot, corpus report, or clinical text is tracked.

## 21. Tests and static checks

```text
env PYTHONPATH=src python3 -m pytest -q          -> 497 passed  (487 + 10 new)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 214 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
corpus-analysis (old / new / active)              -> valid; active reproduces new hash 3a81c4b2…
phase1c doctor                                    -> unchanged 0014 output; 7 manifests; RxNorm Full+Prescribable intact
validate-resource-manifest (all 7)               -> OK (unchanged)
classify_input_change (real snapshots)           -> CONTENT_CHANGED, 100 modified, 0 added/removed
```

## 22. Remaining limitations

- `official_checksum: unavailable` — provenance rests on local SHA-256 + integrity checks.
- The organizer's exact "turn2 / vong1" semantics are unverified; a neutral ID is used.
- The active-input descriptor is a config, not a runtime-enforced registry; commands still
  use the stable path directly.

## 23. Implications for later training (not acted on)

The corpus grew ~54% and shifted toward medication/narrative-heavy text with more
sentences and longer lines. This suggests that, **once BTC confirms the task**, L1/L2
calibration and medication-grammar coverage should be re-examined and any local
dev/eval folds rebuilt from the new active input. **No** conclusion that models must be
retrained is drawn yet — labels/ground truth are not present in input-only data, and no
model exists to retrain. No thresholds were tuned.

## 24. Exact NEW_AUDIT_0015 tracked files

```text
 M docs/audits/README.md            (0015 index line only)
 M docs/repository-layout.md        (organizer snapshots/archives rows only)
 M src/mednorm_vi/round2/__init__.py
?? configs/organizer/active_input.yaml
?? docs/audits/0015-btc-input-turn2-vong1-update.md
?? src/mednorm_vi/round2/input_update.py
?? tests/unit/test_btc_input_update.py
```
Ignored 0015 additions: `data/organizer_test/{input,snapshots,archives}/**`,
`reports/btc_input_update/**`.

## 25. PRE_EXISTING_AUDIT_0014 tracked files

See §2 (unchanged by this audit).

## 26. Final Git status

Branch `main`, HEAD `b0fe762` (unchanged). Tracked working-tree changes are the union of
PRE_EXISTING_AUDIT_0014 (§2) and NEW_AUDIT_0015 (§24). No commit performed.

## 27. Readiness

Audit 0015 is **ready for human review** and safe to commit after review. No labels/ground
truth changes are claimed; no retraining is asserted.

## 28. Separability of 0014 and 0015 for later commits

**Logically separable.** All code/config/manifest/test/ignored-data changes belong cleanly
to one audit or the other. The only shared files are `docs/audits/README.md` and
`docs/repository-layout.md`, where each audit added **distinct, non-overlapping append-only
lines** — separable per-hunk (`git add -p`). Recommended: commit Audit 0014 first (its files
+ its README/layout hunks), then Audit 0015 (its files + its README/layout hunks).

**Recommended next action before any training:** commit 0014 then 0015 after human review;
once BTC confirms the updated task/labels, rebuild local dev/eval folds from the new active
input and re-examine L1/L2 + medication-grammar coverage before touching model settings.
