# Audit 0016 — Organizer Data Layout, BTC Contract Review, and Repository Data Hygiene

- **Date:** 2026-07-23
- **Author:** Claude (AI agent), for human review
- **Change type:** Post-0015 correction + hygiene milestone. **No commit. No push.**
- **Spec reference:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (§4 offsets, §14 Data
  Engine, output contract §1/§16/§19). Also read `README.md`,
  `docs/repository-layout.md`, Audits 0001–0015, `configs/organizer/active_input.yaml`,
  the output schemas/serializer/validator, `round2`/`corpus_analysis` tooling, and
  `.gitignore`.

This audit supersedes the **layout** described in Audit 0015. Audits 0014 (`9fc16fa`,
RxNorm Full) and 0015 (`f587ff5`, BTC input activation) are **immutable committed
historical milestones** and were not altered here beyond removing a mistaken
correction addendum that had been appended to the 0015 document (0015 is now byte-
identical to its `f587ff5` form). All post-0015 corrective work is recorded here.

## 1. Real Git state

```text
branch : main
HEAD   : f587ff589f1be1086b85c396c7350bf8f236725b  activate updated BTC organizer input
parents: f587ff5 -> 9fc16fa (integrate RxNorm Full 2026 snapshot) -> b0fe762
0014 committed as 9fc16fa ; 0015 committed as f587ff5 (both verified present)
```

Initial working tree at the start of this milestone contained the uncommitted
post-0015 edits (organizer layout correction, contract review, lab/position tests);
Audit 0015's document also carried a mistaken 2026-07-23 addendum, since removed.

## 2. Canonical organizer-data layout (final)

The **only** persistent extraction is the active input; historical releases are
retained **archive-only** and are recoverable on demand. No persistent `snapshots/`
directory and no hash-named path exist.

```text
data/organizer_test/                         (all git-ignored)
  input/                                      # ACTIVE = turn2_vong1, tree d7ace183… (100 files, 265,147 B)
    1.txt … 100.txt
  archives/
    initial_release/input.zip                 # original release
    turn2_vong1/input_turn2_vong1.zip         # updated release
```

## 3. Archive verification (reconstruction proof, before any deletion)

Both ZIPs were verified (PK signature, `testzip` OK, exactly 100 `.txt`, no
absolute/traversal/encrypted/symlink/duplicate members, UTF-8), extracted into an
ignored `.temporary_verification/`, and their payloads re-fingerprinted:

| Release | Archive (md5) | Recomputed payload tree | Reconstructs |
| --- | --- | --- | --- |
| initial_release | `input.zip` `4148360722506d51532d0f6837581912` | `56244aa3e772dd67…377af23e` | prior extracted snapshot (exact) |
| turn2_vong1 | `input_turn2_vong1.zip` `59a776041d004d625e1a9ad22d49f817` | `d7ace1838a3433ba…dd58620d` | active input **and** duplicate new snapshot (exact) |

(`reports/btc_input_update/organizer_archive_verification.json`.) Only after these
proofs were the redundant extractions deleted.

## 4. Deletions (files and bytes)

```text
data/organizer_test/snapshots/btc-input-prior-56244aa3e772dd67/   100 files   170,948 B  (== initial_release ZIP)
data/organizer_test/snapshots/btc-input-turn2-vong1/              100 files   265,147 B  (== active input / turn2_vong1 ZIP)
data/organizer_test/.temporary_verification/                     200 files   436,095 B  (transient)
TOTAL DELETED                                                    400 files   872,190 B
```

Ledger: `reports/data_hygiene/organizer_data_cleanup_ledger.json`. Recovery: unzip the
corresponding archive under `data/organizer_test/archives/<release>/`. **No hash-based
organizer directory, no `snapshots/`, and no `.staging`/`.temporary_verification`
path remains** (verified on the filesystem).

## 5. Data naming policy (applied)

Operational data paths use lowercase `snake_case`; release directories use documented
identifiers (`initial_release`, `turn2_vong1`, `full-2026-07-06`, `tt06-2026-official`);
official dataset names retained (`phoner_covid19`, `vimedner`, `vimq`, `vietmed_ner`).
**Content hashes are never used as directory names** — they live only in manifests,
configs (`configs/organizer/active_input.yaml`), and reports. No ambiguous names
(`old`/`new`/`final`/`copy`/`temp`) exist under `data/`.

## 6. `data/` hygiene review

Every `data/` root was inventoried and classified
(`reports/data_hygiene/data_path_inventory.json`): all are `REQUIRED_*`
(active data, archive provenance, canonical sources, derived resources, tracked
metadata). No `__pycache__`, `.pyc`, `.DS_Store`, `.part`, `.tmp`, or stale lock file
was present under `data/` or the repo. ZIP catalog
(`reports/data_hygiene/zip_archive_catalog.json`): 3 ZIPs, all in canonical locations,
none tracked, none at repo root. Rename plan and final layout:
`reports/data_hygiene/data_path_rename_plan.json`, `.../final_data_layout.json`.

## 7. BTC updated task-contract review

`docs/task_contracts/btc_turn2_vong1_contract_review.md`. The updated task's **output
contract is materially unchanged** — one JSON per TXT; five Vietnamese entity labels;
assertion scope (symptom/diagnosis/medication); `CHẨN_ĐOÁN→ICD-10`, `THUỐC→RxNorm`
candidates — all already implemented and enforced (`schemas/constants.py`,
`schemas/prediction.py`).

- **Patient information** → `UNSUPPORTED_BY_OUTPUT_SCHEMA`: no patient-info type/field;
  usable internally as context only. Did **not** invent `THÔNG_TIN_BỆNH_NHÂN` or add
  JSON fields.
- **Inter-concept relations** → `UNSUPPORTED_BY_OUTPUT_SCHEMA`: no `relations` field;
  output remains entity-local assertions. No relation schema invented.
- **Position** → half-open `[start, end)` **end-exclusive**, confirmed by the prior
  organizer example (`original_text[start:end] == text`, `POSITION_IS_END_EXCLUSIVE =
  True`) and **preserved** (no internal spans changed). The BTC "0…n-1" wording alone
  does not re-distinguish inclusive vs exclusive; residual ambiguity documented
  (resolved to end-exclusive by repository-local evidence). If BTC later confirms
  inclusive output, the change is localized: `output_end = internal_end_exclusive - 1`.
- **Laboratory result boundaries** → supports value-only, value+unit, qualitative, and
  comparison; units not required. `bình thường` and `tăng nhẹ` were added to the lab
  qualitative lexicon (coverage, **not** threshold tuning).

## 8. Exact code / config / test / doc changes (this milestone = Audit 0016)

```text
 M configs/laboratory/parser_v1.yaml        (+2 qualitative terms: bình thường, tăng nhẹ)
 M configs/organizer/active_input.yaml       (rewritten to active_input/previous_input;
                                              only canonical release-archive paths; no hash/snapshot paths;
                                              previous_input.extracted_path: null)
 M docs/repository-layout.md                 (organizer rows -> input/ + archives/<release>/; archive-only history)
 M src/mednorm_vi/round2/input_update.py     (+ temporary_input_extraction, fingerprint_archive)
 M src/mednorm_vi/round2/__init__.py         (export new helpers)
 M tests/unit/test_btc_input_update.py       (descriptor test -> new schema; + temp-extraction/no-hash-path tests)
?? docs/task_contracts/btc_turn2_vong1_contract_review.md
?? tests/unit/test_organizer_output_contract.py   (position boundary + per-type contract + lab boundary)
?? docs/audits/0016-organizer-data-layout-contract-and-hygiene.md
 M docs/audits/README.md                     (0016 index entry)
```

No change to the active command path (`data/organizer_test/input/` is stable), to the
architecture, or to model thresholds. Audit 0015's document was restored to its
`f587ff5` state (mistaken addendum removed) and is otherwise untouched; Audit 0014's
files are untouched.

## 9. Tests and static checks

```text
env PYTHONPATH=src python3 -m pytest -q          -> 521 passed
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 214 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
git ls-files data/organizer_test                  -> (empty)
phase1c doctor / all 7 manifests / 3 indexes      -> unchanged & valid; RxNorm Full+Prescribable intact
active-input L1/corpus analysis                    -> deterministic
```

New tests cover: one-character / start-at-zero / end-at-last-character / Vietnamese
Unicode / line-break / empty / invalid spans; per-type organizer field contract;
patient-information and relation absence; lab value-only/unit/qualitative/comparison;
safe temporary archive extraction; archive fingerprint; active-input descriptor
consistency; and a no-hash/snapshot-path guard on the descriptor.

## 10. Raw-data and protected-item verification

`git ls-files data/organizer_test` → empty. All organizer TXT/ZIP, RxNorm RRF/index,
ICD PDFs/derived, public NER data, and generated reports remain ignored. Architecture
PDF unchanged; RxNorm Full/Prescribable and ICD resources preserved; no credentials or
assistant-specific files tracked.

## 11. Final repository state

HEAD `f587ff5` (unchanged). Working tree carries only the Audit 0016 changes listed in
§8. No tracked documentation of the **current** layout claims persistent organizer
snapshots (Audit 0015 remains a dated historical record of the activation milestone,
superseded here). This milestone is a single, self-contained changeset.

## 12. Readiness

Audit 0016 is **ready for human review and safe to commit** as one follow-up commit on
top of `f587ff5`. It is disjoint from Audit 0014's files and only extends/corrects
Audit 0015's, so the three milestones remain cleanly separable. No model was trained,
no organizer prediction or `output.zip` was generated, and no network was accessed.

**Recommended next action before any training:** commit this changeset after review;
once BTC formally confirms the updated task/labels, rebuild local dev/eval folds from
the new active input and re-examine L1/L2 + medication-grammar coverage — no thresholds
or models were tuned here.
