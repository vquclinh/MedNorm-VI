# Audit 0058A: Competition KB v3 Runtime Index Activation

Milestone: 3B.1 - Activate the competition KB v3 runtime indices
Date: 2026-07-31
Verdict: `COMPETITION_V3_INDICES_ACTIVE_WITH_CONTAINER_NOTE`

## 1. Initial Git State

Read-only precondition checks, run before any modification:

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `12ccf38 feat: build competition KB v3 and governed candidate policy` |
| Working tree | clean |
| Staged files | none |
| `git rev-list --left-right --count origin/main...HEAD` | `0 0` |
| Milestone 3B / Audit 0058 committed | **yes** — 31 files in `12ccf38`, including `docs/audits/0058-*.md`, `docs/kb/KB_COMPETITION_0058.md` and all of `src/mednorm_vi/kb/competition/` |

The protected architecture PDF was verified unmodified by hash
(`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`) and was read in
full earlier in this session; it is the primary source of truth for this milestone and
was neither modified nor moved.

## 2. Verified v3 Artifacts and Hashes

All three required paths exist. Nothing was rebuilt — every artifact verified on first
check, which is the precondition the milestone brief set for not rebuilding.

Snapshot `data/derived/kb_freeze/0058_competition_candidate_v3/`:

| Field | Value |
| --- | --- |
| Snapshot ID | `competition-kb-v3-candidate-4d206538e7d24c32` |
| Validation status | `VALID_COMPETITION_CANDIDATE`, 0 violations |
| Builder / policy | `competition-kb-builder-v1` / `competition-kb-policy-v1` |
| Predecessor (read-only input) | `data/derived/kb_freeze/0057_candidate_v2` |
| Validation report digest | matches the manifest |

All 8 snapshot artifacts re-verified on row count, `artifact_sha256` **and**
`canonical_content_sha256`:

```text
OK  competition_candidate_index_v1.csv           rows=227257
OK  competition_provenance_v1.csv                rows=15
OK  icd10_advisory_hierarchy_v1.csv              rows=12968
OK  icd10_competition_concepts_v1.csv            rows=15308
OK  inn_rxnorm_canonical_bridges_v1.csv          rows=12
OK  rxnorm_competition_concepts_v1.csv           rows=211949
OK  rxnorm_competition_relations_v1.csv          rows=960086
OK  rxnorm_governed_synonyms_v1.csv              rows=245401
```

Runtime index hashes, checked against the values Audit 0058 recorded:

| Index | `deterministic_index_hash` | Matches 0058 |
| --- | --- | --- |
| `indices/candidate/icd10_vi/competition-v3/index.json` | `b050a2d5ead90c8410101d257b08b347c129f5a8faa713d7e5f2757c7931a975` | **yes** |
| `indices/candidate/rxnorm/competition-v3/index.json` | `d3c45f349bad1532135b70214c48e54c270ede6ae6988e1444a9949887d92ac3` | **yes** |

Both carry `source_snapshot_id = competition-kb-v3-candidate-4d206538e7d24c32`. The
RxNorm index declares `searchable_concepts: 82429`, `closure_only_concepts: 129520` and
`closure_only_in_postings: false`; the ICD index declares `suspect_names_indexed: true`.

## 3. Old Versus New Active Paths

| Ontology | Before (Audit 0014 default) | After (this audit) |
| --- | --- | --- |
| ICD-10 | `indices/icd10_vi/tt06-2026/index.json` | `indices/candidate/icd10_vi/competition-v3/index.json` |
| RxNorm | `indices/rxnorm/prescribable-2026-07-06/index.json` | `indices/candidate/rxnorm/competition-v3/index.json` |

Legacy indices were not deleted, moved or modified. Re-hashed after activation:

| Path | SHA-256 | Unchanged |
| --- | --- | --- |
| `indices/icd10_vi/tt06-2026/index.json` | `94a1378b0f355afc6da23bb0da4d1e015dbbd4f51a1f2c0508f06e7b1727a768` | yes |
| `indices/rxnorm/prescribable-2026-07-06/index.json` | `4e02ecbfc61e586e041a424177fa682f048618e8e0431e8204f6d2afc2692aec` | yes |
| `indices/rxnorm/full-2026-07-06/index.json` | `9ec49f060683d5af9078c6df58fa95792cc3ca841866ad8de1c743684084c80d` | yes |

## 4. Exact Configuration Changes

Only the two keys the runtime actually reads decide the active index: `PipelineConfig.load`
consumes `icd_index` and `rxnorm_index` and nothing else from the snapshot registry
blocks. Both were repointed.

`configs/pipeline/full_v1.yaml`:

```text
- icd_index: indices/icd10_vi/tt06-2026/index.json
+ icd_index: indices/candidate/icd10_vi/competition-v3/index.json
- rxnorm_index: indices/rxnorm/prescribable-2026-07-06/index.json
+ rxnorm_index: indices/candidate/rxnorm/competition-v3/index.json
```

The registry blocks were updated too, because leaving them behind would have made the
file a false statement about its own behaviour — the exact defect Audits 0056a and 0055
repeatedly corrected elsewhere in this repository. A `competition_v3` entry was added to
`rxnorm_snapshots`, `rxnorm_selection.active` became `competition_v3`,
`competition_v3_only` was added to `modes`, and a new `rxnorm_selection.rollback` block
names both previous paths.

`configs/linking/rxnorm_snapshots_v1.yaml` gained the matching `competition_v3` entry
(with `derived_from: [prescribable, full]`, the searchable/closure counts, the index
hash and governance `COMPETITION_RUNTIME_ELIGIBLE_NOT_CLINICALLY_APPROVED`), plus
`active: competition_v3`, `default_mode: competition_v3_only` and a `rollback` block.

`current_prescribable_preference` stays `true`, and that is not an oversight.
Audit 0014 set it to express a *candidacy* policy, not a path, and the policy is
unchanged: v3 is derived from the Prescribable subset, only prescribable-searchable
concepts may be emitted, and the Full package contributes graph reach only.

No runner, linker, decoder or validator code was changed by this milestone. Activation
is config-only.

## 5. Rollback

Declared in configuration rather than described in prose, so it cannot drift:

```yaml
rxnorm_selection:
  rollback:
    icd_index: indices/icd10_vi/tt06-2026/index.json
    rxnorm_index: indices/rxnorm/prescribable-2026-07-06/index.json
```

To roll back, set `icd_index` and `rxnorm_index` to those two values. Both files exist,
both are byte-identical to their pre-activation hashes, and two tests assert exactly
that — one checks the files exist and hash correctly, another checks the active paths
are not equal to the rollback paths, so a "rollback" that silently pointed at the
active index would fail.

## 6. Targeted Tests

New file `tests/unit/test_competition_v3_activation_0058a.py`, 15 tests, covering all
14 checks the brief required. The index fixtures are the **real activated artifacts**,
module-scoped so each 144 MB index loads once — a miniature cannot answer "did
activation change what the runtime emits?", which is the only question this milestone
exists to answer.

| # | Check | Result |
| --- | --- | --- |
| 1 | Active config resolves to competition-v3 ICD and RxNorm paths | pass |
| 2 | ICD index exposes 15,308 searchable records | pass |
| 3 | RxNorm index exposes 211,949 total records | pass |
| 4 | Exactly 82,429 RxNorm records searchable | pass |
| 5 | Exactly 129,520 closure-only | pass |
| 6 | Closure-only records cannot be final candidates | pass (not retrievable; linker drops with `DROP_CLOSURE_ONLY`) |
| 7 | Advisory ICD hierarchy cannot independently offer candidates | pass (`may_offer_candidate=false` on all 12,968 rows) |
| 8 | `RETRIEVAL_ONLY` bridges widen lookup but never emit codes | pass (returns names, no digits) |
| 9 | `paracetamol` reaches RxCUI 161 through governed retrieval | pass |
| 10 | Candidate top-k remains deterministic | pass (3 identical runs) |
| 11 | L8 cannot introduce a candidate outside the offered set | pass |
| 12 | L9 rejects non-final-eligible candidates | pass |
| 13 | Rollback config paths still exist | pass |
| 14 | No legacy index was changed | pass (hash-verified) |

Also updated: `tests/unit/test_rxnorm_full_intake.py`. Two tests encoded Audit 0014's
`active: prescribable` default and would otherwise have failed silently-wrongly. They
now assert the superseding contract — the v3 entry, both source packages still
registered, `current_prescribable_preference` still true, and the rollback block
present and pointing at real files — with the reason recorded in the test docstring.

`pytest tests/unit/test_competition_v3_activation_0058a.py tests/unit/test_rxnorm_full_intake.py -q`
-> `23 passed`.

## 7. Canonical Runtime Smoke

Deterministic mode, **no model loaded**, two tracked synthetic fixtures
(`tests/fixtures/phase1b/synthetic_medication_list.txt` and
`synthetic_medical_document.txt`) copied to a scratch input directory outside the
repository. No organizer data, no `internal_test`, no public-test input.

The first attempt through the CLI reached `package_output_zip` and stopped on the
organizer's 100-file count — L1-L9 had already completed and, importantly,
`_gate_final_output` had already passed, since packaging runs after it. Rather than
fabricate 100 synthetic documents to satisfy a packaging count, the smoke was re-run
through the canonical `run_document` + `to_submission_json` + `_gate_final_output` path,
which is precisely what `run_input_dir` does before packaging. **No ZIP was produced.**

```text
config icd_index     indices/candidate/icd10_vi/competition-v3/index.json
config rxnorm_index  indices/candidate/rxnorm/competition-v3/index.json
loaded icd snapshot     competition-kb-v3-candidate-4d206538e7d24c32  records 15308
loaded rxnorm snapshot  competition-kb-v3-candidate-4d206538e7d24c32  records 211949

L9 FINAL GATE PASSED (both runs)
serialized output byte-identical across two runs:  True
documents 2 | entities 21
candidate-list sizes emitted: {0, 1, 2}
entities whose emitted codes escaped the L5 offered set: 0
closure-only codes in final output: NONE
```

Emitted medication candidates, through the activated indices:

```text
amlodipine besylate 10 mg po daily  -> ["2726068", "1600716"]
paracetamol 500mg khi cần           -> ["161"]
metformin 500 mg ER bid             -> ["1043578", "1243846"]
amoxicillin 250 mg/5 mL suspension  -> ["723"]
aspirin 81 mg                       -> ["315431"]
insulin 10 UI sc                    -> ["1986354", "1992169"]
```

Serialized organizer JSON carries the Vietnamese label (`"type": "THUỐC"`) with
end-exclusive positions, and every list is 1 or 2 codes — the competition top-k policy
is demonstrably in force on real output.

**The governed crosswalk expansion is load-bearing, not incidental.** Measured directly
against the activated index:

```text
'paracetamol' in exact postings        False
'paracetamol' in accent-folded postings False
linker with governed bridge ENABLED    ['161', '1007587', '1042675']
linker with governed bridge DISABLED   NOTHING
```

So the `161` above cannot have come from lexical retrieval; it exists only because the
`RETRIEVAL_ONLY` bridge supplied the name `acetaminophen` as a query. The bridge never
supplied the code.

One reporting gap worth naming: the `inn_bridge:*` traversal note lives on the linker
report and is not propagated to `PipelineResult.warnings`, so a run-level reader cannot
see that expansion fired. The behaviour is correct and provable; only its visibility at
the run level is missing. Recorded as a follow-up, not fixed here, because this
milestone is scoped to activation.

## 8. Full-suite, Static and Type Results

One full-suite pass, run after every change was complete:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

1892 passed, 1 skipped in 393.01s (0:06:33)
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

The single skip is the pre-existing `pyarrow` absence, unchanged since Audit 0057.

`tests/unit/test_competition_v3_activation_0058a.py` was reformatted while that run was
in flight, so the file it executed was not byte-identical to the file on disk. The
reformat changed line wrapping only, and the two affected files were re-run afterwards
to close the gap: `23 passed` (15 activation tests + 8 in `test_rxnorm_full_intake.py`).
Full-suite collection reports 1,893 tests, one more than the 1,892 that ran, because
`test_no_audit_was_deleted` now also sees this audit file.

Static and type gates:

| Command | Result |
| --- | --- |
| `ruff check .` | `All checks passed!` |
| `ruff format --check tests/unit/test_competition_v3_activation_0058a.py` | `1 file already formatted` |
| `mypy src/mednorm_vi` | `Success: no issues found in 288 source files` |
| `compileall -q src tests` | passed |
| `git diff --check` | passed |

`tests/unit/test_rxnorm_full_intake.py` was already not `ruff format`-clean at HEAD, so
it was left in its existing style rather than reformatted — the same treatment Audit
0058 gave `decoder.py`, `rxnorm_graph.py` and `test_e4_retirement.py`. Repository-wide
formatting is out of scope and would bury the behavioural diff.

## 9. Container Verification

Docker was inspected read-only; nothing was pruned, deleted or rebuilt, and no container
was left running (`--rm` throughout, `docker ps` empty afterwards). Both runs used
`--network=none` with explicit memory, CPU and PID limits.

**The `indices/` mount contract needs no change.** The v3 files sit *inside* the
existing `indices/` tree, so the established `-v "$PWD/indices:/app/indices:ro,Z"` mount
already exposes them. Confirmed inside `mednorm-vi:framework-frozen`:

```text
/app/indices/candidate/icd10_vi/competition-v3/index.json   exists=True  bytes=11,560,325
/app/indices/candidate/rxnorm/competition-v3/index.json     exists=True  bytes=144,229,494
/app/indices/icd10_vi/tt06-2026/index.json                  exists=True
/app/indices/rxnorm/prescribable-2026-07-06/index.json      exists=True
```

**But the image bakes the config.** `Dockerfile` line 185 is `COPY configs/ ./configs/`,
and `mednorm-vi:framework-frozen` was built at Audit 0056g — before this activation. Its
baked `full_v1.yaml` still reads:

```text
icd_index: indices/icd10_vi/tt06-2026/index.json
rxnorm_index: indices/rxnorm/prescribable-2026-07-06/index.json
```

**A rebuild is therefore required** before the container runs on v3. This is a stale
artifact, not a contract defect: with the repository `configs/` mounted as a one-off
diagnostic overlay, the same image resolved the v3 paths, reported `deterministic READY`
with no index errors, loaded the v3 ICD index in-container (15,308 records, snapshot ID
`competition-kb-v3-candidate-4d206538e7d24c32`) and confirmed the network was blocked
(`OSError 101`). No Docker contract redesign is needed and none was made.

`specialist` reported `NOT_READY` in that diagnostic because the checkpoint and HF
mounts were deliberately omitted — no model was to be loaded — not because of anything
about the indices.

## 10. Model Loading

**No model or tokenizer was loaded.** No `torch.load`, no CUDA initialization, no E3
forward pass. The canonical smoke ran in `deterministic` mode, which does not construct
E3, and the container checks used `evaluate_readiness` and index loading only. The
milestone's allowance for a single narrowly-scoped model load was not needed and not
used.

## 11. Exact Changed-file Inventory

Tracked modifications:

```text
M configs/linking/rxnorm_snapshots_v1.yaml
M configs/pipeline/full_v1.yaml
M docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
M docs/kb/KB_COMPETITION_0058.md
M docs/kb/icd10/SNAPSHOT.md
M docs/kb/rxnorm/SNAPSHOT.md
M tests/unit/test_rxnorm_full_intake.py
```

New tracked-intended files:

```text
docs/audits/0058a-competition-kb-v3-runtime-index-activation.md
tests/unit/test_competition_v3_activation_0058a.py
```

No source module under `src/mednorm_vi/` was changed. `docs/kb/KB_COMPETITION_0058.md`
is not an audit; its "Runtime Status" section asserted "staged, not active", which this
milestone made false, so it was corrected with a pointer here. Audits 0001-0058 were not
modified.

Generated data remains Git-ignored (`data/**`, `indices/**`), so the v3 snapshot and
both activated indices are absent from the inventory above by policy, not by omission.

## 12. Remaining Blockers Before the Pre-baseline Review

1. **The frozen container image must be rebuilt** to pick up the activated config. The
   mount contract is unchanged and the rebuild is expected to be routine (§9).
2. **`inn_bridge` expansion is invisible at run level** — the note does not reach
   `PipelineResult.warnings`, so a run manifest cannot show that a governed bridge
   fired. Behavioural only; no incorrect output.
3. **Candidate accuracy remains unmeasurable.** The governed corpus carries zero ICD-10
   and zero RxNorm gold codes, so Recall@K and candidate Jaccard are still
   `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`. Activation changed *coverage and policy*,
   and this audit claims no score improvement.
4. **The competition top-k thresholds (0.98 / 0.95 / 0.80) are still unvalidated**, for
   the same reason.
5. **Advisory ICD hierarchy and `name_quality` are still not read by the ICD ranker.**
   The data is emitted and the weights exist; wiring them is the cheapest remaining
   improvement.

None of these blocks the pre-baseline repository review; items 1 and 2 should be
resolved before any image is treated as final.

## 13. Verdict

`COMPETITION_V3_INDICES_ACTIVE_WITH_CONTAINER_NOTE`

The competition KB v3 indices are the active runtime KB. The canonical config opens
them, the canonical runner produces valid L9-passing output through them with
deterministic ordering and no closure leakage, the legacy indices are retained
byte-identical as a declared rollback, and no runner or linker logic was redesigned to
make any of it work. The container note is the single qualifier: the frozen image bakes
its config and therefore needs a rebuild before it runs on v3.

**Clinical-approval disclaimer.** Nothing activated here is clinically approved. All 12
INN crosswalk bridges are `RETRIEVAL_ONLY`, zero are `CLINICALLY_APPROVED`, and the ICD
searchability policy is a competition-scoring judgement explicitly distinguished from a
clinical one. `assert_automated_decision` raises `ClinicalApprovalForbidden` rather than
permit any automated path to claim otherwise.

## 14. Next Recommendation

1. Review and commit this milestone. Nothing was staged, committed or pushed.
2. Rebuild the offline image so its baked config matches the activated one, then re-run
   the established offline readiness and deterministic smokes against it.
3. Propagate the `inn_bridge` traversal note to `PipelineResult.warnings` so the run
   manifest records governed expansion.
4. Proceed to the pre-baseline repository review.
5. Do not begin model download, training, fine-tuning, calibration or organizer
   inference from this audit alone.
