# Audit 0058B: Competition v3 Container Refresh and Governed-Bridge Observability

Milestone: 3B.2 - Container refresh and run-level governed-bridge observability
Date: 2026-07-31
Verdict: `COMPETITION_V3_RUNTIME_REPRODUCIBLE`

## 1. Initial Git State

Read-only precondition checks, run before any modification:

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `6c12153 feat: activate competition KB v3 runtime indices` |
| Working tree | clean |
| Staged files | none |
| `git rev-list --left-right --count origin/main...HEAD` | `0 0` |
| Audit 0058A committed | **yes** — 9 files in `6c12153`, including `docs/audits/0058a-*.md` and `tests/unit/test_competition_v3_activation_0058a.py` |
| Running containers before work | none |
| RAM available / disk free | 7.9 GiB / 25 G |

The protected architecture PDF was verified unmodified
(`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`) and read in full
earlier in this session. It was neither modified nor moved.

A note on the brief's reading list: `src/mednorm_vi/pipeline/`, `src/mednorm_vi/runtime/`
and `src/mednorm_vi/runner/` **do not exist** in this repository. The canonical runner
is `src/mednorm_vi/inference/pipeline.py`, as Audits 0051 and 0055 established. That is
where the propagation change was made.

## 2. The Bridge-Observability Gap

Audit 0058A §7 recorded it precisely: the governed `RETRIEVAL_ONLY` bridges work and are
load-bearing — `paracetamol` is absent from the index's exact and accent-folded
postings, so with the bridge disabled the linker returns nothing for it — but the
`inn_bridge:*` note stopped at the linker report.

The note was created in `linking/rxnorm.py` and travelled on `LinkerResult.warnings`.
The runner assembled `PipelineResult.warnings` from exactly two sources:

```python
warnings=tuple(phase1b.warnings) + tuple(resolved.warnings),
```

`links` was computed on the line above and consumed by L6, L7, L8 and the offered-set
capture — but its warnings were never folded in. So a run could use a governed
crosswalk bridge with nothing above L5 recording that it had, which is the kind of
silent behaviour the rest of this repository refuses to ship.

## 3. Exact Implementation

Three edits, all additive, no new channel and no warning redesign.

**`src/mednorm_vi/linking/rxnorm.py`** — the module that writes the note now also owns
how it is found:

```python
INN_BRIDGE_NOTE_PREFIX = "inn_bridge:"

def governed_bridge_notes(warnings: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({w for w in warnings if w.startswith(INN_BRIDGE_NOTE_PREFIX)}))
```

Deduplication belongs here rather than at the call site. One document can mention
`paracetamol` several times and every mention emits the same note; without the `set`
the run-level record would vary with *mention count* rather than with *which bridges
were used*. Sorting makes repeated runs byte-identical.

**`src/mednorm_vi/inference/pipeline.py`** — the one-line gap, closed:

```python
warnings=(
    tuple(phase1b.warnings)
    + tuple(resolved.warnings)
    + governed_bridge_notes(note for link in links for note in link.warnings)
),
```

It reads `links`, which L8 has already consumed, and adds nothing to `predictions`.

**`src/mednorm_vi/inference/manifest.py`** — the run-level aggregate. The existing
`linking` counter gained `governed_inn_bridge_expansions`, summing the *distinct*
bridges used per document.

Filtering to the `inn_bridge:` prefix is deliberate. The linker's warnings also carry
per-mention suppression codes (`suppressed:DROP_STRENGTH_CONFLICT:…`) and traversal
diagnostics (`ingredient_absent_from_snapshot`, `traversal_truncated_at:…`). Folding
all of them into the document channel would be the "unrelated warning redesign" the
brief forbids, and would bury the governance signal in noise.

### Governance properties held

| Requirement | How |
| --- | --- |
| No bridge RxCUI exposed | The note carries **names**: `inn_bridge:<surface>-><rxnorm_name>:<status>`. Asserted digit-free. |
| Candidate ordering/scores unchanged | Only `warnings` is touched; §4. |
| RETRIEVAL_ONLY policy unaltered | `INN_BRIDGE_STATUS` and `bridged_ingredient_names` untouched. |
| No clinical-approval claim | Every note ends `RETRIEVAL_ONLY_NOT_CLINICALLY_APPROVED`. |
| No nondeterministic duplication | Dedup + sort. |
| L5/L8/L9 boundaries preserved | L5 emits, the runner reads, L8 and L9 are untouched. |

**No clinical text leaves the document.** A bridge fires only when the normalized
ingredient surface is a key of the 12-entry `INN_TO_RXNORM_NAME` table, so the surface
in any note is provably one of twelve source-code constants — not free text lifted from
the record. The run manifest is stricter still: it stores a **count**, never the note,
and `write_run_manifest` refuses to write a manifest reporting clinical text.

## 4. Proof That Candidate Semantics Did Not Change

Established by measurement, not assertion. The canonical runner was captured over four
tracked synthetic fixtures **immediately before** the propagation edit, and again after,
over the same inputs:

```text
BASELINE serialized-output sha256: cac6e87cc4f9ddcf8d6289e8c296f6fba8384df54120d80307fe6b92e515b086
POST     serialized-output sha256: cac6e87cc4f9ddcf8d6289e8c296f6fba8384df54120d80307fe6b92e515b086
CANDIDATE OUTPUT BYTE-IDENTICAL: True
```

Per document, before -> after:

```text
doc 1  entities 6   warnings 0 -> 1   bridge notes 1
doc 2  entities 15  warnings 3 -> 4   bridge notes 1   (two paracetamol mentions, one note)
doc 3  entities 15  warnings 1 -> 1   bridge notes 0
doc 4  entities 0   warnings 3 -> 3   bridge notes 0
```

Only the warning count moved, and only where a governed bridge actually fired.

That pre/post comparison cannot be re-run from inside the suite once the change is in
the tree, so the suite carries two forward guards instead, and the difference between
them is stated in their own docstrings rather than blurred:

- `test_warnings_are_not_an_input_to_candidate_selection` is a **structural proof**: it
  strips every warning from the linker result and re-decodes. If L8 read the warning
  channel at all, directly or through L6/L7, the decoded candidates and per-candidate
  decisions would move. They do not.
- `test_serialized_candidate_output_matches_its_regression_pin` is a **regression pin**
  on the serialized digest, explicitly documented as a guard rather than as the original
  proof.

## 5. New Image

| Field | Value |
| --- | --- |
| Tag | `mednorm-vi:competition-v3-prebaseline` |
| Image ID | `sha256:381ad95e23f5837bb6ab1c1aa1717918c7a69710969317fc13a1ae56a44f2137` |
| Docker inspect size | 428,899,331 bytes |
| `docker images` table size | 1.94 GB |
| Runtime identity | `USER mednorm`, `WORKDIR /app`, entrypoint `python -m mednorm_vi.inference.cli` |
| Base | unchanged digest pin `python:3.14.5-slim-bookworm@sha256:ec58d916…32d9d6` |

Exactly one build. **The Dockerfile was not modified** — `git diff Dockerfile` is empty
— so the mount contract is unchanged and nothing was redesigned. The prior images
(`framework-frozen`, `2d-c-verified`, `framework-closed`) were **not** deleted or
overwritten, and no Docker data was pruned.

`indices/` remains excluded from the build context by `.dockerignore`, so the indices
are mounted, never baked; only `configs/` needed refreshing, which is precisely why the
rebuild was required.

## 6. Baked Active Config Paths

Read from inside the new image:

```text
icd_index   : indices/candidate/icd10_vi/competition-v3/index.json
rxnorm_index: indices/candidate/rxnorm/competition-v3/index.json
```

This is the defect Audit 0058A §9 identified, now closed: `framework-frozen` still bakes
the legacy paths, and this image bakes the activated ones.

## 7. Offline Readiness and Deterministic Smoke

One bounded run: `--rm --network=none --memory=4g --memory-swap=5g --cpus=2
--pids-limit=256`, thread caps set to 1, mounting `indices/` read-only and two tracked
synthetic fixtures as `/input`. No organizer, public-test or `internal_test` data. No
ZIP was produced — the canonical `run_document` + `to_submission_json` +
`_gate_final_output` path was used, which is what `run_input_dir` does before packaging.

```text
1. baked config          competition-v3 ICD + RxNorm;  readiness deterministic READY, index_errors []
2. ICD records           15,308     snapshot competition-kb-v3-candidate-4d206538e7d24c32
3. RxNorm records        211,949    snapshot competition-kb-v3-candidate-4d206538e7d24c32
4. L9 final gate         PASSED (both runs)
5. paracetamol           -> ['161']
6. bridge evidence       inn_bridge:paracetamol->acetaminophen:RETRIEVAL_ONLY_NOT_CLINICALLY_APPROVED
7. closure-only leaked   NONE
8. deterministic         True   (identical serialized bytes across two runs)
9. rollback indices      both present through the same mount
   network               blocked, OSError 101
ALL CONTAINER SMOKE CHECKS PASSED
```

No container remained running afterwards (`docker ps` empty; `--rm` throughout).

Legacy indices re-hashed on the host after all container work, unchanged:

```text
94a1378b0f355afc6da23bb0da4d1e015dbbd4f51a1f2c0508f06e7b1727a768  indices/icd10_vi/tt06-2026/index.json
4e02ecbfc61e586e041a424177fa682f048618e8e0431e8204f6d2afc2692aec  indices/rxnorm/prescribable-2026-07-06/index.json
9ec49f060683d5af9078c6df58fa95792cc3ca841866ad8de1c743684084c80d  indices/rxnorm/full-2026-07-06/index.json
```

## 8. Model Loading

**No model or tokenizer was loaded.** No `torch.load`, no CUDA initialization, no E3
forward pass. The deterministic smoke covered every required check, so the brief's
allowance for one narrow specialist smoke was not needed and not used.

## 9. Targeted, Full-suite, Static and Type Results

Targeted: `pytest tests/unit/test_governed_bridge_observability_0058b.py -q` ->
`12 passed`, covering all six required proofs:

| # | Proof | Test |
| --- | --- | --- |
| 1 | `paracetamol` bridge activity appears in run-level warnings | `test_paracetamol_bridge_activity_reaches_run_level_warnings` |
| 2 | Note states RETRIEVAL_ONLY, not clinically approved | `test_the_note_states_retrieval_only_and_denies_clinical_approval` |
| 3 | No bridge RxCUI emitted directly | `test_a_bridge_note_carries_names_not_rxcuis`, `test_no_bridge_rxcui_is_emitted_directly` |
| 4 | A mention needing no bridge creates no warning | `test_a_mention_needing_no_bridge_creates_no_bridge_warning` |
| 5 | Repeated runs produce identical warnings | `test_repeated_runs_produce_identical_warnings`, `test_a_document_repeating_one_drug_records_one_bridge_note` |
| 6 | Candidate outputs unchanged | `test_warnings_are_not_an_input_to_candidate_selection`, `test_serialized_candidate_output_matches_its_regression_pin` |

One full-suite pass, run after the implementation and image verification:

```text
1904 passed, 1 skipped in 485.96s (0:08:05)
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

The single skip is the pre-existing `pyarrow` absence, unchanged since Audit 0057.

`tests/unit/test_governed_bridge_observability_0058b.py` was formatted **after** that
run rather than during it — the previous milestone reformatted a file mid-run and had to
qualify the result, which is avoidable. The reformat changed line wrapping only, and the
file was re-run afterwards: `12 passed`.

Static and type gates:

| Command | Result |
| --- | --- |
| `ruff check .` | `All checks passed!` |
| `ruff format --check tests/unit/test_governed_bridge_observability_0058b.py` | `1 file already formatted` |
| `mypy src/mednorm_vi` | `Success: no issues found in 288 source files` |
| `compileall -q src tests` | passed |
| `git diff --check` | passed |

`inference/pipeline.py` and `linking/rxnorm.py` were `ruff format`-clean at HEAD and
remain so. `inference/manifest.py` was **already** not clean at HEAD, so it was left in
its existing style — the same treatment Audits 0058 and 0058A gave `decoder.py`,
`rxnorm_graph.py`, `test_e4_retirement.py` and `test_rxnorm_full_intake.py`.
Repository-wide formatting is out of scope and would bury the behavioural diff.

## 10. Resource Usage

| Metric | Observation |
| --- | --- |
| RAM available before work | 7.9 GiB |
| Lowest observed available RAM | 5.6 GiB |
| Stop threshold (3 GiB) | never approached |
| Disk free before / after | 25 G / 25 G |
| Container limits | `--memory=4g --memory-swap=5g --cpus=2 --pids-limit=256` |
| Concurrent heavy workloads | none; the build and the smoke ran sequentially |

## 11. Exact Changed-file Inventory

Tracked modifications:

```text
M src/mednorm_vi/inference/manifest.py
M src/mednorm_vi/inference/pipeline.py
M src/mednorm_vi/linking/rxnorm.py
```

New tracked-intended files:

```text
docs/audits/0058b-competition-v3-container-and-bridge-observability.md
tests/unit/test_governed_bridge_observability_0058b.py
```

`Dockerfile`, `configs/`, both KB snapshot docs and every previous audit are unchanged.
No KB artifact was rebuilt: the v3 snapshot and both activated indices are exactly the
ones Audit 0058A verified.

## 12. Remaining Blockers Before the Pre-baseline Review

1. **Candidate accuracy remains unmeasurable.** The governed corpus carries zero ICD-10
   and zero RxNorm gold codes, so Recall@K and candidate Jaccard are still
   `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`. Nothing in this milestone claims a score
   improvement — the change is observability plus a container refresh.
2. **Competition top-k thresholds (0.98 / 0.95 / 0.80) are still unvalidated**, for the
   same reason.
3. **Advisory ICD hierarchy and `name_quality` are still not read by the ICD ranker.**
   The data is emitted and the weights exist; wiring them remains the cheapest available
   improvement.
4. **`mednorm-vi:framework-frozen` still bakes the pre-activation config.** It was
   deliberately retained as the rollback image. Whichever image is treated as final must
   be named explicitly, or a stale one could be run by mistake.

None blocks the pre-baseline repository review.

## 13. Verdict

`COMPETITION_V3_RUNTIME_REPRODUCIBLE`

The offline image now bakes the activated competition-v3 configuration and was verified
end to end under `--network=none`: readiness resolves the v3 paths, both snapshots load
at their expected sizes, the deterministic runtime reaches and passes L9, `paracetamol`
resolves to RxCUI 161 through governed retrieval, closure-only concepts do not leak,
output is byte-identical across two runs, and the rollback indices remain available and
unmodified.

The observability gap Audit 0058A recorded is closed at both levels — the document
warning channel carries the deterministic note, the run manifest carries a count — and
the candidate output is provably byte-identical to what it was before the change.

**Clinical-approval disclaimer.** Nothing here is clinically approved. The propagated
note exists precisely to say so on every use: it ends
`RETRIEVAL_ONLY_NOT_CLINICALLY_APPROVED`, carries ingredient names rather than RxCUIs,
and records a retrieval aid — never an adjudicated mapping.

## 14. Next Recommendation

1. Review and commit this milestone. Nothing was staged, committed or pushed.
2. Proceed to the pre-baseline repository review, naming
   `mednorm-vi:competition-v3-prebaseline` as the candidate runtime image.
3. Build a small code-bearing gold set. It remains the blocker on every quantitative
   claim, and until it exists the top-k thresholds cannot be validated.
4. Wire `name_quality` and the advisory hierarchy weight into ICD ranking.
5. Do not begin model download, training, fine-tuning, calibration or organizer
   inference from this audit alone.
