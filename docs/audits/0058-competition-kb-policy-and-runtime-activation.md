# Audit 0058: Competition KB Policy, Candidate v3 and Runtime Activation

Milestone: 3B - Competition-oriented KB policy, candidate v3, runtime activation
Date: 2026-07-31
Verdict: `COMPETITION_KB_ACTIVE_WITH_NOTES`

## 1. Initial Git and Protected State

Read-only precondition checks, run before any modification:

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `6afb81b feat: build governed KB candidate freeze and clean superseded outputs` |
| `git status --short --untracked-files=all` | clean |
| `git diff --cached --name-status` | empty |
| `git rev-list --left-right --count origin/main...HEAD` | `0 0` |
| Milestone 3A / 3A.1 committed | **yes** — Audits 0057 and 0057A, `schemas/kb/`, `src/mednorm_vi/kb/freeze.py` and `tests/unit/test_kb_freeze_0057.py` are all in `6afb81b` |

The Git precondition in the milestone brief — "if Audit 0057/0057A source, schema, test
and documentation changes are still uncommitted, stop" — was therefore satisfied. Audit
0057A recorded those files as uncommitted at the time it was written; the owner has
since committed them.

Protected artifacts were not modified. `docs/MedNorm-VI_Architecture.pdf` was read in
full (14 pages) and left untouched. The E3 checkpoint was not opened or deserialized.

## 2. Interrupted-run Recovery Assessment

The brief warned that a previous 3B attempt may have been interrupted. It was not.

| Evidence | Finding |
| --- | --- |
| `data/derived/kb_freeze/` contents | only `0057_candidate_v2` |
| Grep for `0058`, `competition_candidate_v3`, `COMPETITION_RUNTIME_ELIGIBLE` | no hits in any tracked file |
| `docs/audits/0058-*` | absent |
| Working tree | clean |

**Milestone 3B had not started.** Nothing was reused, nothing was discarded, and no
partial work needed preserving. The expensive builds were run once each.

Resume safety was nonetheless built into the v3 builder rather than assumed away,
because the RxNorm pass streams two 500 MB+ relation files and an interruption is a
realistic event. `build_state_v1.json` records each artifact's row count and both
digests as soon as it is written; a later invocation re-verifies every recorded
artifact against the bytes on disk and rebuilds only what is missing, altered or
unverifiable. A silently reused *corrupt* artifact would be worse than a rebuild, so
verification is by digest rather than by existence.

Exercised twice, on fixtures and then on the real snapshot:

```text
snapshot_id:        competition-kb-v3-candidate-4d206538e7d24c32
status:             VALID_COMPETITION_CANDIDATE
reused artifacts:   8 of 8
```

The snapshot ID was stable and the expensive RxNorm pass did not re-run.

## 3. Task and Metric-derived Competition Policy

The spec (§1) weights `candidates` 40%, `text` 30%, `assertions` 30%, and scores
candidates by **Jaccard**. Jaccard is symmetric in its punishments: with a single gold
code, one correct answer scores `1/1 = 1.0`, and the correct answer plus one wrong one
scores `1/2 = 0.5`. A speculative extra candidate is not a free lottery ticket — it
costs exactly as much as a miss.

3A froze a single governance state ("has a human reviewed this?") and answered it "no"
for everything uncertain. Correct for clinical deployment, self-defeating for a scored
competition: it blocked the runtime behind thousands of manual decisions the metric
never asks about, and it left the most common Vietnamese medication surface linking to
nothing at all.

3B separates the two questions 3A had fused:

```text
may a human rely on this mapping clinically?      ->  CLINICALLY_APPROVED
may the runtime offer this code to the scorer?    ->  COMPETITION_RUNTIME_ELIGIBLE
```

Roles: `CLINICALLY_APPROVED`, `COMPETITION_RUNTIME_ELIGIBLE`, `RETRIEVAL_ONLY`,
`EXCLUDED`, `UNMAPPED`. Allowed automated decisions are exactly
`{RETRIEVAL_ONLY, EXCLUDED, UNMAPPED}`;
`kb/competition/policy.py::assert_automated_decision` raises
`ClinicalApprovalForbidden` if any code path attempts `CLINICALLY_APPROVED`, and fails
loudly rather than silently downgrading, because a quiet rewrite would hide the policy
bug behind output that looks correct.

## 4. ICD Competition Policy

Searchability follows **structural code validity**, not name quality.

| Metric | Count |
| --- | ---: |
| Concepts | 15,308 |
| Structurally valid | 15,308 |
| Searchable | 15,308 |
| Excluded | 0 |
| Clean names | 11,838 |
| Suspect names (searchable, penalised) | 3,470 |
| Missing names | 0 |
| Advisory hierarchy edges | 12,968 |
| Source-supported hierarchy edges | 0 |
| Edges that may offer a candidate | **0** |

Policy points, as implemented:

1. every structurally valid concept is searchable;
2. original names, normalized names and provenance are preserved;
3. suspect names stay searchable with explicit `name_quality` and `quality_flags`;
4. no repaired canonical name is invented — `source_name` is verbatim;
5. suspect status is a **ranking penalty** (25.0 / 50.0), never an exclusion;
6. exact lexical evidence outweighs hierarchy expansion 1000:1
   (`EXACT_MATCH_WEIGHT` 1000.0 against `ADVISORY_HIERARCHY_WEIGHT` 1.0);
7-8. prefix-derived hierarchy is `advisory_prefix_inference`, `source_authoritative`
   is `false`;
9. an advisory edge may contribute a weak ranking feature;
10. `advisory_hierarchy_may_offer` is hard-wired `False`, so an advisory edge can
    **never independently introduce a final candidate**;
11. no manual review of names or edges is required before competition activation;
12. candidate ordering is deterministic.

An edge whose endpoint is not searchable is dropped rather than carried, so the
hierarchy table cannot describe a graph the runtime does not have.

This is a competition-scoring judgement and **not** a clinical one. No ICD record in
this snapshot claims human adjudication.

## 5. RxNorm Searchable and Closure Policy

| Policy point | Implementation |
| --- | --- |
| Prescribable concepts are searchable final candidates | `membership = searchable`, `runtime_role = COMPETITION_RUNTIME_ELIGIBLE` |
| Full RxNorm supplies missing graph endpoints where needed | closure built from `full-2026-07-06`, minimum required only |
| Closure-only concepts support traversal | present in `records` and `graph` |
| Closure-only concepts are not automatically final candidates | refused in three independent places (§9) |
| RxCUI, TTY, REL, RELA, direction preserved | all five columns retained per relation |
| IN/PIN/MIN/SCD/SBD not collapsed | `tty_values` is a typed collection per concept |
| Ranking priority | exact normalized match → ingredient+strength+dose-form → compatible SCD/SBD → IN/PIN fallback (unchanged from Audit 0054, verified by targeted tests) |
| Prefer SCD/SBD when strength/dose form present | verified: `structured_exact` tier |
| IN/PIN fallback for ingredient-only mentions | verified: `ingredient_only` tier |
| Combination mentions not reduced to one ingredient | verified: MIN retained, constituent INs not substituted |
| Searchability and graph-endpoint membership are separate flags | `searchable` and `graph_endpoint` columns |
| A closure-only node never leaks into final output | §9 |

Retained-relation policy: concept-level relations whose `RELA` is in the competition
structure set (ingredient / precise-ingredient / component / dose-form / form /
tradename / quantified-form, 14 labels). `has_inactive_ingredient` alone is 1.5M rows
and cannot identify a drug product; carrying it would inflate both the snapshot and the
closure for no candidate benefit. Excluded rows are counted, never silently dropped.

## 6. Crosswalk Normalization Correction

Independently verified from the frozen 0057 artifact:

| Measurement | Expected | Observed |
| --- | ---: | ---: |
| Physical rows | 46 | **46** |
| Unique normalized local surfaces | 12 | **12** |
| Unique RxCUIs | 11 | **11** |
| Canonical `(surface, rxcui)` bridges | 12 | **12** |

The 12 surfaces, all confirmed against local snapshot evidence rather than taken on
trust from the brief: `adrenaline`→epinephrine (3992), `amoxycillin`→amoxicillin (723),
`cephalexin`→cefalexin (2231), `co-trimoxazole`→sulfamethoxazole / trimethoprim
(10831), `frusemide`→furosemide (4603), `glibenclamide`→glyburide (4815),
`lignocaine`→lidocaine (6387), `noradrenaline`→norepinephrine (7512),
`paracetamol`→acetaminophen (161), `rifampicin`→rifampin (9384),
`salbutamol`→albuterol (435), `trimethoprim-sulfamethoxazole`→sulfamethoxazole /
trimethoprim (10831).

**Cause of the 46-to-12 inflation.** 3A emitted one row per matching *atom*, so
`adrenaline` produced six rows because RxCUI 3992 carries the strings `EPINEPHRINE`,
`EPINEPHrine`, `Epinephrine` and `epinephrine` across term types `IN`, `SU` and `TMSY`.
Different casing, source atoms and term types for the same RxCUI are corroborating
evidence for one bridge, not six competing claims.

**A term-type collection is not a concept type.** 3A wrote the pipe-joined string
`IN|SU|TMSY` into a field named `rxnorm_tty`, which reads as though RxNorm has a term
type of that name. It does not. `BridgeEvidence.observed_tty` is now a typed `tuple`;
the pipe-joined form survives only as a storage encoding, and a schema comment says so
explicitly.

**Two surfaces may share one RxCUI.** `co-trimoxazole` and
`trimethoprim-sulfamethoxazole` both name RxCUI 10831 and remain **two** bridges,
because they are two lookup keys the runtime must be able to normalize; collapsing them
would lose one. The uniqueness invariant is on the pair, never on either half.

Each canonical bridge retains: local surface, normalized local surface, target RxCUI,
preferred target name, observed term-type collection, supporting atom count, supporting
provenance IDs, source evidence, competing RxCUIs, ambiguity flags, automated decision,
decision reason, confidence band and runtime role.

Deterministic canonical content hash:
`2ef033277496624bdcfa431ada11b0ce3ab304a1b047f9e33e56e62ecd75c7c1`. Reversing the
source row order reproduces it exactly, so grouping does not depend on input order.

## 7. Automated Bridge Decisions

| Decision | Count |
| --- | ---: |
| `RETRIEVAL_ONLY` | **12** |
| `EXCLUDED` | 0 |
| `UNMAPPED` | 0 |
| `CLINICALLY_APPROVED` | **0** |

This matches the expected disposition. No clearly invalid ingredient/product or
single/combination mapping was found: both combination surfaces resolve to a `MIN`
concept, and the ten single-ingredient surfaces resolve to `IN` concepts. `cephalexin`
carries the flag `no_concept_defining_tty_in_evidence` (it matched only an `SY` string)
and remains `RETRIEVAL_ONLY`. No surface maps to more than one RxCUI in the current
inventory; the machinery to preserve and flag competing bridges exists and is tested
against a synthetic case.

**What a `RETRIEVAL_ONLY` bridge may do.** It may widen *lookup*. It may not emit a
code. `bridged_ingredient_names` returns RxNorm-side **names, never RxCUIs**, so the
result cannot be mistaken for a candidate even in principle; those names re-enter
retrieval as queries and whatever concept they reach faces every downstream check.
Measured effect on `paracetamol`, which occurs in **zero** records of the locked
snapshot:

| Mention | 3A | 3B |
| --- | ---: | --- |
| `paracetamol` | **0 candidates** | RxCUI 161 `ACETAMINOPHEN` (`IN`), reached via the `exact` channel |
| `paracetamol 500mg` | 0 candidates | top tier `structured_exact` |
| `salbutamol` | 0 candidates | RxCUI 435 `ALBUTEROL` (`IN`) |

Every use is recorded as
`inn_bridge:<surface>-><name>:RETRIEVAL_ONLY_NOT_CLINICALLY_APPROVED`.

## 8. Endpoint Closure Construction

**Corrected finding.** Audit 0057 reported 2,010,514 prescribable relation rows with
missing endpoints and listed it as an activation blocker. Re-measured against the same
local RRF files, the count is an artifact of the check, not a defect in the data.

`RXNREL.RRF` records relations at two levels. A `CUI`-level row populates
`RXCUI1`/`RXCUI2`; an `AUI`-level row leaves those columns empty **by design** and
identifies its endpoints in `RXAUI1`/`RXAUI2`. The 3A builder read only the RXCUI
columns, so every atom-level row looked like a relation with two missing endpoints —
which is why the count was `missing_both: 2,010,514` with **zero** one-sided misses, a
distribution no genuine endpoint defect would produce.

| Measurement | Before | After |
| --- | ---: | ---: |
| Prescribable relation rows | 2,563,978 | 2,563,978 |
| `CUI`-level rows | 553,464 | 553,464 |
| `CUI`-level rows with a missing endpoint | 0 | 0 |
| `AUI`-level rows | 2,010,514 | 2,010,514 |
| Unresolvable atoms | — | **0** |
| Missing endpoints, 3A CUI-only check | 2,010,514 | — |
| Missing endpoints, after atom resolution | — | **0** |

The prescribable relation set was already endpoint-complete and needed **no** closure.
What Full RxNorm supplies is additional *reach*.

| Metric | Value |
| --- | ---: |
| Concepts before closure (searchable) | 82,429 |
| Closure-only concepts added | 129,520 |
| Concepts after closure | 211,949 |
| Full structure rows scanned | 1,527,146 |
| Retained relations | **960,086** |
| — from prescribable | 350,114 |
| — added from Full | 609,972 |
| Excluded, non-structure `RELA` | 2,213,864 |
| Excluded, no searchable endpoint | 445,006 |
| Duplicate relation rows collapsed | 472,168 |
| **Missing endpoints after closure** | **0** |
| Distinct relation labels | 14 |
| Disk footprint | 345 MB |

Relation-label distribution: `has_ingredient`/`ingredient_of` 189,228 each,
`dose_form_of`/`has_dose_form` 108,241 each, `has_tradename`/`tradename_of` 88,539
each, `consists_of`/`constitutes` 69,002 each,
`has_precise_ingredient`/`precise_ingredient_of` 10,881 each, `form_of`/`has_form`
9,182 each, `has_quantified_form`/`quantified_form_of` 4,970 each.

TTY distribution over searchable concepts is preserved in full (26 term types,
including IN 5,820, PIN 1,924, MIN 962, SCD 12,055, SBD 8,083).

A relation with neither endpoint searchable is excluded, which is what keeps the
closure minimal: only endpoints a retained relation actually requires are added.
Duplicates are collapsed deterministically by `(source, target, rel, rela)`.

## 9. Closure-only Concepts Cannot Leak

Enforced in three independent places, so no single bug re-opens the path:

1. **Retrieval** — closure-only surfaces never enter the index postings, verified on
   the built index: 0 of 10 sampled closure nodes were reachable by searching their own
   name.
2. **Linker** — `linking/rxnorm.py` drops one that traversal reached with
   `DROP_CLOSURE_ONLY`, distinct from `DROP_NOT_IN_SNAPSHOT` because the concept *is*
   in the snapshot. Verified on real data: linking `MESNA` reached closure node
   `334034` and dropped it, retaining 9 searchable candidates.
3. **L9** — `validator/kb_membership.py` refuses one with
   `kb.candidate_not_final_eligible`, **without importing any linker**, because the
   linker is the component whose bug this gate exists to catch.

A snapshot that states no membership is treated as permissive. Silence must not mean
exclusion: the legacy indices draw no such distinction, and reading their silence as
"closure-only" would empty every candidate list.

## 10. Candidate Top-k Policy

| Type | Policy |
| --- | --- |
| Medication | top-1; a second only at `score2/score1 >= 0.98` within the same evidence tier |
| Diagnosis | top-1; a second at `>= 0.95`, or on a specific/unspecified code pair at `>= 0.80` |
| Symptom / test name / test result | none (spec §7.3) |
| Hard ceiling | 2 |

A runner-up from a weaker evidence tier is never a tie, however close its score: the
tiers are ordinal and the scores are not comparable across them.

`apply_competition_topk` returns a **sub-sequence** of the list `_select_candidates`
already produced, so ordering is inherited rather than recomputed and **no code can be
introduced**. The L8/L9 offered-set invariant therefore holds structurally, not by a
downstream check. `decode_entities(..., competition_topk=False)` reproduces the
Audit-0054 behaviour exactly, so the wiring is additive.

Candidate identity remains `ontology + code + snapshot_id`. Thresholds are fixed and
written down rather than searched: no stage produces a calibrated probability
(Audit 0053 §19), so a tuned threshold would be fitted to a score that is not one.

## 11. Legacy, v2 and v3 Comparison

| Area | Legacy active | 0057 candidate v2 | 0058 competition v3 |
| --- | --- | --- | --- |
| ICD searchable coverage | 15,308 index records | 15,308 rows, 11,640 runtime-authoritative | **15,308 searchable, 0 excluded** |
| ICD suspect names | not flagged | 3,470 flagged, non-authoritative | 3,470 flagged, **searchable + penalised** |
| ICD hierarchy | 12,968 unlabeled symmetric edges | 12,968 review-required | 12,968 **advisory**, 0 may offer a candidate |
| RxNorm searchable | 82,429 | 82,429 | 82,429 |
| Closure-only concepts | 0 | 0 | **129,520** |
| RxNorm concepts total | 82,429 | 82,429 | 211,949 |
| Relations | unlabeled symmetric adjacency | 2,563,978 rows, labels retained | **960,086 directed labelled, structure-filtered** |
| Endpoint validity | not recorded | 553,464 ok / 2,010,514 "missing" (miscounted) | **0 missing after closure** |
| TTY coverage | one TTY per concept (first atom) | full collection | full collection + representative TTY |
| Crosswalk physical rows vs bridges | 12-entry hard-coded dict, disabled | 46 rows, all review-required | **46 rows → 12 canonical bridges**, all `RETRIEVAL_ONLY` |
| Exact-match behaviour | works | n/a (not an index) | works (`acetaminophen` → 161 top hit) |
| INN surface behaviour | `paracetamol` → **0 candidates** | n/a | `paracetamol` → RxCUI 161 |
| Strength / dose-form behaviour | structured, unchanged | n/a | structured, unchanged, richer graph |
| Combination behaviour | MIN retained | n/a | MIN retained, not reduced |
| Candidate-list sizes | up to 20 per mention | n/a | retrieval up to 20, **L8 emits ≤ 2** |
| Deterministic ordering | yes | yes | yes (5 repeated runs identical) |
| L9 offered-set enforcement | intact | intact | **intact, plus eligibility refusal** |
| Disk footprint | 10 MB + 103 MB indices | 3.2 GB | **345 MB** + 12 MB / 138 MB candidate indices |

## 12. Runtime Migration Result

Activation criteria from the milestone brief:

| Criterion | Status |
| --- | --- |
| All KB validators pass | **pass** — `VALID_COMPETITION_CANDIDATE`, 0 violations |
| Every retained relation endpoint exists | **pass** — 0 missing after closure |
| Closure-only nodes cannot leak | **pass** — three independent guards, verified on real data |
| `RETRIEVAL_ONLY` bridges cannot directly emit | **pass** — returns names, never codes |
| 46 physical rows collapse correctly | **pass** — 46 → 12 |
| No duplicate canonical bridge key | **pass** — asserted in the builder and tested |
| ICD advisory hierarchy cannot offer codes | **pass** — 0 edges may offer |
| Candidate top-k is deterministic | **pass** |
| Canonical runner requires no redesign | **pass** — candidate indices write the same symmetric adjacency, so the TTY-role walk is unchanged |
| L8/L9 invariants intact | **pass** |
| Manifests, snapshot IDs and hashes updated | **pass** |

**Activated now**, because none of it requires an index migration:

- competition top-k decoding in L8, default-on;
- governed `RETRIEVAL_ONLY` crosswalk expansion in L5 medication retrieval;
- the closure-only leak guards in the linker and in L9.

**Staged, not activated**: the v3 snapshot and its derived runtime indices. The
verified runtime configs still point at `indices/icd10_vi/tt06-2026/index.json` and
`indices/rxnorm/prescribable-2026-07-06/index.json`.

The remaining step is a config path change, and it is deliberately left to the owner
rather than taken here. Switching the active KB indices alters retrieval inputs for the
whole `FRAMEWORK_RUNTIME_FROZEN` L5 stack and for the verified container image; that is
a freeze-affecting decision, and the owner alone stages, commits and pushes. Nothing
blocks it technically: the candidate indices are built, verified and behaviour-checked,
and the previous indices are untouched so rollback is immediate.

| Candidate index | Records | deterministic_index_hash |
| --- | ---: | --- |
| `indices/candidate/icd10_vi/competition-v3/` | 15,308 | `b050a2d5ead90c8410101d257b08b347c129f5a8faa713d7e5f2757c7931a975` |
| `indices/candidate/rxnorm/competition-v3/` | 211,949 | `d3c45f349bad1532135b70214c48e54c270ede6ae6988e1444a9949887d92ac3` |

## 13. Validation Results

Targeted, during implementation:

| Command | Result |
| --- | --- |
| `pytest tests/unit/test_competition_kb_0058.py -q` | `53 passed` |
| `pytest tests/unit/test_competition_kb_0058.py tests/unit/test_kb_freeze_0057.py -q` | `59 passed` |
| `pytest tests/unit/test_l5_l7_deterministic_stack.py tests/unit/test_route_gating_and_l8_l9.py -q` | `142 passed` |

One final full-suite pass:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Result: see §13a and §13b below.

Static and type gates:

| Command | Result |
| --- | --- |
| `ruff check .` | `All checks passed!` |
| `ruff format --check <changed Python files>` | `14 files already formatted` |
| `mypy src/mednorm_vi` | `Success: no issues found in 288 source files` |
| `compileall -q src tests` | passed |
| `git diff --check` | passed |

Note: `metric_decoder/decoder.py` and `linking/rxnorm_graph.py` were already not
`ruff format`-clean at HEAD. They were left in their existing style rather than
reformatted, because repository-wide formatting is out of scope for this milestone and
reformatting them would bury the behavioural diff.

## 13a. Full-suite Result and a Pre-existing Failure It Exposed

The first full-suite run surfaced one failure, and it was **not** caused by this
milestone's code:

```text
FAILED tests/unit/test_e4_retirement.py::test_no_audit_was_deleted
1 failed, 1876 passed, 1 skipped in 348.75s (0:05:48)
```

`test_no_audit_was_deleted` globbed `[0-9][0-9][0-9][0-9]-*.md`, which requires a digit
immediately followed by `-` and therefore **silently excluded every lettered audit**.
Milestone 56 was only ever written as `0056a` through `0056g`, so audit number 56
looked deleted the moment any later audit existed. Verified directly: with this
milestone's `0058-*.md` removed from the calculation, the numbers are
`[1..55, 57]` and the gap at 56 is still there.

The failure survived undetected because the last full suite ran during Audit 0057
*before* `0057-kb-audit-repair-and-freeze.md` was created (which is why 0057 recorded
1824 passed), and Audit 0057A explicitly ran no full suite. Adding this milestone's
audit file did not introduce the bug; running the suite again is what exposed it.

Fix: the test now globs `[0-9][0-9][0-9][0-9]*-*.md` and de-duplicates the parsed
numbers, so an instalment letter no longer hides a milestone. Its intent — numbering,
not file naming — is unchanged, and its `max(numbers) >= 51` floor is untouched.

Because that fix invalidated the first run's result, the suite was run once more to end
on a verified state. That confirming run's actual output is recorded in §13b.

## 13b. Confirming Full-suite Result

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

1877 passed, 1 skipped in 347.38s (0:05:47)
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

The suite ran twice rather than the single pass the brief allows: once as specified,
and once more only because the first run's result had been invalidated by the
`test_no_audit_was_deleted` fix in §13a. Reporting the first run as the outcome would
have described a state the tree was no longer in.

The single skip is the pre-existing `pyarrow` absence, unchanged from Audit 0057.

## 14. Snapshot IDs and Hashes

Snapshot ID: `competition-kb-v3-candidate-4d206538e7d24c32`
Validation status: `VALID_COMPETITION_CANDIDATE`, 0 violations.

| Artifact | Rows | canonical_content_sha256 |
| --- | ---: | --- |
| `competition_candidate_index_v1.csv` | 227,257 | `aff0f1415a8fd088…` |
| `competition_provenance_v1.csv` | 15 | `78ba63971f0dc908…` |
| `icd10_advisory_hierarchy_v1.csv` | 12,968 | `6db5aa8b6b966b18…` |
| `icd10_competition_concepts_v1.csv` | 15,308 | `ed5c02c712282f32…` |
| `inn_rxnorm_canonical_bridges_v1.csv` | 12 | `5dd6997ba9ed2c41…` |
| `rxnorm_competition_concepts_v1.csv` | 211,949 | `349528811f71eee9…` |
| `rxnorm_competition_relations_v1.csv` | 960,086 | `fa8cdced9cbcf410…` |
| `rxnorm_governed_synonyms_v1.csv` | 245,401 | `2bd227e5d9ad80b3…` |

All 8 artifacts were independently re-verified against the manifest after formatting:
row counts, `artifact_sha256` and `canonical_content_sha256` all match. Full digests
are in `competition_snapshot_manifest_v1.json`.

Canonical crosswalk bridge hash:
`2ef033277496624bdcfa431ada11b0ce3ab304a1b047f9e33e56e62ecd75c7c1`, reproduced from a
reversed-order input.

## 15. Disk and Memory Observations

| Metric | Observation |
| --- | --- |
| `free -h` before the KB build | 14 GiB total, 6.9 GiB available |
| Lowest observed available RAM | **5.9 GiB**, during the v3 index build |
| Stop threshold | 3 GiB — never approached |
| `df -h .` before | 100 G, 26 G available, 75% used |
| `df -h .` after | 100 G, 25 G available, 76% used |
| v3 snapshot | 345 MB |
| v3 candidate indices | 12 MB (ICD) + 138 MB (RxNorm) |
| 0057 candidate v2 | 3.2 GB, **unmodified** |

Memory was bounded by construction: RRF files are streamed line by line, the Full
concept file is read twice (a map-only pass, then aggregation restricted to the known
closure set) rather than aggregating all 1.2M concepts and discarding most, and the
retained-relation map holds one compact entry per surviving relation rather than per
source row. No two heavy jobs ran concurrently.

## 16. Exact Changed-file Inventory

Tracked modifications:

```text
M docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
M docs/kb/icd10/SNAPSHOT.md
M docs/kb/rxnorm/SNAPSHOT.md
M src/mednorm_vi/linking/rxnorm.py
M src/mednorm_vi/linking/rxnorm_graph.py
M src/mednorm_vi/linking/structured_medication.py
M src/mednorm_vi/metric_decoder/decoder.py
M src/mednorm_vi/validator/kb_membership.py
M tests/unit/test_e4_retirement.py
M tests/unit/test_kb_freeze_0057.py
M tests/unit/test_l5_l7_deterministic_stack.py
```

`tests/unit/test_e4_retirement.py` is unrelated to the competition KB: it carries the
one-line glob fix for the pre-existing audit-numbering failure described in §13a.

New tracked-intended files:

```text
docs/audits/0058-competition-kb-policy-and-runtime-activation.md
docs/kb/KB_COMPETITION_0058.md
schemas/kb/canonical-crosswalk-bridge-schema-v1.json
schemas/kb/competition-icd10-view-schema-v1.json
schemas/kb/competition-rxnorm-view-schema-v1.json
schemas/kb/competition-snapshot-manifest-schema-v1.json
src/mednorm_vi/kb/competition/__init__.py
src/mednorm_vi/kb/competition/build.py
src/mednorm_vi/kb/competition/crosswalk.py
src/mednorm_vi/kb/competition/eligibility.py
src/mednorm_vi/kb/competition/icd10_view.py
src/mednorm_vi/kb/competition/indices.py
src/mednorm_vi/kb/competition/policy.py
src/mednorm_vi/kb/competition/rxnorm_view.py
src/mednorm_vi/kb/competition/topk.py
tests/fixtures/kb/competition/full/RXNCONSO.RRF
tests/fixtures/kb/competition/full/RXNREL.RRF
tests/fixtures/kb/competition/prescribable/RXNCONSO.RRF
tests/fixtures/kb/competition/prescribable/RXNREL.RRF
tests/unit/test_competition_kb_0058.py
```

Ignored generated evidence:

```text
data/derived/kb_freeze/0058_competition_candidate_v3/
indices/candidate/icd10_vi/competition-v3/
indices/candidate/rxnorm/competition-v3/
```

Two pre-existing tests changed meaning and say so in their own docstrings:
`test_the_inn_bridge_is_review_inventory_not_runtime_authority` and
`test_legacy_inn_bridge_inventory_does_not_widen_runtime_retrieval` asserted that the
bridge returns nothing. Under the 3B governance split they assert that it widens lookup
while never returning a code. `data/derived/kb_freeze/0057_candidate_v2/` was read
only and is byte-identical.

## 17. Remaining Leaderboard Hypotheses

Untested, ranked by expected candidate-score impact:

1. **Top-k thresholds are unvalidated.** 0.98 / 0.95 / 0.80 are reasoned from Jaccard's
   shape, not fitted — no code-bearing gold exists to fit them against. If a gold set
   appears, these are the first numbers to search.
2. **Advisory hierarchy contributes no ranking signal yet.** The weight exists and the
   policy permits it; the ICD linker does not yet read `name_quality` or the advisory
   weight. Wiring them is the cheapest remaining ICD improvement.
3. **The labelled relation table is unused at runtime.** The candidate index writes
   symmetric unlabeled adjacency so the walk needs no redesign. Traversing by *relation*
   instead of by inferred TTY would tighten candidate generation, and the data is now
   there.
4. **The INN bridge covers 12 surfaces.** Vietnamese clinical text uses many more
   INN/WHO names absent from RxNorm's USAN vocabulary. Extending it is high-value and
   must stay governed — a long hand-written drug table is a safety hazard.
5. **129,520 closure-only concepts may include genuinely prescribable products** absent
   from the prescribable extract. Promoting any of them is a source-evidence decision,
   not a lexical one.
6. **Suspect-name penalties (25.0 / 50.0) are unmeasured**, as is `MIN_TOKEN_OVERLAP`.
7. **Candidate accuracy remains unmeasurable.** Recall@K and candidate Jaccard are
   `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`. Everything above is a behavioural or
   coverage change, **not** a measured score gain.

## 18. Confirmation That No Human Review Was Falsely Represented

No record produced by this milestone claims human review.

- 12 of 12 crosswalk bridges are `RETRIEVAL_ONLY`; **0** are `CLINICALLY_APPROVED`.
- `assert_automated_decision` raises `ClinicalApprovalForbidden` on any attempt to
  record clinical approval, and a test asserts it.
- Every ICD and RxNorm record's role is `COMPETITION_RUNTIME_ELIGIBLE`,
  `RETRIEVAL_ONLY` or `EXCLUDED` — never `CLINICALLY_APPROVED`.
- The `reviewer` and `review_timestamp` fields are absent from the canonical bridge
  schema entirely, so an empty reviewer cannot be misread as a completed review.
- The ICD searchability decision is documented in three places as a
  competition-scoring judgement, explicitly not a clinical one.

No clinical adjudication was performed, and none is represented as having been.

## 19. Verdict

`COMPETITION_KB_ACTIVE_WITH_NOTES`

The competition KB **policy** is active in the runtime: Jaccard-aware top-k decoding,
governed `RETRIEVAL_ONLY` crosswalk expansion that closes a real coverage hole, and
closure-only leak guards at both the linker and the independent L9 gate. The candidate
v3 **snapshot** is built, validated, hash-verified and behaviour-checked, with derived
runtime indices staged and a verified no-redesign activation path — but the active
index paths are unchanged, because switching them affects the frozen framework and the
container image, and that is the owner's call.

The "notes" are exactly that gap, plus the standing fact that candidate accuracy
remains unmeasurable without code-bearing gold.

## 20. Next Recommendation

1. Review and commit this milestone. Nothing here was staged, committed or pushed.
2. Decide on index activation. To activate, point `configs/pipeline/full_v1.yaml` at
   `indices/candidate/*/competition-v3/`; to roll back, point it back. Both index sets
   exist side by side. Re-run the full suite and the container smoke after switching,
   since that changes the frozen runtime's inputs.
3. Wire `name_quality` and the advisory hierarchy weight into ICD ranking — the
   cheapest remaining improvement, and the data is already emitted.
4. Build a small code-bearing gold set. It is the blocker on *every* quantitative claim
   in this audit, and until it exists the top-k thresholds cannot be validated.
5. Do not begin model download, training, fine-tuning, calibration or organizer
   inference from this audit alone.
