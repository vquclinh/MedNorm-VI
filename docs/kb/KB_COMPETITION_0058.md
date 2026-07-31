# Milestone 3B Competition KB Report

Milestone 3B re-reads the frozen Milestone-3A evidence under a competition policy,
builds the candidate v3 snapshot, and activates the parts of it that need no index
migration. It downloads nothing, trains nothing and loads no model.

## Why 3A's Posture Had To Change

3A froze one governance state — "has a human reviewed this?" — and answered it "no"
for everything uncertain. That is correct for clinical deployment and wrong for a
scored competition, because it blocked the runtime behind thousands of manual
decisions the organizer's metric never asks about.

3B separates the two questions 3A had fused:

```text
may a human rely on this clinically?     ->  CLINICALLY_APPROVED
may the runtime offer this to the scorer? ->  COMPETITION_RUNTIME_ELIGIBLE
```

An automated builder may answer the second and **never** the first.
`kb/competition/policy.py::assert_automated_decision` raises
`ClinicalApprovalForbidden` rather than allowing any code path to record
`CLINICALLY_APPROVED`. No record in this milestone claims human review.

Roles: `CLINICALLY_APPROVED`, `COMPETITION_RUNTIME_ELIGIBLE`, `RETRIEVAL_ONLY`,
`EXCLUDED`, `UNMAPPED`.

## Corrected Finding: The "2,010,514 Missing Endpoints" Were Not Missing

Audit 0057 reported 2,010,514 prescribable relation rows with missing endpoints and
listed it as an activation blocker. Re-measured against the same local RRF files, the
count is an artifact of the check, not a defect in the data.

`RXNREL.RRF` records relations at two levels. A `CUI`-level row populates
`RXCUI1`/`RXCUI2`; an `AUI`-level row leaves those columns empty **by design** and
identifies endpoints in `RXAUI1`/`RXAUI2`. The 3A builder read only the RXCUI columns,
so every atom-level row looked like a relation with two missing endpoints.

| Measurement | Result |
| --- | ---: |
| Prescribable relation rows | 2,563,978 |
| `CUI`-level rows | 553,464 |
| `CUI`-level rows with a missing endpoint | **0** |
| `AUI`-level rows | 2,010,514 |
| `AUI` rows resolvable to concepts via `RXNCONSO.RRF` | **2,010,514** |
| Unresolvable atoms | **0** |
| Missing endpoints after resolution | **0** |

The prescribable relation set was already endpoint-complete. It needed no closure at
all; what Full RxNorm supplies is additional *reach*, not repair.

## Canonical Crosswalk: 46 Source Rows Are 12 Bridges

Verified from the frozen artifact, not assumed:

| Measurement | Value |
| --- | ---: |
| Physical rows in `inn_rxnorm_crosswalk_v1.csv` | 46 |
| Unique normalized local surfaces | 12 |
| Unique RxCUIs | 11 |
| Canonical `(normalized_local_surface, rxcui)` bridges | 12 |
| `RETRIEVAL_ONLY` | 12 |
| `CLINICALLY_APPROVED` | 0 |
| `EXCLUDED` / `UNMAPPED` | 0 / 0 |

The 46-to-12 inflation had one cause: 3A emitted one row per matching *atom*, so
`adrenaline` produced six rows because RxCUI 3992 carries `EPINEPHRINE`,
`EPINEPHrine`, `Epinephrine` and `epinephrine` across term types `IN`, `SU` and
`TMSY`. Different casing and term types for one RxCUI are corroborating evidence for
one bridge, not six competing claims.

Two consequences are load-bearing:

- **A term-type collection is not a concept type.** 3A wrote `IN|SU|TMSY` into a field
  named `rxnorm_tty`, which reads as though RxNorm has a term type of that name. It
  does not. `observed_tty` is now a typed collection; the pipe-joined form is a storage
  encoding only.
- **Two surfaces may share one RxCUI.** `co-trimoxazole` and
  `trimethoprim-sulfamethoxazole` both name RxCUI 10831 and remain two bridges,
  because they are two lookup keys the runtime must normalize. The uniqueness invariant
  is on the pair, never on either half.

Canonical bridge content hash:
`2ef033277496624bdcfa431ada11b0ce3ab304a1b047f9e33e56e62ecd75c7c1`.

## What a RETRIEVAL_ONLY Bridge May Do

It may widen **lookup**. It may not emit a code.

`bridged_ingredient_names` returns RxNorm-side **names**, never RxCUIs, so the result
cannot be mistaken for a candidate even in principle. Those names re-enter retrieval as
queries, and whatever concept they reach faces every downstream check: snapshot
membership, TTY terminality, closure-only membership and structured compatibility.

Measured effect on the most common Vietnamese medication surface, which occurs in zero
records of the locked snapshot:

| Mention | 3A behaviour | 3B behaviour |
| --- | ---: | --- |
| `paracetamol` | **0 candidates** | 20 retrieved, top concept RxCUI 161 `ACETAMINOPHEN` (`IN`) |
| `paracetamol 500mg` | 0 candidates | top tier `structured_exact` |
| `salbutamol` | 0 candidates | top concept RxCUI 435 `ALBUTEROL` (`IN`) |

Every use is recorded in the traversal notes as
`inn_bridge:<surface>-><name>:RETRIEVAL_ONLY_NOT_CLINICALLY_APPROVED`.

## ICD Competition Policy

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

A suspect name is a negative ranking feature, never an exclusion: a fragment label does
not make its code wrong, and a concept the runtime refuses to index can never be
retrieved. No name is repaired; `source_name` is preserved verbatim.

Hierarchy remains advisory prefix inference. `advisory_hierarchy_may_offer` is hard
`False`, so an advisory edge may re-rank a code lexical retrieval already reached but
can never put a new code on the table. Exact lexical evidence outweighs advisory
hierarchy by three orders of magnitude (`1000.0` against `1.0`).

## RxNorm Searchable View and Endpoint Closure

Retained-relation policy: concept-level relations whose `RELA` is in the competition
structure set — the ingredient/component/dose-form/tradename relations the
`IN -> SCDC -> SCD -> SBD` walk traverses. `has_inactive_ingredient` alone is 1.5M rows
and cannot identify a product, so carrying it would inflate the snapshot and the
closure for no candidate benefit.

| Metric | Value |
| --- | ---: |
| Searchable concepts | 82,429 |
| Closure-only concepts | 129,520 |
| Concepts after closure | 211,949 |
| Retained relations | 960,086 |
| — from prescribable | 350,114 |
| — added from Full | 609,972 |
| Full structure rows scanned | 1,527,146 |
| Excluded, non-structure `RELA` | 2,213,864 |
| Excluded, no searchable endpoint | 445,006 |
| Duplicate relation rows collapsed | 472,168 |
| **Missing endpoints after closure** | **0** |
| Distinct relation labels | 14 |

`searchable` and `graph_endpoint` are separate flags throughout. A closure-only concept
is traversable and never emittable, enforced in three independent places:

1. it is absent from the index postings, so retrieval cannot reach it;
2. `linking/rxnorm.py` drops it with `DROP_CLOSURE_ONLY` when traversal reaches it;
3. `validator/kb_membership.py` refuses it with `kb.candidate_not_final_eligible`,
   without importing any linker.

The L9 gate treats an unstated membership as permissive, because the legacy indices
draw no such distinction and reading their silence as exclusion would empty every
candidate list.

## Candidate Top-k Policy

The organizer scores `candidates` by Jaccard. With one gold code, one correct answer
scores `1/1`; the correct answer plus one wrong one scores `1/2`. A speculative extra
candidate costs exactly as much as a miss.

| Type | Policy |
| --- | --- |
| Medication | top-1; a second only when `score2/score1 >= 0.98` in the same tier |
| Diagnosis | top-1; a second on `>= 0.95`, or on a specific/unspecified code pair at `>= 0.80` |
| Symptom, test name, test result | none (spec §7.3) |

Hard ceiling of 2. `apply_competition_topk` returns a sub-sequence of the list L8
already selected, so ordering is inherited and **no code can be introduced** — the
L8/L9 offered-set invariant holds structurally rather than by a later check. Passing
`competition_topk=False` reproduces the Audit-0054 behaviour exactly.

Thresholds are fixed and written down rather than searched: no stage produces a
calibrated probability, so a tuned threshold would be fitted to a score that is not one.

## Snapshot

| Field | Value |
| --- | --- |
| Snapshot ID | `competition-kb-v3-candidate-4d206538e7d24c32` |
| Path | `data/derived/kb_freeze/0058_competition_candidate_v3/` |
| Validation status | `VALID_COMPETITION_CANDIDATE` |
| Violations | none |
| Footprint | 345 MB (v2 is 3.2 GB) |

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

Full digests are in `competition_snapshot_manifest_v1.json`.

## Runtime Status

**Active now** (needs no index migration):

- competition top-k decoding, default-on in L8;
- governed `RETRIEVAL_ONLY` crosswalk expansion in L5 medication retrieval;
- the closure-only leak guards in `linking/rxnorm.py` and `validator/kb_membership.py`.

**Staged, not active**: the v3 snapshot and its derived runtime indices under
`indices/candidate/`. The verified runtime configs still point at
`indices/icd10_vi/tt06-2026/index.json` and
`indices/rxnorm/prescribable-2026-07-06/index.json`. Switching them is a one-line
config change the repository owner makes; the previous indices are untouched, so
rollback is immediate.

| Candidate index | Records | deterministic_index_hash |
| --- | ---: | --- |
| `indices/candidate/icd10_vi/competition-v3/` | 15,308 | `b050a2d5ead90c84…` |
| `indices/candidate/rxnorm/competition-v3/` | 211,949 | `d3c45f349bad1532…` |

The candidate indices write `graph` as the same symmetric unlabeled adjacency the
legacy indices use, so `rxnorm_graph`'s TTY-role traversal works unchanged and
activating v3 needs **no redesign of the canonical runner**. The directed, labelled
relations remain in `rxnorm_competition_relations_v1.csv` for a future relation-aware
walk.

## What Remains Unmeasurable

Candidate accuracy. The governed corpus carries zero ICD-10 and zero RxNorm gold
codes, so Recall@K and candidate Jaccard remain
`UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`. Every improvement recorded here is a
behavioural or coverage change, not a measured score gain.
