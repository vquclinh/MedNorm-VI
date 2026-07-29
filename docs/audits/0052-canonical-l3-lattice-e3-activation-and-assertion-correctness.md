# Audit 0052 — Canonical L3 Lattice, E3 Activation, and Assertion Correctness

- **Date:** 2026-07-29
- **Author:** Claude (AI agent), for human review
- **Change type:** Architecture spine repair. The canonical runner now builds the
  unified L3 span lattice, runs registered experts through it, and resolves through
  **one** L4. The existing E3 checkpoint runs in production for the first time. A
  measured assertion false-positive defect is fixed. **No commit. No push.**
- **Constraints held:** no model downloaded; no training or fine-tuning; `internal_test`
  never opened; no organizer inference; no `output.zip`; no Docker build; the
  architecture PDF unmodified; no assistant control file created.
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1, SHA-256
  `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b` — read in full,
  **unchanged**.
- **Status:** `CANONICAL_SPINE_WIRED_AND_E3_ACTIVE`

---

## 1. Initial state

```text
branch  main   (in sync with origin/main)
HEAD    7afcab2c48cea7bf7d825cd4e2c0c88186dd8528
        "chore: retire E4 from the architecture and consolidate the repository"
working tree  CLEAN  (0 staged, 0 unstaged, 0 untracked)
tracked files 599
latest audit  0051
```

The read-only review at `7afcab2` is the verified starting state. Its two blocking
findings are what this milestone repairs:

* **A** — `build_span_lattice` had **zero callers under `inference/`**. The runner
  passed Phase-1B proposals straight to `resolution/resolver.py`, so E3 — trained and
  validated since Audit 0031 — could not run in production at all, and every new
  expert would have required editing the runner.
* **B** — `specialists/assertion/hydra.py` assigned all three assertion labels to
  essentially every entity in any document containing any cue.

## 2. E3 checkpoint verification and backup gate

Step 1 of the milestone was a gate: no source work until the checkpoint was verified
and a backup established. Both were done **read-only**; nothing was moved or written.

```text
path      checkpoint/s1_mention_full_training_v1/best.pt
size      1,615,513,303 bytes            expected 1,615,513,303   MATCH
sha256    a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
          expected a64cc173…1017c                                 MATCH
mtime     2026-07-27 00:53:41.047814400 +0700
tracked   no (.gitignore:15 /checkpoint/)
```

**External backup: VERIFIED.** Audit 0051 §19 listed this as the one blocker risking
loss of work, because Audit 0032's Drive copy could not be checked from this machine.
It can: an rclone remote of `type = drive` is configured, and Drive stores an MD5 per
object, so byte-identity is checkable **without downloading**:

```text
remote    vquclinh:MedNorm-VI/artifacts/s1_mention_full_training_v1/checkpoints/
          best.pt    size 1,615,513,303   ModTime 2026-07-26T17:41:55Z
          latest.pt  size 1,615,546,719

md5  remote  01fda557e3481303f3e654a03b6489a1
md5  local   01fda557e3481303f3e654a03b6489a1        MATCH
```

Size and MD5 agree, so the Drive copy is byte-identical and the local file is no
longer an unverified only-copy. Nothing was transferred; only object metadata was
read. **Audit 0051 §19 blocker 4 is resolved.**

After every operation in this milestone, re-verified: SHA-256 unchanged, mtime
unchanged (asserted by test).

## 3. Contract lineage, before and after

### 3.1 Before

Four types named `SpanProposal`, two named `DocumentGraph`, no declared ownership, and
**no conversion function** between the deterministic and expert contracts — the
conversion existed only inside `build_span_lattice`, which the runner never called.

### 3.2 After

One versioned lineage, one owner per type, each marked runtime or not:

```text
L1   document_intelligence.models.DocumentGraph        RUNTIME  (8 fields)
     schemas.document.DocumentGraph                    NON-RUNTIME sketch (marked)

L3   E1/E2  -> mention_factory.models.SpanProposal     RUNTIME  (22 fields)
     E3..E7 -> lattice.models.ExpertSpanProposal       RUNTIME  (20 fields)
     schemas.hypotheses.SpanProposal                   NON-RUNTIME sketch (marked)
              |
              v  lattice.builder.build_span_lattice
     merged -> lattice.models.SpanProposal             RUNTIME  (10 fields)
              |
              v  resolution.canonical.resolve_lattice_to_hypotheses
L4   resolution.models.EntityHypothesis                RUNTIME  (+expert_ids,
                                                                +components)
```

`schemas/hypotheses.py` and `schemas/document.py` now carry explicit
**NON-RUNTIME** headers naming the runtime type that supersedes each, so four
same-named types no longer have unclear ownership. `TypedHypothesis` is documented as
runtime, because `resolver_v1` emits it and `resolution/canonical` adapts it.

Nothing was collapsed: a 22-field provenance record, a 20-field per-expert record
with digests, and a 10-field merged node carry different information, and merging
them would destroy it.

### 3.3 Requirements from §2 of the milestone, and where each is enforced

| Requirement | Enforced at |
| --- | --- |
| complete provenance preserved | `SourceEvidence` per contributing expert, never averaged |
| original end-exclusive coordinates preserved | `ExpertSpanProposal.__post_init__` rejects mismatched `original_start`/`original_end` |
| components and relation references preserved | `SourceEvidence.components` / `.relation_refs` → `EntityHypothesis.components` / `.has_result_pair_group_ids` |
| source routes, node ids, scores preserved | `SourceEvidence` fields, unchanged |
| model revision + checkpoint SHA for neural experts | `ExpertSpanProposal`, plus `ExpertRunRecord` per document |
| reject text ≠ `original_text[start:end]` | four independent checks: expert `propose`, registry, lattice `register`, canonical L4 |
| no expert emits `EntityHypothesis` or organizer JSON | the `L3Expert` protocol has no such method; the registry accepts only `ExpertSpanProposal` |
| future experts enter without editing `inference/pipeline.py` | `mention_factory/registry.py`; a test asserts the runner names no expert |

## 4. Canonical flow, before and after

### Before (`7afcab2`)

```text
L1 → L2+E1/E2 (run_phase1b) → resolution.resolver.resolve → L5…L9
                                ^ no lattice, no neural expert reachable
```

### After

```text
L1  document_intelligence.analyze_document                 -> DocumentGraph
L2  case_router (inside run_phase1b)                       -> NodeRouting[]
L3  E1/E2 (run_phase1b)                                    -> SpanProposal[]
    + registry.run_registered_experts(scoped flags)        -> ExpertSpanProposal[]
    -> lattice.builder.build_span_lattice(...)             -> ONE SpanLattice
L4  resolution.canonical.resolve_lattice_to_hypotheses     -> EntityHypothesis[]
L5  specialists.assertion.resolve_assertions
    linking.icd10 / linking.rxnorm
L6  evidence_graph.build_evidence_graph
L7  confidence_cascade.apply_confidence_cascade
L8  metric_decoder.decode_expected_jaccard
L9  serialization -> validator.submission -> output/ + output.zip
```

Only `inference/pipeline.py` was modified; **no second runner was created.** The
runner now names no expert other than the deterministic pair Phase 1B owns — asserted
by `test_the_runner_names_no_expert_of_its_own`.

## 5. Files reused rather than rewritten

This milestone is mostly wiring. What already existed and is used unchanged:

| Reused | Role |
| --- | --- |
| `lattice/builder.py::build_span_lattice` | the unified lattice, including its coordinate-identity merge policy |
| `lattice/builder.py::build_from_phase1b` | already-existing convenience wiring |
| `resolution/resolver_v1.py::resolve_lattice` | **all** L4 decision logic: boundary trim/expand, type utilities, abstention, boundary-group selection, overlap competition, `has_result` protection |
| `resolution/config_v1.py` | the L4 configuration and its hash |
| `mention_factory/neural/runtime.py` | E3 forward pass, checkpoint fingerprinting, strict state-dict load, `local_files_only=True` |
| `mention_factory/neural/decoding.py` | `NeuralSpan` decoding and span validation |
| `specialists/assertion/cues.py` | cue lexicons and the directional scope model (extended, not replaced) |
| `evaluation/exact_mention.py` | the exact character-offset evaluator used for §9's metrics |
| `training/governed_splits.py` | split resolution by SHA-256 |

New code is four modules: `mention_factory/registry.py`,
`mention_factory/experts/` (E3 adapter), `resolution/canonical.py`, and the §8.1
scope additions inside `cues.py`.

## 6. E3 integration result

E3 is now a registered L3 expert and **really runs**.

```text
adapter        src/mednorm_vi/mention_factory/experts/e3_vihealthbert.py
registration   expert_id E3_vihealthbert_span_type
               feature_flag enable_e3_vihealthbert
               checkpoint_key mention/vihealthbert
load           LAZY — constructing the adapter touches no model (asserted)
backbone       demdecuong/vihealthbert-base-word @ f89e80b4…24b6c5
               NOT downloaded: architecture rebuilt from the locally cached
               config.json, every parameter from the checkpoint's state dict,
               local_files_only=True
segmenter      optional; a configured VnCoreNLP dir without a .jar is REFUSED
               rather than silently falling back to whitespace
load time      26.0 s (CPU)
provenance     model_revision and checkpoint_sha256 recorded per document
```

Executed proof, `mode=specialist` on a tracked fixture:

```text
experts that ran     ('E3_vihealthbert_span_type',)
proposals by expert  E1 13, E2 14, E3 3
lattice nodes        30   (deterministic mode: 27)
L4 accepted          18   (deterministic mode: 15)
record.model_revision      f89e80b461e86f9cfc1c84019bd819830c24b6c5
record.checkpoint_sha256   a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
```

The checkpoint was **not modified**: digest and mtime re-verified after a full run
(`test_the_checkpoint_is_unchanged_after_a_full_run`).

## 7. L4 consolidation

Two live L4 implementations existed on disjoint contracts, and the runner used the
weaker one. `resolution/canonical.py` is now the single entry point: it runs
`resolver_v1`'s decision logic **unchanged** and adapts the result to the
`ResolutionResult` / `EntityHypothesis` contract L5-L9 already consume.

| Behaviour required by §5 | Source |
| --- | --- |
| boundary selection (trim / expand to competitor) | `resolver_v1` |
| type assignment with evidence weighting | `resolver_v1` → `resolution/typing.py` |
| overlap and nested handling | `resolver_v1` → `resolution/overlap.py` |
| global conflict resolution | `resolver_v1` near-complete-overlap competition |
| relation evidence (`has_result` pair protection) | `resolver_v1` `_protected_partners` |
| route priors | lattice `SourceEvidence.routes` → `resolver_v1` |
| expert agreement | lattice `SourceEvidence` families → `resolver_v1` |
| abstention | `resolver_v1` `TypeDecision.abstained` → `STATUS_UNRESOLVED` |
| deterministic ordering | asserted by `test_the_canonical_l4_is_deterministic` |
| provenance preservation | `expert_ids`, `source_proposal_ids`, `components` on every hypothesis |

`resolution/resolver.py` was **not deleted**. Its per-type boundary policy
(`medication_boundary`, `test_result_boundary`) has no equivalent in the canonical
path yet, and §5 forbids deleting an implementation before its useful behaviour is
migrated. It is off the canonical path, and the manifest says so.

`resolution/learned_v2.py` already consumes `SpanLattice`; it remains **disabled**
because no checkpoint exists.

Status mapping, chosen to preserve information rather than flatten it:
`accepted → STATUS_ACCEPTED`, `abstained → STATUS_UNRESOLVED` (a type-confidence
decision, spec §7.2), `suppressed → STATUS_REJECTED` (an overlap outcome, §7.3).

## 8. Assertion defect: root cause and regression proof

### 8.1 Root cause — three independent defects

The old implementation asked whether *any* cue of a family appeared anywhere in a
symmetric ±80-character window:

```python
local = text[start-80 : end+80].casefold()
if any(cue in local for cue in NEGATION_CUES): labels.append("isNegated")
```

| # | Defect | Consequence |
| --- | --- | --- |
| A | no direction, and no §8.1 sentence/clause boundary | a cue in another sentence or clause counted |
| B | no contrast handling | "không A **nhưng** B" negated B as well as A |
| C | **bare substring matching** | the family cue `"ông"` (grandfather) matches inside `"kh**ông**"` (not), so every negated sentence also read as family context |

Measured on `tests/fixtures/phase1b/synthetic_medical_document.txt` before the fix:
**10 entities received all three labels**. Defect C was previously unreported — it was
found while writing the regression tests for A and B.

The damage was **masked, not absent**: TEST_NAME/TEST_RESULT carry no `assertions`
field, so ten wrong decisions were computed and silently discarded. Activating E3 puts
narrative DIAGNOSIS/SYMPTOM entities beside lab blocks, which is exactly where they
would have reached L9. Assertions are 30% of the organizer metric under Jaccard, so a
spurious label against an empty gold set scores zero for that entity (spec §13.3).

### 8.2 The fix

`hydra.py` now delegates every decision to `cues.decide_from_cues`, which implements
the §8.1 boundaries:

```text
direction            only cues PRECEDING the mention
sentence boundaries  . ! ? ; newline    end a cue's reach unconditionally
clause separators    , : – — |          end a cue's reach for the next clause
contrast markers     nhưng, tuy nhiên, mà, song, ngược lại, trái lại, còn
word boundaries      find_cue_occurrences() — Unicode-aware, combining marks handled
type eligibility     MEDICATION / DIAGNOSIS / SYMPTOM only
nearest cue          the closest in-scope cue per label governs
section priors       §4.2, and only for the header's own line or a colon-terminated
                     header, and only within the header's own sentence
conservative default an unscopable cue yields NO labels and flags `uncertain`
```

Two design points worth stating. **Cue evidence and section priors combine** — spec §8
A1 and A2/A3 are separate contributing stages, so "Tiền sử gia đình: bố bị tăng huyết
áp" yields `isFamily` (cue "bố") **and** `isHistorical` (header prior). And a label
resting only on a prior sets `uncertain=True`, because §4.2 says a local sentence may
override a prior and nothing at this stage has read that sentence.

Every blocked cue is still returned, with `blocked_by` naming the reason, so a
withheld label is as inspectable as an applied one.

### 8.3 Regression proof

`tests/unit/test_assertion_scope_regression.py` — **32 tests**, all passing:

```text
headline        the defect fixture now yields 0 all-three decisions (was 10)
cause A         cue after the mention; cue in a previous sentence; cue in a
                previous clause; scope_start boundary; in-scope cue still applies
cause B         contrast blocks negation; 4 markers parametrised
cause C         "ông" does not match inside "không"; real family cue still matches
eligibility     lab types never receive assertions (2 parametrised);
                eligible types do (3 parametrised)
repeated        two occurrences of "sốt" judged independently (spec §5 C7)
clauses         two cues in separate clauses do not both apply
family/current  a family header does not reach the patient-current sentence
priors          colon header governs following lines; a non-header line does not
                carry its prior forward; prior-only labels flagged uncertain
conservative    no cue and no prior = confident empty; blocked cue = empty +
                uncertain; empty never means false
```

One test in this file failed on first run and exposed a further leak — a section prior
crossing a sentence boundary *within* a header line ("Tiền sử: X. Hiện tại Y"). Fixed
by applying the same §8.1 boundary to priors.

## 9. Profile behaviour, before and after

| | Before | After |
| --- | --- | --- |
| `deterministic` | E1+E2 | E1+E2 (unchanged) |
| `specialist` | **identical to deterministic**, reported `READY` | **E1+E2+E3 through the lattice**, reported `READY` |
| `full` | fails closed, 12 missing checkpoints | unchanged: fails closed |

Two mechanisms make readiness describe real behaviour:

* **`MODE_EXPERTS` + `flags_for_mode`** — a mode is a *scope*, not a label. A profile
  that enables E3 still runs E1+E2 only in `deterministic` mode, so the two modes
  cannot collapse onto each other again.
* **`evaluate_readiness` → `ReadinessReport`** with `READY` / `DEGRADED_FALLBACK` /
  `NOT_READY`, plus `expected_experts` and `runnable_experts`. A `specialist` profile
  whose runnable set does not exceed the deterministic set now reports
  **`DEGRADED_FALLBACK`** instead of claiming readiness — this is the Audit-0051 §16-7
  overstatement, closed.

`validate_readiness` keeps its signature (blocking errors only), so every existing
caller is unaffected; a degraded profile may still run, and the report is what says it
will do less than its name implies.

No enabled expert is silently skipped: `run_registered_experts` raises
`ExpertNotReady` naming the expert, its role and the exact missing path. A disabled
expert is never even constructed (asserted).

## 10. Structured E1/E2 evidence

Spec §10.1's field decomposition and §6.2's test-result pairing previously died at the
lattice boundary — reduced to `grammar_component_count`, a single float. Now carried
through verbatim.

Measured on `synthetic_medication_list.txt`, after L4:

```text
accepted medications with components: 6/6
roles observed: name, salt, strength_value, strength_unit, concentration,
                dose_form, release, route, frequency, prn, duration
every component satisfies original_text[start:end] == component.text
```

On `synthetic_laboratory.txt`:

```text
accepted hypotheses with pair groups: 15/15
TEST_NAME 'WBC'    -> labpg-d942946fba8f819b-0001
TEST_RESULT '14,43' -> labpg-d942946fba8f819b-0001     (same group)
```

Full RxNorm graph traversal is **not** implemented here, as §8 instructs; the fields
are now available to L5 so Milestone 2 needs no further contract migration.

## 11. Validation metrics

Bounded governed **validation** run — 200 of 1,045 rows, resolved by SHA-256
`ed7cdd2d…f103`. `internal_test` was never opened. Three arms, all through the
canonical lattice + canonical L4 path. Exact character-offset span **and** type.

```text
                        P        R       F1     TP    FP    FN
A  E1+E2 deterministic  0.0000  0.0000  0.0000    0    11   406
B  E3 only              0.6220  0.5025  0.5559  204   124   202
C  E1+E2+E3 lattice     0.6018  0.5025  0.5477  204   135   202
```

Per type:

```text
arm B   DIAGNOSIS   P 0.6600  R 0.4783  F1 0.5546   TP 132  FP 68  FN 144
        SYMPTOM     P 0.5625  R 0.5538  F1 0.5581   TP  72  FP 56  FN  58
arm C   DIAGNOSIS   identical to arm B
        SYMPTOM     identical to arm B
        TEST_NAME   0 TP,  2 FP     (E2 false positives)
        TEST_RESULT 0 TP,  9 FP     (E2 false positives)
```

Counts and diagnostics:

```text
arm  proposals by expert          lattice nodes  merged  L4 acc/rej/unres  offsets
A    E2 16                                  15       0      11/0/4              0
B    E3 328                                328       0     328/0/0              0
C    E2 16, E3 328                          343       0     339/0/4              0

assertion decisions with any label:  A 0    B 13   C 13
assertion decisions with all three:  A 0    B  0   C  0     (was the defect)
schema/offset violations:            0 in every arm
runtime:   10.3 s for 200 docs x 3 arms (0.05 s/doc), + 26.0 s one-time E3 load
peak RSS:  2.49 GiB
E3 checkpoint after the run: digest UNCHANGED
```

**Three honest readings of this table.**

1. **E1/E2 score exactly zero on this corpus**, and that is expected, not a
   regression: the governed validation split is narrative DIAGNOSIS/SYMPTOM text from
   vimedner/vimq, and E1/E2 are a medication grammar and a laboratory parser. Their 11
   proposals are all false positives. This reproduces Audit 0034's Arm-2 result
   (exact F1 0.0000) on the same kind of data.
2. **Adding E1/E2 to E3 lowers F1** — 0.5477 against 0.5559 — because they add 11
   false positives and no true positive. This is why `deterministic` and `specialist`
   are separate modes, and it is an argument for route-gating the deterministic
   experts, not for deleting them: they exist for medication lists and lab blocks,
   which this split does not contain.
3. **0.5559 is not comparable to Audit 0032's 0.7194.** That figure was a
   *token-index span proxy* over all 1,045 rows computed during training; this is
   *exact character offsets plus exact type* over 200 rows. Neither is the organizer's
   complete metric — candidates (40%) and assertions (30%) are still unmeasured. **No
   organizer-metric claim is made.**

Thresholds were **not** tuned, as §9 instructs.

## 12. Tests and static checks

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q
    1574 passed, 1 skipped in 231.59s
    (skip: tests/unit/test_vietmed_adapter.py:399 — pyarrow, pre-existing)

ruff check .                                          All checks passed!
ruff check notebooks                                  All checks passed!
env PYTHONPATH=src .venv/bin/python -m mypy           Success: no issues found
                                                      in 268 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src   clean
git diff --check                                      clean
whole-package import sweep                            0 broken
```

Test movement: **1504 → 1574 (+70)**, all new. Every pre-existing test still passes;
none was weakened or deleted.

New tests, mapped to §9's required proofs:

| §9 requirement | Test |
| --- | --- |
| canonical runner calls `build_span_lattice` | `test_the_runner_calls_build_span_lattice` |
| E1 and E2 enter the lattice | `test_e1_and_e2_enter_the_lattice` |
| E3 enters the same lattice | `test_e3_runs_through_the_canonical_runner` (real weights) |
| a synthetic future expert registers without editing the runner | `test_a_future_expert_reaches_the_runner_without_editing_it`, `test_the_runner_names_no_expert_of_its_own` |
| disabled experts do not load | `test_a_disabled_expert_is_never_constructed_and_loads_nothing` |
| enabled missing experts fail closed | `test_an_enabled_but_unready_expert_fails_closed_with_role_and_path` |
| no expert bypasses L4 | `test_no_expert_output_reaches_l5_without_passing_l4`, `test_e3_emits_no_final_entity_and_cannot_bypass_l4` |
| one canonical L4 entry point | `test_the_canonical_l4_emits_the_contract_l5_consumes`, `test_the_canonical_l4_is_deterministic` |
| structured E1/E2 evidence survives L4 | `test_medication_component_spans_survive_l4`, `test_laboratory_pair_groups_survive_l4`, `test_expert_provenance_survives_l4` |
| specialist differs from deterministic | `test_mode_scopes_the_expert_set`, `test_deterministic_mode_runs_no_neural_expert`, `test_specialist_differs_from_deterministic_with_real_weights` |
| assertion false positives fixed | 32 tests in `test_assertion_scope_regression.py` |
| L9 output remains schema/offset valid | `test_l9_output_stays_schema_and_offset_valid` (3 fixtures) |

Model-backed tests skip with a reason naming exactly what is missing when the
checkpoint or `torch`/`transformers` is absent, so the suite stays green elsewhere
without passing quietly.

## 13. Changed files

Reconciled against `git status --short --untracked-files=all` and
`git diff --name-status`, which are the only source of truth here. The first draft of
this section was internally inconsistent: it claimed "Added (6)" while listing seven
paths and omitting **this audit file itself**, and it claimed "Modified (10)" while
omitting `docs/audits/README.md`, which §16 nevertheless staged. Both are corrected
below; no other claim in this audit changed.

```text
added     8
modified 11
deleted   0
renamed   0
total    19 paths
diffstat 11 tracked files changed, 943 insertions(+), 119 deletions(-)
         (the 8 added files are untracked and therefore absent from `git diff --stat`)
```

**Added (8):**
```text
docs/audits/0052-canonical-l3-lattice-e3-activation-and-assertion-correctness.md
                                                           this audit
src/mednorm_vi/mention_factory/registry.py                 L3 expert registry
src/mednorm_vi/mention_factory/experts/__init__.py          registration point
src/mednorm_vi/mention_factory/experts/e3_vihealthbert.py   E3 adapter
src/mednorm_vi/resolution/canonical.py                     canonical L4
tests/unit/test_canonical_l3_l4_spine.py                   26 tests
tests/unit/test_assertion_scope_regression.py              32 tests
tests/unit/test_e3_canonical_integration.py                12 tests
```

**Modified (11):**
```text
src/mednorm_vi/inference/pipeline.py         lattice + registry + canonical L4;
                                             PipelineResult reports lattice,
                                             expert_records, l4_counts
src/mednorm_vi/inference/config.py           l4_config, expert_settings,
                                             MODE_EXPERTS, flags_for_mode,
                                             ReadinessReport, evaluate_readiness
src/mednorm_vi/lattice/models.py             SourceEvidence.components /
                                             .relation_refs; SpanProposal
                                             accessors
src/mednorm_vi/lattice/builder.py            components + relation pair groups
                                             carried through; relations= param
src/mednorm_vi/resolution/models.py          EntityHypothesis.expert_ids /
                                             .components / components_by_role()
src/mednorm_vi/specialists/assertion/cues.py §8.1 boundaries, contrast markers,
                                             word-boundary matching, type
                                             eligibility, section priors
src/mednorm_vi/specialists/assertion/hydra.py rewritten to delegate to cues
src/mednorm_vi/schemas/hypotheses.py         marked NON-RUNTIME + ownership table
src/mednorm_vi/schemas/document.py           marked NON-RUNTIME
docs/architecture/ACTIVE_RUNTIME_MANIFEST.md truthful update + §9a lineage
docs/audits/README.md                        0052 index entry
```

**Deleted (0)** and **renamed (0)** — this milestone removed and moved nothing.

Per-file diffstat for the 11 modified files, as reported by `git diff --stat`:

```text
docs/architecture/ACTIVE_RUNTIME_MANIFEST.md  | 112 ++++++++--
docs/audits/README.md                         |   1 +
src/mednorm_vi/inference/config.py            | 198 +++++++++++++++--
src/mednorm_vi/inference/pipeline.py          | 132 +++++++++++-
src/mednorm_vi/lattice/builder.py             |  53 ++++-
src/mednorm_vi/lattice/models.py              |  45 ++++
src/mednorm_vi/resolution/models.py           |  17 ++
src/mednorm_vi/schemas/document.py            |  15 +-
src/mednorm_vi/schemas/hypotheses.py          |  33 ++-
src/mednorm_vi/specialists/assertion/cues.py  | 299 ++++++++++++++++++++++++--
src/mednorm_vi/specialists/assertion/hydra.py | 157 ++++++++++----
11 files changed, 943 insertions(+), 119 deletions(-)
```

## 14. Remaining blockers

1. **Assertion supervision does not exist.** The §8.1 fix is verified by constructed
   regression tests, **not** by a gold score, because the governed corpus has zero
   assertion labels (`annotation_coverage.json`: `assertions: false` for all four
   sources). 30% of the metric remains unmeasurable.
2. **Candidates are still a fixed top-10.** `metric_decoder.decode_expected_jaccard`
   does no expected-Jaccard decoding; 40% of the metric is unoptimized. Its name still
   overstates it.
3. **TEST_NAME / TEST_RESULT have no neural supervision**, confirmed by this
   milestone's per-type table: E3 predicts only DIAGNOSIS and SYMPTOM.
4. **E1/E2 are not route-gated**, so on narrative text they contribute only false
   positives (§11 reading 2). This is a scoring loss available to fix cheaply.
5. **No `Dockerfile`, no `requirements.lock`.** Spec §19 and Appendix A require both.
6. **Two L4 implementations still exist**; `resolution/resolver.py` is off the
   canonical path but retained pending migration of its per-type boundary policy.
7. **E5/E6/E7 have no local weights**, and E5's head is untrained.

*Resolved this milestone:* the unreachable-lattice blocker, the E3-cannot-run blocker,
the assertion false-positive defect, the false `specialist` readiness claim, and the
unverified-only-copy checkpoint risk.

## 15. Safe-to-commit verdict

```text
SAFE_TO_COMMIT
```

```text
architecture PDF                 unchanged, SHA-256 0d5eaa20…81e09b
E3 checkpoint                    unchanged (digest + mtime), Drive backup verified
audits 0001-0051                 unmodified; 0052 added
governed corpus / ontologies     untouched
internal_test                    never opened
model downloads                  none (local_files_only=True, backbone rebuilt
                                 from cached config + checkpoint state dict)
training / fine-tuning           none
organizer inference              did not run
output.zip                       not created
Docker image                     not built
assistant control files          none
nothing artifact-shaped staged   verified
tests / lint / types / compile   all green (§12)
```

## 16. Exact commit commands

Stage explicitly; do not use `git add -A`.

```bash
cd /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI

git add \
  docs/audits/0052-canonical-l3-lattice-e3-activation-and-assertion-correctness.md \
  docs/audits/README.md \
  docs/architecture/ACTIVE_RUNTIME_MANIFEST.md \
  src/mednorm_vi/mention_factory/registry.py \
  src/mednorm_vi/mention_factory/experts/ \
  src/mednorm_vi/resolution/canonical.py \
  src/mednorm_vi/inference/pipeline.py \
  src/mednorm_vi/inference/config.py \
  src/mednorm_vi/lattice/models.py \
  src/mednorm_vi/lattice/builder.py \
  src/mednorm_vi/resolution/models.py \
  src/mednorm_vi/specialists/assertion/cues.py \
  src/mednorm_vi/specialists/assertion/hydra.py \
  src/mednorm_vi/schemas/hypotheses.py \
  src/mednorm_vi/schemas/document.py \
  tests/unit/test_canonical_l3_l4_spine.py \
  tests/unit/test_assertion_scope_regression.py \
  tests/unit/test_e3_canonical_integration.py

# Preflight
git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors|bin|zip)$' && echo FAIL || echo "no weights staged"
git status --porcelain | grep '^??' && echo "untracked remain" || echo "clean"
git diff HEAD --quiet -- docs/MedNorm-VI_Architecture.pdf && echo "PDF unchanged"

git commit -m "feat: wire the canonical L3 lattice, activate E3, and fix assertion scope

Route the canonical runner through the unified SpanLattice and one canonical L4,
add an L3 expert registry so future experts need no runner edit, and activate the
existing verified E3 checkpoint in specialist mode. Fix a measured assertion
false-positive defect (no direction/boundary, no contrast handling, bare-substring
cue matching) with 32 regression tests. Carry E1 components and E2 relation pairs
through to L5.

Audit: docs/audits/0052-canonical-l3-lattice-e3-activation-and-assertion-correctness.md"
```

## 17. Exact next milestone

**Milestone 2 — Route-gate the deterministic experts, then finish deterministic L5.**

Ordered so each step is measurable with what now exists:

1. **Route-gate E1/E2** so they contribute only on the routes they are for
   (C1 medication-list, C2 lab). §11 reading 2 shows this recovers the 0.0082 F1 that
   arm C loses, and the L2 router already produces the tags.
2. **Consume the structured evidence this milestone preserved**: RxNorm §10.1
   structured linking from `EntityHypothesis.components`, and ICD §9.3 specificity
   from `index["graph"]` (14,534 nodes, already materialized and loaded).
3. **L6 edges + §11 deterministic consistency**, then make L8 read the graph.
4. **Honest deterministic L8**: replace the fixed top-10 and rename
   `decode_expected_jaccard` to what it does until it earns the name.
5. **`Dockerfile` + `requirements.lock`.**
6. **Migrate `resolution/resolver.py`'s per-type boundary policy** into the canonical
   L4, then delete it.

Explicitly **not** in Milestone 2: model downloads, training, calibration, threshold
search, organizer inference, `output.zip`.
