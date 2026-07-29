# Audit 0053 — Route Gating, Honest L8, L9 KB Membership, and Reproducibility Files

- **Date:** 2026-07-29
- **Author:** Claude (AI agent), for human review
- **Change type:** Correctness and truthfulness repair at L2/L3, L8 and L9, plus the
  first reproducibility infrastructure. **No commit. No push.**
- **Scope status:** **PARTIAL MILESTONE.** The Milestone 2 prompt contained fifteen
  numbered sections. Sections 1 and 15 were starting-state verification and this
  consolidated audit; **sections 2 through 14 contained thirteen implementation
  tasks.** Of those thirteen, **six were completed** (2, 8, 9, 11, 13, 14) and
  **seven were not attempted** (3, 4, 5, 6, 7, 10, 12). The milestone's own
  **ACCEPTANCE CRITERIA are therefore NOT MET.** §17-§19 name every gap; §22 orders
  the remaining work. Nothing in this audit should be read as "L5-L9 is complete."

  An earlier revision of this audit said "fifteen work items, six implemented, seven
  not attempted", which does not add up: it counted the two non-implementation
  sections into the denominator. The denominator is **thirteen**.
- **Constraints held:** no model downloaded; no training or fine-tuning; `internal_test`
  never opened; no organizer inference; no `output.zip`; **the Docker image was
  authored but never built**; no broad threshold optimization; the architecture PDF
  unmodified; no assistant control file created or tracked.
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1, SHA-256
  `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b` — read in full,
  **unchanged** (§2).
- **Status:** `ROUTE_GATED_AND_L8_L9_HONEST__L5_L6_L7_STILL_INCOMPLETE`

### Correction pass (post-approval, same audit — append-only)

Audit 0053 was approved as a partial Milestone 2 commit, with three required
corrections. All three are applied and folded into the sections below:

1. **Task accounting corrected** (header above). "Fifteen work items, six
   implemented, seven not attempted" did not add up; the denominator is the
   **thirteen implementation tasks** in prompt sections 2-14.
2. **`.claude/AGENTS.md` deleted** by explicit owner decision — §1.1, with its hash
   and size recorded before deletion.
3. **The L9 membership gate was found to have no caller and is now wired into the
   canonical packaging path** — §10.1. This is the most consequential of the three:
   §10 originally described a validator that existed but that **nothing invoked**, so
   its guarantee was unenforced. It now runs on every packaged submission and stops
   the run. Three tests were added (48 → 51 in the module).

---

## 1. Initial Git state

```text
branch        main
HEAD          ce18755b29a5f331648bf24323846752a18bc306
              "feat: wire the canonical L3 lattice, activate E3, and fix assertion scope"
              (committed by the repository owner, not by this agent)
origin/main   ce18755   — in sync, nothing to push
working tree  CLEAN     — 0 staged, 0 unstaged, 0 untracked
tracked files 607
latest audit  0052
```

### 1.1 Assistant-specific workspace file: found, then deleted by owner decision

A pre-existing, git-ignored assistant workspace file was found on disk during this
milestone's starting-state check. It was **not created by this milestone** and was
never tracked, so it was reported rather than removed unilaterally — deleting an
operator-owned file is the owner's call, not the agent's. **The owner then decided
explicitly: DELETE.** It was deleted before commit.

Recorded before deletion, so the decision is auditable:

```text
path            .claude/AGENTS.md
untracked       yes — git ls-files returned 0 entries for it
ignored by      .gitignore:108  (".claude/")
size            2,254 bytes
sha256          ab1edc2d18ab73304abb389f6d57a928cb2c8d613c7cf347258f7e81d1c5620f
mtime           2026-07-11 19:38   (predates Audits 0051-0053)
action          DELETED; the .claude/ directory was then empty and was removed too
```

No replacement was created, no ignore **exception** was added, and no
assistant-specific file is tracked. The `.gitignore` rule that prevents such a file
from ever being committed is deliberately **left in place** — it is the guard, and
removing it would weaken the very constraint this step enforces.

Post-deletion verification, both the filesystem and the Git index:

```text
                filesystem                       git ls-files
CLAUDE.md       absent                           absent
AGENTS.md       absent                           absent
.claude/        absent (directory removed)        absent
```

Baseline test suite before any change:

```text
1574 passed, 1 skipped
```

## 2. Architecture PDF hash

```text
sha256  0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b
        docs/MedNorm-VI_Architecture.pdf
```

Identical to the value recorded in Audits 0051 and 0052. The file was read, never
written, moved, renamed or regenerated.

E3 checkpoint, verified before and after every run in this milestone:

```text
path    checkpoint/s1_mention_full_training_v1/best.pt   (git-ignored)
sha256  a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
size    1,615,513,303 bytes
mtime   2026-07-27 00:53:41.047814400 +0700
```

`verify_checkpoint_unchanged()` was called at the end of each measurement run and
passed every time. The Drive backup MD5 `01fda557e3481303f3e654a03b6489a1` verified
in Audit 0052 still matches; no operation in this milestone writes to the checkpoint
directory.

## 3. Before / after runtime flow

Unchanged from Audit 0052 in shape — one runner, one lattice, one L4. Three stages
changed **behaviour**, one gained a **report**:

```text
BEFORE (Audit 0052)                        AFTER (Audit 0053)

L2  score >= activate  -> case active      L2  score >= activate  AND
                                               required_evidence satisfied
                                               -> case active
                                           L2  suppression recorded on
                                               NodeRouting.gate_reasons

L3  E1/E2 run on their routes              L3  unchanged execution, but route
    (eligibility invisible)                    eligibility, skips, proposals and
                                               suppression are REPORTED on
                                               PipelineResult.route_gate

L8  decode_expected_jaccard(...)           L8  decode_entities(...)
      candidates[:10]  unconditional             evidence tier + relative score
      name promised expected-Jaccard             band; per-candidate reason codes;
      search that did not exist                  safety bound 25 (a bound, not
                                                 the rule)
                                           L8  decode_expected_jaccard_calibrated
                                               RAISES CalibratedDecoderUnavailable

L9  candidate SYNTAX only; membership      L9  + validator/kb_membership.py:
    enforced inside linking/snapshot.py        independent, retriever-agnostic
    (the component whose bug the gate          membership + ontology checking
    exists to catch)                       L9  run_input_dir CALLS that gate on the
                                               serialized JSON before writing
                                               anything; a violation raises and no
                                               output/ or output.zip is produced
                                               (offered-set / P7 stays unwired —
                                                see 10.1 and 11)
```

Everything between L4 and L7 is byte-for-byte the same code path. The E3-only
measurement is therefore expected to be identical to Audit 0052's, and §15 confirms
it is (0.5559 both times) — which is the control that says route gating did not
perturb the neural arm.

## 4. Route-gating design and measured effect

### 4.1 The diagnosis, corrected

The milestone brief assumed E1/E2 were running on nodes they were not routed to.
**They were not.** `run_phase1b` already checked route tags before invoking either
parser. Measurement located the real defect one layer up: **the router was attaching
C2 (lab semi-structured) to ordinary narrative prose**, so E2 was correctly running
on a route it should never have been given.

The mechanism: C2's signals are individually weak and additive. A prose sentence
containing any numeral (`numeric_value`, 0.30) plus any faint structural or cue hint
could sum past `activate` without a single piece of evidence that the text is a lab
row. Gating E2 harder would have been the wrong fix — it would have hidden a routing
bug inside an expert.

### 4.2 The fix

A case may now declare **positive evidence it requires**. Reaching `activate` becomes
necessary but not sufficient:

```yaml
  C2:
    name: lab_semi_structured
    required_evidence: [numeric_key_value, lab_unit, reference_range, flag, lab_section]
```

Implemented in `case_router/signals.py` (`CaseSpec.required_evidence`,
`CaseSpec.evidence_satisfied`) and applied in `case_router/router.py`. It is
**config-driven**: no case is gated in code, and the before-state is reproducible by
neutralising one field (which is exactly how §15's before-arm was measured).

`numeric_value` is deliberately **excluded** from `required_evidence`. It fires on any
numeral and was present in **every** measured false positive; including it would have
made the gate a no-op.

One new detector was required. `structural.numeric_key_value` matches a *short key*,
a colon, and a **numeric** value:

```python
_NUMERIC_KEY_VALUE = re.compile(
    r"(?:^|[;\n])\s*([^:;\n]{1,40}?)\s*:\s*[<>~]?\s*\d+(?:[.,]\d+)?")
```

**Why it exists, honestly:** the first version of this gate (the four cue/section
signals alone) **broke five laboratory tests** — `lab-colon-01`,
`lab-decimal-point-01`, `lab-missing-unit-01`,
`test_decimal_point_value_is_not_split`, `test_missing_unit_still_yields_a_result`.
Cause: `PLT: 250` is a legitimate unit-less lab row, and spec §14.2 lists **"missing
units"** as a required synthesis variation, so a gate that needs a unit is a gate
that fails the spec. `numeric_key_value` is the signal that distinguishes a lab row
from prose *without* requiring a unit. It was verified to fire on all four lab forms
and on **none** of three prose forms.

Suppression is not silent. `NodeRouting.gate_reasons` records
`{case}:suppressed_no_required_evidence:score=…:requires=…` and
`{case}:below_activate:score=…`, and `route_gate.py` carries them into the pipeline
result prefixed by node id.

### 4.3 The five required diagnostics

`src/mednorm_vi/mention_factory/route_gate.py`, contract
`route-gate-diagnostics-v1`, surfaced as `PipelineResult.route_gate`:

| Required view | Field |
| --- | --- |
| which nodes each expert was eligible for | `eligible_nodes_by_expert` |
| which nodes each expert skipped | `skipped_nodes_by_expert` |
| how many proposals each expert produced | `proposals_by_expert` |
| which routes were active per node | `active_routes_by_node` |
| why a node was skipped or suppressed | `route_gate_reasons` |

`as_dict()` reports `contains_clinical_text: False`, and a test asserts no fixture
substring appears anywhere in the serialized diagnostics. Expert routes are declared
in one place (`EXPERT_ROUTES`: E1 → C1/C5/C3, E2 → C2) rather than re-derived.

### 4.4 Measured effect

200 governed validation documents (narrative DIAGNOSIS/SYMPTOM text):

```text
                                              before    after
E2 proposals                                      16        6     -62%
documents producing any E1/E2 proposal             8        4     -50%
E1/E2 false positives surviving L4                11        5     -55%
C2 routes suppressed for lack of evidence          —        7 nodes
```

Fixture recall, the thing a gate must not cost — **identical**, not merely close:

```text
fixture                        E1    E2      vs Audit 0052
synthetic_medication_list      19     0      identical
synthetic_laboratory            0    18      identical
synthetic_medical_document     13    14      identical
synthetic_hard_negatives        0     0      identical
```

## 5. ICD hierarchy implementation

**NOT ATTEMPTED.** Milestone item 3 was not implemented in this turn.

State, unchanged from the Audit-0052 review: `LocalIndex.graph` carries **14,534
nodes**, is materialized on disk, is loaded into memory on every run, and
`linking/icd10.py` **never reads it**. There is no hierarchy traversal and no
specificity controller (spec §9.3). Nothing in this audit improves ICD-10 candidate
quality. See §22, item 2.

## 6. RxNorm structured-linking implementation

**NOT ATTEMPTED.** Milestone item 4 was not implemented in this turn.

State: Audit 0052 preserved E1's structured medication parse all the way to L5
(`EntityHypothesis.components`), specifically so the RxNorm linker could use it.
**The linker still ignores it.** There is no `ingredient → component → SCD → SBD`
traversal over `index["graph"]` (**77,055 nodes**, loaded and unused), and no
strength / dose-form / release hard negatives. This is the highest-value open gap in
the repository — candidates are 40% of the metric and the required inputs already
exist. See §22, item 1.

## 7. L6 edge and consistency implementation

**NOT ATTEMPTED.** Milestone items 5 and 6 were not implemented in this turn.

Measured over the same 200 documents, L6 emits exactly three relations:

```text
has_assertion   333
has_candidate 4,000
supports        333
```

Still absent: `has_result`, `modified_by`, `in_section`, `overlaps`, `same_surface`,
`treats`. There is **no consistency-check module** and **no typed public contract**,
so nothing downstream consumes the graph — L7 and L8 both read L4 output directly.

Consequence for §15: **"consistency violations" is not a measurable quantity in this
repository today.** It is reported as unmeasurable, not as zero. Reporting zero would
be the same class of untruth this audit exists to remove from L8. See §22, item 4.

## 8. L7 deterministic escalation contract

**NOT ATTEMPTED.** Milestone item 7 was not implemented in this turn.

`confidence_cascade/` still does one thing: accept or reject on resolver score. The
constrained-decision **parsing** and schema validation in `llm/` exist and are tested;
the **locked option sets** an escalation must choose from (assertion label sets,
candidate sets, boundary options — spec §12.1, principle P7) do not exist. No
escalation can be raised, so none can be adjudicated. See §22, item 5.

## 9. L8 fallback replacement

**IMPLEMENTED.** This was the most misleading name in the repository.

What was there:

```python
def decode_expected_jaccard(...):
    ...
    candidates=tuple(link.candidates[:max_candidates])   # max_candidates = 10
```

A function named for spec §13's expected-Jaccard set search that performed an
**unconditional fixed top-10 slice** — precisely the "fixed top-K" §13.2 rules out.
A reader auditing the repository against the spec would have ticked §13 off.

What replaced it, contract `l8-deterministic-evidence-ranked-v1`:

```text
decode_entities(accepted, cascade, assertions, links)

  tiers          exact_alias > strong_lexical > weak_lexical
  rule           an exact alias wins ALONE;
                 otherwise keep the best tier down to a relative score floor
                 of 0.60 x that tier's top score
  safety bound   CANDIDATE_SAFETY_BOUND = 25   (a bound, NOT the selection rule)
  per candidate  CandidateDecision(reason=KEEP_EXACT | KEEP_WITHIN_TIER |
                 DROP_BELOW_BAND | DROP_WEAKER_TIER | DROP_SAFETY_BOUND |
                 DROP_DUPLICATE | DROP_TYPE_MISMATCH)
```

Three properties are asserted by tests rather than claimed here:

- `decode_expected_jaccard` **cannot be imported** — the name is gone, not aliased.
- selection is **not a fixed slice**: candidate-set size varies with evidence (§15).
- `decode_expected_jaccard_calibrated()` **raises `CalibratedDecoderUnavailable`**.
  It fails loudly instead of returning a plausible-looking set, because a silent
  stand-in for §13 is exactly what the old name was.
- `DecodedEntity.as_dict()` reports `performs_expected_jaccard_decoding: False`.

**This is not spec §13.** There is still no probability calibration, no expected-WER
boundary utility, no per-label assertion thresholds and no wrong-type-risk retention
utility. The honest blocker is S6 calibration: nothing downstream of L4 produces a
calibrated probability, so there is nothing to take an expectation over. The change
here is that the code now says so.

## 10. L9 KB-membership enforcement

**IMPLEMENTED.** `src/mednorm_vi/validator/kb_membership.py`, contract
`l9-kb-membership-v1`.

Spec §16 requires stopping a file when "a code is absent from the frozen KB
snapshot". Before this change L9 checked candidate **syntax** only (an RxCUI must be
numeric; ICD-10 was an opaque string) and membership was enforced **inside
`linking/snapshot.py`** — the wrong place for a final gate, because the linker is the
component whose bug the gate exists to catch, and any future retriever, reranker or
LLM selector could introduce a code without passing through it.

The validator is therefore deliberately **model- and retriever-independent**: it
imports no linker, no index builder and no model. A test asserts this by inspecting
the module's imports, so the independence cannot rot silently.

Six negative cases, each verified:

| Case | Issue code |
| --- | --- |
| code absent from the required snapshot | `kb.candidate_not_in_snapshot` |
| validated against the wrong ontology's index | `kb.wrong_snapshot` |
| candidates emitted on a type that takes none (§7.3) | `kb.candidates_on_type_without_ontology` |
| the same code twice | `kb.duplicate_candidate` |
| snapshot not supplied → **cannot validate**, never "pass" | `kb.snapshot_unavailable` |
| code exists but was **not offered** for this mention (P7) | `kb.candidate_not_offered` |

Two things it deliberately does **not** do. It never repairs, reorders or drops a
code — spec §16's "never silently repair output" means the caller stops the file.
And it does not pretend to check candidate *ordering* determinism, which is a
property of the decoder across runs, not of one list; that is asserted by running the
pipeline twice in tests.

Measured over 200 documents against the live locked snapshots
(`icd10-vi-local-c6b9568cf35b413a`, `rxnorm-local-75a100c0b70b67d0`): 333 entities,
3,390 candidates, **0 violations**.

### 10.1 The gate had no caller — now it is on the canonical packaging path

Stated plainly, because it undercut §10's original claim: as first written, this
validator **was never invoked**. `grep` for `validate_document_candidates` under
`inference/` returned nothing. It was reachable only from tests. A gate that nothing
calls does not gate; §10 described a guarantee the runtime did not provide.

Wired in the correction pass:

```text
inference/pipeline.py
  _gate_kb_membership(payloads, config)
    loads the configured snapshots ONCE per run
    validates the SERIALIZED organizer JSON — the payload the organizer receives,
      so the serializer is inside the checked region, not outside it
    raises KbMembershipViolation, listing "{file}:{code}" per violation
  called from run_input_dir BEFORE write_output_directory
```

Placement is deliberate: the gate runs **before anything is written**, so a violating
run leaves neither an `output/` directory nor an `output.zip` for someone to submit by
mistake. `KbMembershipViolation` lives in `validator/kb_membership.py` but is raised
only by the caller — the validator still reports and never decides, per §16's "never
silently repair".

**One false positive was fixed in the same pass.** An entity emitting `candidates: []`
was reported as `kb.snapshot_unavailable` when no KB was configured, which would have
rejected a legitimate `deterministic` run with no indices. An empty candidate list
makes no membership claim, so it cannot fail one; the validator now returns early. The
codes-present rule is **unchanged** — `test_a_missing_snapshot_fails_rather_than_passing`
still holds it, and `test_an_entity_with_no_candidates_needs_no_snapshot` pins the
other half. No validation rule was weakened.

`MembershipSource`'s two members became read-only properties rather than plain
annotations, because a plain annotation declares a *mutable* attribute that the frozen
`LocalIndex` dataclass cannot satisfy — mypy caught this the moment the Protocol had a
real production caller.

**What the wired gate does not yet check:** `offered_codes`. The P7 offered-set rule
is implemented and unit-tested, but the canonical path cannot supply it, because
nothing records which codes were offered per mention — that is the run manifest
(§11, not attempted). So the wired gate enforces *membership and ontology*; P7
remains available and unwired.

## 11. Inference run manifest

**NOT ATTEMPTED.** Milestone item 10 was not implemented in this turn.

The building blocks now exist and are each designed to be leakage-free —
`RouteGateDiagnostics.as_dict()`, `MembershipReport.as_dict()` and
`DecodedEntity.as_dict()` all report `contains_clinical_text: False` — but nothing
assembles them into a per-run manifest, and no run writes one.

The concrete consequence, after §10.1: L9 now **detects a membership violation and
stops the run**, and there is still nothing that **records the stop** to disk. The
violation reaches the caller as an exception carrying `{file}:{code}` pairs, which is
enough to fail closed but not an audit trail. It also means the gate cannot yet
enforce spec P7's offered-set rule, because "what was offered for this mention" is
exactly the state a manifest would carry. See §22, item 6.

## 12. Reproducibility files

**IMPLEMENTED, NOT BUILT.** Spec §19 and Appendix A require a one-command offline
rebuild; the repository had neither a `Dockerfile` nor a lock file.

`requirements.lock` — every version read from `importlib.metadata` **in the
environment that produced these measurements**, not chosen from a changelog:

```text
interpreter  CPython 3.14.5
core         PyYAML==6.0.3
E3 stack     torch==2.13.0+cu126, transformers==4.57.6, tokenizers==0.22.2,
             safetensors==0.8.0, huggingface-hub==0.36.2, numpy==2.5.1,
             sentencepiece==0.2.2
segmentation py_vncorenlp==0.1.4
transitive   filelock, packaging, regex, requests, tqdm  (enumerated so
                                                          --no-deps is safe)
verification pytest, ruff, mypy, types-PyYAML
install      python -m pip install --no-deps -r requirements.lock
```

Hashes are deliberately **not** pinned: the torch wheel is a CUDA 12.6 build and
pinning its hash would tie the lock to one CUDA minor. `--require-hashes` is the
right hardening step once the submission CUDA target is fixed.

`Dockerfile`:

```text
base            python:3.14.5-slim-bookworm
system packages openjdk-17-jre-headless ONLY (py_vncorenlp needs a JRE)
offline         HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
determinism     PYTHONHASHSEED=0  OMP_NUM_THREADS=1
copied          pyproject.toml, src/, configs/, schemas/   — no data, no weights
NOT baked in    model weights, governed corpus, KB payloads (RxNorm is
                UMLS-licensed and redistribution-restricted)
mounts          /models/e3  /models/hf  /models/vncorenlp  /kb/indices
                /input  /output
user            non-root  mednorm  uid 10001
smoke test      imports mednorm_vi + inference.pipeline + validator.kb_membership
                with no model and no network
entrypoint      exactly ONE: python -m mednorm_vi.inference.cli
```

**The image has not been built.** The milestone forbids it, and the example
`docker build` / `docker run --network=none` lines in the file's header are
documentation. A static test asserts the file's contract properties (single
entrypoint, offline env, non-root, no weight COPY, lock-based install) without
invoking Docker.

## 13. Old L4 migration and deletion

**NOT ATTEMPTED.** Milestone item 12 was not implemented in this turn.

`src/mednorm_vi/resolution/resolver.py` still exists. It is **off the canonical path**
since Audit 0052 — the runner calls `resolution.canonical` only — but its remaining
per-type boundary policy has not been proven equivalent to the canonical L4, no
equivalence tests were written, and the file was not deleted. Two L4 implementations
must not coexist indefinitely; deleting it before proving coverage would risk silently
losing behaviour, which is why it was not deleted blind. See §22, item 7.

## 14. Tests and static checks

New module `tests/unit/test_route_gating_and_l8_l9.py` — **51 tests**, five sections:

```text
A  route gating (17)   numeric_key_value fires on real lab rows and NOT on prose;
                       fixture recall preserved per fixture; multi-label routes
                       survive gating; suppressed routes are recorded with score
                       and requirement
B  diagnostics (8)     all five required views present; no clinical text in the
                       serialized diagnostics; expert routes declared once
C  L8 (13)             the misleading name is gone; selection is not a fixed
                       slice; the maximum is a safety bound not the rule; exact
                       alias wins alone; reason codes; run-to-run determinism;
                       the calibrated path raises
D  L9 (10)             all six negative cases; membership validation is
                       independent of the retriever (import inspection);
                       + the three added in the correction pass, below
E  reproducibility (3) Dockerfile / requirements.lock static contract
```

The three L9 tests added when the gate was wired (§10.1). The first is the
integration test the correction pass required:

```text
test_packaging_stops_when_an_injected_code_is_outside_the_snapshot
    tests/unit/test_route_gating_and_l8_l9.py

    1. control: the medication fixture packages successfully, producing output.zip
       (a document yielding no candidate could not exercise the gate at all, which
       is why the real fixture is used rather than a one-line input)
    2. asserts the forged RxCUI "999999999999" is genuinely ABSENT from the locked
       snapshot, so the test cannot pass for the wrong reason
    3. injects it UPSTREAM by monkeypatching link_rxnorm — the linker is the
       component whose bug this gate exists to catch
    4. drives the real run_input_dir -> serialize -> validate -> package path
    5. asserts KbMembershipViolation carries kb.candidate_not_in_snapshot
    6. asserts NEITHER output.zip NOR the output/ directory was written

test_the_canonical_packaging_path_actually_invokes_the_membership_gate
    records every call the packaging path makes to validate_document_candidates and
    asserts it validated document "1" then "2". This is the regression guard for the
    defect found in the correction pass: a test that only calls the validator
    directly would not have noticed that nothing else did.

test_an_entity_with_no_candidates_needs_no_snapshot
    pins the false-positive fix without weakening the codes-present rule.
```

Full suite and static checks, run at the end:

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q
    1625 passed, 1 skipped in 263.43s        (baseline was 1574 passed, 1 skipped;
                                              the 1 skip is pyarrow, absent locally)
ruff check .                    All checks passed!
ruff check notebooks            All checks passed!
env PYTHONPATH=src .venv/bin/python -m mypy
    Success: no issues found in 270 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src     clean
git diff --check                                             clean
```

Defects found and fixed by these checks during the work, reported rather than
smoothed over: six lint errors (three `I001` import-order, two `F401` unused imports,
one `F841` dead store), and two mypy errors that appeared only once the L9 Protocol
had a real production caller (`LocalIndex` is frozen, so it could not satisfy a
Protocol declaring mutable attributes — §10.1). The counts above are post-fix.

Canonical fixture validation, both modes, plus the fail-closed check:

```text
readiness[deterministic]  READY       E1 + E2
readiness[specialist]     READY       E1 + E2 + E3
readiness[full]           NOT_READY   12 blockers (11 missing checkpoints +
                                      enabled_feature_missing_checkpoint on E3's
                                      full_v1 path)
run_document(mode="full") -> RuntimeError    full mode fails closed, as required

mode=deterministic   4 fixtures, 11.1s
mode=specialist      4 fixtures, 41.5s   (E3 contributes 3 spans on the mixed doc)
every emitted offset re-verified: original_text[start:end] == text, 0 violations
```

## 15. Bounded validation evidence

**Split identity, resolved by digest, never by name** (`internal_test` never opened):

```text
data/derived/training_corpora/mednorm_vi_training_v1/splits/validation.jsonl
sha256 ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103
bounded to the first 200 of 1,045 rows   (same bound as Audit 0052, for comparability)
```

Three arms, all through the canonical runner path. The before-arm is the real
pre-gating behaviour, reproduced by neutralising `CaseSpec.evidence_satisfied` in
memory — **no file was edited to obtain it**.

### 15.1 Proposal counts, lattice, L4

```text
arm                            proposals by expert            nodes  merges
before gating (E1+E2)          E2 16                             15       0
after gating (E1+E2)           E2  6                              6       0
after gating (E1+E2+E3)        E2  6   E3 328                    334       0
E3 only (reference)            E3 328                            328       0

arm                            L4 accepted  rejected  unresolved
before gating (E1+E2)                   11         0           4
after gating (E1+E2)                     5         0           1
after gating (E1+E2+E3)                333         0           1
E3 only (reference)                    328         0           0
```

Zero coordinate merges on this corpus is expected, not a defect: coordinate-identity
merging is a medication-list phenomenon (`Paracetamol 500mg và 650mg`) and these are
narrative documents.

### 15.2 Route eligibility and skips

```text
E1  eligible 230 nodes    skipped   5 nodes
E2  eligible   6 nodes    skipped 229 nodes
C2 suppressed for lack of required evidence:  7 nodes
```

E2's eligibility collapsing from "most of the corpus" to 6 nodes is the fix working:
this is narrative text and almost none of it is a lab row.

### 15.3 Exact P/R/F1 (character offsets **and** type)

```text
arm                       P        R       F1     TP    FP    FN
before gating (E1+E2) 0.0000   0.0000   0.0000     0    11   406
after  gating (E1+E2) 0.0000   0.0000   0.0000     0     5   406
E3 only               0.6220   0.5025   0.5559   204   124   202
gated E1+E2 + E3      0.6126   0.5025   0.5521   204   130   202
```

Per type, gated E1+E2+E3:

```text
type            P        R       F1     TP    FP    FN
DIAGNOSIS   0.6600   0.4783   0.5546    132    68   144
SYMPTOM     0.5625   0.5538   0.5581     72    56    58
TEST_NAME   0.0000   0.0000   0.0000      0     1     0
TEST_RESULT 0.0000   0.0000   0.0000      0     4     0
```

Error categories, gated E1+E2+E3: `exact_match` 204, `missed` 124, `spurious` 51,
`right_boundary` 45, `left_boundary` 27, `both_boundary` 3, `wrong_type` 3,
`duplicate_overlap` 0.

Read this honestly, in three parts.

1. **E1/E2 scoring 0.0000 is correct, not a regression.** This corpus is narrative
   DIAGNOSIS/SYMPTOM prose; E1 and E2 are medication and laboratory parsers. The only
   improvement available to them here is *fewer false positives*, and that is what
   gating delivered: **11 → 5**.
2. **The E3-only arm is bit-identical to Audit 0052 (0.5559).** That is the control
   proving route gating did not perturb the neural path.
3. **Adding gated E1+E2 to E3 still costs F1** (0.5521 against 0.5559). Gating
   narrowed the Audit-0052 gap (0.5477 → 0.5521) but did not close it, so
   `deterministic` and `specialist` remain separate modes rather than one merged
   profile. The residual cost is the 5 surviving TEST_NAME/TEST_RESULT false
   positives — the two types with **zero** governed supervision.

**This is not the organizer score.** It is exact mention span + type on 200 governed
validation rows. The candidate (40%) and assertion (30%) halves of the organizer
metric are not measured here and have never been measured.

### 15.4 Candidate Recall@K — **not measurable, not reported**

The milestone asked for Recall@K "where gold or governed weak evidence exists". It
does not exist. All 406 gold entities in these rows carry:

```text
fields present   start, end, text, target_type, source_label,
                 mapping_status, mapping_confidence, provenance, quality_flags
mapping_status   MAP_EXACT   406 / 406
target_type      DIAGNOSIS 276, SYMPTOM 130
ontology codes   NONE — no candidate field, no ICD-10 code, no RxCUI,
                 no weak-evidence code field
```

The governed corpus carries **zero ontology codes**, so there is no gold set to
compute recall against and no weak substitute. Reporting a number here would require
inventing one. This is a corpus gap, and it blocks measuring the 40% of the metric
that §5/§6 of this audit did not implement either.

### 15.5 Assertion regression

```text
entities receiving any assertion label      13 of 333
entities receiving ALL THREE labels          0
entities flagged assertion_uncertain        35
```

Zero all-three entities holds the Audit-0052 fix (the defect assigned all three
labels to 10 entities in one fixture). The 32 assertion regression tests still pass.
**No assertion metric is computable**: the governed corpus has zero assertion
supervision (Audit 0042), so this is a regression check, not a score.

### 15.6 Consistency violations — **not measurable**

There is no L6 consistency-check module (§7). Reported as unmeasurable rather than
as zero.

### 15.7 Decoder candidate-set-size distribution

Where the old decoder returned an unconditional 10:

```text
size   0:133   1:7   2:4   3:4   4:1   5:4   6:1   7:3   8:1   9:6  10:2
      11:4  12:4  13:1  14:1  15:1  16:2  17:2  18:3  19:4  20:145

candidates offered by L6   4,000
candidates emitted         3,390        (610 dropped by tier / band / duplicate)
per-candidate tiers        exact_alias 10, strong_lexical 3,859, weak_lexical 131
retention reasons          resolver_accepted 333
```

Two features of this distribution deserve stating plainly rather than being spun.
The **133 empty sets** are SYMPTOM and TEST_\* entities, which take no candidates at
all (§7.3) — correct behaviour. The **145 sets of exactly 20** are the *retriever's*
top-K cap, not a decoder rule: those mentions had 20 same-tier candidates all inside
the score band, so the decoder kept them all. That is a signal the **lexical
retriever is undiscriminating** — which is §5/§6's unimplemented work, not something
L8 can fix by truncating harder.

### 15.8 Runtime and peak memory

```text
E3 checkpoint load (once)                       28.2 s
3-arm span/type run, 200 docs                    7.6 s   (0.038 s/doc/3 arms)
L5-L9 run with linking, 200 docs                14.7 s   (0.074 s/doc)
peak RSS                                    2.68 GiB
```

CPU forward passes throughout; no GPU was used and no model was downloaded.

## 16. Changed files

Verified against real Git output, not assumed.

**Modified (10)** — `git diff --stat`:

```text
 configs/case_router/signals_v1.yaml          |  24 ++-
 docs/architecture/ACTIVE_RUNTIME_MANIFEST.md | 147 ++++++++++---
 docs/audits/README.md                        |   1 +
 src/mednorm_vi/case_router/models.py         |   4 +
 src/mednorm_vi/case_router/router.py         |  29 ++-
 src/mednorm_vi/case_router/signals.py        |  41 ++++
 src/mednorm_vi/inference/pipeline.py         |  62 +++++-
 src/mednorm_vi/metric_decoder/__init__.py    |  30 ++-
 src/mednorm_vi/metric_decoder/decoder.py     | 308 ++++++++++++++++++++++++---
 src/mednorm_vi/validator/__init__.py         |  12 ++
 10 files changed, 595 insertions(+), 63 deletions(-)
```

`inference/pipeline.py` grew from +21 to +62 in the correction pass: that delta is
`_gate_kb_membership` and its wiring (§10.1).

`docs/audits/README.md` is a one-line addition: this audit's index entry. It is
listed here because it **is** modified, and Audit 0052 §13 was corrected for exactly
this omission.

**Added (6)** — untracked, therefore **absent from `git diff --stat`** by
construction; line counts are `wc -l`:

```text
 Dockerfile                                        99
 requirements.lock                                 55
 src/mednorm_vi/mention_factory/route_gate.py     119
 src/mednorm_vi/validator/kb_membership.py        294
 tests/unit/test_route_gating_and_l8_l9.py        593
 docs/audits/0053-…-l9-kb-membership.md         1,037   this file
```

`kb_membership.py` 263 → 294 and the test module 480 → 593 are the correction pass:
`KbMembershipViolation`, the empty-candidates early return, the Protocol property fix,
and the three new tests.

**Deleted: 0. Renamed: 0.**

Total: **10 modified + 6 added = 16 paths.** Untracked files do not appear in
`git diff --stat` at all, which is why the two lists are gathered from different
commands (`git diff --stat` and `git ls-files --others --exclude-standard`) rather
than from one.

## 17. Remaining pretrained-model gaps

| Gap | State |
| --- | --- |
| E5 XLM-R MRC-NER | contract + trainer exist; **task head randomly initialized**; must never run untrained |
| E6 GLiNER | **no local weights**; strict adapter fails closed |
| E7 Qwen proposer | **no local weights**; loader fails closed |
| S3 dense retrieval (ICD + RxNorm) | no embedder weights; retrieval is lexical only |
| S4 cross-encoder reranker | none; depends on S3 |
| S5 Qwen LoRA critic/adjudicator | no local Qwen weights; **L7 has no model stage at all** |
| S6 calibration meta-model | none — and this is what blocks real §13 decoding |

Every loader is lazy and `local_files_only=True`. **No model was downloaded in this
milestone.** `full` mode still fails closed with 12 named blockers (§14).

## 18. Remaining training / fine-tuning gaps

| Stage | State | Blocker |
| --- | --- | --- |
| S0 domain adaptation | `SCAFFOLD_ONLY` | notebook is a design draft |
| **S1 mention** | **EXECUTED** | complete — E3 is the only trained model in the repository |
| S2 assertion | `IMPLEMENTED_NOT_RUN` | **zero assertion supervision** in the governed corpus |
| S3 retrieval | `SCAFFOLD_ONLY` | no embedder; and **zero ontology codes** in the corpus (§15.4) |
| S4 reranking | `SCAFFOLD_ONLY` | depends on S3 |
| S5 Qwen LoRA | `NOT_STARTED` | no local weights |
| S6 calibration | `NOT_STARTED` | needs out-of-fold predictions no stage produces |

**No training or fine-tuning ran in this milestone.** TEST_NAME and TEST_RESULT still
have zero governed supervision, which is visibly the source of the residual
false positives in §15.3.

## 19. Genuine blockers

1. **The governed corpus has no ontology codes** (§15.4). Candidate quality is 40% of
   the organizer metric and **cannot be measured at all** — not by Recall@K, not by
   Jaccard. Any candidate work after this point is unvalidatable until a code-bearing
   evaluation set exists. This is the single most consequential blocker in the
   repository and it is a *data* blocker, not a code one.
2. **No calibrated probabilities anywhere below L4.** Spec §13 is an expectation over
   probabilities; there are none. S6 is the prerequisite, and S6 needs out-of-fold
   predictions no stage produces. L8's honest deterministic decoder is the correct
   interim state, not a stepping stone toward §13.
3. **Zero assertion supervision** (Audit 0042). 30% of the metric is held by
   regression tests against constructed cases only.
4. **Route accuracy has never been measured against route gold**, because none
   exists. §4's gate is validated by its *effect* (false positives −62% at unchanged
   fixture recall), which is real evidence but is not routing accuracy.
5. **Two L4 implementations coexist** (§13). Not urgent — one is off the canonical
   path — but it must not persist.
6. **The Docker image has never been built**, so the Appendix A one-command rebuild
   is authored and unproven.
7. **Seven of this milestone's thirteen implementation tasks were not attempted**
   (prompt sections 2-14; sections 1 and 15 were verification and this audit), so its
   acceptance criteria are not met (§20).

## 20. Safe-to-commit verdict

```text
VERDICT: SAFE_TO_COMMIT  (as a PARTIAL milestone, explicitly labelled as such)

MILESTONE 2 ACCEPTANCE CRITERIA: NOT MET

met:
  E1/E2 are route-gated                                          YES (§4)
  valid medication/lab fixture recall preserved                  YES, exactly (§4.4)
  narrative E1/E2 false positives reduced                        YES, 11 -> 5 (§4.4)
  L8 no longer pretends to do expected-Jaccard decoding          YES (§9)
  L8 no longer uses unconditional fixed top-10 selection         YES (§9, §15.7)
  L9 independently validates ontology membership                 YES (§10)
  ...and the canonical packaging path INVOKES that gate           YES (§10.1) —
                                                                 wired in the
                                                                 correction pass;
                                                                 it had no caller
  a violating run writes no output/ and no output.zip            YES (§10.1)
  the assistant-specific workspace file is gone                  YES (§1.1)
  milestone task accounting adds up                              YES (header)
  Dockerfile and requirements.lock exist                         YES, unbuilt (§12)
  deterministic and specialist modes remain operational          YES (§14)
  full mode still fails closed                                   YES, 12 blockers
  no model downloaded                                            YES
  no training or fine-tuning ran                                 YES
  no internal_test access                                        YES
  no organizer inference, no output.zip                          YES
  tests and static checks pass                                   YES (§14)
  Audit 0053 contains real evidence                              YES (§15)
  no commit or push occurred                                     YES

NOT met:
  ICD hierarchy and specificity consumed by the linker           NO  (§5)
  structured E1 medication evidence consumed by RxNorm linking   NO  (§6)
  RxNorm graph traversal functional                              NO  (§6)
  all deterministic L6 edges and consistency outputs             NO  (§7)
  L7 complete locked-option escalation contracts                 NO  (§8)
  inference runs create deterministic no-leakage manifests       NO  (§11)
  obsolete non-canonical resolver removed after migration        NO  (§13)
```

The change is safe to commit **because every claim it makes is true and every gap is
named**. It is not a complete milestone and the commit message must not imply
otherwise.

Safety checks before staging: working tree contains only the 16 paths in §16; no
tracked file was deleted; the architecture PDF is unmodified (§2); the E3 checkpoint
is unmodified (§2); no assistant control file is tracked (§1); `git diff --check`
clean.

## 21. Exact explicit commit commands

To be run by the repository owner. **Nothing below was executed by this audit.**
`git add -A`, `git add .` and `git commit -a` are deliberately not used.

```bash
cd /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI

# 1. Re-verify the preflight
env PYTHONPATH=src .venv/bin/python -m pytest -q
ruff check . && ruff check notebooks
env PYTHONPATH=src .venv/bin/python -m mypy
env PYTHONPATH=src .venv/bin/python -m compileall -q src
git diff --check

# 2. Stage exactly the 16 paths from section 16 (the README index line is already
#    written; it is staged below with everything else)
git add configs/case_router/signals_v1.yaml
git add docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
git add src/mednorm_vi/case_router/models.py
git add src/mednorm_vi/case_router/router.py
git add src/mednorm_vi/case_router/signals.py
git add src/mednorm_vi/inference/pipeline.py
git add src/mednorm_vi/metric_decoder/__init__.py
git add src/mednorm_vi/metric_decoder/decoder.py
git add src/mednorm_vi/validator/__init__.py
git add Dockerfile
git add requirements.lock
git add src/mednorm_vi/mention_factory/route_gate.py
git add src/mednorm_vi/validator/kb_membership.py
git add tests/unit/test_route_gating_and_l8_l9.py
git add docs/audits/0053-route-gating-honest-l8-and-l9-kb-membership.md
git add docs/audits/README.md

# 3. Confirm what is staged before committing
git status --porcelain
git diff --cached --stat

# 4. One commit
git commit -F - <<'MSG'
feat: route-gate the deterministic experts and make L8/L9 honest

Partial Milestone 2: 6 of the 13 implementation tasks (prompt sections 2-14;
sections 1 and 15 were verification and the audit). Route gating, the L8
replacement, L9 KB membership and the reproducibility files are done. ICD
hierarchy, RxNorm structured linking, L6 edges/consistency, the L7 escalation
contract, the run manifest and the old-L4 deletion are NOT done. Acceptance
criteria not met.

L2/L3 - the real defect was the router, not the experts: E1/E2 were already
route-gated, but C2 attached to narrative prose because its weak signals sum
additively. Cases may now declare `required_evidence`, so reaching `activate`
is necessary but not sufficient, and suppression is recorded on
NodeRouting.gate_reasons. A new `numeric_key_value` detector keeps unit-less
lab rows (`PLT: 250`) eligible, which spec 14.2 requires. New
route_gate.py reports eligibility, skips, proposals and reasons with no
clinical text. Narrative false positives 11 -> 5 with fixture recall
unchanged exactly.

L8 - `decode_expected_jaccard` applied an unconditional top-10 under a name
promising spec-13 set search. Replaced by `decode_entities`: evidence tiers,
a relative score band, per-candidate reason codes, and a safety bound that is
a bound rather than the rule. The calibrated path raises instead of
substituting. Candidate-set size now varies 0..20 across 21 distinct values.

L9 - kb_membership.py validates ontology membership independently of the
linker and of any model, covering absent codes, wrong ontology, candidates on
types that take none, duplicates, missing snapshots and codes never offered
for that mention (P7). 0 violations on 200 governed rows. The gate is wired
into run_input_dir ahead of any write, so an injected out-of-snapshot code
stops packaging and leaves neither output/ nor output.zip; an integration test
drives that path. P7's offered-set check stays unwired until a run manifest
records what was offered per mention.

Reproducibility - Dockerfile and requirements.lock authored; the image is NOT
built and no organizer inference ran.

Evidence: Audit 0053. 1625 passed / 1 skipped; ruff, mypy strict, compileall
clean. No model downloaded, no training, no internal_test, no output.zip.
The pre-existing ignored .claude/AGENTS.md was deleted by owner decision.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG

# 5. Verify, do not push automatically
git log -1 --stat
git status
```

## 22. Recommended next milestone

**MAKE CANDIDATE LINKING REAL AND MEASURABLE.** In this order, because the order is
forced by dependency rather than by preference:

1. **A code-bearing evaluation set** (§19.1). Without it, everything below is built
   blind. Even a few hundred human-checked ICD-10 codes over existing governed
   DIAGNOSIS mentions would turn 40% of the metric from unmeasurable into measurable.
   This is the first item because it gates the value of the rest.
2. **RxNorm structured linking** (item 4, §6) — consume `EntityHypothesis.components`,
   traverse `ingredient → component → SCD → SBD` over the 77,055-node graph, add
   strength/dose-form/release hard negatives. Highest value per unit of work: the
   inputs exist, are loaded, and are ignored.
3. **ICD-10 hierarchy + specificity controller** (item 3, §5) — the 14,534-node graph
   is likewise loaded and ignored. §15.7's 145 saturated candidate sets are the
   symptom.
4. **L6 edges, the eight consistency checks, and a typed contract L7/L8 consume**
   (items 5-6, §7) — until this lands, L6 is write-only and "consistency violations"
   stays unmeasurable.
5. **L7 locked-option escalation contract** (item 7, §8).
6. **The inference run manifest** (item 10, §11) — small, and it closes the loop on
   L9's new gate.
7. **Migrate then delete `resolution/resolver.py`** (item 12, §13) — equivalence
   tests first, deletion second.
8. **Build and verify the Docker image** — authored here, unproven.

Explicitly **not** next, and for the same reason spec §20 gives: the L7 LLM cascade,
broad threshold optimization, and any organizer submission. Nothing should be tuned
against a metric that 40% of is currently unmeasurable.

---

**Audit 0053 ends. No commit. No push. Seven of the thirteen implementation tasks
were not attempted; the milestone's acceptance criteria are not met.**
