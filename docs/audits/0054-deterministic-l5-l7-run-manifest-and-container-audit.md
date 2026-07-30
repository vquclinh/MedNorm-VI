# Audit 0054 — Deterministic L5–L7, the Inference Run Manifest, and the Container Audit

- **Date:** 2026-07-30
- **Author:** Claude (AI agent), for human review
- **Change type:** Completion of the deterministic L5–L7 architecture, plus the run
  manifest, the code-linking evaluation contract, and a real (failed) container build.
  **No commit. No push.**
- **Status:** `PARTIAL_MILESTONE`
- **Verdict:** `ACCEPTANCE_CRITERIA_NOT_MET` — two criteria are unmet: the obsolete
  resolver is characterized but **not** migrated or deleted (§11–§12), and the Docker
  build is blocked by the environment (§14). Everything else in the criteria list is
  met and evidenced. §22 is the full table.
- **Constraints held:** no model downloaded; no training or fine-tuning; no calibration
  fitted; no broad threshold search; `internal_test` never opened; no organizer
  inference; **no organizer `output.zip`**; no leaderboard submission; no governed data,
  weights or restricted KB payloads added to Git; no candidate-quality metric claimed;
  no synthetic or model-generated ontology labels created; the architecture PDF
  unmodified; no assistant control file created or tracked.
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1, SHA-256
  `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b` — read in full,
  **unchanged** (§2).

---

## 1. Starting Git state

```text
git branch --show-current   main
git log -1 --oneline        8c9e1a7 feat: route-gate deterministic experts and harden L8/L9
                            (committed by the repository owner)
origin/main                 8c9e1a7500330a1d763a5b9c3fc98ac6ddf2b7af — in sync
git status --short -uall    (empty — clean tree)
tracked files               607
```

Audit 0053 **is** committed: `8c9e1a7` contains
`docs/audits/0053-route-gating-honest-l8-and-l9-kb-membership.md`, `Dockerfile`,
`requirements.lock`, `src/mednorm_vi/mention_factory/route_gate.py` and
`src/mednorm_vi/validator/kb_membership.py`.

Assistant control files, filesystem **and** index:

```text
CLAUDE.md   absent / absent
AGENTS.md   absent / absent
.claude/    absent / absent   (deleted by owner decision in Audit 0053 §1.1)
```

Mode behaviour before editing, and unchanged after (§16):

```text
readiness[deterministic]  READY      E1 + E2
readiness[specialist]     READY      E1 + E2 + E3
readiness[full]           NOT_READY  12 blockers
run_document(mode="full") -> RuntimeError    fails closed
```

Audit 0053's three deliverables verified present and active: route gating
(`CaseSpec.required_evidence` + `numeric_key_value`), the honest L8
(`decode_entities`; `decode_expected_jaccard` unimportable), and the L9 KB-membership
gate wired into `run_input_dir`.

**Baseline suite before any change: `1625 passed, 1 skipped`.** No unexplained changes
in the tree.

## 2. Architecture PDF and E3 checkpoint verification

```text
PDF     sha256 0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b
E3      sha256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
        size   1,615,513,303 bytes
        mtime  2026-07-27 00:53:41.047814400 +0700
```

Both identical to Audits 0051–0053. `verify_checkpoint_unchanged()` was called at the
end of every measurement run in this audit and passed each time. No pretrained weights
other than the existing E3 artifact were materialized: the only model download path is
`local_files_only=True`, and no network fetch occurred.

## 3. Actual KB graph schema findings

**This section exists because the milestone forbade assuming field names from the spec,
and that instruction was load-bearing.** Both graphs turned out to be weaker artifacts
than spec §9.3/§10.2 assume, and the implementations in §4 and §5 are shaped by that.

### 3.1 ICD-10 VI (TT06-2026)

```text
file       indices/icd10_vi/tt06-2026/index.json           (generated, git-ignored)
manifest   data/manifests/icd10-vi-tt06-2026-official.yaml
snapshot   icd10-vi-local-c6b9568cf35b413a
index_id   icd10-index-4202a320e8be2978
index hash 4202a320e8be297852fcb96c1ec65f0614731285b36260bf957a4694a9cffdd5
source hash 7d975de0f8b9bd16ded36e8abd29a3f5b4c4c1864183ec7593943a8fd59dc7d4
builder    kb-index-v1;  normalization casefold+NFKC+accent_optional, ngram_n 3

node schema      {concept_id, canonical_name, aliases[], metadata}
metadata         {block, chapter, dotted_code, specificity}
edge schema      NONE — graph is dict[concept_id, list[concept_id]]
relation labels  NONE
direction        NOT EXPLICIT — 0 of 25,936 sampled edges lack a reverse
records          15,308     graph nodes 14,534
code lengths     3 chars 2,011 | 4 chars 9,790 | 5 chars 3,507
specificity      "0" 2,011 | "1" 9,790 | "2" 3,507   ==  len(code) - 3, always
aliases          0 records carry any alias (0.0%)
exact-name collisions   1,340 normalized names map to >1 code
```

Malformed / missing records, all reported rather than assumed away:

```text
records absent from the graph entirely          774   (e.g. J189 "Viêm phổi, không xác định")
4+char codes whose 3-char prefix is NOT a record 410   (A390 exists, A39 does not)
3-char codes with no children                    454
graph nodes absent from records                    0
```

Loader behaviour before this audit: `load_index` materialized `graph` into
`LocalIndex.graph` on **every run**, and `linking/icd10.py` never read it — recorded as
a gap by Audits 0051, 0052 and 0053.

Two design consequences. **Direction must be reconstructed from code length** (a shorter
neighbour is an ancestor), because the artifact does not carry it. And
**`metadata.specificity` is structural depth, not clinical precision** — it equals
`len(code) - 3` for every record, so using it to rank would be exactly the error spec
§9.3 warns against. It is reported for provenance and excluded from ranking.

A data-quality observation for KB intake, not for the linker: some `canonical_name`
values are visibly truncated PDF fragments (`'- Bao gồm: viêm'`, `'(chương xx) để'`).
That limits lexical retrieval quality and is worth fixing upstream.

### 3.2 RxNorm (Prescribable 2026-07-06)

```text
file       indices/rxnorm/prescribable-2026-07-06/index.json   (generated, git-ignored)
manifest   data/manifests/rxnorm-prescribable-2026-07-06.yaml
snapshot   rxnorm-local-75a100c0b70b67d0
index_id   rxnorm-index-ecaf487b86e27f05
index hash ecaf487b86e27f0575ddba1904147fce9f214d148887e1519025ea34d1b16b10
builder    kb-index-v1;  attributes_seen 3,340,763;  relations_seen 2,563,978

node schema      {concept_id, canonical_name, aliases[], metadata}
metadata         {tty, sab, suppress} + optional RXN_STRENGTH,
                 RXN_AVAILABLE_STRENGTH, RXN_QUANTITY, RXN_HUMAN_DRUG,
                 RXN_OBSOLETED, RXN_BN_CARDINALITY
edge schema      NONE — graph is dict[concept_id, list[concept_id]]
relation labels  NONE
direction        NOT EXPLICIT — 0 of 21,353 sampled edges lack a reverse
records          82,429 (deduplicated from metadata.record_count 245,401)
graph nodes      77,055;  records absent from graph 5,374;  nodes absent from records 0
degree           min 1 | p50 3 | p99 20 | max 6,643
aliases          82,429 records carry aliases (100.0%)
```

TTY census over all records:

```text
SCDC 10,116  DP 7,516  SCD 7,007  SBDC 6,968  SBDG 6,689  SCDG 6,229  SCDF 5,529
SBDF 4,941   SBD 4,385 BN 4,128   SU 3,592    IN 2,905    SCDGP 2,489 SCDFP 2,023
SBDFP 1,767  SY 1,550  PIN 1,314  MIN 957     PSN 859     GPCK 475   BPCK 289
TMSY 269     MTH_RXN_DP 259  DF 120  DFG 43   PT 10
```

**The finding that shapes §5:** the builder saw **2,563,978 relations and stored none of
their names.** Spec §10.2 asks for traversal along `has_ingredient` / `consists_of` /
`tradename_of`; those relation names are not in the artifact. What *is* present is `tty`
on every record, and a bounded neighbour-TTY census confirms the chain is walkable by
role:

```text
SCDG->IN 4,909   SCDG->SBDG 4,250   SCD->SCDC 955   SBDC->SCDC 922
SBD->SCDC 851    SCDC->SCD 686      SBD->BN 531     SCD->DF 488
```

Verified end to end on one live concept (structure only, no restricted content dumped):

```text
IN 1191 -> SCDC 1665355 -> SCD 1665356 -> SBD 1665362
```

Strength is embedded in SCDC names (`aspirin 162.5 MG`), and dose form / release /
route in SCD names (`Extended Release Oral Capsule`, `Rectal Suppository`), which is
what makes the compatibility checks in §5 possible at all.

### 3.3 The INN naming gap — a governed-data finding

```text
records whose canonical_name or any alias contains "paracetamol":  0
exact["paracetamol"]:                                              None
exact["acetaminophen"]:                                            ["161"]
```

RxNorm is a US vocabulary using USAN names; Vietnamese clinical text uses INN/WHO
names. So a mention of `paracetamol 500mg` was **unlinkable by name**, and the old
lexical linker answered it with trigram noise — `ALCOHOL 0.7 L in 1 L TOPICAL GEL`
scored highest. The 12-entry bridge added in §5.4 is listed there in full for clinical
review.

## 4. ICD-10 hierarchy and specificity implementation

New: `src/mednorm_vi/linking/icd10_hierarchy.py` (378 lines), contract
`icd10-hierarchy-v1`. Rewritten: `src/mednorm_vi/linking/icd10.py`, contract
`icd10-hierarchy-linker-v1`.

Implemented, all deterministic and all against the real schema:

```text
exact + normalized alias retrieval        (existing channels, unchanged)
ancestor expansion                        ancestor_path(), bounded to depth 3
descendant expansion                      descendants(), bounded to 24
bounded sibling evidence                  siblings(), bounded to 8 — opened ONLY when
                                          the anchor's own added detail is unsupported
graph-depth + hierarchy provenance        HierarchyContext on every decision
broader-vs-specific competition           tier order exact > supported_specific >
                                          broader > lexical
explicit-text specificity detection       assess_specificity()
conservative broader fallback             KEEP_BROADER_FALLBACK
descendant rejection                      DROP_UNSUPPORTED_SPECIFICITY + missing tokens
specific-code preservation                KEEP_SPECIFIC_SUPPORTED
snapshot membership                       index.exists() before emission
deterministic tie-breaking                (tier, -score, code)
```

**The specificity rule, in one sentence:** a descendant may outrank its ancestor only
when the mention's own text contains the content tokens the descendant's canonical name
adds over its ancestor's. Otherwise it is suppressed, the missing tokens are recorded,
and the ancestor is preferred.

Depth is never treated as correctness. `_score` gives depth a small positive weight that
applies **inside** a tier only — a supported specific code has already earned its tier,
so depth breaks ties among equally supported codes rather than promoting an unsupported
one. `test_depth_alone_never_outranks_a_parent` holds this.

Every retained or suppressed candidate records: lexical sources and score, hierarchy
relationship, ancestor path, `declared_specificity` (for provenance), graph depth,
child/sibling counts, `in_graph`, token overlap, the added/supported/**missing** tokens,
the snapshot id, and the decision reason.

One improvement made after measurement: a candidate with **zero** content-token overlap
with the mention arrived on trigram coincidence alone (the live snapshot returned
`A679 "Bệnh pinta"` for `Bệnh tả`). Those are now dropped as
`DROP_NO_LEXICAL_SUPPORT`; an exact name match is exempt.

## 5. RxNorm structured-linking implementation

Three files. New: `linking/structured_medication.py` (410 lines),
`linking/rxnorm_graph.py` (333 lines). Rewritten: `linking/rxnorm.py`, contract
`rxnorm-structured-linker-v1`.

### 5.1 The structured representation (spec §10.1)

`StructuredMedicationMention` with `ingredient, salt, strength_value, strength_unit,
concentration, dose_form, release, brand, route, frequency, prn, duration`,
`unresolved_fields`, `incoherent_fields` and `provenance`. Built **only** from the E1
`ComponentSpan` records Audit 0052 preserved through the lattice and L4 — the exact
evidence Audit 0053 recorded as still ignored. Verified against the tracked medication
fixture: E1 already emits every role needed.

Three properties, each tested:

- **Nothing is invented.** No default route, no assumed tablet, no inferred brand. A
  field with no evidence lands in `unresolved_fields`.
- **Missing evidence is not negative evidence.** A mention that states no dose form
  cannot conflict on dose form; the comparison returns `NOT_STATED`, a first-class
  verdict distinct from `MATCH`.
- **Normalization never destroys the surface.** `500mg` and `500 MG` compare equal as
  `(500.0, "mg")`, and the original text plus absolute offsets survive on the field.

Incoherent-but-present evidence is recorded rather than repaired: two components for one
role, an unrecognised unit, or a strength value without its unit
(`strength:value_unit_pair_incomplete`).

### 5.2 Retrieval, corrected

The old linker searched on `hypothesis.text` — `amlodipine besylate 10 mg po daily` —
which no RxNorm concept is named, so the exact and normalized channels never fired.
Retrieval now queries brand, ingredient, salt-qualified name **and** the full surface,
and each candidate records which query found it.

### 5.3 Graph traversal (spec §10.2), and what it honestly is

`traverse_ingredient_chain` walks `IN|PIN|MIN → SCDC|SBDC → SCD → SBD` by **endpoint
TTY**, because §3.2 showed the graph has no relation labels. Every path is recorded as a
TTY sequence (`IN:1191->SCDC:1665355->SCD:1665356`), so a reader can see exactly what
was inferred. Bounds (64 neighbours/step, 48 nodes/step, 256 paths) are **bounds, not
rules**: exceeding one is reported as `traversal_truncated_at:<TTY>`, never presented as
"no more candidates exist".

Stated plainly so nobody over-reads it: **these are TTY-role transitions inferred from an
unlabeled graph, not RxNorm's named relations.** Rebuilding the index with labeled edges
would let this tighten from "a neighbour with the right TTY" to "a neighbour across the
right relation"; that is recorded as KB-intake work (§25), not hidden here.

### 5.4 Compatibility and hard negatives

Six per-field comparisons, each returning `MATCH` / `CONFLICT` / `NOT_STATED` /
`NOT_COMPARABLE`, and each conflict suppressing the candidate with its own reason code:

```text
strength        cross-unit mass conversion (1 g == 1000 mg)   DROP_STRENGTH_CONFLICT
unit family     mass vs volume is a CATEGORY ERROR            DROP_UNIT_CONFLICT
concentration   mg-per-mL ratio                               DROP_CONCENTRATION_CONFLICT
dose form       conservative VI/EN vocabulary                 DROP_DOSE_FORM_CONFLICT
release         extended / delayed / immediate                DROP_RELEASE_CONFLICT
route           oral, IV, IM, SC, topical, rectal, …          DROP_ROUTE_CONFLICT
```

Plus: `DROP_SUPPRESSED_CONCEPT` (snapshot `suppress`/`RXN_OBSOLETED`),
`DROP_NON_TERMINAL_TTY` (a grouper is evidence, never a prescribable code),
`DROP_INGREDIENT_MISMATCH`, `DROP_NOT_IN_SNAPSHOT`, `DROP_BUDGET`.

Conservative fallback: when only the ingredient is justified, the IN concept is kept at
tier `ingredient_only` with every unstated field recorded as `NOT_STATED`.

Every candidate decision records retrieval sources, matched fields, conflicting fields,
not-stated fields, the graph path, TTY, snapshot id, retain/suppress reason, confidence
tier, whether the product is a combination, and its brand names.

### 5.5 The INN bridge — a stopgap, marked as one

```text
paracetamol -> acetaminophen        salbutamol -> albuterol
adrenaline -> epinephrine           noradrenaline -> norepinephrine
lignocaine -> lidocaine             frusemide -> furosemide
glibenclamide -> glyburide          amoxycillin -> amoxicillin
cephalexin -> cefalexin             rifampicin -> rifampin
trimethoprim-sulfamethoxazole -> sulfamethoxazole / trimethoprim
co-trimoxazole -> sulfamethoxazole / trimethoprim

status: REQUIRES_CLINICAL_REVIEW      12 entries      widens RETRIEVAL only
```

It never asserts a code: a bridged mention still passes every structured compatibility
check and snapshot-membership check, and the bridge is recorded on the report as
`inn_bridge:paracetamol->acetaminophen:REQUIRES_CLINICAL_REVIEW`. It is deliberately
short — **a long hand-written drug-name table is a patient-safety hazard dressed as a
convenience**, and a test asserts the table stays under 20 entries. The real fix is a
governed crosswalk at KB-intake time.

Measured effect on the tracked medication fixture: `paracetamol 500mg` went from **0**
retained candidates (everything failed the ingredient check) to 5, all at tier
`structured_exact`.

## 6. L6 edge implementation

`evidence_graph/graph.py` → contract `clinical-evidence-graph-v2`. Audit 0053 measured
three relations and no consumer. All nine are now built, from the evidence each one
actually requires:

```text
relation       source of truth                          tier       measured (200 rows)
supports       L3 proposal provenance                   derived    333
has_assertion  L5 assertion decision                    derived    333
has_candidate  L5 linker                                derived    2,099
has_result     E2 pair_group_id preserved through L4     explicit   1
modified_by    E1 ComponentSpan (head role excluded)     explicit   7
in_section     L1 section node containing the span       explicit   333
overlaps       verified character intervals              inferred   0
same_surface   normalized surface, coordinates kept      inferred   15
treats         explicit text trigger OR governed KB      explicit   0
```

`overlaps` and `treats` are 0 on this corpus, and both zeros are explained rather than
excused: L4 leaves no surviving same-type overlap on these documents, and narrative
DIAGNOSIS/SYMPTOM text contains no medications for `treats` to relate. Both fire on unit
fixtures. On the tracked laboratory fixture `has_result` reaches 7 and `modified_by` 14.

**`treats` is deliberately hard to create.** It requires an explicit Vietnamese trigger
sitting *between* the two mentions, within a 120-character window, with no sentence
terminator in between — or a governed KB relation. **Co-occurrence produces no edge at
all**, and the declined pair is recorded in `declined_treats` so the consistency layer
can report a refused claim. A discharge summary mentions a dozen medications and a dozen
diagnoses; asserting treatment from proximity would manufacture clinical claims.

Every edge carries `evidence_source`, `confidence_tier`, `provenance` and `rule`. Edges
are de-duplicated by `(source, target, relation)`, endpoints must exist as nodes, and
`graph_hash` covers every field — so serialization is deterministic and the hash is
stable across runs.

## 7. Deterministic consistency contract

New: `evidence_graph/consistency.py` (579 lines), contract `graph-consistency-v1`.
Audit 0053 §7 reported that "consistency violations" was **not a measurable quantity**
and refused to report zero. That refusal was correct, and this closes it.

Typed public result: `GraphConsistencyReport`, `ConsistencyDecision`, `ConsistencyIssue`,
`SupportSignal`. Twelve rules:

```text
C01 section_compatibility                     C07 overlap_competition
C02 test_pair_completeness                    C08 duplicate_candidate_codes
C03 assertion_entity_type_compatibility       C09 repeated_mention_agreement
C04 candidate_entity_type_compatibility       C10 conflicting_assertion_evidence
C05 icd_hierarchy_compatibility               C11 unresolved_structured_medication_conflict
C06 rxnorm_structured_compatibility           C12 unsafe_or_unsupported_treats
```

**Uncertainty is never encoded as false.** Every rule returns one of four verdicts —
`SUPPORTED`, `CONTRADICTED`, `UNRESOLVED`, `NOT_APPLICABLE` — because folding
`UNRESOLVED` into `CONTRADICTED` invents contradictions and folding it into `SUPPORTED`
hides them.

Every issue identifies its rule id and version, affected hypothesis / candidate / edge
ids, supporting and blocking signals, a downstream recommendation (`EMIT`,
`EMIT_WITH_CAUTION`, `ESCALATE`, `WITHHOLD`), and whether it is **fatal or advisory**.
The report contains no raw clinical text — asserted by a test that scans every long
alphabetic token of the real fixture against the serialized report — and serializes
deterministically (`report_hash` stable across runs).

**A defect this audit introduced and its own tests caught, recorded because it matters:**
`CANDIDATE_ONTOLOGY_BY_TYPE` is keyed by internal English type names (`MEDICATION`),
while `EntityHypothesis.entity_type` carries the organizer-facing Vietnamese label
(`THUỐC`). Looking the Vietnamese label up directly returned `None` for every type,
which C04 read as "this type takes no candidates", which made every linked medication a
fatal contradiction, which **withheld 100% of output** — 6 predictions became 0. The
translation now lives in one named function, `consistency.ontology_for`, with the failure
mode written above it and a test pinning all three cases.

## 8. L6 consumption by L7 and L8

Neither consumer touches a private dictionary; both take the typed report.

**L7** (`build_evidence_bundle` / `evaluate_entry_conditions`) consumes fatal issues,
advisory issues, repeated-mention support, assertion uncertainty, candidate
compatibility, laboratory pair completeness, section support and overlap competition.

**L8** (`decode_entities(..., consistency=, escalation=)`) consumes cascade disposition,
candidate compatibility, graph contradiction status, hierarchy specificity support,
structured RxNorm support, assertion uncertainty and repeated-mention evidence.

Required behaviour, each held by a test:

```text
a fatal contradiction cannot be silently accepted   entity withheld
unsupported candidates may not be emitted           codes dropped as
                                                    dropped_l6_fatal_contradiction
an advisory issue may lower confidence / escalate    UNRESOLVED does NOT withhold
consistency evidence appears in decoder reasons      retention_reason carries
                                                     consistency:<rule>:<verdict>
no fixed top-K returns                               15 distinct sizes measured
the calibrated decoder stays disabled                CalibratedDecoderUnavailable raised
```

One design correction, made because a test failed: **a fatal issue that names candidate
codes condemns those codes; a fatal issue that names none condemns the entity.** A
duplicate RxCUI is a reason to drop the duplicate, not to delete a correctly-found
medication. Conflating the two would have made C08 delete entities.

Omitting both arguments reproduces the Audit-0053 behaviour exactly, so the wiring is
additive rather than a silent change of policy — `test_a_fatal_contradiction_cannot_be_
silently_accepted` asserts both halves.

## 9. L7 locked-option contract

New: `confidence_cascade/escalation.py` (762 lines), contract
`l7-locked-option-escalation-v1`. Every type the milestone named:

```text
CascadeTier          deterministic | critic | adjudicator
CascadeDisposition   ACCEPT | REJECT | UNRESOLVED | ESCALATE
EvidenceBundle       scalars and ids only — mention text is deliberately ABSENT
LockedBoundaryOption LockedTypeOption LockedAssertionOption LockedCandidateOption
LockedOptionSet      + option_set_hash + offered_codes()
EscalationRequest    EscalationDecision    EscalationValidationResult
```

**Spec P7 lives in the plumbing, not in a prompt.** An `EscalationDecision` names its
choices by **option id**, never by value, so it cannot express a code, label or
coordinate that was not offered. `validate_escalation_decision` re-checks every id and
refuses with `REFUSE_UNKNOWN_OPTION`, `REFUSE_INVENTED_VALUE`, `REFUSE_WRONG_SUBJECT`,
`REFUSE_EMPTY_DECISION` or `REFUSE_UNKNOWN_TIER`. A refused decision **degrades to the
deterministic fallback** rather than propagating.

Option sets are built only from alternatives that already exist: boundaries from L4's
chosen span plus its retained alternatives, types from the resolved type plus types an
expert actually proposed (never the whole organizer vocabulary — a type nobody proposed
is a guess, not an alternative), assertion **label sets** rather than individual labels
(so no model can assemble a combination nobody vetted — which is what the Audit-0052
all-three-labels defect looked like from outside), and candidates from the linker.

Ten entry conditions, and **both the fired and the not-fired ones are recorded**, so
"why did this not escalate?" is as answerable as "why did it?".

`EvidenceBundle` deliberately omits mention text. A future critic will need it, and will
receive it through a separate explicitly-audited prompt-construction step; keeping it out
means the bundle is safe to log, hash and put in a manifest.

**No model is loaded, and none can be from here.** A test parses the module's AST and
asserts it imports no `transformers`, `torch`, `requests`, `urllib`, `socket` or `llm`
module. The test double used in tests is named `_TestDecisionSource` with
`name = "test_fixture_decision_source"` — explicitly a fixture, never described as a
model.

Thresholds are fixed and written down (`LOW_CONFIDENCE_BELOW = 0.35`,
`AMBIGUOUS_CANDIDATE_COUNT = 8`), **not searched**. Two were tightened after
measurement showed 6/6 subjects escalating on the medication fixture:
`boundary_competition` from `> 0` to `> 1` retained alternatives (L4 retains a runner-up
for almost every mention), and `graph_unresolved` narrowed to issues whose own
recommendation is `ESCALATE`. Both changes are recorded in the source with the measured
reason.

## 10. Inference run manifest

New: `inference/manifest.py` (489 lines), schema `inference-run-manifest-v1`.
Opt-in via the CLI's `--run-manifest PATH`; **nothing is written without it**, and a
test asserts that constructing a manifest touches no filesystem.

Recorded: schema version, git commit + branch + dirty state, architecture PDF SHA-256,
eight contract versions, config hashes, mode, readiness status and errors, expected vs
actually-run experts, degradation and fail-closed state, checkpoint roles / SHA-256 /
revisions / parameter-count status, snapshot ids and index identity, document count, and
aggregate counts for route gating, proposals by expert, lattice, L4
accepted/rejected/unresolved, assertions, linking, L6 edges, consistency decisions and
issues, L7 dispositions and conditions, decoder candidate-size distribution and reasons,
L9 issue counts, deterministic seeds, interpreter, runtime, peak memory, and output
hashes **only when output writing was authorized**.

Not recorded: source text, entity text, prompts, absolute home-directory paths, tokens,
secrets. `role_safe_path` reduces `/home/someone/checkpoint/s1/best.pt` to
`role:e3_checkpoint/best.pt`. `write_run_manifest` refuses to write anything whose
`contains_clinical_text` is not `False`.

**It is written even when L9 stops the run**, with `l9_stopped_the_run: true` — closing
the loop Audit 0053 §11 recorded: the gate could detect a violation and nothing recorded
the stop.

Determinism verified by two back-to-back runs: byte-identical apart from
`runtime_seconds` and `peak_memory_gib`, which are enumerated in
`RUNTIME_VARIABLE_FIELDS` rather than quietly excluded.

**A real defect the manifest immediately caught:** `output_zip_sha256` differed between
two identical runs, because `package_output_zip` used `ZipFile.write`, which copies each
file's mtime into the entry header. Entries now use a fixed timestamp (1980-01-01) and
fixed mode, and the archive is byte-reproducible. An archive whose hash changes every
second cannot be pinned or attested to, which matters for Appendix A.

## 11. Old-L4 behaviour inventory

`src/mednorm_vi/resolution/resolver.py` (5,402 bytes) compared against
`canonical.py`, `resolver_v1.py`, `boundary.py`, `typing.py`, `overlap.py`.

Behaviour present in the old resolver and **absent from the canonical L4**:

```text
ResolverConfig.medication_boundary    "full" | "name_only" | "name_strength"
ResolverConfig.test_result_boundary   "value_only" | "value_unit" | "full"
ResolverConfig.abstain_on_conflict    bool
```

That configurable per-type boundary policy is the whole of the unique surface. The
canonical L4 shapes boundaries by evidence-weighted trim/expand actions and has no
equivalent knob.

Behaviour the canonical L4 **also** has, and does with more evidence: proposal grouping,
type assignment, same-type overlap resolution, `has_result` retention, scoring,
punctuation/unit handling (inside `boundary.py`, shared by both), per-type tie-breaking,
and warning/provenance emission.

Live importers of the obsolete module, taken by inventory:

```text
src/mednorm_vi/resolution/__init__.py        re-exports ResolverConfig, resolve
src/mednorm_vi/phase1c_foundation/cli.py     a separate CLI that runs it
src/mednorm_vi/phase1c_foundation/doctor.py  validates its config loads
tests/unit/test_resolution.py                its existing test suite
```

The canonical runner does **not** import it — asserted by
`test_the_old_resolver_is_off_the_canonical_path`, which parses `inference/pipeline.py`'s
AST.

## 12. Migrated behaviour and deletion proof — **NOT DONE**

**This is one of the two unmet acceptance criteria, and it is unmet in full.**

Milestone 2B §10 asked for five steps: characterize, migrate, prove equivalence,
document, delete. **Step 1 is complete. Steps 2–5 were not attempted.**

Done — `tests/unit/test_old_l4_characterization.py`, **13 tests**, pinning:

```text
medication_boundary  full -> "aspirin 81 mg po"      (3 parametrized cases)
                     name_only -> "aspirin"
                     name_strength -> "aspirin 81 mg"
test_result_boundary value_only -> "14.43"           (2 parametrized cases)
                     value_unit -> "14.43 K/uL"
policy + alternatives recorded on the chosen boundary
pair-group evidence retained on both endpoints
unsupported types -> UNRESOLVED + a warning
abstain_on_conflict loads and resolves under both settings
config loads from YAML and hashes deterministically
the module is still present (guards an undocumented deletion)
the module is off the canonical path
the importer set is the documented one (fails if a new importer appears)
```

Not done: no migration into the canonical L4, no equivalence proof, no deletion, and
`resolution/resolver.py` remains importable.

**Why, stated plainly rather than excused:** deleting the module means migrating a
separate CLI surface (`phase1c_foundation/cli.py`, `doctor.py`), a package re-export and
an existing test suite. Doing that in the same turn as six other stages would have meant
proving none of it properly. The characterization is the honest half — it is now recorded
what would be lost — and §25 puts the migration first.

## 13. Docker and requirements audit

Audited, and **two real defects found**:

**(1) The lock cannot install.** `requirements.lock` pins `torch==2.13.0+cu126`. The
`+cu126` local version identifier does not exist on PyPI, so
`pip install --no-deps -r requirements.lock` cannot resolve it without PyTorch's own
index — and pulling the CUDA build into a slim image adds roughly 2.5 GB of CUDA runtime
the inference path never touches. Audit 0053 authored the Dockerfile against that lock
and never built it, so this went unnoticed.

Fix: new `requirements-image.lock` (74 lines) installs the **same upstream version**
from the CPU index. This is a recorded deviation, not a loosened pin — the version is
identical (2.13.0, no range), every recorded E3 validation run in Audits 0052–0054 was a
**CPU forward pass**, and the checkpoint loads on CPU. A GPU image is the same Dockerfile
with two build args overridden. The file's header states all of this.

**(2) The output ZIP was not byte-reproducible** — see §10.

Dockerfile hardening in this audit:

```text
explicit indexes        --index-url + --extra-index-url named in the RUN, so the build
                        cannot silently resolve a different torch build from ambient
                        pip configuration
TORCH_INDEX_URL / TORCH_SPEC build args, defaulting to the CPU wheel, documented
torch import verified   inside the build, so a wrong wheel fails the build not the run
heavy deps declared     torch, transformers, tokenizers, safetensors, huggingface-hub,
                        numpy, sentencepiece + full transitive closure enumerated
VnCoreNLP declared      py_vncorenlp + openjdk-17-jre-headless; the jar is MOUNTED
offline runtime         HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
read-only mounts        /models /kb /input documented `:ro`; only /output writable
non-root                uid 10001; owns /app and /output only
one entrypoint          python -m mednorm_vi.inference.cli
no weights or KB copied a test asserts no COPY line touches checkpoint/, models/,
                        data/, indices/ or notebooks
import smoke in-build   now also imports consistency, escalation and manifest
```

## 14. Actual Docker build and offline smoke — **BLOCKED, exact blocker recorded**

The build **was attempted**, and it failed. Nothing was faked and the Dockerfile was not
rewritten to avoid reporting this.

```text
$ docker build -t mednorm-vi:0054 .
#2 ERROR: failed to authorize: failed to fetch oauth token: unexpected status from
   GET https://auth.docker.io/token?scope=repository%3Alibrary%2Fpython%3Apull
   &service=registry.docker.io: 401 Unauthorized: incorrect username or password
ERROR: failed to build: failed to solve: failed to fetch oauth token: ... 401 ...
```

Diagnosis:

```text
docker daemon            RUNNING — server 29.5.3, storage overlayfs
base image cached?       NO — no `python` image present locally
anonymous pull attempt   docker pull python:3.14.5-slim-bookworm
                         -> "authentication required - incorrect username or password"
~/.docker/config.json    exists; `auths` holds three stale Docker Hub entries:
                           https://index.docker.io/v1/
                           https://index.docker.io/v1/access-token
                           https://index.docker.io/v1/refresh-token
                         credsStore: none;  credHelpers: none
```

**Exact unavailable capability: pulling any image from Docker Hub.** The daemon sends the
stored (expired) token on every pull, so even an anonymous pull of a *public* base image
is rejected. This is an operator credential-store problem, not a Dockerfile problem and
not a network problem.

It was **not** fixed here because the fix means deleting or editing the owner's stored
Docker credentials (`docker logout`, or removing the `auths` entries), which is their
environment and their call — not something to do silently inside a repository audit.

What the owner runs to unblock it:

```bash
docker logout                      # clears the stale Docker Hub tokens
docker build -t mednorm-vi:0054 .
docker run --rm --network=none mednorm-vi:0054 --help
```

The offline smoke — package imports, canonical CLI help, deterministic readiness, full
mode failing closed, no network access, no model download, no organizer inference, no
`output.zip` — therefore **did not run**. Its static counterpart did: 4 tests assert the
Dockerfile's offline env, single entrypoint, non-root user, absence of any weight/data
COPY, documented mounts, and that every lock line is `==`-pinned. Those are contract
checks, not a build, and this audit does not present them as one.

The equivalent in-process smoke **did** run: the bounded fixture validation in §17 drives
`run_document` over tracked fixtures in both operational modes with no network and no
download, and `full` mode fails closed.

## 15. Code-bearing evaluation contract

New: `evaluation/code_linking.py` (512 lines), contract `code-linking-eval-v1`. Every
type the milestone named: `CodeLinkingGoldRecord`, `CodeLinkingPredictionRecord`,
`RecallAtKResult`, `CandidateJaccardResult`, `LinkingErrorCategory` (+ `CodeLinkingReport`).

Requirements, each tested:

```text
strict schema                     end > start; known ontology; snapshot_id required;
                                  ADJUDICATED_SINGLE carries exactly one code;
                                  ADJUDICATED_NONE carries none
locked ontology + snapshot ids     on every record
exact mention coordinates          end-exclusive, verified against source
zero / one / many acceptable codes all three are legitimate answers
explicit adjudication status        ADJUDICATED_{SINGLE,MULTIPLE,NONE} | PENDING | DISPUTED
source provenance                  annotators + adjudication_rule
leakage-safe split identity        split_sha256, never a filename
Recall@1 / @5 / @10                excludes correctly-empty gold from the denominator
candidate Jaccard                  a correct withholding scores 1.0
exact candidate-set match          counted separately
broader-vs-specific error          separated using an ancestor map, not guessed
wrong-ontology error               distinct category
stale-code error                   candidate outside the snapshot
missing-gold refusal               GoldSetUnavailable, never 0.0
```

**It refuses rather than fabricating.** `load_gold_records` raises on a missing file and
on any `PENDING` or `DISPUTED` record — skipping one would shrink the denominator and
inflate every metric. `evaluate_code_linking` raises `GoldSetUnavailable` rather than
returning zeros, because a 0.0 looks like a measurement and ends up in a report.
`candidate_quality_status()` returns the sentinel `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`
that audits must print.

**No gold data ships.** There is no function that turns predictions into labels, the
tests use synthetic miniature records only, and a test asserts no gold-looking artifact
exists under `data/`.

The human artifact still required is specified in code as
`REQUIRED_ANNOTATION_ARTIFACT`: JSONL from the **governed validation split only** (never
`internal_test`), **every** DIAGNOSIS and MEDICATION mention in the sampled documents
(partial coverage makes recall uninterpretable — a missing annotation is
indistinguishable from a correctly empty one), at least two independent clinical
annotators per mention, an explicit adjudication step, inter-annotator agreement reported
before use, split identity by SHA-256; and explicitly forbidden: codes from this
repository's linkers, codes from any language model, codes transferred from another
corpus without clinical review, and any record still `PENDING` or `DISPUTED`.

## 16. Tests and static checks

New test modules:

```text
tests/unit/test_l5_l7_deterministic_stack.py   1,228 lines,  91 tests
   A structured medication representation (6)
   B RxNorm traversal + hard negatives (13)  — miniature governed graph fixture
   C ICD hierarchy + specificity (13)        — miniature fixture + live schema/determinism
   D L6 edges (8)
   E typed consistency (7)
   F L7 locked options (8)
   G L8 consumes L6/L7 (6)
   H run manifest (6)
   I code-linking contract (14)              — synthetic miniature records only
tests/unit/test_old_l4_characterization.py       288 lines,  13 tests
```

The mini-fixture / live-snapshot split is deliberate: decision tests use miniature
governed graphs with the **real schema shape** (unlabeled `dict[str, list[str]]`
adjacency, TTY in `metadata`), and the live locked snapshots are used only for schema
invariants, membership and determinism. A decision test pinned to 82,429 concepts would
break on every snapshot refresh; a schema test against a toy graph would prove nothing.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q
    1729 passed, 1 skipped in 331.09s      (baseline 1625 passed, 1 skipped;
                                            +104 tests; the skip is pyarrow, absent)
ruff check .                    All checks passed!
ruff check notebooks            All checks passed!
env PYTHONPATH=src .venv/bin/python -m mypy
    Success: no issues found in 277 source files
env PYTHONPATH=src .venv/bin/python -m compileall -q src     clean
git diff --check                                             clean
whole-package import sweep      268 modules imported, 0 failures
```

Defects these checks found during the work, reported rather than smoothed over:

```text
1. C04 keyed CANDIDATE_ONTOLOGY_BY_TYPE by the Vietnamese organizer label instead of
   the internal type name -> every medication became a fatal contradiction ->
   100% of output withheld (6 predictions -> 0).  Fixed; §7.
2. A fatal issue naming candidate codes withheld the whole ENTITY instead of the
   codes.  Fixed; §8.
3. NodeKind is a str Enum, so `str(node.kind)` renders "NodeKind.SECTION" and
   `in_section` never fired.  Fixed; §6.
4. output.zip was not byte-reproducible (mtimes in ZIP headers).  Fixed; §10.
5. requirements.lock could not install in the image (+cu126 not on PyPI).  Fixed; §13.
6. Six mypy errors from a frozen-dataclass/Protocol mismatch and two return-type
   mismatches.  Fixed.
7. A test-helper bug (components not forwarded) and a test that scanned a docstring
   instead of the imports.  Both were test defects, both fixed.
```

## 17. Bounded validation evidence

Split identity resolved **by digest, never by name**; `internal_test` never opened:

```text
data/derived/training_corpora/mednorm_vi_training_v1/splits/validation.jsonl
sha256 ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103
bounded to the first 200 of 1,045 rows — the same bound as Audits 0052 and 0053
```

### 17.1 Exact span + type (unchanged, as expected)

```text
arm                       P        R       F1     TP    FP    FN
E1+E2 (deterministic) 0.0000   0.0000   0.0000     0     5   406
E3 only               0.6220   0.5025   0.5559   204   124   202
E1+E2 gated + E3      0.6126   0.5025   0.5521   204   130   202
```

Identical to Audit 0053 to four decimal places. That is the **control**: this milestone
changed L5–L9, so the span/type numbers must not move, and they did not. Per type
(merged arm): DIAGNOSIS P 0.6600 R 0.4783 F1 0.5546; SYMPTOM P 0.5625 R 0.5538 F1
0.5581; TEST_NAME 1 FP; TEST_RESULT 4 FP.

### 17.2 Route eligibility, lattice, L4

```text
E1 eligible 230 / skipped 5      E2 eligible 6 / skipped 229
C2 suppressed for lack of required evidence      7 nodes
proposals: E2 6, E3 328
lattice nodes 334, merges 0
L4 accepted 333 / rejected 0 / unresolved 1      offset violations 0
```

### 17.3 Assertions (regression check, not a score)

```text
with labels 13 of 333      all three labels 0      uncertain 35
```

Zero all-three holds the Audit-0052 fix. **No assertion metric is computable** — the
governed corpus has zero assertion supervision.

### 17.4 ICD candidate sizes and hierarchy behaviour

```text
size  0:12  1:3  2:4  3:8  4:11  5:2  6:12  7:7  8:21  9:12 10:8
     11:15 12:9 13:11 14:9 15:11 16:5 17:15 18:3 19:3  20:19     (21 distinct values)

decision reasons
  DROP_NO_LEXICAL_SUPPORT       11,609     KEEP_LEXICAL              1,128
  DROP_UNSUPPORTED_SPECIFICITY   3,328     KEEP_SPECIFIC_SUPPORTED     478
  DROP_BUDGET                      113     KEEP_SIBLING_COMPETITION    290
                                           KEEP_BROADER_FALLBACK       193
                                           KEEP_EXACT_NAME              10

hierarchy relationships of retained candidates
  self 95   sibling 76   ancestor 37   descendant 20   unrelated 1,871
```

The 3,328 unsupported-specificity suppressions are the §9.3 controller doing its job:
each is a descendant whose added detail the mention did not state. The 193 broader
fallbacks are the conservative answer in those cases.

### 17.5 RxNorm structured behaviour

```text
on the 200 governed rows:  0 medications  ->  no RxNorm candidates
```

Reported honestly rather than padded: this corpus is narrative DIAGNOSIS/SYMPTOM text
and contains no medication mentions, so the RxNorm path has nothing to link. Its
behaviour is measured on the tracked medication fixture instead:

```text
candidate sizes per mention       [14, 1, 1, 1, 2, 2]   (old lexical: 20,1,20,20,1,20)
KEEP_STRUCTURED_MATCH                 59
KEEP_LEXICAL_ONLY                     20
KEEP_INGREDIENT_FALLBACK              10
KEEP_EXACT_INGREDIENT                  4
DROP_STRENGTH_CONFLICT                74
DROP_NON_TERMINAL_TTY                146
DROP_DOSE_FORM_CONFLICT                9
DROP_CONCENTRATION_CONFLICT            9
DROP_INGREDIENT_MISMATCH               6
DROP_SUPPRESSED_CONCEPT                5
DROP_BUDGET                           19
traversal: IN -> SCDC -> SCD -> SBD walked; depth up to 3; truncation reported
```

### 17.6 L6 edges, consistency, L7, L8, L9

```text
L6 edges (specialist arm)
  supports 333   has_assertion 333   has_candidate 2,099   in_section 333
  same_surface 15   has_result 1   modified_by 7   overlaps 0   treats 0
  declined_treats 0

consistency decisions   SUPPORTED 1,193   UNRESOLVED 81   CONTRADICTED 1
                        NOT_APPLICABLE 1,005
consistency issues      C02 3   C04 12   C05 28   C09 4   C10 35

L7 dispositions         ACCEPT 239   UNRESOLVED 94   REJECT 0   ESCALATE 0
L7 conditions fired     graph_unresolved 66   assertion_uncertainty 35
                        candidate_ambiguity 33   repeated_mention_conflict 8
                        graph_contradiction 1

L8 candidate sizes      0:145 1:42 2:57 3:30 4:22 5:9 6:8 7:4 8:5 9:1 10:1
                        11:3 12:3 13:2 18:1        (15 distinct values)
L8 candidate reasons    kept_within_tier_score_band 617
                        kept_exact_ontology_evidence 10
                        dropped_below_tier_score_band 812
                        dropped_weaker_tier_than_best_evidence 660

L9 issues               0
```

`ESCALATE 0` is truthful: no decision source exists, so the deterministic fallback
answers `UNRESOLVED` instead. `overlaps 0` / `treats 0` are explained in §6.

### 17.7 Fixture validation, both modes

```text
readiness[deterministic]  READY       readiness[specialist]  READY
readiness[full]           NOT_READY   12 blockers; run_document raises -> fails closed

mode=deterministic   4 fixtures, 13.0s
mode=specialist      4 fixtures, 35.6s
every emitted offset re-verified: original_text[start:end] == text, 0 violations
laboratory fixture: has_result 7, modified_by 14, in_section 15
mixed fixture:      has_result 5, modified_by 25, in_section 15/18
```

### 17.8 Run-manifest determinism

Two back-to-back runs of the same document produced byte-identical manifests apart from
`runtime_seconds` and `peak_memory_gib`, and an identical `output_zip_sha256` after the
ZIP fix. A leakage check over the medication fixture's drug names found none present in
the serialized manifest.

### 17.9 Runtime and memory

```text
E3 checkpoint load (once)          22.6 s
3-arm L1-L9 run, 200 documents     38.3 s   (0.192 s/doc for all three arms)
peak RSS                            3.16 GiB
fixture validation, deterministic   13.0 s   specialist  35.6 s
```

CPU forward passes throughout; no GPU, no download.

### 17.10 Candidate quality

```text
Recall@1 / @5 / @10        UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD
candidate Jaccard          UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD
exact candidate-set match  UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD
```

No independently human-checked governed set exists. Audit 0053 §15.4 measured the cause
and it is unchanged: all 406 gold entities carry `target_type` and `mapping_status:
MAP_EXACT`, and **not one carries an ontology code**. §15 built the contract and the
refusal path; it did not and must not invent labels.

**No organizer score is claimed. No broad threshold optimization was performed.**

## 18. Exact changed-file inventory from Git

`git diff --stat` — **13 modified**:

```text
 Dockerfile                                    |  44 +-
 docs/architecture/ACTIVE_RUNTIME_MANIFEST.md  | 167 +++---
 docs/audits/README.md                         |   1 +
 src/mednorm_vi/confidence_cascade/__init__.py |  67 ++-
 src/mednorm_vi/evidence_graph/__init__.py     |  52 +-
 src/mednorm_vi/evidence_graph/graph.py        | 394 ++++++++++++-
 src/mednorm_vi/inference/cli.py               |  18 +-
 src/mednorm_vi/inference/packaging.py         |  45 +-
 src/mednorm_vi/inference/pipeline.py          | 164 +++++-
 src/mednorm_vi/linking/icd10.py               | 366 +++++++++++-
 src/mednorm_vi/linking/rxnorm.py              | 770 +++++++++++++++++++++++++-
 src/mednorm_vi/metric_decoder/__init__.py     |   8 +
 src/mednorm_vi/metric_decoder/decoder.py      |  87 ++-
 13 files changed, 2014 insertions(+), 169 deletions(-)
```

`git ls-files --others --exclude-standard` — **11 added** (untracked files do not appear
in `git diff --stat`, which is why the two lists come from different commands):

```text
    74  requirements-image.lock
   762  src/mednorm_vi/confidence_cascade/escalation.py
   512  src/mednorm_vi/evaluation/code_linking.py
   579  src/mednorm_vi/evidence_graph/consistency.py
   489  src/mednorm_vi/inference/manifest.py
   378  src/mednorm_vi/linking/icd10_hierarchy.py
   333  src/mednorm_vi/linking/rxnorm_graph.py
   410  src/mednorm_vi/linking/structured_medication.py
  1228  tests/unit/test_l5_l7_deterministic_stack.py
   288  tests/unit/test_old_l4_characterization.py
  1289  docs/audits/0054-…-container-audit.md   (this file)
```

Total: **13 modified + 11 added = 24 paths.** `ACTIVE_RUNTIME_MANIFEST.md` and
`docs/audits/README.md` are in the modified list because they are, and Audit 0052 §13 was
corrected once for omitting exactly that kind of entry.

**Deleted: 0. Renamed: 0.** No model weights, governed data or generated artifacts are in
the change set — verified by pattern (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`,
`.zip`) and by the Dockerfile COPY test.

## 19. Remaining pretrained-model gaps

| Gap | State |
| --- | --- |
| E5 XLM-R MRC-NER | contract + trainer exist; **task head randomly initialized**; must never run untrained |
| E6 GLiNER | **no local weights**; strict adapter fails closed |
| E7 Qwen proposer | **no local weights**; loader fails closed |
| S3 dense retrieval (ICD + RxNorm) | no embedder weights; retrieval is lexical + graph only |
| S4 cross-encoder reranker | none; depends on S3 |
| S5 Qwen LoRA critic/adjudicator | no local weights. **The L7 contract is now complete**, so a backend is a drop-in — but there is no backend |
| S6 calibration meta-model | none — still what blocks real spec §13 decoding |

Every loader is lazy and `local_files_only=True`. **No model was downloaded.** `full`
mode still fails closed with 12 named blockers.

## 20. Remaining training / fine-tuning and data gaps

| Stage | State | Blocker |
| --- | --- | --- |
| S0 domain adaptation | `SCAFFOLD_ONLY` | notebook is a design draft |
| **S1 mention** | **EXECUTED** | complete — E3 is the only trained model |
| S2 assertion | `IMPLEMENTED_NOT_RUN` | **zero assertion supervision** in the governed corpus |
| S3 retrieval | `SCAFFOLD_ONLY` | no embedder; and **zero ontology codes** to train or evaluate against |
| S4 reranking | `SCAFFOLD_ONLY` | depends on S3 |
| S5 Qwen LoRA | `NOT_STARTED` | no local weights |
| S6 calibration | `NOT_STARTED` | needs out-of-fold predictions no stage produces |

Data gaps, which are now the dominant constraint:

```text
ICD-10 / RxNorm gold codes    ZERO — 40% of the metric is unmeasurable
assertion supervision         ZERO — 30% of the metric is held by regression tests only
route gold                    ZERO — routing accuracy has never been measured
INN <-> RxNorm crosswalk      ABSENT — a 12-entry stopgap covers the common cases
ICD canonical_name quality    some records are truncated PDF fragments
```

**No training or fine-tuning ran. No calibration was fitted. No thresholds were searched
broadly.**

## 21. Genuine blockers

1. **The governed corpus has no ontology codes.** Candidates are 40% of the organizer
   metric and remain **unmeasurable**. Audit 0054 built the contract that will measure
   them and the refusal that prevents faking them; the missing piece is a human
   annotation round. This is a *data* blocker and it gates the value of §4 and §5.
2. **Docker Hub is unreachable from this environment** (§14) — stale credentials in the
   operator's `~/.docker/config.json`. One `docker logout` away, but not this agent's
   file to modify.
3. **No calibrated probabilities below L4.** Spec §13 is an expectation over
   probabilities; there are none. L8's deterministic decoder is the correct interim
   state, not a step toward §13.
4. **Zero assertion supervision.** 30% of the metric is held by regression tests against
   constructed cases.
5. **Both KB graphs discarded their relation labels at build time.** ICD direction is
   inferred from code length, RxNorm direction from endpoint TTY. Both are recorded as
   inferences and both are correct on the samples verified — but they are inferences.
6. **RxNorm is a US vocabulary against Vietnamese INN text.** `paracetamol` is absent
   from the snapshot entirely; a governed crosswalk is needed.
7. **Two L4 implementations still coexist** (§12).
8. **The Docker image has still never been built**, so Appendix A's one-command rebuild
   is corrected-but-unproven.

## 22. Acceptance-criteria table

```text
MILESTONE 2B ACCEPTANCE CRITERIA                                        STATUS
RxNorm consumes EntityHypothesis.components                             MET     §5.1
RxNorm graph traversal against the real snapshot schema                 MET     §5.3
structured hard negatives enforced                                      MET     §5.4
ICD hierarchy traversal implemented                                     MET     §4
ICD general/specific decisions auditable                                MET     §4
no invented ICD code or RxCUI can be emitted                            MET     tests
all supportable L6 edge types implemented                               MET     §6
a typed deterministic consistency report exists                         MET     §7
L7 and L8 consume that report                                           MET     §8
L7 has locked boundary/type/assertion/candidate contracts               MET     §9
decisions outside locked options are rejected                           MET     §9
the deterministic no-model fallback is explicit                         MET     §9
inference run manifests are deterministic and text-free                 MET     §10
old resolver behaviour characterized before migration                   MET     §12
useful old behaviour is migrated                                    NOT MET     §12
obsolete resolver.py is deleted and unimportable                    NOT MET     §12
one canonical L4 public entry point remains                             MET     §11
Dockerfile and requirements.lock are validated                          MET     §13
Docker built and smoke-tested offline when the environment supports it  BLOCKED §14
a strict code-bearing evaluation contract without fabricated labels     MET     §15
deterministic and specialist modes remain operational                   MET     §17.7
full mode remains fail-closed                                           MET     §17.7
E3 checkpoint remains byte-identical                                    MET     §2
no pretrained model was downloaded                                      MET
no training or fine-tuning occurred                                     MET
internal_test was not accessed                                          MET
no organizer inference or output.zip was produced                       MET
tests and static checks pass                                            MET     §16
Audit 0054 contains real evidence                                       MET     §17
no commit or push occurred                                              MET
```

**Two criteria are NOT MET and one is BLOCKED, so the milestone is
`PARTIAL_MILESTONE` / `ACCEPTANCE_CRITERIA_NOT_MET`.**

## 23. Safe-to-commit verdict

```text
VERDICT: SAFE_TO_COMMIT as a PARTIAL milestone, explicitly labelled

Safe because every claim in this audit is measured and every gap is named. NOT safe to
describe as completing Milestone 2B: the old-L4 migration was not done and the container
was not built.
```

Safety checks: working tree contains only the paths in §18; no tracked file deleted; the
architecture PDF unmodified (§2); the E3 checkpoint unmodified (§2); no assistant control
file present or tracked (§1); no weights, governed data or generated artifacts staged;
`git diff --check` clean; 1729 tests pass; ruff, mypy strict and compileall clean.

## 24. Exact explicit staging and commit commands

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

# 2. Stage exactly the modified files
git add Dockerfile
git add src/mednorm_vi/confidence_cascade/__init__.py
git add src/mednorm_vi/evidence_graph/__init__.py
git add src/mednorm_vi/evidence_graph/graph.py
git add src/mednorm_vi/inference/cli.py
git add src/mednorm_vi/inference/packaging.py
git add src/mednorm_vi/inference/pipeline.py
git add src/mednorm_vi/linking/icd10.py
git add src/mednorm_vi/linking/rxnorm.py
git add src/mednorm_vi/metric_decoder/__init__.py
git add src/mednorm_vi/metric_decoder/decoder.py
git add docs/architecture/ACTIVE_RUNTIME_MANIFEST.md

# 3. Stage exactly the new files
git add requirements-image.lock
git add src/mednorm_vi/confidence_cascade/escalation.py
git add src/mednorm_vi/evaluation/code_linking.py
git add src/mednorm_vi/evidence_graph/consistency.py
git add src/mednorm_vi/inference/manifest.py
git add src/mednorm_vi/linking/icd10_hierarchy.py
git add src/mednorm_vi/linking/rxnorm_graph.py
git add src/mednorm_vi/linking/structured_medication.py
git add tests/unit/test_l5_l7_deterministic_stack.py
git add tests/unit/test_old_l4_characterization.py
git add docs/audits/0054-deterministic-l5-l7-run-manifest-and-container-audit.md
git add docs/audits/README.md

# 4. Confirm what is staged before committing
git status --porcelain
git diff --cached --stat

# 5. One commit
git commit -F - <<'MSG'
feat: complete the deterministic L5-L7 stack, run manifest and container audit

Partial Milestone 2B. Six of the seven carried-forward tasks are done, plus the
code-linking evaluation contract. The old-L4 migration is characterized but NOT
migrated or deleted, and the Docker build is blocked by stale Docker Hub
credentials in the operator environment. Acceptance criteria not met.

Both KB graphs were inspected before any traversal was written, and both turned
out weaker than the spec assumes: the generated indexes store unlabeled,
symmetric adjacency and discarded all 2,563,978 RxNorm relation names. So ICD
direction is reconstructed from code length and RxNorm direction from endpoint
TTY, and every emitted candidate records the path it was inferred along.

L5 RxNorm - consumes the E1 ComponentSpans that Audit 0052 preserved to L5 and
the linker had ignored since. Retrieval now queries the ingredient rather than
the whole surface; traversal walks IN -> SCDC -> SCD -> SBD; strength (with
cross-unit mass conversion), unit family, concentration, dose form, release and
route conflicts are hard negatives with distinct reason codes. Suppressed
concepts and grouper TTYs are never offered. A finding: `paracetamol` occurs in
zero records of the locked snapshot, so a 12-entry INN bridge marked
REQUIRES_CLINICAL_REVIEW widens retrieval only, and is recorded per decision.

L5 ICD-10 - hierarchy traversal plus the section 9.3 specificity controller. A
descendant outranks its ancestor only when the mention states the detail it
adds; otherwise it is suppressed with the missing tokens recorded and the
ancestor is the conservative fallback. Depth never wins on its own.
metadata.specificity equals len(code)-3 throughout the snapshot, so it is
reported and excluded from ranking.

L6 - all nine edge types, each with provenance, an evidence source and a tier.
`treats` requires an explicit trigger between the two mentions inside one
sentence-scale window; co-occurrence produces no edge and the declined pair is
recorded. New typed consistency contract: 12 rules, 4 verdicts, and uncertainty
is never encoded as false.

L7/L8 - L7 gains the full locked-option escalation contract. Decisions name
option ids, not values, so a code, label or coordinate that was not offered
cannot be returned; a refused decision degrades to the deterministic fallback.
No model is loaded and a test asserts the module imports none. L8 now consumes
both reports: a fatal contradiction can no longer be silently accepted, and
consistency verdicts appear in its reason codes.

Reproducibility - opt-in `--run-manifest` writes a deterministic, clinical-text-
free record, including when L9 stops the run. It immediately caught two real
defects: output.zip was not byte-reproducible (ZIP entry mtimes) and
requirements.lock could not install in the image (+cu126 is not on PyPI).
Both fixed; requirements-image.lock records the CPU-wheel deviation.

Evaluation - a strict code-linking contract that refuses to score without
human-adjudicated gold rather than returning zeros, and ships no gold.
Candidate quality remains UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD.

Evidence: Audit 0054. 1729 passed / 1 skipped (+104); ruff, mypy strict,
compileall, import sweep clean. Span/type F1 unchanged at 0.5559 / 0.5521,
which is the control. No model downloaded, no training, no internal_test, no
organizer inference, no output.zip.
MSG

# 6. Verify, do not push
git log -1 --stat
git status
```

The `docs/audits/README.md` line to add before staging it:

```text
- `0054-deterministic-l5-l7-run-manifest-and-container-audit.md`
```

## 25. Recommended next milestone

**UNBLOCK MEASUREMENT, THEN FINISH THE TWO LOOSE ENDS.** The order is forced by
dependency, not preference.

1. **Produce the code-bearing gold set** specified in
   `evaluation.code_linking.REQUIRED_ANNOTATION_ARTIFACT`. Even a few hundred
   human-adjudicated ICD-10 codes over existing governed DIAGNOSIS mentions turns 40% of
   the metric from unmeasurable into measurable. **Everything built in §4 and §5 is
   currently unvalidatable**, and that is the single most consequential fact in this
   repository.
2. **Clear the Docker credential blocker and build the image** (§14) — one `docker
   logout` by the owner, then the build and the offline smoke. Small, and it closes
   Appendix A.
3. **Migrate and delete `resolution/resolver.py`** (§12) — the characterization exists;
   what remains is porting the per-type boundary policy, moving
   `phase1c_foundation/cli.py` and `doctor.py` to the canonical L4, proving equivalence,
   then deleting.
4. **A governed INN ↔ RxNorm crosswalk at KB-intake time**, retiring the 12-entry
   stopgap. And while in the KB builder: **keep the relation labels** — both graphs
   discarded them, and labeled edges would turn §4's and §5's careful inferences into
   assertions.
5. **Fix the truncated ICD `canonical_name` extraction** — lexical retrieval cannot do
   better than the names it is given, and some are PDF fragments.
6. **Calibration (S6)**, which is the only thing that turns L8's honest deterministic
   decoder into spec §13.
7. **Assertion supervision**, for the other 30% of the metric.
8. **L6 global optimization** and the **L4 boundary-offset head**, both of which want
   measurable candidates first.

Explicitly **not** next, for the same reason spec §20 gives: the L7 model stages, broad
threshold optimization, and any organizer submission. Nothing should be tuned against a
metric that 40% of is still unmeasurable.

---

**Audit 0054 ends. No commit. No push. `PARTIAL_MILESTONE` /
`ACCEPTANCE_CRITERIA_NOT_MET`: the old-L4 migration was characterized but not performed,
and the container build is blocked by the operator's Docker credential store.**
