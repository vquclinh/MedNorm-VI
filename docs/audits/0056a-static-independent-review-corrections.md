# Audit 0056a — static independent-review corrections (Milestone 2D-A)

**This is a 2D-A audit, not the 0056 closure audit.** It records static source
corrections and targeted tests only. No notebook was executed, no model was loaded,
no corpus validation ran, no container was built. Audits 0001-0055 are append-only
and were not edited.

- **Scope:** Milestone 2D-A — static architecture corrections
- **Date:** 2026-07-30
- **Preceded by:** Audit 0055 (framework closure) and an independent Codex review
  that challenged it with eleven findings
- **Followed by:** Milestone 2D-B (one light validation pass), then 2D-C (isolated
  container verification)

---

## 1. Initial clean Git state

Verified before any edit, and again before writing this audit:

```text
branch                                   main
HEAD                                     bbc3d02 feat: close the framework and verify the offline container
git status --short --untracked-files=all (empty)
git rev-list --left-right --count origin/main...HEAD   0    0
```

Protected artifacts, hashed before and after the milestone — both unchanged:

```text
docs/MedNorm-VI_Architecture.pdf
  0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b   (1,304,976 bytes)

checkpoint/s1_mention_full_training_v1/best.pt
  a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c   (1,615,513,303 bytes)
```

The PDF was neither modified nor moved. The checkpoint was hashed by streaming
`sha256sum` only; it was never opened by `torch.load` and never deserialized.

**Nothing was staged, committed, pushed or amended.** The repository owner alone
performs every Git operation.

## 2. Memory-safety compliance

The previous session exhausted RAM and crashed the machine. This milestone observed
the absolute limits without exception.

```text
lowest available RAM observed            6.6 GiB
highest available RAM observed           8.4 GiB
floor that would have stopped work       3.0 GiB   (never approached)
threshold below which no command starts  4.0 GiB   (never violated)
swap growth                              none observed
```

`free -h` was run before every command with a meaningful footprint. Work was strictly
sequential — never two expensive commands at once, no parallel workers, no `pytest -n`.

Not done, per the milestone's absolute limits: notebook execution, Jupyter or any
kernel, E3 loading, `torch.load`, transformer or tokenizer loading, CUDA
initialization, training, the 200-document validation, corpus-wide inference,
`internal_test` access, Docker build, container runs, the full pytest suite,
repository-wide mypy, and whole-package import sweeps.

One nuance worth stating precisely: E3 readiness calls `importlib.util.find_spec`,
which resolves a module's location **without importing it**. Confirming that `torch`
and `transformers` are installed therefore cost no memory and initialized no CUDA.

## 3. Codex findings reproduced

All eleven findings were reproduced by static evidence in the preceding
context-acquisition turn. Two carried nuance the original report omitted, one was
broader than reported, and one was partly over-scoped. Dispositions after 2D-A:

| # | Finding | Reproduced | 2D-A disposition |
| --- | --- | --- | --- |
| A | Full-profile readiness may report READY for enabled-but-unregistered E5/E6/E7; fake checkpoint dirs satisfy path-only checks | CONFIRMED | **Fixed** (§4, §5) |
| B | Learned L4 v2 has a slot but is not dispatched | CONFIRMED, and broader — **both** L4 flags were inert | **Fixed** (§6) |
| C | E4 remains in executable cells in two notebooks | CONFIRMED for both; S1 `phobert_alignment` hits were **not** E4 | **Fixed** (§9) |
| D | Final L9 packaging lacks serialized schema and offset/text validation | CONFIRMED | **Fixed** (§7) |
| E | L9's candidate-not-offered rule never receives offered sets | CONFIRMED | **Fixed** (§8) |
| F | ACTIVE_RUNTIME_MANIFEST contains contradictory/stale statements | CONFIRMED — three distinct contradictions | **Fixed** (§13) |
| G | `specialist_requires_checkpoints` declared but unread | CONFIRMED | **Fixed** (§4) |
| H | Docker uses a mutable base tag, not a digest | CONFIRMED | **Deferred to 2D-C** (§16), documented in place |
| I | E3 uses `torch.load(..., weights_only=False)` | CONFIRMED, with digest verification already preceding it | **Source hardening done** (§10); runtime verification deferred to 2D-C |
| J | Output-directory writing recursively deletes without a guard | CONFIRMED at function level | **Fixed** (§11) |
| K | `mention_factory/adapters.py`, stale docstrings, obsolete configs | CONFIRMED | **Fixed** (§12, §13) |

### The precise defect behind finding A

Three mechanics combined, and each alone was survivable:

1. `expected` was built by iterating `MODE_EXPERTS["full"]`, which lists only
   E1/E2/E3. `_expert_readiness` — the only function that asks whether an expert can
   actually run — was therefore **never called** for an enabled E5, E6 or E7.
2. `flags_for_mode` in `full` mode turned on *any* `enable_e*` flag the profile set,
   so those experts were scoped ON and handed to the runner.
3. Their only check was `Path.exists()` on a role directory. **`Path.exists()`
   returns `True` for an empty directory**, so `mkdir -p` was sufficient.

Downstream, `run_registered_experts` iterates the registry, so an unregistered expert
produced no proposals, raised nothing, and received **no `ExpertRunRecord` at all** —
it was invisible even in the run record. Net effect: READY, with silent absence.

That contradicted the repository's own stated intent. `mention_factory/experts/__init__.py`
already documented why E5/E6/E7 are deliberately unregistered ("registering it would
make an untrained head reachable from a profile"). The intent was right; readiness
did not honour it.

## 4. Readiness corrections

`src/mednorm_vi/inference/config.py` was substantially rewritten.

| Correction | Before | After |
| --- | --- | --- |
| Which experts are evaluated | `MODE_EXPERTS[mode]` — E1/E2/E3 in every mode | `expected_expert_flags()` reads the **scoped flags**, so every enabled expert is evaluated |
| Enabled but unregistered | invisible; profile could be READY | **NOT_READY** with a named implementation blocker quoting the reason and the remedy |
| Artifact check | `Path.exists()` | `artifact_is_valid()` — a directory artifact must contain a non-empty entry matching declared patterns |
| `specialist_requires_checkpoints` | parsed, never read | enforced in `specialist` mode |
| Declared role owned by an expert | n/a | resolved **through that expert**, because the adapter is the authority on where its asset lives |
| Spec/registry drift | possible, silent | reported by name (`"the specification and the L3 registry have drifted"`) |
| Degradation | any unready expert could degrade | an expert with **no implementation** is never "degraded" — there is nothing to fall back from, so it errors in every mode |

Readiness now also reports `l4_route`, so an operator can see which resolver a run
will use.

### `assertion/assertion_hydra` removed from the specialist requirement

Owner decision: keep and enforce `specialist_requires_checkpoints`, but correct it to
list only real checkpoint-backed dependencies.

L5's wired assertion path is deterministic A1-A3 cue/scope logic in
`specialists/assertion/`. It has no checkpoint and the architecture plans none for it
— the future learned slot is the S2 head, a different artifact. Enforcing the list as
written would have made `specialist` permanently NOT_READY for an asset nobody
intends to produce. A false negative is no more honest than a false positive.

`full_requires_checkpoints` retains the role: `full` is the everything-trained
profile, and the S2 head is a genuine future checkpoint. **Observation for a later
milestone:** the role is named `assertion/assertion_hydra` rather than
`assertion/s2_assertion`, which reads as though the deterministic Hydra path itself
had a checkpoint. Renaming it is a config-contract change and was left out of 2D-A
scope; it is recorded in §18.

### Measured effect on the shipped profile

```text
mode            before 0056a                     after 0056a
deterministic   READY                            READY
specialist      READY                            READY          (route: deterministic_v1)
full            NOT_READY, 12 blockers           NOT_READY, 11 blockers
```

`specialist` remains READY with the real E3 assets, as required. The `full` count
moved from 12 to 11 because `assertion/assertion_hydra` is no longer double-counted
across the two lists; the blockers are now *named implementation* blockers for
E5/E6/E7 rather than bare missing-path strings.

## 5. Expert implementation / registry / asset matrix

The new single declaration is `src/mednorm_vi/mention_factory/expert_spec.py`
(`l3-expert-spec-v1`). It covers, per expert: id, enable flag, registration state,
implementation readiness, checkpoint role, artifact kind and validation, model
revision requirement, parameter-count status, lazy construction, runtime invocation
and the exact fail-closed sentence.

| Expert | Flag | Implementation | Registered | Checkpoint role | Artifact | Revision | Param count | Enabling it today |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 medication grammar | `enable_e1_medication_grammar` | `DETERMINISTIC_PHASE1B` | not required | — | none | no | `NOT_APPLICABLE` | runs |
| E2 laboratory parser | `enable_e2_laboratory_parser` | `DETERMINISTIC_PHASE1B` | not required | — | none | no | `NOT_APPLICABLE` | runs |
| E3 ViHealthBERT | `enable_e3_vihealthbert` | `REGISTERED` | **yes** | `mention/vihealthbert` | dir with `*.pt`/`*.safetensors`/`*.json` | yes | `NOT_VERIFIED` | runs (lazy; digest-verified) |
| E4 PhoBERT-W2NER | — | **RETIRED** | refused by construction | — | — | — | — | flag refused at load |
| E5 XLM-R MRC | `enable_e5_xlmr_mrc` | `NOT_IMPLEMENTED_OR_NOT_REGISTERED` | no, deliberately | `mention/xlmr_mrc` | dir with weights | yes | `NOT_VERIFIED` | **NOT_READY**, named blocker |
| E6 GLiNER | `enable_e6_gliner` | `NOT_IMPLEMENTED_OR_NOT_REGISTERED` | no, deliberately | `mention/gliner` | dir with weights | yes | `NOT_VERIFIED` | **NOT_READY**, named blocker |
| E7 Qwen proposer | `enable_e7_qwen_proposer` | `NOT_IMPLEMENTED_OR_NOT_REGISTERED` | no, deliberately | `mention/qwen3_1_7b_proposer` | dir with weights/config | yes | `NOT_VERIFIED` | **NOT_READY**, named blocker |

**No registration was added to make readiness pass.** E5's task head is randomly
initialized; registering it would make an untrained head reachable from a profile,
which spec §6 and the project's fail-closed rule both forbid. E6 and E7 have no local
weights and no complete runtime contract. All three are declared `IMPLEMENTATION_ABSENT`
precisely so that enabling one is loud.

The blockers are quotable sentences, not codes. For example, E5's:

> E5 has no registered expert: its MRC task head is randomly initialized and no
> trained checkpoint exists, so registering it would make an untrained head reachable
> from a profile (spec §6). Train S5/E5 and register an adapter before enabling this
> flag

## 6. Learned-L4 dispatcher

New module `src/mednorm_vi/resolution/learned_dispatch.py` (`l4-learned-dispatch-v1`).

**Both L4 flags now have real semantics, and no inert flag remains.** They select a
route, and exactly one must be selected:

```text
enable_l4_deterministic_v1: true   ->  deterministic canonical L4 (shipped default)
enable_l4_learned_v2: true         ->  learned route, fail-closed
both true                          ->  ValueError at load: ambiguous L4 route
both false                         ->  ValueError at load: no L4 route selected
```

"Neither" is a configuration error rather than a mode, because L4 is not optional:
L5-L9 consume `EntityHypothesis` and no other stage produces it.

`configs/pipeline/full_v1.yaml` had `enable_l4_deterministic_v1: false` while the
canonical deterministic resolver ran on every document. The profile was a false
statement about its own behaviour. It is now `true`, in the top-level block and in
all three nested per-mode blocks. **The executed behaviour is unchanged** — the same
resolver runs — so no measurement recorded in Audits 0034/0052/0053/0054 is affected.

### Dispatch structure

`resolve_lattice_to_hypotheses` gained an optional `learned_backend`. There is still
**one canonical L4 entry point and one `EntityHypothesis` output contract**; the
existing AST test asserting a single entry point still passes unmodified.

```text
inference/pipeline.resolve_l4_backend(config)
    route == deterministic  ->  None            -> canonical.resolve_lattice_to_hypotheses (deterministic)
    route == learned        ->  backend OR RAISE -> canonical.resolve_lattice_to_hypotheses (learned)
```

`build_learned_l4_backend` has no code path that returns `None` or yields a
deterministic resolver. The no-silent-fallback guarantee is structural, not
documented. Its error message states the fact explicitly:

> learned L4 v2 is enabled but cannot run: no usable learned checkpoint (...). The
> deterministic L4 is NOT substituted — a profile that selects the learned route must
> either supply a valid checkpoint or disable the flag

Readiness surfaces the same fact **before** a run starts, as
`l4_learned_not_ready:learned_l4_no_checkpoint_configured:...`, so the failure is not
discovered on document 1 of 100.

### Learned output is untrusted

`_adapt_result` re-verifies every returned span against `original_text` and rejects
the **document** on the first violation — out-of-range coordinates, a text/offset
mismatch, or an accepted hypothesis with no type. A mis-trained or corrupted
checkpoint fails closed at L4 rather than emitting coordinates L9 would have to catch.

Dispatch was verified with an injected stub backend. **No model was loaded, trained or
fabricated.**

## 7. Final L9 serialized validation

New module `src/mednorm_vi/validator/final_gate.py` (`l9-final-serialized-gate-v1`),
called from `inference/pipeline._gate_final_output` inside `run_input_dir`, before
anything is written.

### What was actually missing

Audit 0053 wired a *candidate-membership* gate. It received the serialized JSON and
the snapshots — and never the source document. `validate_organizer_document` takes
only parsed JSON, so it **structurally cannot** check the offset invariant. Meanwhile
`validator.validate_document(original_text, entities)`, which does, had **no caller
anywhere in the repository** outside its own package.

So Appendix A's "every entity satisfies `original_text[start:end] == text`" was
enforced four times upstream — expert, registry, lattice, L4 — and not once on the
bytes that would be submitted. Serialization is precisely the stage those four checks
do not cover.

### What the gate now receives and validates

Receives: original source text, final serialized entities, document identity, frozen
ontology snapshots, and offered codes aligned with the final entities.

Validates, in one pass over the serialized payload:

| Check | Issue code |
| --- | --- |
| safe filename (`{digits}.json`, no separators, no traversal) | `final.unsafe_filename`, `final.unexpected_filename` |
| UTF-8 / JSON shape, root is a list | `final.bad_json`, `final.root_not_list` |
| organizer schema, exact required/forbidden fields per type, type vocabulary, assertion applicability, candidate syntax and ontology | `organizer.*` |
| end-exclusive offsets inside the document | `final.invalid_span`, `final.span_outside_document` |
| **exact `original_text[start:end] == text`** | `final.text_offset_mismatch` |
| duplicate candidates within an entity | `final.duplicate_candidate` |
| snapshot membership and correct ontology per type | `kb.candidate_not_in_snapshot`, `kb.*` |
| **candidate membership in the exact offered set** | `kb.candidate_not_offered` |
| deterministic emission order (non-decreasing `(start, end)`) | `final.nondeterministic_order` |

`final.text_offset_mismatch` deliberately reports **lengths and coordinates, never the
clinical text**, so a validation log carries no patient data.

The order check locks in an invariant that already holds by construction —
`build_span_lattice` sorts its merged proposals, L4 preserves that order, L8 and the
serializer iterate — rather than imposing a new requirement. That is what makes
Appendix A's "deterministic ordering" checkable on a single payload instead of only by
running the pipeline twice.

**Nothing is repaired.** The gate reports; the caller raises `FinalValidationError`;
a violating run leaves no `output/` and no `output.zip`. The CLI returns exit code 3
and the run manifest records `l9_stopped_the_run: true`.

## 8. Offered-set wiring (spec P7)

`validate_document_candidates` has always accepted `offered_codes_by_index`,
implementing "a model may select from retrieved candidates, never introduce a code".
The canonical caller never passed it, so `allowed is None` and the rule was a no-op on
every real run — a unit contract that had never executed on real output.

Per owner decision, enforcement is **immediate and fail-closed**; there is no
warn-only or record-only transition.

### Where the offered set comes from, and why

It is captured from `LinkerResult` at L5 linking — **not** from L8's surviving
candidates:

```python
offered_by_mention = {
    result.mention_id: tuple(c.code for c in result.candidates) for result in links
}
```

Taking it from L8's survivors would make the check tautological: a code L8 introduced
would appear in its own audit trail and validate against itself. Taking it from the
retriever is what makes a **post-L8 injection detectable**, which is the scenario the
rule exists for and which is now covered by an integration test.

Alignment is by construction: `to_entity_predictions` preserves the decoder's order
and `to_submission_json` preserves the list, so index *i* of the payload is index *i*
of the decoded entities.

**A missing offered set fails.** An entity that emits candidates with no recorded
offered set raises `final.offered_set_missing` rather than passing by default —
otherwise a future wiring regression would silently disable P7 again, exactly as
before.

## 9. E4 notebook cleanup

Static JSON editing only. **No notebook was executed, no kernel started, no stored
output regenerated.** Legitimate E3 `phobert_alignment` infrastructure was not touched.

### `notebooks/MedNorm_Phase2_Validation_Ablation.ipynb`

Removed from executable cells: `E4_FULL_ARTIFACT`, the `validate_e4_artifact` import,
the `"e4"` artifact-report entry, the `e4_phobert_w2ner` checkpoint path, and the
`enable_e4_phobert_w2ner` flag. The declared ablation-arm set is now E3, E5 and
learned L4 v2 — architecture-valid components only.

Two facts worth recording, because they show the arm was not merely stale but broken:

- **`validate_e4_artifact` does not exist.** `training/phase2/artifacts.py` defines
  `validate_e5_artifact` and `validate_l4_artifact` only; the E4 validator went with
  the retirement in Audit 0048. The cell raised `ImportError` before doing anything.
- **`enable_e4_phobert_w2ner` is actively refused.** `PipelineConfig.load` raises on
  any profile declaring it, so the flag could only have produced an unloadable config.

### `notebooks/MedNorm_S0_DomainAdaptation.ipynb`

Per owner decision, **not retargeted**. The E4 workflow was removed and the affected
section converted to an explicit architecture-neutral `DESIGN_DRAFT`.

Removed: `MODEL = "vinai/phobert-base-v2"`, `OUT = CHECKPOINT_ROOT + '/' + "mention/phobert_w2ner"`,
the manifest `role: "mention/phobert_w2ner"`, and the instruction to copy results into
`models/checkpoints/full_v1/mention/phobert_w2ner`.

Replaced with `S0_TARGET_MODEL` / `S0_TARGET_REVISION` / `S0_CHECKPOINT_ROLE`, all
empty, and a `raise SystemExit` when any is unset. The notebook header states that a
target may be selected only after:

1. code-bearing and assertion evaluation data exist;
2. the final ≤9B profile is selected;
3. exact model revisions and licences are approved.

Retargeting to ViHealthBERT or XLM-R would have been a model-selection decision taken
without evidence, and S0's purpose is to adapt whichever backbone the final profile
uses — choosing one here would prejudge that.

### Static test

`test_no_executable_notebook_cell_references_active_e4` parses every notebook's JSON,
inspects **code cells only**, strips comments, and fails on any of
`enable_e4_phobert_w2ner`, `validate_e4_artifact`, `E4_FULL_ARTIFACT`,
`mention/phobert_w2ner`, `e4_phobert_w2ner`, `phobert-base`. Comments that record the
retirement are explicitly allowed; live code is not.

### What was deliberately NOT changed

The `MedNorm_S1_*` notebooks import `mednorm_vi.training.phobert_alignment`. This is
**not** an E4 remnant. `demdecuong/vihealthbert-base-word` declares
`tokenizer_class = PhobertTokenizer`, which has no fast implementation, so the module
reconstructs the offset mapping E3's supervision depends on. It is active E3
infrastructure and a test now asserts it remains present.

## 10. Checkpoint-loading source hardening

New module `src/mednorm_vi/mention_factory/neural/checkpoint_loading.py`.

The E3 path always verified the checkpoint's SHA-256 **before** loading, so finding I
was never an open door. The real weakness was that the ordering was a convention held
by one call site, and nothing prevented a future checkpoint from being loaded the same
way with no digest at all. A guarantee that depends on remembering to hash first is
not a guarantee.

One policy was split into two:

| Path | Admits | `weights_only` | Why |
| --- | --- | --- | --- |
| `TRUSTED_LEGACY_PICKLE` | **only** the file whose digest equals `TRUSTED_LEGACY_E3_SHA256` | `False` | E3's `best.pt` predates safetensors and carries non-tensor metadata (`mode`, `entity_type_order`, `pinned_model_revision`) that `weights_only=True` cannot read. The allowance is by identity, named and limited to one artifact |
| `SAFETENSORS` | any `.safetensors` | `True` | safe by format; the reader executes no code |
| `WEIGHTS_ONLY` | every other checkpoint | `True` | the legacy allowance must not become the house style |

Two invariants are now enforced rather than documented:

1. **an unverified checkpoint can never reach `weights_only=False`** — the caller must
   pass an already-computed digest, so there is no ordering left to get wrong;
2. **a pinned digest that disagrees with the file on disk is refused** here as well as
   at fingerprint time.

`runtime.py` now calls `assert_load_is_permitted(...)` and passes
`weights_only=policy.weights_only`. A source-level test asserts no bare
`weights_only=False` survives in executable code on the E3 path. The chosen policy is
recorded in the backend record, so a run manifest shows that the legacy allowance was
used and for which artifact.

**Deferred to 2D-C, as instructed:** `weights_only=True` compatibility was **not**
tested, because that requires loading a real checkpoint and model loading is
prohibited in 2D-A. Runtime verification that E3's forward pass still produces
identical output under this policy is recorded as 2D-C work (§16). All tests here use
tiny synthetic files in `tmp_path`; the real 1.6 GB artifact was never read.

## 11. Output-directory safety

`write_output_directory` began with `if root.exists(): shutil.rmtree(root)` — no guard
of any kind — and is exported, so any caller passing any path destroyed it. The
canonical runner bounded the damage by forcing the *name* to `output`, but a
pre-existing unrelated `output/` beside the requested ZIP was still deleted silently,
and a direct caller had no protection at all.

`assert_safe_output_directory` now runs **before anything is removed** and refuses:

| Refused | Reason |
| --- | --- |
| filesystem root | catastrophic |
| user home directory | catastrophic |
| repository root | catastrophic |
| a symlink, or a path reached through one | deleting through a link destroys the target, not the link |
| any directory not named `output` | the workflow only ever writes to a directory it owns |
| an existing directory holding files this workflow did not write | a packaging run must never destroy unrelated data |
| an existing non-directory destination | — |

Permitted: an `output` directory that is absent, empty, or contains only numbered
prediction JSON — i.e. what a previous packaging run left. Normal inference behaviour
is preserved, including re-running over a previous output directory. Tests use
temporary directories exclusively.

## 12. Dead-code decisions

### Deleted: `src/mednorm_vi/mention_factory/adapters.py`

Reconfirmed before deletion that its only real importer was the legacy test path. The
`data_engine/cli.py` hit found by grep resolves to `data_engine/adapters.py` via
relative import — a different module.

It was a parallel, obsolete adapter layer: `NeuralExpertAdapter.propose` always
raised, `default_neural_adapters` duplicated the registry's job on the **wrong
proposal type** (`mention_factory.models.SpanProposal` instead of
`lattice.models.ExpertSpanProposal`), and its docstring pointed at
`mention_factory.qwen_proposer`, a module that does not exist. It had no active
architecture role.

No alias, shim, stale import, `qwen_proposer` reference or unused `fallback_to_empty`
behaviour was left behind. Three tests assert this: the file is absent, the module is
unimportable (`ModuleNotFoundError`), and no tracked source references it.

### Migrated tests

`tests/unit/test_full_pipeline_v1.py::test_missing_checkpoint_and_anchored_qwen_contract`
was replaced by two tests against the canonical contracts:

- `test_an_enabled_expert_without_its_asset_fails_closed_by_name` — exercises
  `ExpertRegistry` / `ExpertRegistration` / `run_registered_experts`, asserting that an
  enabled-but-unready expert raises `ExpertNotReady` naming the expert, role and exact
  missing path, and that a disabled expert is never constructed;
- `test_offsets_for_a_text_only_source_are_resolved_not_invented` — replaces the
  legacy `anchor_substring` helper with `mention_factory.offsets.resolve_occurrence`,
  the real path for a proposal-only source, including its refusal of an ambiguous
  surface form.

### Kept: E5/E6/E7 typed-slot YAML

`configs/mention_factory/{xlmr_mrc_ner_v1,gliner_v1,qwen_proposer_v1}.yaml` were
inspected, **not deleted**. They are legitimate typed-slot configuration: each records
the expert id, `status: IMPLEMENTED_UNTRAINED` / `UNAVAILABLE_UNTRAINED`,
`enabled: false`, an unpinned-revision placeholder and a planned parameter count that
a future download must satisfy.

**They are currently consumed by no source module.** That is the same defect class this
audit closed elsewhere, so rather than leave them unread, a test now consumes them: a
slot file naming an expert the canonical specification does not declare, or claiming
`enabled: true` while the expert is unimplemented, now fails.

## 13. Active-documentation corrections

Audits 0001-0055 were **not** edited. Corrections were applied to active documents only.

| Document | Correction |
| --- | --- |
| `ACTIVE_RUNTIME_MANIFEST.md` line 70 | Said "**The image has still never been built**" while line 82 of the same table said it was built and smoke-tested in Audit 0055. Corrected, and the not-yet-digest-pinned base recorded |
| `ACTIVE_RUNTIME_MANIFEST.md` §4 | Said `resolution/resolver.py` "still exists as a second, non-canonical implementation pending test migration", contradicting the same table's statement that Audit 0055 deleted it. The file is genuinely absent; the stale line is gone |
| `ACTIVE_RUNTIME_MANIFEST.md` §4 | Listed `configs/resolution/resolver_v1.yaml` as a configuration source. That file was deleted in Audit 0055 |
| `ACTIVE_RUNTIME_MANIFEST.md` §3, §9, headline | E5/E6/E7 rows now record that enabling them is a named NOT_READY blocker; the L9 rows record the complete final gate and real offered-set wiring; six new headline rows record the 0056a corrections |
| `Dockerfile` header | Credited Audit 0054 with building and smoke-testing the image; 0054 *attempted* the build and 0055 performed it. Also now states plainly that the base is **not yet digest-pinned** and that 2D-C will pin it |
| `validator/submission.py` | "KB membership of candidate codes remains deferred (no frozen KB yet)" — false since Audit 0014 froze the snapshots and 0053 added `kb_membership.py`. Replaced with an accurate statement of what this module can and cannot see |
| `validator/__init__.py` | Described `submission.py` as a "STUB"; it has not been one since Audit 0002. Now lists the KB-membership and final-gate modules too |
| `mention_factory/laboratory/synthetic.py` | Cited the deleted `configs/resolution/resolver_v1.yaml` as the tracked deterministic default. Now names the surviving key `boundary.group_preference.test_result` |
| `configs/pipeline/full_v1.yaml` | The `profiles.specialist.boundary_type_resolver_v1` block claimed to DISABLE L4 v1. Nothing read it (`PipelineConfig.load` consumes only top-level `feature_flags`), and the canonical L4 ran regardless. Replaced with an accurate `l4_measurement_note` that retains the real Audit-0034 numbers and states `runtime_effect: none` |

`phase1c_foundation/doctor.py`'s mention of `resolver_v1.yaml` was **left alone**: it
is a truthful historical note explaining what replaced it, not a stale active reference.

## 14. Targeted tests run

Only tests directly covering modified areas were run, sequentially. **The full suite
was not run** — that is 2D-B.

```text
tests/unit/test_static_review_corrections_0056a.py        54 passed   (new)
tests/unit/test_full_pipeline_v1.py                        7 passed   (migrated)
tests/unit/test_framework_closure.py                      ---.
tests/unit/test_canonical_l3_l4_spine.py                     |- 67 passed
tests/unit/test_e3_canonical_integration.py               ---'
tests/unit/test_e4_retirement.py                          included above, 116 with closure
tests/unit/test_e5_s2_readiness_and_budget.py             ---.
tests/unit/test_phase2_flags_registry_notebooks.py           |- 65 passed
tests/unit/test_l4_learned_v2.py                          ---'
tests/unit/test_notebooks.py                              ---.
tests/unit/test_notebook_execution.py                        |- 112 passed (static only)
tests/unit/test_notebook_placeholders.py                  ---'
tests/unit/test_l5_l7_deterministic_stack.py              ---.
tests/unit/test_organizer_output_contract.py                 |- 179 passed
tests/unit/test_migrated_contracts.py                        |
tests/unit/test_organizer.py                              ---'
tests/unit/test_neural_expert.py                          ---.
tests/unit/test_phase1c_doctor.py                            |- 26 passed
                                                          ---'
tests/integration/                                        76 passed
```

Lint and compile, changed files only:

```text
ruff check <17 changed .py files>                         All checks passed
python -m compileall -q src/mednorm_vi/{inference,resolution,validator,mention_factory} tests/unit   OK
git diff --check                                          clean
```

### The new suite, by finding

`tests/unit/test_static_review_corrections_0056a.py`, 54 tests:

- **A (7):** enabled-and-unregistered ⇒ NOT_READY; empty directory ⇒ NOT_READY; a
  *fabricated but non-empty* artifact still fails on the missing implementation;
  registered-but-missing-asset ⇒ NOT_READY; a disabled expert is never constructed;
  spec↔registry drift in both directions.
- **G (3):** specialist requirements enforced; the shipped list names only real
  checkpoint-backed assets; specialist stays READY with the real E3 asset.
- **B (9):** neither-route and both-routes refused at load; the shipped profile
  selects deterministic; the deterministic route builds no backend; missing checkpoint
  fails closed *and* is a readiness blocker; an injected stub is reached; both routes
  return one contract; invalid learned output rejected (text mismatch, out of range).
- **D/E (11):** a valid entity passes; offset/text mismatch fails; span outside the
  document fails; snapshot-valid-but-unoffered fails; invented code fails; wrong
  ontology fails; duplicate candidate fails; missing offered set fails; out-of-order
  entities fail; unsafe filenames fail; the gate raises and names violations.
- **E on the canonical path (3):** a **post-L8 candidate injection** stops packaging;
  a stopped run leaves **no `output/` and no `output.zip`**; the runner carries source
  text and offered sets.
- **I (6):** only the trusted-legacy digest reaches `weights_only=False`; an
  unrecognised checkpoint is forced to safe loading; an unverified checkpoint cannot
  load at all; a digest mismatch is refused; safetensors is safe by format; no bare
  `weights_only=False` in E3 executable source.
- **J (6):** refuses a non-`output` name, an `output` holding unrelated files, a
  symlink, home, filesystem root, the repository root; accepts absent and
  previous-output directories.
- **C (4):** no executable notebook cell references active E4; S0 is an explicit
  `DESIGN_DRAFT` that fails closed; the ablation notebook declares only valid arms;
  E3's `phobert_alignment` is untouched.
- **K (4):** the legacy module is absent, unimportable, unreferenced; the E5/E6/E7
  slot configs agree with the canonical specification.

### Pre-existing tests updated, and why

Two assertions encoded the untruth they were meant to guard, and were corrected:

- `test_framework_closure.py::test_the_learned_l4_v2_stays_disabled_and_fail_closed`
  asserted **both** L4 flags were `False` — while the deterministic L4 ran on every
  document. It now asserts the route semantics and that the learned slot reports a
  blocker.
- `test_phase2_flags_registry_notebooks.py::test_phase2_default_feature_flags_preserve_measured_baseline`
  asserted `enable_l4_deterministic_v1 is False` for the same reason. The measured
  baseline is unaffected: the same resolver executes.

No test was weakened or deleted to make a change pass.

## 15. Tests deliberately deferred to 2D-B

- the **full** unit suite in one pass (`pytest tests/`);
- `ruff check .` and `ruff format --check .` over the whole repository;
- `mypy src/mednorm_vi` (scoped, once — never repository-wide);
- `python -m compileall` over the whole tree;
- lightweight import/static scans across all packages;
- review of the Audit 0056 claims.

Anything outside the modules changed here was intentionally not exercised, so a
regression in an untouched package would surface in 2D-B rather than being masked by
selective running now.

## 16. Container work deliberately deferred to 2D-C

- **digest-pinning the base image** (finding H). `Dockerfile` still names the mutable
  tag `python:3.14.5-slim-bookworm`. Obtaining the digest requires a network pull,
  which 2D-A forbids. The Dockerfile now states the gap in place;
- the single authorized `docker build`;
- deterministic offline smoke under `--network=none`;
- specialist/E3 smoke on one small tracked fixture;
- read-only checkpoint behaviour (EROFS from inside the container);
- **verification that the new final L9 gate fires inside the container**, including
  the offset/text and offered-set checks;
- **runtime verification of the checkpoint-loading policy** — that E3 loads under
  `assert_load_is_permitted` and produces identical output. This is the one part of
  finding I that source hardening cannot prove.

## 17. Exact changed-file inventory

Created (5):

```text
src/mednorm_vi/mention_factory/expert_spec.py
src/mednorm_vi/mention_factory/neural/checkpoint_loading.py
src/mednorm_vi/resolution/learned_dispatch.py
src/mednorm_vi/validator/final_gate.py
tests/unit/test_static_review_corrections_0056a.py
```

Deleted (1):

```text
src/mednorm_vi/mention_factory/adapters.py
```

Modified (18):

```text
Dockerfile
configs/pipeline/full_v1.yaml
docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
notebooks/MedNorm_Phase2_Validation_Ablation.ipynb        (static JSON edit; never executed)
notebooks/MedNorm_S0_DomainAdaptation.ipynb               (static JSON edit; never executed)
src/mednorm_vi/inference/cli.py
src/mednorm_vi/inference/config.py
src/mednorm_vi/inference/packaging.py
src/mednorm_vi/inference/pipeline.py
src/mednorm_vi/mention_factory/laboratory/synthetic.py
src/mednorm_vi/mention_factory/neural/runtime.py
src/mednorm_vi/resolution/canonical.py
src/mednorm_vi/validator/__init__.py
src/mednorm_vi/validator/submission.py
tests/unit/test_framework_closure.py
tests/unit/test_full_pipeline_v1.py
tests/unit/test_phase2_flags_registry_notebooks.py
docs/audits/0056a-static-independent-review-corrections.md   (this file, new)
```

```text
git diff --stat   18 files changed, 865 insertions(+), 522 deletions(-)
                  (plus 5 untracked new files)
```

## 18. Unresolved issues

1. **The base image is not digest-pinned** (finding H). Deferred to 2D-C; requires a
   network pull. Until then Appendix A's reproducible private rebuild is not
   guaranteed.
2. **`weights_only=True` is unverified at runtime** (finding I). The policy is
   enforced in source and unit-tested with synthetic files; whether E3 loads
   identically under it is a 2D-C question.
3. **`full_requires_checkpoints` names `assertion/assertion_hydra`.** The role reads
   as though the deterministic Assertion Hydra had a checkpoint; the real future
   artifact is the S2 head. Renaming it to `assertion/s2_assertion` is a
   config-contract change and was left out of 2D-A scope.
4. **The `profiles:` block in `full_v1.yaml` is still largely unread.**
   `PipelineConfig.load` consumes only the top-level `feature_flags`; the nested
   per-mode `feature_flags` are scanned for retired E4 flags but otherwise unused, and
   mode scoping comes from `MODE_EXPERTS`. The misleading `boundary_type_resolver_v1`
   sub-block was corrected, but the structural question — should nested profiles be
   read, or removed? — is open.
5. **E3's checkpoint role resolution is indirect.** `mention/vihealthbert` in
   `specialist_requires_checkpoints` is satisfied through the E3 adapter rather than by
   an artifact under `checkpoint_root`, because that is where the real file lives. This
   is correct today but means the role string is a label rather than a path for E3
   alone.
6. **`final.nondeterministic_order` encodes an invariant, not a spec requirement.**
   The organizer has not confirmed that entity order matters. The check locks in
   existing behaviour; if ordering is later confirmed irrelevant it can be downgraded
   to a warning.
7. **No packaging-stress suite** (spec §18.3) still exists, and `output.zip` has never
   been produced for a real submission.
8. Everything data-blocked from Audit 0055 §13 remains blocked: zero ICD-10 codes,
   zero RxCUIs, zero assertion labels in the governed corpus.

## 19. Verdict

```text
SAFE_FOR_2D_B
```

Rationale:

- every 2D-A scope item (A-G) is implemented, and each is held by a targeted test;
- all targeted suites pass: 54 new tests plus every pre-existing suite covering a
  modified area, and the full integration directory (76 tests);
- `ruff check` passes on all changed files; `compileall` succeeds; `git diff --check`
  is clean;
- both protected artifacts hash identically to the start of the milestone;
- no notebook, model, corpus validation, container or Docker workload ran at any point;
- memory never fell below 6.6 GiB available, far above the 3 GiB stop threshold;
- the two pre-existing test assertions that changed were assertions encoding a defect,
  and both were corrected rather than removed.

The residual risk 2D-B must cover is regression in packages **not** touched here,
which selective running cannot rule out. That is precisely what the single full-suite
pass in 2D-B is for.

Nothing was staged, committed or pushed.
