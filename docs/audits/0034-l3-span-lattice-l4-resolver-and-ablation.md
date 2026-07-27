# Audit 0034 — Unified L3 Span Lattice, L4 Boundary & Type Resolver v1, Laboratory Stress Suite, and the Exact Four-Way Ablation

- **Date:** 2026-07-27
- **Author:** Claude (AI agent), for human review
- **Change type:** Unified L3 span lattice over the three experts that actually exist, the L4
  Boundary & Type Resolver v1 with one tracked config, a synthetic laboratory stress suite, the
  completed SYMPTOM error attribution, and the exact four-way ablation executed on the governed
  `internal_test` split with the validated S1 checkpoint. **No commit. No push. No training, no
  retraining, no backward pass, no optimizer or scheduler. No S2–S6 training. No checkpoint
  mutation. No organizer inference. No `output.zip`. No model-weight download. Governed corpus
  unmodified. Architecture PDF unmodified. Audit 0033 unmodified.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — read in full and **unchanged** (SHA-256
  `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`, `git status` clean).
- **Status:** `L3_LATTICE_AND_L4_RESOLVER_V1_IMPLEMENTED_AND_ABLATED — RESOLVER_V1_LEFT_DISABLED_ON_MEASURED_EVIDENCE`.

## 1. Headline, stated before the detail

The milestone is complete: all four deliverables are implemented, executed and measured. **The
principal measured result is unfavourable and is reported as such.**

| Arm | Exact F1 | Δ vs Arm 1 |
| --- | --- | --- |
| 1. ViHealthBERT neural only | **0.7103** | — |
| 2. Deterministic medication/laboratory only | **0.0000** | −0.7103 |
| 3. Unified L3 lattice, before L4 | **0.6963** | −0.0139 |
| 4. Unified lattice, after L4 resolver v1 | **0.7039** | −0.0063 |

Arm 1 reproduces Audit 0033's 0.7103 **exactly** — every count, not just the F1. The unified
lattice *costs* precision on this corpus because E1/E2 contribute only false positives here, and
the L4 resolver recovers a little over half of that cost but does not close it. **Resolver v1 is
therefore left DISABLED** in the development/specialist profile, recorded in
`configs/pipeline/full_v1.yaml`.

The 253 boundary errors that Audit 0033 quantified as the L4 opportunity moved to **252**. That
is the honest answer to the milestone's central question, and it is one error, not a trend.

## 2. Exact initial Git state

```text
pwd     /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
branch  main   (## main...origin/main)
HEAD    a00af3c feat: add exact character-offset S1 mention evaluator
        15d0b07 feat: validate S1 checkpoint and add local held-out evaluation
        36c520d fix: protect S1 astral characters and finalize preflight

git status --porcelain      (clean — working tree had no changes at start)
```

Audits 0030–0033 are committed. Audit 0033 was **not** edited, moved, or renumbered. The next
available number is **0034** and this is the only audit this milestone creates.

## 3. Architecture sections applied

| Section | Applied as |
| --- | --- |
| §4 offset preservation | `original_text[start:end] == text` enforced at every layer; violations raise, never repair |
| §4.2 section priors | `section_priors` in the resolver config — evidence, never an absolute rule |
| §4.3 lexicon coverage | measured gap recorded for bare qualitative lab rows (§7) |
| §5 C1/C2/C3, C7 | route priors; identity is coordinates, never text |
| §6 Mention Factory | one unified lattice; **no expert emits a final entity** |
| §6.1/6.2 grammar, `TEST_NAME –has_result→ TEST_RESULT` | E1 completeness feature; relation preserved through L4 |
| §7.1 boundary ensemble | bounded trim; expand only onto a competing boundary. **No learned boundary-offset head exists or was trained** |
| §7.2 type scoring, abstention | weighted evidence families; abstain on weak/near-tie/conflicting evidence |
| §7.3 global span optimization | near-complete same-type overlap penalty; ICD↛MEDICATION, RxNorm↛DIAGNOSIS |
| §14.1/14.2 synthetic control | laboratory cases with exact offsets **by construction** |
| §18.1/18.2/18.3 | Audit 0033's evaluator reused unchanged; four-way ablation; offset stress |
| §19.1 contracts | `SpanProposal` → `TypedHypothesis`, never `FinalEntity` |

## 4. Audit 0033's evaluator was reused, not replaced

`src/mednorm_vi/evaluation/exact_mention.py` is **unmodified**. Every arm, the synthetic
laboratory suite, and the SYMPTOM attribution are scored by that one evaluator, which is why the
four arms are directly comparable and why Arm 1 reproduces the published baseline bit for bit.

## 5. Checkpoint and corpus integrity

```text
checkpoint  checkpoint/s1_mention_full_training_v1/best.pt      (git-ignored)
SHA-256     a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c
size        1,615,513,303 bytes
mtime       2026-07-27 00:53:41.047814400 +0700   (mtime_ns 1785088421047814400)

verified BEFORE inference   digest + size + mtime_ns   MATCH
verified AFTER  inference   digest + size + mtime_ns   UNCHANGED

governed corpus   corpus_manifest_sha256  a3fd365d…fb8e3     unchanged
                  train / validation / internal_test        33826 / 1045 / 1045
                  internal_test_sha256    e23acde0…e135      unchanged
architecture PDF  0d5eaa20…1e09b                             unchanged
model weights tracked by git                                 0
```

Execution was forward-only: `AutoConfig` + `AutoModel.from_config` (so **no base weights were
downloaded**), strict state-dict load, `model.eval()`, `torch.no_grad()`, every parameter
`requires_grad_(False)`, `trainable_parameters: 0`. No optimizer, no scheduler, no `backward`.
A static AST test asserts the neural package contains no `backward`/`step`/`zero_grad`/`train`
call and no optimizer or scheduler name anywhere.

## 6. What was built

### 6.1 E3 — the trained expert, now tracked code

Audit 0033 executed the neural path by hand; it is now a tracked module, split so that every
decoding rule is testable without a model:

```text
src/mednorm_vi/mention_factory/neural/decoding.py   pure: token predictions -> character spans
src/mednorm_vi/mention_factory/neural/runtime.py    forward-only execution + checkpoint custody
```

Decoding policy: `maximal_contiguous_positive_subtoken_runs_per_type_v1`. The S1 head is a
multi-label token classifier trained with a character-overlap rule, so decoding is its inverse —
a maximal run of consecutive positive subtokens for one type is one mention, spanning the first
subtoken's `original_start` to the last one's `original_end`. Because the slow-tokenizer
alignment backend gives every subtoken its segmented word's original span, decoded spans always
land on real word boundaries in `original_text`.

### 6.2 L3 — the unified span lattice

```text
src/mednorm_vi/lattice/models.py       SourceEvidence, SpanProposal, SpanLattice
src/mednorm_vi/lattice/builder.py      L1 DocumentGraph -> L2 routes -> L3 lattice
src/mednorm_vi/lattice/validation.py   hard invariants; every rule raises
```

Experts that contribute: **E1** medication grammar, **E2** laboratory parser, **E3** ViHealthBERT.
W2NER, MRC-NER, GLiNER and the Qwen proposer have no checkpoint, contribute nothing, and
**nothing was fabricated on their behalf**.

Each node preserves start/end, exact original text, per-type scores, source expert, local score,
route, section, normalized view, evidence/features and complete per-source provenance. Invariants
held and tested:

- `original_text[start:end] == text`, end-exclusive, non-empty;
- **no text-only deduplication** — identity is coordinates, so repeated text at different offsets
  stays distinct (spec §5, case C7);
- exact-coordinate duplicates merge evidence while **every source is retained** individually;
- normalization never mutates `original_text`;
- invalid proposals raise `LatticeError` instead of being trimmed into something plausible;
- deterministic replay — both expert streams are canonically ordered before ids are assigned, so
  the determinism hash does not depend on collection order.

### 6.3 L4 — Boundary & Type Resolver v1

```text
src/mednorm_vi/resolution/config_v1.py     the one tracked config + its digest
src/mednorm_vi/resolution/resolver_v1.py   lattice -> TypedHypothesis
src/mednorm_vi/resolution/boundary.py      + trim_span / expand_to_competitor  (extended)
src/mednorm_vi/resolution/typing.py        + type_utilities / decide_type      (extended)
src/mednorm_vi/resolution/overlap.py       + near-complete overlap competition (extended)
src/mednorm_vi/resolution/features.py      + boundary_kind_of_rule             (extended)
```

**Tracked config and its SHA-256:**

```text
configs/resolution/boundary_type_resolver_v1.yaml
SHA-256  9ad1abada9b70b576cbe1dc566f9464c988ecfcf7556b051620489ca5b1a36d6
version  boundary-type-resolver-v1
```

That digest is written into every report the resolver produces.

Configurable evidence, all bounded to `[0,1]` before weighting: neural span/type score, medication
grammar completeness, laboratory evidence, section priors, route tags, delimiter/boundary shape,
expert agreement and disagreement, overlap structure, and ontology evidence.

Behaviour delivered:

- **choose among competing boundaries** — boundary alternatives sharing a `boundary_group_id`
  (E1's name_only→full ladder, E2's value_only/value+unit pair) collapse to exactly one
  hypothesis, using the tracked `group_preference` (medication `full`, test_result `value_only`,
  mirroring `configs/resolution/resolver_v1.yaml` so the two cannot drift);
- **controlled trim/expand** — trimming is capped at 4 characters per side and can never empty a
  span; expansion may only adopt a boundary another expert already proposed, so no invented offset
  can exist;
- **list numbers, separators and unrelated context are not swallowed** — leading `1.`/`2)`/`-`/`•`
  markers and edge separators are trimmed;
- **all five entity types** are resolvable;
- **abstention on wrong-type risk** — below `min_type_utility` 0.30, below `min_type_margin` 0.05,
  on a type-vs-ontology conflict, or when the only support is a deterministic expert that flagged
  its own parse (`unknown_test_name`, `unknown_medication_name`, `incomplete_parse_no_strength`);
- **repeated mentions at different positions are preserved** — their interval IoU is 0, so they
  never compete;
- **near-complete same-type overlaps are penalised** (IoU ≥ 0.60, penalty 0.20);
- **strong `TEST_NAME –has_result→ TEST_RESULT` pairs are protected** from mutual suppression;
- **ICD evidence can never reinforce MEDICATION and RxNorm evidence can never reinforce
  DIAGNOSIS** — forbidden evidence contributes zero and raises an abstention conflict; the config
  loader rejects a config that pairs them, checked against `CANDIDATE_ONTOLOGY_BY_TYPE`;
- output is `schemas.hypotheses.TypedHypothesis`, never organizer JSON.

**No learned boundary-offset head exists, was built, or was trained.** Spec §7.1 describes one;
this resolver is entirely deterministic and the module records that fact as
`LEARNED_BOUNDARY_OFFSET_HEAD_TRAINED = False`.

## 7. Laboratory path and the synthetic stress suite — **SYNTHETIC, NOT REAL PERFORMANCE**

> The governed corpus contains **no TEST_NAME or TEST_RESULT supervision**. Nothing in this
> section is a claim about real laboratory performance, and every generated payload carries
> `synthetic: true` and `real_performance_claim: false`.

`src/mednorm_vi/mention_factory/laboratory/synthetic.py` builds 14 cases across 13 families. Every
gold span is recorded **as the text is written** — no case calls `str.find`, so an offset cannot
drift. Executed through L1 → L2 → E2 → L3 → L4:

```text
SYNTHETIC laboratory stress suite
cases 14   gold mentions 42   gold has_result relations 21

exact precision  1.0000
exact recall     0.9048
exact F1         0.9500     (TP 38, FP 0, FN 4)

has_result pairing   precision 1.0000  recall 0.9048  F1 0.9500  (19/21)

without the L4 resolver:  precision 0.7170  F1 0.8000  (FP 15)
```

Every family recovered exactly: colon `name:value`, `name: value unit`, decimal comma, decimal
point, H/L flags, reference ranges, missing units, multiline rows, semicolon rows, table-like tab
cells, and a repeated test name at two offsets kept as two mentions. The distractor case (date,
age, phone number, medication dose) produced **zero** laboratory mentions, as required.

### 7.1 Three real defects the suite found, and what was done

1. **A decimal value was shredded by list-marker trimming.** `14.43` matched the ordered-list
   marker pattern `\d{1,2}[.)]` and was trimmed to `43`. Fixed: the numeric marker form now
   requires trailing whitespace (`resolution/boundary.py`). Regression-tested.
2. **`%` was folded into the value, leaving no value-only alternative.** `76.4 %` was emitted as a
   single `result_percent` value, so — unlike `mmol/L` — there was no value_only/value+unit pair
   for the resolver to choose between. Fixed with a role priority in
   `laboratory/patterns.py:find_value` so the bare number wins at the same offset and `%` is
   picked up as the unit it is configured to be.
3. **Boundary alternatives both survived L4.** Their IoU is low, so the overlap rule never fired.
   Fixed by resolving `boundary_group_id` groups before the global overlap competition.

### 7.2 One measured gap, not fixed

A **bare** qualitative row (`HbA1c: dương tính`) with no number, no unit and no laboratory section
header does not route to C2, so E2 never runs and nothing is proposed. The identical rows under a
`Xét nghiệm:` header are recovered exactly. This is recorded as a measured routing gap rather than
silently patched: adding a lexicon-test-name signal to the tracked router config is a routing
decision that needs its own evidence, and tuning the router to a synthetic case would be the wrong
way to make that decision. Both cases are kept in the suite so the difference stays visible.

## 8. SYMPTOM error attribution — real data, completed

Audit 0033 could only report the coarse taxonomy. With L2 routes and L3 provenance available, the
finer categories the milestone asked for are now produced. Attributed against Arm 1 (the
neural-only arm, so the attribution describes the model whose baseline is published):

| Category | Count | Share |
| --- | --- | --- |
| complete_miss | 102 | 31.3% |
| boundary_too_short | 57 | 17.5% |
| boundary_too_long | 42 | 12.9% |
| low_neural_confidence | 40 | 12.3% |
| unknown_other | 28 | 8.6% |
| treatment_purpose_phrase | 16 | 4.9% |
| diagnosis_symptom_confusion | 15 | 4.6% |
| overlap_competition | 14 | 4.3% |
| deterministic_evidence_absent | 12 | 3.7% |
| section_router_error | 0 | 0.0% |
| **total** | **326** | |

By route: C3 154, C3+C4 52, C2/C2+C3/C2+C3+C4 5, unrouted 115.

Two honest caveats. `section_router_error` is **0** because this corpus is single-paragraph
narrative with no section headers — the category is implemented and unit-tested, but the data
cannot exercise it. And E1/E2 **cannot emit SYMPTOM by construction**, so deterministic
corroboration is structurally unavailable for this type; `deterministic_evidence_absent` therefore
counts regions where *no* expert proposed SYMPTOM at all, and the report says so in a
`structural_note` field. 8.6% remain `unknown_other` and are reported as unattributed rather than
pushed into a plausible-looking bucket.

**S1 was not retrained.** Tracked summaries carry counts and 16-hex `privacy_safe_example_id`
handles only; clinical text exists solely in the git-ignored `reports/` tree, asserted by test.

## 9. The exact four-way ablation

Split: governed `internal_test`, 1,045 examples, 2,046 gold mentions. One evaluator, one set of
inputs, four arms.

| | Arm 1 neural only | Arm 2 deterministic only | Arm 3 lattice before L4 | Arm 4 lattice after L4 v1 |
| --- | --- | --- | --- | --- |
| exact precision | **0.7470** | 0.0000 | 0.7169 | 0.7326 |
| exact recall | **0.6769** | 0.0000 | 0.6769 | 0.6774 |
| **exact F1** | **0.7103** | **0.0000** | **0.6963** | **0.7039** |
| exact matches | 1385 | 0 | 1385 | 1386 |
| proposal count | 1854 | 79 | 1932 | 1932 |
| final count | 1854 | 79 | 1932 | 1892 |
| missed | 385 | 2045 | 385 | 385 |
| spurious | 193 | 78 | 262 | 231 |
| wrong type | 23 | 0 | 23 | 23 |
| left boundary | 87 | 0 | 87 | 86 |
| right boundary | 140 | 0 | 140 | 140 |
| both boundary | 26 | 1 | 26 | 26 |
| duplicate/overlap | 0 | 0 | 9 | 0 |
| **boundary errors** | **253** | 1 | **253** | **252** |
| resolver abstained | — | — | — | 37 |
| resolver suppressed | — | — | — | 3 |
| boundary shaped | — | — | — | 2 |

Per-type exact F1:

| Type | Arm 1 | Arm 2 | Arm 3 | Arm 4 |
| --- | --- | --- | --- | --- |
| DIAGNOSIS | 0.7621 | 0.0 | 0.7624 | 0.7632 |
| MEDICATION | 0.7448 | 0.0 | 0.6926 | 0.7386 |
| SYMPTOM | 0.5793 | 0.0 | 0.5793 | 0.5793 |
| TEST_NAME | 0.0 | 0.0 | 0.0 | 0.0 |
| TEST_RESULT | 0.0 | 0.0 | 0.0 | 0.0 |

Deltas from Arm 1:

| | Arm 2 | Arm 3 | Arm 4 |
| --- | --- | --- | --- |
| F1 | −0.7103 | −0.0139 | −0.0063 |
| precision | −0.7470 | −0.0302 | −0.0145 |
| recall | −0.6769 | +0.0000 | +0.0005 |
| exact matches | −1385 | +0 | **+1** |
| missed | +1660 | +0 | +0 |
| spurious | −115 | +69 | +38 |
| wrong type | −23 | +0 | +0 |
| **boundary errors** | −252 | **+0** | **−1** |
| final count | −1775 | +78 | +38 |
| DIAGNOSIS F1 | −0.7621 | +0.0003 | +0.0011 |
| MEDICATION F1 | −0.7448 | −0.0522 | −0.0062 |
| SYMPTOM F1 | −0.5793 | +0.0000 | +0.0000 |

**Arm 1 reproduces the baseline with no deviation to explain**: F1 0.7102564…, TP/FP/FN
1385/469/661, and every error-category count identical to Audit 0033.

### 9.1 Reading the result honestly

- **Arm 2 scores 0.0000.** E1 and E2 produce 79 proposals on this corpus and **not one** matches a
  gold mention. That is not a bug in the ablation: `internal_test` is ViMedNER/ViMQ narrative
  prose, which contains no medication lists and no laboratory rows. The laboratory parser fires on
  narrative clauses that happen to contain a colon and a qualitative term (`có`, `bình thường`),
  producing whole-clause TEST_NAME spans. Those 79 proposals are the entire cost of Arms 3 and 4.
- **Arm 3 is Arm 1 plus that cost.** Recall is identical to four decimal places; only precision
  moves, and only downward (spurious +69). Merging deterministic experts into the lattice on
  narrative text is currently a net negative.
- **Arm 4 recovers a little over half of it.** 37 abstentions and 3 suppressions remove 31
  spurious predictions, and boundary shaping produced exactly **+1** exact match and **−1**
  boundary error. Arm 4 still sits 0.0063 below Arm 1.
- **The 253-boundary-error opportunity is essentially untouched: 253 → 252.** A deterministic
  trim/expand resolver moves it by one. This is the clearest evidence in the milestone that the
  remaining boundary headroom needs the *learned* boundary-offset head of spec §7.1 — which does
  not exist — rather than more deterministic edge shaping.

### 9.2 Resolver enable/disable decision

**DISABLED.** Recorded in `configs/pipeline/full_v1.yaml` under the specialist profile, with the
measured numbers and the condition for revisiting it (`Arm 4 ≥ Arm 1`). Thresholds were **not**
tuned against `internal_test`: that is the held-out split every published number depends on, and
fitting to it would destroy the only trustworthy measurement in the project. Threshold search
belongs on the validation split, in a later milestone.

The one resolver rule added after seeing the ablation — abstain when the only support is a
deterministic expert that flagged its own parse — is an evidence-quality rule derived from spec
§7.2, not a threshold fitted to the split: the expert itself reported that its evidence was weak.
It is disclosed here so a reader can discount it if they disagree.

## 10. Tests and static checks

```text
env PYTHONPATH=src python -m pytest -q         -> 1279 passed, 1 skipped (pyarrow reader)
ruff check .                                    -> All checks passed!
env PYTHONPATH=src python -m mypy               -> Success: no issues found in 238 source files
env PYTHONPATH=src python -m compileall -q src  -> clean
git diff --check                                -> clean
```

129 new tests across six files:

| File | Covers |
| --- | --- |
| `tests/unit/test_span_lattice.py` (19) | exact/end-exclusive offsets, decomposed Unicode, repeated spaces and newlines, repeated text at different offsets, exact-coordinate evidence merging, no text-only deduplication, adjacent and overlapping spans, full provenance, narrowest-route lookup, deterministic replay, loud failure on every invariant violation |
| `tests/unit/test_resolver_v1.py` (32) | config digest and forbidden-ontology rejection, controlled trim (incl. the decimal-shredding regression) and expand, medication neural/grammar agreement and conflict, wrong-type abstention, flagged-evidence abstention, ICD↛MEDICATION and RxNorm↛DIAGNOSIS, near-complete overlap, protected `has_result` pairs, boundary-group collapse, repeated mentions preserved, offset invariant, deterministic replay |
| `tests/unit/test_neural_expert.py` (22) | run decoding, end-exclusive offsets, two runs stay two mentions, multi-label tokens, special tokens, decomposed Unicode, repeated whitespace, mean-probability scores, checkpoint digest/mtime custody, **no backward/optimizer/scheduler anywhere (AST scan)**, no tracked model weights |
| `tests/unit/test_laboratory_synthetic_stress.py` (31) | offsets exact by construction, every family, decimal comma/dot, units, flags, ranges, missing unit, multiline, semicolon, table, repeated tests, distractors, pairing, synthetic labelling, determinism |
| `tests/unit/test_symptom_attribution.py` (15) | every attribution rule, cue window, privacy-safe records and summaries |
| `tests/integration/test_four_way_ablation.py` (10) | all four arms executed, required fields per arm, deltas measured against Arm 1, determinism, privacy-safe attribution |

## 11. Changed tracked files

**Added (16):**

```text
docs/audits/0034-l3-span-lattice-l4-resolver-and-ablation.md
configs/resolution/boundary_type_resolver_v1.yaml
src/mednorm_vi/lattice/__init__.py
src/mednorm_vi/lattice/models.py
src/mednorm_vi/lattice/builder.py
src/mednorm_vi/lattice/validation.py
src/mednorm_vi/mention_factory/neural/__init__.py
src/mednorm_vi/mention_factory/neural/decoding.py
src/mednorm_vi/mention_factory/neural/runtime.py
src/mednorm_vi/mention_factory/laboratory/synthetic.py
src/mednorm_vi/resolution/config_v1.py
src/mednorm_vi/resolution/resolver_v1.py
src/mednorm_vi/evaluation/ablation.py
src/mednorm_vi/evaluation/ablation_cli.py
src/mednorm_vi/evaluation/laboratory_stress.py
src/mednorm_vi/evaluation/symptom_attribution.py
```

Plus six test files:

```text
tests/unit/test_span_lattice.py
tests/unit/test_resolver_v1.py
tests/unit/test_neural_expert.py
tests/unit/test_laboratory_synthetic_stress.py
tests/unit/test_symptom_attribution.py
tests/integration/test_four_way_ablation.py
```

**Modified — milestone changes (6):**

```text
configs/pipeline/full_v1.yaml                          resolver v1 disabled + measured rationale
docs/audits/README.md                                  0034 index entry
src/mednorm_vi/resolution/boundary.py                  + trim_span / expand_to_competitor
src/mednorm_vi/resolution/typing.py                    + type_utilities / decide_type
src/mednorm_vi/resolution/overlap.py                   + near-complete overlap competition
src/mednorm_vi/resolution/features.py                  + boundary_kind_of_rule
src/mednorm_vi/mention_factory/laboratory/patterns.py  value-role priority (percent/unit fix)
```

**Modified — incidental static-check hygiene (27 files, one line each):**

`mypy` could not run at all at HEAD: the local virtualenv now contains Torch (and therefore
NumPy), whose stubs require `python_version >= 3.12` while the tracked config declares 3.10, so
the check aborted before analysing any project file. Two changes make the tracked check
runnable and reproducible:

1. `pyproject.toml` — an override skipping the heavy frameworks the project deliberately does not
   depend on (`torch`, `numpy`, `transformers`, `py_vncorenlp`); they are imported lazily inside
   functions on the inference path only, and their internals are not this repository's contract.
   `types-PyYAML` added to the dev extra so the declared environment is definite.
2. With mypy actually running, 26 `# type: ignore` comments were reported as **stale**. A stale
   suppression hides future real errors, so they were removed — one line per file, no behaviour
   change, all 1,279 tests still pass.

This is disclosed as incidental scope: it was necessary to run the static check the milestone
requires, and it touches pre-existing files.

## 12. Ignored reports and local assets

```text
reports/l3_l4_ablation/four_way_ablation.json                     ignored (reports/**)
reports/l3_l4_ablation/arm1..arm4_*.md                            ignored
reports/l3_l4_ablation/symptom_attribution.md                     ignored
reports/l3_l4_ablation/laboratory_stress_synthetic.json           ignored
reports/l3_l4_ablation/neural_spans_internal_test.json            ignored
checkpoint/s1_mention_full_training_v1/best.pt   1.6 GB           ignored (/checkpoint/)
.venv/                                                            ignored
~/.cache/huggingface/hub, ~/.cache/mednorm-vi/vncorenlp           outside the repository
```

Verified with `git check-ignore -v`. `git ls-files` tracks **zero** `.pt/.pth/.ckpt/.safetensors/.bin`
files, asserted by test.

## 13. Limitations

1. **Arm 4 is below Arm 1.** The unified lattice plus resolver v1 does not beat the neural
   baseline on the governed `internal_test` split, and the resolver is disabled accordingly.
2. **The 253 boundary errors moved to 252.** Deterministic trim/expand is not the lever; the
   learned boundary-offset head of spec §7.1 remains unbuilt and untrained.
3. **E1/E2 have no measurable value on this corpus** (Arm 2 exact F1 0.0000). Their value is
   assumed for medication lists and laboratory rows, which `internal_test` does not contain — the
   synthetic suite is the only evidence they work at all, and it is synthetic.
4. **TEST_NAME and TEST_RESULT remain unsupervised.** Their 0.0 in every arm is an absent-data
   fact. The synthetic suite's 0.9500 says nothing about real performance.
5. **SYMPTOM is unchanged at 0.5793** across all four arms — nothing in this milestone touched it,
   and 8.6% of its failures remain unattributed.
6. **Resolver weights are designed, not learned.** They were not optimised on any split, by
   deliberate choice; the ablation measures the design, not a tuned system.
7. **A bare qualitative laboratory row does not route to C2** (§7.2), unfixed by design.
8. **No ontology evidence flows into the resolver yet.** The ICD/RxNorm consistency rules are
   implemented and unit-tested but contribute 0.0 in every executed arm, because no linker
   evidence is attached at L3 in this milestone.
9. **This is not the organizer score.** Mention span and type only; assertions (30%) and candidate
   Jaccard (40%) are not integrated.
10. S2–S6 remain untrained; L5–L9 unchanged; no organizer inference and no `output.zip`.

## 14. Prohibited actions — none performed

```text
training / fine-tuning        NO       backward pass                 NO
optimizer or scheduler        NO       S2-S6 training                NO
model-weight download         NO       external API call             NO
organizer inference           NO       output.zip                    NO
checkpoint mutation           NO       tracked model weights         NO (0 files)
architecture PDF modified     NO       governed corpus modified      NO
audit 0033 modified           NO       additional audits created     NO (0034 only)
```

## 15. Safe-to-commit verdict

**SAFE TO COMMIT.** Working tree contains source, tests, configs and this audit only. No model
weights, no checkpoints, no corpus files, no reports, no caches. `ruff`, `mypy`, `compileall`,
`git diff --check` and the full 1,279-test suite pass. The checkpoint, the governed corpus and the
architecture PDF are byte-identical to their pre-milestone state.

The headline result is negative and is reported as such: resolver v1 is disabled on measured
evidence, and the boundary-error opportunity Audit 0033 quantified remains open.
