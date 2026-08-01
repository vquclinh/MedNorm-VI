# Audit 0070: Hierarchy-Aware ICD Specificity Arbitration and v4.1 Acceptance

Milestone: 4K - hierarchy arbitration, v4.1 acceptance, public leaderboard ablation
Date: 2026-08-01
Verdict: `ICD_HIERARCHY_V41_READY_FOR_LEADERBOARD`

No clinical text appears in this audit.

## 1. Preconditions

| Check | Result |
| --- | --- |
| Branch / HEAD | `main` / `6cb8604 feat: stabilize ICD v4.1 ranking and recover wrapped titles` |
| Audit 0069 committed | yes, in `6cb8604` |
| Staged / `origin/main...HEAD` | none / `0 0` |
| Architecture PDF | `0d5eaa20…81e09b` unchanged |
| Active E3 / rollback checkpoint | `524ece1e…dde3a` / `a64cc173…1017c` unchanged |
| RxNorm index | `415f5972…cff99` unchanged |
| ICD v3 / v4 | `d72ed17f…5fdd` / `3a629085…4de3a` unchanged |
| **ICD v4.1** | **`d1803be3…ea09c` - matches the Audit-0069 frozen expectation** |
| Scored 11.9188 ZIP | `eae8a348…4c05` unchanged |

No model trained or downloaded, no notebook, no `internal_test`, no public-test labels, no
external API, no Docker pruning. **The KB was not rebuilt**: v4.1 is a frozen input and its
digest was verified before and after the milestone.

## 2. The Hierarchy Problem, Formalized (§3)

The existing gate decides a family in three steps. `assess_specificity` computes
`added = child_tokens − parent_tokens`, drops a descendant whose added tokens are not all
present in the mention (`DROP_UNSUPPORTED_SPECIFICITY`), then assigns a linker tier:

| Condition | Tier | Order |
| --- | --- | ---: |
| exact name channel | `exact_name` | 1 |
| `assessment.supported_tokens` non-empty | `supported_specific` | 2 |
| reached by upward expansion | `broader` | 3 |
| sibling or plain lexical | `lexical` | 4 |

**The defect is structural, not code-specific.** A child is measured only by what it *adds*.
Once Audit 0069 repaired B06's title to `Bệnh rubella [sởi Đức]`, its child B069
(`Bệnh rubella`) added *nothing*, earned no `supported_specific` credit and fell to
`lexical` - while the parent arrived by upward expansion and took `broader`, which sorts
above it. So **any repaired parent whose title is a token-superset of its child's
automatically demotes that child**, and repair made that configuration far more common.

Q01 fails the mirror-image way: before repair its title was the fragment `- Bao gồm: thoát`,
so it could not compete at all; after repair (`Dị tật thoát vị não`) its children Q018/Q019
match the same mention and outrank it.

Neither B06/B069 nor Q01 is special-cased anywhere in the implementation.

## 3. Specificity Features (§4)

`src/mednorm_vi/linking/icd10_specificity.py`. Every feature is derived from the governed KB,
the mention and optional bounded context. No LLM, no medical fact the KB does not state.

The two that carry the decision:

| Feature | Definition |
| --- | --- |
| `unsupported_extra` | content tokens in the candidate's title **absent** from the mention/context - detail claimed but not stated |
| `supported_added` | content tokens the candidate adds over its family root that the text **does** state - detail claimed and stated |

Also computed and recorded: `evidence_tier`, `depth`, `is_family_root`,
`parent_child_distance`, `mention_exact_to_title`, `mention_exact_to_alias`,
`semantic_synonym_match`, `child_is_unspecified` (`.9`), `child_is_other` (`.8`),
`sibling_competition_count`, `lexical_score`, and the token lists themselves.

Context participates **only in support**: it can justify a qualifier the mention omits but can
never introduce one. Widening what counts as *stated* is safe; widening what counts as
*asserted* is not.

### A defect found while implementing

The first version zeroed `root_tokens` when the candidate *was* the family root, which
credited the category with its entire title as "added detail" and let it out-argue every child
it contains. Two CASE-A tests caught it before any measurement was taken. The root's added
detail is now correctly zero.

## 4. Arbitration Contract (§5, §6)

Ranking is lexicographic over an ordered tuple. There are **no additive bonuses**, so no
quantity of weaker evidence can accumulate past a stronger criterion:

```
(evidence_tier_index, unsupported_extra, -supported_added, not is_family_root, depth, code)
```

| Position | Resolves | Behaviour |
| --- | --- | --- |
| `evidence_tier_index` | — | Audit-0069 tiers A-F, unchanged and **dominant**. A weak fuzzy child can never defeat an exact parent. |
| `unsupported_extra` | **B, E** | A candidate asserting unstated detail loses. `.8` cannot win on lexical similarity; a repaired parent carrying extra words loses to its exact child. |
| `-supported_added` | **A** | When the text states the qualifier, the child carrying it wins. |
| `not is_family_root` | **C, D** | Ties resolve **upward**: a category represents its family, and no child is picked arbitrarily when several remain equally plausible. |
| `depth`, `code` | — | Total order, so a repeat run cannot reorder. |

**CASE C rule, stated explicitly:** when a parent and its unspecified child are
indistinguishable on evidence and neither asserts unstated detail, **the category
represents the family**. `.9` children add only stopwords (`không xác định`), so they tie and
lose upward. This is a documented convention, not a discovered fact.

**Containment.** Arbitration operates *within* one ICD family only. A family keeps the best
position any member already held; only *which* member sits there can change, and a family
never moves past another family. Choosing between unrelated concepts remains a lexical
question that the Audit-0069 tiers already answer - which is structurally why this cannot
reintroduce the `sởi`/`sỏi` false friend.

## 5. Policy Sweep (§7)

Three policies, no hyperparameter search.

| Policy | non-empty | set | @1 | @2 | @5 | @10 | MRR | p→c | c→p | sib |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **H0** none (control) | 43/50 | 11.5 | 3 | 5 | 6 | 8 | 0.563 | 0 | 0 | 0 |
| **H1** conservative | 43/50 | 11.5 | 5 | 5 | 6 | 8 | 0.688 | 2 | 0 | 0 |
| **H2** hierarchy-aware | 43/50 | 11.5 | **5** | **6** | 6 | 8 | **0.719** | 2 | 2 | 1 |

Clean-exact regressions: 0 under both H1 and H2. False friends: 0 under all three
(`sởi` → B05 rank 1 in every case).

**Selected: H2.** H1 recovers @1 by undoing over-assertion only, but leaves @2 and MRR below
v3 because it cannot promote a parent over a child that asserts unstated detail - the Q01
shape. H2 handles both directions with one rule and is the only policy that reaches v3's @2
and exceeds v3's MRR.

## 6. Offline Evaluation (§8)

Same fixed Audit-0067/0069 50-ICD diagnostic packet. AI consensus remains weak-supervision
engineering evidence, never clinical gold. Selected subset **n = 8**.

| | non-empty | mean set | @1 | @2 | @5 | @10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R0** v3 + existing linker | 39/50 | 9.0 | 5 | 6 | 6 | 8 | 0.714 |
| **R1** v4.1 + existing linker | 43/50 | 11.5 | 3 | 5 | 6 | 8 | 0.563 |
| **R2** v4.1 + hierarchy linker | **43/50** | 11.5 | **5** | **6** | 6 | **8** | **0.719** |

| vs v3 | R1 | R2 |
| --- | ---: | ---: |
| empty → non-empty | 5 | 5 |
| non-empty → empty | 1 | 1 |
| top-1 changed | 24 | 24 |
| parent → child | 0 | 0 |
| child → parent | 2 | 1 |
| sibling | 0 | 1 |

### 16-concept natural-mention probe

| | retrievable | at rank 1 |
| --- | ---: | ---: |
| R0 v3 | 1/16 | 0/16 |
| R1 v4.1 | 9/16 | 3/16 |
| **R2 v4.1 + hierarchy** | **9/16** | **3/16** |

### Accent false friend

Mention `sởi` (measles, B05): v3 → `N200` (kidney calculus); **R1 and R2 → `B05`, `B06`.**
Arbitration leaves it untouched, as containment requires.

## 7. The Three Audit-0069 Losses (§11)

| Record | gold | v3 | R1 (v4.1) | **R2 (+hierarchy)** | Outcome |
| --- | --- | --- | --- | --- | --- |
| `lg1-09477de4` | B069 | HIT | r2 | **HIT** | **fixed** |
| `lg1-2d50af74` | B069 | HIT | r2 | **HIT** | **fixed** |
| `lg1-3b655cce` | Q01 | r2 | r4 | **r2** | **fixed** (restored to v3 rank) |

Mechanisms: both B069 records were the repaired-parent-demotes-child shape, resolved by
`unsupported_extra` (B06 carries `sởi`/`đức`, which the mention does not). Q01 was the
child-asserts-unstated-qualifier shape, resolved by the same field acting in the opposite
direction plus the upward tie-break.

Other selected records: `lg1-0eea7eeb` (I25) r10 → **r9**, better than v3;
`lg1-3376ea28` (K37) r9 → **r7**, better than v3. Both improvements come from the KB repair,
not the policy. F329, H524, H480 remain HIT throughout.

**Nothing in the selected set remains worse than v3.** Full classification of the
non-selected changes is not possible without gold labels and is not claimed.

Improvement mechanisms: title recovery (5 empty→non-empty), evidence tiers (I25, K37),
hierarchy arbitration (B069 ×2, Q01).

## 8. Hard Acceptance Gate (§9) — all pass

| Requirement | Threshold | Result |
| --- | --- | --- |
| v4.1 coverage retained | non-empty ≥ 43/50 | **43/50 PASS** |
| 16-concept probe | ≥ 9/16 | **9/16 PASS** |
| @10 | ≥ 8/8 | **8/8 PASS** |
| @1 | ≥ 5/8 (v3) | **5/8 PASS** (preferred > 5/8 **not** met) |
| No new catastrophic accent false friend | — | **PASS** |
| `sởi` → B05 rank 1 | — | **PASS** |
| No new clean exact-match regression | — | **PASS** (0) |
| No RxNorm regression | — | **PASS** (0 selected-code changes) |
| E1 / E2 outputs unchanged | — | **PASS** |
| E3 spans / types unchanged | — | **PASS** (0 / 0) |
| Assertion outputs unchanged | — | **PASS** (0) |
| Deterministic repeat | — | **PASS** (byte-identical rerun) |

**Small-n limitation, stated and not used to relax anything:** the selected subset is 8
records of AI consensus. @1 = 5/8 *equals* the v3 baseline rather than exceeding it, so the
preferred target was not reached. MRR (0.719 vs 0.714) and @2 are the finer-grained signals
that separate R2 from v3, and both favour R2. The gate as written passes; the evidence behind
it is thin, and that is a real risk carried into the public run.

## 9. Public Leaderboard Ablation (§12) — run once

Profile `configs/pipeline/icd_hierarchy_v41_0070.yaml`, a copy of `full_v1.yaml` differing in
**exactly two fields**, verified field-by-field against the baseline config object:
`icd_index` → competition-v4.1, and `icd_hierarchy_policy` → H2.

`runs/icd_hierarchy_v41_0070_final_dotted/`

| Comparison vs the 11.9188 scored baseline | Value | Required |
| --- | ---: | ---: |
| Documents | 100 | — |
| Entities compared | 1,147 | — |
| Entity-count mismatches | 0 | 0 |
| Documents changed | 68 | — |
| **Spans changed** | **0** | **0** |
| **Entity types changed** | **0** | **0** |
| Mention texts changed | 0 | — |
| **Assertions changed** | **0** | **0** |
| **RxNorm selected codes changed** | **0** | **0** |
| ICD candidate sets changed | 150 | — |
| ICD top-1 changed | 150 | — |
| Output schema | identical | — |
| Runtime | 128.3 s | — |
| Warnings / L9 stop | none / no | — |

ICD top-1 change direction — **the hierarchy policy is not what moves most of the public
output**:

| Direction | Count |
| --- | ---: |
| cross-family (KB repair) | **126** |
| child → parent (policy) | 7 |
| parent → child (policy) | 5 |
| sibling (policy) | 4 |

307 ICD codes emitted, all well-formed; 243 dotted, 64 three-character categories.

**Dotted serialization.** The raw pipeline emits dotless internal identities; dotted form is
applied by the governed post-processing step `mednorm_vi.inference.derive_submission
--derivation icd_dotted`, exactly as the scored baseline was produced. Its validation report
is clean (`ok: true`, 1,147 entities checked, 0 issues, medication lists identical).

| Artifact | SHA-256 |
| --- | --- |
| **`runs/icd_hierarchy_v41_0070_final_dotted/output.zip`** | **`29d77f1c8bb202dbae6e8a3f438c64528f72cf21267b39b9c41d676edbf132fb`** |
| raw pre-derivation ZIP | `5ad64477027895d2746c8ca9c27ad8cf8e2b0daa6a08e6d0ac535112f3427b95` |

Artifacts written: `output/`, `output.zip`, `run-manifest.json`, `kb-manifest.json`,
`hierarchy-policy.json`, `validation-report.json`, `diagnostic-summary.json`, `run.log`,
plus `derivation-manifest.json` and `output_raw_dotless/` as derivation provenance.

**Not submitted.** The owner submits manually.

## 10. Parameter Policy (§13)

**Zero learned parameters added.** No model trained, downloaded or fine-tuned. Laptop VRAM
was not a design constraint; the hard constraint for future deployment remains **< 9B total
deployed parameters**.

## 11. Validation (§16)

| Gate | Result |
| --- | --- |
| `test_icd_hierarchy_specificity_0070.py` + `test_icd_evidence_tiers_0069.py` | 49 passed |
| `ruff check .` | clean |
| `ruff format --check` (files I changed) | clean |
| `mypy src/mednorm_vi` | no issues, 298 source files |
| `compileall src tests scripts` | clean |
| `git diff --check` | clean |

`src/mednorm_vi/linking/icd10.py` is **not** ruff-format-clean, and was already not clean at
HEAD before this milestone. It was left alone rather than reformatted, because normalizing it
would produce churn unrelated to this change.

The 23 new tests are synthetic throughout and cover every §10 pattern: parent + exact child,
parent + unspecified child, parent + `.8`, parent + `.9`, two siblings, parent exact + child
fuzzy, child exact + parent exact, child requiring an unsupported qualifier, context
supporting the qualifier, generic mention supporting parent only, hierarchy distance > 1,
accent false friend still blocked, untiered v3 unchanged, RxNorm unchanged - plus
candidate-set preservation, determinism, H0-is-a-no-op, and cross-family containment.

## 12. Changed-File Inventory

New (tracked): `src/mednorm_vi/linking/icd10_specificity.py`,
`tests/unit/test_icd_hierarchy_specificity_0070.py`,
`configs/pipeline/icd_hierarchy_v41_0070.yaml`,
`docs/audits/0070-icd-hierarchy-specificity-arbitration.md`.

Modified (tracked): `src/mednorm_vi/linking/icd10.py` (policy hook, inert by default),
`src/mednorm_vi/inference/config.py` (`icd_hierarchy_policy`, refused if unknown),
`src/mednorm_vi/inference/pipeline.py` (threads the policy to the ICD linker).

New (untracked, ignored): `runs/icd_hierarchy_v41_0070_final_dotted/`.

Unchanged: all three ICD KBs, RxNorm, both checkpoints, the scored baseline ZIP, and
`configs/pipeline/full_v1.yaml`, which remains the rollback profile.

## 13. Remaining Risks

1. **n = 8.** @1 merely *equals* v3; the separation is MRR and @2. The public run changes 150
   ICD top-1 codes on evidence this thin. This is the dominant risk.
2. **126 of 150 public top-1 changes are cross-family**, driven by the Audit-0069 KB repair
   rather than by anything measured here. The offline probe (1/16 → 9/16) is the main reason
   to believe they are improvements, and it is an indirect argument.
3. **Bounded context is plumbed but unused in production.** `context_text` defaults to empty
   from the pipeline, so CASE A currently fires only on mention text. Wiring document context
   is a real, tested improvement left undone.
4. **CASE C is a convention, not a measurement.** Resolving parent/unspecified-child ties
   upward matched the observed Q01 evidence; a different governed convention would change it.
5. **9.4% of ICD titles remain damaged**, unchanged from Audit 0069.

## 14. Next Milestone (§14)

`PRETRAINED SEMANTIC RETRIEVAL ON REPAIRED ICD KB`. Candidate architecture: lexical tiered
retrieval + semantic dense retrieval → union/fusion Top-K → hierarchy-aware arbitration →
optional learned reranking only if ranking remains a *measured* bottleneck. Model selection
must optimize expected leaderboard quality, not laptop compatibility; heavy training may use
Colab A100/H100. **No pretrained model was chosen in this audit.**

## 15. Verdict

**`ICD_HIERARCHY_V41_READY_FOR_LEADERBOARD`**

The parent/child ambiguity Audit 0069 left open was a structural defect in how specificity was
measured, not a property of two codes. Replacing "what does the child add?" with "what does
this candidate assert that the text does not say?" resolves all three losses with one
lexicographic rule, no learned parameters and no magic constants. R2 retains every v4.1
coverage gain, matches v3 on @1/@2/@5/@10, exceeds it on MRR, and holds `sởi` → B05.

Every hard gate passes and the public ablation changed only ICD candidates - spans, types,
assertions and RxNorm are byte-identical to the scored baseline.

**Recommendation: SUBMIT**, with the small-n caveat above stated plainly. The decision is the
owner's; the ZIP is not submitted automatically.
