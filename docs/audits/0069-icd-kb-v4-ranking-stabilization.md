# Audit 0069: ICD KB v4.1 Ranking Stabilization and Wrapped-Title Recovery

Milestone: 4J - ranking stabilization, wrapped-title recovery, public ablation gate
Date: 2026-08-01
Verdict: `ICD_KB_V41_ACCEPTED_OFFLINE_ONLY`

No clinical text appears in this audit.

## 1. Preconditions

| Check | Result |
| --- | --- |
| Branch / HEAD | `main` / `5e9e645 feat: repair ICD KB titles and add governed aliases` |
| Audit 0068 committed | yes, in `5e9e645` |
| Staged / `origin/main...HEAD` | none / `0 0` |
| Architecture PDF | `0d5eaa20…81e09b` unchanged |
| Active E3 / rollback checkpoint | `524ece1e…dde3a` / `a64cc173…1017c` unchanged |
| RxNorm index | `415f5972…cff99` unchanged |
| ICD competition-v3 (rollback) | `d72ed17f…5fdd` unchanged |
| ICD competition-v4 (frozen) | `3a629085…4de3a` unchanged |
| Scored 11.9188 ZIP | `eae8a348…4c05` unchanged |

No model trained or downloaded, no notebook, no `internal_test`, no public-test labels, no
external API, no Docker pruning. v3 and v4 are immutable references; v4.1 is append-only.

## 2. Wrapped-Title Recovery (§3)

Audit 0068 read only a row's **first physical line**. `pdftotext -layout` renders one logical
table row across several lines, so a wrapped title was stored truncated. C14.2 is the shape:
the anchor line ends `U ác tính ở vòng bạch` and the remainder, `huyết Waldeyer`, sits on the
next line at the same column origin.

The layout is recoverable because column origins are exact. In that block every segment of
the Vietnamese title column begins at x=260 and nothing else does, so rows are rejoined by
**x-position alone** - no semantics, no model, no guessing. Row reconstruction now runs
*before* column extraction (`kb/icd10/repair/row_reconstruction.py`).

Two rules keep the join honest: a continuation segment joins a column only at that column's
**exact** origin (`X_TOLERANCE = 0`), and a column **closes at the first gap** - a later line
reusing the x belongs to a different logical row, and resuming is precisely how a title
acquires another concept's words.

### Two defects found and fixed during implementation

**The anchor must be measured before merging.** pdftotext sometimes glues the code onto a
title column (`U ác tính ở trực C20`). Once that column is rejoined with its remainder the
trailing code is no longer at the end, so re-deriving the anchor after merging lost 160
codes - including C20, the flagship Audit-0068 recovery. The anchor is now taken from the
unmerged line. Losses went to **zero**.

**`_CODE_TOKEN` never matched an undotted specific code.** The pattern was
`^[A-Z]\d{2}(\.\d+)?[†*]?$`, which matches `C15` and `C15.0` but **not** `C150`. Since the
anchor pair is `<dotted> <undotted>`, no 4-character code could ever be anchored - and
indeed every one of Audit 0068's 673 recoveries was a 3-character category. Adding the
undotted branch opened 667 specific codes.

### Precision guards

A wrong title is worse than a damaged one: it makes a concept confidently reachable by the
wrong name. Every guard below costs recall deliberately.

| Guard | Rejects |
| --- | --- |
| `MAX_JOINED_LINES = 6` | B18 joined 14 lines of subclassification instructions |
| `INSTRUCTION_PHRASES` | mid-row apparatus `is_note` cannot see from the start |
| `is_title_start` | strings starting lower case - the middle of a wrapped phrase |
| `has_repeated_phrase` | interleaved columns (`Bệnh lao phổi, được khẳng - Bệnh lao phổi, …`) |
| `_APPARATUS` | dagger/asterisk cross-references, `+` note sub-items |
| `DANGLING_TAIL_WORDS` | titles ending on `và/hoặc`, `của`, `do` - an incomplete wrap |
| post-trim Vietnamese check | J81, where the diacritics lived only in the trimmed note |
| multi-code rows | a row carrying two anchor pairs; either title could belong to either |

### Results

| | Audit 0068 (v4) | Audit 0069 (v4.1) |
| --- | ---: | ---: |
| Damaged titles in v3 | 2,777 | 2,777 |
| **Recovered** | **673** | **1,342** |
| — single-line / wrapped | 673 / 0 | 371 / **971** |
| — category / specific codes | 673 / **0** | 675 / **667** |
| Embedded-note trims (§9 finding) | 0 | 183 |
| **Remaining damaged** | 2,104 (13.7%) | **1,435 (9.4%)** |
| Titles completed vs v4 | — | 392 |

Deterministic spot checks: `B05 → Bệnh sởi`, `C20 → U ác tính ở trực tràng`,
`C15 → U ác tính ở thực quản`, `M10 → Gút [thống phong]`, `A192 → Bệnh lao kê cấp tính,
không xác định` (wrapped), `D51 → Thiếu máu do thiếu vitamin B12` (wrapped),
`M03 → Bệnh lý khớp sau nhiễm trùng và/hoặc bệnh lý khớp phản ứng …` (wrapped),
`A04 → Nhiễm trùng đường ruột` (embedded note trimmed). A 24-record deterministic sample
(seed 0) read as correct Vietnamese ICD titles in 22 cases; the two exceptions were
`D43 → 'U tân sinh không tiên'` (still truncated) and one source-mangled string
(`F514 → 'Cácnhớ sự'`, where the PDF text layer itself has lost the word boundary).

## 3. Evidence-Tier Scoring Contract (§4)

v4's ranking was a single additive float: exact 100, accent-stripped exact 80, and **+1 per
shared trigram**. With 29,586 aliases in the index a concept can accumulate more than 100
points of trigram noise and outrank a concept the query matched exactly. That is not a
tuning problem - a sum cannot express *"no amount of fuzzy evidence beats one exact match"*.

Ranking is now lexicographic over an explicit ordered tuple
(`kb/indexing/evidence.py`), with no arbitrary score constants deciding order:

```
(evidence_tier, exactness, semantic_alias_support,
 lexical_score, hierarchy_signal, deterministic_code_tiebreak)
```

| Tier | Evidence | Channels |
| --- | --- | --- |
| **A** | accent-sensitive exact canonical title or code | `exact_canonical` |
| **B** | accent-sensitive exact governed alias | `exact_alias` |
| **C** | governed semantic synonym rule | `exact_synonym` |
| **D** | accent-sensitive sparse / char-n-gram | `sparse_accent`, `char_ngram_accent` |
| **E** | accent-stripped exact canonical or alias | `exact_ascii` |
| **F** | accent-stripped fuzzy | `sparse`, `char_ngram` |

The accent boundary between D and E is the point of the ordering: `sởi` and `sỏi` are the
same string once accents are stripped, so accent-stripped evidence must sit below every
accent-preserving channel. It stays in the index - someone typing without diacritics still
finds the concept - it simply can no longer outrank an accent-exact match.

**competition-v3 and RxNorm are untouched.** They carry no tier postings, so they rank at
`UNTIERED_RANK` and their tuple reduces to `(-score, concept_id)` - exactly the previous
ordering. This was verified directly: v3 and RxNorm query results are unchanged.

### A third defect found here

Building the accent-sensitive channel exposed a subtle bug. `char_ngrams` pads with spaces,
and the filter compared each gram against `normalize_text(gram, strip_accents=True)` - which
also **strips whitespace**. So `"  s"` compared unequal to `"s"` and was admitted as
"accent-bearing", along with every other boundary gram. Tier D became a match-anything
channel and `sởi` still returned a gallstone. Comparing against a raw accent-strip that
leaves whitespace alone cut `ngrams_accent` from 11,391 postings to 3,799 and fixed it.

## 4. Alias Quality Control (§6)

| Alias type | Count | Evidence class |
| --- | ---: | --- |
| `canonical_title` | 13,876 | strong |
| `punctuation_variant` | 999 | strong |
| `synonym_rule` | 761 | medium |
| `unaccented_variant` | 15,594 | **recall_only** |
| **Total** | **31,230** | |

A `recall_only` alias generates candidates but feeds **only** the accent-insensitive
channels, so it can never reach tiers A-D. That is the containment for `sởi`/`sỏi`.

| Collision measure | Value |
| --- | ---: |
| Distinct normalized aliases | 19,509 |
| Unique to one code | 17,066 |
| Shared by 2 codes | 1,142 |
| Shared by ≥3 codes | 1,301 |
| Max codes per alias | 96 |
| Accented alias keys / unaccented keys | 9,781 / 9,728 |
| Demoted from exact tiers (≥10 codes) | 472 |

The worst offenders are ICD block phrasings, not names: `rối loạn tâm thần và/hoặc …` (96
codes), `tác động bất lợi của …` (82), `người đi xe bán tải …` (80). An alias shared by ≥10
codes is a generic fragment that could only produce ties, so it is kept out of the exact
tiers. **No provenance is deleted** - every alias stays in `alias_manifest.json` with its
collision count and a `demoted_from_exact_tier` flag.

## 5. competition-v4.1 (§7)

`indices/candidate/icd10_vi/competition-v4.1/` — index SHA-256
`d1803be3a1a1bea2f72f6ca20f98e687c4a32d14f3018fb738bd0184942ea09c`.
A clean rebuild reproduced that digest byte-for-byte.

| | v3 | v4 | v4.1 |
| --- | ---: | ---: | ---: |
| Concepts | 15,308 | 15,308 | 15,308 (identical set) |
| Repaired titles | 0 | 673 | **1,342** (+183 note trims) |
| Remaining damaged | 2,777 | 2,104 | **1,435** |
| Aliases | 0 | 29,586 | 31,230 |

Postings: `exact` 38,379 · `exact_canonical` 36,823 · `exact_alias` 724 · `exact_synonym`
609 · `exact_ascii` 40,954 · `ngrams` 8,125 · `ngrams_accent` 3,799 · `sparse_terms` 17,223
· `sparse_accent` 1,524.

Artifacts: `index.json`, `manifest.json`, `repair_manifest.json`, `alias_manifest.json`,
`alias_collision_report.json`, `retrieval_policy.json`, `diagnostics.json`. ICD identity
stays dotless internally; dotted serialization remains output-only.

## 6. Offline A/B/C Regression (§8)

Same fixed Audit-0067 50-ICD diagnostic packet. AI consensus is an engineering
weak-supervision signal, never clinical gold; the selected subset is **n = 8**.

| | non-empty | mean set | @1 | @2 | @5 | @10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R0 v3** | 39/50 | 9.0 | **5** | 6 | 6 | **8** | 0.714 |
| **R1 v4** | 43/50 | 11.5 | 3 | 5 | 6 | 7 | 0.558 |
| **R2 v4.1** | **43/50** | 11.5 | 3 | 5 | 6 | **8** | 0.563 |

| Change vs v3 | v4 | v4.1 |
| --- | ---: | ---: |
| empty → non-empty | 5 | 5 |
| non-empty → empty | 1 | 1 |
| identical candidate list | 6 | 6 |
| top-1 changed | 19 | 24 |

### 16-concept natural-mention probe

| | retrievable | at rank 1 |
| --- | ---: | ---: |
| v3 | 1/16 (A169 only) | 0/16 |
| v4 | 9/16 | 2/16 |
| **v4.1** | **9/16** | **3/16** |

### Accent false friend - the headline fix

Mention `sởi` (measles, B05):

| | top 3 |
| --- | --- |
| v3 | `N200` (kidney calculus), `D48`, `K81` |
| v4 | `K851` (gallstone pancreatitis), `B05`, `N200` |
| **v4.1** | **`B05`**, `B06` |

Measles now ranks first, and the only competitor is rubella - a clinically adjacent concept,
not a stone. v4.1 recovered @10 to v3's level (8/8) and improved MRR over v4.

## 7. Error Analysis (§9)

Every record still behind v3, classified:

| Record | gold | v3 | v4.1 | Class | Cause |
| --- | --- | --- | --- | --- | --- |
| `lg1-09477de4` | B069 | HIT | r2 | **hierarchy ambiguity** | B06's title was repaired to `Bệnh rubella [sởi Đức]`, so the parent now legitimately competes with its child `Bệnh rubella` |
| `lg1-2d50af74` | B069 | HIT | r2 | **hierarchy ambiguity** | same pair |
| `lg1-3b655cce` | Q01 | r2 | r4 | **hierarchy ambiguity** | Q01 repaired from `- Bao gồm: thoát` to `Dị tật thoát vị não`; children Q018/Q019 now match too |

Improvements relative to v3, classified:

| Class | Count | Why |
| --- | ---: | --- |
| title recovery | 5 | concepts had no searchable name at all in v3 (empty → non-empty) |
| title recovery (probe) | 8 | C20, B05, C15, C25, G10, C18, Q210, F03 reachable by their own names |
| accent normalization | 1 | `sởi` no longer returns a calculus |
| scoring / evidence tier | 2 | `lg1-0eea7eeb` r10 → r9, `lg1-3376ea28` r9 → r7 |

**The dominant residual error class is parent/child hierarchy ambiguity, and it is caused by
repair succeeding.** In v3 a damaged parent could not compete because it had no real name;
now that B06 and Q01 carry correct titles, both parent and child match the mention and the
linker's specificity gate - not the KB - decides. That is a linker policy question, out of
scope for a KB milestone, and it is the single highest-value next fix.

`lg1-0eea7eeb` (I25) also exposed a **gap in the damage predicate**: I25 is stored as
`Bệnh tim thiếu máu cục bộ Loại trừ: bệnh`, an *embedded* note. `is_damaged_title` anchors at
the start, so it never saw it and I25 was never a repair target. Trimming glued notes across
the whole KB fixed 183 such titles; the previous string is preserved in the repair manifest.

## 8. Acceptance Gate (§8) — not passed

| Requirement | Result |
| --- | --- |
| Retain or exceed v4 coverage gains | **PASS** — non-empty 43/50, probe 9/16, both equal to v4 |
| **Restore @1 to at least v3** | **FAIL** — 3 vs 5 |
| Restore @10 to at least v3 | **PASS** — 8/8, recovered from v4's 7 |
| No new clean exact-case regression | **PASS** — no case that v4 got right was lost |
| No catastrophic accent false friend | **PASS** — `sởi` → B05 at rank 1 |
| Deterministic repeat | **PASS** — identical SHA-256 on clean rebuild |
| No change to E1/E2/E3/assertion/RxNorm | **PASS** by construction; v3 and RxNorm untiered and verified unmoved |

Preferred targets (`@1 > v3`, `probe ≥ 9/16`, `non-empty ≥ 43/50`): two of three met. The @1
requirement is not, so the gate fails. n = 8 is small and the three losses are all one
explainable cause, but the gate is written to refuse this trade and it refuses. No number
here has been adjusted to make a target.

## 9. Public Ablation (§10)

**Not run.** §10 permits public inference only when v4.1 clearly passes all offline gates.
It does not. `runs/icd_kb_v41_0069_final_dotted/` was not created, no `output.zip` was
produced, and the scored 11.9188 submission remains the current best. The remaining
leaderboard attempt is preserved.

## 10. Validation (§12)

| Gate | Result |
| --- | --- |
| `tests/unit/test_icd_evidence_tiers_0069.py` | 26 passed |
| `ruff check .` | clean |
| `ruff format --check` (changed files) | clean |
| `mypy src/mednorm_vi` | no issues, 297 source files |
| `compileall src tests scripts` | clean |
| `git diff --check` | clean |

The regression tests are synthetic throughout - no clinical record, no private annotation,
no public-test text. They cover the accent false friend (`sởi` vs `sỏi`), accented exact vs
unaccented alias, accented alias vs accent-stripped canonical, semantic synonym vs fuzzy
spelling, exact canonical against 40 fuzzy competitors, tier ordering under a 10,000× score
disadvantage, untiered-index equivalence, wrapped-title rejoining, parent-never-inherits-
child, gap-stops-a-join, instruction-text rejection, embedded-note trimming and interleaved-
column rejection.

## 11. Changed-File Inventory

New (tracked):
`src/mednorm_vi/kb/icd10/repair/row_reconstruction.py`,
`src/mednorm_vi/kb/indexing/evidence.py`,
`scripts/build_icd_kb_v41_0069.py`,
`tests/unit/test_icd_evidence_tiers_0069.py`,
`docs/audits/0069-icd-kb-v4-ranking-stabilization.md`.

Modified (tracked):
`src/mednorm_vi/kb/icd10/repair/title_recovery.py` (wrapped recovery + precision guards),
`src/mednorm_vi/kb/icd10/repair/alias_layer.py` (evidence classes),
`src/mednorm_vi/kb/indexing/normalization.py` (accent-marked forms),
`src/mednorm_vi/kb/indexing/retrieval.py` (tier postings, rank tuple),
`src/mednorm_vi/linking/icd10.py` (evidence ordering, gated on `index.tiered`).

New (untracked, ignored): `indices/candidate/icd10_vi/competition-v4.1/` (7 files).

Unchanged: competition-v3, competition-v4, RxNorm, both checkpoints, the scored ZIP, and
every pipeline config - nothing points at v4.1.

## 12. Remaining Risks

1. **Parent/child ambiguity is now the binding constraint**, and repair created it by giving
   parents real names. Fixing it means the linker's specificity gate, not the KB.
2. **9.4% of titles are still damaged** (1,435). Most are wraps the strict x-alignment
   refuses; loosening it would trade precision for recall, which this milestone declined.
3. **Weak labels, n = 8.** The @1 comparison rests on eight AI-consensus records. It is a
   signal, not a measurement - which cuts both ways, and the gate treats it conservatively.
4. **`I10` (essential hypertension) is absent from the index entirely.** No repair reaches it.
5. **Tier D is narrow by design** (1,524 accent tokens). It fires only for genuinely accented
   queries; unaccented mentions fall to tier E and rank below any accent-exact competitor.

## 13. Next-Milestone Handoff (§11)

`PRETRAINED_SEMANTIC_RETRIEVER_ON_REPAIRED_KB`, on a v4.1 whose parent/child ordering has
been settled - training a retriever against text that is 9.4% still-damaged, with an
unresolved parent/child preference, would bake both into the weights. Sequence: settle the
specificity gate, re-run this A/B/C regression, then select the retriever.

Model selection must optimize expected leaderboard quality. The hard constraint is the
competition's **< 9B total deployed parameters**; the owner's RTX 4060 is **not** a
constraint - heavy training may use Colab A100/H100 or other permitted GPU infrastructure.

## 14. Verdict

**`ICD_KB_V41_ACCEPTED_OFFLINE_ONLY`**

v4.1 doubles title recovery (673 → 1,342), cuts remaining damage from 13.7% to 9.4%,
replaces an additive score with an explicit lexicographic evidence contract, and eliminates
the catastrophic accent false friend that motivated the milestone - `sởi` returns measles at
rank 1 for the first time. It holds v4's coverage gains and recovers @10 to v3's level.

It does not restore @1, and the three losses are all the same newly-created parent/child
ambiguity. The gate refuses that trade, so v4.1 is retained for further offline work and is
**not** promoted to the leaderboard. **Recommendation: do not submit.**
