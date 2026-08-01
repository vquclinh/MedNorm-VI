# Audit 0067: AI-to-AI Linker Adjudication Handoff

Milestone: 4H - AI-to-AI adjudication handoff replacing interactive owner review
Date: 2026-08-01
Verdict: `AI_CONSENSUS_READY` (superseding `CHATGPT_HANDOFF_READY` — see §15)

This audit contains **no clinical text and no individual record content**. Every packet
described here lives under `data/private_linker_gold/`, which is gitignored.

## 1. Why this milestone exists

Audit 0066 left the owner with an 80-record queue to adjudicate one record at a time. That
is the correct scientific process and an unacceptable engineering cost. This milestone
replaces the interactive loop with an offline exchange: a blind packet goes to a second AI
system, its decisions come back through a validating importer, and the two independent
opinions are compared to produce a consensus set usable as **weak supervision**.

What it does not do is manufacture ground truth. `AI_CONSENSUS_HIGH` is the strongest label
this project can produce without a person, and it is still not `HUMAN_GOLD`: two models
that share a blind spot agree exactly as readily as two models that are right, and nothing
in this pipeline can tell those apart.

## 2. Git and Safety State

| Check | Result |
| --- | --- |
| Branch / HEAD | `main` / `7f74cb2 feat: establish governed ICD and RxNorm adjudication` |
| Staged / `origin/main...HEAD` | none / `0 0` |
| Architecture PDF | `0d5eaa20…81e09b`, unchanged |

No model trained or downloaded, no public inference, no `internal_test` access, no external
API. No clinical text left the machine — the packets are files on disk for the owner to
transfer manually.

### One legitimate change to `pack.jsonl`, investigated and explained

The pack digest moved from `1898d726…8226d` (recorded in Audit 0066) to
`245ef156…14cb`. This was checked rather than assumed, because that file is the one thing
this pipeline must never write.

**Cause: the owner adjudicated one record themselves** through the Audit-0066 blind-review
CLI, at 04:59 UTC, before requesting this milestone. The record is one RxNorm mention; the
owner recorded `NO_VALID_CANDIDATE` at high confidence, the pre-reveal decision was
preserved as designed, and the AI draft was shown only afterwards.

Three things follow, all verified:

* No tool in this milestone wrote to the pack — asserted by test, and the round-trip
  rehearsal wrote only to a scratch directory.
* The blind packet's recorded `pack_sha256` matches the current file, so the manifest is
  truthful rather than stale.
* **The first human-vs-AI datapoint agrees exactly.** The owner independently reached
  `NO_VALID_CANDIDATE`; the Claude draft for the same record is `NO_VALID_CANDIDATE`,
  `HIGH`, flagged `drug_class`. One record is not evidence, but it is the first.

The pack now holds **1 `HUMAN_GOLD`** and 299 `UNLABELLED` records. That single human
decision remains authoritative wherever it exists: the consensus builder records
`human_gold_present` and `human_gold_is_authoritative`, and writes AI consensus beside a
human decision, never over it.

## 3. Blind Packet (§1–§2)

| Artifact | SHA-256 |
| --- | --- |
| `chatgpt_handoff/minimum_benchmark_blind.jsonl` | `957151bc2244cdb0ca5b3f2222c5f7ea71e2830a6030cd3489bd24cd4af9cdb5` |
| `chatgpt_handoff/minimum_benchmark_blind.md` | `3520672df20f16a43cda685ac4247e45f9b337a2665fbea2ff61e0b735d194d8` |
| `chatgpt_handoff/handoff_manifest.json` | `93ad2d1615088161ebd837416a891a464c942347d594256a8ede4f928ea4228c` |

**80 records exported: 50 ICD-10 and 30 RxNorm** — every id in
`queue_minimum_benchmark.txt`, verified set-equal by test. Markdown is 104 KB.

Each record carries `annotation_id`, ontology, entity type, source dataset, stratum,
mention text, bounded context before and after, and every offered candidate with its full
local evidence: for ICD the display index, dotless internal code, dotted display code,
canonical name, aliases, parent, children, hierarchy depth, specificity class and tier,
retrieval channels, lexical score and `linker_rank`; for RxNorm the display index, RxCUI,
canonical name, aliases, TTY, ingredient, precise ingredient, strength, dose form, brand,
graph relations, retrieval channels, lexical score and `linker_rank`. The KB snapshot id
and both index digests travel in the manifest.

**The packet is blind, enforced structurally.** `FORBIDDEN_KEYS` names every field that
would leak the first opinion — decision, confidence, rationale, selected codes, status,
`linker_top1`, agreement flags, any "recommended" field — and the exporter raises if one
would be written. Tests assert their absence from the JSONL records, from the candidate
objects, and from the Markdown prose. `linker_rank` is retained deliberately: it is factual
metadata about what the production retriever returned, not a recommendation.

The reason for blindness is not procedural tidiness. A second opinion that has already seen
the first is a ratification, and the agreement rate it produces would measure nothing.

## 4. Response Schema (§3)

`schemas/annotation/ai_second_opinion_v1.schema.json` — tracked, because the contract is
public; the returned decisions are private, because they reference clinical records.

Required: `annotation_id`, `ontology`, `decision` (`SELECTED` / `NO_VALID_CANDIDATE` /
`ABSTAIN`), `selected_codes`, `confidence` (`HIGH` / `MEDIUM` / `LOW`), `rationale`,
`evidence_used`, `competing_codes`, `ambiguity_flags`. Conditional rules make
`NO_VALID_CANDIDATE` and `ABSTAIN` carry no selected codes — unresolved alternatives belong
in `competing_codes` — and `SELECTED` carry at least one. ICD codes stay dotless.

## 5. Importer (§4)

`scripts/import_ai_second_opinion.py` refuses a return that: names an unknown
`annotation_id`; names one twice; changes a record's ontology; selects a code that record
never offered; uses a dotted ICD code; carries codes on a non-`SELECTED` decision; uses an
unknown decision or confidence; or omits any expected record unless `--allow-partial` is
given explicitly. All problems are collected and reported together rather than failing on
the first.

Accepted results are written to `data/private_linker_gold/0065/ai_second_opinion/` with
`adjudication_status = AI_SECOND_OPINION`. **`pack.jsonl` is never opened for writing.**

## 6. Consensus and Reconciliation (§5–§6)

`scripts/build_ai_consensus_0067.py` classifies every record into one of the eight required
categories — `EXACT_AGREEMENT`, `DECISION_AGREEMENT_CODE_DISAGREEMENT`,
`CLAUDE_SELECTED_CHATGPT_REJECTED`, `CHATGPT_SELECTED_CLAUDE_REJECTED`, `CLAUDE_ABSTAIN`,
`CHATGPT_ABSTAIN`, `BOTH_ABSTAIN`, `OTHER_DISAGREEMENT` — and writes `consensus.jsonl`,
`disagreements.jsonl`, `high_confidence_consensus.jsonl` and `summary.json`.

A record reaches `AI_CONSENSUS_HIGH` only when both systems independently agree on the
decision; the selected code sets are exactly equal when `SELECTED`; neither side is `LOW`
confidence; neither flags unresolved ontology ambiguity; every code is present in the
frozen KB offer set; and ICD identity is dotless. Each of those is covered by a test.

The disagreement packet (`disagreements_for_chatgpt.jsonl` / `.md`) is round 2 and **may**
show both prior opinions — blindness applied to round 1 only. It adds local KB enrichment:
ICD sibling concepts under the shared parent and hierarchy context; RxNorm graph neighbours
alongside the ingredient, TTY, strength, dose-form and brand fields already present. No
external API is used.

### Round-trip rehearsal

The whole path was exercised end to end with synthetic decisions over the real 80-record
packet, writing only to a scratch directory: import validated 80/80, consensus classified
all eight categories, and the disagreement packet generated. This proves the mechanics; the
resulting agreement rate is an artifact of synthetic input and is deliberately **not**
reported as a finding. No synthetic data was placed in the real input paths, which remain
absent.

## 7. Status Separation (§7)

```text
HUMAN_GOLD            a person decided                     ← the only human annotation
AI_ASSISTED_DRAFT     first model's pre-annotation
AI_SECOND_OPINION     second model, independently
AI_CONSENSUS_HIGH     both agree under strict conditions
AI_WEAK_HIGH_CONFIDENCE  promotable weak label
SILVER_UNIQUE_EXACT   automatic unique exact alias
```

All five AI statuses are in `AI_DRAFT_STATUSES`; `HUMAN_GOLD_STATUSES` remains exactly
`{HUMAN_GOLD, SECOND_REVIEW_AGREED}`; the intersection is empty and `assert_not_ai_gold()`
checks that as a property of the vocabularies. `AI_CONSENSUS_HIGH` may later serve as weak
supervision if a milestone explicitly permits it, and may never be reported as clinical
benchmark accuracy.

## 8. Diagnostics for the Next Intervention (§8)

`summary.json` carries agreement and disagreement rates, `SELECTED` and
`NO_VALID_CANDIDATE` agreement counts, per-ontology breakdowns and high-consensus counts —
computed only from real returned decisions. The intervention choice
(`COVERAGE_FIRST_PRETRAINED_RETRIEVER`, `RERANKER_FIRST`, `KB_ALIAS_REPAIR_FIRST`,
`HYBRID_RETRIEVAL_PLUS_RERANKING`) is deferred until those numbers exist. Audit 0066's
evidence points at coverage rather than ranking, but that is a hypothesis this exchange is
designed to test, not a conclusion to record in advance.

## 9. Tests and Static Validation

`tests/unit/test_ai_to_ai_handoff_0067.py` — **20 tests** covering all eleven required
proofs: the blind export contains no first opinion (records, candidates and Markdown); all
80 expected ids exported; no handoff artifact is tracked or trackable while the schema is;
the importer rejects unknown ids, duplicates, un-offered codes, dotted ICD codes, mismatched
ontologies and silent partial returns; `AI_SECOND_OPINION` and `AI_CONSENSUS_HIGH` are never
human gold; consensus requires exact code agreement; disagreement export works; and no tool
writes to `pack.jsonl`.

```text
targeted   ai_to_ai_handoff_0067 + ai_linker_drafts_0066     35 passed
ruff check .                                                 All checks passed!
ruff format --check <5 changed files>                        5 files already formatted
mypy src/mednorm_vi              Success: no issues found in 292 source files
compileall -q src tests scripts                              OK
git diff --check                                             OK
```

## 10. Privacy Verification

`git check-ignore` confirms every handoff, second-opinion and consensus path is ignored;
`git ls-files` lists nothing under `data/private_linker_gold/`; the tracked schema is
confirmed *not* ignored. This audit names no mention, no context and no patient content.

## 11. Changed-File Inventory

New (tracked): `schemas/annotation/ai_second_opinion_v1.schema.json`,
`scripts/export_chatgpt_handoff_0067.py`, `scripts/import_ai_second_opinion.py`,
`scripts/build_ai_consensus_0067.py`, `tests/unit/test_ai_to_ai_handoff_0067.py`,
`docs/audits/0067-ai-to-ai-linker-adjudication-handoff.md`.

Modified (tracked): `src/mednorm_vi/annotation/linker_gold.py` — adds
`STATUS_AI_SECOND_OPINION` and `STATUS_AI_CONSENSUS_HIGH` to `AI_DRAFT_STATUSES`.

New private/ignored: `chatgpt_handoff/` (3 files). `ai_second_opinion/`, `ai_consensus/`
and the disagreement packet are created on import and do not yet exist.

No earlier audit was modified.

## 12. Next Owner Action

1. Upload **`data/private_linker_gold/0065/chatgpt_handoff/minimum_benchmark_blind.md`**
   to the second AI (the `.jsonl` is the machine-readable equivalent if preferred).
2. Save its reply as
   `data/private_linker_gold/0065/chatgpt_handoff/chatgpt_decisions.jsonl` — one JSON
   object per line, 80 lines, matching `ai_second_opinion_v1.schema.json`.
3. Run the importer, then the consensus builder. Both commands are in the final response.

## 13. Remaining Risks

1. **Two AIs agreeing is not truth.** Shared blind spots — particularly on Vietnamese
   clinical vocabulary — would produce confident agreement that is confidently wrong. The
   single human decision available agrees with the AI draft, which is encouraging and is
   one datapoint.
2. **The second system sees only what the local KB offered.** Where the retriever returned
   nothing or only false friends, no adjudicator can recover the right code; those records
   will show agreement on `NO_VALID_CANDIDATE` that reflects KB coverage, not linking skill.
3. **`AI_CONSENSUS_HIGH` will be tempting to treat as gold.** The status separation and
   tests exist precisely because that temptation will recur under schedule pressure.
4. **The second AI's compliance with the schema is unverified until it returns.** The
   importer will refuse a malformed return, which may cost a round trip.

## 14. Verdict

**`CHATGPT_HANDOFF_READY`**

The blind 80-record packet is exported and verified free of the first opinion, the response
schema is defined and tracked, the importer and consensus builder are written and
round-trip tested, and every AI status is provably outside human gold. No second-opinion
data exists yet, so no consensus or intervention claim is made.


---

# APPENDIX A — Real Import and Consensus Results

Appended after the second adjudication was returned. Nothing above is modified. No clinical
text appears here.

## 15. Import (§1–§3)

`chatgpt_decisions.jsonl` verified before use: **SHA-256
`1aea32f8ef6280236b5fd82dc8909e93de69f62e334d3b354907ccf656d415dd` matches**, 80 non-empty
records, 80 distinct annotation ids, 50 ICD-10 / 30 RxNorm, and the file is gitignored.

`scripts/import_ai_second_opinion.py` accepted **80/80** with no `--allow-partial`. All
seven validations passed: no unknown ids, no duplicates, ontology unchanged on every
record, every `SELECTED` code was among those offered, ICD codes dotless,
`NO_VALID_CANDIDATE`/`ABSTAIN` carry no codes, and no expected record missing.

| | |
| --- | --- |
| Output | `ai_second_opinion/chatgpt_second_opinion.jsonl`, sha256 `da42ef87b500139793e7233c3e5e568ff8e7b6a42f158aff3ac44e48610e4311` |
| Status | `AI_SECOND_OPINION` |
| Second-opinion decisions | NO_VALID_CANDIDATE 56, SELECTED 17, ABSTAIN 7 |
| Confidence | HIGH 65, MEDIUM 15 |
| `pack.jsonl` modified | **no** |

## 16. Consensus (§4–§6)

| Metric | Value |
| --- | ---: |
| Compared | 80 |
| **EXACT_AGREEMENT** | **69** |
| **Agreement rate** | **0.8625** |
| — SELECTED, exact code-set agreement | 15 |
| — NO_VALID_CANDIDATE agreement | 54 |
| BOTH_ABSTAIN (abstention agreement) | 4 |
| Disagreements | 11 |
| **AI_CONSENSUS_HIGH** | **69** (ICD 39, RxNorm 30) |

Disagreements are `CLAUDE_ABSTAIN` 3, `CHATGPT_ABSTAIN` 3, `BOTH_ABSTAIN` 4,
`DECISION_AGREEMENT_CODE_DISAGREEMENT` 1 — **all 11 are ICD-10; RxNorm produced zero
disagreements.** No record fell into `CLAUDE_SELECTED_CHATGPT_REJECTED`,
`CHATGPT_SELECTED_CLAUDE_REJECTED` or `OTHER_DISAGREEMENT`: the two systems never
contradicted each other outright, they only differed in when to abstain.

### Production linker versus consensus

| Measure | Value |
| --- | ---: |
| Consensus `SELECTED` records | 15 |
| Linker top-1 equals the consensus code | 11 (**73.3%**) |
| Consensus `NO_VALID_CANDIDATE` records | 54 |
| — where the linker nonetheless offered ≥1 candidate | **43** |

## 17. Failure Taxonomy

| Failure class | Count |
| --- | ---: |
| **Coverage** — no offered candidate is correct | **54 / 69 (78%)** |
| — empty offered set | 11 |
| — candidates offered but all wrong | 43 |
| **Ranking** — correct code offered but not top-1 | **4 / 69 (5.8%)** |

| ICD-10 (39 consensus) | | RxNorm (30 consensus) | |
| --- | ---: | --- | ---: |
| No candidate represents the mention | 12 | Vocabulary / brand / class coverage | 21 |
| Empty offered set | 11 | TTY / ingredient-level resolution | 8 |
| Hierarchy / specificity related | 9 | Strength or dose-form ranking failure | 0 |
| Lexical false friend | 3 | | |

**Ranking is not the bottleneck.** Where the correct concept reached the candidate set, the
existing linker already ranked it first 73.3% of the time, and only 4 records in 69 were
ranking failures. Coverage outnumbers ranking **13.5 : 1**.

## 18. Why the KB, not the retriever, is first

The obvious reading of §17 is "retrieval is weak, train a retriever". That was the
Audit-0066 hypothesis and the evidence does not support acting on it yet.

**The correct concepts are already in the index.** Probing 16 concepts the consensus judged
missing from the offered sets, **15 are present** in the frozen ICD index. Only essential
hypertension (I10) is genuinely absent. So these are not missing-concept failures.

**They are unfindable because their searchable text is not a disease name.** Every ICD
record is reachable by at least one exact posting, but for a substantial minority that
posting is an ICD *tabular instruction* rather than a name:

| Concept | Only searchable strings in the index |
| --- | --- |
| rectal cancer (C20) | `bao gồm: bóng` (“includes: …”), `c20` |
| measles (B05) | `- bao gồm: bệnh`, `b05` |

No retriever — lexical, dense, pretrained or fine-tuned — can match a Vietnamese mention to
a concept whose entire searchable surface is a truncated cross-reference note. Index-wide:

| ICD index condition | Count | Share |
| --- | ---: | ---: |
| `canonical_name` is an ICD instruction (`Bao gồm` / `Loại trừ` / `Incl.` / `Excl.`) | 1,280 | 8.4% |
| `canonical_name` ends mid-phrase (`,` `:` `(` `-`) | 1,347 | 8.8% |
| `canonical_name` ≤ 6 characters | 318 | 2.1% |
| **union of damaged-name conditions** | **2,746** | **17.9%** |
| KB's own `name_quality: suspect` flag | 3,470 | 22.7% |
| `records[].aliases` populated | **0** | **0%** |

The last row is the structural gap. RxNorm carries 82,429 populated alias lists; the ICD
side carries **none** — it never received a synonym layer at all, so its only searchable
text is one canonical name per concept, damaged in roughly a fifth of cases.

Where the name *is* clean the failure mode is a plain synonym gap: pancreatic cancer is
stored as *u ác tính ở tụy* (“malignant neoplasm of…”) while the clinical text says *ung
thư tụy* (“cancer of…”). That is exactly what an alias layer fixes, and exactly what a
dense retriever would have to learn around.

**RxNorm's failures are a different thing and are not repairable by retrieval at all.** 21
of 30 RxNorm consensus records are Vietnamese drug *class* terms and local *brand* names.
RxNorm contains no concept for either, so `NO_VALID_CANDIDATE` is the correct output, not a
defect. Chasing those with a better retriever would be chasing concepts that do not exist.

## 19. Next Intervention

**`KB_ALIAS_REPAIR_FIRST`.**

This overrides the Audit-0066 hypothesis rather than preserving it. The reasoning:

1. Coverage dominates ranking 13.5 : 1, so `RERANKER_FIRST` addresses 4 records out of 69.
2. The concepts already exist (15/16), so this is not missing KB content.
3. Their searchable text is damaged (17.9%) or vocabulary-mismatched, and the ICD alias
   layer is entirely absent — **the KB side of any similarity function is broken**.
4. Training a retriever now would fit it to corrupted text: it would learn to embed
   “Bao gồm: Bóng” as rectal cancer, and that error would be baked into the weights.
5. Alias repair is deterministic, needs no GPU and no parameter budget, and improves both
   the current lexical channel and any future dense retriever.

`COVERAGE_FIRST_PRETRAINED_RETRIEVER` and `HYBRID_RETRIEVAL_PLUS_RERANKING` remain the right
*second* step, and the 9B budget and Colab/A100 availability make them practical — but they
should be trained against a repaired index, not this one.

Concretely, the repair is: restore real ICD-10 Vietnamese titles for the 2,746 damaged
records from the organizer's frozen source, populate an alias layer (Vietnamese synonym
variants such as *ung thư* / *u ác tính* / *bướu*, accented and unaccented forms,
abbreviations), and re-measure this same 80-record consensus set before any training.

## 20. Round-2 Disagreement Packet (§8)

The 11 disagreements are exported for a second ChatGPT round rather than returned to the
owner for manual adjudication. This packet **may** show both prior opinions — blindness
applied to round 1 only — and adds local KB enrichment (ICD sibling concepts and hierarchy;
RxNorm graph neighbours). No external API was used.

| Artifact | SHA-256 |
| --- | --- |
| `chatgpt_handoff/disagreements_for_chatgpt.jsonl` | `a77d86c423cb6f34bf1f5ebb7f349e64205b4ffd64ef2b8672f08e2847d96f5e` |
| `chatgpt_handoff/disagreements_for_chatgpt.md` | `aaca5756919b023b81da7fdde596ad53469f5d91aabd1a858203d0d963d6ad61` |

All 11 are ICD-10 and all are abstention-related, so reconciliation is about *how much
specificity the context supports*, not about competing code claims.

## 21. Status Discipline

The 69 consensus records carry `AI_CONSENSUS_HIGH`. None is `HUMAN_GOLD`. The benchmark
still reports zero human-reviewed records for both ontologies, and
`evaluate_linker_gold.py` still refuses to emit metrics. The one human decision in the pack
is untouched, is flagged `human_gold_present` where consensus overlaps it, and **agrees with
both AI systems**.

86.25% agreement between two independently-prompted systems is a strong engineering signal
and is not a measurement of correctness. Both systems read the same damaged index; a shared
blind spot would produce exactly this agreement. That is the reason `AI_CONSENSUS_HIGH`
exists as a separate status and the reason no accuracy claim is made here.

## 22. Verdict

**`AI_CONSENSUS_READY`**

80/80 second opinions imported under full validation, 69 records reached
`AI_CONSENSUS_HIGH` at 86.25% agreement, the 11 disagreements are packaged for automated
round-2 reconciliation, and the evidence selects **`KB_ALIAS_REPAIR_FIRST`** over the prior
retriever hypothesis. No model was trained or downloaded; `pack.jsonl` was never modified;
nothing was staged, committed or pushed.
