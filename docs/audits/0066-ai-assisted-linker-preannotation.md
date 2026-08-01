# Audit 0066: AI-Assisted Linker Pre-Annotation

Milestone: 4G - AI-assisted linker pre-annotation, blind human verification, weak-label quality gate
Date: 2026-08-01
Verdict: `AI_DRAFTS_INSUFFICIENT`

## 1. Git and Resource State

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `7f74cb2 feat: establish governed ICD and RxNorm adjudication` |
| Working tree | clean |
| Staged / `origin/main...HEAD` | none / `0 0` |
| Audit 0065 committed | yes |
| RAM / disk / containers | 5.6 GiB available / 24 G free / 0 |
| `docs/MedNorm-VI_Architecture.pdf` | `0d5eaa20…81e09b`, unchanged |
| `pack.jsonl` | `1898d726…8226d`, unchanged |

No model was trained, downloaded or run. No public inference. No `internal_test` access. No
external API and no clinical text left the machine.

## 2. Coverage — stated first, because it is the limitation

**85 of the 300 records were drafted: 50 ICD-10 and 35 RxNorm.** The brief asked for all
300. Each drafted record was read individually — mention, bounded context, and all offered
candidates with their structural evidence — and decided on that evidence. The remaining
**215 records carry no draft**, and none was fabricated, rule-filled, or given a placeholder
decision to reach a count.

What the 85 do cover is the owner's stated first target exactly: **the 80-record minimum
benchmark queue (50 ICD + 30 RxNorm) is fully pre-annotated**, so human review can begin
immediately with a rationale and the relevant KB evidence already attached to every record
in it.

The verdict is `AI_DRAFTS_INSUFFICIENT` on that basis: the drafts that exist are usable and
the queue that matters is complete, but the stated scope was not met.

## 3. Draft Status and Storage

Two new statuses in `src/mednorm_vi/annotation/linker_gold.py`, deliberately **not** a reuse
of `HUMAN_UNCERTAIN` or `SILVER_UNIQUE_EXACT`:

```text
AI_ASSISTED_DRAFT          a model's pre-annotation
AI_WEAK_HIGH_CONFIDENCE    the most a draft may ever be promoted to
```

Both sit in `AI_DRAFT_STATUSES` and neither is in `HUMAN_GOLD_STATUSES`. `assert_not_ai_gold()`
checks that as a property of the vocabularies rather than of any single record, because the
failure it prevents — a model's draft quietly counting as ground truth — is invisible
record-by-record.

Private, gitignored artifacts under `data/private_linker_gold/0065/ai_drafts/`:

| File | SHA-256 |
| --- | --- |
| `claude_code_v1.jsonl` (85 drafts) | `a793f601e97bce3f4bc22dec4c6e6d5bf9b0887dd933481208825b74406afabd` |
| `manifest.json` | `45d2083521c01c7585218fe8865ea985c1b9cbfc10aa1908a9de3721737c196d` |
| `summary.json`, `queues/` | derived |

The manifest records git commit `7f74cb2e…`, the architecture PDF hash, pack hash, KB
snapshot `competition-kb-v3-candidate-4d206538e7d24c32`, both index hashes, method
`claude-code-two-pass-evidence-grounded-v1`, timestamp, processed count (85 of 300),
deterministic processing order (ascending `annotation_id`) and schema version
`ai_linker_draft_v1`.

Drafts are written to a separate file and **never** touch `pack.jsonl`, so a human decision
cannot be overwritten by a model's; the writer additionally skips any record already
adjudicated by a person.

## 4. Annotation Rules Applied

Only local evidence was used: mention text, bounded context, candidate canonical names and
aliases, ICD hierarchy and specificity class, RxNorm TTY, ingredient, strength, dose form
and brand fields. No candidate was invented and no memorised external mapping was applied
without a local candidate to attach it to.

`NO_VALID_CANDIDATE` was used freely and is the majority outcome, which is the finding
rather than a failure to decide. `ABSTAIN` was used where specificity genuinely could not
be resolved from context. No code was forced to raise a completion rate.

### ICD-10

Dotless identity preserved throughout; the dotted form is display only. A `.8`/`.9` code
was never taken merely because it ranked first — `hạ đường huyết` took **E16.1** over the
rank-1 **P70.4**, which is *neonatal* hypoglycaemia in an adult context. Organ, cause,
acuity and complication were never inferred: `viêm phổi` was refused because every offered
code named a specific organism the context did not state, and `sa sút trí tuệ` was refused
because every candidate attributed a cause the context explicitly left open.

### RxNorm

TTY was respected. Nine drafts resolved to an ingredient (`IN`) where the mention was the
ingredient with no strength or dose form — `vitamin E → 11256`, `omega-3 → 4301` — which is
the correct level, not a fallback. No SCD was chosen without matching strength and form, and
no brand concept without brand evidence. Exact score ties were never broken arbitrarily.

## 5. Two-Pass Procedure

**PASS A** — independent draft from evidence. Candidate display order was already
randomized by the pack; `linker_rank` was retained as metadata and did not drive any
decision.

**PASS B** — self-critique over all 22 `SELECTED` and 22 `MEDIUM`-confidence drafts (34
distinct records), attempting to disprove each PASS-A selection.

**One decision did not survive and was reversed**, recorded rather than silently changed:

> `2d7edf67` — *teo thực quản* (oesophageal atresia). PASS A selected **Q390**. PASS B
> rejected it: Q39.0 is atresia **without** fistula and Q39.1 is **with** fistula, and the
> context states neither. Selecting Q39.0 would infer the absence of a fistula. Changed to
> **ABSTAIN**, flag `multiple_plausible`.

One PASS-B challenge *strengthened* a draft rather than reversing it: `11e4c1c5` selected
H48.0 (optic atrophy in diseases classified elsewhere), and re-reading the context confirmed
an underlying cause — ischaemia after head-and-neck radiotherapy — which is exactly what
that code requires.

`pass_b_changed_decisions: 1` is recorded in the manifest.

## 6. Draft Distribution

| | ICD-10 | RxNorm | total |
| --- | ---: | ---: | ---: |
| Records drafted | 50 | 35 | **85** |

| Decision | count | | Confidence | count |
| --- | ---: | --- | --- | ---: |
| NO_VALID_CANDIDATE | 57 | | HIGH | 62 |
| SELECTED | 21 | | MEDIUM | 23 |
| ABSTAIN | 7 | | LOW | 0 |

**The drafts agree with the production linker's top-1 in 13 of 85 records (15.3%).** That
is the headline number of this milestone. It is not yet an accuracy claim — no human gold
exists — but it means the benchmark's disagreement queue is not a corner case: it is 61 of
85 records.

Ambiguity flags (top): `drug_class` 11, `empty_set` 11, `no_candidate_represents` 9,
`ingredient_only_appropriate` 9, `brand_not_in_kb` 7, `specificity_ok` 5,
`lexical_false_friend` 4, `exact_concept` 4, `no_valid_product` 3, `multiple_plausible` 3.

### What the evidence shows

Two failure modes dominate, and they differ by ontology.

**ICD** fails on *coverage and false friends*. `sởi` (measles) retrieved only urinary
calculus codes — `sỏi` differs by one diacritic. `hội chứng qt kéo dài` (long QT) retrieved
three iliotibial-band syndrome codes. `ung thư trực tràng` (rectal cancer, C20) was offered
only benign neoplasms and carcinoma-in-situ — the wrong malignancy class entirely. Eleven
diagnoses had no candidate at all.

**RxNorm** fails on *vocabulary mismatch*. Vietnamese consumer-health text is dominated by
drug **class** terms (`thuốc hạ sốt`, `men vi_sinh`, `thuốc tránh thai`) and local **brand**
names (`Skinbibi`, `Eclaran`, `Biogaia`, `Flexsa`) — 18 of 35 records. RxNorm represents
neither, so the retriever returns whatever shares character n-grams, which was frequently a
topical salicylic-acid product. Only the vitamin/supplement mentions linked cleanly.

Neither pattern is a ranking problem. Both are coverage problems, which is a different
intervention from the one Audit 0063 tentatively favoured — and precisely why human review
must confirm before anything is trained.

## 7. Blind Review Without Anchoring

`scripts/adjudicate_linker_gold.py` gained four modes. In `--blind-ai-review` the reviewer
sees the normal randomized candidate evidence and **must decide first**; the decision is
saved, the pre-reveal decision is preserved verbatim in
`human_decision_pre_ai_reveal`, and only then is the AI draft shown with an option to
revise. Any revision is marked in the notes. The AI-selected candidate is never displayed
before the initial human decision, so the draft cannot act as an anchor, and the human
decision remains authoritative in every path.

### Exact owner commands

```bash
# progress
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py --status

# the priority session: the 80-record minimum benchmark queue, blind
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py \
    --reviewer-id <your-id> --blind-ai-review --resume \
    --queue data/private_linker_gold/0065/ai_drafts/queues/queue_minimum_benchmark.txt

# where the AI and the production linker disagree (61 records)
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py \
    --reviewer-id <your-id> --blind-ai-review --review-ai-disagreements --resume

# audit a deterministic sample of HIGH-confidence drafts
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py \
    --reviewer-id <your-id> --blind-ai-review --resume \
    --queue data/private_linker_gold/0065/ai_drafts/queues/queue_high_confidence_audit.txt

# AI quality, once 50 ICD + 30 RxNorm human records exist
env PYTHONPATH=src .venv/bin/python scripts/evaluate_ai_linker_drafts.py
```

## 8. Review Queues

| Queue | size | ICD | RxNorm |
| --- | ---: | ---: | ---: |
| `queue_minimum_benchmark` | **80** | 50 | 30 |
| `queue_ai_disagreement` | 61 | 34 | 27 |
| `queue_high_confidence_audit` | 20 | 11 | 9 |
| `queue_ai_abstain` | 7 | 7 | 0 |
| `queue_ai_low_confidence` | 0 | 0 | 0 |

`queue_minimum_benchmark` is stratified round-robin across ambiguity flags, not taken in id
order: reviewing 50 easy exact-alias records would satisfy the count and teach nothing.
The low-confidence queue is empty because no draft was assigned LOW — where evidence was
that weak the decision became `ABSTAIN` instead.

## 9. Quality Evaluation — refused, as designed

`scripts/evaluate_ai_linker_drafts.py` exists and **refuses to report accuracy**:

```text
ICD10       0 / 50 required
RXNORM      0 / 30 required
REFUSING TO REPORT AI ACCURACY.
```

An AI draft compared against itself is not a measurement. The weak-label gate
(HIGH-confidence exact agreement ≥ 90%, HIGH-confidence no-valid precision ≥ 90%, ≥ 30
HIGH-confidence reviewed per ontology, no collapsed ICD specificity or RxNorm TTY subgroup)
is closed and reports `passed: false`. Passing it would promote drafts only as far as
`AI_WEAK_HIGH_CONFIDENCE` — never to `HUMAN_GOLD`.

## 10. Tests

`tests/unit/test_ai_linker_drafts_0066.py` — **15 tests** covering all six required proofs:
AI statuses are never gold (checked over the vocabularies themselves); 200 AI drafts leave
the benchmark exactly as closed as zero would; a record already adjudicated by a human is
skipped and the pack left byte-identical; promotion has no code path (the writer's source
is asserted not to contain a human-gold status); a selected code must be among those
offered; ICD codes must be dotless; ontology restrictions hold; and the private artifacts
are gitignored.

```text
targeted   ai_linker_drafts_0066 + linker_gold_schema/benchmark_0065     50 passed
full       pytest tests/ -q                    2046 passed, 2 skipped in 482.68s
ruff check .                                                            All checks passed!
ruff format --check <6 changed Python files>                            6 files already formatted
mypy src/mednorm_vi                     Success: no issues found in 292 source files
compileall -q src tests scripts                                         OK
git diff --check                                                        OK
```

## 11. Changed-File Inventory

New (tracked):

```text
scripts/build_ai_linker_drafts_0066.py
scripts/build_ai_review_queues_0066.py
scripts/evaluate_ai_linker_drafts.py
tests/unit/test_ai_linker_drafts_0066.py
docs/audits/0066-ai-assisted-linker-preannotation.md
```

Modified (tracked):

```text
src/mednorm_vi/annotation/linker_gold.py   AI_ASSISTED_DRAFT / AI_WEAK_HIGH_CONFIDENCE,
                                           AI_DRAFT_STATUSES, is_ai_draft, assert_not_ai_gold
scripts/adjudicate_linker_gold.py          blind-AI review modes and queue filtering
```

New private/ignored: `data/private_linker_gold/0065/ai_drafts/{claude_code_v1.jsonl,
manifest.json,summary.json,queues/*}`, `runs/diagnostics/0066_ai_quality/ai_quality.json`.

No earlier audit was modified. No clinical text is tracked: rationales name codes and
concepts, never patient content, and the drafts live under `data/**`, which is gitignored.

## 12. No AI Draft Is Treated as Human Gold

Stated plainly because it is the milestone's central risk: every one of the 85 drafts
carries `AI_ASSISTED_DRAFT`; `is_human_gold` is false for all of them; the benchmark counts
zero of them; the weak-label gate is closed; and no code path can promote one to
`HUMAN_GOLD`. The pack's human decisions were never written to by this milestone.

## 13. Remaining Risks

1. **215 of 300 records have no draft.** The stated scope was not met, and a follow-up pass
   is required before the whole pack is review-ready.
2. **Single-annotator drafts with no verification.** 15.3% agreement with the production
   linker means the drafts are either a substantial correction or substantially wrong, and
   nothing here distinguishes those two until humans review.
3. **`NO_VALID_CANDIDATE` is 67% of drafts.** If that rate survives review it is a strong
   result about KB coverage; if the reviewer disagrees, the drafts are systematically too
   conservative. Either way the benchmark's answerable population will be small.
4. **35 RxNorm drafts, minimum 30** leaves almost no margin: a few reviewer disagreements
   on eligibility could drop the count below the reporting threshold.
5. **The Vietnamese class/brand vocabulary gap** (18 of 35 medication mentions) may not be
   fixable by any retriever or reranker; it may require KB extension or a decision that
   such mentions legitimately receive no candidate.

## 14. Verdict

**`AI_DRAFTS_INSUFFICIENT`**

85 of 300 records were drafted from evidence over two passes, with one self-critique
reversal recorded. The infrastructure is complete and tested — draft status that cannot be
gold, a writer that validates against the frozen pack, blind review that cannot anchor,
five prioritised queues, and a quality evaluator that refuses to speak before human gold
exists. The owner's 80-record minimum benchmark queue is fully pre-annotated and ready for
blind review. The remaining 215 records are not, and are not pretended to be.
