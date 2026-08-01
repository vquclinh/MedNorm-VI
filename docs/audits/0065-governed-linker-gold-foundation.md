# Audit 0065: Governed Linker Gold Foundation

Milestone: 4F - Linker gold-set foundation, human adjudication workflow, honest ICD/RxNorm benchmark
Date: 2026-08-01
Verdict: `LINKER_GOLD_PACK_READY_FOR_REVIEW`
Next intervention: `MORE_HUMAN_ADJUDICATION_REQUIRED`

## 1. Git and Resource State

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `55cf161 research: evaluate source-balanced E3 refinement` |
| Working tree | clean |
| Staged files | none |
| `origin/main...HEAD` | `0 0` |
| Audit 0064 committed | yes |
| RAM | 14 GiB total, 5.9 GiB available |
| Disk | 24 G free (77% used) |
| GPU | RTX 4060 Laptop, 8,188 MiB, 13 MiB used |
| Containers | 0 |

Protected digests verified before any work and again at the end:

```text
docs/MedNorm-VI_Architecture.pdf                 0d5eaa20…81e09b   unchanged
checkpoint/e3_boundary_refinement_0062/best.pt   524ece1e…dde3a    ACTIVE, unchanged
checkpoint/s1_mention_full_training_v1/best.pt   a64cc173…1017c    ROLLBACK, unchanged
```

No model was trained, downloaded or activated. No public inference was run.

## 2. The Cleanup-Guard Defect, Fixed

Audit 0064 §2 recorded two defects in the ad-hoc cleanup that preceded this milestone.
Both are now fixed **structurally**, in `scripts/safe_remove_project_artifact.py`.

**Defect 1 — a substring grep counted a provenance field as a live reference.**
`e3_checkpoint_profiles.yaml` records `experiment_origin:` to document where a checkpoint
came from. Grepping for the path matched that line and reported a runtime reference that
does not exist. The tool now **parses** the three governed configs and treats only
load-bearing fields as live:

| Config | Live fields | Provenance-only (never blocking) |
| --- | --- | --- |
| `e3_checkpoint_profiles.yaml` | `path` | `experiment_origin`, `source_checkpoint`, `notes` |
| `full_v1.yaml` | `expert_settings.e3_checkpoint_path` | — |
| `candidate_model_registry.yaml` | `active_checkpoint_path` | `notes` |

**Defect 2 — the script printed `ABORT` and continued.** A printed warning is not a guard.
Every failed precondition now raises `RemovalRefused` and the process exits **2** before
anything is unlinked. Verified directly:

```text
active --execute      -> exit 2
rollback --execute    -> exit 2
absent candidate      -> exit 2
unbounded directory   -> exit 2
outside repo          -> exit 2
```

Other properties, all enforced rather than documented: dry-run is the default; `--execute`
is required; paths resolve canonically and must stay inside the repository; symlinks are
refused for the target **and every parent**; the active and rollback checkpoints are
refused **by digest** as well as by path, so renaming or relocating one changes nothing; a
directory target must be bounded (no subdirectories); no wildcards, no shell, no `eval`; a
receipt is written for every executed removal.

`tests/unit/test_safe_remove_project_artifact_0065.py` — **13 tests**, covering all eight
required proofs plus a group-atomicity case (one bad target refuses the whole group, since
a partial deletion is not safer).

## 3. The Linker-Gold Schema

`schemas/annotation/linker_gold_v1.schema.json` plus a strict validator at
`src/mednorm_vi/annotation/linker_gold.py`.

Every record carries all 22 required fields. The rules that matter, all enforced in code
and covered by tests:

* `CHẨN_ĐOÁN → ICD10` and `THUỐC → RXNORM` only (spec §7.3 forbids the cross-links);
* `selected_codes` may hold several codes; `no_valid_candidate` is explicit and **mutually
  exclusive** with any selection;
* a `HUMAN_GOLD` record must carry a decision — codes or an explicit "none valid";
* an `UNLABELLED` record may carry none;
* a reviewer may only select from candidates they were actually shown;
* **ICD identity stays dotless** — a dotted `code` is refused; the dot lives in
  `display_code` (Audit 0061);
* RxNorm candidates must record TTY;
* every selected code must exist in the frozen snapshot when the snapshot is supplied;
* `annotation_id` is derived from document hash + offsets, so re-sampling a mention cannot
  create a second record for it.

**Silver can never be counted as gold.** `HUMAN_GOLD_STATUSES` is exactly
`{HUMAN_GOLD, SECOND_REVIEW_AGREED}`; `SILVER_UNIQUE_EXACT` sits outside it, and
`is_human_gold` is the only predicate the benchmark uses.

`tests/unit/test_linker_gold_schema_0065.py` — **23 tests**.

## 4. Sampling Design and the Private Pack

`data/private_linker_gold/0065/` (ignored by `data/**`).

| | |
| --- | --- |
| Pack | `pack.jsonl`, **300 records**, sha256 `1898d726…8226d` |
| Manifest | `manifest.json`, sha256 `93915fdf4730d10a901e3a6ab12c6640592a36ea1d486c76802308ebd2fdb06c` |
| Silver | `silver/silver.jsonl`, 77 records, sha256 `4a56a829…40af49` |
| Seed | 20260801 |
| Validation split | `ed7cdd2d…f68f103` |
| Snapshot | `competition-kb-v3-candidate-4d206538e7d24c32` |
| Eligible gold mentions | 1,458 (1,302 DIAGNOSIS + 156 MEDICATION) |
| Deduplicated away | 0 (all 1,458 distinct by normalized mention + context) |

Sampling uses **gold** mention spans and types only — E3 predictions never decide
eligibility. Had predictions chosen the sample, the benchmark would measure the linker on
whatever the mention layer happens to find, and every later comparison would inherit that
bias.

### A stratification design that had to be corrected

The first implementation assigned each mention to the **first** stratum it matched. That
produced a pack in which the brief's medication strata — strength-bearing, dose-form,
ingredient+strength+form, ingredient-only — were all **empty**, because
`rx_exact_ranking_tie` and `rx_saturated_20_candidates` apply to almost every medication
mention (Audit 0063 §7.4: 46.2% ties, 83.3% saturated) and absorbed the entire population.

The fix: record **every** stratum a mention matches as a facet, and assign the primary
stratum used for quotas to the **rarest** matching facet. Rare, hard buckets now claim
their mentions before abundant ones can swallow them, and every declared stratum with any
population is represented.

| DIAGNOSIS stratum | primary | present as facet |
| --- | ---: | ---: |
| dx_empty_candidate_set | 27 | 41 |
| dx_exact_kb_alias | 27 | 27 |
| dx_low_margin_ranking | 27 | 45 |
| dx_vague_top1_category_or_unspecified | 27 | 53 |
| dx_len_1_word | 26 | 26 |
| dx_len_2_3_words | 26 | 86 |
| dx_len_4_7_words | 26 | 74 |
| dx_len_8plus_words | 14 | 14 |
| dx_hierarchy_parent_child_ambiguity | 0 | 101 |
| dx_fuzzy_only | 0 | 132 |
| **total** | **200** | |

| MEDICATION stratum | primary | present as facet |
| --- | ---: | ---: |
| rx_exact_ranking_tie | 23 | 36 |
| rx_tty_ambiguity_in_pin_scd_sbd | 22 | 72 |
| rx_exact_kb_alias | 15 | 15 |
| rx_brand_like | 14 | 20 |
| rx_saturated_20_candidates | 14 | 81 |
| rx_dose_form | 8 | 10 |
| rx_ingredient_strength_form | 2 | 2 |
| rx_strength_bearing | 2 | 4 |
| rx_ingredient_only | 0 | 88 |
| **total** | **100** | |

All 19 declared strata are represented. The two DIAGNOSIS strata and one MEDICATION
stratum with a primary count of 0 are the most *abundant* ones — every member had a rarer
facet — and they remain heavily present as facets, so no analysis loses them.

Records are drawn from both validation sources: 200 `vimedner` (all DIAGNOSIS) and 100
`vimq` (all MEDICATION), which follows from the type-disjoint corpus recorded in Audit
0064 §3.1.

## 5. Review Evidence

Each record carries up to **ten** candidates with the full evidence the brief requires —
ICD: dotless code, dotted display code, canonical name, aliases, parent, children,
hierarchy depth, retrieval channels, lexical score, specificity tier and
category/specific/other/unspecified class. RxNorm: RxCUI, canonical name, synonyms, TTY,
ingredient, strength, dose form, brand, graph relations, channels, score and matched
structured fields. A reviewer is never asked to choose between bare codes.

**Candidate display order is deterministically shuffled** (`display_order_seed` derived
from the annotation id) while `linker_rank` is retained in the record. A reviewer shown
"rank 1" tends to ratify it, and the benchmark would then be measuring agreement with the
system it exists to judge. The evaluator reads `linker_rank`, never display order — there
is a test that reverses the display list to prove it.

The first record in the pack illustrates why this work is needed: the mention
*kháng_sinh* ("antibiotic", a drug **class**, not a product) retrieves ten amino-acid
mixtures through `char_ngram` alone. That is the Audit-0063 finding — 82% of matches are
fuzzy — made concrete for a human to adjudicate.

## 6. Adjudication Workflow

`scripts/adjudicate_linker_gold.py`. One record at a time, validated against the schema
before saving, written **atomically** (temp file + `os.replace`) after every decision, so
an interrupt cannot truncate the pack. Nothing is transmitted anywhere and the corpus is
never printed in bulk.

Exact commands for the owner:

```bash
# progress at any time
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py --status

# review the 200 ICD records (resume skips anything already decided)
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py \
    --reviewer-id <your-id> --ontology ICD10 --resume

# review the 100 RxNorm records
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py \
    --reviewer-id <your-id> --ontology RXNORM --resume

# optional independent second pass over already-adjudicated records
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py \
    --reviewer-id <second-id> --second-review

# export a summary
env PYTHONPATH=src .venv/bin/python scripts/adjudicate_linker_gold.py \
    --export-summary runs/diagnostics/0065_benchmark/adjudication_summary.json
```

In-loop keys: `1 3 5` select those displayed candidates · `n` no valid candidate · `?`
uncertain · `s` skip · `u` undo the previous saved decision · `ch|cm|cl` set confidence ·
`!text` add a note · `q` save and quit.

The minimum useful session is **50 ICD + 30 RxNorm**; the full pack is 300.

## 7. Silver Labels, Kept Separate

77 silver records in `silver/silver.jsonl` — 72 ICD, 5 RxNorm — every one
`SILVER_UNIQUE_EXACT` with exactly one code. The rule is narrow by design: the normalized
mention matches **exactly one** active searchable concept by exact alias, and for RxNorm
the concept must additionally carry an unambiguous single TTY.

Fuzzy top-1 is never silver. No LLM prediction is used. Silver lives in its own directory,
is excluded from `HUMAN_GOLD_STATUSES`, and a test proves that 160 silver records do not
unlock the benchmark.

## 8. Benchmark Status

`scripts/evaluate_linker_gold.py` **refuses to report metrics** below 50 reviewed ICD and
30 reviewed RxNorm records.

```text
ICD10       0 reviewed / 50 required
RXNORM      0 reviewed / 30 required
REFUSING TO REPORT BENCHMARK METRICS.
```

That refusal is the deliverable, not a limitation. Audit 0063 had to report Recall@k as
*undefined*; a confident number computed from a handful of annotations would be worse,
because it would look like evidence. The output JSON carries `reportable: false` and
contains **no** `icd10` or `rxnorm` metric block at all.

When the minimums are met the evaluator reports, separately for each ontology and never
combined: KB coverage, Recall@1/2/5/10, MRR, top-1 exact accuracy, empty-set rate,
no-valid-candidate rate, candidate-set size distribution, and metrics stratified by
stratum, confidence, exact-alias vs fuzzy-only, and short vs long mentions. ICD adds
specificity distribution, parent/child confusion, `.8`/`.9` overuse and alias-language
coverage; RxNorm adds TTY accuracy, ingredient/strength/dose-form accuracy, TTY confusion
pairs, saturated-set metrics and tie-breaking metrics.

`tests/unit/test_linker_gold_benchmark_0065.py` — **12 tests**, including proof that rank
is taken from `linker_rank` and not display order, that `no_valid_candidate` records are
excluded from recall but counted, and that each §9 decision rule fires correctly.

## 9. Next-Intervention Decision

**`MORE_HUMAN_ADJUDICATION_REQUIRED`.**

Zero records are adjudicated, so no retrieval-versus-ranking claim is supportable. The
brief's own rule applies directly: *"Do not select a dense retriever or reranker without
meeting the minimum human gold counts."* The decision function returns this verdict
whenever `ready` is false, and it is the only branch reachable today.

Audit 0063's evidence still points at `ICD_SYNONYM_OR_RETRIEVAL_REPAIR` (8.8% verbatim
alias coverage) and `RXNORM_STRUCTURED_RERANKER` (46.2% exact score ties), but those
figures were computed **without gold codes** and cannot select an intervention on their
own. That is exactly the gap this milestone was built to close.

## 10. Checkpoint Cleanup (§13)

The owner confirmed `checkpoint/` is backed up off-machine to Google Drive under
`MedNorm-VI/checkpoint/`.

**The filesystem was inspected as the source of truth, and it was already in the §13.1
required final state.** The rejected weights were removed in Milestone 4E (Audit 0064 §9);
nothing remained that §13.3 authorises deleting.

Pre-cleanup inventory — `checkpoint/` 3.1 G, 25 files:

```text
1,615,513,431  checkpoint/e3_boundary_refinement_0062/best.pt      ACTIVE
1,615,513,303  checkpoint/s1_mention_full_training_v1/best.pt      ROLLBACK
        (23 small metadata files: manifests, resolved configs, validation metrics, histories)
```

All **19** authorized deletion candidates from §13.3 were checked with the safe-removal
tool. **0 were present.** Both protected artifacts were dry-run against the tool and
refused twice over — by digest *and* by live runtime reference — exiting 2.

| | |
| --- | --- |
| Deleted paths | **none** |
| Bytes reclaimed | **0** |
| Retained weight files | 2 (active + rollback) |
| Retained metadata files | 23 |
| Receipt | `runs/diagnostics/0065_checkpoint_cleanup/deletion_receipt.json` |
| Post-cleanup inventory sha256 | `95e2d217284e5c67bab5986a214317a8d38346d4684d32d7eaed64ea45f6a597` |

No additional weight file remains beyond the two required, so no individual justification
is needed. §13.5 verification — all twelve conditions pass, including a bounded real
checkpoint load (4 proposals) whose recorded runtime digest equals the configured digest.

## 11. Validation

```text
targeted   safe_remove_project_artifact_0065                        13 passed
targeted   linker_gold_schema_0065                                  23 passed
targeted   linker_gold_benchmark_0065                               12 passed
full       pytest tests/ -q                    2031 passed, 2 skipped in 491.03s
ruff check .                                                        All checks passed!
ruff format --check <9 changed Python files>                        9 files already formatted
mypy src/mednorm_vi                     Success: no issues found in 292 source files
compileall -q src tests scripts                                     OK
git diff --check                                                    OK
```

The mypy surface grew from 291 to 292 files: `src/mednorm_vi/annotation/linker_gold.py`.

## 12. Changed-File Inventory

New (tracked):

```text
schemas/annotation/linker_gold_v1.schema.json
src/mednorm_vi/annotation/linker_gold.py
scripts/safe_remove_project_artifact.py
scripts/build_linker_gold_pack_0065.py
scripts/adjudicate_linker_gold.py
scripts/evaluate_linker_gold.py
tests/unit/test_safe_remove_project_artifact_0065.py
tests/unit/test_linker_gold_schema_0065.py
tests/unit/test_linker_gold_benchmark_0065.py
docs/audits/0065-governed-linker-gold-foundation.md
```

Modified (tracked): **none.**

New untracked/private (ignored): `data/private_linker_gold/0065/{pack.jsonl,manifest.json,
silver/silver.jsonl}`, `runs/diagnostics/0065_checkpoint_cleanup/deletion_receipt.json`,
`runs/diagnostics/0065_benchmark/linker_benchmark.json`.

Deleted: **none.**

### No private clinical text is tracked

The pack carries bounded windows of clinical text and lives entirely under `data/**`,
which `.gitignore` covers. A test asserts the policy rather than the momentary state: the
schema must be trackable, and `pack.jsonl`, `manifest.json` and `silver/silver.jsonl` must
each be gitignored. `git status --untracked-files=all` lists no path under
`data/private_linker_gold/`.

## 13. Remaining Risks

1. **Zero records are adjudicated.** Everything downstream waits on human review; the pack
   is infrastructure, not evidence.
2. **A single reviewer is a single point of bias.** The second-review mode exists and is
   unused. For the strata where the linker is near-arbitrary (46% RxNorm ties), one
   opinion may not be enough.
3. **300 records is a small benchmark**, drawn deliberately toward hard cases. It will
   estimate hard-case behaviour well and overall accuracy poorly — a stratified estimator,
   not a random sample. Any headline figure must be reweighted or clearly labelled.
4. **`no_valid_candidate` will be the hardest judgement.** Distinguishing "the KB lacks
   this concept" from "I cannot find it in ten candidates" needs guidance the tool does not
   provide.
5. **Validation-only sampling.** The pack draws exclusively from governed validation, so it
   is an evaluation set. Promoting any partition to training is an explicit owner decision
   and has not been taken.
6. **RxNorm silver is thin** (5 records) because the unique-exact rule is strict. That is
   the correct trade, but it means silver will not meaningfully supplement RxNorm.

## 14. Verdict

**`LINKER_GOLD_PACK_READY_FOR_REVIEW`**

The blocker Audit 0063 identified — no ICD or RxNorm gold codes anywhere — now has the
machinery to be removed: a versioned schema with a strict validator, a stratified
300-record evidence pack covering all 19 declared strata, an atomic resumable adjudication
CLI, silver labels kept structurally separate, and a benchmark that refuses to speak before
it has enough evidence to speak honestly. The cleanup tooling that failed in Audit 0064 is
fixed and tested. Nothing was trained, submitted, or activated.
