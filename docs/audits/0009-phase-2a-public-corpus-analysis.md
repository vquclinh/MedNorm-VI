# Audit 0009 — Phase 2A: Public Corpus Analysis

- **Date:** 2026-07-17
- **Author:** Claude (AI agent), for human review
- **Change type:** Additive, read-only descriptive analysis of the organizer's
  100 PUBLIC INPUT documents. No prediction, no entity extraction, no linking,
  no LLM/ML, no training, no synthetic generation. **No existing L1 behavior was
  modified.**

> **This audit is tracked, so it contains aggregate statistics only.** No
> organizer document text is reproduced here. The full report (including heading
> surface forms) is written to `reports/corpus_analysis/`, which is git-ignored.

## 1. Objective and scope

Measure the organizer's public input corpus structurally so later phases are
driven by evidence rather than assumption: document lengths, section headings,
canonical routable-unit frequencies, list/table/key-value distributions,
laboratory layouts, medication-list patterns, imaging/report styles, and overall
document structure — exported as reproducible JSON + Markdown.

**Out of scope (not implemented, not run):** training, synthetic-data
generation, NER, linking, inference, assertion classification, LLM/ML of any
kind. The L3 mention specialists are deliberately **not** run, so derived route
cases C6/C7 are not computed.

## 2. Git state before implementation

Branch `main`, clean working tree, latest commit
`43e347f feat: add policy and KB forensics foundation` (Phase 1C-A).

## 3. Corpus acquisition and placement

The organizer archive `input.zip` (84 KB, 100 `.txt` files + one directory entry)
was extracted to `data/organizer_test/input/` — the repo's **INPUT ONLY**
namespace per `data/README.md` — and the pristine archive moved to
`data/organizer_test/input.zip`. Verified: files `1.txt … 100.txt`, all present,
no gaps. Nothing under `data/organizer_test/` is trackable
(`git status --porcelain data/organizer_test/` is empty; `git check-ignore`
confirms both the `.txt` files and the archive). The corpus has **no ground
truth** and is never labeled, evaluated, or committed.

- Encoding: UTF-8, **LF only** (0 files contain CR).
- `corpus_sha256`: `de63331e70b896380f60b807290bdede67ce3a306f4ebb8aa8b9b7594a802b73`

## 4. Files created

- **Config:** `configs/corpus_analysis/analysis_v1.yaml` (cue inventories +
  descriptive thresholds; cues live in config, not Python, per convention).
- **Package:** `src/mednorm_vi/corpus_analysis/` — `models.py`, `config.py`,
  `loader.py`, `distributions.py`, `layouts.py`, `document.py`, `corpus.py`,
  `pipeline.py`, `reporting.py`, `cli.py`, `__init__.py`.
- **Fixtures:** `tests/fixtures/corpus_analysis/input/` (5 synthetic documents).
- **Tests:** `tests/unit/test_corpus_analysis.py`,
  `tests/integration/test_corpus_analysis_pipeline.py`.
- This audit.

## 5. Files modified

**None.** Phase 2A is purely additive:
`git status --short src/mednorm_vi/{document_intelligence,case_router,mention_factory,resolution}`
is empty. L1 preprocessing behavior is unchanged.

## 6. Design — how L1 is consumed, not changed

Each document is parsed with the **existing** `analyze_text` using the **existing**
`configs/document_intelligence/base.yaml` (L1 `config_hash` recorded in the
report). Statistics are read off the resulting `DocumentGraph` (`SECTION`,
`LINE`, `SENTENCE`, `LIST_ITEM`, `TABLE_ROW`, `PARAGRAPH`, `TOKEN` nodes) and off
the existing L2 canonical routable units. `source_path` is deliberately not
passed, so `document_id` stays content-derived and machine-independent.

A route case is a **processing case**, not an entity type and not a prediction;
only structural cases C1-C5 are counted.

## 7. Determinism

No timestamps, no wall-clock, no set iteration in output; every frequency table
is ordered (count desc, key asc) and every mapping is sorted. Identity comes from
`corpus_sha256` + analysis `config_hash` + L1 `config_hash`. Running the CLI
twice over the real 100-document corpus produced **byte-identical**
`report.json` and `report.md`.

- Report `determinism_hash`:
  `a117cca904904b9b754f43bd10537d4dddf5057498cf21bec8d53d629ee531e1`

## 8. Corpus findings (aggregate)

**Scale.** 100 documents, 132,336 characters (170,948 UTF-8 bytes), 2,964 lines
(2,827 non-blank), 30,470 tokens, 970 sentences. L1 validation: **100/100 OK**.

| measure | min | p25 | median | p75 | p90 | max | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| characters | 136 | 611.75 | 1222.5 | 1741.0 | 2437.0 | 4428 | 132336 |
| lines | 3 | 14.0 | 29.5 | 36.25 | 52.1 | 140 | 2964 |
| max_line_length | 38 | 109.0 | 133.0 | 206.0 | 280.0 | 743 | — |
| sections | 1 | 2.0 | 2.0 | 3.0 | 3.0 | 6 | 240 |
| list_items | 0 | 6.75 | 16.0 | 25.0 | 40.1 | 126 | 1913 |
| routable_units | 5 | 17.0 | 31.5 | 42.0 | 54.2 | 143 | 3203 |

Length histogram (chars): 1-250 → 3 docs; 251-500 → 14; 501-1000 → 25;
1001-2000 → 39; 2001-4000 → 18; 4001-8000 → 1; >8000 → 0.

**Structure.** Documents are overwhelmingly list-shaped: profile
`list_dominant` 78, `mixed` 18, `narrative_dominant` 4. 98/100 documents contain
list items and numbered items; 96/100 contain key-value rows; **0 documents
contain tab-separated (`table_like`) rows** — all 320 detected rows are
`key_value_like`.

**Routable units (3,203).** `list_item` 1913, `sentence` 970, `table_row` 320,
`line` (fallback) **0** — every line decomposed into a more specific unit.

**Processing cases (C1-C5).** C3 2609, C4 503, C1 191, C2 185, C5 6.

**Sections.** 240 sections across 100 documents; only **13 distinct heading
surface forms**, i.e. a small, highly repetitive heading vocabulary. L1 section
categories: `unknown` 96, `admission_information` 78, `laboratory` 36,
`pre_admission_medications` 12, `treatment` 6, `diagnosis` 4, `symptoms` 4,
`medical_history` 2, `current_examination` 1. 14 documents raised
`weak_section_header_confidence`.

**Laboratory layouts** (139 lab-shaped lines in 46 documents): bare numeric
results 75, qualitative results 42, normality phrases 33, numeric-with-unit 22,
**reference ranges 0**.

**Medication-list patterns** (64 med-shaped lines in 30 documents): route cues
89, strength 48, dose form 18, frequency 17; the full
strength+route+frequency pattern appears **2 times, in 1 document**.

**Imaging/report styles** (141 modality lines in 56 documents): 58
imaging-section-cue lines; the trailing `(<modality>)` parenthetical style
appears 11 times across 8 documents.

## 9. Engineering implications (observations, not decisions)

These are measurements to inform later phases; nothing was changed on their
basis in this phase.

1. **Reference-range parsing is currently unexercised** — the corpus contains
   zero reference ranges, though Phase 1B supports them.
2. **Lab values are mostly bare numerics** (75) rather than numeric+unit (22),
   and 33 lines state normality in prose. The lab parser's unit-anchored paths
   will under-fire on this corpus.
3. **No tab-delimited tables exist**; only key-value rows. `table_like` handling
   is currently dead weight on this corpus.
4. **Medication lines rarely carry a full grammar** (frequency on only 17 lines);
   most carry a route cue and often a strength. Boundary policy should not assume
   a full `name strength route frequency` span is typical here.
5. **96 sections classify as `unknown`** and 14 documents raise weak-header
   warnings, against only 13 distinct heading forms — the L1 section lexicon
   appears to under-cover a small, highly repetitive heading vocabulary. This is
   a high-leverage, low-risk lexicon gap to review.
6. **C3 dominates routing** (2609 units) — narrative handling matters more than
   structured lab/medication handling on this corpus.

## 10. Tests and exact results

```
python3 -m pytest -q   ->  460 passed in ~16s
ruff check .           ->  All checks passed!
python3 -m mypy        ->  Success: no issues found in 161 source files
git diff --check       ->  clean
```

27 new tests. Unit: numeric-not-lexical document ordering, verbatim loading,
corpus-hash determinism, distribution/percentile/histogram math, and lab /
medication / imaging / narrative / blank-line layout detection. Integration
(over synthetic fixtures): all documents analyzed and L1-valid, lengths, routing
units and cases (asserting **C6/C7 are absent**), structure, layout families,
sections/headings, determinism (`to_json`/`to_markdown`/hash), no-timestamp
guard, JSON round-trip, **scope contract** (descriptive_only, no
extraction/prediction/linking/ML), a forbidden-field scan, an **L1-unchanged**
guard (same graph before/after analysis), heading-offset preservation, and CLI
determinism + empty-corpus error. One test analyzes the real 100-document corpus
**only if present locally** and skips otherwise, so CI passes without the
git-ignored data.

## 11. CLI

```
python -m mednorm_vi.corpus_analysis.cli \
    --input-dir data/organizer_test/input \
    --output-dir reports/corpus_analysis
```

Output (real corpus):

```
documents           : 100
corpus sha256       : de63331e70b896380f60b807290bdede67ce3a306f4ebb8aa8b9b7594a802b73
L1 validation OK    : 100/100
characters (total)  : 132336
routable units      : 3203
units by kind       : list_item=1913, sentence=970, table_row=320
cases (C1-C5)       : C3=2609, C4=503, C1=191, C2=185, C5=6
profiles            : list_dominant=78, mixed=18, narrative_dominant=4
docs w/ lab layout  : 46
docs w/ med layout  : 30
docs w/ imaging     : 56
determinism hash    : a117cca904904b9b754f43bd10537d4dddf5057498cf21bec8d53d629ee531e1
```

## 12. Reports

`reports/corpus_analysis/report.json` (204 KB) and `report.md` (9 KB).
`reports/**` is git-ignored (only `README.md`/`.gitkeep` track) — correct, since
the reports summarise organizer INPUT documents and must not be committed.

## 13. No prediction / extraction / ML / network

Confirmed. `grep -rE "requests|urllib|http|socket|torch|transformers|sklearn"`
over `src/mednorm_vi/corpus_analysis` returns nothing. The mention specialists
are not imported. The report carries an explicit `scope` block
(`descriptive_only: true`, `entity_extraction: false`, `prediction: false`,
`linking: false`, `ml_or_llm: false`, `l1_behavior_modified: false`), asserted by
tests.

## 14. Data hygiene

Organizer inputs and the archive live under `data/organizer_test/` and are fully
git-ignored; the tracked tree gained no organizer content. Test fixtures are
**synthetic** and live under `tests/fixtures/`, not `data/`. This audit contains
aggregate statistics only. `.claude/` remains ignored; no `CLAUDE.md`/`AGENTS.md`
tracked; neither architecture PDF was modified or moved.

## 15. `git status --short`

```
?? configs/corpus_analysis/
?? src/mednorm_vi/corpus_analysis/
?? tests/fixtures/corpus_analysis/
?? tests/integration/test_corpus_analysis_pipeline.py
?? tests/unit/test_corpus_analysis.py
?? docs/audits/0009-phase-2a-public-corpus-analysis.md
```
(Everything uncommitted, left for human review. No existing file modified.)

## 16. Known limitations

- Layout detection is regex/shape-based and descriptive; a "lab-shaped" line is
  not a claim that the line contains a lab result.
- Cue inventories are representative, not exhaustive; recall of the layout
  families is a floor, not a measurement of true prevalence.
- Section categories/headings are whatever the current L1 lexicon produces —
  the `unknown` count measures our lexicon's coverage, not the corpus.
- C6/C7 are structurally uncomputable here by design (specialists not run).

## 17. Is Phase 2A complete?

**Yes.** All 100 public documents parse through the unchanged L1 pipeline
(100/100 valid); lengths, headings, routing-unit frequencies, list/table/
key-value distributions, laboratory layouts, medication patterns, imaging styles,
and document structure are measured; JSON + Markdown reports are byte-reproducible;
tests and static checks are green; no L1 behavior changed and no data committed.

## 18. Recommended next milestone

**Phase 2B — corpus-driven L1/L2 calibration review:** use these measurements to
review the section-heading lexicon (the 96 `unknown` sections and 14 weak-header
warnings against only 13 distinct heading forms), and re-examine lab/medication
cue coverage against the observed bare-numeric and route-heavy reality — each as
an explicit, audited change with before/after corpus statistics.
