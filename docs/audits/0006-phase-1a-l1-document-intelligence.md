# Audit 0006 — Phase 1A: L1 Document Intelligence

- **Date:** 2026-07-12
- **Author:** Claude (AI agent), for human review
- **Change type:** First real processing layer (L1). Deterministic, no models,
  no datasets, no KB indices, no network, no heavy ML dependencies.
- **Note:** This audit is consolidated — it describes the final Phase 1A
  implementation including the completed post-review hardening (see §26 for a
  short summary; details are integrated throughout).

## 1. Objective and scope

Implement a complete, deterministic **L1 Document Intelligence** layer that turns
an exact UTF-8 clinical document into a `DocumentGraph` while preserving
reversible mappings to the immutable `original_text`. L1 identifies **structure
only** ("possible section header", "list-like", "table-like") and never claims a
medical entity. Core invariant: `original_text[start:end] == node.text` for every
node, `[start, end)` end-exclusive over Unicode code points.

Out of scope (deferred to L2+): NER, medication/lab entity extraction, assertion
classification, ICD-10/RxNorm linking, Case Router, model/dataset/KB downloads,
external APIs, LLM calls, organizer JSON prediction.

## 2. Files created

- Code (`src/mednorm_vi/document_intelligence/`): `unicode_utils.py`, `io.py`,
  `alignment.py`, `models.py`, `normalization.py`, `lines.py`, `sections.py`,
  `sentences.py`, `lists.py`, `table_rows.py`, `tokens.py`, `builder.py`,
  `validation.py`, `serialization.py`, `cli.py` (and rewrote `__init__.py`).
- Configs: `configs/document_intelligence/base.yaml`,
  `configs/document_intelligence/section_lexicon_v1.yaml`.
- Docs: `docs/document_intelligence/{OVERVIEW,OFFSET_ALIGNMENT,SECTION_PARSING,
  SEGMENTATION,KNOWN_LIMITATIONS}.md`.
- Fixtures: `tests/fixtures/document_intelligence/synthetic_multisection.txt`,
  `tests/fixtures/document_intelligence/section_hard_negatives_v1.txt`.
- Tests: `tests/unit/test_l1_{loading,alignment,segmentation,sections,graph}.py`,
  `tests/unit/test_l1_{bom,keyvalue,alignment_properties,section_hard_negatives}.py`,
  `tests/integration/test_l1_{cli,stress}.py`.
- This audit.

## 3. Files modified

- `src/mednorm_vi/document_intelligence/__init__.py` (stub → real L1 API).
- `README.md` (Phase 1A status + L1 usage), `docs/architecture/LAYER_CONTRACTS.md`
  (L1 marked implemented), `docs/audits/README.md` (audit index).
- Hardening updated three L1 docs created in §2:
  `docs/document_intelligence/{OFFSET_ALIGNMENT,SEGMENTATION,KNOWN_LIMITATIONS}.md`.

## 4. Exact-input loading policy

`io.load_document`/`load_text` decode strict UTF-8 with **no** silent
normalization: `\n`, `\r\n`, repeated spaces, tabs, blank lines, punctuation,
combining marks, duplicated substrings, and leading/trailing whitespace are kept
verbatim. Newlines are never translated (style only recorded: LF/CRLF/MIXED/CR/
NONE). BOM policy is explicit; the **default is `preserve`** (exact text, no
offset shift). `error` fails deterministically. `strip` is explicit opt-in and
**audited**: it removes the BOM only at load time (before offsets exist), records
`DocumentSource.bom_stripped=True`, and emits a structured `bom.stripped` warning
(propagated into `DocumentGraph.warnings`) stating the working text begins after
the BOM. Invalid UTF-8, unreadable input, and BOM policy violations raise
`DocumentLoadError`. Source metadata: path, encoding, byte length, code-point
length, SHA-256, newline style, `has_bom`, `bom_policy`, `bom_stripped`.

## 5. Three text representations

1. **`original_text`** — immutable, exact, sole source of output offsets.
2. **normalized views** (`match`, `accent`) — transformed copies for matching,
   each with a reversible alignment; **never** output coordinates.
3. **token/segment graph** — structural nodes with absolute offsets.

## 6. Alignment design

`alignment.CharAlignment` stores one canonical monotonic **boundary map** `o2n`
(length N+1; `o2n[0]==0`, `o2n[-1]==normalized_length`). Covering original spans
are derived by **binary search** over `o2n`, returning the **smallest valid
covering** original span (`os` = largest original boundary at/before the
normalized start; `oe` = smallest original boundary at/after the normalized end).
This supports one-to-one, one-to-many, many-to-one, view-only insertions, and
original-only deletions. In particular, selecting a partial expansion is correct:
for `ß`→`ss`, `normalized_span_to_original(0,1)`, `(1,2)`, and `(0,2)` all recover
the covering original `ß` (`[0,1)`) — the earlier bug that returned an empty span
for the first half is fixed.

APIs: `original_span_to_normalized`, `normalized_span_to_original` (smallest
covering), `recover_original` (reads only `original_text`), `then` (compose
stages), `consistency_errors`, `to_offset_alignment` (bridge to the reused
`schemas.spans.OffsetAlignment` / `NormalizedView`). The map is built in **O(n)**
from exact per-character output counts (`from_counts`) — no diffing — so
long/repetitive inputs are fast. These properties are checked exhaustively over
every boundary and span in `tests/unit/test_l1_alignment_properties.py`.

## 7. Normalization stages

Independently toggleable, config-defined per view: `nfc`, `casefold`,
`whitespace_collapse`, `punctuation_alias`, `decimal_view`, `accent_strip`. Each
stage records name, version, input/output length, transformation count, and a
config hash. NFC/accent operate per normalization cluster (starter + following
combining marks); every stage yields exact reversible alignment. The clinical
abbreviation expander is intentionally **not** built (framework supports future
one-to-many expansion).

## 8. Section detection design

Versioned, extensible lexicon (`section_lexicon_v1.yaml`) with per-alias and
per-category metadata (semantic group, language, accent/abbreviation status,
prior label + strength, positive/negative examples). Matching runs on a
casefold + accent-stripped + whitespace-collapsed **key** (text before the first
`:`): exact-alias hits accepted directly; fuzzy hits (`SequenceMatcher` ≥
`fuzzy_threshold`) require structural evidence — **colon-terminated or upper-case**
(a merely short/isolated line is not sufficient) — bounded by `max_header_chars`.
A key ending in sentence punctuation (`. ! ? …`) is never a header, so ordinary
standalone sentences are not promoted. Exact aliases (including unaccented and
bilingual headings) are still detected without colon/upper. Subsections nest by
indentation. Sections store priors as **evidence only** — no assertions are
assigned. Hard negatives are pinned in
`tests/fixtures/document_intelligence/section_hard_negatives_v1.txt`
(e.g. `Thuốc này dùng tại nhà.` is not classified as `home_medications`).

## 9. Sentence / list / table-like segmentation

- **Lines/paragraphs:** content excludes terminators; newline delimiters kept;
  lines + newline delimiters tile `[0, len)`; paragraphs never cross a section
  start.
- **Sentences:** conservative — period not a boundary between digits or after a
  known abbreviation/initial; `;` and `\n` split; terminating punctuation kept.
- **Lists:** `1.`/`1)`/`(1)`/`a.`/`a)`/`-`/`–`/`—`/`*`/`•`; marker span and
  content span stored separately (number/bullet never folded into content).
- **Table-like/key-value rows (non-exclusive):** tab-split → `table_like`;
  `name: value` → `key_value_like`; `;`-heavy → `table_like`. A colon line is
  represented non-exclusively — all of: the full line, the `key_value_like` row,
  the `key` cell, the `:` delimiter (kept in the original text between the cells),
  the `value` cell, **and** sentence-like segment(s) over the narrative value
  (nested inside the value cell). Tokens reference the smallest covering segment
  (the value sentence over the row). No medical entities emitted.

## 10. Token-view design

`tokens.tokenize` produces word/number tokens keeping internal decimals, unit
slashes, and hyphens (`14.43`, `14,43`, `325-650`, `mg/ml`), plus one token per
punctuation mark; each token stores absolute offsets, normalized form, category,
punctuation flag, whitespace-before. Not a model tokenizer — later tokenizers map
back through this view.

## 11. DocumentGraph design

Immutable `DocumentGraph`: `document_id`, source metadata, `original_text`,
normalized views, and a deterministic node tree (document → sections → paragraphs
→ lines → sentences/list-items/table-rows → tokens; plus markers, cells, newline
delimiters). Deterministic ids (`sec-0001`, `line-0007`, `tok-000012`, …) and
ordering; node identity is id + absolute coordinates (never text-only), so
repeated surface forms at different positions stay distinct.

## 12. Validation invariants

`validation.validate_graph` detects: invalid ranges, text/offset mismatch,
duplicate node ids, broken parent containment, line-coverage overlaps/gaps, and
broken alignment maps. For normalization deletions/collapses the alignment
invariants are: the boundary map is **monotonic and in bounds** with anchored
endpoints; a normalized span maps to the **smallest valid covering** original
span; **intentionally deleted original characters need not be included** in every
normalized full-range covering span (e.g. trailing collapsed whitespace is
legitimately excluded), so the validator checks covering integrity (start at 0,
image reaches the full normalized length) rather than demanding the full original
length; and every deletion/transformation stays **recorded in provenance**
(per-stage `NormalizationStageRecord`: input/output length, transformation count,
version, config hash). Errors fail fast; weak section-header confidence is a
warning.

## 13. Configuration and lexicon format

`base.yaml` controls IO/BOM policy, normalization views, sentence/list/table
rules, section thresholds (fuzzy + structural), token options, validation
strictness, and deterministic id padding. `section_lexicon_v1.yaml` is a
versioned, extensible category→aliases inventory with rich metadata (no
exhaustive medical lexicon embedded in Python).

## 14. Tests and exact results

```
python3 -m pytest -q   ->  261 passed in ~4.4s
ruff check .           ->  All checks passed!
python3 -m mypy        ->  Success: no issues found in 74 source files
git diff --check       ->  clean
```

L1 tests cover: valid/invalid UTF-8, LF/CRLF/mixed/no-trailing-newline, tabs,
repeated blanks; **BOM policies** (default-preserve, no invisible offset shift,
strip-is-explicit-and-audited, error-fails-deterministically); alignment
identity/casefold/NFC-NFD/one-to-many/many-to-one/covering/composition/broken-
rejected/exact-recovery plus **exhaustive small-string property tests** (every
boundary/span, smallest covering, original-only recovery); sentences
(abbreviation, decimal, semicolon, narrative), lists (numbered/bulleted, marker
excluded), table/**key-value dual representation** (key/value offsets, value
sentence, containment, deterministic serialization), tokens (decimals/units/
ranges), repeated text distinct; sections (VN/unaccented/English/abbrev/colon/
uppercase/false-positive/nested/preamble/prior-recorded) plus a **hard-negative
regression fixture**; graph (substring invariant, determinism, unique ids,
containment, no-mutation, round-trip serialization, UTF-8, normalized-not-output);
stress (NFC/NFD, mixed VN-EN, spaces/tabs, CRLF, punctuation, long line,
duplicates, noisy headings, medication-like, lab-like, realistic multi-section,
empty/whitespace-only).

## 15. Positive CLI smoke-test output

```
$ python -m mednorm_vi.document_intelligence.cli \
    --input tests/fixtures/document_intelligence/synthetic_multisection.txt \
    --config configs/document_intelligence/base.yaml \
    --output /tmp/mednorm-document-graph.json
MedNorm-VI L1 Document Intelligence
input path        : tests/fixtures/document_intelligence/synthetic_multisection.txt
characters        : 416
bytes             : 506
newline style     : LF
has BOM           : False
lines             : 28
sections          : 8
paragraphs        : 8
sentences         : 16
list items        : 7
table-like rows   : 5
tokens            : 110
delimiters        : 27
normalization     : accent, match
graph validation  : OK (0 errors, 0 warnings)
structural warns  : 0
determinism hash  : a305270deaabb45f24e22c74736476dfdf1304c2512288ab3020d76e75f12b6c
output path       : /tmp/mednorm-document-graph.json
```

The `sentences: 16` count reflects the non-exclusive key-value representation
(value narratives are also emitted as sentence-like segments).

## 16. Determinism test result

Running the CLI twice on the fixture produced **byte-identical** JSON output
(`diff` empty; identical `determinism_hash`
`a305270deaabb45f24e22c74736476dfdf1304c2512288ab3020d76e75f12b6c`). A test
(`test_l1_cli.test_cli_is_deterministic`) asserts this.

## 17. Negative smoke-test results

- **Invalid UTF-8:** CLI exits `2` with `DocumentLoadError` ("L1 never silently
  repairs decoding").
- **Broken alignment:** `CharAlignment(4,3,(0,2,1,0,3)).consistency_errors()`
  → non-empty (non-monotonic `o2n` + bad endpoints).
- **Out-of-bounds span:** injecting a node `[2,99)` into a length-3 document →
  `validate_graph` reports `l1.invalid_range` (not ok).

## 18. Items intentionally deferred to L2+

NER and all medical entity extraction; assertion classification; ICD-10/RxNorm
linking; Case Router; model/dataset/KB usage. Also (see `KNOWN_LIMITATIONS.md`):
subsections use indentation only; table cells are basic; abbreviation expander
not built. (The earlier "colon lines are rows, not sentences" limitation is
**resolved** — see §9's non-exclusive representation.)

## 19. Risks and known limitations

Sentence segmentation is heuristic (representative abbreviation list); per-cluster
Unicode composition ignores multi-starter compositions (irrelevant for
Vietnamese); fuzzy section matching is deliberately conservative (colon/upper +
threshold). All documented in `KNOWN_LIMITATIONS.md`. None violate the offset
invariant or emit entities.

## 20. No models/datasets/KBs/external APIs

Confirmed. `grep` over `document_intelligence/` for
`torch|transformers|faiss|requests|urllib.request|numpy|pandas` returns nothing.
No network calls; only stdlib + `PyYAML` for config.

## 21. AI-assistant files remain ignored/untracked

`.claude/AGENTS.md` is git-ignored (`git check-ignore` confirms). No `CLAUDE.md`
or `AGENTS.md` is tracked (the only matches are the audit **filenames** 0004/0005).

## 22. Architecture PDFs not modified or moved

`git status --short -- '*.pdf'` is empty; both PDFs remain at
`docs/MedNorm-VI_Architecture.pdf` and `docs/MedNorm-VI_Architecture_Vietnamese.pdf`
(the latter still ignored). Neither was edited or relocated in this change.

## 23. `git status --short`

```
 M README.md
 M docs/architecture/LAYER_CONTRACTS.md
 M docs/audits/README.md
 M src/mednorm_vi/document_intelligence/__init__.py
?? configs/document_intelligence/
?? docs/audits/0006-phase-1a-l1-document-intelligence.md
?? docs/document_intelligence/
?? src/mednorm_vi/document_intelligence/alignment.py
?? src/mednorm_vi/document_intelligence/builder.py
?? src/mednorm_vi/document_intelligence/cli.py
?? src/mednorm_vi/document_intelligence/io.py
?? src/mednorm_vi/document_intelligence/lines.py
?? src/mednorm_vi/document_intelligence/lists.py
?? src/mednorm_vi/document_intelligence/models.py
?? src/mednorm_vi/document_intelligence/normalization.py
?? src/mednorm_vi/document_intelligence/sections.py
?? src/mednorm_vi/document_intelligence/sentences.py
?? src/mednorm_vi/document_intelligence/serialization.py
?? src/mednorm_vi/document_intelligence/table_rows.py
?? src/mednorm_vi/document_intelligence/tokens.py
?? src/mednorm_vi/document_intelligence/unicode_utils.py
?? src/mednorm_vi/document_intelligence/validation.py
?? tests/fixtures/document_intelligence/
?? tests/integration/test_l1_cli.py
?? tests/integration/test_l1_stress.py
?? tests/unit/test_l1_alignment.py
?? tests/unit/test_l1_alignment_properties.py
?? tests/unit/test_l1_bom.py
?? tests/unit/test_l1_graph.py
?? tests/unit/test_l1_keyvalue.py
?? tests/unit/test_l1_loading.py
?? tests/unit/test_l1_section_hard_negatives.py
?? tests/unit/test_l1_sections.py
?? tests/unit/test_l1_segmentation.py
```
(Everything uncommitted, left for human review. `tests/fixtures/
document_intelligence/` contains both `synthetic_multisection.txt` and
`section_hard_negatives_v1.txt`.)

## 24. Is Phase 1A complete?

**Yes.** L1 loads exactly, normalizes reversibly with O(n) covering alignment,
segments sections/sentences/lists/table-rows/tokens, builds a deterministic
validated `DocumentGraph`, serializes deterministically, and ships a CLI + docs +
comprehensive tests. pytest/ruff/mypy/`git diff --check` are green; positive,
determinism, and negative smoke tests pass; no models/data/KB/APIs; PDFs
untouched; AI-assistant files ignored.

## 25. Recommended next milestone

**Phase 1B — deterministic medical structure specialists:** medication grammar,
laboratory parser, and an initial rule-based multi-label Case Router, all reading
the L1 `DocumentGraph` and measurable against the provisional local evaluator.

## 26. Post-review hardening summary

The post-review hardening is complete and its results are **integrated into the
sections above** (this audit describes the final implementation consistently).
In brief, four items were hardened:

1. **BOM policy** — default `preserve`; `strip` is explicit opt-in with
   `bom_stripped` metadata + a `bom.stripped` warning (§4). Tests:
   `test_l1_bom.py`.
2. **Non-exclusive key-value + narrative** — a colon line yields the full line,
   row, key cell, `:` delimiter, value cell, **and** value sentence(s) (§9); the
   old "colon rows are not sentence-like" limitation is removed (§18/§19). Tests:
   `test_l1_keyvalue.py`.
3. **Alignment** — one canonical `o2n` map; smallest-valid-covering via binary
   search; the `ß`→`ss` partial-expansion bug is fixed (§6); validation wording
   updated for deletions/collapses (§12). Exhaustive property tests:
   `test_l1_alignment_properties.py`.
4. **Section fuzzy hard negatives** — colon/upper evidence + sentence-punctuation
   guard (§8); fixture `section_hard_negatives_v1.txt`; tests
   `test_l1_section_hard_negatives.py`.

Final verified validation (also in §14/§15/§16): `pytest -q` → **261 passed**;
`ruff check .` → clean; `mypy` → clean (74 files); `git diff --check` → clean;
CLI run twice → byte-identical output, determinism hash
`a305270deaabb45f24e22c74736476dfdf1304c2512288ab3020d76e75f12b6c`. No new audit
was created (this consolidation is recorded here per instruction).
