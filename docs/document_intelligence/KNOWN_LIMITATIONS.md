# L1 Document Intelligence — Known Limitations (Phase 1A)

L1 is deliberately conservative and structural. The following are known,
intentional limitations to be addressed later (they never violate the offset
invariant or emit medical entities).

## Structural

- **Colon lines carry BOTH structures (resolved in hardening).** A labeled
  narrative line like `Chẩn đoán: Viêm phổi cộng đồng.` is represented
  *non-exclusively*: a `key_value_like` row with `key`/`value` cells **and**
  sentence-like segment(s) over the narrative value (nested in the value cell).
  Tokens reference the smallest covering segment. No medical entity is emitted.
- **Sentence segmentation is heuristic.** The abbreviation list is representative,
  not exhaustive; unusual abbreviations followed by a capitalized word may
  over-split. Segmentation is deliberately conservative to avoid losing text.
- **Subsections use indentation only.** Non-indented subsections (e.g. deeper
  headings that are not indented) are treated as siblings, not children.
- **Table cells are basic.** Only tab-split cells and single `key: value` splits
  produce cell nodes; aligned-column tables without tabs are labelled
  `table_like` without per-column cells.
- **List nesting depth is not fully inferred.** `indent` is recorded, but nested
  list depth beyond the marker's leading indentation is not resolved.

## Section detection (hardening)

- **Fuzzy header matching requires colon-terminated or upper-case evidence** — a
  merely short/isolated line is **not** sufficient. Keys ending in sentence
  punctuation (`. ! ? …`) are never headers. Exact-alias headers are still
  detected without colon/upper (e.g. a standalone `Chẩn đoán`). Hard negatives
  are pinned in `tests/fixtures/document_intelligence/section_hard_negatives_v1.txt`.

## Normalization

- **Unicode NFC/NFD alignment is per-cluster** (starter + following combining
  marks). Compositions that span multiple starters (e.g. Hangul L+V+T) are not
  composed by the per-cluster stage — irrelevant for Vietnamese clinical text but
  documented for completeness.
- **Covering is *smallest*.** A normalized span maps back to the smallest valid
  covering original span, so trailing collapsed/deleted whitespace is not
  included when it contributes nothing to the normalized content.
- **`match`/`accent` views collapse all whitespace** (including newlines) to a
  single space. This is fine for matching; section matching uses each line's own
  normalized key, not the global view.
- The **clinical abbreviation expander is intentionally not built** in Phase 1A.
  The alignment framework supports future one-to-many expansions.

## Scope (deferred to L2+ by design)

- No NER, medication/lab entity extraction, assertion classification, ICD-10 /
  RxNorm linking, Case Router classification, model inference, or organizer JSON.
- Section **priors** are stored as evidence only; **final assertions are never
  assigned** in L1 and local text may override a section prior later.

## Performance

- Alignment is O(n) per stage (no diffing), so long/repetitive documents are
  handled quickly. Section fuzzy matching runs `SequenceMatcher` per short header
  **key** against the lexicon; it is bounded by `max_header_chars`.
