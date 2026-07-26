# Audit 0027 — S1 Character-Level Alignment and a Single Segmentation Policy

- **Date:** 2026-07-26
- **Author:** Claude (AI agent), for human review
- **Change type:** Correctness fix in the slow-tokenizer alignment backend plus one
  segmentation policy covering raw and pre-segmented sources. **No commit. No push. No local
  training, no local GPU, no model/tokenizer/VnCoreNLP downloads, no organizer inference, no
  `output.zip`, no backbone change. Governed corpus files NOT modified.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — §4 (offset preservation and the
  `original_text[start:end] == entity["text"]` invariant; "never trim whitespace or rewrite
  punctuation before storing a span"), §15 (S1 objective). The PDF was not modified or moved.
- **Status:** `IMPLEMENTED_UNEXECUTED` — requires the **v3 Colab smoke rerun**.

Audit 0026 is committed (`ca1e473`) and is not modified. The next immutable number is 0027.

## 1. The two real failures

Real smoke-v2 diagnostics: 12 considered, 10 aligned, **2 unalignable**, counters reconciled,
boundary-merge counters both zero — so neither failure reached the boundary-merge policy.

| # | Handle | Source | Split | Stage | Reason code |
| --- | --- | --- | --- | --- | --- |
| 1 | `6c3b519cd0100bec` | `vimq` | validation | `word_mapping` | `EMPTY_SEGMENTED_WORD` |
| 2 | `c06995a162f6eb76` | `vimedner` | train | `word_mapping` | `SEGMENTED_SYLLABLE_NOT_FOUND` |

Both were reproduced **exactly** from the governed examples' structure before any code was
changed, so neither root cause is a guess.

### Failure 1 — the segmenter emits a standalone join character

The ViMQ example is already word-segmented (the join character sits between word characters).
Feeding it back through RDRSegmenter makes the segmenter treat the join character as
punctuation and split it off as **its own token**. `map_segmented_words()` then computed
`"_".split("_")` → `["", ""]` → filtered to `[]` → `EMPTY_SEGMENTED_WORD`.

Worse than the crash: this also **shreds the word**. A two-syllable compound that the corpus
stores as one word becomes three model tokens, destroying exactly the word-level input
ViHealthBERT-Word was pretrained on.

### Failure 2 — the segmenter regrouped a spaced compound

The ViMedNER example contains a hyphen compound written with spaces around the hyphen
(`… <syllable> <syllable> - <syllable> <syllable> …`). RDRSegmenter pulls such compounds
together, emitting a single token whose surface form contains the hyphen with **no** spaces.
`map_segmented_words()` required every emitted syllable to appear **verbatim** in the original
text; the regrouped syllable never does, so `str.find` failed →
`SEGMENTED_SYLLABLE_NOT_FOUND`.

## 2. Was either annotation invalid?

**No.** Both examples satisfy the §4 invariant `original_text[start:end] == entity["text"]`,
carry in-range monotonic offsets, and use valid entity types. Nothing about either annotation
is malformed or inherently ambiguous.

Both failures were **implementation defects in the alignment backend**, which assumed a word
segmenter preserves surface forms verbatim. It does not — regrouping text is precisely what a
segmenter is for. `configs/training/s1_governed_exclusions.yaml` therefore **remains empty**,
and no governed corpus file was touched.

## 3. The general correction

### (a) Character-level monotonic alignment

`map_segmented_words()` no longer searches for syllables. It walks the segmented output and
the original text together with a strictly monotonic cursor and matches them **character by
character**, skipping *separators* (whitespace and the join character) on either side. Each
word's span runs from its first to its last matched original character.

This is agnostic to *how* the segmenter regrouped the text — merges, splits, moved whitespace,
standalone join characters — while remaining strict about what matters: **a real character may
never be changed, added, or dropped.** Any such divergence raises the new reason code
`SEGMENTER_ALTERED_CHARACTERS`.

A token consisting only of separators (failure 1) carries no characters of its own, so it
contributes no word and no label rather than being an error. If a segmentation yields no
mappable words at all, `EMPTY_SEGMENTED_WORD` still fires.

`ALIGNMENT_BACKEND` is bumped to `phobert-slow-char-alignment-v3`.

### (b) One segmentation policy for both source kinds

```text
text already in RDRSegmenter form  ->  use it verbatim (no re-segmentation)
everything else                    ->  VnCoreNLP RDRSegmenter
```

`looks_pre_segmented()` detects the structural signature — the join character directly between
two word characters — and `resolve_segmented_text()` applies the rule. Detection is purely
structural: no dataset names, no per-corpus table, no example ids anywhere in the backend
(enforced by a test that strips comments and string literals before checking).

This is what keeps ViMQ/PhoNER words intact instead of shredded, and it avoids ~18k pointless
segmenter calls in full training.

**Both fixes are independent and both are needed:** (b) prevents the shredding, (a) makes the
mapping correct for whatever the segmenter actually returns — including the shredded form, so
the pipeline is correct even if detection ever misses.

## 4. Final segmentation/alignment policy

| Situation | Behaviour |
| --- | --- |
| Raw clinical text | RDRSegmenter (production path, asserted, never degraded) |
| Already pre-segmented source | used verbatim; words preserved |
| Literal join character in the source | a separator: skipped when matching, included inside a word's span |
| Standalone join token from the segmenter | carries no characters; contributes no word |
| Segmenter merged/split across whitespace or punctuation | mapped; span covers the original spacing |
| Segmenter changed/added/dropped a real character | **fails** with `SEGMENTER_ALTERED_CHARACTERS` |
| A word straddling a gold entity boundary | unchanged: masked and counted (Audit 0026) |
| Truncation cutting an entity | unchanged: `PARTIAL_TRUNCATION_POLICY` |

Offsets remain half-open `[start, end)` into the untouched `original_text`; no token ever
receives an incompatible label.

## 5. Test evidence for both real examples

Both patterns are covered by tests written from the examples' **shape**, never their text and
never their ids:

- `test_standalone_join_token_is_mapped_not_treated_as_empty` — failure 1's exact structure;
  the shredded segmentation maps and every span reproduces the original substring;
- `test_segmenter_compound_merge_across_spaces_is_mapped` — failure 2's exact structure; the
  merged compound maps back onto the original spacing;
- `test_both_real_failure_shapes_encode_end_to_end` — both shapes now produce labelled
  features, with no masked token retaining a label;
- `test_pre_segmented_detection_is_structural` (7 cases), `test_pre_segmented_text_is_used_verbatim_and_raw_text_is_segmented`;
- `test_offsets_are_monotonic_and_reconstruct_the_original` (6 regroupings) — spans are
  ordered, non-overlapping, in range, and cover every non-separator character exactly once;
- `test_a_genuinely_unmappable_word_still_raises_with_a_reason_code` and
  `test_a_dropped_punctuation_character_is_still_a_failure` — real drift still fails;
- `test_segmentation_that_produces_only_separators_fails`;
- `test_structural_mismatch_diagnostics_carry_no_characters` — the message gives the offset and
  the Unicode *category*, never the character;
- `test_alignment_contains_no_example_specific_special_cases` — parses the backend, strips
  comments and string literals, and asserts no dataset name or privacy-safe id reaches
  executable code.

Locally verified against the two real governed examples (debugging only, nothing written to
the repo): failure 1 is detected as pre-segmented and maps to 14 words with exact spans;
failure 2 goes through the segmenter and its merged compound maps to the exact original span.

## 6. Readiness and counter semantics — unchanged

`unalignable_example_count` still counts **unexpected** failures only and still blocks
`full_training_readiness`; `governed_exclusion_count` still counts tracked decisions and still
does not. The reconciliation invariant is unchanged:

```text
aligned + unalignable + governed_exclusions == examples_considered
tokenizer_equivalence_considered           == examples_considered
```

`NON_SEPARATOR_GAP_INSIDE_WORD` is retained as a declared reason code so v2 manifests remain
readable, but the character-level aligner no longer produces it. Two diagnostics are added to
the manifest: `pre_segmented_example_count` and `segmenter_example_count`, plus
`segmentation_policy`.

The validator is **not** weakened: no threshold was relaxed and no nonzero failure count is
permitted.

## 7. Artifact lifecycle

```text
v1  artifacts/s1_mention_first_run_smoke      historical; full_training_readiness: false
v2  artifacts/s1_mention_first_run_smoke_v2   historical; 2 unalignable examples
v3  artifacts/s1_mention_first_run_smoke_v3   the corrected rerun target
```

The tracked smoke config now records `artifact_version: v3` plus every previous artifact with
its version and status. `SmokeArtifactPaths.validate()` rejects an `artifact_dir` equal to
**any** historical one, and the smoke notebook asserts the same at runtime. v1 and v2 are never
written to and never edited.

The validation and full-training notebooks default to v3 through the existing runtime inputs
(`MEDNORM_SMOKE_ARTIFACT_VERSION`, `MEDNORM_SMOKE_ARTIFACT_DIR`) and still take the expected
SHA-256 from the operator (`MEDNORM_EXPECTED_SMOKE_CHECKPOINT_SHA256`, empty by default). **No
source edit is needed to validate v3** — the three-way hash contract (recomputed == manifest ==
operator-supplied) is unchanged. The validation notebook warns when pointed at v1 or v2; the
full-training notebook refuses both.

## 8. Validation

```text
env PYTHONPATH=src python3 -m pytest -q          -> 986 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 224 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
S1 notebook code-cell AST parse                   -> 4 notebooks, 0 syntax errors
```

Preserved and still passing: two-pass Colab bootstrap, scoped dependency health, NumPy
`RandomState` + dummy AdamW ABI checks, ViHealthBERT slow `PhobertTokenizer`, tokenizer
equivalence, strict offset invariants, boundary-merge and partial-truncation masking, VietMed
span/type-only supervision, governed corpus hashes and split policy, privacy-safe diagnostics,
checkpoint save/reload, and the artifact-validator three-way SHA contract.

## 9. Exact changed files

**Added (1):**
```text
docs/audits/0027-s1-character-level-alignment-and-segmentation-policy.md
```

**Modified (12):**
```text
configs/training/s1_mention_first_run_smoke.yaml     (v3 output contract, v1+v2 history)
docs/audits/README.md                                (0027 index entry)
docs/notebooks/notebook_execution_integrity.md       (S1 smoke status note)
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb    (segmentation policy, v3 output, source counters)
notebooks/MedNorm_S1_Mention_Full_Training.ipynb     (segmentation policy, v3 inputs, source counters)
notebooks/MedNorm_S1_Smoke_Artifact_Validation.ipynb (v3 default, warns on v1/v2)
src/mednorm_vi/training/phobert_alignment.py         (character-level alignment, segmentation policy)
src/mednorm_vi/training/s1_mention_smoke.py          (multi-version artifact paths)
tests/unit/test_notebook_placeholders.py             (v3 + multi-history assertions)
tests/unit/test_notebooks.py                         (versioned output path)
tests/unit/test_s1_alignment_diagnostics.py          (+16 Audit 0027 tests)
tests/unit/test_s1_artifact_validation.py            (v1/v2/v3 lifecycle)
```

Counted precisely: **1 added, 12 modified — 13 changed tracked files.**

## 10. Remaining risks

1. **Nothing has been rerun.** No v3 artifact exists; no claim is made that the smoke now
   validates.
2. The character-level aligner is strict about real characters. If VnCoreNLP normalizes a
   character on some other example (a quote or ellipsis form, say), that example fails loudly
   with `SEGMENTER_ALTERED_CHARACTERS` and a structural offset — visible and diagnosable rather
   than silent, but it would need its own decision.
3. `looks_pre_segmented()` treats any word-character/`_`/word-character sequence as
   segmentation. A raw text containing such a token would skip the segmenter; offsets stay
   exact, but that example's words would be whitespace-delimited. Measured prevalence outside
   the pre-segmented sources is 1 train example in 5,796 for ViMedNER and 0 for VietMed.

## 11. Honest status

`IMPLEMENTED_UNEXECUTED`. The corrections are unit-tested and the notebooks are statically
verified, but **the v3 rerun has not been executed**. Full S1 training has not run.
