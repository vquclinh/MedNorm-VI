# Audit 0028 — S1 Vietnamese Tone-Placement Equivalence

- **Date:** 2026-07-26
- **Author:** Claude (AI agent), for human review
- **Change type:** Offset-safe orthographic equivalence in the slow-tokenizer alignment
  backend, plus a dropped-character check. **No commit. No push. No local training, no local
  GPU, no model/tokenizer/VnCoreNLP downloads, no organizer inference, no `output.zip`, no
  backbone change. Governed corpus files NOT modified.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — §4 (offset preservation, the
  `original_text[start:end] == entity["text"]` invariant, and "perform Unicode normalization on
  a copy and preserve alignment between original and normalized code points"), §15 (S1
  objective). The PDF was not modified or moved.
- **Status:** `IMPLEMENTED_UNEXECUTED` — requires the **v4 Colab smoke rerun**.

Audit 0027 is committed (`aa9ef73`) and is not modified. The next immutable number is 0028.

## 1. Exact privacy-safe structural transformation

Real v3 diagnostics: 12 considered, 11 aligned, **1 unalignable**, equivalence failures 0,
counters reconciled. The remaining failure was `c06995a162f6eb76` (`vimedner`, train,
`word_mapping`, `SEGMENTER_ALTERED_CHARACTERS`).

**RDRSegmenter rewrites Vietnamese tone-mark placement.** In an `oa` / `oe` / `uy` vowel
cluster the tone mark may sit on either vowel. Both spellings denote the identical word; the
segmenter normalizes to the traditional placement (tone on the second vowel).

For the failing example the divergence is exactly one cluster:

```text
original  : U+00F3 (Ll) + U+0061 (Ll)      tone on the first vowel
segmented : U+006F (Ll) + U+00E1 (Ll)      tone on the second vowel
length    : 2 code points on each side, order preserved, same base letters
```

### Character census of the failing example (structural only)

| Property | Value |
| --- | --- |
| Length | 395 code points |
| NFC-normalized | yes (`NFC == NFKC == original`) |
| Unicode categories | `Ll` 289, `Zs` 97, `Po` 6, `Ps` 1, `Pe` 1, `Pd` 1 |
| Non-letter code points | `U+002C U+002E U+003A U+0028 U+0029 U+002D` (plain ASCII) |
| Punctuation already space-separated | yes (9 standalone punctuation tokens, 0 attached) |
| Double spaces / edge whitespace | none |
| Tone clusters, traditional placement | 4 |
| Tone clusters, alternative placement | **1** |

Because the text is already NFC/NFKC and every non-letter code point is plain ASCII, a
tone-placement rewrite is the only character-level change available to the segmenter.

### Proof that RDRSegmenter performs this normalization

Two governed sources ship text that **is** RDRSegmenter output (they are pre-segmented). Their
tone-cluster distribution is a direct sample of the segmenter's own convention:

| Source | Kind | Traditional | Alternative |
| --- | --- | --- | --- |
| `phoner_covid19` | RDRSegmenter output | 3,204 | **0** |
| `vimq` | RDRSegmenter output | 1,496 | **0** |
| `vimedner` | raw source text | 4,617 | 1,331 |
| `vietmed_ner` | raw source text | 2,474 | 796 |

**4,700 clusters of segmenter output contain zero alternative-placement forms**, while raw text
runs roughly 23% alternative. The segmenter normalizes.

This also predicts the observed v3 outcome exactly. Across the 12 smoke examples, the count of
alternative-placement clusters is `1` for the failing example and `0` for all eleven that
aligned — a perfect discriminator. One aligned example contains a *traditional* cluster, which
correctly needs no rewrite.

**Corpus-wide impact:** 1,518 supervised train examples (6.4% of the 23,799 supervised),
129 validation and 133 internal-test examples would have failed alignment in full S1 training.

## 2. Is the annotation valid?

**Valid.** The example satisfies `original_text[start:end] == entity["text"]` for all 17
entities, offsets are in range and monotonic, entity types are legal, and the text is
well-formed NFC with plain ASCII punctuation. Both tone placements are correct Vietnamese
orthography.

This was an **implementation gap**, not bad data: the aligner treated a documented orthographic
equivalence as character corruption. `configs/training/s1_governed_exclusions.yaml` therefore
**remains empty**, and no governed corpus file was touched.

## 3. Smallest general correction

A tracked equivalence table of the 15 tone-placement pairs, consulted only when a character
comparison has already failed:

```text
oà↔òa  oá↔óa  oả↔ỏa  oã↔õa  oạ↔ọa
oè↔òe  oé↔óe  oẻ↔ỏe  oẽ↔õe  oẹ↔ọe
uỳ↔ùy  uý↔úy  uỷ↔ủy  uỹ↔ũy  uỵ↔ụy
```

Both directions and both letter cases are accepted. This is **not** fuzzy matching: only these
listed two-code-point clusters are equivalent, and every other difference still fails with
`SEGMENTER_ALTERED_CHARACTERS`. No dataset name, privacy-safe id, or sentence appears anywhere
in the backend (enforced by an existing test that strips comments and string literals).

Two supporting changes:

- **Enriched mismatch diagnostic.** The message now reports both Unicode categories and
  `same_base_letter`, which separates a diacritic-level rewrite from a genuinely different
  letter. The characters themselves are still never reported.
- **Dropped-character check.** Non-separator characters left unconsumed in the original after
  the last word are now a failure. Previously a character the segmenter dropped from the *tail*
  of an example was silently ignored, which would have stripped supervision without any
  counter moving. Found while writing the regression tests for this milestone.

`ALIGNMENT_BACKEND` is bumped to `phobert-slow-char-alignment-v4`.

## 4. Proof that offsets remain exact

Each side of every pair is **exactly two code points**, so the original cursor and the
segmented cursor advance by the same amount and the alignment stays synchronized. The word span
is still `[first matched original offset, last matched original offset + 1)` — an index into
**untouched** `original_text`, never into the segmenter's spelling.

Verified against the real failing example locally (debugging only; nothing written to the
repo): it now maps to 94 words, spans are monotonic and in range, and the concatenation of
`original_text[start:end]` over all words reproduces the original character sequence exactly
(separators aside).

Tests assert the same invariants directly: `test_tone_placement_rewrite_aligns_with_exact_original_offsets`
checks literal start offsets and that a span slices the **original** spelling rather than the
segmenter's; `test_entity_offsets_survive_a_tone_placement_rewrite` checks the §4 invariant end
to end through `encode_mention_example_slow`; and
`test_offsets_remain_monotonic_and_exact_under_tone_rewrites` checks ordering and exactness
across a mixed sequence.

## 5. Tests and validation

Ten new tests, written from the **transformation**, never from the example's text or id:

- `test_the_equivalence_table_is_a_bijection_over_two_code_points` — 15 pairs, no duplicates on
  either side, sides disjoint, equal length, and identical letters once the tone mark is
  stripped (so each pair really is the same word);
- `test_equivalence_is_accepted_in_both_directions_and_both_cases`;
- `test_tone_placement_rewrite_aligns_with_exact_original_offsets`;
- `test_entity_offsets_survive_a_tone_placement_rewrite`;
- `test_tone_rewrite_combined_with_a_compound_merge_aligns` — both segmenter behaviours at once,
  the real example's shape;
- `test_a_genuinely_different_letter_still_fails`;
- `test_non_equivalent_diacritic_changes_still_fail` — a different tone, a dropped character, an
  inserted character, and reordered characters all still fail;
- `test_mismatch_diagnostic_reports_the_transformation_class_not_the_text`;
- `test_offsets_remain_monotonic_and_exact_under_tone_rewrites`.

```text
env PYTHONPATH=src python3 -m pytest -q          -> 998 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 224 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
S1 notebook code-cell AST parse                   -> 4 notebooks, 0 syntax errors
```

Preserved and still passing: character-level monotonic alignment, the pre-segmented-text
policy, ViHealthBERT slow `PhobertTokenizer`, VnCoreNLP production segmentation, tokenizer
equivalence, boundary-merge and truncation masking, privacy-safe diagnostics, two-pass
bootstrap and scoped dependency health, governed corpus hashes and split policy, and the
artifact three-way SHA contract. No validator threshold was relaxed and no nonzero unexpected
failure count is permitted.

## 6. Artifact lifecycle

```text
v1  artifacts/s1_mention_first_run_smoke      historical; full_training_readiness: false
v2  artifacts/s1_mention_first_run_smoke_v2   historical; 2 unalignable examples
v3  artifacts/s1_mention_first_run_smoke_v3   historical; 1 unalignable example
v4  artifacts/s1_mention_first_run_smoke_v4   the corrected rerun target
```

The tracked smoke config records `artifact_version: v4` and all three historical artifacts with
their versions and statuses. `SmokeArtifactPaths.validate()` rejects an `artifact_dir` equal to
**any** of them, and the smoke notebook asserts the same at runtime. v1–v3 are never written to
and never edited.

The validation and full-training notebooks default to v4 through the existing runtime inputs
(`MEDNORM_SMOKE_ARTIFACT_VERSION`, `MEDNORM_SMOKE_ARTIFACT_DIR`) and still take the expected
SHA-256 from the operator (`MEDNORM_EXPECTED_SMOKE_CHECKPOINT_SHA256`, empty by default). **No
run hash is hardcoded anywhere and no source edit is needed to validate v4.** The validation
notebook warns when pointed at v1–v3; the full-training notebook refuses all three.

## 7. Exact changed files

**Added (1):**
```text
docs/audits/0028-s1-tone-placement-equivalence.md
```

**Modified (10):**
```text
configs/training/s1_mention_first_run_smoke.yaml     (v4 output contract, v1-v3 history)
docs/audits/README.md                                (0028 index entry)
docs/notebooks/notebook_execution_integrity.md       (S1 smoke status note)
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb    (v4 output, v3 added to history)
notebooks/MedNorm_S1_Mention_Full_Training.ipynb     (v4 default, refuses v1-v3)
notebooks/MedNorm_S1_Smoke_Artifact_Validation.ipynb (v4 default, warns on v1-v3)
src/mednorm_vi/training/phobert_alignment.py         (tone equivalence, dropped-character check, richer diagnostic)
tests/unit/test_notebook_placeholders.py             (v4 assertion)
tests/unit/test_s1_alignment_diagnostics.py          (+10 Audit 0028 tests)
tests/unit/test_s1_artifact_validation.py            (v1-v4 lifecycle)
```

Counted precisely: **1 added, 10 modified — 11 changed tracked files.**

## 8. Remaining risks

1. **Nothing has been rerun.** No v4 artifact exists; no claim is made that the smoke now
   validates.
2. The equivalence covers the 15 documented tone-placement pairs. If RDRSegmenter performs some
   other character rewrite on an example outside the smoke subset, that example fails loudly
   with `SEGMENTER_ALTERED_CHARACTERS` plus the Unicode categories, `same_base_letter`, and the
   original offset — visible and diagnosable, never silent.
3. The dropped-character check is new and strictly stronger; it could surface previously hidden
   tail-truncation behaviour on other examples. That would be a real defect being revealed, not
   introduced.

## 9. Honest status

`IMPLEMENTED_UNEXECUTED`. The correction is unit-tested and the notebooks are statically
verified, but **the v4 rerun has not been executed**. Full S1 training has not run.
