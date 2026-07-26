# Audit 0029 — S1 Orthographic Equivalence and the Full-Corpus Alignment Preflight

- **Date:** 2026-07-26
- **Author:** Claude (AI agent), for human review
- **Change type:** Offset-correctness fix in the slow-tokenizer alignment backend, a general
  Vietnamese orthographic-equivalence rule replacing the fixed pair table, and a whole-corpus
  alignment preflight that gates full training. **No commit. No push. No local training, no
  local GPU, no model/tokenizer/VnCoreNLP downloads, no organizer inference, no `output.zip`,
  no backbone change. Governed corpus files NOT modified.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — §4 (offset preservation, the
  `original_text[start:end] == entity["text"]` invariant, and "perform Unicode normalization on
  a **copy** and preserve alignment between original and normalized code points"), §15 (S1
  objective). The PDF was not modified or moved.
- **Status:** `IMPLEMENTED_UNEXECUTED` — requires the Colab full-corpus preflight, then the v5
  smoke rerun.

Audit 0028 is committed (`cbda043`) and is not modified. The next immutable number is 0029.

## 1. The four failures: exact privacy-safe transformation

All four are **one** transformation: the governed source stores a partially **decomposed**
sequence (a base letter followed by a standalone combining mark), and the segmenter emits the
**precomposed** character.

| Handle | Split idx | Offset | Original code points | Segmented | Category | NFD base |
| --- | --- | --- | --- | --- | --- | --- |
| `4af7df7adddccd7c` | 17 | 97 | `U+01A1` + `U+0309` (Ll + Mn) | `U+1EDF` (Ll) | Ll/Ll | `U+006F` |
| `d62adbe8e4933b74` | 33 | 36 | `U+0069` + `U+0303` (Ll + Mn) | `U+0129` (Ll) | Ll/Ll | `U+0069` |
| `ac60fe9c3cfdbf68` | 43 | 34 | `U+0069` + `U+0300` (Ll + Mn) | `U+00EC` (Ll) | Ll/Ll | `U+0069` |
| `b824940635ad6eaf` | 60 | 42 | `U+00E2` + `U+0300` (Ll + Mn) | `U+1EA7` (Ll) | Ll/Ll | `U+0061` |

Each is `2 code points → 1`, canonically equivalent (identical NFD), same base letter — which
is exactly the reported `same_base_letter: true`, `Ll/Ll` signature. The combining marks are
`COMBINING HOOK ABOVE`, `COMBINING TILDE` and `COMBINING GRAVE ACCENT`; the non-tone marks
(`COMBINING HORN`, `COMBINING CIRCUMFLEX ACCENT`) stay on their own base letter throughout.

**Reproduced locally with no VnCoreNLP at all.** Feeding the untouched original text back as
its own segmentation already raised the identical diagnostic
(`word 21 character 0 does not match original offset 97 (categories Ll/Ll, same_base_letter=True)`),
because the aligner normalized one side and not the other.

### A second, silent defect found while investigating

`map_segmented_words()` began with `text = unicodedata.normalize("NFC", original_text)` and then
computed every span against that **normalized copy**. For non-NFC sources this shifts every
offset after a composed mark, so `original_text[start:end]` silently stops matching the entity
text — a direct §4 violation that no counter would ever have reported. Measured shifts on the
four examples: 1, 1, 2 and 4 code points, with entities located after the first mark in two of
them.

**Prevalence:** 245 of 25,889 supervised examples (0.95%) are non-NFC — 180 train, 33
validation, 32 internal-test — all from one source.

## 2. The generalized rule

The 15-pair table is **removed**. Equivalence is now decided structurally over NFD
decompositions, so it covers every cluster in the language rather than an enumerated subset.

A cluster is accepted only when **all** of these hold:

1. the **base-letter sequence** is identical, in the same order;
2. each base letter carries the **identical non-tone Vietnamese marks** (horn `U+031B`, breve
   `U+0306`, circumflex `U+0302`);
3. the **multiset of tone marks** over the cluster is identical (grave `U+0300`, acute
   `U+0301`, tilde `U+0303`, hook `U+0309`, dot-below `U+0323`);
4. if the tone sits on a **different** base letter, every base letter in the cluster is a
   Vietnamese vowel (`a e i o u y`) — a tone may only move between vowels;
5. clusters are at most three vowels, matching Vietnamese orthography (`uyê`, `oai`, `ươi`).

Two transformation classes result, and both are reported in diagnostics:

- `canonical_composition` — same letters, same marks, same carrier; the sides differ only in
  composed/decomposed spelling. Unicode guarantees these denote identical text.
- `tone_placement` — the tone moved to another vowel of the same cluster.

Two supporting changes:

- **The original is matched verbatim.** The NFC normalization of `original_text` is gone, so
  offsets index untouched source by construction.
- **Cluster-aware matching.** The plain character fast-path is skipped when either side has a
  following combining mark; otherwise a base letter matches while its standalone mark is left
  stranded. The full-corpus preflight caught exactly this in 3 further examples where a tone
  **both** moved and composed in the same cluster.

## 3. Why it cannot accept a different tone or base letter

- **Different tone** → the tone multiset differs (condition 3) at every cluster width. Rejected.
- **Different base letter** → the base sequence differs (condition 1). Rejected.
- **Reordered base letters** → the base sequence order differs (condition 1). Rejected.
- **Insertion / deletion** → the base sequences have different content, and any residue fails at
  the next comparison or at the trailing dropped-character check. Rejected.
- **Tone added or removed** → the tone multiset differs. Rejected.
- **Non-tone diacritic moved** → the per-letter non-tone mark sets differ (condition 2).
  Rejected.
- **Arbitrary Unicode normalization** → not accepted as a blanket operation; only canonical
  equivalence of a single cluster, decided by the four structural conditions above.

Verified through the real aligner:

| Change | Result |
| --- | --- |
| different tone | `SEGMENTER_ALTERED_CHARACTERS` |
| different base letter | `SEGMENTER_ALTERED_CHARACTERS` |
| reordered base letters | `SEGMENTER_ALTERED_CHARACTERS` |
| inserted character | `SEGMENTED_SYLLABLE_NOT_FOUND` |
| dropped character | `SEGMENTER_ALTERED_CHARACTERS` |
| tone inserted / removed | `SEGMENTER_ALTERED_CHARACTERS` |
| non-tone quality mark moved | `SEGMENTER_ALTERED_CHARACTERS` |
| dropped trailing word | `SEGMENTER_ALTERED_CHARACTERS` |

The rule still accepts all 30 directed pairs the old table enumerated, plus clusters it could
not express (three-vowel clusters, and composed/decomposed spellings of any of them).

## 4. Full supervised train + validation preflight

`run_alignment_preflight()` runs segmentation, character alignment, mention encoding and
tokenizer equivalence over every supervised example, with **no model forward or backward**, and
returns both the report and the encoded features so the corpus is not encoded twice.

Executed locally over the complete supervised corpus, with the segmenter's two proven
transformations modelled (canonical composition and traditional tone placement) and a
whitespace stand-in tokenizer — **no VnCoreNLP, no model, no download**:

```text
alignment_preflight_passed       True
counters_reconciled              True
examples_considered              24844
aligned_example_count            24844
unalignable_example_count            0
governed_exclusion_count             0
equivalence_checked_count        24844
equivalence_failure_count            0
reason_code_counts                  {}
stage_counts                        {}
```

Offset exactness over the same 24,844 examples: **566,461** segmented words mapped, 0
non-monotonic examples, 0 out-of-range spans, 0 examples with incomplete original coverage, and
**13,711 / 13,711** entities satisfying `original_text[start:end] == entity["text"]` against
**untouched** original text.

This is a local model of the segmenter, not the segmenter itself; the authoritative run is the
Colab preflight, which is now a hard gate before model construction.

**The gate immediately proved its worth.** The first full-corpus run surfaced 3 failures that
no 12- or 64-example sample had reached (`0edca39339b893da`, `74f99f574679d0df`,
`ffac288546d60284`, all one source, `SEGMENTER_ALTERED_CHARACTERS` at
`subtoken_encoding`) — the stranded-combining-mark case fixed in §2.

## 5. Counters

| Split | Considered | Aligned | Unalignable | Governed exclusions |
| --- | --- | --- | --- | --- |
| train | 23,799 | 23,799 | 0 | 0 |
| validation | 1,045 | 1,045 | 0 | 0 |

| Source | Considered | Aligned | Unalignable | Governed exclusions |
| --- | --- | --- | --- | --- |
| `vietmed_ner` | 9,267 | 9,267 | 0 | 0 |
| `vimedner` | 6,702 | 6,702 | 0 | 0 |
| `vimq` | 8,875 | 8,875 | 0 | 0 |

Reason codes: none. Stages: none. Reconciliation
`aligned + unalignable + governed_exclusions == examples_considered` holds for every split and
source. `configs/training/s1_governed_exclusions.yaml` **remains empty**: every failure was an
implementation defect, and no governed annotation was found invalid.

## 6. Tests and static validation

`tests/unit/test_s1_alignment_preflight.py` (11, new): clean corpus passes and reconciles;
encoding failures block and are recorded by stage and reason code; equivalence failures are
counted separately and block; governed exclusions are separate and do **not** block;
unsupervised examples are never considered; per-split and per-source counters sum to the total;
diagnostics carry only the six permitted fields and leak no text or verbatim id; `passed`
requires zero unexpected failures; the report serializes every counter.

`tests/unit/test_s1_alignment_diagnostics.py` (Audit 0029 section): each of the four real
failures reproduced **from its code points alone** (parametrized over base letter and combining
mark name); composition accepted in both directions; offsets index untouched original text when
the source is decomposed; the general rule covers all 30 directed pairs of the retired table
(kept only as a coverage oracle) plus a three-vowel cluster; a decomposed source whose tone also
moves; nine unsupported changes still fail; a tone may not move onto a consonant; identical tone
multiset and identical non-tone marks are both required.

`tests/unit/test_notebook_placeholders.py` (+2): the full-training notebook gates on the
full-corpus preflight, covers **both** supervised splits (asserting no `[:64]` sample remains),
and the gate precedes model acquisition, the optimizer and any backward pass; the manifest
records the preflight report.

```text
env PYTHONPATH=src python3 -m pytest -q          -> 1034 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 224 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
S1 notebook code-cell AST parse                   -> 4 notebooks, 0 syntax errors
```

Preserved: character-level monotonic original-offset alignment, the pre-segmented-source policy,
the VnCoreNLP production path, the ViHealthBERT slow tokenizer, boundary-merge and truncation
masking, governed corpus files and hashes, privacy-safe diagnostics, three-way checkpoint SHA
validation, the pinned model revision, and the two-pass bootstrap with scoped dependency health.

## 7. Artifact lifecycle

```text
v1  s1_mention_first_run_smoke      historical; full_training_readiness: false
v2  s1_mention_first_run_smoke_v2   historical; 2 unalignable examples
v3  s1_mention_first_run_smoke_v3   historical; 1 unalignable example
v4  s1_mention_first_run_smoke_v4   historical; VALIDATED, but predates the full-corpus preflight
v5  s1_mention_first_run_smoke_v5   the corrected rerun target
```

v4 is real execution evidence and is **not** edited or overwritten — the alignment backend
changed underneath it, so it no longer reflects current behaviour. `SmokeArtifactPaths.validate()`
rejects an `artifact_dir` equal to any of v1–v4, and the smoke notebook asserts the same at
runtime. The validation and full-training notebooks default to v5 through the existing runtime
inputs, and the expected checkpoint SHA-256 is still operator-supplied — **no run hash is
hardcoded and no source edit is needed to validate v5.**

Ordering is deliberate: the **full-corpus preflight must pass on Colab before** the v5 smoke is
run, so v5 is only ever produced by a backend the whole corpus has already exercised.

## 8. Exact changed files

**Added (2):**
```text
docs/audits/0029-s1-orthographic-equivalence-and-full-corpus-preflight.md
tests/unit/test_s1_alignment_preflight.py
```

**Modified (11):**
```text
configs/training/s1_mention_first_run_smoke.yaml     (v5 target, v1-v4 history)
docs/audits/README.md                                (0029 index entry)
docs/notebooks/notebook_execution_integrity.md       (S1 status note)
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb    (v5 output, v4 added to history)
notebooks/MedNorm_S1_Mention_Full_Training.ipynb     (full-corpus preflight gate, v5 inputs)
notebooks/MedNorm_S1_Smoke_Artifact_Validation.ipynb (v5 default, warns on v1-v4)
src/mednorm_vi/training/phobert_alignment.py         (verbatim original, general equivalence rule, cluster-aware matching)
src/mednorm_vi/training/s1_mention_smoke.py          (alignment preflight report and runner)
tests/unit/test_notebook_placeholders.py             (+2 preflight-gate assertions, v5)
tests/unit/test_s1_alignment_diagnostics.py          (Audit 0029 section, table retired to an oracle)
tests/unit/test_s1_artifact_validation.py            (v1-v5 lifecycle)
```

Counted precisely: **2 added, 11 modified — 13 changed tracked files.**

## 9. Remaining risks

1. **Nothing has been rerun on Colab.** The authoritative preflight uses the real VnCoreNLP; the
   local run models it from two proven transformations. No v5 artifact exists.
2. If the segmenter performs a rewrite outside the accepted rule, the preflight fails loudly
   with the reason code, stage, Unicode categories and `same_base_letter` — before any GPU time
   is spent, which is the point of the gate.
3. The preflight encodes the whole supervised corpus once (a few minutes on Colab) before
   training starts. That cost is deliberate.

## 10. Honest status

`IMPLEMENTED_UNEXECUTED`. The corrections are unit-tested and the full supervised corpus aligns
cleanly under a local model of the segmenter, but **the Colab preflight and the v5 smoke have
not been run**. Full S1 training has not run.
