# Audit 0026 — S1 Pre-Segmented-Source Alignment Fix and Privacy-Safe Diagnostics

- **Date:** 2026-07-26
- **Author:** Claude (AI agent), for human review
- **Change type:** Correctness fix in the slow-tokenizer alignment backend, plus privacy-safe
  alignment diagnostics and a counter-scope repair. **No commit. No push. No local training,
  no local GPU, no model/tokenizer/VnCoreNLP downloads, no organizer inference, no
  `output.zip`, no backbone change. Governed corpus files NOT modified.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 — §4 (offset preservation,
  `original_text[start:end] == entity["text"]`), §15 (S1 objective), §15.2 (reproducibility).
  The PDF was not modified or moved.
- **Status:** `IMPLEMENTED_UNEXECUTED` — requires a **new Colab smoke rerun**.

Audit 0025 is committed (`7db1641`) and is not modified. The next immutable number is 0026.

## 1. The blocker

The real S1 smoke manifest failed two validator conditions:

```text
alignment:
  aligned_example_count: 7
  tokenizer_equivalence_examples: 10
  tokenizer_equivalence_failures: 0
  unalignable_example_count: 1
full_training_readiness: false
```

## 2. Exact root cause

**Several governed sources are distributed ALREADY word-segmented, so `_` is a literal
character of `original_text` — and `map_segmented_words()` treated it as an illegal
separator.**

Measured across the governed corpus (local inspection, no files changed):

| Source | Train examples containing a literal `_` | Supervised |
| --- | --- | --- |
| `phoner_covid19` | 9,978 / 10,027 (99.51%) | 0 (coverage `boundary: false`) |
| `vimq` | 8,114 / 8,736 (92.88%) | **8,114** |
| `vimedner` | 1 / 5,796 (0.02%) | 1 |
| `vietmed_ner` | 0 / 9,267 | 0 |

Validation: `vimq` 123/139 (88.49%). Internal test: `vimq` 111/125 (88.80%).

**21,338 of 21,339 ViMQ underscores sit between two letters** — the RDRSegmenter
syllable-joiner shape. ViMQ and PhoNER-COVID19 ship pre-segmented text, and the governed
corpus stored it verbatim. Entity spans legitimately contain the underscore (e.g. a
MEDICATION span of shape `xxxxxxx_x`), and the §4 invariant
`original_text[start:end] == entity["text"]` holds throughout — the annotation is **not**
invalid.

The failure mechanism: `map_segmented_words()` splits each segmented word on `_` to recover
syllables, then requires the gap between consecutive syllables to be whitespace:

```python
gap = text[end:found]
if gap.strip():                       # a literal '_' is not whitespace
    raise AlignmentError("non-whitespace gap inside segmented word")
```

For pre-segmented sources the gap **is** the literal `_`, so every such example raised. The
code conflated *the segmenter's join character* with *a character of the source text*, which
are indistinguishable from the segmenter output alone.

**Impact far beyond the smoke:** 8,238 supervised train+validation examples (~34% of the
23,799 supervised training examples) would have been silently discarded in full S1 training.

## 3. Privacy-safe identity of the failed example, and the honest limit

The observed manifest predates the diagnostics added here, so it records only counts. What is
established from the governed data alone:

- the smoke selects **8 train + 4 validation = 12** examples, deterministically (seed
  20260723);
- exactly one selected example — `privacy_safe_example_id` **`6c3b519cd0100bec`**, source
  `vimq`, split `validation` — contains literal underscores (2 of them) and therefore **must**
  have failed at `stage: word_mapping`, with `reason_code: NON_SEPARATOR_GAP_INSIDE_WORD`
  (or `EMPTY_SEGMENTED_WORD` if the segmenter emits the underscore as a standalone token —
  either way it could not be mapped). It was one of the two examples the equivalence loop
  skipped, and it was **counted nowhere**;
- none of the 8 selected **train** examples contains an underscore, so the single
  `unalignable_example_count: 1` (a train-scope counter) has a **different** cause. The two
  train examples with adjacent entities separated by a single space are the candidates for a
  segmenter boundary merge: `privacy_safe_example_id` **`c06995a162f6eb76`** (source
  `vimedner`, 17 entities, a DIAGNOSIS/SYMPTOM pair one space apart) and
  **`fa52c9305b116301`** (source `vietmed_ner`, 3 entities, a MEDICATION/MEDICATION pair one
  space apart).

**Which of those two failed cannot be determined from the current manifest.** That is exactly
the diagnostic gap this audit closes; the rerun will name it. Both candidate causes are fixed
below, so the rerun should yield `unalignable_example_count: 0` either way.

## 4. Why equivalence covered 10 while aligned + unalignable was 8

The counters were **correct but had different, undocumented scopes**, and one scope had a hole:

| Counter | Scope before | Behaviour on failure |
| --- | --- | --- |
| `tokenizer_equivalence_*` | train **+** validation = 12 | `map_segmented_words` failures silently `continue`d |
| `aligned_example_count` / `unalignable_example_count` | **train only** = 8 | counted |
| validation encoding (a later cell) | validation = 4 | `except AlignmentError: continue`, **counted nowhere** |

So `12 − 2 skipped = 10` equivalence examples, and `7 + 1 = 8` over the train scope. The two
numbers were never meant to reconcile — but nothing said so, and a validation example that
failed alignment was **invisible**: it blocked nothing and appeared in no counter.

**Semantics kept, scope repaired.** The equivalence check and the alignment preflight now
cover the **same** 12 examples, and the manifest records both denominators explicitly plus a
reconciliation invariant:

```text
aligned_example_count + unalignable_example_count + governed_exclusion_count
    == examples_considered
tokenizer_equivalence_considered == examples_considered
```

`tokenizer_equivalence_examples` retains its distinct meaning — examples that *reached and
passed* the equivalence check — with the new `tokenizer_equivalence_skipped_unmappable`
accounting for those whose word mapping failed, so equivalence is undefined rather than
passing.

## 5. The fix

Two changes, both minimal; one is the proven defect, the other removes a disproportionate
policy that the same module had already solved better elsewhere.

**(a) Literal join character (the proven bug).** `is_legal_syllable_gap()` accepts whitespace
**or** `SEGMENTER_JOIN_CHARACTER`. A pre-segmented word then maps to the exact original
substring, underscore included; genuinely drifted mappings (punctuation, letters between
syllables) still raise. `ALIGNMENT_BACKEND` is bumped to
`phobert-slow-char-alignment-v2`.

**(b) Boundary merge: mask, don't discard.** `BOUNDARY_MERGE_POLICY` changes from
`unalignable_example_counted_and_skipped` to
`mask_straddling_word_and_affected_entity_subtokens`. Previously one ambiguous word threw away
the entire example — for the 17-entity candidate that meant losing 16 well-formed entities.
The straddling word and every subtoken of the entities it touches are now masked (labels
zeroed, `label_mask = 0`) and counted in `boundary_merge_masked_word_count` /
`boundary_merge_affected_entity_count`. **No token is ever given an incompatible label** —
this is the identical rule `PARTIAL_TRUNCATION_POLICY` already applies to truncation.

This is **not** a weakening of the validator: `unalignable_example_count == 0` is still
required, and the masking is recorded, not hidden.

## 6. Readiness policy: expected vs unexpected

`AlignmentError` now carries a stable, text-free `reason_code`, and every failure is classified:

| Family | Reason codes | Counter | Blocks readiness |
| --- | --- | --- | --- |
| Unexpected | `SEGMENTED_SYLLABLE_NOT_FOUND`, `NON_SEPARATOR_GAP_INSIDE_WORD`, `EMPTY_SEGMENTED_WORD`, `TOKENIZER_NOT_DECOMPOSABLE`, `TOKENIZER_PIECE_ID_MISMATCH`, `ENTITY_OFFSET_INVARIANT_VIOLATED`, `MAX_LENGTH_TOO_SMALL` | `unalignable_example_count` | **YES** |
| Expected | `GOVERNED_EXCLUSION` | `governed_exclusion_count` | no |

Stages: `word_mapping`, `tokenizer_equivalence`, `subtoken_encoding`, `governed_exclusion`.

`configs/training/s1_governed_exclusions.yaml` holds the deterministic exclusion policy, keyed
by `privacy_safe_example_id`, each entry requiring a `justification`. **It is deliberately
empty**: no invalid governed annotation was found, so nothing is excluded and every failure
still blocks. Skipping an example is a governance decision, never a way to silence a bug.

## 7. Privacy-safe diagnostics

Recorded per failed example — and nothing else:

```text
source · split · privacy_safe_example_id · stage · reason_code · exception_type
```

`privacy_safe_example_id = sha256(example_id).hexdigest()[:16]` — non-reversible, stable, and
recomputable locally by a reviewer. No raw clinical text, no raw entity text, no verbatim
example id, and no exception message reaches a manifest or this audit. The artifact validator
enforces it: any diagnostic with an unexpected key or a non-16-hex handle fails validation.

## 8. Governance observation (NOT acted on)

Storing pre-segmented text as `original_text` is in tension with spec §4, which requires
`original_text` to be the untouched source used to emit text and position. For S1 training the
corpus is self-consistent and the fix above is sufficient. **No governed corpus file was
modified.** A corpus rebuild that recovers raw ViMQ/PhoNER text is a separate decision needing
its own evidence and sign-off; it is recorded in the exclusion policy file and here so it is
not lost.

## 9. Preserved behaviour

Two-pass Colab bootstrap with the fingerprint marker and single forced restart; scoped S1
dependency health with full `pip check` diagnostics; NumPy `RandomState` + dummy AdamW ABI
preflight; ViHealthBERT slow `PhobertTokenizer` (`is_fast=False`); VnCoreNLP production
segmentation (asserted); tokenizer equivalence preflight; VietMed span/type-only supervision;
smoke limits; checkpoint save/reload; and the artifact-validator hash contract (recomputed
SHA-256 must match both the manifest and the expected hash).

## 10. Tests and validation

New `tests/unit/test_s1_alignment_diagnostics.py` (29). Updated: `test_phobert_alignment.py`
(boundary-merge test now asserts masking instead of raising),
`test_s1_artifact_validation.py` (+6, fixture updated to the new alignment shape),
`test_notebook_placeholders.py` (+3).

Covered: the exact failing pattern (literal `_` in pre-segmented source maps, and an entity
span containing `_` labels exactly its own word); space-joined compounds unchanged; punctuation
between syllables still illegal; a genuinely unmappable word still raises with a reason code;
boundary merge masks rather than discards, with no masked token retaining a label;
privacy-safe id is stable and non-reversible; a diagnostic carries only the six permitted
fields and leaks no text or verbatim id; unexpected and expected failures use different
counters; unexpected failures block readiness while governed exclusions do not; the exclusion
policy is tracked, empty, requires justifications, and contains no verbatim ids; the validator
rejects non-reconciling counters, mismatched equivalence/alignment scopes, and diagnostics
carrying content.

```text
env PYTHONPATH=src python3 -m pytest -q          -> 966 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 224 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
notebook code-cell AST parse                      -> 3 S1 notebooks, 0 syntax errors
```

## 11. Exact changed files

**Added (3):**
```text
configs/training/s1_governed_exclusions.yaml       (deterministic exclusion policy, empty)
docs/audits/0026-s1-pre-segmented-source-alignment-fix.md
tests/unit/test_s1_alignment_diagnostics.py        (29 tests)
```

**Modified (16):**
```text
configs/training/s1_mention_first_run_smoke.yaml   (versioned v2 output contract)
configs/training/s1_mention_full_training.yaml     (smoke_artifact_dir -> v2)
docs/audits/README.md                              (0026 index entry)
docs/notebooks/notebook_execution_integrity.md     (S1 smoke status note)
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb  (unified preflight, diagnostics, reconciliation, versioned output)
notebooks/MedNorm_S1_Mention_Full_Training.ipynb   (diagnostics path, runtime artifact inputs, v1 refusal)
notebooks/MedNorm_S1_Smoke_Artifact_Validation.ipynb (runtime artifact dir + operator-supplied hash)
src/mednorm_vi/training/phobert_alignment.py       (join-character fix, reason codes, boundary-merge masking)
src/mednorm_vi/training/s1_artifact_validation.py  (no hardcoded hash; reconciliation + privacy conditions)
src/mednorm_vi/training/s1_full_training.py        (runtime smoke_artifact_dir override)
src/mednorm_vi/training/s1_mention_smoke.py        (masking, diagnostics, exclusions, versioned output paths)
tests/unit/test_notebook_placeholders.py           (+8 preflight/diagnostic/lifecycle assertions)
tests/unit/test_notebooks.py                       (versioned smoke output path)
tests/unit/test_phobert_alignment.py               (boundary-merge masking)
tests/unit/test_s1_artifact_validation.py          (+15 reconciliation/privacy/lifecycle tests)
tests/unit/test_s1_full_training.py                (+4 validated-artifact consumption tests)
```

Counted precisely: **3 added, 16 modified — 19 changed tracked files.**

## 12. Artifact lifecycle

```text
v1  artifacts/s1_mention_first_run_smoke      historical, full_training_readiness: false
                                              IMMUTABLE evidence - never overwritten
v2  artifacts/s1_mention_first_run_smoke_v2   corrected rerun after this alignment fix
validator                                     runtime artifact path + user-confirmed SHA
full training                                 allowed ONLY after v2 validation passes
```

**v1 stays untouched.** The existing checkpoint (`7310f69a…92213`) and its manifest are not
edited; they remain the honest record of the pre-fix run.

**v2 is where the rerun writes.** The smoke output path is now explicit and versioned in
`configs/training/s1_mention_first_run_smoke.yaml`:

```yaml
output:
  artifact_version: v2
  artifact_dir: .../artifacts/s1_mention_first_run_smoke_v2
  previous_artifact_dir: .../artifacts/s1_mention_first_run_smoke
  previous_artifact_status: HISTORICAL_FULL_TRAINING_READINESS_FALSE
```

`smoke_artifact_paths_from_config()` raises if the two are equal, and the smoke notebook
asserts both that its `OUTPUT_DIR` differs from v1 and that its version matches the tracked
config.

**The validator carries no run-specific hash.** `EXPECTED_SMOKE_CHECKPOINT_SHA256` has been
**removed from `s1_artifact_validation.py`** — a reusable validator must not need a source
edit to accept the next rerun. `expected_checkpoint_sha256` is now a required argument
supplied by the operator, and validation requires all three to agree:

```text
recomputed SHA-256  ==  manifest artifacts.checkpoint_sha256  ==  operator-supplied hash
```

A missing or malformed expected hash **never** grants readiness. Instead the validator reports
the recomputed digest inside the failure message, and the notebook prints the exact line to
paste — no Python changes. Whitespace and letter case are tolerated; anything that is not a
64-hex digest is rejected.

**Both notebooks take runtime inputs**, defaulting to v2 and to an *empty* hash, so the old
artifact can never be silently accepted as the new one:

| Variable | Environment override | Default |
| --- | --- | --- |
| `SMOKE_ARTIFACT_DIR` | `MEDNORM_SMOKE_ARTIFACT_DIR` | `…_v2` (via `MEDNORM_SMOKE_ARTIFACT_VERSION`) |
| `EXPECTED_SMOKE_CHECKPOINT_SHA256` | `MEDNORM_EXPECTED_SMOKE_CHECKPOINT_SHA256` | **empty** |

The validation notebook warns loudly if pointed at v1. The full-training notebook **refuses**
v1 outright, and passes the directory it actually validated into
`load_full_training_config(..., smoke_artifact_dir=...)`, so training is authorized by the
artifact that passed rather than by a hardcoded path. `configs/training/s1_mention_full_training.yaml`
now defaults `output.smoke_artifact_dir` to the v2 directory.

## 13. Honest status

`IMPLEMENTED_UNEXECUTED`. The fix and the artifact lifecycle are unit-tested and the notebooks
are statically verified, but **nothing has been rerun**. No v2 artifact exists yet. No claim is made that the smoke artifact now validates, and the
identity of the single train-scope failure remains unconfirmed until the rerun emits the new
diagnostics. Full S1 training has not run.
