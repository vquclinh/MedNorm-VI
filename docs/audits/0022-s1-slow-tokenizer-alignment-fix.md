# Audit 0022 — S1 ViHealthBERT Slow-Tokenizer Alignment Fix and Colab Smoke Repair

- **Date:** 2026-07-25
- **Author:** Claude (AI agent), for human review
- **Change type:** Correctness fix for the S1 mention smoke tokenization path. **No commit.
  No push. No local training, no model/tokenizer downloads, no GPU, no external API.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (L3 Mention Factory, S1 training stage,
  offset-preservation contract). No architecture change; the selected backbone is unchanged.
- **Status:** `IMPLEMENTED_UNEXECUTED` — the repaired notebook still requires a **new fresh
  Colab GPU run**.

Audit 0021 is committed (`e540c33 feat: prepare S1 mention first-run smoke`) and is **not
modified** by this milestone; this is the next immutable audit.

## 1. Verified Colab failure

A fresh Colab GPU run of `notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb` acquired the
tokenizer successfully and then stopped at:

```text
assert getattr(tokenizer, "is_fast", False), \
    "S1 smoke requires offset_mapping from a fast tokenizer."
AssertionError: S1 smoke requires offset_mapping from a fast tokenizer.
```

## 2. Exact root cause

`demdecuong/vihealthbert-base-word` declares `tokenizer_class = PhobertTokenizer`.
**`PhobertTokenizer` has no fast (Rust) implementation.** `AutoTokenizer.from_pretrained(...,
use_fast=True)` therefore falls back to the slow tokenizer and returns
`tokenizer.is_fast == False` — this is correct library behaviour, not a bug and not a
credentials problem.

The Audit 0021 notebook and `encode_mention_example()` incorrectly assumed a fast tokenizer
and depended on `return_offsets_mapping=True` to recover character offsets. That assumption,
not the assertion, was the defect.

**Unrelated to the failure:** the `HF_TOKEN` warning (anonymous download of a public model
succeeds) and the transformers `FutureWarning` (deprecation notice only). Neither influences
`is_fast`.

## 3. Tokenizer status (accepted, not asserted against)

```text
model id        : demdecuong/vihealthbert-base-word   (registry id: vihealthbert_span_type)
tokenizer class : PhobertTokenizer
tokenizer.is_fast: False          <- accepted as the supported path
loaded with     : AutoTokenizer.from_pretrained(..., use_fast=False)
alignment backend: phobert-slow-char-alignment-v1
```

The backbone was **not** replaced and no substitute vocabulary was introduced. The notebook
prints and records the tokenizer class, `is_fast`, vocabulary size, special-token IDs, and
the alignment backend. Requested and resolved revisions are both recorded in the manifest.

## 4. Slow-tokenizer alignment architecture

New tracked module **`src/mednorm_vi/training/phobert_alignment.py`** (not notebook-only
logic), with typed structures `SegmentedWord` and `AlignedSubtoken` and three explicit stages:

1. **segmented words → original characters** (`map_segmented_words`): each RDRSegmenter word
   (syllables joined by `_`) is located with a **strictly monotonic cursor** — never a global
   `str.find` — so repeated words, repeated whitespace, tabs, newlines and punctuation map to
   distinct ordered half-open `[start, end)` spans. A non-whitespace gap inside a segmented
   word, or a word that cannot be located, raises `AlignmentError` (fail fast, no guessing).
2. **words → BPE subtokens** (`align_subtokens`): each word is tokenized with the real slow
   tokenizer; every piece inherits its word's **original** character span, its source word
   index, and an `is_continuation` flag. Special tokens use the tokenizer's real IDs and carry
   `(0, 0)` spans.
3. **subtokens → supervision**: labels are derived from those original spans by the **same
   character-overlap rule** the fast path used, via a shared `_labels_from_offsets()` helper —
   so supervision semantics are provably unchanged.

`return_offsets_mapping`, `word_ids()`, `token_to_chars()` and `char_to_token()` are **not
used** on the slow path (test-enforced).

## 5. Word segmentation contract (hardened)

Production S1 smoke **requires** VnCoreNLP RDRSegmenter. The notebook defaults to
`SEGMENTER_MODE = "vncorenlp"`; resources are acquired **in Colab only**
(`py_vncorenlp==0.1.4`), cached under `DRIVE_ROOT/model_cache/vncorenlp`, and the run
**fails fast** when resources are missing (`"VnCoreNLP resources missing after
acquisition"`) or when no resource hashes can be computed (`"VnCoreNLP resource hashes are
empty"`). A successful production-path smoke therefore requires:

```text
word_segmenter   = "VnCoreNLP RDRSegmenter"
segmenter_mode   = "vncorenlp"
degraded_fallback = false
word_segmenter_resource_hashes  non-empty
```

These four conditions are combined into `PRODUCTION_SEGMENTATION`.

`whitespace_fallback` exists **only as explicit opt-in diagnostics**
(`MEDNORM_SEGMENTER_MODE=whitespace_fallback`). It never activates automatically, prints a
prominent banner ("DEGRADED MODE … cannot be classified as a successful production-path S1
smoke"), records `degraded_fallback=true`, and forces `full_training_readiness=false`.

The manifest records segmenter name, version, resource filenames, resource SHA-256 hashes,
mode, degraded flag, and acquisition source. Segmentation is **separated from alignment**, so
all local tests use synthetic segmented-word fixtures — no Java, no downloads, no network.

## 6. Subtoken supervision policy

```text
all_subtokens_inherit_word_span_overlap_labels
```

Every non-special subtoken inherits its word's original span and receives the multi-label
type vector by character overlap; **special tokens and padding get a zero label and
`label_mask = 0`**. This matches the existing multi-label mention-loss implementation
(`ENTITY_TYPE_ORDER`, overlap-based labels, `label_mask` for supervised positions) rather
than inventing a first-subtoken-only rule.

## 6b. Tokenizer equivalence preflight (before model acquisition)

The alignment backend tokenizes **word by word** to attach character spans, which is only
valid if the tokenizer is per-word decomposable. `verify_tokenizer_equivalence()` re-tokenizes
the **full segmented sentence** with the real tokenizer using `add_special_tokens=False` and
requires the IDs to equal the concatenated per-word IDs **exactly**. The notebook runs this on
**every** selected smoke example (train + validation), immediately after loading the real
`PhobertTokenizer` and **before** `AutoModel.from_pretrained`; any mismatch raises and the run
stops before weights are downloaded. Recorded: `tokenizer_equivalence_checked=true`,
`tokenizer_equivalence_examples`, `tokenizer_equivalence_failures` (must be 0).

Local unit tests exercise this helper with deterministic fake slow tokenizers (decomposable,
non-decomposable, and a no-`__call__` fallback). **These do not replace the real Colab check
against the actual PhobertTokenizer.**

## 7. Truncation and boundary policies

- **Truncation:** explicit and reported. The sequence is cut to `max_length` including special
  tokens; `truncated` and `truncated_entity_count` are returned per example and aggregated in
  the manifest. An entity that loses all supervised subtokens is **counted**, never silently
  mislabeled.
- **Partial truncation** (`PARTIAL_TRUNCATION_POLICY =
  "mask_all_retained_subtokens_of_partially_truncated_entity"`): when an entity's supervised
  subtokens only **partially** survive, the encoder masks **every retained subtoken of that
  entity** (labels zeroed, `label_mask = 0`) so no partial supervision leaks, and counts it.
  `classify_truncated_entities()` reports three disjoint classes and the encoder surfaces
  `fully_dropped_entity_count`, `partially_truncated_entity_count`, and
  `truncated_entity_count` (= dropped + partial); the notebook aggregates them plus
  `truncated_example_count`.
- **Boundary merges:** if word segmentation merges across a gold entity boundary (one token
  would need two incompatible labels), the example raises `AlignmentError` and is counted as
  `unalignable_example_count` under the documented policy
  `unalignable_example_counted_and_skipped` — never assigned an arbitrary label.

## 8. Loss and VietMed policy (unchanged)

VietMed entities remain `DRUGCHEMICAL → MEDICATION`, `MAP_APPROXIMATE`, with masks
`span=True, entity_type=True, assertions=False, icd_candidates=False, rxnorm_candidates=False`;
validation/internal_test remain MAP_EXACT-only. **No governed corpus file, split membership,
or artifact hash was touched** by this milestone.

## 9. Notebook changes

- removed the invalid `assert tokenizer.is_fast`;
- loads the tokenizer with `use_fast=False` and asserts it **is** slow;
- new word-segmentation cell (Colab-only acquisition, hash recording, fail-fast);
- new **alignment preflight** cell that runs **before** `AutoModel.from_pretrained` and
  verifies id/mask/label/label-mask dimensions on the selected smoke examples;
- encoding now calls `encode_mention_example_slow(...)`; deterministic repeat check retained;
- only aggregate diagnostics are printed (no clinical text);
- the bounded forward/backward/optimizer/checkpoint smoke is unchanged;
- the training manifest additionally records `tokenizer_class`, `tokenizer_is_fast`,
  `tokenizer_revision`, `alignment_backend`, `alignment_backend_version`,
  `subtoken_supervision_policy`, `word_segmenter`, `word_segmenter_version`,
  `word_segmenter_resource_hashes`, `aligned_example_count`, `unalignable_example_count`,
  `truncated_example_count`, `truncated_entity_count`, and — added by this pass —
  `segmenter_mode`, `degraded_fallback`, `tokenizer_equivalence_checked`,
  `tokenizer_equivalence_examples`, `tokenizer_equivalence_failures`,
  `fully_dropped_entity_count`, `partially_truncated_entity_count`,
  `partial_truncation_policy`, and `full_training_readiness`
  (= production segmentation **and** zero equivalence failures **and** zero unalignable
  examples).

## 10. Tests (no downloads)

`tests/unit/test_phobert_alignment.py` — 32 tests with a deterministic **fake slow
tokenizer** and synthetic Vietnamese text: slow tokenizer accepted; no fast-only APIs; single-
word and multi-word entities; multi-syllable underscore words; punctuation; repeated word
occurrences; repeated whitespace/tabs/newlines; adjacent entities; text with no entities; BPE
word split into multiple pieces (all pieces share the word span, continuation flagged);
special tokens unsupervised; padding ignored; truncation without entity loss; truncation
cutting an entity (counted); segmentation merge crossing an entity boundary (fails fast);
determinism; entity offset-invariant enforcement; VietMed masks span/type-only; dimension
agreement; plus an **adversarial test** proving that deleting the `is_fast` assertion alone —
without the alignment backend — still fails (the slow tokenizer rejects
`return_offsets_mapping`).

Added by this pass: **truncation classification** — entity fully retained; entity fully
removed (`fully_dropped`); entity partially cut (masked, `partially_truncated`, no residual
labels); multiple entities where only one is cut (the retained one keeps labels); special and
padding positions remain ignored after truncation; deterministic repeat. **Tokenizer
equivalence** — decomposable tokenizer passes, non-decomposable fails fast, and a
no-`__call__` tokenizer uses the `tokenize()` fallback.

`tests/unit/test_notebook_placeholders.py` — new S1 assertions: no `assert is_fast`, no
`use_fast=True`, no `return_offsets_mapping`/`word_ids`/`token_to_chars`/`char_to_token`,
slow tokenizer explicitly supported, tracked alignment helper invoked, segmentation resources
verified, preflight ordered before model download, unalignable examples counted, smoke limits
unchanged, no full-training branch, no placeholders.

## 11. Validation

```text
env PYTHONPATH=src python3 -m pytest -q          -> 690 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 221 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
```

No model acquisition, no segmentation-resource acquisition, and no training ran locally.

## 12. Exact changed files

Verified against `git status --short`, `git diff --name-status`, and `git diff --stat`.

Added (3):
```text
docs/audits/0022-s1-slow-tokenizer-alignment-fix.md
src/mednorm_vi/training/phobert_alignment.py
tests/unit/test_phobert_alignment.py
```
Modified (5):
```text
src/mednorm_vi/training/s1_mention_smoke.py        (encode_mention_example_slow + shared label helper)
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb  (slow tokenizer, segmentation, preflight, manifest)
tests/unit/test_notebook_placeholders.py           (slow-tokenizer notebook integrity assertions)
docs/audits/README.md                              (0022 index entry)
docs/notebooks/notebook_execution_integrity.md     (S1 status note)
```

## 13. Honest status

`IMPLEMENTED_UNEXECUTED`. The alignment backend is unit-tested locally, but the repaired
notebook **has not been run on Colab**. No smoke success, checkpoint, or training manifest may
be claimed until the user completes a **new fresh Colab GPU run** and returns the artifacts.
