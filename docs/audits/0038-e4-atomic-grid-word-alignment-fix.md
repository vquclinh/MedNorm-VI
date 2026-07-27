# Audit 0038 - E4 Atomic Grid-Word Alignment Fix

Date: 2026-07-27

## 1. Objective and Scope

This append-only bug-fix audit records the repository fix for a real Colab E4
failure that occurred **after** Audit 0037, on:

`notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`

Audit 0037 fixed the Colab bootstrap and the slow-`PhobertTokenizer` path. The
fresh run reached the official tokenizer successfully and then failed while
building governed **validation** W2NER targets. That second failure is the
subject of this audit.

This milestone does not edit `docs/MedNorm-VI_Architecture.pdf`, Audit 0036, or
Audit 0037. It does not train locally, run internal_test, run organizer
inference, create `output.zip`, or commit/push. It does not change
`src/mednorm_vi/evaluation/exact_mention.py`.

Final status:

- E4 atomic grid-word alignment fix: `IMPLEMENTED`, `PREFLIGHT_EXECUTED`,
  `READY_FOR_COLAB_SMOKE`.
- E4 PhoBERT W2NER training artifact: `UNAVAILABLE_UNTRAINED`.

No `SMOKE_EXECUTED`, `ARTIFACT_VALIDATED`, or `FULLY_TRAINED` claim is made.

## 2. Initial Git State

```text
pwd
/mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI

git branch --show-current
main

git status -sb
## main...origin/main

git status --short
<clean>

git log --oneline -6
020cae7 fix: repair E4 Colab bootstrap and PhoBERT alignment
03fc55f feat: complete phase2 Colab training readiness
c6107a4 feat: add phase2 mention ensemble and learned l4 v2 contracts
23065f9 feat: implement L3 span lattice and L4 resolver ablation
a00af3c feat: add exact character-offset S1 mention evaluator
15d0b07 feat: validate S1 checkpoint and add local held-out evaluation

git diff --check
<clean>
```

Audit 0037 is committed at `020cae7` and was left immutable. Architecture PDF
read in full before any code change; SHA-256
`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`, unchanged.
Audits 0036 and 0037 read in full.

## 3. Exact Failed Example

The operator's Colab run reached the official PhoBERT tokenizer:

```text
tokenizer_class = PhobertTokenizer
tokenizer_is_fast = false
slow_path_required_for_official_phobert = true
```

and then raised, while constructing governed validation contracts:

```text
W2NERError: entity SYMPTOM 85:102 is not word-aligned
```

**Reproduced locally, byte for byte.** The example is in the governed
**validation** split (not train, despite the `:train:` fragment inside its id):

```text
example_id  vimedner:train:train-000054
split       validation
row_index   4

original text
tìm kiếm các dấu hiệu của các bệnh khác , chẳng hạn như bệnh tuyến giáp ,
có thể gây rối loạn nhịp tim .

gold entity
type  = SYMPTOM
start = 85
end   = 102
text  = rối loạn nhịp tim
```

The §4 invariant holds on the gold span:

```text
original_text[85:102] == "rối loạn nhịp tim"     True
```

VnCoreNLP segmentation of that sentence:

```text
tìm_kiếm các dấu_hiệu của các bệnh khác , chẳng_hạn như bệnh tuyến_giáp ,
có_thể gây_rối loạn nhịp tim .
```

Relevant segmented model words and their original spans:

```text
gây_rối    81:88
loạn       89:93
nhịp       94:98
tim        99:102
```

The gold entity starts at character 85, **three characters inside** the single
model word `gây_rối`.

Observed alignment against the model-word surface:

```text
left_boundary_aligned  = false
right_boundary_aligned = true
```

The second governed entity in the same example, `DIAGNOSIS 56:71
"bệnh tuyến giáp"`, aligned fine, so the failure is specific to the merge.

## 4. Root Cause

The W2NER relation grid was indexed by **VnCoreNLP segmented model words**
(`build_w2ner_batch_contract_from_segmented_words` passed the segmented words
straight into `build_w2ner_grid`). A W2NER grid can only express an entity whose
start and end coincide with word boundaries, and a word segmenter is entitled to
merge syllables that a gold annotation splits.

So the grid coordinate system and the PhoBERT/VnCoreNLP model-token coordinate
system were the same system, and they must not be. The gold span is correct; the
grid was wrong. This is **not** caused by the unauthenticated Hugging Face
warning, and it is not corrupt annotation.

## 5. Full-Corpus Diagnostic

New module `src/mednorm_vi/training/phase2/e4_alignment_diagnostic.py`, executed
locally over the **complete** governed train and validation splits with real
VnCoreNLP segmentation. internal_test is refused by the module itself.

```text
diagnostic_version   e4-alignment-diagnostic-v1
report sha256        62cb541f84a9c19aecda9cd486067366a03507d1a832bf1b33450b6baebd1af6
report path          reports/e4_alignment/e4_alignment_diagnostic.json   (git-ignored)

total examples                                34,871
total entities                                13,711

VnCoreNLP segmented model-word surface (Audit 0037 grid)
  fully aligned                               13,552
  misaligned left only                           119
  misaligned right only                           40
  misaligned both boundaries                       0
  TOTAL MISALIGNED                               159

atomic original-text word surface (Audit 0038 grid)
  fully aligned                               13,711
  misaligned (any)                                 0

entities fixed by atomic word boundaries         159
entities still not alignable after atomic
  deterministic tokenization                       0
silent exclusions                                  0
atomic/model projection violations                 0

maximum atomic word count                        162
maximum VnCoreNLP model-word count               162
maximum PhoBERT subtoken count           unavailable locally (see below)
examples exceeding max_words (256)                 0
examples exceeding max_model_tokens (512) unavailable locally
```

Counts by split:

| Split | Examples | Entities | Segmented aligned | Left only | Right only | Atomic aligned |
| --- | --- | --- | --- | --- | --- | --- |
| train | 33,826 | 11,720 | 11,574 | 113 | 33 | 11,720 |
| validation | 1,045 | 1,991 | 1,978 | 6 | 7 | 1,991 |

Counts by source:

| Source | Examples | Entities | Segmented aligned | Left only | Right only |
| --- | --- | --- | --- | --- | --- |
| vimedner | 6,702 | 10,822 | 10,696 | 96 | 30 |
| vietmed_ner | 9,267 | 2,114 | 2,081 | 23 | 10 |
| vimq | 8,875 | 775 | 775 | 0 | 0 |
| phoner_covid19 | 10,027 | 0 | 0 | 0 | 0 |

Counts by entity type (segmented surface):

| Type | Aligned | Left only | Right only | Both |
| --- | --- | --- | --- | --- |
| DIAGNOSIS | 7,530 | 86 | 24 | 0 |
| SYMPTOM | 3,166 | 10 | 6 | 0 |
| MEDICATION | 2,856 | 23 | 10 | 0 |

The report contains the required record:

```json
{
  "example_id": "vimedner:train:train-000054",
  "split": "validation",
  "row_index": 4,
  "source": "vimedner",
  "entity_type": "SYMPTOM",
  "start": 85,
  "end": 102,
  "segmented_category": "misaligned_left_only",
  "atomic_category": "aligned",
  "fixed_by_atomic_words": true,
  "straddling_model_word": "gây_rối",
  "straddling_model_word_start": 81,
  "straddling_model_word_end": 88
}
```

All 159 mismatch records are enumerated in the report; none were truncated.

**PhoBERT subtoken statistics.** `vinai/phobert-large` is not in the local
Hugging Face cache and this milestone does not download model artifacts, so
`max_phobert_subtoken_count` and `examples_exceeding_max_model_tokens` are
reported as `-1` with `phobert_subtoken_statistics_available: false`. They are
**not** reported as zero. The notebook passes the real tokenizer to the same
function in Colab, where both are measured before the encoder is acquired.

## 6. Atomic Grid-Word Implementation

`src/mednorm_vi/mention_factory/w2ner.py`:

```text
ATOMIC_WORD_POLICY_VERSION = "atomic-original-word-v1"
tokenize_atomic_words(original_text) -> tuple[WordToken, ...]
entity_atomic_alignment(words, entity) -> (left_aligned, right_aligned)
is_atomic_boundary_character(character) -> bool
```

Rules, applied to `original_text` alone:

- whitespace separates words and never belongs to one;
- every punctuation/symbol character (Unicode category `P*`/`S*`) is its own
  atomic word, so punctuation is represented explicitly;
- every other maximal run of characters is one atomic word;
- exact end-exclusive offsets, `original_text[start:end] == surface`, validated
  per word on construction;
- combining marks are ordinary characters, so decomposed Unicode survives and no
  normalized copy is ever built;
- repeated spaces and newlines shift offsets without changing the word set;
- no gold-dependent splitting, no snapping, no trimming, no silent exclusion.

`build_w2ner_grid` now defaults to this surface, and
`build_w2ner_batch_contract_from_segmented_words` builds the grid from it while
retaining the VnCoreNLP words verbatim for PhoBERT.

For the failed example the atomic surface contains exactly what the milestone
required:

```text
gây  81:84
rối  85:88
loạn 89:93
nhịp 94:98
tim  99:102
```

so `SYMPTOM 85:102` aligns to `rối | loạn | nhịp | tim`, and the grid decodes
back to `(85, 102, "SYMPTOM", "rối loạn nhịp tim")`.

### One documented exception

The segmenter join character `_` is treated as **word-internal**, not
punctuation. Several governed sources (ViMQ, PhoNER-COVID19) ship text that is
already RDRSegmenter output, so `_` occurs literally inside `original_text` and
inside gold entity text. Both policies were measured over the full corpus before
choosing: splitting on `_` and not splitting on `_` each align **13,711 / 13,711**
entities, but keeping it word-internal caps the atomic word count at 162 instead
of 244. The relation grid is O(n²), so the smaller surface was chosen on measured
evidence, and the constant is tracked as
`ATOMIC_WORD_INTERNAL_CHARACTERS`.

### Error message

`_word_bounds_for_entity` now names which edge failed and which word straddled
it. Audit 0037's message was only `is not word-aligned`, which cost a full Colab
round trip to diagnose. The entity is still never moved to make the check pass.

## 7. PhoBERT Projection Implementation

`src/mednorm_vi/training/phase2/e4_w2ner_training.py`:

```text
ATOMIC_PROJECTION_VERSION = "atomic-projection-v1"
ATOMIC_FEATURE_DIM = 3

AtomicWordProjection
build_atomic_projection(original_text, segmented_words, encoding, *, atomic_words)
project_to_atomic_word_embeddings(sequence_output, projection)
atomic_relation_head_input_dim(hidden_size) -> hidden_size + 3
```

Projection chain:

```text
PhoBERT subtoken -> VnCoreNLP segmented model word -> atomic original-text word
```

**Neither surface is a refinement of the other**, which the corpus scan proved
rather than assumed:

- one model word may cover several atomic words — `gây_rối` covers `gây` and
  `rối` (the reported failure);
- one atomic word may span several model words — VnCoreNLP splits at
  letter/digit transitions, so `beta1` becomes `beta` + `1` while the atomic
  surface keeps it whole. A first, containment-based implementation of the
  projection raised on **31 governed examples** for exactly this reason; the
  invariant check that caught it is part of the diagnostic and now reports 0.

The projection is therefore **overlap-based**: an atomic word maps to every model
word it overlaps and pools the union of their subtokens. No assumption about
which tokenizer is finer is embedded anywhere.

**Explicit pooling rule.** An atomic word's representation is the mean of the
subtoken states of every model word it overlaps, with three deterministic
features appended:

| Feature | Meaning |
| --- | --- |
| `start_ratio` | atomic word start, relative to its primary model word's span |
| `end_ratio` | atomic word end, relative to its primary model word's span |
| `index_ratio` | atomic word ordinal among those sharing that primary model word |

The *primary* model word is the one sharing the most characters with the atomic
word, ties broken by earliest index — fully deterministic.

Atomic words sharing a merged model token therefore share contextual content but
are **not** indistinguishable. For the audited example:

```text
gây -> model word 14 (gây_rối), subtokens (15,), features (0.0,      0.428571, 0.0)
rối -> model word 14 (gây_rối), subtokens (15,), features (0.571429, 1.0,      1.0)
```

A test asserts the subtokens are identical **and** the features differ, so the
behaviour is documented and pinned rather than left implicit. When a model word
contains exactly one atomic word (the overwhelming majority) the features are
`(0.0, 1.0, 0.0)` and the projection reduces to plain mean pooling.

The subtoken-slice alternative was rejected deliberately and the reason is
recorded in the class docstring: the official slow `PhobertTokenizer` exposes no
subtoken character offsets, so per-atomic-word slicing would require
reconstructing BPE piece boundaries from the `@@` continuation marker.

Training and inference use the identical atomic tokenizer and the identical
projection — the notebook routes both through one `_atomic_word_embeddings`
helper. Gold entities create labels only; they never touch atomic word
construction.

## 8. Contract and Schema Version Changes

| Field | Old | New |
| --- | --- | --- |
| `W2NER_CONFIG_VERSION` | `phobert-w2ner-v1` | `phobert-w2ner-v2` |
| `E4_STAGE_ID` | `phase2-e4-phobert-w2ner-v1` | `phase2-e4-phobert-w2ner-v2` |
| E4 input contract | (absent) | `e4-atomic-grid-word-v1` |
| E4 checkpoint schema | (absent) | `phase2-e4-checkpoint-v2` |
| Grid word surface | (implicit) | `atomic-original-word-v1` |
| Atomic projection | (absent) | `atomic-projection-v1` |

`build_e4_resolved_config` now carries `e4_input_contract_version`,
`e4_checkpoint_schema_version`, `grid_word_surface`,
`atomic_projection_version`, `atomic_feature_dim` and `w2ner_config_version`, so
the resolved-config SHA-256 changes and an Audit-0037 artifact can never validate
against a v2 run.

`e4_checkpoint_payload()` writes the same identity into every checkpoint, and
`reject_incompatible_e4_checkpoint()` refuses any payload whose input contract,
checkpoint schema, or projection version does not match. The notebook calls it
during save/reload validation, so an Audit-0037 E4 checkpoint or smoke artifact is
**rejected, not resumed**.

The shared `CHECKPOINT_SCHEMA_VERSION` was deliberately **not** bumped: it also
covers E5 and L4, whose input contracts did not change, and bumping it would
invalidate artifacts this milestone did not touch. The new versions are E4-scoped.

## 9. Notebook Execution Order

`notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`, safe for one fresh
`Runtime -> Run all`:

1. mount Google Drive;
2. clone/update the repository and record the 40-hex commit;
3. set repository root, `sys.path` and `PYTHONPATH`, verify `mednorm_vi` imports
   from the clone;
4. resolve the governed corpus by authoritative SHA-256;
5. resolve immutable model/tokenizer revisions and gate on 40-hex values;
6. **acquire the tokenizer only** (plus the VnCoreNLP singleton);
7. create atomic original-word surfaces;
8. create VnCoreNLP model-word surfaces;
9. validate the atomic/model-word projection on the audited failing sentence;
10. run the complete train + validation corpus alignment preflight;
11. print and save the diagnostic summary (`e4_alignment_diagnostic.json` plus
    its SHA-256);
12. **only if the preflight passes**, acquire the 1.48 GB encoder;
13. run smoke/full training;
14. save, reload, hash and validate the artifact.

The encoder is guarded inside `run_training` by
`if not globals().get("PREFLIGHT_PASSED", False): raise RuntimeError(...)`, so it
cannot be downloaded or instantiated before alignment passes.

**VnCoreNLP singleton.** `py_vncorenlp` starts a JVM and `chdir`s into
`save_dir`; a second instance in one process raises. The annotator is cached in
`globals()["VNCORENLP_ANNOTATOR"]` and is **not** reset when a cell is rerun, and
the working directory is restored immediately after construction. The notebook
contains exactly one `py_vncorenlp.VnCoreNLP(` call, asserted by test.

The relation head is now built with `atomic_relation_head_input_dim(hidden_size)`
so its input width matches the projected atomic representation.

## 10. Entity Policy

After atomic word construction, over the complete governed train and validation
splits:

- every one of the 13,711 governed entities aligns to atomic word boundaries;
- no snapping, no trimming, no change to entity text, no change to entity
  offsets, no silent exclusion, no gold-dependent splitting;
- entities that cannot align raise loudly — verified by a test that a genuinely
  unalignable span raises `W2NERError` rather than being moved;
- training readiness passes under the explicit governed policy with
  `entities_unalignable_after_atomic_words == 0`,
  `silent_exclusions == 0` and `atomic_projection_violations == 0`.

## 11. Tests and Static Checks

New: `tests/unit/test_e4_atomic_grid_alignment.py` (43 tests), covering the exact
failed example `vimedner:train:train-000054`; `gây_rối` mapping to atomic `gây`
and `rối`; `rối loạn nhịp tim` aligning to 85:102; atomic word construction;
punctuation with and without surrounding spaces; repeated spaces; repeated
newlines; decomposed Unicode; the segmenter join character staying word-internal;
one model word mapping to multiple atomic words; one atomic word spanning
multiple model words; subtokens reaching atomic words through their model word;
projection output shape and feature dimension; atomic words under one merged
token remaining distinguishable; no gold-dependent tokenization; no snapping; no
silent exclusion; every diagnostic category; diagnostic determinism; the
internal_test refusal; contract/schema version bumps; Audit-0037 checkpoint
rejection; preflight-before-encoder ordering; VnCoreNLP singleton initialization;
and notebook cell parsing.

Modified: `tests/unit/test_e4_phobert_alignment_fix.py` —
`test_segmented_underscore_word_decodes_to_exact_original_slice` asserted the
Audit-0037 coupling `contract.grid.words[1].text == "suy tim"`, i.e. that a grid
word *is* a segmented model word. That is precisely the coupling this milestone
removes, so the test was updated to check the two surfaces separately and, in
addition, to assert the decode round trip it previously only implied. This is
stated plainly because it is a deliberate contract change to a committed test, not
a weakening: the test now asserts strictly more.

```text
env PYTHONPATH=src python -m pytest -q
1374 passed, 1 skipped in 124.89s (0:02:04)

env PYTHONPATH=src python -m pytest -q tests/unit/test_e4_atomic_grid_alignment.py
43 passed

ruff check .
All checks passed!

ruff check notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb
All checks passed!

env PYTHONPATH=src python -m mypy
Success: no issues found in 257 source files

env PYTHONPATH=src python -m compileall -q src
<passed>

git diff --check
<passed>
```

The single skip is the pre-existing local `pyarrow` skip in
`tests/unit/test_vietmed_adapter.py`, unrelated to E4.

Every modified notebook code cell was parsed with `compile(...)`; an AST scan
confirmed no undefined or unused names.

## 12. Unchanged Protected Files

Verified no diff via `git status --porcelain`:

- `docs/MedNorm-VI_Architecture.pdf`
  (`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`)
- `docs/audits/0036-phase2-training-readiness-and-artifact-validation.md`
- `docs/audits/0037-e4-colab-bootstrap-and-phobert-alignment-fix.md`
- `src/mednorm_vi/evaluation/exact_mention.py`

Governed corpus hashes recomputed unchanged:

```text
892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a  splits/train.jsonl
ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103  splits/validation.jsonl
e23acde0d87154d1750d25d1e47c92c86e457e42f2f97cb985661253dea7e135  splits/internal_test.jsonl
a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3  manifests/corpus_manifest.json
```

`git ls-files` tracks **0** files matching `*.pt|*.pth|*.ckpt|*.safetensors|*.bin|*.zip`;
`output.zip` does not exist; the generated diagnostic is git-ignored
(`.gitignore:60:reports/**`).

## 13. Changed Files

Added:

- `docs/audits/0038-e4-atomic-grid-word-alignment-fix.md`
- `src/mednorm_vi/training/phase2/e4_alignment_diagnostic.py`
- `tests/unit/test_e4_atomic_grid_alignment.py`

Modified:

- `notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb`
- `src/mednorm_vi/mention_factory/w2ner.py`
- `src/mednorm_vi/training/phase2/e4_w2ner_training.py`
- `tests/unit/test_e4_phobert_alignment_fix.py`

Generated locally and git-ignored:

- `reports/e4_alignment/e4_alignment_diagnostic.json`

## 14. Non-execution Guarantees

This milestone did not train locally, run backward on governed data, download the
PhoBERT encoder, call an external API, mutate any checkpoint, run internal_test,
run organizer inference, create `output.zip`, or add any corpus, model,
checkpoint or artifact file to Git. E4 remains disabled in every profile feature
flag.

## 15. Remaining Colab Work

The operator must rerun the E4 notebook in Colab to produce real smoke artifacts:

- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/checkpoints/best.pt`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/checkpoints/latest.pt`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/logs/training_history.jsonl`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/resolved_config.json`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/validation_metrics.json`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/training_manifest.json`
- `/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_smoke_v1/e4_alignment_diagnostic.json`

Two facts the local run could not establish and Colab will:
`max_phobert_subtoken_count` and `examples_exceeding_max_model_tokens`, both
measured by the same diagnostic before the encoder is acquired.

Any existing Audit-0037 E4 smoke artifact must be discarded, not resumed: it is
rejected by `reject_incompatible_e4_checkpoint`.

## 16. Safe-to-Commit Verdict

Safe to commit after review. Required tests and static checks pass, no protected
architecture/audit/evaluator file changed, governed corpus hashes are unchanged,
and no weight, checkpoint, corpus, report, cache or archive artifact is staged.
Do not include runtime directories or model weights in `git add`.

```bash
git add \
  docs/audits/0038-e4-atomic-grid-word-alignment-fix.md \
  src/mednorm_vi/training/phase2/e4_alignment_diagnostic.py \
  src/mednorm_vi/mention_factory/w2ner.py \
  src/mednorm_vi/training/phase2/e4_w2ner_training.py \
  notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb \
  tests/unit/test_e4_atomic_grid_alignment.py \
  tests/unit/test_e4_phobert_alignment_fix.py

git commit -m "fix: decouple E4 W2NER grid words from VnCoreNLP model words"
git push origin main
```
