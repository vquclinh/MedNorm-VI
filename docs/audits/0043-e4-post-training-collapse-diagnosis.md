# Audit 0043 - E4 Post-Training Collapse Diagnosis

Date: 2026-07-28

## 1. Objective and Scope

The real E4 PhoBERT W2NER full-training run **completed at the engineering
level** and **failed its quality gate**. This append-only audit records what the
run actually did, what has been ruled out with evidence, what remains unproven,
and the one bounded experiment that will close the remaining gap.

**No fix is implemented in this milestone.** The loss is not changed, the model
is not retrained, no threshold is tuned and no repair is claimed. Everything
below is a measurement or a static fact about code that exists at the audited
commit.

Nothing was trained locally. `internal_test` was never opened. No organizer
inference ran. No `output.zip` was produced. The immutable local artifact was
read and hashed, never modified, renamed, moved or regenerated.

### Status

| Subject | Status |
| --- | --- |
| E4 full run (engineering) | `FULLY_TRAINED`, `ARTIFACT_VALIDATOR_OK`, `INTERNAL_TEST_UNTOUCHED` |
| E4 full run (quality) | `QUALITY_GATE_FAILED`, `ZERO_PREDICTIONS_ON_GOVERNED_VALIDATION` |
| Root cause | `ROOT_CAUSE_NOT_YET_PROVEN` |
| Leading hypothesis | `ALL_BACKGROUND_LOSS_COLLAPSE` (strongly supported, not proven) |
| Tiny-overfit diagnostic | `IMPLEMENTED_NOT_AUTHORIZED` |

**No repair is claimed and no model quality is claimed.**

## 2. Initial Git State

```text
pwd                       /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
git branch --show-current main
git status --short        <clean>

git log --oneline -3
df74a6d feat: add E5 S2 and parameter budget readiness
4a55682 feat: add E4 training progress heartbeats and ETA
2dade89 fix: harden E4 Colab corpus IO and stream W2NER contracts
```

Audit 0041 is committed at `4a55682` and Audit 0042 at `df74a6d`, so this
milestone creates the append-only Audit 0043 and leaves both untouched.

Architecture PDF read in full before any change; SHA-256
`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`, unchanged.

### Artifact directory correction

The milestone brief named the artifact `.local_artifacts/e4_phobert_w2ner_full_v1`.
The directory on disk is `local-artifacts/e4_phobert_w2ner_full_v1` (no leading
dot, hyphen not underscore), covered by the `local-artifacts/` rule at
`.gitignore:83`. That path is what was diagnosed. `.local_artifacts/` does not
exist and was not created.

## 3. Artifact Integrity

### 3.1 What is present

```text
local-artifacts/e4_phobert_w2ner_full_v1/
  training_manifest.json          52b8673a2bd5f4b99bda1e6b10eb2cf8debdb08b5ee7eb6a51f5cbe71d948d73
  resolved_config.json            c7d47737b46facc1df024f7cc8c2910cc6a9e3de3231f3d904f34b3f4c476a96
  validation_metrics.json         c563ea63fde15c174ca43b1c7c7e341e20e0c3f3a6e941f271f56311a5b1cd50
  grid_target_statistics.json     0d959c8f4eea428babece37106175d3f0db7a10cbfb3a5ea2f837c4ae0bbf05a
  e4_alignment_diagnostic.json    86ad2fc50bb9084d3a6ba2328c9b571a77aa0067c82fcc0481bb70b5d2c34bc7
  logs/training_history.jsonl     2a2127de035cfd8f4875b8026c091a34636e7b20fbef984886670506382363b6
  logs/training_progress.jsonl    30e9e359a04b19f39a0b52d295f34b480e45f9542df182ac31023dceb9177ff8
```

Every file cross-checks: manifest and metrics declare identical checkpoint
hashes, the recorded label space equals the repository type order, and
`grid_target_statistics.label_count` (7) equals `len(W2NERLabelVocab().labels)`.
`verify_artifact_integrity()` reports **zero inconsistencies**.

### 3.2 What is absent — and why it matters

```text
  checkpoints/best.pt      ABSENT
  checkpoints/latest.pt    ABSENT
```

The weights were **not** downloaded from Colab. The artifact declares their
digests, which match the values supplied in the milestone brief:

```text
best     4cc934eb5d072bcf827e46745bbcc308beda3552b4156c4c4504f571aa0bd16f
latest   22b9017da3e5a56b7086c7f03dda1aed7e78b5e7b9f844d6171ee9333355bf07
```

but the files themselves are not on disk, so their SHA-256 could not be
verified, `torch.load` was never called, the checkpoint schema and W2NER head
keys were not inspected, and **section E could not be executed at all**.

This is recorded as a bounded *evidence gap*, not as an artifact inconsistency.
`require_checkpoint()` raises a dedicated `CheckpointEvidenceUnavailable`
precisely so that "the classifier emits background" and "we could not look at the
classifier" can never collapse into the same outcome.

### 3.3 Recorded run identity

```text
expert_id                E4_phobert_w2ner
stage_id                 phase2-e4-phobert-w2ner-v2
mode                     full
model_id / revision      vinai/phobert-large @ 1c7880f20db59c0054c6de5afd71b012369f6ee4
input contract           e4-atomic-grid-word-v1
grid word surface        atomic-original-word-v1
label space              DIAGNOSIS, MEDICATION, SYMPTOM, TEST_NAME, TEST_RESULT
relation labels          7
parameter_count          371,289,161
epochs completed         12
optimizer steps          50,748   (expected 4,229 x 12)
backward passes          405,912  (expected 33,826 x 12)
precision                bf16 on cuda, no GradScaler
accumulation             micro1-accum8-effective8
weight format            pytorch_model.bin, use_safetensors False
GPU                      NVIDIA A100-SXM4-40GB
elapsed                  40,288.6 s
internal_test_accessed   false
best epoch               2
best validation exact F1 0.0019960079840319364
```

The encoder load report shows only the expected PhoBERT MLM-head keys as
unexpected; `missing_keys` is empty. `parameter_count` 371,289,161 is a real
programmatic count and is the first verified figure available for
`e4_phobert_w2ner`, whose registry entry is still `parameter_count_verified:
false` (Audit 0042). Recording that verification is deliberately left to a
separate milestone, since this one must not modify the model registry.

## 4. Epoch-by-Epoch Metrics

Reconstructed by joining `logs/training_history.jsonl` (loss, precision, recall,
F1) with the per-epoch validation heartbeats in `logs/training_progress.jsonl`
(the raw predicted/gold counts, which the history rows do not carry).

```text
epoch  train_loss   pred  gold    tp    fp    fn   precision      recall          f1  new_best
    1  0.01364423      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000      True
    2  0.00647322     13  1991     2    11  1989  0.15384615  0.00100452  0.00199601      True
    3  0.00593435      4  1991     1     3  1990  0.25000000  0.00050226  0.00100251     False
    4  0.00925738      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
    5  0.01368798      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
    6  0.01378138      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
    7  0.01362625      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
    8  0.01360618      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
    9  0.01359775      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
   10  0.01360213      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
   11  0.01360450      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
   12  0.01360332      0  1991     0     0  1991  0.00000000  0.00000000  0.00000000     False
```

`checkpoint_hash` per epoch is **UNAVAILABLE**: the `epoch_checkpoint_persisted`
records in the progress log do not carry a digest field. It is marked
`UNAVAILABLE` rather than filled in, because "no hash was recorded" and "the hash
is empty" are different facts.

### 4.1 When predictions vanished

```text
peak predicted mentions          13, at epoch 2
last epoch with any prediction   3
first zero-prediction epoch      4      (and every epoch thereafter)
final predicted mentions         0
```

The model was never good. It briefly emitted a handful of mentions in epochs 2-3
(2 true positives at epoch 2, 1 at epoch 3), then stopped predicting anything at
all from epoch 4 onward and stayed there for eight consecutive epochs.

### 4.2 The loss stops moving

Epochs 5-12 span a loss range of **1.9e-4**:

```text
0.01368798  0.01378138  0.01362625  0.01360618
0.01359775  0.01360213  0.01360450  0.01360332
```

Epoch 1's loss (0.01364423) is within that same band. The run therefore ended
where it started, having briefly departed and returned. The 100-sample rolling
loss in the progress log tells the same story from inside the epoch: it starts at
**1.970317** (essentially `ln 7 = 1.9459`, an untrained 7-way classifier), falls
to 0.0190 by the end of epoch 1, reaches 0.0142 at epoch 3, jumps to **0.0318**
during epoch 4, and then sits at 0.0318 ± 0.0005 for every remaining epoch.

## 5. Gold-Grid Round-Trip

```text
gold character spans -> atomic W2NER words -> target grid -> gold-grid decoder
                     -> reconstructed character spans and types
```

No model is involved anywhere in this path.

```text
split        examples   gold  reconstructed     tp   fp   fn  precision  recall     F1
train           33826  11720          11720  11720    0    0        1.0     1.0    1.0
validation       1045   1991           1991   1991    0    0        1.0     1.0    1.0
```

```text
failures by entity type   {}   (both splits)
representative failures   none
acceptance criterion      exact P = R = F1 = 1.0     MET
```

**The supervision signal is lossless.** The target builder and the decoder are
mutually inverse on every governed entity in both splits. `TARGET_DECODER_MISMATCH`
is ruled out by measurement, not by inspection.

A secondary confirmation falls out of this: the validation gold total the full
run scored against (1,991) is exactly what the round-trip reconstructs, so the
denominator in the failed metric was correct.

## 6. Grid Class Distribution

Measured from the governed targets, and cross-checked against the run's own
`grid_target_statistics.json` (`train_contracts` 33,826, `validation_contracts`
1,045, `label_count` 7, `max_atomic_words` 162 — all reproduced exactly).

```text
                                       train        validation
valid grid cells                  25,635,699         1,491,764
background cells (NONE)           25,591,359         1,485,126
positive cells                        44,340             6,638
positive / background               0.0017326         0.0044696
positive cell rate                    0.1730%           0.4450%
examples with zero positive cells 27,513 (81.34%)          0 (0%)
entities represented                  11,720             1,991
mean atomic words per example          24.14             33.76
max atomic words                         162               120
```

Per relation class:

```text
label              train    validation
NONE          25,591,359     1,485,126
NNW               32,620         4,647
THW:DIAGNOSIS      6,338         1,302
THW:MEDICATION     2,733           156
THW:SYMPTOM        2,649           533
THW:TEST_NAME          0             0
THW:TEST_RESULT        0             0
```

11,720 + 1,991 = **13,711**, matching the aligned-entity total established in
Audit 0038. Every aligned entity produces its expected target structure; the
mention-to-pattern breakdown accounts for all of them (899 single-word
`THW`-only, 3,182 two-word `NNW1+THW1`, the remainder multi-word).

### 6.1 The padded region is not in the loss

`_pad_grid` builds `padded_labels` and `padded_pair_mask` at `max_words` = 256,
but the executed loop passes `item.grid.labels` and `item.grid.pair_mask` — the
**unpadded** `n x n` grid. The 256x256 padding therefore never reaches the loss.
Padded cell counts are reported separately by the diagnostic for completeness and
are not folded into the valid-cell totals.

### 6.2 Two label classes have no training signal at all

`THW:TEST_NAME` and `THW:TEST_RESULT` occur **zero** times in train and zero
times in validation. Two of the five declared entity types, and two of the seven
classifier outputs, were never supervised. This is reported, not repaired.

### 6.3 The converged loss against an input-independent predictor

The absolute loss number 0.0136 looks small, which is exactly the trap. The
meaningful comparison is against the best predictor that **ignores its input
entirely** and emits the class marginal at every cell. Because the loss is a
per-example mean, that marginal is the example-weighted one, and the resulting
loss is its entropy:

```text
weighted marginal          NONE 0.9981910669   NNW 0.0013139097
                           THW:DIAGNOSIS 0.0001585728
                           THW:MEDICATION 0.0002740519
                           THW:SYMPTOM 0.0000623986
                           THW:TEST_NAME 0.0   THW:TEST_RESULT 0.0

best constant-predictor loss H(q)      0.014764124850679539
observed converged loss (epoch 12)     0.013603323943778346
ratio observed / constant              0.9213769242240074
improvement over constant predictor    7.86%
implied cross-entropy per positive cell  7.52 nats  (p_true = 5.4e-4)
```

After 405,912 backward passes and 50,748 optimizer steps, a 371-million-parameter
model finished **7.9% better than a constant that never looks at the input**.
Epoch 1 already stood at 92.4% of the same baseline.

## 7. Corpus Composition and Ordering

`GovernedW2NERContractSource.iter_contracts()` preserves deterministic file order
and the E4 loop does not shuffle, so how the corpus is laid out on disk is part
of what the optimizer actually experienced. It was measured rather than assumed.

```text
train (in file order)
  source           examples  entities  first row  types
  phoner_covid19     10,027         0          0  {}
  vimedner            5,796     8,987     10,027  {DIAGNOSIS 6338, SYMPTOM 2649}
  vimq                8,736       619     15,823  {MEDICATION 619}
  vietmed_ner         9,267     2,114     24,559  {MEDICATION 2114}

  longest consecutive zero-entity run   10,027, starting at row 0

validation (in file order)
  vimedner              906     1,835          0  {DIAGNOSIS 1302, SYMPTOM 533}
  vimq                  139       156        906  {MEDICATION 156}
```

Two facts follow, both visible in the progress log:

1. **Every epoch opens with 10,027 consecutive examples containing no entity** —
   about 1,253 consecutive optimizer steps whose only gradient signal is "predict
   NONE". The per-sample `current_loss` in the progress log is `0.0` or `~1e-6`
   from sample ~2,000 to sample ~10,000 in every epoch, then rises once
   `vimedner` begins.
2. **Validation covers only two of the four training sources.** `phoner_covid19`
   (10,027 examples) and `vietmed_ner` (9,267 examples) appear in train and are
   entirely absent from validation — 57% of the training set is drawn from
   distributions the metric never scores.

Neither is changed here. Both are recorded as evidence.

## 8. Best/Latest Checkpoint Probe

**BLOCKED — not executed.**

`checkpoints/best.pt` and `checkpoints/latest.pt` are not present in the local
artifact, and this milestone does not download them. Every quantity section E
asks for therefore has no value:

```text
predicted mention total / gold / tp / P / R / F1     NOT MEASURED
predictions by entity type                           NOT MEASURED
predicted grid labels by relation class              NOT MEASURED
background vs non-background label counts            NOT MEASURED
background logit mean and quantiles                  NOT MEASURED
strongest non-background logit quantiles             NOT MEASURED
gold-positive-cell predicted labels                  NOT MEASURED
gold-positive-cell background rate                   NOT MEASURED
decoder input positive relations / output mentions   NOT MEASURED
w2ner_head restoration from either checkpoint        NOT MEASURED
```

The probe is fully implemented (`CheckpointProbeReport`,
`inspect_checkpoint_payload`, `require_checkpoint`) and fails closed with the
exact SHA-256 to fetch. To unblock it, place the two checkpoints under
`local-artifacts/e4_phobert_w2ner_full_v1/checkpoints/` and re-run
`scripts/diagnose_e4_collapse.py`.

Nothing was substituted for this evidence. No logit distribution is estimated,
inferred or assumed anywhere in this audit.

## 9. Label and Loss Contract Audit

### 9.1 One label ordering, traced end to end

```text
id  label             builder writes         head output   CE target   decoder reads
 0  NONE              default fill                    0           0    (not a relation)
 1  NNW               labels[i][i+1]                  1           1    path edge
 2  THW:DIAGNOSIS     labels[end][start]              2           2    DIAGNOSIS
 3  THW:MEDICATION    labels[end][start]              3           3    MEDICATION
 4  THW:SYMPTOM       labels[end][start]              4           4    SYMPTOM
 5  THW:TEST_NAME     labels[end][start]              5           5    TEST_NAME
 6  THW:TEST_RESULT   labels[end][start]              6           6    TEST_RESULT
```

There is exactly one source: `W2NERLabelVocab.labels`. The head is built with
`relation_count = grid_target_statistics["label_count"]` (7); the loss is
`cross_entropy(logits.reshape(-1, item.label_count), ...)` where
`label_count = len(grid.vocab.labels)` (7); the decoder reads
`grid.vocab.thw_type(label_id)` off the same tuple. `trace_label_contract()`
round-trips every id through builder -> classifier index -> cross-entropy target
-> argmax -> decoder relation and finds **zero differences**.
`LABEL_MAPPING_MISMATCH` is ruled out.

### 9.2 What the executed loss does

```python
pair_mask = torch.tensor([item.grid.pair_mask], dtype=torch.bool, device=device)
labels    = torch.tensor([item.grid.labels],    dtype=torch.long, device=device)
logits    = head(word_embeddings, pair_mask)
loss      = nn.functional.cross_entropy(
    logits.reshape(-1, item.label_count), labels.reshape(-1))
```

```text
ignore_index                 NOT USED
class weights                NOT USED
padded cells in the loss     NO   (the unpadded n x n grid is passed)
triangular masking           NONE (build_w2ner_grid sets pair_mask all True over
                                   n x n; NNW lives at [i][i+1] in the upper
                                   triangle, THW at [tail][head] in the lower/
                                   diagonal, and both triangles are scored)
normalization                mean over all n*n cells, then mean over examples
background class id          0
positive relation ids        1, 2, 3, 4, 5, 6
learning rate schedule       NONE; 2e-05 held constant for all 50,748 steps
```

Because the mean is over every cell and no weighting offsets the 577:1 background
majority, **an all-background solution is a reachable stationary point of this
objective**, and §6.3 measures how close the run finished to it. Per-example
normalization compounds it: a 5-word example (25 cells) counts as much as a
162-word one (26,244 cells).

### 9.3 The decoder applies no threshold

`decode_argmax_relation_grid` takes a plain `argmax` with no cutoff.
`decode_w2ner_grid` is then called with `scores=None` and `threshold=0.0`, so
`score` is 1.0 and `1.0 < 0.0` is never true. No decoding policy can be
suppressing predictions. `DECODER_THRESHOLD_FAILURE` is ruled out statically,
without weights.

**The loss is not changed by this milestone.**

## 10. Verdict

```text
ROOT_CAUSE_NOT_YET_PROVEN
```

### Ruled out, with evidence

| Hypothesis | Evidence |
| --- | --- |
| `TARGET_DECODER_MISMATCH` | gold-grid round-trip exact P = R = F1 = 1.0 on 33,826 train and 1,045 validation examples, 13,711 entities, zero failures, no model in the loop |
| `LABEL_MAPPING_MISMATCH` | one `W2NERLabelVocab.labels` tuple is shared by builder, head size, cross-entropy target and decoder; every id round-trips |
| `DECODER_THRESHOLD_FAILURE` | the decoder applies no threshold at all — `score` 1.0, `threshold` 0.0 |

### Leading hypothesis — supported, not proven

`ALL_BACKGROUND_LOSS_COLLAPSE`:

* converged loss is 92.1% of the best **input-independent** predictor's loss, a
  7.9% improvement after 405,912 backward passes (§6.3);
* the loss is flat to within 1.9e-4 across epochs 5-12 and equals epoch 1's;
* the rolling loss starts at `ln 7` and returns to a fixed value after epoch 4;
* 0.173% positive cells, 577:1 background, with no class weighting and no
  `ignore_index` (§6, §9.2);
* 81.34% of training examples ask for nothing but background, and every epoch
  begins with 10,027 consecutive such examples (§7);
* 11 of 13 validation passes predicted exactly 0 mentions; the peak was 13.

### Why it is not returned as the verdict

The brief is explicit that `ALL_BACKGROUND_LOSS_COLLAPSE` must be supported by
**grid logits, positive-cell predictions and a passing gold-grid round-trip**.
The round-trip passes. The other two require the checkpoints, which are absent
(§8). `resolve_verdict()` enforces this gate in code, and a test asserts that a
zero prediction count alone can never produce the collapse verdict.

### Missing evidence, named

1. grid logit distributions from `best.pt` and `latest.pt`;
2. predicted labels at gold-positive cells, and the background rate there;
3. `w2ner_head` restoration from both checkpoints — so
   `CHECKPOINT_RESTORE_FAILURE` is neither confirmed nor ruled out.

## 11. Tiny-Overfit Colab Diagnostic

`notebooks/MedNorm_E4_TinyOverfit_Diagnostic.ipynb` and
`configs/training/phase2_e4_tiny_overfit_diagnostic.yaml`, backed by
`src/mednorm_vi/training/phase2/e4_tiny_overfit.py`.

Train on a handful of governed training examples and evaluate on **the same**
examples. A correctly wired W2NER pipeline must be able to memorize them.

```text
selected examples          12          (deterministic, single pass, no seed)
row indices                10027 10028 10029 10030 10037 10038 10039 10040
                           15862 15868 15894 15907
entities                   22          DIAGNOSIS 10, SYMPTOM 8, MEDICATION 4
required types covered     DIAGNOSIS, SYMPTOM, MEDICATION   (all)
atomic words per example    23 55 50 37 38 23 31 28 13 12 11 10
```

No single governed example carries all three types, so the selection reserves a
per-type quota and takes the earliest eligible example covering each. An example
whose entity does not align on the atomic surface is **skipped, never repaired** —
repairing it here would smuggle in the very failure this diagnostic detects.

`TEST_NAME` and `TEST_RESULT` are deliberately not required: the corpus contains
zero instances of either (§6.2), so requiring them would make the selection
unsatisfiable.

### Three metrics, never one

A background-everywhere model already scores ~0.998 grid cell accuracy on this
corpus, so a single accuracy number would call total failure a near-perfect
result. `TinyOverfitScore` reports `grid_cell_accuracy`,
`positive_cell_accuracy` and `background_cell_accuracy` separately, alongside
decoded exact precision / recall / F1.

### Interpretation

| observed | conclusion |
| --- | --- |
| exact train F1 -> ~1.0 | target/loss/decoder are coherent; the collapse is optimization + class imbalance on the real corpus |
| loss -> ~0 while exact train F1 = 0.0 | the loss is minimized without producing decodable mentions: a target/loss/decoder inconsistency |
| neither, within the epoch cap | inconclusive; must not be reported as a root cause |

### Safety

```text
authorization required          I_AUTHORIZE_E4_TINY_OVERFIT_DIAGNOSTIC
committed run flag              RUN_TINY_OVERFIT_DIAGNOSTIC = False
committed confirmation value    ""
stop rule                       exact F1 >= 0.95, or 200 epochs, whichever first
evaluation cadence              every 5 epochs, plus first and last
artifact directory              e4_tiny_overfit_diagnostic_v1   (separate)
full artifact as write target   REFUSED by assert_artifact_dir_is_not_protected
loss                            UNCHANGED from the full run
internal_test                   never opened
organizer inference             never run
output.zip                      never created
executed locally                NO
```

The loss is deliberately identical to the full run's: changing it inside the
diagnostic would destroy the comparison the experiment exists to make.

## 12. Prohibitions Observed

```text
local training                     NONE
internal_test access               NONE (refused by name in every entry point)
organizer inference                NONE
output.zip                         NOT CREATED
E4 retraining                      NOT PERFORMED
loss change                        NOT MADE
artifact modified/renamed/moved    NO   (read and hashed only)
architecture PDF                   UNCHANGED
committed audits                   UNCHANGED
protected E4 implementation paths  UNCHANGED (SHA-256 re-verified)
```

## 13. Tests and Static Checks

`tests/unit/test_e4_collapse_diagnosis.py` — 75 tests covering immutable artifact
hashes, epoch-history reconstruction (including `UNAVAILABLE` marking), the
gold-grid round-trip on both full splits, class-distribution accounting,
label-ID consistency, checkpoint-head restoration, the blocked probe, the verdict
gate, `internal_test` refusal, absence of clinical text in every payload, the
diagnostic notebook and config, and that no artifact or checkpoint is tracked in
Git. Nine protected E4 paths are asserted byte-identical by recomputed SHA-256.

No existing test was weakened. `tests/unit/test_notebooks.py` gains one
registration entry for the new notebook, which is how that file records every
notebook added after Audit 0017.

```text
env PYTHONPATH=src python -m pytest -q          1670 passed, 1 skipped, 2 failed
tests/unit/test_e4_collapse_diagnosis.py        75 passed
ruff check .                                    All checks passed
ruff check notebooks                            All checks passed
env PYTHONPATH=src python -m mypy               Success: no issues found in 266 source files
env PYTHONPATH=src python -m compileall -q src  clean
git diff --check                                clean
```

### 13.1 Two pre-existing failures, reported rather than hidden

```text
tests/unit/test_parallel_post_e4_readiness.py::test_count_parameters_is_programmatic_not_an_estimate
tests/unit/test_parallel_post_e4_readiness.py::test_s2_head_parameter_count_is_programmatic

  ModuleNotFoundError: No module named 'torch'
```

Both belong to Audit 0042 and both are **environmental**: `torch` is no longer
installed in this local environment (`pip show torch` -> not found), so
`build_l4_mlp()` and `build_s2_assertion_head()` cannot instantiate the modules
they count. Neither test, nor the code they exercise, is touched by this
milestone — `git status --short` reports both paths clean.

They were left alone rather than made to skip: gating them on a missing optional
dependency is a change to an Audit-0042 test and is out of scope here. Installing
`torch` restores both. Everything this milestone adds is pure Python and imports
no deep-learning framework, which is why all 75 new tests pass without it.

The 1 skip is the long-standing `pyarrow not installed locally` skip in
`tests/unit/test_vietmed_adapter.py`.

## 14. Changed Files

```text
A  docs/audits/0043-e4-post-training-collapse-diagnosis.md
M  docs/audits/README.md
A  src/mednorm_vi/training/phase2/e4_collapse_diagnosis.py
A  src/mednorm_vi/training/phase2/e4_tiny_overfit.py
A  scripts/diagnose_e4_collapse.py
A  notebooks/MedNorm_E4_TinyOverfit_Diagnostic.ipynb
A  configs/training/phase2_e4_tiny_overfit_diagnostic.yaml
A  tests/unit/test_e4_collapse_diagnosis.py
M  tests/unit/test_notebooks.py
```

## 15. What Comes Next (not done here)

1. Download `best.pt` and `latest.pt` into the artifact directory and run the
   probe. That single step either confirms `ALL_BACKGROUND_LOSS_COLLAPSE` or
   redirects the whole diagnosis.
2. Run the tiny-overfit diagnostic on Colab under explicit authorization.
3. Only then decide on a fix. The evidence already assembled points at class
   imbalance (§6), corpus ordering (§7) and loss normalization (§9.2), but no
   change is justified until the probe has spoken.

## 16. Safe-to-Commit Verdict

Safe to commit after review. Required tests and static checks pass, no protected
architecture, audit, evaluator, corpus or E4 implementation file changed,
governed hashes are unchanged, no existing test was weakened, and no weight,
checkpoint, corpus, cache or archive artifact is staged.

```bash
git add \
  docs/audits/0043-e4-post-training-collapse-diagnosis.md \
  docs/audits/README.md \
  src/mednorm_vi/training/phase2/e4_collapse_diagnosis.py \
  src/mednorm_vi/training/phase2/e4_tiny_overfit.py \
  scripts/diagnose_e4_collapse.py \
  notebooks/MedNorm_E4_TinyOverfit_Diagnostic.ipynb \
  configs/training/phase2_e4_tiny_overfit_diagnostic.yaml \
  tests/unit/test_e4_collapse_diagnosis.py \
  tests/unit/test_notebooks.py

git commit -m "feat: diagnose E4 post-training collapse with bounded evidence"
git push origin main
```
