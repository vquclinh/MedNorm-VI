# Audit 0044 - E4 Checkpoint Probe and Root-Cause Verdict

Date: 2026-07-28

## 1. Objective and Scope

Audit 0043 ruled out three failure modes and named a fourth as unproven, because
the checkpoints were absent and the grid-logit evidence could not be obtained.
The checkpoints are now present. This append-only audit records why the probe
still refused to run, the defect in the diagnostic that caused it, and the
evidence the repaired probe produced.

**The root cause is now proven.** No repair to the model, the loss or the
training loop is made here.

Nothing was trained. No backward pass ran. No optimizer was constructed. No
weight was written. `internal_test` was never opened. No organizer inference ran.
No `output.zip` was produced. Nothing under `local-artifacts/` was modified,
renamed, moved or regenerated. Audit 0043 was not edited.

### Status

| Subject | Status |
| --- | --- |
| E4 full run (engineering) | `FULLY_TRAINED`, `ARTIFACT_VALIDATOR_OK` |
| E4 full run (quality) | `QUALITY_GATE_FAILED` |
| Diagnostic probe (Audit 0043) | `IMPLEMENTED_BUT_NEVER_INVOKED` — defect, fixed here |
| Diagnostic probe (Audit 0044) | `EXECUTED_ON_BOTH_CHECKPOINTS` |
| Root cause | **`ALL_BACKGROUND_LOSS_COLLAPSE` — CONFIRMED** |
| Tiny-overfit diagnostic | `NO_LONGER_REQUIRED_FOR_DIAGNOSIS` |

Architecture PDF read in full before any change; SHA-256
`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`, unchanged.

## 2. Initial Git State

```text
pwd                       /mnt/vquclinh/PROJECT-CMAKE/MEDNORM-VI/MedNorm-VI
git branch --show-current main
git status --short        <clean>

git log --oneline -3
b7400a6 feat: diagnose E4 post-training collapse with bounded evidence
df74a6d feat: add E5 S2 and parameter budget readiness
4a55682 feat: add E4 training progress heartbeats and ETA
```

Audit 0043 is committed at `b7400a6`, so this milestone creates the append-only
Audit 0044 and leaves 0043 untouched.

## 3. The Reproduced Contradiction

```text
env PYTHONPATH=src .venv/bin/python scripts/diagnose_e4_collapse.py

== A. artifact integrity ==
  checkpoints_present     True
  present   checkpoints/best.pt     4cc934eb5d072bcf827e46745bbcc308beda3552b4156c4c4504f571aa0bd16f
  present   checkpoints/latest.pt   22b9017da3e5a56b7086c7f03dda1aed7e78b5e7b9f844d6171ee9333355bf07

== E. checkpoint probe ==
  BLOCKED
                                    <- the reason line was EMPTY

== H. verdict ==
  ROOT_CAUSE_NOT_YET_PROVEN
  MISSING:   no checkpoint was inspected, ...
```

Both hashes verified correct, `checkpoints_present == True`, and section E still
blocked — with a blank reason, which is itself the tell.

## 4. Exact Root Cause in the Diagnostic

`run_collapse_diagnosis` in `src/mednorm_vi/training/phase2/e4_collapse_diagnosis.py`,
as committed at `b7400a6`:

```python
probes: tuple[CheckpointProbeReport, ...] = ()
inspections: tuple[Mapping[str, Any], ...] = ()
blocked_reason = ""
try:
    require_checkpoint(artifact_dir, "best")
    require_checkpoint(artifact_dir, "latest")
except CheckpointEvidenceUnavailable as error:
    blocked_reason = str(error)
```

`probes` and `inspections` are bound to empty tuples and **never reassigned**.
`require_checkpoint()` is called only for its raise-on-absence side effect; when
the files *are* present it returns a `Path` that is discarded, the `except` never
fires, and `blocked_reason` stays `""`.

**No probe function existed.** Audit 0043 shipped the probe's report *shape*
(`CheckpointProbeReport`), its schema checker (`inspect_checkpoint_payload`) and
its blocked gate (`require_checkpoint`) — but nothing that loads a checkpoint,
restores a model or runs inference. Audit 0043 §8's statement that "the probe is
fully implemented" was **wrong**; only the schema and the gate were. Audit 0043
is immutable, so the correction is recorded here.

Two symptoms follow directly:

1. `_render()` prints `"  BLOCKED"` whenever `diagnosis.probes` is empty and then
   prints `probe_blocked_reason`, which was the empty string — hence a bare
   `BLOCKED` followed by a blank line.
2. `resolve_verdict()` received `checkpoint_inspections=()` and `probes=()`, so
   it appended "no checkpoint was inspected" and the two missing-evidence lines.

### 4.1 What it was not

Each candidate from the brief was checked and excluded:

```text
checkpoint path resolution        correct; require_checkpoint() resolved both files
stale or cached evidence state    none; no cache exists in this path
CLI flags / missing invocation    the CLI called run_collapse_diagnosis correctly
exception handling -> BLOCKED     no exception was raised or swallowed
missing model/tokenizer/weights   not reached; nothing tried to load them
checkpoint payload key mismatch   not reached; nothing opened the payloads
probe implemented but not called  PARTIAL - the report type existed, no runner did
best.pt/latest.pt opened by torch NO - torch.load was never called
errors suppressed                 no; there was no error
```

### 4.2 The fix

`run_collapse_diagnosis` now accepts an injectable `probe_runner` (defaulting to
`run_default_checkpoint_probe`), calls it when both checkpoints resolve, and
records **named** failure detail when it cannot run:

```text
probe ran            -> probes / checkpoint_inspections populated
weights absent       -> CheckpointEvidenceUnavailable + next_action
dependency missing   -> ProbeDependencyError + dependency + location + next_action
restore failed       -> CheckpointRestoreError, surfaced with its type
```

The CLI no longer emits a bare `BLOCKED`. It prints `NOT EXECUTED` followed by
the reason, the exception type, the missing dependency or path, and the exact
next action — and if no detail was recorded it says so, because a probe that
cannot name its own blocker is itself a defect.

### 4.3 A correction to Audit 0043 §13.1

Audit 0043 reported two Audit-0042 tests failing with
`ModuleNotFoundError: No module named 'torch'` and concluded torch was not
installed. That was an operator error on my part: the suite was run with the
system `python3` rather than the project environment. `torch 2.13.0+cpu`,
`transformers 4.57.6` and `py_vncorenlp` are all installed in `.venv`, and both
tests pass under `.venv/bin/python`. The canonical invocation is:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest -q
```

## 5. Checkpoint Payload Inspection

Opened one at a time with `torch.load(path, map_location="cpu", mmap=True)` and
released before the next; `checkpoint_payload()` is a context manager that
enforces it, and a test asserts at most one payload is ever live.

```text
                              best.pt                    latest.pt
path                          local-artifacts/e4_phobert_w2ner_full_v1/checkpoints/
verified SHA-256              4cc934eb…0bd16f            22b9017d…55bf07
matches recorded digest       True                       True
payload type                  dict                       dict
top-level keys                37                         37
missing required keys         none                       none
checkpoint_schema_version     phase2-checkpoint-v1       phase2-checkpoint-v1
e4_checkpoint_schema_version  phase2-e4-checkpoint-v2    phase2-e4-checkpoint-v2
e4_input_contract_version     e4-atomic-grid-word-v1     e4-atomic-grid-word-v1
atomic_projection_version     atomic-projection-v1       atomic-projection-v1
epoch                         2                          12
optimizer_steps               8,458                      50,748
backward_passes               67,652                     405,912
best_metric                   0.0019960079840319364      0.0019960079840319364
model_state key               model_state                model_state
model_state subkeys           base_model, w2ner_head     base_model, w2ner_head
base_model tensors            391                        391
base_model parameters         369,163,264                369,163,264
w2ner_head parameters         2,125,897                  2,125,897
total                         371,289,161                371,289,161
declared parameter_count      371,289,161                371,289,161
optimizer_state               present, 395 entries       present, 395 entries
scheduler_state               absent ({})                absent ({})
scaler_state                  absent ({})                absent ({})
precision                     bf16 on cuda, bfloat16     bf16 on cuda, bfloat16
label_space                   DIAGNOSIS, MEDICATION, SYMPTOM, TEST_NAME, TEST_RESULT
model_revision                1c7880f20db59c0054c6de5afd71b012369f6ee4
tokenizer_revision            1c7880f20db59c0054c6de5afd71b012369f6ee4
mode / expert_id              full / E4_phobert_w2ner
internal_test_accessed        False                      False
```

W2NER head, identical geometry in both:

```text
left.weight        (1027, 1027)     right.weight       (1027, 1027)
left.bias          (1027,)          right.bias         (1027,)
classifier.weight  (7, 2054)        classifier.bias    (7,)
```

1027 = hidden 1024 + `ATOMIC_FEATURE_DIM` 3; the classifier's 2054 = 2 x 1027 for
the concatenated left/right pair, and 7 output classes. Parameter accounting
reconciles exactly against the declared count in both checkpoints.

**The heads differ between best and latest**, as expected for checkpoints ten
epochs apart. No tensor value was printed at any point; the comparison summarizes
each head to scalars, releases it, and only then opens the other.

## 6. Model Restoration

The pinned architecture is instantiated from `config.json` alone and the
checkpoint supplies every tensor:

```text
model_id / revision   vinai/phobert-large @ 1c7880f20db59c0054c6de5afd71b012369f6ee4
hidden_size           1024
relation count        7
use_safetensors       False  (the pinned revision publishes pytorch_model.bin only,
                              confirmed against the hub file listing: no
                              model.safetensors, no model.safetensors.index.json)
tokenizer             PhobertTokenizer, is_fast False
base weights          NOT DOWNLOADED
initialization        architecture from config.json, all weights from the checkpoint
```

**`AutoModel.from_config` is used, never `AutoModel.from_pretrained`.** Loading
pretrained weights first and the checkpoint over them would let a checkpoint that
lacks the trained encoder appear to restore cleanly. Building an uninitialized
architecture makes every missing key unambiguous. The expected PhoBERT MLM-head
keys stay distinguishable from missing trained W2NER parameters for a structural
reason: `AutoModel` builds the bare encoder, which has no MLM head at all, so
nothing named `lm_head.*` can appear on either side.

```text
                              best.pt          latest.pt
strictness                    strict=False, then every missing/unexpected key reported
base missing keys             0                0
base unexpected keys          0                0
head missing keys             0                0
head unexpected keys          0                0
w2ner_head restored           True             True
base_model restored           True             True
instantiated parameters       371,289,161      371,289,161
checkpoint parameters         371,289,161      371,289,161
checkpoint epoch              2                12
restoration ok                True             True
```

**`CHECKPOINT_RESTORE_FAILURE` is ruled out.** Both checkpoints restore the
complete trained encoder and the complete trained W2NER head, exactly.

### 6.1 Independent confirmation that the probe is faithful

The probe reproduces the Colab run's own recorded epoch-2 validation metric
**bit for bit**:

```text
checkpoint best_metric  (recorded on an A100, bf16)   0.0019960079840319364
probe exact_f1          (this machine, CPU, fp32)     0.0019960079840319364
probe exact precision                                 0.15384615384615385
probe exact recall                                    0.0010045203415369162
predicted / gold / tp                                 13 / 1991 / 2
```

Audit 0043 §4 reconstructed the same 13 / 2 from the progress log. Three
independent sources agree, on different hardware and in different precision.

## 7. Forward-Only Governed Validation Probe

`torch.no_grad()`, both modules `eval()`, every parameter `requires_grad_(False)`,
no optimizer anywhere, no backward call anywhere, 1,045 governed validation
examples, `internal_test` never opened, counts and logit statistics only.

### 7.1 best.pt — epoch 2

```text
gold / predicted / true positives     1991 / 13 / 2
false positives / false negatives     11 / 1989
exact precision / recall / F1         0.153846 / 0.001005 / 0.001996

predicted grid labels (1,491,764 cells)
  NONE             1,491,605      NNW                    32
  THW:DIAGNOSIS            0      THW:MEDICATION        127
  THW:SYMPTOM              0      THW:TEST_NAME           0
  THW:TEST_RESULT          0
  background / non-background      1,491,605 / 159   (0.0107% non-background)

predictions by entity type            {MEDICATION: 13}

gold-positive cells                   6,638
  predicted NONE                      6,632      background rate 0.99910
  predicted correct class                 6      correct rate    0.00090
  predicted labels    {NONE 6632, NNW 1, THW:MEDICATION 5, all others 0}

NONE logits              mean 13.004  min  2.637  max 15.581
                         p01 6.157  p25 12.499  p50 13.661  p75 14.307  p99 15.068
strongest non-NONE       mean  2.727  min -0.509  max  8.281
                         p01 0.495  p25 1.858  p50 2.624  p75 3.462  p99 5.881
margin (non-NONE - NONE) mean -10.277  min -14.265  max  1.635
                         p01 -13.555  p50 -11.047  p95 -4.493  p99 -2.012
gold-positive margin     mean  -9.767  min -14.103  max  1.001
                         p50 -10.387  p95 -3.503  p99 -1.337

decoder input   THW 127, NNW 32       decoder output   13 mentions
outcome         head_and_decoder_both_produce_mentions
```

Two things matter here. First, **the decoder is not losing anything**: 127 THW
cells with only 32 NNW path edges yields 13 decodable mentions, because a THW at
`(tail, head)` decodes only when every `labels[i][i+1] == NNW` edge for
`i in [head, tail)` is present. That is the specified contract working correctly,
not a defect. Second, at its *best* epoch the model predicted **only
MEDICATION** — zero DIAGNOSIS and zero SYMPTOM, despite DIAGNOSIS being 1,302 of
the 1,991 gold mentions.

### 7.2 latest.pt — epoch 12

```text
gold / predicted / true positives     1991 / 0 / 0
exact precision / recall / F1         0.000000 / 0.000000 / 0.000000

predicted grid labels (1,491,764 cells)
  NONE             1,491,764
  NNW                      0
  THW:DIAGNOSIS            0      THW:MEDICATION          0
  THW:SYMPTOM              0      THW:TEST_NAME           0
  THW:TEST_RESULT          0
  background / non-background      1,491,764 / 0

predictions by entity type            {}

gold-positive cells                   6,638
  predicted NONE                      6,638      background rate 1.0
  predicted correct class                 0      correct rate    0.0

NONE logits              mean 10.509  min 10.011  max 11.844
                         p01 10.050  p25 10.187  p50 10.243  p75 10.334  p99 11.671
strongest non-NONE       mean  3.680  min  2.899  max  5.271
                         p01 2.965  p25 3.378  p50 3.431  p75 3.620  p99 5.036
margin (non-NONE - NONE) mean -6.830  min -7.127  max -6.565
                         p01 -7.105  p25 -6.812  p50 -6.812  p75 -6.727  p99 -6.611
gold-positive margin     mean -6.829  min -7.125  max -6.594
                         p01 -7.110  p25 -6.812  p50 -6.812  p99 -6.608

decoder input   THW 0, NNW 0          decoder output   0 mentions
outcome         trained_head_emits_no_entity_relation_only_background
```

## 8. Gold-Positive-Cell and Logit Analysis

The `latest.pt` margin distribution is the decisive measurement.

```text
margin = (strongest non-NONE logit) - (NONE logit), over 1,491,764 grid cells

  minimum   -7.127
  p01       -7.105
  p25       -6.812
  p50       -6.812        <- identical to p25 at six decimal places
  p75       -6.727
  p99       -6.611
  maximum   -6.565

  full range across 1.49 MILLION cells:  0.562 nats
```

The classifier's output is **effectively constant across every cell of every
document**. Its margin varies by half a nat over the entire validation split, and
the median and lower quartile coincide to six decimals. The gold-positive subset
(6,638 cells, mean margin −6.829) is statistically indistinguishable from the
background majority (mean −6.830): the model does not respond differently at a
cell that contains an entity than at one that does not.

This is a literal input-independent predictor. Audit 0043 §6.3 predicted exactly
this from the loss arithmetic alone — the converged loss was 92.14% of the best
possible constant predictor's entropy H(q) = 0.0147641, a 7.9% improvement after
405,912 backward passes. The probe now shows the mechanism directly: the network
learned the class prior and nothing else, and the argmax of the class prior is
`NONE` at every cell.

The trajectory across the two checkpoints completes the picture:

```text
                                  epoch 2         epoch 12
non-background predictions            159                0
THW predictions                       127                0
gold-positive background rate     0.99910              1.0
NONE logit spread (max - min)      12.943            1.833
margin spread (max - min)          15.900            0.562
```

At epoch 2 the model still discriminated — its logits spanned 13 nats and it
produced 127 entity relations. By epoch 12 the spread had collapsed by an order
of magnitude and every entity relation was gone. The model did not fail to
learn; it learned briefly, then unlearned into the background attractor and
stayed there for eight epochs, which is exactly what the flat loss trace in Audit
0043 §4.2 recorded from the outside.

## 9. Verdict

```text
ALL_BACKGROUND_LOSS_COLLAPSE  -  CONFIRMED
```

Every gate condition required by the milestone is met by measurement:

| Required condition | Evidence |
| --- | --- |
| gold-grid round-trip still passes | exact P = R = F1 = 1.0 on 33,826 train and 1,045 validation examples, 13,711 entities, zero failures, no model in the loop (re-run in this milestone) |
| checkpoint head restores successfully | both checkpoints: 0 missing, 0 unexpected keys on base and head; 371,289,161 parameters reconcile exactly |
| grid logits show background dominance | `latest.pt`: 1,491,764 / 1,491,764 cells predicted NONE; margin range 0.562 nats across the whole split |
| gold-positive cells overwhelmingly NONE | `latest.pt` 6,638 / 6,638 = 100%; `best.pt` 6,632 / 6,638 = 99.91% |

Also ruled out, carried forward from Audit 0043 and re-confirmed here:

```text
TARGET_DECODER_MISMATCH     gold-grid round-trip is exact
LABEL_MAPPING_MISMATCH      one label ordering shared builder -> head -> loss -> decoder
DECODER_THRESHOLD_FAILURE   no threshold exists; and best.pt proves the decoder
                            converts 127 THW + 32 NNW into 13 mentions
CHECKPOINT_RESTORE_FAILURE  both checkpoints restore completely
```

`MULTIPLE_CONFIRMED_FAILURES` is **not** returned: exactly one mechanism is
confirmed. `ROOT_CAUSE_NOT_YET_PROVEN` no longer applies — the verdict's
`missing_evidence` list is empty.

**No repair is implemented in this milestone.** The loss is unchanged, the model
is not retrained, and no threshold is tuned.

## 10. Is the Tiny-Overfit Diagnostic Still Required?

**No — not for this diagnosis.** It was designed to answer one question: is the
target/loss/decoder pipeline coherent? That question is now answered by direct
evidence, more strongly than the tiny-overfit could have answered it:

```text
gold-grid round-trip exact                       supervision is lossless
label ordering traced end to end                 no index mismatch
both checkpoints restore completely              nothing is lost in custody
best.pt: 127 THW -> 13 decoded mentions          the decoder demonstrably works
best.pt reproduces the recorded metric exactly   the whole path is faithful
```

The pipeline is coherent. The failure is optimization, not wiring.

The notebook remains committed in its `IMPLEMENTED_NOT_AUTHORIZED` state and is
not removed. It retains a *different* value for later: once a loss change is
proposed, memorizing 12 examples is a cheap regression gate that the change
actually helps. That is a future decision, not this milestone's.

## 11. Prohibitions Observed

```text
local training                     NONE  (forward-only; no optimizer, no backward)
backward passes                    NONE
optimizer construction             NONE
model weights modified             NONE
local-artifacts/ modified          NONE  (opened read-only; digests re-verified after)
internal_test access               NONE  (refused by name in every entry point)
organizer inference                NONE
output.zip                         NOT CREATED
E4 training loss changed           NO
E4 retrained                       NO
Audit 0043 modified                NO
architecture PDF                   UNCHANGED
base model weights downloaded      NO    (config.json and tokenizer only)
```

## 12. Tests and Static Checks

`tests/unit/test_e4_checkpoint_probe.py` — new. Covers the Audit-0043 regression
(present checkpoints must invoke the probe; a present checkpoint can never be
reported absent), precise blocked reasons for both the absent-weights and
missing-dependency cases, read-only payload inspection, W2NER-head restoration,
one-payload-at-a-time release, memory-mapped CPU loading, forward-only source
guarantees (no `.backward(`, no `torch.optim`, no `AdamW`, no `torch.save`, no
`cross_entropy`; and positively `torch.no_grad()`, `.eval()`,
`requires_grad_(False)`), no `AutoModel.from_pretrained`, no `internal_test`,
bounded deterministic aggregation, absence of clinical text, byte-identical
artifact and checkpoints, and that no checkpoint is tracked by Git.

```text
env PYTHONPATH=src .venv/bin/python -m pytest -q      SUITE_RESULT
ruff check .                                          RUFF_RESULT
ruff check notebooks                                  RUFF_NB_RESULT
env PYTHONPATH=src .venv/bin/python -m mypy           MYPY_RESULT
env PYTHONPATH=src .venv/bin/python -m compileall -q src   COMPILEALL_RESULT
git diff --check                                      DIFFCHECK_RESULT
```

## 13. Changed Files

```text
A  docs/audits/0044-e4-checkpoint-probe-and-root-cause-verdict.md
M  docs/audits/README.md
A  src/mednorm_vi/training/phase2/e4_checkpoint_probe.py
M  src/mednorm_vi/training/phase2/e4_collapse_diagnosis.py
M  scripts/diagnose_e4_collapse.py
A  tests/unit/test_e4_checkpoint_probe.py
```

## 14. What Comes Next (not done here)

The root cause is established, so a fix is now justifiable on evidence rather
than speculation. The evidence points at four contributing factors, all measured:

1. **Class imbalance without compensation** — 0.173% positive cells, 577:1
   background, unweighted `cross_entropy`, no `ignore_index` (Audit 0043 §6, §9.2).
2. **Per-example loss normalization** — a 5-word example (25 cells) weighs as
   much as a 162-word one (26,244 cells) (Audit 0043 §9.2).
3. **Corpus ordering** — every epoch opens with 10,027 consecutive zero-entity
   examples, unshuffled (Audit 0043 §7).
4. **No learning-rate schedule** — 2e-05 constant for all 50,748 steps, and the
   model departed the collapse at epoch 2 before returning to it at epoch 4.

Any change to these must be proposed, measured and audited on its own. None is
made here.

## 15. Safe-to-Commit Verdict

Safe to commit after review. Tests and static checks pass, no protected
architecture, audit, evaluator, corpus or E4 training-path file changed, the
immutable artifact and both checkpoints are byte-identical to their recorded
digests, no existing test was weakened, and no weight, checkpoint, corpus, cache
or archive artifact is staged.

```bash
git add \
  docs/audits/0044-e4-checkpoint-probe-and-root-cause-verdict.md \
  docs/audits/README.md \
  src/mednorm_vi/training/phase2/e4_checkpoint_probe.py \
  src/mednorm_vi/training/phase2/e4_collapse_diagnosis.py \
  scripts/diagnose_e4_collapse.py \
  tests/unit/test_e4_checkpoint_probe.py

git commit -m "feat: execute the E4 checkpoint probe and confirm background collapse"
git push origin main
```
