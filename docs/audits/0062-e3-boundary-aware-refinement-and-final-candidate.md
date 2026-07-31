# Audit 0062: E3 Boundary-Aware Refinement and the Final Candidate

Milestone: 4C - Boundary-aware E3 refinement, governed model selection, final leaderboard candidate
Date: 2026-07-31
Verdict: `E3_REFINEMENT_READY_FOR_LEADERBOARD`

## 1. Initial Git State

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `2394565 feat: govern model acquisition and resolve dotted ICD output` |
| Working tree | clean |
| Staged files | none |
| `origin/main...HEAD` | `0 0` |
| Audit 0061 committed | yes |
| RAM / disk / GPU / containers | 14 GiB total, 7.1 GiB available / 24 G free / RTX 4060 Laptop 8,188 MiB (13 MiB used) / 0 |

The architecture PDF is unmodified (`0d5eaa20…81e09b`) and was read in full at the start
of this milestone.

## 2. Exact Starting Checkpoint

| Field | Value |
| --- | --- |
| Path | `checkpoint/s1_mention_full_training_v1/best.pt` |
| SHA-256 | `a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c` |
| Size | 1,615,513,303 bytes |
| Base model | `demdecuong/vihealthbert-base-word` @ `f89e80b461e86f9cfc1c84019bd819830c24b6c5` |
| Architecture | encoder + dropout(0.1) + `Linear(768, 5)` multi-label token head |
| Parameters | 135,002,117 (recounted from the state dict this milestone) |

The digest was re-verified before and after every run in this milestone and never
changed. Nothing in this milestone writes to it.

## 3. Reproduced Baseline

The brief requires the existing checkpoint's validation evaluation to be reproduced
through the current evaluation code before any training, within an explicitly justified
tolerance. **No tolerance is needed: the reproduction is exact.**

Governed validation split `ed7cdd2d…f68f103`, 1,045 examples, 1,991 gold entities.

```text
micro span+type   P=0.6270  R=0.5344  F1=0.5770   TP=1064  FP=633  FN=927
  DIAGNOSIS       P=0.6607  R=0.5146  F1=0.5786
  SYMPTOM         P=0.5520  R=0.5572  F1=0.5546
  MEDICATION      P=0.7578  R=0.6218  F1=0.6831
  TEST_NAME       0 gold, 3 FP        TEST_RESULT  0 gold, 14 FP

error categories  exact_match 1064 | missed 537 | spurious 242
                  right_boundary 216 | left_boundary 111 | both_boundary 37 | wrong_type 26
```

Every figure matches Audit 0061 digit for digit, including all eight error categories.
The E3 span cache is **byte-identical** to the Audit-0061 run (`762139fc…3bd075`), so the
forward pass is bit-reproducible on this machine.

One benign difference is recorded rather than glossed: in 1 of 1,045 documents the
internal `hypothesis_id` ordinal differed (`-0003` vs `-0001`). Spans, types, texts,
scores and expert IDs were identical in every row of every document, and `hypothesis_id`
is internal — it is used to key offered-set lookups within a run and never reaches the
organizer output. It is an ID-assignment ordering artifact, not a content difference.

## 4. Error Analysis

### 4.1 The label-projection ceiling — is the architecture the limit?

Two rules bound what E3 can emit: supervision labels a subtoken positive when its
ORIGINAL character span **overlaps** a gold entity, and decoding turns a maximal run of
consecutive positive subtokens into one mention. Every subtoken inherits the span of its
segmented *word*, so a gold entity that starts or ends inside a word gets a supervision
target WIDER than gold — an error no amount of training can remove.

Running both rules on GOLD labels (a model with zero classification error) measures that
ceiling exactly:

| | |
| --- | ---: |
| Gold entities | 1,991 |
| Exactly reachable | **1,915 (96.18%)** |
| Unreachable | 76 (3.82%) |
| **Oracle micro span+type** | **P=0.9810 R=0.9618 F1=0.9713** |
| Dropped by truncation at `max_length=256` | **0** |

All 76 unreachable entities are adjacent same-type golds that merge into one run, which
is a *decoding* limit (the head has no B/I distinction), not a labelling one. Truncation
and chunk-boundary loss contribute nothing at all.

**The architecture is not the ceiling.** Baseline 0.5770 against an achievable 0.9713
leaves **39.4 F1 points that are model capability**. That is what justifies spending this
milestone on training rather than on plumbing.

### 4.2 Direction of the residual error

`right_boundary 216 / left_boundary 111` says a boundary is wrong but not *which way*,
and the two imply opposite fixes. Measured against the widest overlapping same-type
prediction:

| Direction | Count | Share |
| --- | ---: | ---: |
| too narrow (right) | 142 | 36.1% |
| too narrow (left) | 98 | 24.9% |
| too wide (right) | 74 | 18.8% |
| too wide (left) | 46 | 11.7% |
| too narrow (both) | 29 | 7.4% |
| too wide (both) | 4 | 1.0% |
| **TOTAL too narrow** | **269** | **68.4%** |
| **TOTAL too wide** | **124** | 31.6% |

The model **under-fires by 2.2 : 1**. Note this is the opposite of what the word-snap
oracle produces — word snapping can only make spans *wider* — so the narrowing is model
behaviour, not an alignment artifact.

### 4.3 Where the missed entities are lost

```text
E3 proposed it EXACTLY, L4 dropped it          0
E3 proposed an overlapping span, L4 dropped    0
E3 never proposed anything overlapping       534
```

**L4 discards nothing.** Every missed entity is missing because E3 never fired. The
resolver is not the bottleneck and tuning it would be wasted effort.

### 4.4 Length, punctuation, repetition, document length

| Gold length | gold | exact | boundary | missed | exact % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 word | 132 | 71 | 19 | 42 | 53.8% |
| 2-3 words | 1,065 | 616 | 151 | 298 | 57.8% |
| 4-7 words | 752 | 374 | 201 | 177 | 49.7% |
| 8+ words | 42 | 3 | 22 | 17 | **7.1%** |

Long spans collapse, exactly as "runs break early" predicts: the more tokens a run needs,
the more chances it has to break.

* **Punctuation proximity: 0 of 1,991** gold entities are adjacent to punctuation, so the
  annotation convention already excludes it and no punctuation-aware intervention is
  warranted. (Stated because the brief asks for the distribution — the answer is that
  this category is empty, not that it was skipped.)
* **Repeated surfaces**: 282 gold entities repeat a surface within their document; 78 of
  those are missed entirely (27.7%, close to the 26.8% overall miss rate) — repetition is
  not a distinct failure mode.
* **Document length**: 1,981 of 1,991 gold entities live in documents under 500
  characters. There is no long-document population to analyse and no chunking effect.

### 4.5 Per-type imbalance

| Type | gold | missed | spurious |
| --- | ---: | ---: | ---: |
| DIAGNOSIS | 1,302 | 373 | 101 |
| SYMPTOM | 533 | 135 | 126 |
| MEDICATION | 156 | 26 | 3 |
| TEST_NAME | 0 | 0 | 3 |
| TEST_RESULT | 0 | 0 | 14 |

### 4.6 A split-composition mismatch, recorded as a risk

| Source | train (supervised) | validation |
| --- | ---: | ---: |
| vietmed_ner | 38.9% | **0%** |
| vimq | 36.7% | 13.3% |
| vimedner | 24.4% | **86.7%** |
| MEDICATION share of entities | 23.3% | 7.8% |

The governed splits are grouped by source family to prevent leakage (spec §15.2), and the
consequence is that validation is **not** iid with training: it is dominated by a source
that is a quarter of the training signal.

This is deliberately **not** acted on. Reweighting training toward `vimedner` because
validation is 86.7% `vimedner` would be fitting the validation composition, and nothing
is known about the composition of the organizer's test set — the brief forbids using
public organizer input for training or model selection, so that question cannot be
answered here. It is carried as a risk in §14, not converted into a recipe.

### 4.7 Does the loss under-weight the positive class? Yes, provably.

The training loss (unchanged from the original run) is:

```python
alpha_t = focal_alpha * labels + (1.0 - focal_alpha) * (1.0 - labels)
per_element = alpha_t * (1.0 - p_t).pow(focal_gamma) * bce
```

With `focal_alpha = 0.25`, **positive tokens carry weight 0.25 and negatives 0.75**.
Positives are 8.26% of supervised tokens, so an already 11:1 imbalance is multiplied by a
further 3:1 against the rare class.

That is a mechanistic explanation for precisely what §4.2 and §4.3 measured — runs that
stop early and 534 entities that never fire — and it identifies the smallest possible
intervention: one scalar.

## 5. Governed Threshold Calibration (diagnostic)

Brief §2 permits threshold selection on the governed validation split.
`e3_decision_threshold` is an existing governed runtime setting, so the sweep runs the
same checkpoint through the same L1→L4 path with only that value changed.

| threshold | P | R | F1 | ΔF1 | ΔP | boundary | missed | predictions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.5913 | 0.6067 | 0.5989 | **+2.19** | −3.57 | 399 | 327 | 2,043 |
| 0.30 | 0.5513 | 0.6288 | 0.5875 | +1.05 | −7.57 | 408 | 271 | 2,271 |
| 0.35 | 0.5747 | 0.6223 | 0.5975 | +2.05 | −5.23 | 407 | 290 | 2,156 |
| 0.40 | 0.5959 | 0.5977 | 0.5968 | +1.98 | −3.11 | 400 | 357 | 1,997 |
| 0.45 | 0.6097 | 0.5665 | 0.5873 | +1.03 | −1.73 | 382 | 448 | 1,850 |
| **0.50 (shipped)** | **0.6270** | 0.5344 | 0.5770 | — | — | 364 | 537 | 1,697 |
| 0.60 | 0.6157 | 0.4264 | 0.5039 | −7.31 | −1.13 | 361 | 769 | 1,379 |

Three conclusions, all of which shaped the recipes:

1. The model **does** under-fire — recall rises monotonically as the threshold falls.
2. **Every threshold that gains F1 loses precision**, and the gate forbids any precision
   decrease. A threshold change alone therefore cannot pass, because it slides along the
   existing ROC instead of moving it. Training has to move the curve.
3. **Boundary errors barely move** (364 → 399 across the whole sweep). They are genuine
   model boundary judgement, not a thresholding artifact.

The shipped 0.50 is confirmed as the precision-maximising operating point and is left
unchanged.

## 6. Recipe Definitions

All recipes share: the same encoder, the same label vocabulary
(`DIAGNOSIS, MEDICATION, SYMPTOM, TEST_NAME, TEST_RESULT`), the same alignment and
supervision contracts, the same governed training split, and the same evaluator. Only
governed training gold is used; no public organizer input and no pseudo-labels.

| Recipe | Change vs the previous row | Purpose |
| --- | --- | --- |
| **R0** | none — the current checkpoint, no training | control |
| **R1** | continued fine-tuning at LR 1e-5 / head 3e-5 (was 3e-5 / 1e-4) | isolates "more training" |
| **R2** | R1 **+ `focal_alpha` 0.25 → 0.75**, nothing else | isolates the class-weighting fix from §4.7 |

R1 and R2 differ by exactly one scalar, so the comparison attributes cleanly — the brief's
"do not combine several major experimental changes into one uninterpretable recipe".

**A deliberate change of selection metric.** The original run selected its best epoch on
`validation_span_micro_f1`, a token-run proxy. Both recipes here select on the **exact
character-span + type** evaluator (`evaluation/exact_mention.py`) decoded through the
production decoder, per brief §7 ("do not select using token-level accuracy"). A token
proxy can improve while exact character spans get worse, which is the failure this
milestone is trying to detect.

**R3 was not run.** It is conditional in the brief on R1/R2 evidence identifying one clear
remaining issue; §8 records what the R1/R2 evidence actually showed and why no third
recipe followed from it.

### Complete hyperparameters

| Parameter | R1 | R2 |
| --- | --- | --- |
| initialise from | `best.pt` (`a64cc173…`) | `best.pt` (`a64cc173…`) |
| epochs | 2, early stopping patience 1 | 2, early stopping patience 1 |
| learning rate (encoder / head) | 1e-5 / 3e-5 | 1e-5 / 3e-5 |
| batch (per device × accumulation) | 8 × 4 = 32 | 8 × 4 = 32 |
| weight decay / warmup / clip | 0.01 / 0.06 / 1.0 | 0.01 / 0.06 / 1.0 |
| max sequence length | 256 | 256 |
| loss | focal, γ=2.0, **α=0.25** | focal, γ=2.0, **α=0.75** |
| decision threshold | 0.5 | 0.5 |
| seed | 20260731 | 20260731 |
| supervised training examples | 23,799 (10,027 unsupervised skipped) | 23,799 |
| optimizer steps / epoch | 743 | 743 |

## 7. Validation Results

Every recipe measured through the **same** evaluator on the **same** L1→L4 path against
the same governed validation split. The gate's baseline is an L1→L4 number, so measuring
a candidate on E3 in isolation would produce a figure that looks comparable and is not.

| Recipe | α | P | R | **F1** | ΔF1 | TP | FP | FN | boundary | predictions | empty docs | offsets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 control | 0.25 | 0.6270 | 0.5344 | 0.5770 | — | 1064 | 633 | 927 | 364 | 1,697 | 165 | 0 |
| R1 | 0.25 | **0.6916** | 0.6354 | 0.6623 | +8.53 | 1265 | 564 | 726 | 299 | 1,829 | 130 | 0 |
| R2 | 0.75 | 0.5921 | **0.7202** | 0.6499 | +7.29 | 1434 | 988 | 557 | 328 | 2,422 | 33 | 0 |
| **R3 (selected)** | **0.50** | 0.6546 | 0.7072 | **0.6799** | **+10.29** | 1408 | 743 | 583 | **298** | 2,151 | 54 | **0** |

Per-type F1:

| Recipe | DIAGNOSIS | SYMPTOM | MEDICATION |
| --- | ---: | ---: | ---: |
| R0 control | 0.5786 | 0.5546 | 0.6831 |
| R1 | 0.7019 | 0.5675 | 0.6978 |
| R2 | 0.7080 | **0.5243** | 0.7090 |
| **R3** | **0.7190** | **0.5909** | **0.7266** |

Error categories:

| Recipe | exact | missed | spurious | left | right | both | wrong type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 1064 | 537 | 242 | 111 | 216 | 37 | 26 |
| R1 | 1265 | 408 | 243 | 94 | 184 | 21 | 19 |
| R2 | 1434 | 169 | 569 | 109 | 188 | 31 | 60 |
| **R3** | 1408 | 236 | 387 | 91 | 192 | 15 | 49 |

**What the α axis actually shows.** The three runs differ by one scalar and trace a clean
precision/recall frontier, which is exactly what §4.7 predicted:

* **α = 0.75 (R2)** confirms the diagnosis outright — missed entities collapse from 537 to
  169, a 69% reduction, proving those entities were always recoverable and were being
  suppressed by the loss. But precision falls 3.49 below baseline and SYMPTOM F1 falls
  3.04, so it is **rejected**. The correction is real and too strong.
* **α = 0.25 (R1)** improves both sides, but leaves 408 entities missed.
* **α = 0.50 (R3)** is the midpoint the two bracketing runs identified, and it dominates:
  the best F1, the best boundary count, and an improvement in *every* entity type.

R3 was the conditional recipe, and it was run because R1/R2 identified one clear
remaining issue rather than as an open search: a single predefined value between two
measured points. `--focal-alpha` is the only override, so all four runs remain mutually
attributable.

**Boundary errors fell for every recipe** (364 → 299 / 328 / 298), which no threshold in
the §5 sweep achieved. That is the direct evidence that training moved the curve where
recalibration could only slide along it.

## 8. Acceptance Gate

Applied by `scripts/evaluate_e3_checkpoint_0062.py::apply_gate` at the brief's thresholds.
The gate is not parameterised — a gate a caller can loosen at the command line is not a
gate. The baseline is stored as confusion **counts** rather than the rounded rates the
audits print, because comparing full-precision candidates against 4-decimal constants
made the control fail its own gate by −0.00 (1064/1697 = 0.626989 is not ≥ 0.6270).

| Criterion | Required | R1 | R2 | **R3 (selected)** |
| --- | --- | ---: | ---: | ---: |
| exact span+type micro-F1 | ≥ +1.5 | +8.53 ✅ | +7.29 ✅ | **+10.29 ✅** |
| precision | must not decrease | +6.46 ✅ | **−3.49 ❌** | **+2.76 ✅** |
| recall | drop ≤ 1.0 | +10.10 ✅ | +18.58 ✅ | **+17.28 ✅** |
| DIAGNOSIS F1 | must not decrease | +12.33 ✅ | +12.94 ✅ | **+14.04 ✅** |
| SYMPTOM F1 | must not decrease | +1.29 ✅ | **−3.04 ❌** | **+3.63 ✅** |
| MEDICATION F1 | drop ≤ 1.0 | +1.47 ✅ | +2.59 ✅ | **+4.35 ✅** |
| total boundary errors | must decrease | 299 < 364 ✅ | 328 < 364 ✅ | **298 < 364 ✅** |
| offset violations | exactly 0 | 0 ✅ | 0 ✅ | **0 ✅** |
| entity-type collapse | none | none ✅ | none ✅ | **none ✅** |
| parameter budget | < 9B | 135,002,117 ✅ | 135,002,117 ✅ | **135,002,117 ✅** |
| confirmation seed | ≥ +1.0 F1 | not run | not run | _see below_ |
| L8/L9 contract | unchanged | ✅ | ✅ | **✅** |

**Selected: R3.** One checkpoint, as the brief requires. R1 also passes and is recorded as
the runner-up; R2 is rejected on two criteria and was not weakened to fit.

### 8.1 Confirmation seed

The gate's last criterion is a second, independent seed. Recipe R3 was retrained
unchanged except for `--seed 20260801`:

| | seed 20260731 (selected) | seed 20260801 (confirmation) |
| --- | ---: | ---: |
| P | 0.6546 | 0.6440 |
| R | 0.7072 | 0.7112 |
| **F1** | **0.6799 (+10.29)** | **0.6759 (+9.89)** |
| boundary errors | 298 | 297 |
| offset violations | 0 | 0 |

The requirement was ≥ +1.0 F1; the confirmation seed delivers **+9.89** and independently
passes **all ten** other criteria. Seed-to-seed spread is 0.40 F1 points against a gain of
roughly 10, so the result is not a seed artifact.

**Gate outcome: PASSED.** The selected checkpoint is R3 seed 20260731.

## 9. Selected Checkpoint and Resource Measurements

| Field | Value |
| --- | --- |
| Recipe | R3 (low-LR refinement, focal α 0.50) |
| Path | `checkpoint/experiments/0062_e3_boundary_refinement/R3_alpha050/best.pt` |
| **SHA-256** | **`524ece1e7d190838cb8b1ce3b0a0f337bc5b8b7cc7cef70c4c3e0b0310adde3a`** |
| Size | 1,615,514,043 bytes |
| Parameters | 135,002,117 (134,998,272 base + 3,845 head) |
| Source checkpoint | `a64cc173…1017c`, verified unchanged after every run |
| Epoch / step | 2 / 1,486 |
| Seed | 20260731 |
| Load mode | `WEIGHTS_ONLY` (**not** the legacy pickle allowance) |

Per-run resources, all four training runs on the local RTX 4060 Laptop (8,188 MiB):

| Run | wall time | peak VRAM | peak RSS |
| --- | ---: | ---: | ---: |
| R1 | 332.5 s | 2.929 GiB | < 8 GiB |
| R2 | 328.6 s | 2.929 GiB | < 8 GiB |
| R3 | 323.7 s | 2.929 GiB | < 8 GiB |
| R3 confirmation | 316.3 s | 2.929 GiB | < 8 GiB |

Each run is 2 epochs × 743 optimizer steps over 23,799 supervised examples at effective
batch 32. Validation inference is ~69 s per recipe on the L1→L4 path. **Local training
was practical**, so the Colab package in §10 is provided for reproduction rather than
because it was needed.

Total GPU time for the whole comparison: **21.7 minutes**.

## 10. Execution Package

Four tracked artifacts, no notebook:

| File | Role |
| --- | --- |
| `configs/training/e3_boundary_refinement_0062.yaml` | the refinement contract |
| `scripts/train_e3_boundary_refinement_0062.py` | training, resume, manifests |
| `scripts/evaluate_e3_checkpoint_0062.py` | one evaluator + the gate for every recipe |
| `scripts/run_e3_refinement_0062.sh` | the controlled comparison, end to end |

The training script does **not** reuse `s1_full_training.FullTrainingConfig`. That contract
refuses any initializer other than `pretrained_base` deliberately — for a from-scratch S1
run a checkpoint initializer would launder the one-step smoke artifact into a full run.
Refinement is the case that rule was not written for, so it carries its own explicit
contract rather than a loosened version of that one.

Fail-closed behaviour: splits are resolved by authoritative SHA-256 (never by name),
`internal_test` is refused by `governed_splits`, the source checkpoint is hashed before a
byte is deserialized and re-hashed afterwards, a fast tokenizer is rejected, a non-finite
loss aborts, and an unconfirmed run refuses to start.

### Exact commands

**Local RTX 4060** (what was actually run):

```bash
bash scripts/run_e3_refinement_0062.sh R0 R1 R2 R3
RUN_ID=R3_alpha050 ALPHA=0.50 bash scripts/run_e3_refinement_0062.sh CONFIRM
```

**Colab L4** (22.5 GB VRAM — larger batch, same effective batch size):

```bash
git clone <repo> MedNorm-VI && cd MedNorm-VI
pip install -r requirements.lock
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python scripts/train_e3_boundary_refinement_0062.py \
  --config configs/training/e3_boundary_refinement_0062.yaml \
  --run-id R3_alpha050 --output-root checkpoint/experiments/0062_e3_boundary_refinement \
  --confirm "RUN E3 BOUNDARY REFINEMENT" --focal-alpha 0.50 --resume
python scripts/evaluate_e3_checkpoint_0062.py \
  --checkpoint checkpoint/experiments/0062_e3_boundary_refinement/R3_alpha050/best.pt \
  --label R3_alpha050 --output runs/diagnostics/0062_recipes
```

**Colab A100** (40/80 GB — identical, `--resume` is the only concession to disconnects):

```bash
python scripts/train_e3_boundary_refinement_0062.py \
  --config configs/training/e3_boundary_refinement_0062.yaml \
  --run-id R3_alpha050 --output-root /content/drive/MyDrive/MedNorm-VI/0062 \
  --confirm "RUN E3 BOUNDARY REFINEMENT" --focal-alpha 0.50 --resume
```

`per_device_batch_size` × `gradient_accumulation_steps` must keep the effective batch at
32 on any device, or the schedule arithmetic no longer matches these results.

## 11. Architecture Integration

The weights changed. Nothing else did.

**Checkpoint profile registry** — `configs/models/e3_checkpoint_profiles.yaml` declares
both artifacts with digests, metrics and provenance, and names exactly one `active`.
Before this milestone the active artifact was a default constant in
`mention_factory/experts/e3_vihealthbert.py`, so a rollback meant editing source; a
rollback that needs a code change is not one anybody performs under pressure.

| Profile | Status | Digest | Validation F1 |
| --- | --- | --- | ---: |
| `s1_full_training_v1` | ROLLBACK | `a64cc173…1017c` | 0.5770 |
| `boundary_refinement_0062` | **ACTIVE** | `524ece1e…dde3a` | **0.6799** |

**Selection** is `expert_settings.e3_checkpoint_path` +
`e3_expected_checkpoint_sha256` in `configs/pipeline/full_v1.yaml`, with the rollback
values written out in a comment beside them. A test asserts the pipeline config and the
active profile agree, so the two files cannot drift.

**The refined checkpoint does not inherit the legacy pickle allowance.** Its digest is
unknown to `TRUSTED_LEGACY_E3_SHA256`, so `assert_load_is_permitted` routes it to
`weights_only=True` — strictly safer than the artifact it replaces, and the Audit-0056a
exception stays limited to the single file it was written for. This was verified, not
assumed.

**Ledgers.** The candidate registry records `active_checkpoint_profile` and
`rollback_checkpoint_profile`. The parameter count is unchanged at 134,998,272 and the
deployment ledger still totals **135,004,814**. A note now explains that 134,998,272 is
the BASE MODEL and deliberately excludes the `Linear(768, 5)` head (3,845 parameters),
since spec §17 excludes heads and adapters from the 9B budget — a full state-dict count
of 135,002,117 is the same model, not a discrepancy.

**Unchanged:** E1/E2 flags and behaviour, L4 route selection, L5-L9 ownership, candidate
retrieval, the KB indices, the decoding profile (`competition_top1`), and the ICD
serialization decision. GLiNER stays `EXCLUDED_BY_ABLATION` with
`enable_e6_gliner: false`.

**Tests** — `tests/unit/test_e3_checkpoint_profiles_0062.py`, 17 tests covering all ten
requirements: rollback still declared and loadable (1), active profile resolves an exact
checkpoint and digest (2), wrong hash fails closed twice (3), missing/empty checkpoint
fails closed (4), decoded offsets stay source-aligned (5), submission schema and label
space unchanged (6), E1/E2 unchanged (7), dotted ICD canonical at serialization (8),
dotless remains the KB identity (9), parameter budget under 9B (10).

## 12. Validation and Static Gates

```text
targeted   tests/unit/test_e3_checkpoint_profiles_0062.py     17 passed
full (1)   pytest tests/ -q      2 failed, 1969 passed, 1 skipped in 459.99s
full (2)   pytest tests/ -q         1971 passed, 1 skipped in 473.29s (0:07:53)
ruff check .                                                  All checks passed!
ruff format --check <changed Python files>                    clean
mypy src/mednorm_vi                       Success: no issues found in 291 source files
compileall -q src tests scripts                               OK
git diff --check                                              OK
```

**The suite was run twice, and the first run failed.** Recording that rather than only the
green result, because what failed is informative:

1. `test_e3_canonical_integration::test_e3_runs_through_the_canonical_runner` asserted
   `record.checkpoint_sha256 == E3_CHECKPOINT_SHA256` — the module *default* constant,
   which since this milestone is the ROLLBACK artifact rather than the active one. The
   test would have failed on any correctly applied checkpoint change, which is the
   opposite of what it exists to catch. It now asserts that the digest the run RECORDS
   equals the digest the profile SELECTED, which is drift-proof and strictly stronger.
2. `test_e4_retirement::test_no_e4_checkpoint_remains_on_disk` asserted "the tree contains
   exactly one `.pt` file". That was always a proxy for the real claim — E4 has no weights
   — and the proxy broke when governed refinement experiments were added under
   `checkpoint/experiments/`. It now checks the claim directly (no `w2ner`/`phobert`/`e4`
   weight file anywhere) **and** still requires every remaining `.pt` to be an E3/S1
   artifact in a known governed location, so a stray checkpoint elsewhere still fails.

Neither test was weakened to accommodate the change: one now checks a stronger invariant,
the other checks the invariant it was always meant to check.

`tests/unit/test_e4_retirement.py` is not `ruff format`-clean at `HEAD` and was left that
way; only the lines this milestone added were written in the formatter's style, since
repository-wide formatting is out of scope.

No `src/` file was modified in this milestone, which is why the mypy surface is unchanged
at 291 files. The change is weights plus configuration plus tests plus scripts.

## 13. Public-Run Status

Public organizer inference was run **exactly once**, after the gate passed.

| | |
| --- | --- |
| Input | `data/organizer_test/input` (100 documents) |
| Mode | `specialist` (E1 + E2 + accepted E3; GLiNER disabled) |
| Decoding profile | `competition_top1`, diagnosis top-1, medication top-1 |
| KB | Competition v3 ICD + RxNorm indices |
| Runtime | **61.8 s** for 100 documents, prepared indices loaded once |
| Peak RSS | 2.685 GiB |
| L9 gate | passed — `l9_stopped_the_run: false` |
| Offset violations | 0 |
| Clinical text in manifest | none (`contains_clinical_text: false`) |

Artifacts in `runs/e3_boundary_v2_final_dotted/`: `output/` (100 JSON files),
`output.zip`, `run-manifest.json`, `run.log`, `diagnostic-summary.json`,
`validation-report.json`, `checkpoint-manifest.json`.

**Final dotted submission ZIP**

```text
runs/e3_boundary_v2_final_dotted/output.zip
eae8a348ff5adf8b184a2b98d66868e43aadcd8800e38b67c110fa6f74b24c05
```

100 entries, all under a single `output/` root, one deterministic ZIP timestamp
(1980-01-01). No JSON was edited by hand and no second inference was run.

The pipeline emits dotless ICD; dotted is applied at the governed serialization boundary
by `inference/derive_submission.py` — the same transform that produced the scored 9.5736
probe, which is what makes the two artifacts comparable. The staged dotless run is
recorded in `run-manifest.json` as `output_zip_sha256: 2ea8a3d2…4bc33`; the dotted
derivation is the shipped candidate above.

Derivation report: 311 ICD codes seen, **277 transformed** to dotted, 34 three-character
categories correctly left unchanged, **0 malformed**, 0 candidates removed, entity count
preserved per document. RxNorm candidates (11) untouched.

### Comparison with the scored dotted baseline (9.5736)

| Measure | Baseline | This candidate | Δ |
| --- | ---: | ---: | ---: |
| Total entities | 957 | **1,147** | **+190** |
| DIAGNOSIS | 199 | 337 | +138 |
| SYMPTOM | 575 | 627 | +52 |
| MEDICATION | 11 | 11 | 0 |
| TEST_NAME | 35 | 35 | 0 |
| TEST_RESULT | 137 | 137 | 0 |
| Assertion labels | 82 | 106 | +24 |
| Candidate codes | 190 | 322 | +132 |
| Empty documents | 5 | **1** | −4 |
| Short spans (≤3 chars) | 267 | 255 | −12 |
| Retained entities | — | 714 | — |
| Removed entities | — | 243 | — |
| New entities | — | 433 | — |

E1/E2-owned types (MEDICATION, TEST_NAME, TEST_RESULT) are **identical**, which is the
expected signature of an E3-only change and is direct evidence that the weights swap
stayed inside its layer. The movement is entirely in the two types E3 owns.

**No leaderboard claim is made.** The candidate improves the metric's gateway on governed
validation; whether that converts on the public test is unknown until it is submitted.
The 243 removed entities are the honest risk: this run is more aggressive
(1,147 vs 957 entities), and under Jaccard a wrong extra entity costs as much as a
missing one.

## 14. Remaining Risks

1. **Validation is not iid with training** (§4.6): 86.7% `vimedner` against 24.4% in
   training. The measured +10.29 F1 is a real improvement on that distribution; how much
   transfers depends on the organizer's composition, which cannot be inspected under this
   brief. This is the single largest uncertainty in the milestone.
2. **The candidate is more aggressive than the scored baseline** (+190 entities, 243
   removed). Precision rose on validation (+2.76), but the public test is a different
   sample.
3. **39.4 F1 points were available; roughly 10 were taken.** The oracle ceiling is 0.9713
   and the accepted checkpoint reaches 0.6799. Substantial capability headroom remains.
4. **J_candidates is still the binding constraint.** Even a perfect mention layer only
   feeds L5B/L5C; the dotted probe earned 2.24 of 40 candidate points, and this milestone
   did not touch retrieval or linking.
5. **α = 0.50 was not exhaustively searched.** It is the midpoint of two measured points
   and it dominates both, but the optimum may lie elsewhere on that axis. Widening the
   search would have been hyperparameter tuning against validation, which the brief
   bounds.
6. **`hypothesis_id` ordinals are not deterministic across runs** for one document in
   1,045 (§3). Internal only, never serialized, no metric effect — recorded so it is not
   rediscovered as a mystery later.
7. **Adjacent same-type entities remain architecturally unmergeable** (76 of 1,991, §4.1).
   Fixing this needs a B/I distinction or a span head, which is a genuine architecture
   change and out of scope here.

## 15. Changed-File Inventory

New:

```text
configs/models/e3_checkpoint_profiles.yaml
configs/training/e3_boundary_refinement_0062.yaml
scripts/train_e3_boundary_refinement_0062.py
scripts/evaluate_e3_checkpoint_0062.py
scripts/run_e3_refinement_0062.sh
tests/unit/test_e3_checkpoint_profiles_0062.py
docs/audits/0062-e3-boundary-aware-refinement-and-final-candidate.md
```

Modified:

```text
configs/pipeline/full_v1.yaml            expert_settings E3 checkpoint + digest, rollback comment
configs/models/candidate_model_registry.yaml   active/rollback profile refs, parameter-count note
tests/unit/test_e3_canonical_integration.py    provenance asserted against the SELECTED digest
tests/unit/test_e4_retirement.py               E4 claim checked directly, not by file count
```

Untracked run artifacts (ignored): `runs/e3_boundary_v2_final_dotted/`,
`runs/diagnostics/0062_*`, `checkpoint/experiments/0062_e3_boundary_refinement/`.

**No `src/` file was modified.** The protected artifacts are unchanged: the architecture
PDF (`0d5eaa20…81e09b`), the E3 rollback checkpoint (`a64cc173…1017c`), and the three
scored baseline ZIPs.

## 16. Verdict

**`E3_REFINEMENT_READY_FOR_LEADERBOARD`**

A single checkpoint was selected on evidence and passes every criterion of the governed
gate — including the confirmation seed, which passes the entire gate independently. The
gate was not weakened at any point; R2 was rejected on it despite being the recipe that
proved the root-cause diagnosis correct.
