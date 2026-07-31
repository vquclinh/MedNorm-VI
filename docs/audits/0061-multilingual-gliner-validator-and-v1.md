# Audit 0061: Multilingual GLiNER Validator, Dotted ICD Resolution, and the Final Attempt

Milestone: 4B - Multilingual GLiNER precision validator and second leaderboard candidate
Date: 2026-07-31
Verdict: `GLINER_VALIDATOR_NOT_ACCEPTED`

## 1. Initial Git State

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `6270cc0 feat: add precision-first profiles and ICD serialization probe` |
| Working tree | clean |
| Staged files | none |
| `origin/main...HEAD` | `0 0` |
| Audit 0060 committed | yes |
| RAM / disk / GPU / containers | 7.8 GiB free / 25 G / RTX 4060 8188 MiB (13 MiB used) / 0 |

The architecture PDF is unmodified (`0d5eaa20…81e09b`) and was read in full earlier in
this session.

## 2. ICD Serialization: RESOLVED = DOTTED

The owner submitted the Audit-0060 probe. Verified result:

| Submission | Score | WER | J_assertion | J_candidates |
| --- | ---: | ---: | ---: | ---: |
| Profile A (dotless) | 9.1625 | 87.0976 | 16.0234 | 1.2119 |
| Dotted ICD probe | **9.5736** | 87.0976 | 16.0234 | **2.2396** |
| Delta | **+0.4111** | **0** | **0** | **+1.0277** |

Probe ZIP `f87db8aead3fd935e2a0b9e3245b359b65953f7d28430ca3b56aeff22f663d59`.

Both scores reproduce exactly under `0.3·(100−WER) + 0.3·J_assert + 0.4·J_cand`, and
`0.4 × 1.0277 = 0.4111` accounts for the entire gain. **The zero delta on the two
components the probe did not touch is what makes the attribution airtight**: the probe
differed from an already-scored submission only by the reversible dot, so nothing else
can explain the movement.

Governance updated:

- `UNRES-ICD-DOTTED` → `status: resolved`, `resolution: dotted`, `confidence: high`,
  `internal_default: dotted`, `leaderboard_experiment_id: EXP-ICD-DOTTED-0060`.
- `ICDF-DOTTED-OUTPUT` → `CONFIRMED`; `ICDF-UNDOTTED-OUTPUT` → `REJECTED_FOR_OUTPUT`.

Scope is deliberately narrow and enforced by tests: **dotted is an output
serialization; dotless remains the KB concept identity and the offered-set key.** The
dot is applied only at the governed serialization boundary
(`inference/code_format.py` → `derive_submission.py`); retrieval, ranking and
offered-set membership are untouched; three-character categories are unchanged; RxNorm
candidates are never dotted. Historical dotless baselines stay read-only and are
hash-pinned.

**A necessary correction, not a sufficient one.** J_candidates rose from 1.21 to 2.24 —
but that is 2.24 of 40, so **39.10 candidate points remain unearned**. The format was
wrong *and* the codes are still largely wrong.

## 3. Model Acquisition

`urchade/gliner_medium-v2.1` was **not** downloaded. The approved multilingual model was
used, acquired through a new governed tool (`model_registry/acquire.py`) that resolves,
pins, hashes and verifies before anything is registered.

| Field | Value |
| --- | --- |
| Model ID | `urchade/gliner_multi-v2.1` |
| Pinned revision | `443d26d654e0324125a96bebd8e796c14ff2efe6` |
| License | **apache-2.0** (read from repository metadata, matched against expectation) |
| Source | `https://huggingface.co/urchade/gliner_multi-v2.1/tree/443d26d6…` |
| Local path | `models/mention/gliner_multi-v2.1/443d26d654e0324125a96bebd8e796c14ff2efe6/` |
| **Parameters** | **288,949,504** — counted from the loaded model |
| Total bytes | 1,155,835,359 |
| Tokenizer identity | `microsoft/mdeberta-v3-base` |

File hashes:

```text
gliner_config.json   477 B          e25f61d91620df84aae8076811ee592e926e94d341b82e7bc1be359718f83017
model.safetensors    1,155,830,112  2100142f31627531497850659dcb3821c99d5e71c08a8e01a98e4b11ef32a199
README.md            4,770          820125f2ea897716cc38645d94f73f390ec9dfd392980c436be1d25c2e106b55
```

Only safetensors were taken. The repository also ships `pytorch_model.bin`; acquiring a
pickle checkpoint would drag it under the Audit-0056a trusted-load policy for no
benefit, so it was excluded by an explicit allow-list.

**Companion tokenizer**, acquired under the same governance because GLiNER ships none
and names a backbone instead:

| Field | Value |
| --- | --- |
| Model ID | `microsoft/mdeberta-v3-base` |
| Pinned revision | `a0484667b22365f84929a935b5e50a51f71f159d` |
| License | mit |
| Files | `config.json` (`bcffcd34…`), `spm.model` (`13c8d666…`), `tokenizer_config.json` (`3f3978e0…`) |

The acquisition tool failed closed three times during this milestone and each refusal
was correct: once on a missing tokenizer file, once on a missing weights file for a
deliberately weights-free tokenizer artifact, and once on an unverifiable parameter
count. The first was a genuine gap (the backbone tokenizer really was missing); the
second exposed that "model" and "tokenizer" are different artifact kinds and must be
verified against different shapes, which is now explicit rather than assumed. No check
was weakened to make an acquisition succeed.

## 4. Resource Measurements

```text
GLiNER load        9.16 s      device cuda
inference         32.21 s      for 2 × 1,045 documents (two label sets)
peak RSS           3.22 GiB
peak VRAM          1.128 GiB   of 8,188 MiB available
offset violations  0
```

## 5. E3 Validation Baseline

Reproduced from the **current runtime**, not from historical numbers. The ablation CLI
was not used: it requires VnCoreNLP, which the shipped active profile does not, so it
would have measured a different configuration. A harness faithful to
`inference/pipeline.run_document` through L4 was used instead.

Governed validation split, SHA-256
`ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103`, 1,045 examples,
1,991 gold entities.

```text
micro span+type   P=0.6270  R=0.5344  F1=0.5770   TP=1064  FP=633  FN=927
  DIAGNOSIS       P=0.6607  R=0.5146  F1=0.5786
  SYMPTOM         P=0.5520  R=0.5572  F1=0.5546
  MEDICATION      P=0.7578  R=0.6218  F1=0.6831
  TEST_NAME       0 gold, 3 FP        TEST_RESULT  0 gold, 14 FP

error categories  exact_match 1064 | missed 537 | spurious 242
                  right_boundary 216 | left_boundary 111 | both_boundary 37 | wrong_type 26
```

Audit 0052's 0.5559 was measured on 200 rows; this is the full split and is the number
the gate is judged against.

## 6. GLiNER Standalone Diagnostics

Governed label contract, fixed before any measurement — no prompt search:

```text
en:  "medical symptom" -> SYMPTOM        "diagnosis or disease" -> DIAGNOSIS
vi:  "triệu chứng y khoa" -> SYMPTOM     "chẩn đoán hoặc bệnh" -> DIAGNOSIS
```

```text
predictions (threshold floor 0.30)   en 1,913   vi 889
E3 proposals for comparison          1,679
offset violations                    0   (every prediction re-anchored to source)
```

## 7. Validator Policies and the Acceptance Gate

A scope limitation, stated plainly: the policies target the **L7-UNRESOLVED**
population, but on this split L7's deterministic low-confidence condition never fires —
`LOW_CONFIDENCE_BELOW = 0.35` and the minimum L4 score is 0.360. The other UNRESOLVED
conditions depend on L5/L6 outputs a mention-level harness does not produce. The sweep
therefore evaluated the **full** E3 symptom/diagnosis population (the superset any
validator could act on) and, separately, the lowest-confidence tertile (≤ 0.62) as the
nearest available proxy for "uncertain".

40 policies: {exact support, overlap support ≥ 0.5 IoU} × {en, vi} ×
{0.30, 0.40, 0.50, 0.60, 0.70} × {all, low-confidence}. Policy 3 (confidence as a
feature) is reported but is not a gate and cannot be accepted as one.

Representative rows, deltas in absolute points against the baseline:

| Policy | P | R | F1 | ΔF1 | ΔP | ΔR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| POLICY_0 baseline | 0.6270 | 0.5344 | 0.5770 | — | — | — |
| **POLICY_1 en 0.30 lowconf** (best) | 0.7390 | 0.4736 | **0.5773** | **+0.03** | +11.20 | −6.08 |
| POLICY_1 en 0.40 lowconf | 0.7391 | 0.4666 | 0.5720 | −0.50 | +11.21 | −6.78 |
| POLICY_2 en 0.30 lowconf | 0.7151 | 0.4766 | 0.5720 | −0.50 | +8.82 | −5.78 |
| POLICY_1 vi 0.30 lowconf | 0.7443 | 0.4621 | 0.5702 | −0.68 | +11.73 | −7.23 |
| POLICY_1 en 0.30 **all** (max precision) | 0.7820 | 0.2612 | 0.3916 | −18.54 | **+15.50** | −27.32 |
| POLICY_1 vi 0.70 all (worst) | 0.7168 | 0.0814 | 0.1461 | −43.09 | +8.98 | −45.30 |

### Gate result

| Criterion | Required | Best achieved |
| --- | --- | --- |
| span+type micro-F1 | **≥ +1.0** | **+0.03** |
| precision | improves | +11.20 ✓ |
| recall | ≥ −3.0 | −6.08 ✗ |
| DIAGNOSIS F1 | ≥ −2.0 | −1.73 ✓ |
| SYMPTOM F1 | ≥ −2.0 | +3.64 ✓ |
| offset violations | 0 | 0 ✓ |

**Policies meeting all criteria: 0 of 40.**

The pattern is consistent and interpretable. GLiNER agreement **is** a genuine precision
signal — it lifts precision by 9–15 points everywhere. But it disagrees with far too
many *correct* E3 spans, so recall collapses by 6 points at best and 45 at worst, and
net F1 never moves. On the full population the effect is catastrophic (−18 to −43 F1),
because GLiNER's zero-shot span conventions do not match the governed annotation
convention: it returns 1,913 spans where E3 returns 1,679, and their exact-span overlap
is low.

This is precisely the risk Audit 0060 §19 named when selecting GLiNER: *"GLiNER is not
trained on the organizer's annotation convention; it must be used as a validator on
low-confidence E3 spans, never as an independent proposer, or it will add spurious
entities and raise WER."* The prediction was right about the mechanism; the measurement
shows the validator framing does not rescue it either.

The gate was **not** weakened despite only one leaderboard attempt remaining. A +0.03
point change on validation is indistinguishable from noise and would be a reckless basis
for spending the last attempt.

## 8. Architecture Integration

**None.** Because the gate failed, E6 was not wired into L3/L4/L7, no feature flag was
added, and the runtime is byte-for-byte unchanged in its inference behaviour. Making a
rejected model reachable from a profile would be the exact silent-capability failure the
expert-spec registry exists to prevent.

E6 is recorded in the candidate registry as researched and rejected:

```text
status                    EXCLUDED_BY_ABLATION      (governed vocabulary: researched, not deployed)
loaded_at_inference       false
total_parameters          288,949,504
parameter_count_method    counted_from_checkpoint
parameter_count_verified  true
```

Deployment ledger **unchanged**:

```text
total loaded (gated)   134,998,272 + 6,542 = 135,004,814
limit                  9,000,000,000
remaining margin       8,864,995,186        within budget: True
```

A verified 288M model that failed its gate consumes no budget, and a test asserts that.

## 9. Test, Static and Type Results

Targeted: `test_governed_model_acquisition_0061.py`, `test_icd_dotted_serialization_0061.py`,
`test_precision_first_0060.py` → **49 passed**.

Full suite: see §9a.

| Command | Result |
| --- | --- |
| `ruff check .` | All checks passed |
| `ruff format --check <changed files>` | formatted |
| `mypy src/mednorm_vi` | Success, 291 source files |
| `compileall -q src tests` | passed |
| `git diff --check` | passed |

One configuration change was required for mypy: `huggingface_hub` was added to the
existing, already-documented `follow_imports = "skip"` override list. Without it mypy
follows huggingface_hub into numpy's stubs, which target a newer `python_version` than
this project declares, and fails for an environment reason rather than a code reason —
exactly the case that comment block was written to describe.

## 9a. Full-suite Result

```text
1954 passed, 1 skipped in 468.00s (0:07:48)
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

The single skip is the pre-existing `pyarrow` absence, unchanged since Audit 0057. The
two new test files were `ruff format`-normalised after that run and re-executed on their
own (`18 passed`); the reformat changed line wrapping only.

## 10. Public-run Status

**No public-test inference was run. The final leaderboard attempt was not consumed.**

The milestone is explicit: run public inference only after the acceptance gate passes.
It did not pass, so no `runs/gliner_v1_precision_validator_dotless/` or
`runs/gliner_v1_precision_validator_dotted/` artifacts exist, and none should be
fabricated.

## 11. Comparison with Profile A

Not applicable — no GLiNER submission was produced. The comparison the milestone
specifies presumes an accepted policy.

## 12. Exact Changed-file Inventory

Tracked modifications:

```text
M configs/models/candidate_model_registry.yaml
M configs/organizer/icd_format_hypotheses_v1.yaml
M configs/organizer/unresolved_policies_v1.yaml
M pyproject.toml
```

New tracked-intended files:

```text
docs/audits/0061-multilingual-gliner-validator-and-v1.md
src/mednorm_vi/model_registry/acquire.py
tests/unit/test_governed_model_acquisition_0061.py
tests/unit/test_icd_dotted_serialization_0061.py
```

Ignored generated evidence:

```text
models/mention/gliner_multi-v2.1/443d26d654e0324125a96bebd8e796c14ff2efe6/
models/mention/mdeberta-v3-base/a0484667b22365f84929a935b5e50a51f71f159d/
runs/diagnostics/0061_gliner/
```

No inference-path source file was modified. The baseline A/B/C artifacts and the dotted
probe are byte-identical.

## 13. Remaining Risks

1. **39.10 of 40 candidate points remain unearned** even with correct serialization. The
   dot fixed the format; the codes are still largely wrong or attached to unmatched
   spans.
2. **WER 87.1% is untouched** and remains the gateway: 26.13 unearned points, gated on
   entity detection quality.
3. `UNRES-ICD-SPECIFICITY` is still unresolved — 3-character categories may or may not
   be accepted.
4. `UNRES-POS-INTERVAL`, `UNRES-POS-COORD-SPACE` and `UNRES-MATCH-MODE` remain open and
   could each independently depress WER.
5. The L7-UNRESOLVED population could not be reproduced on validation, so the "restrict
   to uncertain cases" idea was tested only through a confidence-tertile proxy.
6. Over- versus under-prediction on the public test is still undetermined without gold.

## 14. Verdict

`GLINER_VALIDATOR_NOT_ACCEPTED`

Multilingual GLiNER was acquired under full governance — pinned revision, verified
apache-2.0 license, hashed safetensors, 288,949,504 parameters counted from the loaded
model, zero offset violations — and evaluated honestly against the governed acceptance
gate. It failed: the best of 40 policies moved micro-F1 by +0.03 points against a
required +1.0, because agreement buys precision at a recall cost the gate correctly
refuses to accept.

The milestone still produced durable value: the ICD serialization question is resolved
with hard leaderboard evidence and locked behind regression tests, and the repository now
has a governed model-acquisition path that fails closed — which is what will make the
*next* model decision cheap and safe rather than speculative.

## 15. Recommendation for the Final Attempt

**Do not spend it on GLiNER.** Nothing measured here justifies it.

The one change with verified leaderboard evidence behind it is the dot, and it is
already proven: +0.4111 for a transformation that touches nothing else. If the owner
wants the best available submission from work already validated, it is the existing
dotted probe — but that has **already been submitted and scored at 9.5736**, so
resubmitting it gains nothing.

Therefore the honest recommendation is to **hold the final attempt** until an
intervention exists that is expected to move WER or candidate correctness, rather than
spend it on a validator that measured flat. The next candidate should target the
gateway component:

1. **Fine-tune E3 on the governed training split with the boundary errors in view** —
   327 of the 1,697 predictions are left/right/both-boundary errors, the single largest
   correctable category after `missed`. This needs no new model and no new licence.
2. Failing that, an ontology-aware retriever for the candidate component, where 39.10
   points sit unearned.

## 16. Safety Confirmations

No notebook or Jupyter, no training or fine-tuning, no `internal_test` access, no
external API at inference (both model loads ran with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`), no Docker pruning, no concurrent model jobs. Downloads were
performed only for the explicitly approved acquisition step. Public organizer input was
not read in this milestone at all. No clinical text was copied into tracked audits,
fixtures or tests.
