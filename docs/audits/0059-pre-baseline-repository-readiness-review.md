# Audit 0059: Pre-baseline Repository, Task-contract, Runtime and Submission Readiness Review

Milestone: 3C - Pre-baseline readiness review
Date: 2026-07-31
Verdict: `BASELINE_READY_WITH_NOTES`

## 1. Initial Git State

Read-only precondition checks, run before any other work:

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `5ba04dc feat: refresh competition v3 runtime and bridge observability` |
| Working tree | **clean** |
| Staged files | none |
| `git rev-list --left-right --count origin/main...HEAD` | `0 0` |
| Audit 0058B committed | **yes** — 5 files in `5ba04dc` |

The protected architecture PDF is unmodified
(`0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b`). It was read in
full earlier in this session and neither modified nor moved.

### 1a. State change observed during the review

The tree was clean at the precondition check. Later in the review it contained:

```text
?? runs/baseline_v0_profile_a/derived-manifest.json
?? runs/baseline_v0_profile_b/derived-manifest.json
?? runs/baseline_v0_profile_c/run-manifest.json
?? runs/baseline_v0_profile_c/run.log
?? runs/baseline_v0_profile_c/run.pid
```

**This review did not create them.** Timestamps are 17:18 (profile C) and 17:25
(profiles A and B); the writing process (`pid 77031`) had already exited. They are
untracked artifacts and Git-ignored ZIPs — **no tracked source file changed**, and the
working tree is still clean of modifications.

What they are is established in §12: a complete 100-document `specialist` baseline was
run, and profiles A and B were derived from it. That is squarely outside this
milestone's mandate ("Do not begin organizer/public-test inference. Do not generate
leaderboard submission ZIP files"), and it was not performed by this review. It is
recorded here as a finding, not adopted as this milestone's work — but it is genuinely
useful evidence, and §12 uses it rather than pretending not to have seen it.

## 2. Task-contract Sources Read

**The complete organizer task specification is not stored in this repository.** No
organizer PDF, rulebook or scoring script is tracked. What is tracked is a governed
reconstruction, and it is explicit about which parts are confirmed and which are not.

| Source | What it fixes |
| --- | --- |
| `docs/MedNorm-VI_Architecture.pdf` | Metric weights (candidates 40%, text 30%, assertions 30%), Jaccard candidate scoring, 9B base-parameter cap, Appendix A submission checklist |
| `configs/organizer/confirmed_facts_v1.yaml` | 14 CONFIRMED facts with named sources: submission layout, five labels, multi-label assertions, offset convention, KB not provided, TEST_NAME/TEST_RESULT independence |
| `configs/organizer/unresolved_policies_v1.yaml` | Explicit hypothesis slots for what is **not** disclosed — position coordinate space, char-vs-byte, line endings, interval convention, match mode |
| `docs/task_contracts/btc_turn2_vong1_contract_review.md` | Turn2/vong1 review: output contract materially unchanged; patient-info and relations `UNSUPPORTED_BY_OUTPUT_SCHEMA` |
| `configs/organizer/active_input.yaml` | Active input provenance, hashes, 100 documents |
| `src/mednorm_vi/schemas/constants.py` | The enforced vocabulary (verified live, §11) |

The contract that is actually **supported by tests, architecture and documentation**:

```text
input          100 flat UTF-8 .txt files, 1.txt … 100.txt
output         one JSON list per input, N.txt -> N.json
entity types   TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC
assertions     isNegated, isFamily, isHistorical (multi-label)
               emitted ONLY for SYMPTOM / DIAGNOSIS / MEDICATION
candidates     CHẨN_ĐOÁN -> ICD-10, THUỐC -> RxNorm; cross-links forbidden;
               no candidates on SYMPTOM / TEST_NAME / TEST_RESULT
offsets        half-open [start, end), Unicode code points,
               original_text[start:end] == text        (POSITION_IS_END_EXCLUSIVE = True)
package        output.zip containing exactly one output/ directory with 1.json … 100.json
metric         candidates 40% (Jaccard), text 30% (1-WER), assertions 30% (Jaccard)
parameters     9,000,000,000 base-model cap; adapters/heads excluded
reproducibility  local paths only, no external API at inference, one command to output.zip
```

**Residual ambiguity, honestly carried:** the end-inclusive/exclusive question is
resolved to end-exclusive by a prior organizer example, not by the current wording; the
graded coordinate space and match mode remain `unresolved` in the policy registry. The
repository does not paper over this — `unresolved_policies_v1.yaml` names each open
question, its options and the internal default in force until a leaderboard probe
resolves it.

### Canonical runner, identified from evidence

Not guessed from directory names. The brief's `src/mednorm_vi/pipeline/`,
`src/mednorm_vi/runtime/` and `src/mednorm_vi/runner/` **do not exist**. The evidence:

- `Dockerfile:205` — `ENTRYPOINT ["python", "-m", "mednorm_vi.inference.cli"]`
- `mednorm_vi/inference/pipeline.py` is the only runtime module calling
  `resolve_lattice_to_hypotheses` (the canonical L4); the other callers are the
  Phase-1C debug CLI and the doctor.

Canonical runner: **`src/mednorm_vi/inference/cli.py` -> `inference/pipeline.py`**.
Docker invokes the same Python pipeline through that entrypoint.

## 3. Architecture Conformance Summary

| Spec area | Conformance |
| --- | --- |
| §1 metric-driven design | Followed. L8's top-k policy is derived from Jaccard's shape (Audit 0058 §7) |
| §4 offset preservation | Followed and enforced four times, plus on the serialized bytes (`validator/final_gate.py`) |
| §7.3 no candidates on symptom/test types | Enforced; verified live over 100 documents (§11) |
| §9/§10 linkers | Deterministic lexical + hierarchy + structured/TTY graph. No dense retrieval, no reranker |
| §12 LLM cascade | Contract complete, **no model** — `ESCALATE` never returned |
| §13 metric-aware decoding | **Not implemented as specified.** L8 is deterministic evidence-ranked plus top-k; `decode_expected_jaccard_calibrated` fails closed |
| §16 fail-fast | Followed: L9 gates the serialized payload before anything is written |
| §17 9B budget | 134,998,272 deployed (§7) |
| Appendix A | Satisfied except "one command to output.zip on real input", which is the next milestone |

The architecture describes a *candidate super-architecture*; the repository implements
a deterministic subset with one trained expert. `ACTIVE_RUNTIME_MANIFEST.md` is the
authority on the difference and remains accurate.

## 4. Completed-component Inventory

| Component | Classification | Evidence |
| --- | --- | --- |
| L1 document intelligence | `IMPLEMENTED_AND_ACTIVE` | `document_intelligence/` + `position/`; runs on every document |
| L2 case router | `IMPLEMENTED_AND_ACTIVE` | `case_router/`; `signals_v1` + `required_evidence` gating (Audit 0053) |
| L3 mention lattice | `IMPLEMENTED_AND_ACTIVE` | `mention_factory/registry.py` + `lattice/` |
| E1 medication grammar | `IMPLEMENTED_AND_ACTIVE` | `ExpertSpec(implementation='DETERMINISTIC_PHASE1B')`; 250 THUỐC on the synthetic package |
| E2 laboratory parser | `IMPLEMENTED_AND_ACTIVE` | Same; 300 TÊN_XÉT_NGHIỆM + 350 KẾT_QUẢ on the synthetic package |
| E3 ViHealthBERT | `IMPLEMENTED_AND_ACTIVE` | `implementation='REGISTERED'`; ran in this review's smoke (`experts_that_ran == ('E3_vihealthbert_span_type',)`) |
| E4 PhoBERT-W2NER | `RETIRED` | Absent from `AVAILABLE_EXPERTS`; `governance/e4_retirement.py` |
| E5 XLM-R MRC | `ABSENT` | `NOT_IMPLEMENTED_OR_NOT_REGISTERED`; named blocker: randomly-initialized head |
| E6 GLiNER | `ABSENT` | Same; blocker: no verified local checkpoint |
| E7 Qwen proposer | `ABSENT` | Same; blocker: no local weights, proposal contract unimplemented |
| L4 boundary/type | `IMPLEMENTED_AND_ACTIVE` | `resolution/canonical.py`; `enable_l4_deterministic_v1: true` |
| L4 learned v2 | `IMPLEMENTED_BUT_NOT_ACTIVE` | `enable_l4_learned_v2: false`; `loaded_at_inference: false` |
| ICD-10 linking | `IMPLEMENTED_AND_ACTIVE` | `linking/icd10.py`; 199 CHẨN_ĐOÁN linked in the observed baseline |
| RxNorm linking | `IMPLEMENTED_AND_ACTIVE` | `linking/rxnorm.py` + `rxnorm_graph.py` |
| Governed INN bridge | `IMPLEMENTED_AND_ACTIVE` | 50 expansions recorded in this review's 100-doc manifest |
| Assertion handling | `PARTIAL` | Deterministic A1-A3 only; S2 head `loaded_at_inference: false`, data-blocked |
| L6 evidence graph | `IMPLEMENTED_AND_ACTIVE` | `evidence_graph/graph.py` + `consistency.py`, 9 edges / 12 rules |
| L7 confidence cascade | `PARTIAL` | Contract + deterministic fallback; no model stage can load |
| L8 decoding | `PARTIAL` | Evidence-ranked + competition top-k active; §13 calibrated path fails closed |
| L9 validation | `IMPLEMENTED_AND_ACTIVE` | Passed on 100 synthetic documents with zero contract violations (§11) |
| Output serialization | `IMPLEMENTED_AND_ACTIVE` | Verified field-exact per type over 100 documents |
| Run manifest | `IMPLEMENTED_AND_ACTIVE` | Written; `contains_clinical_text: false` |
| Packaging | `IMPLEMENTED_AND_ACTIVE` | 100-file ZIP, single `output/` root, fixed timestamp |
| Active KB v3 indices | `IMPLEMENTED_AND_ACTIVE` | §6 |
| Rollback paths | `IMPLEMENTED_AND_ACTIVE` | Legacy indices byte-identical; declared in config |
| Local runtime | `IMPLEMENTED_AND_ACTIVE` | 100-document CLI run completed, exit 0 |
| Offline container | `IMPLEMENTED_AND_ACTIVE` | `mednorm-vi:competition-v3-prebaseline`, verified Audit 0058B |

## 5. Incomplete-component Inventory

| Gap | Class | Note |
| --- | --- | --- |
| Public-test package absent | — | **Not a gap**: present and verified (§10) |
| Code-bearing ICD/RxNorm gold | `DOES_NOT_BLOCK_BASELINE` | Zero gold codes; candidate Jaccard `UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD`. The leaderboard is the measurement |
| Assertion gold | `DOES_NOT_BLOCK_BASELINE` | Zero assertion supervision; S2 is code-ready and data-blocked |
| TEST entity gold | `DOES_NOT_BLOCK_BASELINE` | E3 has zero TEST_NAME/TEST_RESULT supervision; E2 covers them deterministically |
| Unvalidated top-k thresholds | `DOES_NOT_BLOCK_BASELINE` | 0.98 / 0.95 / 0.80 reasoned, not fitted. Profiles A/B/C exist to probe exactly this |
| Unmeasured candidate Jaccard | `DOES_NOT_BLOCK_BASELINE` | Same cause |
| Advisory ICD hierarchy unused by ranker | `POST_BASELINE_IMPROVEMENT` | Emitted and weighted; not read by `linking/icd10.py` |
| `name_quality` unused by ranker | `POST_BASELINE_IMPROVEMENT` | In the index metadata; not consumed |
| Labelled RxNorm relation graph unused | `POST_BASELINE_IMPROVEMENT` | 960,086 directed labelled relations in the snapshot; the walk still infers by TTY |
| E5/E6/E7 not selected/downloaded | `DOES_NOT_BLOCK_BASELINE` | Each fails closed with a named blocker |
| Calibration (S6) | `DOES_NOT_BLOCK_BASELINE` | No out-of-fold predictions; §13 decoder fails closed by design |
| Training/fine-tuning | `DOES_NOT_BLOCK_BASELINE` | Only E3 is trained; the baseline is deliberately a deterministic+E3 reference |
| Per-document index reload | `POST_BASELINE_IMPROVEMENT` | §8 — costs ~4.8 min per 100-document run |
| Profile-derivation tooling untracked | `FINAL_SUBMISSION_BLOCKER` | §12 — the A/B derivation is ad-hoc and not in Git, so the submission is not reproducible from the repository |
| Final-container status | `DOES_NOT_BLOCK_BASELINE` | `competition-v3-prebaseline` verified; must be named the final image before submission |
| Model parameter ledger | — | **Not a gap**: verified (§7) |

## 6. Active Runtime Identity

Verified by ID and hash, not by tag:

| Field | Value | Verified |
| --- | --- | --- |
| Image | `mednorm-vi:competition-v3-prebaseline` | — |
| Image ID | `sha256:381ad95e23f5837bb6ab1c1aa1717918c7a69710969317fc13a1ae56a44f2137` | **matches expected** |
| ICD index | `indices/candidate/icd10_vi/competition-v3/index.json` | present |
| ICD `deterministic_index_hash` | `b050a2d5ead90c8410101d257b08b347c129f5a8faa713d7e5f2757c7931a975` | matches Audits 0058/0058A |
| RxNorm index | `indices/candidate/rxnorm/competition-v3/index.json` | present |
| RxNorm `deterministic_index_hash` | `d3c45f349bad1532135b70214c48e54c270ede6ae6988e1444a9949887d92ac3` | matches Audits 0058/0058A |
| KB snapshot ID (both) | `competition-kb-v3-candidate-4d206538e7d24c32` | matches |
| Records | ICD 15,308 / RxNorm 211,949 (82,429 searchable + 129,520 closure-only) | matches |

Rollback indices present and byte-identical to their pre-activation hashes:
`94a1378b…1727a768`, `4e02ecbf…2692aec`, `9ec49f06…84c80d`.

**Stale-image risk is real and named.** Four `mednorm-vi` images exist:
`competition-v3-prebaseline` (25 min old, the only one baking the v3 config),
`framework-frozen` (19 h), `2d-c-verified` (20 h), `framework-closed` (25 h). Only the
first may be used for the baseline; the others bake pre-activation configs. No
container was left running.

## 7. Model / Checkpoint / Parameter Ledger

E3 checkpoint verified by **streaming hash before any load**:

```text
checkpoint/s1_mention_full_training_v1/best.pt
size   1,615,513,303 bytes
sha256 a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c   MATCHES
```

Ledger from `configs/models/candidate_model_registry.yaml`
(`parameter_count_verified: true`, method `counted_from_config`):

| Component | Parameters | Active at inference? |
| --- | ---: | --- |
| `e3_vihealthbert_span_type` | **134,998,272** | **YES** |
| `l4_learned_resolver_v2` | 6,542 | no (`enable_l4_learned_v2: false`) |
| `s2_assertion_head` | 134,998,272 + 9,228 adapter | no (data-blocked) |

```text
deployed base parameters       134,998,272
competition cap              9,000,000,000
remaining budget             8,865,001,728   (98.5% unused)
```

E3 is `demdecuong/vihealthbert-base-word` pinned at revision
`f89e80b461e86f9cfc1c84019bd819830c24b6c5`.

**Rules and KB assets do not count.** E1/E2 are declared
`parameter_count_status='NOT_APPLICABLE'`; the ICD/RxNorm indices are data, not model
weights. Spec §17 caps base-model parameters.

Two nuances worth stating rather than smoothing over. The deployment template
`template_minimal_verified` sums 135,004,814 because it *selects*
`l4_learned_resolver_v2`, which the active profile does not load — the template is
conservative relative to the real runtime. And `ExpertSpec` reports E3's
`parameter_count_status='NOT_VERIFIED'`; that field describes what *readiness* checks
at load time, not the model registry's verified count. The two are not in conflict.

Unknown values: none required for the baseline. No model was downloaded.

## 8. Local Hardware and Resource Assessment

| Resource | Observation |
| --- | --- |
| CPU | AMD Ryzen 7 8745H, 16 threads (8 cores × 2) |
| RAM | 14 GiB total, 7.3–8.0 GiB available throughout |
| GPU | NVIDIA RTX 4060 Laptop, 8,188 MiB, driver 580.159.04 |
| torch | 2.13.0+cu126, `cuda.is_available() == True` |
| Disk | 25 G free |

**One E3 load was performed, exactly as permitted** (checkpoint hash verified first,
two tracked synthetic documents, single process):

```text
specialist readiness                     READY, no errors
doc 1 (includes E3 + index load)         22.86 s   18 entities
doc 2 (warm, model reused)                6.17 s    6 entities
L9 final gate                            PASSED
peak RSS                                  2.69 GiB
CUDA memory allocated                     0.000 GiB  ->  E3 ran on CPU
```

**E3 runs on CPU despite CUDA being available.** VRAM is therefore not a constraint and
the GPU is currently irrelevant to the baseline.

### A measured performance defect

`run_document` calls `_indexes(config)` **per document** (`pipeline.py:270`), reloading
both indices from JSON every time:

```text
ICD index load     0.33 s
RxNorm index load  2.56 s     -> ~2.9 s of pure redundant work per document
```

Over 100 documents that is **~4.8 minutes** of avoidable cost, plus two further loads
in `_gate_final_output` and `_write_manifest`. It is a performance defect, not a
correctness one, and §13 of the brief forbids changing source merely to make this
review pass — so it is recorded, with the minimal fix specified in §15, and not
implemented here.

Measured 100-document run (this review, deterministic, small synthetic documents):
**458 s wall**. The observed specialist baseline over the real 100 documents took
**409 s** (§12). Both are comfortably feasible locally.

### Recommendation: local canonical CLI first

Run the baseline through the **local canonical CLI**, then confirm reproducibility in
**Docker**. Do **not** use Colab: E3 runs on CPU, peak RSS is 2.69 GiB against 8 GiB
free, and the whole 100-document specialist run completes in under seven minutes. A GPU
environment would add upload, environment drift and organizer-data handling risk for no
measured benefit.

## 9. Exact Canonical Baseline Command

From `inference/cli.py`, verified live:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  .venv/bin/python -m mednorm_vi.inference.cli run \
    --input-dir  data/organizer_test/input \
    --output-zip runs/<profile>/output.zip \
    --config     configs/pipeline/full_v1.yaml \
    --mode       specialist \
    --run-manifest runs/<profile>/run-manifest.json
```

| Aspect | Behaviour |
| --- | --- |
| Config | `--config`, required |
| Input directory | `--input-dir`, required; globs `*.txt`, numeric-stem ordering |
| Output directory | **No flag.** Derived as the sibling `output/` of `--output-zip` |
| Mode | `--mode {deterministic,specialist,full}`, required |
| Checkpoint path | Not a flag; resolved by the E3 adapter from `checkpoint_root` |
| Index paths | Not flags; from the config's `icd_index` / `rxnorm_index` |
| Manifest | `--run-manifest`, opt-in; written even when L9 stops the run |
| ZIP packaging | Automatic; requires exactly `expected_documents` (100) files or raises |
| Failure behaviour | L9 violation -> exit 3, **no `output/` and no ZIP written** |
| Resume/overwrite | No resume. `write_output_directory` refuses unowned destinations; an existing ZIP is unlinked and rewritten |

Docker invokes the same pipeline (`ENTRYPOINT ["python","-m","mednorm_vi.inference.cli"]`)
with `indices/`, `checkpoint/`, `/input` read-only and `/output` writable.

## 10. Public-test Package Readiness

Verified structurally, **without reading any clinical text and without inference**:

| Check | Result |
| --- | --- |
| Location | `data/organizer_test/input` |
| File count | **100**, all `.txt`, no extra entries |
| Names | exactly `1.txt` … `100.txt` |
| Missing / duplicate IDs | none |
| Regular files, no symlinks | yes |
| UTF-8 decode | all 100 succeed |
| BOM / CRLF | 0 files with BOM, 0 with CRLF |
| Sizes | min 1,673 B, median 2,378 B, mean 2,651 B, max 5,862 B, total 265,147 B |
| Payload tree hash | `d7ace1838a3433ba40ca3a9df5a8fc3a86be5a2a7ca989c8361e2175dd58620d` |
| Matches `configs/organizer/active_input.yaml` | **yes, exactly** |

`internal_test/` **does not exist** in this repository, so there was no possibility of
touching it. No clinical text was read, quoted or copied.

The output writer expects `1.json … 100.json` (`N_SUBMISSION_DOCUMENTS = 100`) and the
ZIP root is `output/` (`SUBMISSION_DIR_NAME`).

## 11. Output and ZIP Contract

Verified end to end on **100 synthetic scratch documents** built only from tracked
fixtures, outside tracked data. No organizer input was used for this smoke.

```text
CLI exit 0, wall 458 s, processed_documents: 100
output/*.json count            100, names 1..100 exact
zip entries                    100, single root {'output'}, all under output/
zip stems                      1..100 exact
zip sha256                     d6e3259ecf63c88e62d6bafd77c23fe8c2e983e893375890f8c5c9590d4af680
CONTRACT VIOLATIONS            NONE
```

Checked on every entity of all 100 files:

| Requirement | Result |
| --- | --- |
| Each file a JSON list, each entity a dict | pass |
| `original_text[start:end] == text` | pass (end-exclusive) |
| Type in the five organizer labels | pass |
| Per-type field set exactly `ORGANIZER_FIELDS_BY_TYPE` | pass |
| Candidates only on candidate-bearing types | pass — only `THUỐC` appeared |
| Assertions only on symptom/diagnosis/medication | pass — only `THUỐC` appeared |
| Assertion values ⊆ {isNegated, isFamily, isHistorical} | pass |
| Deterministic ordering | pass (Audit 0058B; fixed-timestamp ZIP) |
| Exactly 100 files required for packaging | pass — enforced, and it *failed* correctly at 2 files |

Run manifest: `governed_inn_bridge_expansions: 50`, candidate sizes
`{0: 650, 1: 125, 2: 125}`, `contains_clinical_text: false`, `l9_stopped_the_run: false`.

`CHẨN_ĐOÁN` and `TRIỆU_CHỨNG` are absent from this smoke because the synthetic fixtures
are medication/laboratory text and it ran in `deterministic` mode. Diagnosis-side
contract behaviour is covered by the observed specialist baseline in §12.

## 12. Three-profile Feasibility

Classification: **`ONE_PASS_THREE_DECODES_READY`** — but by a route the canonical
runner does not itself provide, and the tooling is not in Git.

### What the code supports

Re-decoding inside the pipeline is **not** possible today. `decode_entities` needs
`hypotheses, cascade, assertions, links`; `PipelineResult` retains none of those four.
Top-k thresholds are module constants (`MEDICATION_TIE_RATIO` etc.), and
`decode_entities` exposes only a boolean `competition_topk`, not a policy. The CLI has
no `--profile` flag.

### What was actually done

The profiles were produced by running C once and **deriving A and B from C's output
JSON by candidate-list truncation**:

```text
profile C   specialist, 100 documents, 957 entities, runtime 409.3 s, L9 did not stop
            git commit 5ba04dcf…, KB snapshot competition-kb-v3-candidate-4d206538e7d24c32
            E3 sha256 a64cc173…, output_zip_sha256 29ef6670…
profile A   derived from C, medication top-1 + diagnosis top-1,  123 lists truncated
profile B   derived from C, medication top-1 + diagnosis adaptive, 4 lists truncated
```

This review **verified the derivations are sound** rather than trusting them:

```text
DERIVATION VIOLATIONS: NONE
  - text, type, position and assertions identical across A, B and C on all 957 entities
  - A and B candidate lists are exact PREFIXES of C's (ordering preserved, nothing introduced)
  - A never emits more than 1 candidate
```

That is valid precisely because `apply_competition_topk` returns an order-preserving
sub-sequence, so a stricter policy's output is a prefix of a looser one's.

### Are all three worth a leaderboard attempt?

Measured on the real 100-document output:

```text
candidate-bearing entities        210   (199 CHẨN_ĐOÁN + 11 THUỐC)
                       0 candidates: 20
                       1 candidate:  67
                       2 candidates: 123

A vs C differ on 123 entities
B vs C differ on   4 entities        <-- of 957
A vs B differ on 119 entities
```

**Profiles B and C differ on 4 entities out of 957.** Spending two leaderboard attempts
to separate them is close to worthless — any score difference will be inside the noise
of a 100-document set. The real question is A versus B/C: *is diagnosis adaptive top-2
worth it?*, worth 119 entities.

Recommendation: submit **A** and **B** (or A and C, not both B and C). Under Jaccard,
A is the conservative reference — a wrong second candidate halves that mention's
contribution — and B isolates the diagnosis-side bet. If only one attempt is available,
submit A.

## 13. Reproducibility Record Contract

Each profile's baseline run must preserve:

| Field | Source |
| --- | --- |
| Git commit SHA | `git rev-parse HEAD` (manifest `git.commit`) |
| Image ID | `docker image inspect --format '{{.Id}}'` |
| Architecture PDF hash | `0d5eaa20…81e09b` |
| E3 checkpoint hash | `a64cc173…1017c` |
| KB snapshot ID | `competition-kb-v3-candidate-4d206538e7d24c32` |
| ICD index hash | `b050a2d5…931a975` |
| RxNorm index hash | `d3c45f34…d92ac3` |
| Full config hash | manifest `config_hashes` |
| Profile name | A / B / C |
| Decoding thresholds | 0.98 / 0.95 / 0.80, `MAX_CANDIDATES = 2` |
| Input manifest hash | `d7ace183…58620d` |
| Output directory | `runs/<profile>/output/` |
| Output ZIP SHA-256 | recorded per profile |
| Run manifest | `runs/<profile>/run-manifest.json` |
| Entity counts by type | from the output |
| Empty / one / two-candidate counts | manifest `decoder_candidate_sizes` |
| Governed bridge expansion count | manifest `aggregates.linking.governed_inn_bridge_expansions` |
| Warning / error counts | manifest aggregates |
| Runtime duration | manifest `runtime_seconds` |
| Peak RAM / VRAM | `peak_memory_gib`; VRAM n/a (E3 on CPU) |
| L9 result | manifest `l9_stopped_the_run` + `l9_issues` |
| Submission time and score | to be filled after submission |

**Two gaps in the already-produced record**, which the next milestone must close:

1. `git.dirty: true` — the run was made against an unclean tree, so the commit SHA does
   not fully identify the code that ran.
2. The A/B derivation tool is **not tracked in Git**, so those two ZIPs cannot be
   regenerated from the repository. This is why it is a `FINAL_SUBMISSION_BLOCKER`.

Nothing was submitted by this review.

## 14. Baseline Blockers

**None.** Every precondition for running the baseline is satisfied: the public-test
package is present and hash-verified, the runtime identity matches by ID and hash, the
canonical command is confirmed, the output/ZIP contract passes on 100 documents, the
parameter budget is 98.5% unused, and the machine completes the run in under seven
minutes.

## 15. Non-blocking Risks

1. **Profile-derivation tooling is untracked** (`FINAL_SUBMISSION_BLOCKER` for the final
   submission, not for the baseline). Minimal fix: commit the derivation script, or add
   `--profile` to the CLI by parameterising `apply_competition_topk` with a policy
   dataclass (defaults unchanged) and threading it through `decode_entities`.
2. **A baseline was run outside its milestone** (§1a), against a dirty tree. Re-run it
   from a clean tree under the next milestone so the record is reproducible.
3. **Per-document index reload** costs ~4.8 min per 100 documents. Minimal fix: hoist
   `_indexes(config)` out of `run_document` and pass the loaded indices in, as
   `prepared_experts` already does for E3.
4. **Stale images**: three older `mednorm-vi` images bake pre-activation configs.
5. **B and C are near-identical** (4 entities) — do not spend two attempts on them.
6. **Candidate accuracy remains unmeasurable** offline; the leaderboard is the first
   real measurement.
7. **Residual contract ambiguity** (coordinate space, match mode) is unresolved and
   documented; a probe submission is the only way to settle it.

## 16. Exact Files Inspected

```text
docs/MedNorm-VI_Architecture.pdf                    (hash-verified; read in full this session)
docs/audits/0051 … 0058b                            (all)
docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
docs/kb/KB_COMPETITION_0058.md, docs/kb/icd10/SNAPSHOT.md, docs/kb/rxnorm/SNAPSHOT.md
docs/task_contracts/btc_turn2_vong1_contract_review.md
configs/organizer/{confirmed_facts_v1,unresolved_policies_v1,active_input}.yaml
configs/pipeline/full_v1.yaml, configs/linking/rxnorm_snapshots_v1.yaml
configs/parameter_budget.yaml, configs/models/{candidate_model_registry,deployment_budget_template}.yaml
Dockerfile, .dockerignore, README.md
src/mednorm_vi/inference/{cli,config,pipeline,manifest,packaging,serialization}.py
src/mednorm_vi/linking/, src/mednorm_vi/metric_decoder/, src/mednorm_vi/validator/
src/mednorm_vi/schemas/constants.py
src/mednorm_vi/mention_factory/expert_spec.py
tests/unit/{test_competition_v3_activation_0058a,test_organizer_output_contract,test_rxnorm_full_intake}.py
tests/fixtures/phase1b/*.txt
data/organizer_test/input/          (metadata only: names, sizes, encoding, hashes — no content read)
runs/baseline_v0_profile_{a,b,c}/   (manifests and output metadata; no clinical text quoted)
```

## 17. Files Created by This Review

Inside the repository: **only this audit**,
`docs/audits/0059-pre-baseline-repository-readiness-review.md`.

No source file, config, test or fixture was modified. All smoke artifacts (100
synthetic documents, their `output/`, `output.zip` and run manifest) were written to the
session scratchpad outside the repository. The `runs/` artifacts are **not** this
review's (§1a).

## 18. Verdict

`BASELINE_READY_WITH_NOTES`

Every hard precondition is met and verified by evidence rather than assertion: the
public-test package matches its recorded hash exactly, the runtime image and both KB
indices match their expected IDs and hashes, the canonical command is confirmed from
the code, the full output and ZIP contract passes on 100 documents with zero violations,
the deployed parameter total is 134,998,272 against a 9B cap, and the machine runs the
100-document specialist baseline in under seven minutes on CPU.

The notes are governance and reproducibility, not capability: a baseline was run outside
its milestone against a dirty tree, the profile-derivation tooling is untracked, and an
avoidable per-document index reload costs about five minutes per run.

## 19. Exact Next Action

1. Commit this audit. Nothing was staged, committed or pushed.
2. Commit or discard the untracked `runs/` artifacts so the tree is clean.
3. Track the profile-derivation tooling — or add `--profile` to the CLI — so all three
   ZIPs are reproducible from the repository.
4. Re-run the baseline from a **clean tree** with the §9 command, capturing the §13
   record for each profile.
5. Submit **Profile A** first as the conservative reference, then **Profile B**. Do not
   spend a separate attempt on C: it differs from B on 4 of 957 entities.
6. Do not begin training, fine-tuning, calibration or model downloads from this audit.
