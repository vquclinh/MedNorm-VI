# Audit 0057: KB Audit, Repair and Candidate Freeze

Milestone: 3A - Knowledge-base audit, repair and freeze  
Date: 2026-07-30  
Verdict: `CANDIDATE_KB_NOT_ACTIVE`

## 1. Initial Git and Protected-Artifact State

Precondition read-only checks passed before any modification:

| Check | Result |
| --- | --- |
| Branch | `main` |
| HEAD | `b12633e feat: freeze the verified framework and container runtime` |
| `origin/main...HEAD` | `0 0` |
| Staged files | none |
| Working tree | clean |
| Audit 0056g | committed in HEAD |

Protected artifacts were verified through streaming SHA-256 only:

| Artifact | SHA-256 |
| --- | --- |
| `docs/MedNorm-VI_Architecture.pdf` | `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b` |
| `checkpoint/s1_mention_full_training_v1/best.pt` | `a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c` |

The checkpoint was not deserialized.

## 2. Resource-Safety Compliance

No notebook, Jupyter kernel, model/tokenizer load, `torch.load`, CUDA,
training/fine-tuning, Docker workload, corpus-wide inference, organizer data,
`internal_test` content, internet access, external API or data download was used.

Thread caps were set for build/test commands:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

Observed resource floor:

| Metric | Observation |
| --- | --- |
| Lowest available RAM | 6.7 GiB |
| Peak observed swap use | 3.4 GiB |
| Largest observed swap delta | +0.6 GiB across the milestone, not continuing rapidly |
| Lowest observed free disk on repo filesystem | 19 GiB |
| Corrected candidate output size | 3.2 GiB |
| Total ignored KB freeze output size, including superseded development runs | 9.6 GiB |

## 3. Existing-KB Inventory

All inspected KB payloads are local repository or local competition-authorized
assets. Raw/generated data remains ignored by Git; manifests, source code, docs,
schemas and tests are tracked.

| Asset | Role | Size/Rows | SHA-256 or Status |
| --- | --- | ---: | --- |
| `data/external/icd10_vi/tt06-2026/raw/06-byt.pdf` | ICD source PDF | 22 MiB source dir total | `1216d8a2f980c56c5c300a7ab00e8cbb063d0ff0b0cfb404cf195eae6dbc5c92` |
| `data/external/icd10_vi/tt06-2026/raw/06-byt-kem.pdf` | ICD source PDF | included above | `8639f5eeb77b571363dc841923095895d2498748f9bc6620f50710a6da9159e2` |
| `data/derived/icd10_vi/tt06-2026/icd10_vi_normalized.csv` | ICD normalized legacy table | 15,308 rows | `7d975de0f8b9bd16ded36e8abd29a3f5b4c4c1864183ec7593943a8fd59dc7d4` |
| `data/derived/icd10_vi/tt06-2026/icd10_vi_hierarchy.csv` | ICD legacy hierarchy | 12,968 rows | `9e26cabad4aff6c5f7717ba31f938e6895ac36cc8ace4d26bd75baf777e32062` |
| `indices/icd10_vi/tt06-2026/index.json` | Active ICD runtime index | 10,138,449 bytes | `94a1378b0f355afc6da23bb0da4d1e015dbbd4f51a1f2c0508f06e7b1727a768` |
| `data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNCONSO.RRF` | Prescribable RxNorm concepts | 245,401 rows | `4dda3b83bb67af5f8349a5fd455a4c54b8290921cffecc362163cd00f27439a5` |
| `data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNREL.RRF` | Prescribable RxNorm relations | 2,563,978 rows | `88c25038ab62a189dd9f744b84552fcaa3d61d2bb9d9dc5a83f4a8a8f19b9c9b` |
| `data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNSAT.RRF` | Prescribable RxNorm attributes | 3,340,763 rows | `a01d4b60c6386d0f4d13304e78ae4c2e1a0397d223b905f191a4361891ae4793` |
| `indices/rxnorm/prescribable-2026-07-06/index.json` | Active RxNorm runtime index | 103,374,362 bytes | `4e02ecbfc61e586e041a424177fa682f048618e8e0431e8204f6d2afc2692aec` |
| `data/external/rxnorm/full-2026-07-06/RxNorm_full_07072025.zip` | Full RxNorm archive | 259,313,098 bytes | `53523ee9f1fcd7ee426698edf566aedebe548a6ec8cc372c41271fc5b28e784c` |
| `data/external/rxnorm/full-2026-07-06/raw/rrf/RXNCONSO.RRF` | Full RxNorm concepts | 1,202,603 rows | `093e26be3a173770aa6d44ab10fe7b3e8c02fce88a0dc94ca7816fab1e3ffece` |
| `data/external/rxnorm/full-2026-07-06/raw/rrf/RXNREL.RRF` | Full RxNorm relations | 7,423,180 rows | `56fd9fe8cfbac95afb445608a0b6f2377b5c3a18b1585a9dfbdd5e1ec64dffaf` |
| `indices/rxnorm/full-2026-07-06/index.json` | Selectable full RxNorm runtime index | 578,218,016 bytes | `9ec49f060683d5af9078c6df58fa95792cc3ca841866ad8de1c743684084c80d` |

Current consumers:

| Area | Runtime Consumer |
| --- | --- |
| ICD lookup | `src/mednorm_vi/linking/icd10.py`, `src/mednorm_vi/linking/snapshot.py` |
| RxNorm lookup | `src/mednorm_vi/linking/rxnorm.py`, `src/mednorm_vi/linking/rxnorm_graph.py` |
| Candidate membership | `src/mednorm_vi/validator/kb_membership.py`, `src/mednorm_vi/validator/final_gate.py` |
| L6/L7/L8 consumers | `src/mednorm_vi/evidence_graph/`, `src/mednorm_vi/metric_decoder/` |

## 4. Reproduced ICD Defects

Static and generated evidence confirmed:

| Finding | Evidence |
| --- | --- |
| Truncated or fragment-derived names | 3,470 names flagged as `SUSPECT_TRUNCATION`; examples include source labels beginning with fragments such as `+ ...`, `- ...` or incomplete function words |
| Parent relationships not source-supported | 12,968 hierarchy rows retained as `legacy_code_prefix_inference`; 0 source-supported hierarchy edges in the 3A contract |
| Records absent from legacy graph | 774 records absent from the active runtime graph, matching previous audit evidence |
| Broken prefixes | 410 4+ character codes lack a valid 3-character prefix record in the legacy active artifacts |
| Lost source provenance | Legacy runtime index lacks the full per-row provenance table now emitted by 3A |

No name was repaired from medical knowledge. Suspect names are flagged for review
and excluded from clean runtime-authoritative status where appropriate.

## 5. Reproduced RxNorm Defects

Static and generated evidence confirmed:

| Finding | Evidence |
| --- | --- |
| Relation labels discarded by active runtime graph | Legacy index stores unlabeled symmetric adjacency; 3A candidate emits 2,563,978 directed labeled relation rows |
| Directionality not authoritative in legacy runtime | Active graph traversal infers from endpoint TTY; 3A relation rows preserve source-to-target direction |
| TTY distinctions need explicit frozen exposure | Candidate concepts preserve all observed TTY values, including `IN`, `PIN`, `SCD`, `SBD` |
| Synonym provenance unclear in legacy runtime | Candidate synonym rows carry source row/provenance IDs |
| Vietnamese aliases not directly present in RxNorm | Legacy INN bridge surfaces remain review inventory |
| INN-to-RxNorm bridge unreviewed | 46 candidate crosswalk rows, all `REQUIRES_REVIEW`, 0 `APPROVED` |

## 6. Source-Authority and No-External-Data Policy

The only source authority used was the existing local repository/resource
inventory. Clinical mappings were not inferred from model knowledge. Uncertain
records were preserved with provenance and review states rather than promoted.

## 7. Schema Contracts

Added versioned, machine-readable contracts:

| Schema ID | File |
| --- | --- |
| `icd10-kb-schema-v1` | `schemas/kb/icd10-kb-schema-v1.json` |
| `rxnorm-kb-schema-v1` | `schemas/kb/rxnorm-kb-schema-v1.json` |
| `kb-provenance-schema-v1` | `schemas/kb/kb-provenance-schema-v1.json` |
| `governed-synonym-schema-v1` | `schemas/kb/governed-synonym-schema-v1.json` |
| `inn-rxnorm-crosswalk-schema-v1` | `schemas/kb/inn-rxnorm-crosswalk-schema-v1.json` |
| `candidate-representation-schema-v1` | `schemas/kb/candidate-representation-schema-v1.json` |
| `kb-snapshot-manifest-schema-v1` | `schemas/kb/kb-snapshot-manifest-schema-v1.json` |

The schema constants are exported from `src/mednorm_vi/kb/freeze.py` and tested
against the tracked files.

## 8. ICD Canonical-Name Disposition

Each ICD concept row includes stable code, display code, lookup code, canonical
display name, original source name, source file, source row identifier, source
version, extraction method, provenance ID, review status, quality flags and
snapshot ID.

Disposition:

| Category | Count |
| --- | ---: |
| Retained ICD concepts | 15,308 |
| Suspect truncation/name-fragment rows | 3,470 |
| Missing source names | 0 |
| Runtime-authoritative concept rows after name-quality gating | 11,640 |

## 9. ICD Hierarchy Repair

The builder no longer treats code-prefix parentage as authoritative source
evidence. It carries every legacy hierarchy edge forward for review:

| Metric | Count |
| --- | ---: |
| Hierarchy edges | 12,968 |
| Source-supported edges | 0 |
| Prefix-inferred review-required edges | 12,968 |
| Missing parent endpoints | 0 |
| Cycles | 0 |

All retained hierarchy rows include parent code, child code, relation label,
direction, source evidence, provenance ID, review status, quality flags and
runtime-authoritative flag.

## 10. RxNorm TTY Preservation

The prescribable candidate freeze preserves every TTY observed in source atoms
rather than collapsing concept types. Key counts:

| TTY | Count |
| --- | ---: |
| `IN` | 5,820 |
| `PIN` | 1,924 |
| `SCD` | 12,055 |
| `SBD` | 8,083 |

The validation report also records all other observed TTY counts.

## 11. RxNorm Relation Labels and Direction

The authoritative candidate relation table contains:

| Metric | Count |
| --- | ---: |
| Directed labeled relation rows | 2,563,978 |
| Distinct `REL` values | 4 |
| Distinct `RELA` values | 35 |
| Endpoint `ok` rows | 553,464 |
| Missing-endpoint rows in prescribable concept subset | 2,010,514 |

Missing endpoints are review evidence. They are not silently removed or converted
into unlabeled adjacency.

## 12. Governed Synonym Table

RxNorm candidate synonym/source-string rows:

| Metric | Count |
| --- | ---: |
| Synonym rows | 245,401 |
| Provenance-bearing source rows | 245,401 |

Rows are labeled by source-native string/provenance and are not promoted as
clinically reviewed aliases unless review state says so.

## 13. INN-RxNorm Crosswalk and Review States

The legacy bridge surfaces were converted into governed crosswalk candidates:

| Review state | Count |
| --- | ---: |
| `APPROVED` | 0 |
| `REQUIRES_REVIEW` | 46 |
| Total | 46 |

All generated crosswalk rows have `runtime_authoritative=false`. The live
structured medication linker now keeps the legacy bridge as review inventory only:
`bridged_ingredient_names("paracetamol") == ()`.

## 14. Vietnamese Medication Mapping Findings

Known findings from local evidence:

| Finding | Disposition |
| --- | --- |
| Vietnamese/INN surfaces may not exist directly in RxNorm | Preserved as review-required crosswalk candidates |
| One local surface can map to multiple RxCUIs through RxNorm terms | Not auto-approved; rows remain review-required |
| Ingredient/product/combo semantics require clinical review | Not inferred by lexical similarity |
| Accent/token normalization can collide | Captured as governance risk; not auto-promoted |

No clinical-document frequency was used because no approved code-bearing
frequency artifact exists for this purpose.

## 15. Candidate Representation Contract

`candidate-representation-schema-v1` defines:

| Field Family | Contract |
| --- | --- |
| Identity | `ontology + code + snapshot_id` |
| Display | canonical display name and source concept type/TTY |
| Provenance | provenance IDs, snapshot ID, synonym/crosswalk evidence |
| Relation evidence | relation path with labels and direction |
| Scoring/order | deterministic score components and deterministic ordering key |
| Governance | review status, runtime eligibility, exclusion reason |
| L5/L8/L9 semantics | offered versus selected is explicit; L8 cannot introduce a code not offered by L5 |

## 16. Provenance Contract

`kb-provenance-schema-v1` records source ontology, source file, source SHA-256,
source version, row identifier, extraction method, builder version, snapshot ID,
review status and quality flags. Every retained 3A output row links to a
provenance ID.

## 17. Deterministic Builders

Added `src/mednorm_vi/kb/freeze.py`.

Commands used for the corrected candidate output:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -m mednorm_vi.kb.freeze build-icd10 --normalized-csv data/derived/icd10_vi/tt06-2026/icd10_vi_normalized.csv --source-manifest data/manifests/icd10-vi-tt06-2026-official.yaml --output-dir data/derived/kb_freeze/0057_candidate_v2/icd10_vi_tt06_2026

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -m mednorm_vi.kb.freeze build-rxnorm --rrf-root data/external/rxnorm/prescribable-2026-07-06/raw/rrf --source-manifest data/manifests/rxnorm-prescribable-2026-07-06.yaml --output-dir data/derived/kb_freeze/0057_candidate_v2/rxnorm_prescribable_2026_07_06 --source-version 2026-07-06 --snapshot-label prescribable-2026-07-06
```

The builders use staged output directories, deterministic row ordering and
separate `artifact_sha256` and `canonical_content_sha256` values. RxNorm relation
and provenance outputs are streamed from local RRF rows.

## 18. Validation Results

Targeted checks:

| Command | Result |
| --- | --- |
| `pytest tests/unit/test_kb_freeze_0057.py tests/unit/test_l5_l7_deterministic_stack.py -q` | `97 passed in 32.48s` |
| `pytest tests/unit/test_kb_freeze_0057.py -q` after final type fix | `6 passed in 0.48s` |

One final full-suite pass:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Result:

```text
1824 passed, 1 skipped in 319.98s (0:05:19)
SKIPPED [1] tests/unit/test_vietmed_adapter.py:399: pyarrow not installed locally
```

Static/type gates:

| Command | Result |
| --- | --- |
| `ruff check .` | passed |
| `ruff format --check <changed Python files>` | `5 files already formatted` |
| `env PYTHONPATH=src .venv/bin/python -m mypy src/mednorm_vi` | `Success: no issues found in 279 source files` |
| `env PYTHONPATH=src .venv/bin/python -m compileall -q src tests` | passed |
| `git diff --check` | passed |

Note: the first `ruff check .` after formatting reported only import-order issues
in `src/mednorm_vi/kb/freeze.py` and `tests/unit/test_kb_freeze_0057.py`; those
were corrected. The first package-wide mypy pass reported `datetime.UTC` as
unavailable in this environment's type stubs; the builder now uses
`datetime.timezone.utc`.

## 19. Snapshot IDs and Hashes

ICD snapshot:

| Artifact | Rows | artifact_sha256 | canonical_content_sha256 |
| --- | ---: | --- | --- |
| `icd10_concepts_v1.csv` | 15,308 | `92f8319690eee9cb2dfd4e715bcb9d996a4886506af7400b558f29694ee13c2f` | `2dfbab71a58c7f937fad69dfc08fba09f09f979945106a656ecdfe9819537555` |
| `icd10_hierarchy_edges_v1.csv` | 12,968 | `4c2e34e131e1dc2e3019259095b12966f3f07e18691f70b3996c31e6949e8c0f` | `d173e9e17a56b9251cf06d2e77d76fbde965c368137186397f7d4df0794c51a5` |
| `icd10_aliases_v1.csv` | 0 | `6685a21f7bbc19602ef3a24f7bd2433f456b4038fadf34027010cb90380098a6` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `icd10_provenance_v1.csv` | 15,308 | `b7d8bb7fc0c1497f6e5482c16cceb501fa2ea45b89fed17a5eaf1130d95bd484` | `aa18908f6d580e25a1fe26b600a62aa014bcd5a008e2c4d815903e7bef94ecac` |
| `icd10_review_queue_v1.csv` | 17,123 | `095e8a576a2e40547ce91d75dce43e7525bd6fef8cd8fcb2bccbd9009407c230` | `1b7732f5ddedef0d6af68c51d00006c4d16022958c0b5cd50f9aa5bbea23bfdd` |

RxNorm snapshot:

| Artifact | Rows | artifact_sha256 | canonical_content_sha256 |
| --- | ---: | --- | --- |
| `rxnorm_concepts_v1.csv` | 82,429 | `abb46e925419e3ddcd7625f69f7d0de6eaf3b8e73dad50ded6cb2cd438b3d970` | `91a035a8f5b174c3e24648e4b1ad223e7adca3bf8eac2b835625d7f56c330b85` |
| `rxnorm_synonyms_v1.csv` | 245,401 | `2e5be39c1dcc9606a88aec2544eaa3b581eb99c6180b0a0787fefa15eec8b2aa` | `1c0c26369b1800fe83b81df382dd21f01adf1508cb3549b7ee1ae770c3326554` |
| `rxnorm_relations_v1.csv` | 2,563,978 | `84fff2c4dca84d88a4c3febe43582e5aff7fde7b7b36b988a30b234c0474689b` | `b4f6f9f6290b8b9620c7980f763d86a12823437c1b73b847e3ee38756229f330` |
| `inn_rxnorm_crosswalk_v1.csv` | 46 | `3884e722317e17517a818188bc8143741a0c5ec52814a60f6839d850aff633d3` | `3dd6eccd44d06b5ab76d9a97392d3b6c5c888e742939087ebdf83076267b0646` |
| `rxnorm_provenance_v1.csv` | 2,809,379 | `d6d7289eeefe2ca36e639e47265afddd5d334b25600f74530a1bb0500747b696` | `9fb25fe091ae42b351c05bf566b389a6635c7f6d7cc481d6a5e16809e6fbcf84` |
| `rxnorm_review_queue_v1.csv` | 6,031,588 | `017bf17f32bd3931dd51362ca227a8251b6e30823170054babaafd5324206f5c` | `72a18046ff12abd8b3667a9fadd5a5de3ad2abe7f376f8f53413c0121b0c0efc` |

## 20. Old-Versus-New Snapshot Migration Comparison

| Area | Current active runtime view | 3A candidate freeze | Migration outcome |
| --- | --- | --- | --- |
| ICD concepts | 15,308 compact index records | 15,308 provenance-bearing concept rows | Coverage preserved |
| ICD hierarchy | Unlabeled graph, direction inferred by code length | 12,968 review-required prefix-inference rows | Not active until hierarchy source evidence is reviewed |
| ICD names | Compact canonical names, some visibly truncated | Suspect names flagged and queued | Not active until name disposition is reviewed |
| RxNorm concepts | 82,429 compact prescribable records | 82,429 provenance-bearing concept rows | Coverage preserved |
| RxNorm TTY | Available in metadata | Preserved explicitly, including IN/PIN/SCD/SBD | Compatible |
| RxNorm graph | Unlabeled symmetric adjacency | Directed labeled relation rows with endpoint status | Not active until relation endpoint policy is approved |
| INN bridge | Previously widened retrieval despite review-required status | Runtime bridge disabled; crosswalk rows review-required | Safer runtime behavior; no unapproved authoritative bridge |
| L9 membership/offered set | Final gate unchanged | Candidate contract preserves invariant | Compatible |

The current active runtime snapshot was not replaced.

## 21. Exact Changed-File Inventory

Tracked modifications:

```text
M docs/architecture/ACTIVE_RUNTIME_MANIFEST.md
M docs/kb/icd10/SNAPSHOT.md
M docs/kb/rxnorm/SNAPSHOT.md
M src/mednorm_vi/linking/rxnorm.py
M src/mednorm_vi/linking/structured_medication.py
M tests/unit/test_l5_l7_deterministic_stack.py
```

New tracked-intended files:

```text
docs/audits/0057-kb-audit-repair-and-freeze.md
docs/kb/KB_FREEZE_0057.md
schemas/kb/candidate-representation-schema-v1.json
schemas/kb/governed-synonym-schema-v1.json
schemas/kb/icd10-kb-schema-v1.json
schemas/kb/inn-rxnorm-crosswalk-schema-v1.json
schemas/kb/kb-provenance-schema-v1.json
schemas/kb/kb-snapshot-manifest-schema-v1.json
schemas/kb/rxnorm-kb-schema-v1.json
src/mednorm_vi/kb/freeze.py
tests/unit/test_kb_freeze_0057.py
```

Ignored generated evidence:

```text
data/derived/kb_freeze/0057_candidate_v2/
```

No file was staged, committed or pushed.

## 22. Unresolved Human-Review Work

| Item | Blocking Runtime Activation? | Disposition |
| --- | --- | --- |
| ICD suspect canonical names | yes | Review and repair only from authorized source evidence |
| ICD hierarchy source evidence | yes | Replace or approve `legacy_code_prefix_inference` rows |
| RxNorm prescribable missing relation endpoints | yes for labeled graph activation | Approve prescribable-subset policy or rebuild runtime view from full relation closure |
| INN-RxNorm crosswalk | yes for bridge activation | Clinical review required; only `APPROVED` rows may become authoritative |
| Code-bearing and assertion gold | no for KB candidate freeze, yes for metrics | Later label/adjudication phase |
| Full RxNorm candidate freeze | later cleanup | Full raw assets were inventoried; active runtime uses prescribable. Full freeze can be run after disk/resource planning |

## 23. Acceptance-Criteria Table

| Criterion | Status |
| --- | --- |
| ICD retained codes unique | pass |
| ICD source evidence retained | pass |
| ICD truncations flagged or repaired from authorized source | pass, flagged only |
| ICD parent links source-supported | not complete; all legacy parent edges review-required |
| ICD cycles absent | pass |
| ICD unresolved parents/ambiguity reported | pass |
| ICD provenance and hashes exist | pass |
| RxNorm TTY preserved | pass |
| IN/PIN/SCD/SBD distinguishable | pass |
| RxNorm relation labels/direction retained | pass in candidate freeze |
| RxNorm relation endpoints validated | pass, with review queue |
| Synonym provenance retained | pass |
| Vietnamese aliases not auto-promoted | pass |
| Crosswalk governed review states | pass |
| Only approved mappings runtime-authoritative | pass, 0 approved |
| Candidate representation schema versioned | pass |
| L9 offered-set invariant preserved | pass |
| No external KB/data downloaded | pass |
| No `internal_test` access | pass; path inventory only, no content read |
| No model/notebook/training/Docker workload | pass |
| Targeted tests pass | pass |
| One final full suite pass | pass |
| Static/type gates pass | pass |
| Nothing staged, committed or pushed | pass |

## 24. Verdict

`CANDIDATE_KB_NOT_ACTIVE`

3A produced the versioned schemas, deterministic builders, provenance-rich
candidate freeze artifacts, review queues, hash ledger, migration report and
active documentation. The artifacts are not promoted to the active runtime KB
because unresolved ICD hierarchy/name quality and RxNorm endpoint/crosswalk
review work remains.

## 25. Next Phase Recommendation

After owner review and commit of this milestone, proceed to governed KB review:

1. adjudicate ICD suspect names against authorized source evidence;
2. replace or approve ICD hierarchy edges with source-supported parent evidence;
3. decide whether the RxNorm active runtime should use prescribable-only labeled
   relations, full-source relation closure or a derived relation subset;
4. clinically review INN-RxNorm crosswalk rows and expose only `APPROVED` rows;
5. only then activate or migrate runtime KB paths.

Do not begin model download, training, fine-tuning, calibration or organizer
inference from this audit alone.
