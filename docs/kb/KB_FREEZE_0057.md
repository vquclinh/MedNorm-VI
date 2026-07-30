# Milestone 3A KB Freeze Report

Milestone 3A audited and rebuilt candidate KB freeze artifacts from the existing
local ICD-10 Vietnamese and RxNorm resources. The generated artifacts are
candidate freeze outputs, not active runtime replacements.

## Source Authority

No internet, external APIs, external medical datasets, model downloads, notebook
execution, Docker workload, training, `internal_test` access or organizer-test
input was used.

| Source | Path | SHA-256 |
| --- | --- | --- |
| Architecture PDF | `docs/MedNorm-VI_Architecture.pdf` | `0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b` |
| E3 checkpoint, integrity only | `checkpoint/s1_mention_full_training_v1/best.pt` | `a64cc173a284e42ff4bc21b6e0914314d6ff2c6c13efd7fc04d7be0f9be1017c` |
| ICD manifest | `data/manifests/icd10-vi-tt06-2026-official.yaml` | `c19c1a22aabd6e8b7e833e9924fb38781e0475c9e6f4f1d9a73f69df0a26226b` |
| RxNorm full manifest | `data/manifests/rxnorm-full-2026-07-06.yaml` | `1efe98631f2a65b70ee11d4ed69772db82688e1d716cd35c41f21b4bacff36ff` |
| RxNorm prescribable manifest | `data/manifests/rxnorm-prescribable-2026-07-06.yaml` | `24df3c2015497abda21e8ae58a48837124f32df3baa827c6f36d4814c4c5bcae` |
| ICD normalized source table | `data/derived/icd10_vi/tt06-2026/icd10_vi_normalized.csv` | `7d975de0f8b9bd16ded36e8abd29a3f5b4c4c1864183ec7593943a8fd59dc7d4` |
| RxNorm prescribable `RXNCONSO.RRF` | `data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNCONSO.RRF` | `4dda3b83bb67af5f8349a5fd455a4c54b8290921cffecc362163cd00f27439a5` |
| RxNorm prescribable `RXNREL.RRF` | `data/external/rxnorm/prescribable-2026-07-06/raw/rrf/RXNREL.RRF` | `88c25038ab62a189dd9f744b84552fcaa3d61d2bb9d9dc5a83f4a8a8f19b9c9b` |

## Versioned Contracts

| Contract | File |
| --- | --- |
| `icd10-kb-schema-v1` | `schemas/kb/icd10-kb-schema-v1.json` |
| `rxnorm-kb-schema-v1` | `schemas/kb/rxnorm-kb-schema-v1.json` |
| `kb-provenance-schema-v1` | `schemas/kb/kb-provenance-schema-v1.json` |
| `governed-synonym-schema-v1` | `schemas/kb/governed-synonym-schema-v1.json` |
| `inn-rxnorm-crosswalk-schema-v1` | `schemas/kb/inn-rxnorm-crosswalk-schema-v1.json` |
| `candidate-representation-schema-v1` | `schemas/kb/candidate-representation-schema-v1.json` |
| `kb-snapshot-manifest-schema-v1` | `schemas/kb/kb-snapshot-manifest-schema-v1.json` |

The builder is `src/mednorm_vi/kb/freeze.py` with version
`kb-freeze-builder-v1`. It writes to a staged directory first, validates and
hashes logical rows deterministically, then promotes the staged output.

## Candidate Snapshots

| Ontology | Snapshot ID | Status |
| --- | --- | --- |
| ICD-10 Vietnamese | `icd10-vi-tt06-2026-kb-v1-candidate-52c4de3abfd25dc4` | `CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED` |
| RxNorm prescribable | `rxnorm-prescribable-2026-07-06-kb-v1-candidate-d7181e39bd4053e7` | `CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED` |

Generated artifacts are under `data/derived/kb_freeze/0057_candidate_v2/` and are
ignored by Git through the repository `data/**` policy.

## ICD Findings

The ICD candidate freeze retains all 15,308 local records with provenance and
stable normalized codes. It flags 3,470 suspect names that look truncated or
fragment-derived. No name was repaired from general medical knowledge.

The existing hierarchy is not source-supported in the available normalized table.
All 12,968 parent edges are retained as `legacy_code_prefix_inference`,
`REQUIRES_REVIEW`, and `runtime_authoritative=false`. There are no self-parent
or cycle findings in the candidate output.

## RxNorm Findings

The RxNorm candidate freeze preserves TTY distinctions, including 5,820 `IN`,
1,924 `PIN`, 12,055 `SCD` and 8,083 `SBD` concepts. It emits 2,563,978 directed
labeled relation rows with `REL`, `RELA`, source row identity and endpoint status.

The prescribable concept subset leaves 2,010,514 relation rows with missing
endpoints. Those rows are reported in the review queue instead of being hidden or
converted into unlabeled symmetric graph edges.

The legacy 12-surface INN bridge generated 46 possible local-surface/RxCUI rows.
All 46 are `REQUIRES_REVIEW`; 0 are `APPROVED`; 0 are runtime-authoritative.

## Candidate Representation

`candidate-representation-schema-v1` defines a stable candidate identity as
`ontology + code + snapshot_id`. It carries canonical display name, source concept
type or TTY, provenance IDs, matching evidence, relation paths with labels and
direction, synonym or crosswalk evidence, deterministic score components, ordering
key, review/governance status, runtime eligibility and explicit offered/selected
semantics.

The existing L9 invariant is preserved: L8 cannot introduce a code absent from
the L5 offered set, and snapshot-valid but unoffered codes fail closed.

## Migration Outcome

The verified runtime configs still point at:

```text
indices/icd10_vi/tt06-2026/index.json
indices/rxnorm/prescribable-2026-07-06/index.json
```

The 3A candidate freeze is not activated because it contains review queues for
ICD source-supported hierarchy/name quality and RxNorm relation endpoint/crosswalk
policy. Runtime compatibility remains intact because organizer output contracts,
canonical runner paths and L9 offered-set validation were not changed.

## Verdict

`CANDIDATE_KB_NOT_ACTIVE`

The framework remains frozen. The next phase should be human KB review and
owner-approved activation/migration of the candidate KB artifacts, not model
training or organizer inference.
