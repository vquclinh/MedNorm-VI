# RxNorm KB Snapshot Status (Milestone 3A)

Milestone 3A produced a versioned candidate RxNorm KB freeze from the existing
local prescribable 2026-07-06 RRF files. No external RxNorm, drug database, model
or API was downloaded or used.

The candidate snapshot is **not active**. The currently verified runtime index
remains:

```text
indices/rxnorm/prescribable-2026-07-06/index.json
```

## Candidate Snapshot

| Field | Value |
| --- | --- |
| Snapshot ID | `rxnorm-prescribable-2026-07-06-kb-v1-candidate-d7181e39bd4053e7` |
| Generated path | `data/derived/kb_freeze/0057_candidate_v2/rxnorm_prescribable_2026_07_06/` |
| Source manifest | `data/manifests/rxnorm-prescribable-2026-07-06.yaml` |
| Source RRF root | `data/external/rxnorm/prescribable-2026-07-06/raw/rrf/` |
| Validation status | `CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED` |
| Runtime activation | Deferred |

Generated CSV/manifest artifacts under `data/derived/` are ignored by Git. The
tracked contract is in `schemas/kb/rxnorm-kb-schema-v1.json`; the build entry
point is `python -m mednorm_vi.kb.freeze build-rxnorm`.

## 3A Counts

| Metric | Count |
| --- | ---: |
| Concepts | 82,429 |
| Synonym/source-string rows | 245,401 |
| Directed labeled relation rows | 2,563,978 |
| Relation rows with endpoints in the prescribable concept table | 553,464 |
| Relation rows with missing endpoints in the prescribable concept table | 2,010,514 |
| Distinct `REL` values | 4 |
| Distinct `RELA` values | 35 |
| IN concepts | 5,820 |
| PIN concepts | 1,924 |
| SCD concepts | 12,055 |
| SBD concepts | 8,083 |
| Governed INN-RxNorm crosswalk rows | 46 |
| Approved crosswalk rows | 0 |
| Review-required crosswalk rows | 46 |
| Review queue rows | 6,031,588 |

## Disposition

The candidate freeze preserves RxNorm TTY values and the source relation
semantics (`REL`, `RELA`, row identity and source-to-target direction). This fixes
the auditability gap in the legacy runtime index, whose graph remains an unlabeled
symmetric adjacency view.

The local INN-to-RxNorm bridge is now inventory only. Every generated crosswalk row
is `REQUIRES_REVIEW` and `runtime_authoritative=false`; no row can widen runtime
candidate retrieval until it is explicitly approved in a governed crosswalk.

The large missing-endpoint count is expected for the prescribable subset because
many relation endpoints are outside the prescribable concept table. It is retained
as review evidence rather than hidden.

## Hashes

| Artifact | Rows | artifact_sha256 | canonical_content_sha256 |
| --- | ---: | --- | --- |
| `rxnorm_concepts_v1.csv` | 82,429 | `abb46e925419e3ddcd7625f69f7d0de6eaf3b8e73dad50ded6cb2cd438b3d970` | `91a035a8f5b174c3e24648e4b1ad223e7adca3bf8eac2b835625d7f56c330b85` |
| `rxnorm_synonyms_v1.csv` | 245,401 | `2e5be39c1dcc9606a88aec2544eaa3b581eb99c6180b0a0787fefa15eec8b2aa` | `1c0c26369b1800fe83b81df382dd21f01adf1508cb3549b7ee1ae770c3326554` |
| `rxnorm_relations_v1.csv` | 2,563,978 | `84fff2c4dca84d88a4c3febe43582e5aff7fde7b7b36b988a30b234c0474689b` | `b4f6f9f6290b8b9620c7980f763d86a12823437c1b73b847e3ee38756229f330` |
| `inn_rxnorm_crosswalk_v1.csv` | 46 | `3884e722317e17517a818188bc8143741a0c5ec52814a60f6839d850aff633d3` | `3dd6eccd44d06b5ab76d9a97392d3b6c5c888e742939087ebdf83076267b0646` |
| `rxnorm_provenance_v1.csv` | 2,809,379 | `d6d7289eeefe2ca36e639e47265afddd5d334b25600f74530a1bb0500747b696` | `9fb25fe091ae42b351c05bf566b389a6635c7f6d7cc481d6a5e16809e6fbcf84` |
| `rxnorm_review_queue_v1.csv` | 6,031,588 | `017bf17f32bd3931dd51362ca227a8251b6e30823170054babaafd5324206f5c` | `72a18046ff12abd8b3667a9fadd5a5de3ad2abe7f376f8f53413c0121b0c0efc` |

## Migration Rule

Do not switch runtime configs to the 3A candidate RxNorm freeze until relation
endpoint policy and INN crosswalk review are owner-approved. The existing runtime
index remains the verified framework-freeze artifact, with the unapproved INN
bridge disabled for candidate retrieval.
