# ICD-10 Vietnamese KB Snapshot Status (Milestone 3A)

Milestone 3A produced a versioned candidate ICD-10 KB freeze from the existing
local TT06-2026 derived CSV. No external ICD data was downloaded or used.

The candidate snapshot is **not active**. The currently verified runtime index
remains:

```text
indices/icd10_vi/tt06-2026/index.json
```

## Candidate Snapshot

| Field | Value |
| --- | --- |
| Snapshot ID | `icd10-vi-tt06-2026-kb-v1-candidate-52c4de3abfd25dc4` |
| Generated path | `data/derived/kb_freeze/0057_candidate_v2/icd10_vi_tt06_2026/` |
| Source manifest | `data/manifests/icd10-vi-tt06-2026-official.yaml` |
| Source table | `data/derived/icd10_vi/tt06-2026/icd10_vi_normalized.csv` |
| Validation status | `CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED` |
| Runtime activation | Deferred |

Generated CSV/manifest artifacts under `data/derived/` are ignored by Git. The
tracked contract is in `schemas/kb/icd10-kb-schema-v1.json`; the build entry point
is `python -m mednorm_vi.kb.freeze build-icd10`.

## 3A Counts

| Metric | Count |
| --- | ---: |
| Concepts | 15,308 |
| Hierarchy edges retained for review | 12,968 |
| Source-supported hierarchy edges | 0 |
| Inferred parent edges requiring review | 12,968 |
| Suspect canonical names | 3,470 |
| Missing names | 0 |
| Duplicate code rows in freeze output | 0 |
| Review queue rows | 17,123 |

## Disposition

The prior normalized ICD table preserved useful local codes and labels, but its
hierarchy evidence is not source-supported in the frozen contract. Parent edges
are retained as `legacy_code_prefix_inference`, marked `REQUIRES_REVIEW`, and
`runtime_authoritative=false`.

Canonical names that look like PDF fragments or truncated extraction output are
flagged as `SUSPECT_TRUNCATION`. They are not repaired from medical knowledge.
Only another existing authorized source record can repair a name.

## Hashes

| Artifact | Rows | artifact_sha256 | canonical_content_sha256 |
| --- | ---: | --- | --- |
| `icd10_concepts_v1.csv` | 15,308 | `92f8319690eee9cb2dfd4e715bcb9d996a4886506af7400b558f29694ee13c2f` | `2dfbab71a58c7f937fad69dfc08fba09f09f979945106a656ecdfe9819537555` |
| `icd10_hierarchy_edges_v1.csv` | 12,968 | `4c2e34e131e1dc2e3019259095b12966f3f07e18691f70b3996c31e6949e8c0f` | `d173e9e17a56b9251cf06d2e77d76fbde965c368137186397f7d4df0794c51a5` |
| `icd10_aliases_v1.csv` | 0 | `6685a21f7bbc19602ef3a24f7bd2433f456b4038fadf34027010cb90380098a6` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `icd10_provenance_v1.csv` | 15,308 | `b7d8bb7fc0c1497f6e5482c16cceb501fa2ea45b89fed17a5eaf1130d95bd484` | `aa18908f6d580e25a1fe26b600a62aa014bcd5a008e2c4d815903e7bef94ecac` |
| `icd10_review_queue_v1.csv` | 17,123 | `095e8a576a2e40547ce91d75dce43e7525bd6fef8cd8fcb2bccbd9009407c230` | `1b7732f5ddedef0d6af68c51d00006c4d16022958c0b5cd50f9aa5bbea23bfdd` |

## Migration Rule

Do not switch `configs/pipeline/full_v1.yaml` or the runtime index paths to the
3A candidate snapshot until human review resolves the truncated-name and
source-supported hierarchy queues, or an owner-approved policy explicitly accepts
the candidate limitations.

## Milestone 3B Update (Audit 0058)

Milestone 3B is the "owner-approved policy explicitly accepts the candidate
limitations" branch above, for competition purposes only.

Searchability now follows **structural code validity**, not name quality. All 15,308
concepts are structurally valid and searchable, including the 3,470 with suspect names:
a fragment label does not make its code wrong, and a concept the runtime refuses to
index can never be retrieved. Suspect status became a negative ranking feature
(`name_quality`) instead of an exclusion. No name is repaired and `source_name` is
preserved verbatim.

The 12,968 prefix-inferred hierarchy edges are retained as **advisory**. They may
contribute a weak ranking feature but `may_offer_candidate` is `false` on every edge,
so advisory hierarchy can never introduce a code that lexical evidence did not already
reach. Exact lexical evidence outweighs it 1000:1.

| Field | Value |
| --- | --- |
| Snapshot ID | `competition-kb-v3-candidate-4d206538e7d24c32` |
| Path | `data/derived/kb_freeze/0058_competition_candidate_v3/` |
| Searchable / excluded | 15,308 / 0 |
| Suspect names (searchable, penalised) | 3,470 |
| Advisory hierarchy edges | 12,968 |
| Edges that may offer a candidate | 0 |
| Runtime activation | Staged; see `docs/kb/KB_COMPETITION_0058.md` |

No manual review of thousands of names or hierarchy edges is required before
competition activation. This is a competition-scoring judgement, **not** a clinical
one: no ICD record in this snapshot claims human adjudication.
