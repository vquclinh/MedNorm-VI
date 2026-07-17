# Resource manifest — field reference (tracked)

Human-readable schema for `ResourceManifest`
(`src/mednorm_vi/resources/models.py`) and `NerDatasetManifest`
(`src/mednorm_vi/resources/ner.py`). Manifests are tracked; the resource payloads
they describe are **not**. No field here implies a legal conclusion — a human
reviewer sets `license.status`.

## ResourceManifest (RxNorm / ICD-10 / generic)

| field | required to USE | meaning |
| ----- | :-------------: | ------- |
| `resource_id` | ✓ | stable id |
| `title` | | human title |
| `source.{organization,url,description}` | ✓ (org or url) | provenance |
| `snapshot.{snapshot_id,version,release_date,acquisition_date}` | ✓ (version or id) | identity |
| `license.{name,status,text_reference,permits_redistribution,notes}` | ✓ (status) | review record |
| `redistribution.{permission,organizer_submission_implications,notes}` | | redistribution policy |
| `files[]` `{path,sha256,bytes}` | ✓ (≥1 with sha256) | integrity |
| `permitted_use[]` / `intended_use[]` | ✓ (intended) | `domain_adaptation`, `ner_training`, `assertion_training`, `linking`, `evaluation` |
| `raw_path` / `processed_path` | | untracked local locations |
| `transformation.{script,description,deterministic,output_paths}` | | how derived artifacts are built |
| `label_schema[]` | | e.g. loader column mapping |
| `acquisition.{acquired_by,acquisition_date,method,notes}` | | manual acquisition record |
| `provenance`, `review_status`, `reviewer` | | governance |
| `unresolved_legal_questions[]` | | open questions |
| `derived_artifacts[]` | | deterministic outputs |

`license.status` ∈ `LICENSE_UNKNOWN`, `REVIEW_REQUIRED`, `PERMITTED_INTERNAL_USE`,
`REDISTRIBUTION_RESTRICTED`, `REJECTED`. A resource is *usable* only under
`PERMITTED_INTERNAL_USE` or `REDISTRIBUTION_RESTRICTED`.

## NerDatasetManifest (public NER corpora)

Adds: `original_labels[]`, `label_mappings[]`
(`source_label → canonical_intermediate → target`), `incompatible_labels[]`,
`boundary_convention`, `assertion_availability`, `normalization_availability`,
`redistribution_constraints`, `privacy_considerations`, `split_provenance`,
`leakage_risk`, `permitted_use[]`. A mapping `target` is a MedNorm entity type or
`DROP` / `REVIEW`.

Validate any manifest with:

    python -m mednorm_vi.phase1c_foundation.cli validate-resource-manifest --manifest <path>
