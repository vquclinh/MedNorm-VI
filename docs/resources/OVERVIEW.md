# External Resource Governance (Phase 1C-A)

Every external dataset or knowledge base (RxNorm, Vietnamese ICD-10, public NER
corpora) must be described by a **tracked manifest** before it is downloaded,
ingested, compared, or used. The manifests are metadata only; the resource
payloads they describe are **never committed**.

## Why the payloads stay untracked

- Licenses vary and may forbid redistribution; committing RxNorm/ICD/NER content
  could breach them and would bloat the repo.
- The organizer may ask us to *describe* our data/generation process — a manifest
  is exactly that description, without shipping the data.
- Raw and derived data live under `data/**` (git-ignored). Only `README.md`,
  `.gitkeep`, and filled-in manifests under `data/manifests/*.yaml` are tracked;
  templates live under `configs/resources/`.

## No automatic legal conclusions

Validation enforces **governance completeness**, not legality. A human reviewer
sets `license.status` ∈ `LICENSE_UNKNOWN`, `REVIEW_REQUIRED`,
`PERMITTED_INTERNAL_USE`, `REDISTRIBUTION_RESTRICTED`, `REJECTED`. A resource is
*usable* only under `PERMITTED_INTERNAL_USE` / `REDISTRIBUTION_RESTRICTED`. The
validator fails a resource that is used without a version, source, checksum,
license review status, or intended-use declaration.

## Why public NER datasets are not immediately "gold"

`NerDatasetManifest` records original labels, proposed MedNorm-VI label mappings,
incompatible labels, boundary-convention differences, assertion/coding
availability, redistribution + privacy constraints, split provenance, and leakage
risk. A public corpus can use a different label set, different span boundaries,
and a different annotation guideline than the organizer's; and it may share
documents with our dev data (leakage). So a dataset is treated as **evidence to
review and map**, never as drop-in gold. Label mappings are versioned and
auditable (`LabelMappingSet`), and a mapping target is a MedNorm entity type or
`DROP` / `REVIEW`.

## Commands

    # validate a resource or NER manifest
    python -m mednorm_vi.phase1c_foundation.cli validate-resource-manifest \
        --manifest data/manifests/<resource>.yaml [--ner]

See `ACQUISITION.md` for the manual acquisition workflow and
`schemas/resources/RESOURCE_MANIFEST.md` for the field reference.
