# Manual Resource Acquisition (Phase 1C-A)

Phase 1C-A downloads **nothing**. Acquisition is a deliberate, manual, documented
step. This is the workflow to follow later, per resource.

## 1. Acquire locally

- **RxNorm** — obtain a licensed RxNorm RRF release (the exact 2026 release the
  organizer uses is undisclosed; record what you actually acquired). Place the
  RRF files under `data/external/rxnorm/` (git-ignored).
- **Vietnamese ICD-10** — obtain a Vietnamese ICD-10 code+label table (source and
  version undisclosed). Place the CSV/TSV under `data/external/icd10_vi/`.
- **Public NER** — obtain each corpus (ViMedNer, ViMQ, PhoNER_COVID19, …) under
  `data/external/public_ner/`, respecting its license.

## 2. Write a manifest

Copy a template from `configs/resources/` to `data/manifests/<resource>.yaml`,
fill every placeholder (version, source, per-file SHA-256, license review status,
intended use), and have a human set `license.status`.

## 3. Validate the manifest

    python -m mednorm_vi.phase1c_foundation.cli validate-resource-manifest \
        --manifest data/manifests/<resource>.yaml [--ner]

Fix every error before use. A resource is usable only when license review
explicitly permits it.

## 4. Ingest / compare (offline)

    python -m mednorm_vi.phase1c_foundation.cli inspect-rxnorm-snapshot --dir data/external/rxnorm
    python -m mednorm_vi.phase1c_foundation.cli compare-rxnorm-snapshots --left A --right B
    python -m mednorm_vi.phase1c_foundation.cli inspect-icd-snapshot \
        --table data/external/icd10_vi/icd10_vi.csv --columns configs/icd_columns.yaml

## 5. Record derived artifacts

Any deterministic transformation (normalized table, snapshot index, label-mapped
corpus) is described by a `DerivedArtifactRecord` in the resource's manifest
(script + checksum), so it can be rebuilt from the raw resource. Derived files
stay under `data/derived/` (git-ignored).

## Checklist before using a resource in training/linking

- [ ] version + source recorded
- [ ] per-file SHA-256 recorded
- [ ] license reviewed by a human (status set)
- [ ] intended use declared and within permitted use
- [ ] leakage risk assessed (NER datasets)
- [ ] manifest validates with no errors
