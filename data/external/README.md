# data/external/ — untracked external resource payloads

Raw external datasets and knowledge bases live here **locally only**. Everything
under `data/**` is git-ignored except `README.md` / `.gitkeep` markers — no
RxNorm, ICD-10, or NER files are ever committed.

Layout:

- `rxnorm/` — a locally acquired RxNorm RRF snapshot (RXNCONSO/RXNREL/RXNSAT/…).
- `icd10_vi/` — a locally acquired Vietnamese ICD-10 table.
- `public_ner/` — public Vietnamese medical NER corpora (ViMedNer, ViMQ, …).

Each resource **must** have a tracked manifest under `data/manifests/` (or a
template under `configs/resources/`) describing version, source, checksums,
license review status, and intended use before it may be ingested. See
`docs/resources/OVERVIEW.md` and validate with:

    python -m mednorm_vi.phase1c_foundation.cli validate-resource-manifest \
        --manifest data/manifests/<resource>.yaml

Nothing here is downloaded automatically; acquisition is a manual, documented
step (`docs/resources/ACQUISITION.md`).
