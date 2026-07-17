# data/external/public_ner/ — untracked public NER corpora

Place public Vietnamese medical NER datasets here (ViMedNer, ViMQ,
PhoNER_COVID19, …). Git-ignored. Never committed.

Each dataset needs a tracked manifest (`configs/resources/*.manifest.template.yaml`
→ `data/manifests/`) recording labels, proposed MedNorm-VI mappings, license, and
leakage risk BEFORE any use. Licenses are NOT assumed; a human reviewer sets the
license status.
