# scripts/

Operational entry points (planned; not implemented at bootstrap). Per the
architecture spec (section 19), the intended commands are:

- `prepare` — build the DocumentGraph / data prep
- `train`   — per-stage training (S0-S6)
- `index`   — build ICD/RxNorm retrieval indices from frozen KB snapshots
- `infer`   — run the end-to-end inference flow (section 16)
- `evaluate`— local metric clone (text WER, assertion/candidate Jaccard)
- `package` — validate and produce `output.zip`

Appendix A requires a single command from the input directory to `output.zip`,
with all models/tokenizers/KB/indices resolved from local paths (no external
API). Add scripts here as each layer is implemented.
