# indices/

Root for **generated retrieval indices** (ICD-10 / RxNorm multi-representation
indexes: exact-alias, char n-gram, BM25/sparse, dense embeddings, hierarchy
graph). Everything under this directory except this README is git-ignored.

- Indices are **generated artifacts** — rebuildable from the frozen KB snapshots
  in `data/kb/`. Never commit them.
- Each index must record the KB snapshot id/hash it was built from, so results
  are reproducible for the private-test rebuild.

No indices are built at the bootstrap stage.
