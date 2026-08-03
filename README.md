# MedNorm-VI

Vietnamese clinical information extraction with governed ontology normalization: given a
clinical note, MedNorm-VI finds the clinical entities, decides what is asserted about them,
and links diagnoses and medications to governed ICD-10 and RxNorm concepts.

All inference is local. No external API is called, nothing is trained or fine-tuned at run
time, and every emitted code comes from the governed knowledge base.

## Architecture

One pipeline, one inference path.

```
clinical document
  ├─ proposal sources ── E3 mention proposals
  │                    ├ Qwen3-8B contextual passes (diagnosis/symptom, medication, lab)
  │                    └ governed exact-alias support
  ├─ deterministic exact-source alignment      offsets computed locally, never by a model
  ├─ finite span/type lattice                  a closed set of alternatives
  ├─ Qwen finite verifier                      accept / reject / retype, by local index only
  ├─ deterministic span and type resolution
  ├─ contextual assertions
  └─ verified organizer entities
        ├─ Qwen query reformulation            two extra retrieval views, never entities
        ├─ multi-view retrieval ── sparse character n-gram (BioSyn-style)
        │                        ├ SapBERT-XLMR
        │                        └ ClinLinker-KB-GP  (diagnosis only)
        ├─ governed candidate union            deduplicated, deterministically capped
        ├─ ontology / structured pruning       type gate plus evidenced conflicts
        ├─ Qwen3-8B set-wise reranking         one option, or NONE
        └─ organizer JSON                      one governed concept per mention, or none
```

Two properties hold by construction. A model never produces a character offset, so every span
is a literal slice of the source document. A model never produces a concept identifier — it
chooses an option letter — so every emitted code came from the governed knowledge base.

## Entity types and output schema

Five organizer types:

| type | meaning | assertions | candidates |
|---|---|---|---|
| `TRIỆU_CHỨNG` | symptom or sign | yes | — |
| `CHẨN_ĐOÁN` | named disease or condition | yes | ICD-10 |
| `THUỐC` | medication | yes | RxNorm |
| `TÊN_XÉT_NGHIỆM` | name of a test | — | — |
| `KẾT_QUẢ_XÉT_NGHIỆM` | result a named test produced | — | — |

```json
[
  {"position": [12, 21], "text": "viêm phổi", "type": "CHẨN_ĐOÁN",
   "assertions": ["isHistorical"], "candidates": ["J18.9"]},
  {"position": [30, 37], "text": "glucose", "type": "TÊN_XÉT_NGHIỆM"}
]
```

The two laboratory types carry `position`, `text` and `type` alone — an empty `assertions`
list on them is an unsupported field, not a harmless default.

**Assertions.** `isNegated` scopes to the entity's own clause, so "no fever, has cough" does
not negate the cough. `isFamily` requires a kinship reference that is the subject of the
clause and precedes the entity; first-person or patient narration cancels it, and words that
are also Vietnamese address pronouns count only inside a declared family-history section.
`isHistorical` comes from the section or an explicit past-time marker — chronicity alone is
not history, because a chronic condition is a current one.

**Linking.** A diagnosis may only receive an ICD-10 code and a medication may only receive an
RxCUI. Candidates are pruned when governed metadata *proves* a conflict — a stated strength,
dose form or route contradicting a recorded one — and never when an attribute is merely
absent. The reranker sees the mention, its context and the candidates' ontology facts; it
never sees a similarity score, a rank or a retriever name.

## Deployed models

| model | role | licence |
|---|---|---|
| Qwen3-8B | contextual proposal, verification, reformulation, reranking | Apache-2.0 |
| SapBERT-XLMR | multilingual concept retrieval | MIT |
| ClinLinker-KB-GP | hierarchy-aware diagnosis retrieval | Apache-2.0 |
| E3 (ViHealthBERT) | mention proposal | — |

**8,729,759,237 deployed parameters**, under the 9,000,000,000 cap. The count sums every
model the system deploys, not the ones resident at any one moment: sequential loading reduces
peak memory, not what was deployed. Sparse character retrieval is an inverted index and
contributes no parameters. `run_mednorm.py budget` certifies this from the model manifest and
fails if the stack has moved.

Model revisions are resolved to immutable commits **before** download and recorded in
`model-manifest.json`; a floating branch name is refused.

## Governed knowledge base

Candidate identifiers come only from the governed ICD-10 and RxNorm indices under
`indices/candidate/`. Governed alias matches are a *proposal* source: they may support a
mention another source found, but an alias-only span never becomes an entity on its own.
Alias quality is measured from the knowledge base — how many concepts a surface form names,
how common a token is across labels, and the index's own label-quality flags — rather than
from any hand-written word list.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock
```

Required assets, none of which are tracked in this repository:

| asset | location |
|---|---|
| governed ICD-10 / RxNorm indices | `indices/candidate/` |
| structured RxNorm attributes | `artifacts/rxnorm_structured/` |
| organizer documents | `data/organizer_test/input/` |
| E3 checkpoint | `checkpoint/` |
| Qwen3-8B snapshot | local directory, pinned revision |

## Running

```bash
# 1. acquire the frozen retrievers at pinned commits, and certify the stack
python scripts/run_mednorm.py acquire --allow-download --model-root MODELS

# 2. embed the governed knowledge base once per retriever
python scripts/run_mednorm.py index --allow-download --model-root MODELS --cache-root CACHE

# 3. verify the deployed parameter budget
python scripts/run_mednorm.py budget --model-root MODELS

# 4. inference
python scripts/run_mednorm.py run --allow-download \
    --input-dir data/organizer_test/input \
    --seed-entities E3_ENTITIES \
    --model-root MODELS --cache-root CACHE --qwen-root QWEN \
    --output-dir runs/production

# 5. validate, archive and hash
python scripts/run_mednorm.py package --output-dir runs/production \
    --input-dir data/organizer_test/input --expected-documents 100
```

Every stage that touches weights requires `--allow-download`, an explicit statement that this
is a GPU host, and refuses to start below the memory floor for that stage.

[docs/colab/final-production-commands.md](docs/colab/final-production-commands.md) is the
same workflow for a fresh Colab runtime.

## Reproducibility

A run is identified by two artifacts kept together: the SHA256 in `package-summary.json`
identifies the archive, and `model-manifest.json` identifies the weights that produced it.
Semantic caches carry the document checksum, row-order checksum, pooling contract, dtype and
model revision, and retrieval refuses a cache built by different weights. Generation is
greedy throughout.

## Repository structure

```
src/mednorm_vi/
    production/   the pipeline: extraction then linking, one pass
    extraction/   proposal, alignment, lattice, verification, resolution, assertions
    linking/      reformulation, sparse and dense retrieval, pruning, set-wise reranking
    models/       registry, pinned acquisition, encoders, the Qwen host, budget certification
    kb/           governed indices, ontology facts, semantic cache contract
    packaging/    dotted-code derivation, deterministic archive and hashing
    validation/   organizer schema and submission gates
    config.py     runtime configuration

scripts/run_mednorm.py            the only entrypoint
configs/pipeline/production.yaml  the only profile
docs/audits/                      development provenance
```

## Development history

The final system was reached through a long series of audited experiments — earlier entity
extractors, semantic linkers, candidate-tier policies and span correction passes — each
recorded under `docs/audits/`. Those experiments are history: the working tree contains the
final system only, and the git history preserves the rest.
