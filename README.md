# MedNorm-VI

<div align="center">

### Vietnamese Clinical Entity Extraction and Ontology Normalization for Viettel AI Race 2026

Offline, governed-KB inference system for Track 2 — entities, assertions, and ICD-10 / RxNorm linking.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Track](https://img.shields.io/badge/Viettel%20AI%20Race-Track%202-red)
![Parameters](https://img.shields.io/badge/Deployed%20Params-8.73B%20%3C%209B-brightgreen)
![Runtime](https://img.shields.io/badge/Runtime-Fully%20Offline-success)
![Training](https://img.shields.io/badge/Training-None%20(Pretrained%20Only)-informational)
![Output](https://img.shields.io/badge/Output-Organizer%20JSON-orange)

</div>

## What MedNorm-VI is

MedNorm-VI reads a Vietnamese clinical note and produces structured organizer JSON: every clinical
entity with its **exact character span**, its **type**, its **clinical assertions**, and — for
diagnoses and medications — its **governed ontology candidates**.

It runs **four local open-weight models** in a single sequential pipeline (8,729,759,237 deployed
parameters, under the 9B cap) with **no external API and no internet at inference time**. Nothing
is trained, fine-tuned, or adapted at run time; all four models are frozen at pinned commits.

The system is built so that its two most dangerous failure modes are **structurally impossible**
rather than merely unlikely: a model never emits a character offset, and a model never emits an
ontology code.

## Competition / runtime constraints

Built for **Viettel AI Race 2026 — Track 2**, which requires local inference under a hard deployed
parameter ceiling. Submissions are scored on three components: transcription fidelity of extracted
spans, agreement on assertions, and agreement on ontology candidates.

Because an incorrect candidate is penalised, the pipeline is **precision-oriented end to end**: it
abstains rather than guesses whenever the text does not support a concept. `NONE` is a first-class
answer at the final decision stage, not a failure path.

## Final architecture

A single inference path. No alternative pipeline, no variant family, no post-hoc selection between
competing runs.

```
clinical document
  │
  ├─ Proposal sources ─────── E3 mention proposals (ViHealthBERT)
  │                        ├─ Qwen3-8B contextual extraction
  │                        │    · diagnosis / symptom pass
  │                        │    · medication pass
  │                        │    · laboratory pass
  │                        └─ governed exact-alias support
  │
  ├─ Deterministic exact-source alignment    offsets computed locally, never by a model
  ├─ Finite span / type lattice              a closed set of alternatives
  ├─ Qwen3-8B finite verifier                accept / reject / retype, by local index
  ├─ Deterministic span and type resolution
  ├─ Contextual assertion pass
  │
  └─ Verified organizer entities
       │
       ├─ Qwen3-8B query reformulation       two additional retrieval views
       ├─ Multi-view retrieval ──────────── sparse character n-gram
       │                                 ├─ SapBERT-XLMR
       │                                 └─ ClinLinker-KB-GP  (diagnosis only)
       ├─ Governed candidate union           deduplicated, deterministically capped
       ├─ Ontology / structured pruning      type gate plus evidenced conflicts
       ├─ Qwen3-8B set-wise reranking        one option, or NONE
       │
       └─ Organizer JSON
```

Models load **sequentially and are released between phases** — Qwen for extraction, then the
retrievers one at a time, then Qwen again for reranking. Peak device memory therefore tracks the
largest single model rather than their sum, which is what allows the full pipeline to run on a
22.5 GiB card.

## Design invariants

| Invariant | Mechanism |
|---|---|
| **No hallucinated spans** | A model never emits a character offset. It returns a line identifier and text; offsets are resolved by locating that text in the source document. Text absent from the document cannot be aligned and is discarded, never repaired. |
| **No invented concept identifiers** | A model never emits an ontology code. It selects an option label from a list this system constructed from the governed KB. A literal code appearing in a model response is refused outright. |
| **No unsupported specificity** | A candidate adding a subtype, site, laterality, acuity, strength, form or route that the text does not state is a different concept — the reranker is instructed to answer `NONE` rather than assume. |
| **Fail closed** | An unparseable model response, an out-of-range index, or a malformed selection resolves to the safe outcome (drop the proposal / emit no candidate), never to a guess. |

All four are enforced by regression tests, not assumed.

## Output schema

| Type | Meaning | `assertions` | `candidates` |
|---|---|:---:|:---:|
| `TRIỆU_CHỨNG` | Symptom or sign | ✓ | — |
| `CHẨN_ĐOÁN` | Named disease or condition | ✓ | ICD-10 |
| `THUỐC` | Medication | ✓ | RxNorm |
| `TÊN_XÉT_NGHIỆM` | Name of a test or measurement | — | — |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Result a named test produced | — | — |

```json
[
  {
    "position": [12, 21],
    "text": "viêm phổi",
    "type": "CHẨN_ĐOÁN",
    "assertions": ["isHistorical"],
    "candidates": ["J18.9"]
  },
  {
    "position": [30, 37],
    "text": "glucose",
    "type": "TÊN_XÉT_NGHIỆM"
  }
]
```

The two laboratory types carry `position`, `text` and `type` only. An empty `assertions` array on
those types is an **unsupported field**, not a harmless default, and packaging refuses a submission
containing one.

### Assertions

`isNegated`, `isFamily`, `isHistorical` — computed only after spans and types are final, so an
assertion always describes a fixed entity.

- **`isNegated`** — scoped to the entity's own clause. A cue does not cross a clause boundary, so
  *"không sốt, có ho khan"* negates the fever but not the cough.
- **`isFamily`** — requires an unambiguous kinship reference that is the clause subject and
  precedes the entity. First-person or patient-reference narration cancels it. Terms that are
  simultaneously Vietnamese address pronouns (*ông*, *bà*, *cô*, *chú*) count only inside a
  declared family-history section.
- **`isHistorical`** — from section context or an explicit past-time marker. Chronicity alone is
  insufficient: a chronic condition is a *current* condition.

### Ontology linking

Diagnoses receive only ICD-10; medications receive only RxNorm; the other three types are never
linked. Candidates are pruned when governed metadata **proves** a conflict — a stated strength,
dose form or route contradicting a recorded one — and never when an attribute is merely absent.

The set-wise reranker sees the mention, its context and each candidate's ontology facts. It does
**not** see similarity scores, retrieval ranks or retriever names: a similarity value is not
semantic evidence, and exposing one anchors the model to whatever the retrievers happened to
favour. Candidate ordering is deterministic and independent of retrieval rank.

## Deployed models

| Model | Role | Parameters | Licence |
|---|---|---:|---|
| `Qwen/Qwen3-8B` | Extraction, verification, reformulation, reranking | 8,190,735,360 | Apache-2.0 |
| `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | Concept retrieval (diagnosis + medication) | 278,043,648 | MIT |
| `ICB-UMA/ClinLinker-KB-GP` | Hierarchy-aware retrieval (diagnosis only) | 125,978,112 | Apache-2.0 |
| E3 (ViHealthBERT) | Mention proposal | 135,002,117 | Local checkpoint |
| | **TOTAL** | **8,729,759,237** | **< 9,000,000,000** |

The total sums **every model the system deploys**, not those simultaneously resident. Sequential
loading reduces peak memory; it does not reduce what was deployed. Sparse character retrieval is an
inverted index and contributes **zero** parameters.

`run_mednorm.py budget` certifies this figure from the model manifest and exits non-zero if the
stack has changed. Model revisions are resolved to **immutable commit SHAs before download** and
recorded in `model-manifest.json` — a branch name is refused as a revision, since it cannot
identify the weights that produced a result.

## Governed knowledge base

Every candidate identifier originates from the governed ICD-10 and RxNorm indices under
`indices/candidate/`. No other candidate source exists in the system.

Governed alias matches are a **proposal source, not an entity source**: an alias may support a
mention another source found, but an alias-only span never becomes an entity on its own. Alias
quality is measured from the KB itself — how many concepts a surface form names, how frequently a
token occurs across labels, and the index's own label-quality metadata — never from a
hand-maintained word list.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
pip install -e ".[dev]"
```

Python 3.10+ and a CUDA device for inference.

### Required assets — none tracked in this repository

| Asset | Location |
|---|---|
| Governed ICD-10 / RxNorm indices | `indices/candidate/` |
| Structured RxNorm attributes | `artifacts/rxnorm_structured/` |
| Organizer documents | `data/organizer_test/input/` |
| E3 checkpoint | `checkpoint/` |
| Qwen3-8B snapshot | local directory, pinned revision |

### Hardware

| Stage | Minimum VRAM |
|---|---|
| `acquire`, `index` | 6 GiB |
| `run` | 18 GiB |
| Recommended | 40 GiB |

A 22.5 GiB NVIDIA L4 runs the full pipeline. `package` requires no GPU.

## Usage

One entrypoint, five commands.

```bash
# 1 — Acquire the frozen retrievers at pinned commits and certify the stack
python scripts/run_mednorm.py acquire --allow-download \
    --model-root MODELS --qwen-root QWEN --e3-checkpoint CHECKPOINT

# 2 — Embed the governed knowledge base once per retriever
python scripts/run_mednorm.py index --allow-download \
    --model-root MODELS --cache-root CACHE

# 3 — Verify the deployed parameter budget
python scripts/run_mednorm.py budget --model-root MODELS

# 4 — Inference
python scripts/run_mednorm.py run --allow-download \
    --input-dir data/organizer_test/input \
    --seed-entities E3_ENTITIES \
    --model-root MODELS --cache-root CACHE --qwen-root QWEN \
    --output-dir runs/production

# 5 — Validate, archive and hash
python scripts/run_mednorm.py package \
    --output-dir runs/production \
    --input-dir data/organizer_test/input \
    --expected-documents 100
```

Any stage that loads model weights requires `--allow-download` — an explicit assertion that the
host is a GPU machine — and refuses to start below that stage's memory floor. For a fresh Google
Colab runtime, see [`docs/colab/final-production-commands.md`](docs/colab/final-production-commands.md).

### Produced artifacts

| File | Contents |
|---|---|
| `output.zip` | The submission archive |
| `package-summary.json` | Archive path and SHA-256 |
| `diagnostic-summary.json` | Per-stage counters, peak VRAM, certified parameter total |
| `mention-records.jsonl` | Per-mention candidates, retrieval provenance and selection |
| `proposal-records.jsonl` | Per-document proposals, sources and final entities |

The record files contain evidence and decisions only — **no model reasoning traces are stored**.

### Validation gates

`package` refuses to produce an archive on any of: incorrect document count, an unknown entity
type, offsets that do not slice the source exactly, assertions or candidates on an ineligible type,
an unsupported field, a non-dotted-serializable ICD code, or any identifier outside the governed KB.

## Reproducibility

A run is identified by **two artifacts kept together**: the SHA-256 in `package-summary.json`
identifies the archive, and `model-manifest.json` identifies the weights that produced it. Either
alone proves nothing.

Semantic caches record the document checksum, row-order checksum, pooling contract, dtype and model
revision. Retrieval **refuses** a cache built by different weights or against a different knowledge
base, because such a mismatch is otherwise completely silent. Generation is greedy throughout.

## Repository structure

```
src/mednorm_vi/
├── production/      the pipeline: extraction then linking, in one pass
├── extraction/      proposal, alignment, lattice, verification, resolution, assertions
├── linking/         reformulation, sparse and dense retrieval, pruning, set-wise reranking
├── models/          registry, pinned acquisition, encoders, Qwen host, budget certification
├── kb/              governed indices, ontology facts, semantic cache contract
├── packaging/       dotted-code derivation, deterministic archive and hashing
├── validation/      organizer schema and submission gates
└── config.py        runtime configuration

scripts/run_mednorm.py             the only entrypoint
configs/pipeline/production.yaml   the only profile
tests/unit/                        extraction and linking invariants
docs/colab/                        production runbook
```

*Where is the pipeline?* — `production/pipeline.py`. *How do I run it?* — `scripts/run_mednorm.py`.
Each question has exactly one answer.

## Development

```bash
pytest tests/unit -q          # 139 tests
ruff check .
mypy src/mednorm_vi scripts
```

The suite runs entirely offline: language models are replaced by injected fakes that exercise the
same production orchestration path, so **no weights are downloaded during testing**.

## Documentation

- [`docs/colab/final-production-commands.md`](docs/colab/final-production-commands.md) — the
  authoritative fresh-runtime execution guide.
- [`docs/kb/`](docs/kb/) — governed knowledge-base freeze, competition policy and snapshot
  provenance.
- [`docs/decisions/`](docs/decisions/) — organizer contract facts and recorded unresolved policies.
- [`docs/audits/`](docs/audits/) — the full development record.

## Evidence and limitations — stated honestly

- **No leaderboard score is claimed for this system.** The consolidated pipeline in this repository
  has not been executed end to end on the organizer test set. Scores obtained earlier in
  development belonged to superseded configurations and would misrepresent this code.
- **The `8,729,759,237` total is measured, not estimated** — every count comes from the loaded
  checkpoints at acquisition time, and `budget` fails closed on an unmeasured model.
- **Offline behaviour is enforced by construction**, not by observation: the only network call in
  the codebase is Hugging Face model acquisition, which is confined to the `acquire` command and
  gated behind `--allow-download`.
- **Assertion and pruning rules are heuristic.** They encode clinical-language regularities
  (clause-scoped negation, kinship subject position, evidenced structured conflicts) and will not
  cover every construction in Vietnamese clinical prose.
- **The E3 checkpoint is a local artifact.** It is not redistributed here, so a third party cannot
  reproduce a run without it.
- **Peak VRAM is reported per run** in `diagnostic-summary.json` rather than guaranteed; it depends
  on document length and batch size.

## Security and compliance

- **Offline at inference:** no external API, no web retrieval, no telemetry — four local
  open-weight models loaded from disk.
- **Model weights are never committed.** `models/`, `checkpoint/`, `indices/`, caches and generated
  runs are all git-ignored.
- **No credentials** exist in the repository or the image; no provider key is required.
- **Pinned provenance:** immutable commit SHAs for hub models, a SHA-256 file digest for the local
  checkpoint, and checksums on every semantic cache.
- **Governed identifiers only:** the system cannot emit an ICD-10 or RxNorm code that is not in the
  competition knowledge base.
- **No public-test specialisation:** no document IDs, no hard-coded mentions, no answer lookup
  tables anywhere in the production path.

## Research acknowledgements

MedNorm-VI transplants *inference-time* ideas from the biomedical entity-linking literature. No
method below is reproduced exactly, and nothing here is trained — we gratefully acknowledge the
authors whose work shaped this design.

| Paper | Authors | Venue | Link |
|---|---|---|---|
| Biomedical Entity Representations with Synonym Marginalization (BioSyn) | Mujeen Sung, Hwisang Jeon, Jinhyuk Lee, Jaewoo Kang | ACL 2020 | [arXiv:2005.00239](https://arxiv.org/abs/2005.00239) |
| Self-Alignment Pretraining for Biomedical Entity Representations (SapBERT) | Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, Nigel Collier | NAACL 2021 | [arXiv:2010.11784](https://arxiv.org/abs/2010.11784) |
| Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking | Fangyu Liu, Ivan Vulić, Anna Korhonen, Nigel Collier | ACL 2021 | [arXiv:2105.14398](https://arxiv.org/abs/2105.14398) |
| Knowledge-Rich Self-Supervision for Biomedical Entity Linking (KRISSBERT) | Sheng Zhang, Hao Cheng, Shikhar Vashishth, Cliff Wong, Jinfeng Xiao, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, Hoifung Poon | Findings of NAACL 2022 | [arXiv:2112.07887](https://arxiv.org/abs/2112.07887) |
| MedType: Improving Medical Entity Linking with Semantic Type Prediction | Shikhar Vashishth, Rishabh Joshi, Denis Newman-Griffis, Ritam Dutt, Carolyn Rosé | arXiv preprint, 2020 | [arXiv:2005.00460](https://arxiv.org/abs/2005.00460) |
| ClinLinker: Medical Entity Linking of Clinical Concept Mentions in Spanish | Fernando Gallego, Guillermo López-García, Luis Gascó-Sánchez, Martin Krallinger, Francisco J. Veredas | arXiv preprint, 2024 | [arXiv:2404.06367](https://arxiv.org/abs/2404.06367) |
| Qwen3 Technical Report | Qwen Team, Alibaba Group | arXiv preprint, 2025 | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |

Which idea came from where: **BioSyn** — complementary sparse and dense retrieval with
character-level lexical similarity (inference-time only; no training or hard-negative mining).
**SapBERT / cross-lingual SapBERT** — the multilingual retriever, used frozen at its published
pooling contract. **ClinLinker** — retained as a hierarchy-aware *diagnosis-only* complementary
retriever. **KRISSBERT** — contextual query semantics, borrowed as document-context query
reformulation rather than by deploying the model. **MedType** — semantic type compatibility
filtering before disambiguation, implemented as a deterministic type gate rather than a new
classifier.

Two further design influences — small diversified candidate sets with retrieval scores withheld
from the final model, and set-wise rather than pointwise disambiguation with an explicit `NONE` —
were adopted from the candidate-ranking and generative-retrieval literature during development.
They are recorded in [`docs/audits/`](docs/audits/); we have not cited specific works for them here
rather than risk attributing them incorrectly.

Any implementation differences and errors are our own.

---

<div align="center">

Prepared for **Viettel AI Race 2026 — Track 2**.

</div>
