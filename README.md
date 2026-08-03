# MedNorm-VI

**Vietnamese clinical information extraction with governed ontology normalization.**

Submission for **Viettel AI Race 2026 — Track 2**.

Given a Vietnamese clinical note, MedNorm-VI identifies clinical entities, determines what is
asserted about each of them, and normalizes diagnoses and medications to governed ICD-10 and
RxNorm concepts.

Inference is fully local and self-hosted. No external API is called, no model is trained or
fine-tuned at run time, and every emitted concept identifier originates from the governed
knowledge base.

---

## Task

The system consumes a clinical note and produces organizer JSON containing, for each entity:
its exact character span, its type, its clinical assertions, and — for diagnoses and
medications — its governed ontology candidates.

Submissions are scored on three components: transcription fidelity of the extracted spans,
agreement on assertions, and agreement on ontology candidates. The system is therefore
precision-oriented throughout: an incorrect candidate is penalised, so the pipeline abstains
rather than guessing when the text does not support a concept.

---

## Architecture

A single inference path. There is no alternative pipeline, no variant family, and no post-hoc
selection between competing runs.

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
       ├─ Multi-view retrieval ──────────── sparse character n-gram (BioSyn-style)
       │                                 ├─ SapBERT-XLMR
       │                                 └─ ClinLinker-KB-GP  (diagnosis only)
       ├─ Governed candidate union           deduplicated, deterministically capped
       ├─ Ontology / structured pruning      type gate plus evidenced conflicts
       ├─ Qwen3-8B set-wise reranking        one option, or NONE
       │
       └─ Organizer JSON
```

### Design invariants

Two properties hold by construction rather than by validation:

| Invariant | Mechanism |
|---|---|
| **No hallucinated spans** | A model never emits a character offset. It returns a line identifier and text; offsets are resolved by locating that text in the source. Text absent from the document cannot be aligned and is discarded. |
| **No invented concept identifiers** | A model never emits an ontology code. It selects an option label from a list this system constructed from the governed knowledge base. A literal code in a model response is refused. |

Both are enforced by regression tests rather than assumed.

---

## Output schema

Five organizer entity types, with per-type field eligibility:

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

The two laboratory types carry `position`, `text` and `type` only. An empty `assertions`
array on those types is an unsupported field, not a harmless default, and packaging refuses a
submission containing one.

---

## Assertions

Three assertions are supported: `isNegated`, `isFamily`, `isHistorical`. The assertion pass
runs only after spans and types are final, so an assertion always describes a fixed entity.

- **`isNegated`** — scoped to the entity's own clause. A negation cue does not cross a clause
  boundary, so *"không sốt, có ho khan"* negates the fever but not the cough.
- **`isFamily`** — requires an unambiguous kinship reference that is the subject of the clause
  and precedes the entity. First-person or patient-reference narration cancels it. Terms that
  are simultaneously Vietnamese address pronouns (*ông*, *bà*, *cô*, *chú*) are counted only
  within a declared family-history section.
- **`isHistorical`** — derived from section context or an explicit past-time marker.
  Chronicity alone is insufficient: a chronic condition is a current condition.

---

## Ontology linking

A diagnosis may receive only an ICD-10 code; a medication may receive only an RxCUI. The
remaining three types are never linked.

Candidates are pruned when governed metadata **proves** a conflict — a stated strength, dose
form or route contradicting a recorded one — and never when an attribute is merely absent. A
candidate that records no strength is unspecific, not contradictory, and survives to the
reranker.

The set-wise reranker receives the mention, its surrounding context and each candidate's
ontology facts. It does **not** receive similarity scores, retrieval ranks or retriever names:
a similarity value is not semantic evidence, and exposing one anchors the model to whatever
the retrievers happened to favour. Candidate ordering is deterministic and independent of
retrieval rank.

Output is one governed concept per mention, or none.

---

## Deployed models

| Model | Role | Licence |
|---|---|---|
| `Qwen/Qwen3-8B` | Contextual extraction, verification, reformulation, reranking | Apache-2.0 |
| `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | Multilingual concept retrieval (diagnosis + medication) | MIT |
| `ICB-UMA/ClinLinker-KB-GP` | Hierarchy-aware retrieval (diagnosis only) | Apache-2.0 |
| E3 (ViHealthBERT) | Mention proposal | Local checkpoint |

### Parameter compliance

```
SapBERT-XLMR           278,043,648
ClinLinker-KB-GP       125,978,112
ViHealthBERT (E3)      135,002,117
Qwen3-8B             8,190,735,360
──────────────────────────────────
TOTAL                8,729,759,237   <  9,000,000,000
```

The total sums **every model the system deploys**, not those simultaneously resident.
Sequential loading reduces peak memory; it does not reduce what was deployed. Sparse character
retrieval is an inverted index and contributes no parameters.

`run_mednorm.py budget` certifies this figure from the model manifest and exits non-zero if
the stack has changed.

Model revisions are resolved to immutable commit SHAs **before** download and recorded in
`model-manifest.json`. A branch name is refused as a revision, since it cannot identify the
weights that produced a result.

---

## Governed knowledge base

All candidate identifiers originate from the governed ICD-10 and RxNorm indices under
`indices/candidate/`. No other candidate source exists in the system.

Governed alias matches are a **proposal source**, not an entity source: an alias may support a
mention another source found, but an alias-only span never becomes an entity on its own. Alias
quality is measured from the knowledge base itself — how many concepts a surface form names,
how frequently a token occurs across labels, and the index's own label-quality metadata —
rather than from a hand-maintained word list.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
pip install -e ".[dev]"
```

Requires Python 3.10+ and a CUDA device for inference.

### Required assets

None of the following are tracked in this repository:

| Asset | Location |
|---|---|
| Governed ICD-10 / RxNorm indices | `indices/candidate/` |
| Structured RxNorm attributes | `artifacts/rxnorm_structured/` |
| Organizer documents | `data/organizer_test/input/` |
| E3 checkpoint | `checkpoint/` |
| Qwen3-8B snapshot | local directory, pinned revision |

### Hardware

Models load sequentially and are released between phases, so peak device memory tracks the
largest single model rather than their sum.

| Stage | Minimum VRAM |
|---|---|
| `acquire`, `index` | 6 GiB |
| `run` | 18 GiB |
| Recommended | 40 GiB |

A 22.5 GiB NVIDIA L4 is sufficient for the full pipeline. `package` requires no GPU.

---

## Usage

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

Any stage that loads model weights requires `--allow-download`, an explicit assertion that the
host is a GPU machine, and refuses to start below that stage's memory floor.

For a fresh Google Colab runtime, see
[`docs/colab/final-production-commands.md`](docs/colab/final-production-commands.md).

### Produced artifacts

| File | Contents |
|---|---|
| `output.zip` | The submission archive |
| `package-summary.json` | Archive path and SHA-256 |
| `diagnostic-summary.json` | Per-stage counters, peak VRAM, certified parameter total |
| `mention-records.jsonl` | Per-mention candidates, retrieval provenance and selection |
| `proposal-records.jsonl` | Per-document proposals, sources and final entities |

The two record files contain evidence and decisions only — no model reasoning traces are
stored.

### Validation gates

`package` refuses to produce an archive on any of: incorrect document count, an unknown entity
type, offsets that do not slice the source exactly, assertions or candidates on an ineligible
type, an unsupported field, a non-dotted-serializable ICD code, or any identifier outside the
governed knowledge base.

---

## Reproducibility

A run is identified by two artifacts that must be kept together: the SHA-256 in
`package-summary.json` identifies the archive, and `model-manifest.json` identifies the weights
that produced it. Either alone proves nothing.

Semantic caches record the document checksum, row-order checksum, pooling contract, dtype and
model revision. Retrieval refuses a cache built by different weights or against a different
knowledge base, since such a mismatch is otherwise silent. Generation is greedy throughout.

---

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

Two questions have exactly one answer each: *where is the pipeline?* —
`production/pipeline.py`; *how do I run it?* — `scripts/run_mednorm.py`.

---

## Development

```bash
pytest tests/unit -q          # 139 tests
ruff check .
mypy src/mednorm_vi scripts
```

The test suite runs entirely offline. Language models are replaced by injected fakes that
exercise the same production orchestration path, so no weights are downloaded during testing.

---

## Development history

The final system was reached through an audited sequence of experiments — earlier entity
extractors, semantic linkers, candidate-tier policies and span-correction passes. Those
experiments are history: the working tree contains the final system only, and the Git history
preserves the rest.

---

## Licence and attribution

Pretrained model licences are recorded in the model registry alongside their source model
cards, and are reproduced in the deployed-model table above. This repository records what each
model card declares; it does not constitute a legal determination. ICD-10 and RxNorm content
is governed by their respective terms.
