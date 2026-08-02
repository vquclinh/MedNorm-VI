# GraphCENT 0080 — Colab runbook

Multi-encoder biomedical retrieval + ontology context + constrained Qwen3-8B.
All weights frozen; nothing is trained. **One inference pass produces four submission
variants** — Qwen is not re-run per variant.

**Nothing here runs on the development laptop.** Every stage that touches weights requires
`--allow-download` (an explicit statement that this is a GPU host) *and* enough device memory
for that stage. The floors are per operation, not one blanket number:

| stage | needs | why |
|---|---|---|
| `build-index` | ≥ 6 GiB | one retriever resident at a time |
| `smoke`, `run` | ≥ 18 GiB | Qwen3-8B in bf16, retrievers already unloaded |
| any | ≥ 40 GiB recommended | below that the run warns but proceeds |

Models are loaded **sequentially and unloaded** between phases, so a 22.5 GiB L4 is a
supported host. `package` needs no GPU at all.

`--config` is a **global** flag and must come before the subcommand.

## Cell 1 — runtime, Drive, repo

```python
import os
os.environ.update(USE_TF="0", TRANSFORMERS_NO_TF="1", USE_FLAX="0",
                  TOKENIZERS_PARALLELISM="false")
from google.colab import drive; drive.mount('/content/drive')
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
[ -d /content/MedNorm-VI/.git ] || git clone https://github.com/vquclinh/MedNorm-VI /content/MedNorm-VI
cd /content/MedNorm-VI && git fetch -q && git checkout -q main && git log -1 --oneline
pip install -q "transformers==4.51.3" "accelerate==1.6.0" "safetensors==0.4.5" \
               "sentencepiece==0.2.0" "huggingface_hub[cli]"
```

## Cell 2 — restore gitignored assets

```bash
%%bash
cd /content/MedNorm-VI; D=/content/drive/MyDrive/MedNorm-VI
tar xzf $D/0072/kb-inputs.tgz -C .                 # ICD v3/v4.1 + RxNorm indices
mkdir -p artifacts/rxnorm_structured/0074
cp $D/0074/structured.jsonl artifacts/rxnorm_structured/0074/ 2>/dev/null || \
  PYTHONPATH=src python scripts/build_rxnorm_structured_0074.py
# organizer input + E3-only seed entities + Qwen3-8B
cp -r $D/organizer_test/input data/organizer_test/ 2>/dev/null || true
ls runs/e3_verified_8b_allnull_0076/output | wc -l   # expect 100 (E3-only seed)
[ -d /content/qwen3-8b-local ] || huggingface-cli download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /content/qwen3-8b-local
```

The seed directory is the **E3-only** entity/assertion output. GraphCENT never re-extracts
entities and never changes a type or an assertion; it only attaches candidates.

## Cell 3 — download retrievers, resolve provenance, enforce the budget

Models go to `/content/graphcent_models`, **not** Drive. Persist later only if they earn it.

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/graphcent_0080.py download-models \
  --allow-download \
  --model-root /content/graphcent_models \
  --e3-checkpoint checkpoint/e3_boundary_refinement_0062/best.pt \
  --qwen-root /content/qwen3-8b-local
cat /content/graphcent_models/model-manifest.json
```

Prints per model: resolved revision (read from the downloaded snapshot, never assumed),
licence, **measured** parameter count and embedding dim — then the summed total, hard-failing
at ≥ 9,000,000,000. An unmeasured count is a failure, not a zero.

### Licences, as recorded on the model cards

| key | repo | licence on card | default |
|---|---|---|---|
| `sapbert_xlmr` | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | MIT | **enabled** |
| `clinlinker_kb_gp` | `ICB-UMA/ClinLinker-KB-GP` | Apache-2.0 | **enabled** |
| `biobert_mnli` | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` | CC-BY-NC-3.0 (non-commercial) | **disabled** |

This records what the cards state; it is not a legal conclusion. The default profile
`configs/pipeline/graphcent_0080.yaml` ships with `biobert_mnli` **disabled**, so the licence
gate is already clear and nothing is downloaded for it. To run the third encoder anyway:

```bash
python scripts/graphcent_0080.py --config configs/pipeline/graphcent_0080_full_research.yaml \
  download-models --allow-download --model-root /content/graphcent_models
```

That profile sets `enabled: true` **and** `licence_overrides_accepted: [biobert_mnli]`. The
gate fails closed: enabling the model without recording the override is refused.

## Cell 4 — build the semantic indices

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/graphcent_0080.py build-index --allow-download \
  --model-root /content/graphcent_models --cache-root /content/graphcent_cache
```

One cache per retriever — pooling and tokenisation differ, so they are not interchangeable
(SapBERT and ClinLinker are read at CLS; the BioBERT sentence model is mean-pooled). Each
cache carries the document checksum, **row-order checksum**, pooling, dtype and shape, and
retrieval refuses to start if any disagrees. Each encoder is unloaded before the next loads.

## Cell 5 — smoke on 3 documents

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/graphcent_0080.py smoke --allow-download \
  --input-dir data/organizer_test/input \
  --seed-entities runs/e3_verified_8b_allnull_0076/output \
  --model-root /content/graphcent_models --cache-root /content/graphcent_cache \
  --run-dir runs/graphcent_0080/_smoke --documents 1,10,100
python -c "
import json;d=json.load(open('runs/graphcent_0080/_smoke/diagnostic-summary.json'))
print(json.dumps({k:v for k,v in d.items() if k!='mentions'},indent=2))"
```

Read before going further: `null_decisions` vs `select_decisions`, `parse_fallbacks`
(a rising count means the prompt is not landing), `invalid_ids` (ids the model invented —
they are stripped, never emitted), and the tier counts.

## Cell 6 — full 100-document run (ONE pass, four variants)

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/graphcent_0080.py run --allow-download \
  --input-dir data/organizer_test/input \
  --seed-entities runs/e3_verified_8b_allnull_0076/output \
  --model-root /content/graphcent_models --cache-root /content/graphcent_cache \
  --run-dir runs/graphcent_0080
```

Retrieval for every unique span is precomputed first (one encoder resident at a time, each
unloaded before the next), then Qwen loads once and every mention is decided in that single
pass. `mention-records.jsonl` holds the neutral per-mention evidence; the four tier variants
are derived from it afterwards on CPU.

## Cell 7 — validate, derive dotted ICD, package, SHA256

```bash
%%bash
export PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/graphcent_0080.py package --run-dir runs/graphcent_0080 \
  --input-dir data/organizer_test/input --expected-documents 100
cat runs/graphcent_0080/package-summary.json
```

Validation refuses to package on: wrong document count, an unknown type, offsets that do not
slice the source text exactly, assertions or candidates on a type that must not carry them,
an unsupported field, an ICD code that is not dotted-serializable, or any id outside the
governed KB. Produces `output_allnull.zip`, `output_tierA.zip`, `output_tierAB.zip`,
`output_tierABC.zip` with a SHA256 for each. **Nothing is submitted automatically.**

Submission order, given ALL-NULL is the public best at 14.3749: submit `tierA` first (exact
governed alias matches only), and only escalate to `tierAB` / `tierABC` if `tierA` beats
14.3749. `output_allnull.zip` is the control and should reproduce the known score.

## Cell 8 — persist only what earned it

```bash
%%bash
D=/content/drive/MyDrive/MedNorm-VI/0080; mkdir -p $D
cp runs/graphcent_0080/*.zip runs/graphcent_0080/*.json $D/
cp /content/graphcent_models/model-manifest.json $D/
# copy caches/models to Drive ONLY after a variant proves itself on the leaderboard
```

## Disk / RAM / VRAM

| item | size |
|---|---|
| Qwen3-8B snapshot | ~16 GiB disk |
| SapBERT-XLMR | ~1.1 GiB disk |
| ClinLinker-KB-GP | ~0.44 GiB disk |
| BioBERT-MNLI (optional, off by default) | ~0.44 GiB disk |
| semantic caches (227,257 docs × dim, fp16) | ~0.3–0.8 GiB each |
| peak VRAM | ~17 GiB (Qwen bf16), ~3 GiB (index build) |
| peak system RAM | ~12 GiB |
