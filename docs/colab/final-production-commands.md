# MedNorm-VI — production runbook

Everything needed to reproduce final inference on a fresh Colab runtime. There is one
pipeline, one CLI and one profile; nothing else needs to be understood to run the system.

Host: a single CUDA device. Models load **sequentially** and are unloaded between phases, so
peak memory tracks the largest single model (~17 GiB) rather than their sum — a 22.5 GiB L4
is sufficient. 40 GiB is recommended.

## 1 — runtime, Drive, repository

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
               "sentencepiece==0.2.0" "huggingface_hub[cli]" "pyyaml"
```

## 2 — restore governed assets

The knowledge base, the organizer input, the E3 proposal set and Qwen are all gitignored and
restored from Drive. Nothing here is downloaded from an unpinned source.

```bash
%%bash
cd /content/MedNorm-VI; D=/content/drive/MyDrive/MedNorm-VI
tar xzf $D/kb-inputs.tgz -C .                      # governed ICD + RxNorm indices
mkdir -p artifacts/rxnorm_structured/0074
cp $D/structured.jsonl artifacts/rxnorm_structured/0074/
cp -r $D/organizer_test/input data/organizer_test/ # organizer documents
ls $D/e3_entities | wc -l                          # E3 proposal set
cp -r $D/e3_entities /content/e3_entities
[ -d /content/qwen3-8b-local ] || huggingface-cli download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /content/qwen3-8b-local
```

## 3 — acquire the frozen retrievers

The exact commit is resolved from the hub **before** anything is fetched, and the download
happens at that commit. A model that cannot produce an immutable revision is refused.

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/run_mednorm.py acquire --allow-download \
  --model-root /content/mednorm_models \
  --qwen-root /content/qwen3-8b-local \
  --e3-checkpoint checkpoint/e3_boundary_refinement_0062/best.pt
cat /content/mednorm_models/model-manifest.json
```

## 4 — build the semantic caches

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/run_mednorm.py index --allow-download \
  --model-root /content/mednorm_models --cache-root /content/mednorm_cache
```

One cache per retriever — pooling and tokenisation are not interchangeable. Each cache
records the document checksum, the row-order checksum, the pooling contract, the dtype and
the model revision, and retrieval refuses to start if any of them disagrees.

## 5 — verify the deployed budget

```bash
%%bash
export PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/run_mednorm.py budget --model-root /content/mednorm_models
```

Expect `TOTAL 8,729,759,237` under the 9,000,000,000 cap. The command fails if the stack has
moved.

## 6 — inference

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/run_mednorm.py run --allow-download \
  --input-dir data/organizer_test/input \
  --seed-entities /content/e3_entities \
  --model-root /content/mednorm_models \
  --cache-root /content/mednorm_cache \
  --qwen-root /content/qwen3-8b-local \
  --output-dir runs/production
python -c "
import json;d=json.load(open('runs/production/diagnostic-summary.json'))
print(json.dumps({k:v for k,v in d.items()
                  if k!='extraction_generation_reports'},indent=2,ensure_ascii=False))"
```

Read before packaging: `entities` and `governed_codes`; in `extraction_counters` the
`proposer_*_kept` counts, `proposer_passes_failed` (expect 0) and `verifier_invalid_index`
(expect 0); in `linking_counters` `select` versus `none`, `reranker_returned_a_code` and
`reranker_invalid_option` (both expect ~0); and `peak_vram_gib`.

## 7 — package

```bash
%%bash
export PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/run_mednorm.py package --output-dir runs/production \
  --input-dir data/organizer_test/input --expected-documents 100
cat runs/production/package-summary.json
```

Packaging derives dotted ICD codes, then refuses to archive on: wrong document count, an
unknown type, offsets that do not slice the source exactly, assertions or candidates on a
type that must not carry them, an unsupported field, or any identifier outside the governed
knowledge base.

## 8 — back up

```bash
%%bash
D=/content/drive/MyDrive/MedNorm-VI/production; mkdir -p $D
cp runs/production/output.zip runs/production/package-summary.json \
   runs/production/diagnostic-summary.json runs/production/*.jsonl $D/
cp /content/mednorm_models/model-manifest.json $D/
```

The SHA256 in `package-summary.json` identifies the archive; the model manifest identifies
the weights that produced it. Keep both together — either alone proves nothing.
