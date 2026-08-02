# Sprint 0075 — Colab commands

## Leaderboard status (2026-08-02)

| Variant | Score | J_candidates | Status |
| --- | ---: | ---: | --- |
| historical 0073 | **12.4757** | **4.2570** | **PUBLIC BEST** |
| corrected_dense_only | 12.3922 | 4.0483 | **PUBLIC_REJECTED** (−0.0835) |

The row-order fix is correct engineering and worse on the leaderboard. To reproduce the
public best, use `configs/pipeline/public_best_0073_legacy.yaml`, which explicitly selects
`row_order: legacy_0073_misaligned_file_positions` and says why.

**Variant C (8B + Embedding-4B + Reranker-4B) is NOT PRODUCED — it is illegal**: 16,369,296,389
parameters against a 9B cap on the SUM of base models. Sequential loading lowers peak VRAM,
not model count.

## Fresh-Colab dependencies (all gitignored — a clone does NOT contain them)

| Needed | Where from |
| --- | --- |
| `indices/candidate/icd10_vi/competition-v3`, `competition-v4.1`, `indices/candidate/rxnorm/competition-v3` | upload `kb-inputs.tgz` from your laptop |
| `artifacts/semantic/0071/documents.jsonl` (+ `manifest.json`) | `MyDrive/MedNorm-VI/0072/artifacts/` |
| `artifacts/rxnorm_structured/0074/structured.jsonl` | rebuild locally (`scripts/build_rxnorm_structured_0074.py`) or copy from Drive |
| `S1_doc_vectors.pt` (variant A only) | `MyDrive/MedNorm-VI/0072/cache/` |
| Qwen3-Embedding-4B / Reranker-4B (A only) | `MyDrive/MedNorm-VI/0072/models/` |
| Qwen3-8B (E only) | download, see below |
| `data/organizer_test/input` | upload yourself |

Laptop, once:

```bash
cd <repo> && tar czf /tmp/kb-inputs.tgz \
  indices/candidate/icd10_vi/competition-v3 \
  indices/candidate/icd10_vi/competition-v4.1 \
  indices/candidate/rxnorm/competition-v3
```

## Cell 1 — environment (TensorFlow OFF: Colab protobuf conflict breaks Qwen3)

```python
import os
os.environ.update(USE_TF="0", TRANSFORMERS_NO_TF="1", USE_FLAX="0",
                  TOKENIZERS_PARALLELISM="false")
from google.colab import drive; drive.mount('/content/drive')
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Cell 2 — repo + deps + KB restore

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
[ -d /content/MedNorm-VI/.git ] || git clone https://github.com/vquclinh/MedNorm-VI /content/MedNorm-VI
cd /content/MedNorm-VI && git fetch -q && git checkout -q main && git log -1 --oneline
pip install -q "transformers==4.51.3" "accelerate==1.6.0" "safetensors==0.4.5" "sentencepiece==0.2.0"
D=/content/drive/MyDrive/MedNorm-VI
tar xzf $D/0072/kb-inputs.tgz -C /content/MedNorm-VI 2>/dev/null || tar xzf /content/kb-inputs.tgz -C /content/MedNorm-VI
mkdir -p artifacts/semantic/0071 && cp $D/0072/artifacts/documents.jsonl $D/0072/artifacts/manifest.json artifacts/semantic/0071/
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())"
```

---

# PUBLIC-BEST RERUN — historical 0073 (use this if you need to reproduce 12.4757)

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
D=/content/drive/MyDrive/MedNorm-VI/0072
python scripts/run_public_inference_0073.py \
  --input-dir data/organizer_test/input \
  --run-dir runs/public_best_0073_legacy_rerun \
  --config configs/pipeline/public_best_0073_legacy.yaml \
  --semantic-model-root  $D/models \
  --semantic-doc-vectors $D/cache/S1_doc_vectors.pt \
  --semantic-documents   $D/artifacts/documents.jsonl
```

---

# VARIANT A — corrected_dense_only  ← PUBLIC_REJECTED, kept as artifact

Scored 0073 architecture with the dense row-order defect fixed. Nothing else changes.

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
D=/content/drive/MyDrive/MedNorm-VI/0072
python scripts/run_public_inference_0073.py \
  --input-dir data/organizer_test/input \
  --run-dir runs/corrected_dense_0075_final_dotted \
  --config configs/pipeline/corrected_dense_0075.yaml \
  --semantic-model-root  $D/models \
  --semantic-doc-vectors $D/cache/S1_doc_vectors.pt \
  --semantic-documents   $D/artifacts/documents.jsonl
```

Prints the `output.zip` path and SHA-256. Does not submit.

---

# VARIANT E — full_8b (new, unmeasured)

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
huggingface-cli download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /content/drive/MyDrive/MedNorm-VI/0075/models/qwen3-8b
```

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/run_8b_reasoner_0075.py \
  --input-dir data/organizer_test/input \
  --run-dir runs/full_8b_0075 \
  --model-root /content/drive/MyDrive/MedNorm-VI/0075/models/qwen3-8b \
  --config full_8b --device cuda

python -m mednorm_vi.inference.derive_submission \
  --source-dir runs/full_8b_0075/output_raw_dotless \
  --output-dir runs/full_8b_0075/output \
  --derivation icd_dotted --expected-documents 100 \
  --validation-report runs/full_8b_0075/derivation-validation.json

python -c "
from mednorm_vi.inference.packaging import package_output_zip
from pathlib import Path; import hashlib
p=package_output_zip(Path('runs/full_8b_0075/output'),Path('runs/full_8b_0075/output.zip'),expected_count=100)
print(p, hashlib.sha256(Path(p).read_bytes()).hexdigest())"
```

---

# VARIANT D — candidate_8b (historical E3 entities + 8B governed selector)

Two steps: produce E3 entities/assertions with the public-best profile, then let the 8B choose
candidates from a parameter-free governed pool. Only Qwen3-8B is deployed in step 2.

```bash
%%bash
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTHONPATH=src
cd /content/MedNorm-VI
python scripts/run_8b_reasoner_0075.py \
  --input-dir data/organizer_test/input \
  --run-dir runs/candidate_8b_0075 \
  --model-root /content/drive/MyDrive/MedNorm-VI/0075/models/qwen3-8b \
  --config cand_8b \
  --seed-entities runs/public_best_0073_legacy_rerun/output \
  --device cuda

python -m mednorm_vi.inference.derive_submission \
  --source-dir runs/candidate_8b_0075/output_raw_dotless \
  --output-dir runs/candidate_8b_0075/output \
  --derivation icd_dotted --expected-documents 100 \
  --validation-report runs/candidate_8b_0075/derivation-validation.json

python -c "
from mednorm_vi.inference.packaging import package_output_zip
from pathlib import Path; import hashlib
p=package_output_zip(Path('runs/candidate_8b_0075/output'),Path('runs/candidate_8b_0075/output.zip'),expected_count=100)
print(p, hashlib.sha256(Path(p).read_bytes()).hexdigest())"
```

`--config entity_8b` is the C-style entity/assertion-only switch, but it must be paired with a
LEGAL linker — never the 4B pair.

**Budget:** E3 135,002,117 + Qwen3-8B 8,190,735,360 = **8,325,737,477** (< 9B). The script
refuses to co-deploy the Audit-0073 4B pair — that would be 16.4B.
