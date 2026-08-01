# Colab execution commands — Audit 0071 pretrained hybrid semantic linker

The owner's machine has **8 GiB VRAM, ~7 GiB free RAM and 3.1 GiB free on the `/home`
partition that holds the Hugging Face cache**. S1 needs ~16 GB of weights and S2 ~17 GB, so
downloading either locally would fill the disk before the first file finished. Audit 0071
therefore stopped before any multi-GB download, exactly as §17 requires, and the heavy steps
run here instead.

No notebook is required. Every step below is a shell command.

Nothing in this document sends clinical text anywhere. The models are downloaded *to* the
GPU host; the private annotation packs under `data/private_linker_gold/` are **not** uploaded.

---

## 0. Runtime

Choose an **A100 40GB** (sufficient for both systems) or **H100**. A T4/L4 is not enough for
S2's 8B embedding model in bfloat16.

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /content
```

## 1. Repository

```bash
cd /content
git clone <OWNER_REMOTE_URL> MedNorm-VI
cd MedNorm-VI
git checkout main
git log -1 --oneline          # expect the commit carrying Audit 0071
```

## 2. Pinned dependencies

```bash
pip install -q \
  "torch==2.5.1" \
  "transformers==4.51.3" \
  "accelerate==1.6.0" \
  "sentencepiece==0.2.0" \
  "safetensors==0.4.5"
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())"
```

`transformers >= 4.51` is required: Qwen3 model classes are not present in earlier releases.

## 3. Download the exact pinned revisions

Revisions are immutable commit SHAs verified against the Hugging Face API on 2026-08-01.
Do **not** substitute `main`.

```bash
export HF_HOME=/content/hf
huggingface-cli download Qwen/Qwen3-Embedding-4B  --revision 5cf2132abc99 --local-dir /content/models/emb-4b
huggingface-cli download Qwen/Qwen3-Reranker-4B   --revision 22e683669bc0 --local-dir /content/models/rr-4b
# S2 only:
huggingface-cli download Qwen/Qwen3-Embedding-8B  --revision 1d8ad4ca9b3d --local-dir /content/models/emb-8b
huggingface-cli download Qwen/Qwen3-Reranker-0.6B --revision e61197ed4502 --local-dir /content/models/rr-06b
```

| Model | Revision | Params | License |
| --- | --- | ---: | --- |
| Qwen/Qwen3-Embedding-4B | `5cf2132abc99` | 4,021,774,336 | apache-2.0 |
| Qwen/Qwen3-Reranker-4B | `22e683669bc0` | 4,021,784,576 | apache-2.0 |
| Qwen/Qwen3-Embedding-8B | `1d8ad4ca9b3d` | 7,567,295,488 | apache-2.0 |
| Qwen/Qwen3-Reranker-0.6B | `e61197ed4502` | 595,776,512 | apache-2.0 |

## 4. Build semantic documents and the concept-disjoint benchmark

Model-free; already run locally, but re-run so the artifacts exist on the GPU host.

```bash
PYTHONPATH=src python scripts/build_semantic_kb_0071.py
cat artifacts/semantic/0071/manifest.json
```

Expect 227,257 documents and concept-disjoint train/dev/test with zero concept leakage.

## 5. Zero-shot evaluation FIRST (§14 — do not fine-tune before this)

```bash
PYTHONPATH=src python scripts/evaluate_semantic_linker_0071.py --system S1 --device cuda
PYTHONPATH=src python scripts/evaluate_semantic_linker_0071.py --system S2 --device cuda
```

Smoke-test a small slice before committing to the full 227k-document encode:

```bash
PYTHONPATH=src python scripts/evaluate_semantic_linker_0071.py \
  --system S1 --device cuda --limit-documents 5000
```

Engineering targets on the selected-consensus subset: **Recall@10 = 15/15**, top-1 ≥ 13/15,
and a material reduction in the current **43/54** false NO_VALID emissions — without
collapsing valid-candidate recall. Do not force these numbers; report every regression.

## 6. Choose ONE architecture, then optionally fine-tune (§15)

```bash
PYTHONPATH=src python scripts/train_semantic_linker_0071.py \
  --model Qwen/Qwen3-Embedding-4B --revision 5cf2132abc99 \
  --objective contrastive --seed 0
```

One configuration plus one confirmation seed. No broad sweep.

## 7. Run the hybrid linker over public input

```bash
PYTHONPATH=src python scripts/run_semantic_linker_0071.py \
  --input-dir data/organizer_test/input \
  --output-dir runs/pretrained_semantic_linker_0071_final_dotted/output_raw_dotless \
  --system S1 --dense-index artifacts/semantic/0071/dense_S1.faiss --device cuda
```

Then apply the governed dotted derivation — the same step the 11.9188 baseline used:

```bash
PYTHONPATH=src python -m mednorm_vi.inference.derive_submission \
  --source-dir runs/pretrained_semantic_linker_0071_final_dotted/output_raw_dotless \
  --output-dir runs/pretrained_semantic_linker_0071_final_dotted/output \
  --derivation icd_dotted --expected-documents 100 \
  --validation-report runs/pretrained_semantic_linker_0071_final_dotted/validation-report.json
```

## 8. Package artifacts for transfer back

```bash
cd /content/MedNorm-VI
tar czf /content/0071-artifacts.tgz \
  artifacts/semantic/0071/eval \
  artifacts/semantic/0071/manifest.json \
  artifacts/semantic/0071/deployed_parameter_manifest.json \
  runs/pretrained_semantic_linker_0071_final_dotted
sha256sum /content/0071-artifacts.tgz
```

Download `0071-artifacts.tgz`, unpack it into the repository, and re-run the offline
acceptance gate locally before any submission decision.

---

## Constraints that still apply on Colab

- **Do not** access `internal_test`.
- **Do not** train on public-test data or tune per-public-record rules.
- **Do not** use the leaderboard score as a label or calibration target; only the aggregate
  outcome may be recorded.
- Deployed parameter total must stay **< 9,000,000,000**; S1 totals 8,178,561,029 and S2
  totals 8,298,074,117 including the existing 135,002,117-parameter E3 head.
- Do not upload `data/private_linker_gold/` anywhere.
