# Colab runbook — Audit 0072 complete zero-shot hybrid semantic linker

This run answers one question: **how good are the official pretrained models zero-shot, as a
complete system?** Not dense recall — the full pipeline, with real Qwen3 cross-encoder
reranking.

**No training happens here.** Audit 0072 §11 stops after zero-shot evidence exists.
**No public inference runs automatically.** §12 permits it only after the offline gate passes.

Pipeline per system:

```
v3 lexical Top-20  ∪  v4.1 tiered lexical Top-20  ∪  dense Top-50
      → deduplicate by governed identity (nothing outside the frozen KB survives)
      → REAL Qwen3 reranker  (yes/no logit protocol, not cosine similarity)
      → H2 within-family arbitration, applied AFTER reranking
      → NULL gate in SHADOW mode (records evidence, changes nothing)
```

Models are loaded **one at a time** and unloaded between stages, so no runtime holds all four.

---

## Prerequisite (once, on your laptop)

The governed KB indices are gitignored, so a fresh clone does not contain them.

```bash
\
mkdir -p /tmp/0072-upload && \
tar czf /tmp/0072-upload/kb-inputs.tgz \
  indices/candidate/icd10_vi/competition-v3 \
  indices/candidate/icd10_vi/competition-v4.1 \
  indices/candidate/rxnorm/competition-v3 && \
sha256sum /tmp/0072-upload/kb-inputs.tgz
```

Upload to `MyDrive/mednorm-vi/0071/kb-inputs.tgz`. It contains governed ontology indices only —
no clinical text. `data/private_linker_gold/` is never uploaded.

## Cell 1 — Drive + GPU

```python
import os, pathlib, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True,text=True).stdout.strip() or "NO GPU — switch to A100/H100")
from google.colab import drive; drive.mount('/content/drive')
D = pathlib.Path('/content/drive/MyDrive/mednorm-vi'); (D/'0071').mkdir(parents=True, exist_ok=True)
(D/'0071'/'models').mkdir(exist_ok=True); (D/'0072').mkdir(exist_ok=True); (D/'0072'/'logs').mkdir(exist_ok=True)
os.environ.update(DRIVE=str(D), WORK='/content/MedNorm-VI', HF_HOME=str(D/'0071'/'models'/'hf'),
                  REV='<PASTE_COMMIT_SHA_OF_AUDIT_0072>')
assert (D/'0071'/'kb-inputs.tgz').exists(), "upload kb-inputs.tgz first"
print("ok")
```

## Cell 2 — clone at the exact committed revision

```bash
%%bash
set -euo pipefail
[ -d "$WORK/.git" ] || git clone https://github.com/vquclinh/MedNorm-VI "$WORK"
cd "$WORK"; git fetch --all -q; git checkout -q "$REV"
test "$(git rev-parse HEAD)" = "$REV" && echo "pinned $(git log -1 --oneline)"
```

## Cell 3 — dependencies

```bash
%%bash
set -euo pipefail
pip install -q "transformers==4.51.3" "accelerate==1.6.0" "safetensors==0.4.5" "sentencepiece==0.2.0" "huggingface_hub[cli]"
python - <<'PY'
import torch, transformers
assert transformers.__version__.startswith("4.51"), transformers.__version__
assert tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2,4), torch.__version__
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
PY
```

## Cell 4 — restore KB, build semantic documents, download pinned models

```bash
%%bash
set -euo pipefail
cd "$WORK"
[ -f indices/candidate/icd10_vi/competition-v4.1/index.json ] || tar xzf "$DRIVE/0071/kb-inputs.tgz" -C "$WORK"
sha256sum indices/candidate/icd10_vi/competition-v4.1/index.json
echo "expect d1803be3a1a1bea2f72f6ca20f98e687c4a32d14f3018fb738bd0184942ea09c"

if [ -f "$DRIVE/0071/artifacts/documents.jsonl" ]; then
  mkdir -p artifacts/semantic/0071/benchmark
  cp "$DRIVE/0071/artifacts/documents.jsonl" "$DRIVE/0071/artifacts/manifest.json" artifacts/semantic/0071/
  cp "$DRIVE/0071/artifacts"/{train,dev,test,test_2k}.jsonl artifacts/semantic/0071/benchmark/ 2>/dev/null || true
else
  PYTHONPATH=src python scripts/build_semantic_kb_0071.py 2>&1 | tee "$DRIVE/0072/logs/build_kb.log"
  mkdir -p "$DRIVE/0071/artifacts"
  cp artifacts/semantic/0071/documents.jsonl artifacts/semantic/0071/manifest.json "$DRIVE/0071/artifacts/"
  cp artifacts/semantic/0071/benchmark/*.jsonl "$DRIVE/0071/artifacts/"
fi
[ -f artifacts/semantic/0071/benchmark/test_2k.jsonl ] || python - <<'PY'
import json, pathlib
src=pathlib.Path("artifacts/semantic/0071/benchmark/test.jsonl")
rows=[json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
rows.sort(key=lambda r:(r["ontology"],r["positive_concept_id"],r["query"]))
sl=rows[::max(1,len(rows)//2000)][:2000]
src.with_name("test_2k.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in sl),encoding="utf-8")
print("eval slice",len(sl))
PY
cp artifacts/semantic/0071/benchmark/test_2k.jsonl "$DRIVE/0071/artifacts/" 2>/dev/null || true

M="$DRIVE/0071/models"
dl(){ [ -f "$M/$3/config.json" ] || huggingface-cli download "$1" --revision "$2" --local-dir "$M/$3" >/dev/null; echo "ready $1"; }
dl Qwen/Qwen3-Embedding-4B  5cf2132abc99cad020ac570b19d031efec650f2b emb-4b
dl Qwen/Qwen3-Reranker-4B   22e683669bc0f0bd69640a1354a6d0aebcfeede5 rr-4b
dl Qwen/Qwen3-Embedding-8B  1d8ad4ca9b3dd8059ad90a75d4983776a23d44af emb-8b
dl Qwen/Qwen3-Reranker-0.6B e61197ed45024b0ed8a2d74b80b4d909f1255473 rr-06b
du -sh "$M"/* 2>/dev/null
```

## Cell 5 — smoke test (8 queries, full document pool, real reranker)

```bash
%%bash
set -euo pipefail
cd "$WORK"
PYTHONPATH=src python scripts/run_0072_zero_shot_colab.py --stage smoke \
  --models "$DRIVE/0071/models" --device cuda 2>&1 | tee "$DRIVE/0072/logs/smoke.log"
echo "If this printed final R@1/R@10/MRR, the complete pipeline works end to end."
```

## Cell 6 — complete zero-shot S1

```bash
%%bash
set -euo pipefail
cd "$WORK"
PYTHONPATH=src python scripts/run_0072_zero_shot_colab.py --stage s1 \
  --models "$DRIVE/0071/models" --device cuda \
  --embed-batch-size 16 --rerank-batch-size 8 2>&1 | tee "$DRIVE/0072/logs/s1.log"
mkdir -p "$DRIVE/0072/artifacts"; cp artifacts/semantic/0072/S1_*.json* "$DRIVE/0072/artifacts/" 2>/dev/null || true
```

## Cell 7 — complete zero-shot S2

```bash
%%bash
set -euo pipefail
cd "$WORK"
PYTHONPATH=src python scripts/run_0072_zero_shot_colab.py --stage s2 \
  --models "$DRIVE/0071/models" --device cuda \
  --embed-batch-size 8 --rerank-batch-size 16 2>&1 | tee "$DRIVE/0072/logs/s2.log"
cp artifacts/semantic/0072/S2_*.json* "$DRIVE/0072/artifacts/" 2>/dev/null || true
```

## Cell 8 — frozen selection on FINAL reranked metrics

```bash
%%bash
set -euo pipefail
cd "$WORK"
PYTHONPATH=src python scripts/run_0072_zero_shot_colab.py --stage select \
  2>&1 | tee "$DRIVE/0072/logs/select.log"
cp artifacts/semantic/0072/selection.json "$DRIVE/0072/artifacts/"
```

Ordering is frozen: final Recall@10 → final Recall@1 → final MRR → fewer reranker
regressions → deterministic name. Dense metrics are reported but **cannot** decide selection.

## Cell 9 — package

```bash
%%bash
set -euo pipefail
cd "$WORK"
PYTHONPATH=src python scripts/run_0072_zero_shot_colab.py --stage package \
  --package-to "$DRIVE/0072/0072-results.tgz"
cp "$DRIVE"/0072/logs/*.log "$DRIVE/0072/artifacts/" 2>/dev/null || true
sha256sum "$DRIVE/0072/0072-results.tgz"
echo "NO training was run. NO public inference was run."
```

---

## Resume after a disconnect

Re-run Cells 1–4, then continue from the first stage with no artifact on Drive. `--stage all`
skips any system whose `*_complete.json` already exists, and the dense index is cached per
system in `artifacts/semantic/0072/cache/`, so a reconnect never re-encodes 227k documents.

## Public inference — deliberately not in this runbook

The code path exists (`scripts/run_semantic_linker_0071.py`) and Colab A100/H100 is a
legitimate place to run it. It is excluded here for two reasons, neither of which is a claimed
competition rule:

1. the offline acceptance gate has not passed yet — §12 requires that first;
2. `data/organizer_test/input` is gitignored, so it is simply not in the clone. If you choose
   to run public inference on Colab later, upload that input yourself; there is no rule
   against processing it there that this repository has established.
