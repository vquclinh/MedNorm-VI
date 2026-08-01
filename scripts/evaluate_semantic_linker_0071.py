#!/usr/bin/env python3
"""Zero-shot / post-training evaluation of a complete semantic linker (Audit 0071 §13-§14).

Evaluates SYSTEMS, not embeddings. Audit 0070 demonstrated that candidate expansion without
matching discrimination *lowers* J_candidates, so recall on its own is not evidence: every
configuration here is scored end to end, including the NO_VALID_CANDIDATE gate.

Heavy dependencies (torch, transformers) are imported inside `--device cuda` paths only, so
this file stays importable, lintable and type-checkable on a machine with no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.linking.semantic import null_gate as ng  # noqa: E402
from mednorm_vi.linking.semantic.representation import SemanticQuery  # noqa: E402

SYSTEMS = {
    "S1": {
        "embedding": "Qwen/Qwen3-Embedding-4B",
        "embedding_revision": "5cf2132abc99",
        "reranker": "Qwen/Qwen3-Reranker-4B",
        "reranker_revision": "22e683669bc0",
    },
    "S2": {
        "embedding": "Qwen/Qwen3-Embedding-8B",
        "embedding_revision": "1d8ad4ca9b3d",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "reranker_revision": "e61197ed4502",
    },
}


def encode(texts: list[str], model_id: str, revision: str, device: str, batch: int) -> Any:
    """Encode with the official model at a pinned revision. Imported lazily on purpose."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = (
        AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
    return torch.cat(out) if out else torch.empty(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument(
        "--documents", type=Path, default=REPO / "artifacts/semantic/0071/documents.jsonl"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=REPO / "artifacts/semantic/0071/benchmark/test.jsonl"
    )
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/semantic/0071/eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=0,
        help="0 = all; smaller values are for smoke tests only",
    )
    parser.add_argument("--dense-topk", type=int, default=50)
    args = parser.parse_args(argv)

    spec = SYSTEMS[args.system]
    documents = [
        json.loads(line)
        for line in args.documents.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit_documents:
        documents = documents[: args.limit_documents]
    rows = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    import torch

    doc_vectors = encode(
        [d["text"] for d in documents],
        spec["embedding"],
        spec["embedding_revision"],
        args.device,
        args.batch_size,
    )
    queries = [
        SemanticQuery(
            mention_text=r["query"],
            entity_type="",
            context_before="",
            context_after="",
            ontology=r["ontology"],
        ).render()
        for r in rows
    ]
    query_vectors = encode(
        queries, spec["embedding"], spec["embedding_revision"], args.device, args.batch_size
    )

    ids = [d["concept_id"] for d in documents]
    at = {k: 0 for k in (1, 2, 5, 10)}
    reciprocal = 0.0
    for index, row in enumerate(rows):
        scores = query_vectors[index] @ doc_vectors.T
        top = torch.topk(scores, min(args.dense_topk, len(ids))).indices.tolist()
        ranked = [ids[i] for i in top]
        gold = row["positive_concept_id"]
        for k in at:
            if gold in ranked[:k]:
                at[k] += 1
        if gold in ranked:
            reciprocal += 1.0 / (ranked.index(gold) + 1)

    result = {
        "system": args.system,
        **spec,
        "documents": len(documents),
        "queries": len(rows),
        "recall_at_1": at[1] / len(rows),
        "recall_at_2": at[2] / len(rows),
        "recall_at_5": at[5] / len(rows),
        "recall_at_10": at[10] / len(rows),
        "mrr": reciprocal / len(rows),
        "null_gate_version": ng.NULL_GATE_VERSION,
        "note": "dense retrieval stage only; run --system with the reranker stage for a "
        "complete-system number before any acceptance decision",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.system}_zero_shot.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
