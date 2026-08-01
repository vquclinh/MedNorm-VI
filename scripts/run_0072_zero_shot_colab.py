#!/usr/bin/env python3
"""Complete zero-shot hybrid semantic linker orchestrator (Audit 0072 §13).

One entry point for the whole Colab run so the owner never composes intermediate scripts.
Stages are resume-aware: each writes its artifact and is skipped when that artifact already
exists, so a disconnect never costs a 227k-document re-encode.

**No training happens here.** Audit 0072 §11 stops after zero-shot evidence exists.

    --stage smoke     tiny end-to-end pass, proves plumbing
    --stage s1        complete zero-shot S1  (embed -> unload -> rerank -> unload)
    --stage s2        complete zero-shot S2
    --stage select    frozen selection on FINAL reranked metrics
    --stage package   tar the artifacts to bring back
    --stage all       kb -> s1 -> s2 -> select -> package
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.linking.semantic.hybrid import (  # noqa: E402
    NULL_MODE_SHADOW,
    build_pool,
    run_hybrid,
)
from mednorm_vi.linking.semantic.representation import SemanticQuery  # noqa: E402
from mednorm_vi.linking.semantic.reranker import Qwen3Reranker  # noqa: E402

#: Full immutable revisions (Audit 0072 §9). Abbreviated shas and branch names are refused.
SYSTEMS: dict[str, dict[str, str]] = {
    "S1": {
        "embedding": "Qwen/Qwen3-Embedding-4B",
        "embedding_revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
        "embedding_dir": "emb-4b",
        "reranker": "Qwen/Qwen3-Reranker-4B",
        "reranker_revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
        "reranker_dir": "rr-4b",
    },
    "S2": {
        "embedding": "Qwen/Qwen3-Embedding-8B",
        "embedding_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "embedding_dir": "emb-8b",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "reranker_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "reranker_dir": "rr-06b",
    },
}

ART = REPO / "artifacts" / "semantic" / "0072"


def log(message: str) -> None:
    print(f"[0072] {message}", flush=True)


def peak_vram_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except ImportError:
        pass
    return 0.0


def load_documents(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["concept_id"]] = row
    return out


def embed(
    texts: list[str], spec: dict[str, str], models_root: Path, device: str, batch: int
) -> Any:
    """Encode with the pinned embedding model, then release it."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    source = models_root / spec["embedding_dir"]
    where = str(source) if source.exists() else spec["embedding"]
    kwargs = {} if source.exists() else {"revision": spec["embedding_revision"]}
    tokenizer = AutoTokenizer.from_pretrained(where, **kwargs)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(where, torch_dtype=dtype, **kwargs).to(device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            enc = tokenizer(
                texts[start : start + batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(chunks) if chunks else torch.empty(0)


def evaluate_system(system: str, args: argparse.Namespace) -> dict[str, Any]:
    """Complete zero-shot: dense retrieval, union, REAL reranking, H2, shadow null gate."""
    import torch

    from mednorm_vi.kb.indexing.retrieval import load_index

    spec = SYSTEMS[system]
    started = time.time()
    documents = load_documents(args.documents)
    rows = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log(f"{system}: {len(documents):,} documents, {len(rows):,} queries")

    doc_ids = sorted(documents)
    doc_texts = [documents[c]["text"] for c in doc_ids]

    cache = args.cache / f"{system}_doc_vectors.pt"
    if cache.exists():
        log(f"{system}: restoring dense index from {cache}")
        doc_vectors = torch.load(cache)
    else:
        log(f"{system}: encoding documents with {spec['embedding']}")
        doc_vectors = embed(doc_texts, spec, args.models, args.device, args.embed_batch_size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(doc_vectors, cache)

    queries = [
        SemanticQuery(
            mention_text=r["query"],
            entity_type=r.get("entity_type", ""),
            context_before=r.get("context_before", ""),
            context_after=r.get("context_after", ""),
            ontology=r["ontology"],
        ).render()
        for r in rows
    ]
    log(f"{system}: encoding {len(queries):,} queries")
    query_vectors = embed(queries, spec, args.models, args.device, args.embed_batch_size)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()  # embedding model gone before the reranker loads

    v3 = load_index(args.icd_v3) if args.icd_v3.exists() else None
    v41 = load_index(args.icd_v41) if args.icd_v41.exists() else None
    rx = load_index(args.rxnorm) if args.rxnorm.exists() else None

    log(f"{system}: loading reranker {spec['reranker']}")
    local = args.models / spec["reranker_dir"]
    backend = Qwen3Reranker(
        spec["reranker"],
        spec["reranker_revision"],
        device=args.device,
        batch_size=args.rerank_batch_size,
        local_path=str(local) if local.exists() else None,
    )

    dense_at = {k: 0 for k in (1, 2, 5, 10)}
    union_at = {k: 0 for k in (1, 2, 5, 10)}
    final_at = {k: 0 for k in (1, 2, 5, 10)}
    per_ontology: dict[str, dict[str, Any]] = {
        "ICD10": {"n": 0, "final_at": {k: 0 for k in (1, 2, 5, 10)}, "mrr": 0.0},
        "RXNORM": {"n": 0, "final_at": {k: 0 for k in (1, 2, 5, 10)}, "mrr": 0.0},
    }
    mrr = 0.0
    kept = overturned = dense_added = fixes = regressions = hierarchy_changes = 0
    pool_sizes: list[int] = []
    null_shadow: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        ontology = row["ontology"]
        gold = row["positive_concept_id"]
        scores = query_vectors[index] @ doc_vectors.T
        top = torch.topk(scores, min(args.dense_topk, len(doc_ids)))
        dense_hits = [
            (doc_ids[i], float(s))
            for i, s in zip(top.indices.tolist(), top.values.tolist(), strict=True)
            if documents[doc_ids[i]]["ontology"] == ontology
        ]
        dense_ids = [c for c, _ in dense_hits]
        for k in dense_at:
            if gold in dense_ids[:k]:
                dense_at[k] += 1

        governed = (v41 or v3) if ontology == "ICD10" else rx
        pool = build_pool(
            row["query"],
            v3_index=v3 if ontology == "ICD10" else rx,
            v41_index=v41 if ontology == "ICD10" else None,
            dense_hits=dense_hits,
            governed_index=governed,
            v3_topk=args.v3_topk,
            v41_topk=args.v41_topk,
            dense_topk=args.dense_topk,
        )
        pool_sizes.append(len(pool))
        pool_ids = [c.concept_id for c in pool]
        for k in union_at:
            if gold in pool_ids[:k]:
                union_at[k] += 1
        if gold in dense_ids and gold not in [
            c.concept_id for c in pool if c.v3_rank or c.v41_rank
        ]:
            dense_added += 1

        query_text = SemanticQuery(
            mention_text=row["query"],
            entity_type=row.get("entity_type", ""),
            context_before=row.get("context_before", ""),
            context_after=row.get("context_after", ""),
            ontology=ontology,
        ).render(with_instruction=False)
        result = run_hybrid(
            row["query"],
            query_text,
            pool,
            {c: documents[c]["text"] for c in pool_ids if c in documents},
            backend,
            ontology=ontology,
            icd_index=v41 if ontology == "ICD10" else None,
            null_mode=NULL_MODE_SHADOW,
            limit=args.final_limit,
        )
        final = list(result.codes)
        for k in final_at:
            if gold in final[:k]:
                final_at[k] += 1
        if gold in final:
            mrr += 1.0 / (final.index(gold) + 1)
        bucket = per_ontology[ontology]
        bucket["n"] += 1
        for k in bucket["final_at"]:
            if gold in final[:k]:
                bucket["final_at"][k] += 1
        if gold in final:
            bucket["mrr"] += 1.0 / (final.index(gold) + 1)

        if result.v3_top1_kept:
            kept += 1
        if result.v3_top1_overturned:
            overturned += 1
            if result.final_top1 == gold:
                fixes += 1
            elif result.v3_top1 == gold:
                regressions += 1
        if result.hierarchy_applied:
            hierarchy_changes += 1
        null_shadow.append(
            {
                "query_index": index,
                "ontology": ontology,
                "null_decision": result.null_decision,
                "top_reranker_score": result.reranked[0]["reranker_score"]
                if result.reranked
                else 0.0,
                "pool_size": result.pool_size,
            }
        )
        if args.progress and (index + 1) % args.progress == 0:
            log(f"{system}: {index + 1}/{len(rows)} queries")

    backend.unload()
    total = len(rows)
    out = {
        "system": system,
        **{k: v for k, v in spec.items()},
        "documents": len(documents),
        "queries": total,
        "stage_dense": {f"recall_at_{k}": dense_at[k] / total for k in dense_at},
        "stage_union": {f"recall_at_{k}": union_at[k] / total for k in union_at},
        "final_reranked": {
            **{f"recall_at_{k}": final_at[k] / total for k in final_at},
            "mrr": mrr / total,
        },
        "by_ontology": {
            name: {
                "n": b["n"],
                **{
                    f"recall_at_{k}": (b["final_at"][k] / b["n"] if b["n"] else 0.0)
                    for k in b["final_at"]
                },
                "mrr": (b["mrr"] / b["n"] if b["n"] else 0.0),
            }
            for name, b in per_ontology.items()
        },
        "diagnostics": {
            "v3_top1_kept": kept,
            "v3_top1_overturned": overturned,
            "reranker_fixes": fixes,
            "reranker_regressions": regressions,
            "dense_added_correct_concepts": dense_added,
            "hierarchy_changes": hierarchy_changes,
            "mean_pool_size": sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0.0,
            "max_pool_size": max(pool_sizes) if pool_sizes else 0,
        },
        "null_mode": NULL_MODE_SHADOW,
        "null_shadow_records": len(null_shadow),
        "runtime_seconds": round(time.time() - started, 1),
        "peak_vram_gib": peak_vram_gib(),
        "warning": "stage_dense and stage_union are RETRIEVAL stages; only final_reranked is "
        "a complete-system result (Audit 0072 §7).",
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"{system}_complete.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    (ART / f"{system}_null_shadow.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in null_shadow), encoding="utf-8"
    )
    log(
        f"{system}: final R@1={out['final_reranked']['recall_at_1']:.4f} "
        f"R@10={out['final_reranked']['recall_at_10']:.4f} "
        f"MRR={out['final_reranked']['mrr']:.4f}"
    )
    return out


def lexical_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """The baseline the semantic system must beat (Audit 0072 §8)."""
    from mednorm_vi.kb.indexing.retrieval import load_index, search_index

    rows = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexes = {}
    if args.icd_v41.exists():
        indexes["ICD10"] = load_index(args.icd_v41)
    if args.rxnorm.exists():
        indexes["RXNORM"] = load_index(args.rxnorm)
    at = {k: 0 for k in (1, 2, 5, 10)}
    mrr = 0.0
    for row in rows:
        index = indexes.get(row["ontology"])
        if index is None:
            continue
        hits = [h.concept_id for h in search_index(index, row["query"], limit=10)]
        gold = row["positive_concept_id"]
        for k in at:
            if gold in hits[:k]:
                at[k] += 1
        if gold in hits:
            mrr += 1.0 / (hits.index(gold) + 1)
    total = len(rows)
    return {**{f"recall_at_{k}": at[k] / total for k in at}, "mrr": mrr / total, "queries": total}


def select(args: argparse.Namespace) -> dict[str, Any]:
    """Frozen selection on FINAL reranked metrics only (Audit 0072 §8)."""
    systems = {}
    for name in SYSTEMS:
        path = ART / f"{name}_complete.json"
        if path.exists():
            systems[name] = json.loads(path.read_text(encoding="utf-8"))
    if not systems:
        raise SystemExit("no complete-system results found; run --stage s1 and --stage s2")

    baseline = lexical_baseline(args)

    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, int, str]:
        name, data = item
        final = data["final_reranked"]
        return (
            -final["recall_at_10"],  # 1. final Recall@10
            -final["recall_at_1"],  # 2. final Recall@1
            -final["mrr"],  # 3. final MRR
            data["diagnostics"]["reranker_regressions"],  # 4. fewer anchor regressions
            name,  # 5. deterministic tie-break
        )

    ranked = sorted(systems.items(), key=key)
    winner, data = ranked[0]
    final = data["final_reranked"]
    passes = final["recall_at_10"] >= baseline["recall_at_10"]
    out = {
        "selection_rule": [
            "final_recall_at_10",
            "final_recall_at_1",
            "final_mrr",
            "fewer_reranker_regressions",
            "deterministic_name",
        ],
        "selected_on": "final_reranked (complete system, NOT dense retrieval)",
        "lexical_baseline": baseline,
        "systems": {
            n: {
                "final_reranked": d["final_reranked"],
                "stage_dense": d["stage_dense"],
                "diagnostics": d["diagnostics"],
            }
            for n, d in systems.items()
        },
        "selected": winner,
        "selected_final": final,
        "gate_final_recall_at_10_ge_baseline": passes,
        "preferred_recall_at_1_gt_baseline": final["recall_at_1"] > baseline["recall_at_1"],
        "preferred_mrr_gt_baseline": final["mrr"] > baseline["mrr"],
        "null_mode": NULL_MODE_SHADOW,
        "training_performed": False,
        "note": "Zero-shot only. Audit 0072 §11 forbids training in this run; NULL is shadow "
        "so it cannot affect selection.",
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "selection.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


def package(args: argparse.Namespace) -> None:
    target = args.package_to
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "czf", str(target), "-C", str(REPO), "artifacts/semantic/0072"],
        check=True,
    )
    log(f"packaged -> {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=["smoke", "s1", "s2", "select", "package", "all"]
    )
    parser.add_argument(
        "--documents", type=Path, default=REPO / "artifacts/semantic/0071/documents.jsonl"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=REPO / "artifacts/semantic/0071/benchmark/test_2k.jsonl"
    )
    parser.add_argument(
        "--models", type=Path, default=Path("/content/drive/MyDrive/mednorm-vi/0071/models")
    )
    parser.add_argument("--cache", type=Path, default=REPO / "artifacts/semantic/0072/cache")
    parser.add_argument(
        "--icd-v3", type=Path, default=REPO / "indices/candidate/icd10_vi/competition-v3/index.json"
    )
    parser.add_argument(
        "--icd-v41",
        type=Path,
        default=REPO / "indices/candidate/icd10_vi/competition-v4.1/index.json",
    )
    parser.add_argument(
        "--rxnorm", type=Path, default=REPO / "indices/candidate/rxnorm/competition-v3/index.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--v3-topk", type=int, default=20)
    parser.add_argument("--v41-topk", type=int, default=20)
    parser.add_argument("--dense-topk", type=int, default=50)
    parser.add_argument("--final-limit", type=int, default=20)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument(
        "--package-to",
        type=Path,
        default=Path("/content/drive/MyDrive/mednorm-vi/0072/0072-results.tgz"),
    )
    args = parser.parse_args(argv)

    if args.stage == "smoke":
        args.benchmark = args.benchmark.with_name("smoke.jsonl")
        rows = [
            json.loads(line)
            for line in (REPO / "artifacts/semantic/0071/benchmark/test_2k.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ][:8]
        args.benchmark.write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )
        log("smoke: 8 queries, full document pool")
        evaluate_system("S1", args)
        return 0
    if args.stage in ("s1", "s2"):
        evaluate_system(args.stage.upper(), args)
        return 0
    if args.stage == "select":
        select(args)
        return 0
    if args.stage == "package":
        package(args)
        return 0

    for system in ("S1", "S2"):
        if (ART / f"{system}_complete.json").exists():
            log(f"{system}: already complete, skipping (resume)")
            continue
        evaluate_system(system, args)
    select(args)
    package(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
