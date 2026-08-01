#!/usr/bin/env python3
"""Complete-system A/B for structured RxNorm on GPU (Audit 0074 GPU handoff).

    A = frozen scored 0073 S1        frozen documents + frozen S1_doc_vectors.pt
    B = A + accepted 0074 structured RxNorm
                                     ICD text and ICD vectors IDENTICAL to frozen 0072;
                                     only the RxNorm slice is re-encoded

Both arms run the real pipeline: lexical + dense union -> Qwen3-Reranker-4B. NULL stays in
shadow and no threshold is calibrated here. No public-test clinical text is required or used,
and no model is trained.

The frozen 0072 cache is READ ONLY. New vectors are written to a separate 0074 path, built
temporary-first and finalized atomically, so a half-refreshed cache can never be mistaken for
a valid one.

Stages: preflight | build-docs | rebuild-rxnorm-vectors | ab | package | all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.kb.rxnorm.structured import StructuredDrug, semantic_text  # noqa: E402
from mednorm_vi.linking.semantic.hybrid import (  # noqa: E402
    NULL_MODE_SHADOW,
    build_pool,
    run_hybrid,
)
from mednorm_vi.linking.semantic.representation import SemanticQuery  # noqa: E402
from mednorm_vi.linking.semantic.reranker import Qwen3Reranker  # noqa: E402
from mednorm_vi.linking.semantic.runtime import (  # noqa: E402
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_ICD_ROWS,
    EXPECTED_RXNORM_ROWS,
    FROZEN_0072_ROW_ORDER,
    S1_EMBEDDING_DIRNAME,
    S1_EMBEDDING_MODEL,
    S1_EMBEDDING_REVISION,
    S1_RERANKER_DIRNAME,
    S1_RERANKER_MODEL,
    S1_RERANKER_REVISION,
)

MANIFEST_VERSION = "structured-rxnorm-dense-v1"


def log(message: str) -> None:
    print(f"[0074] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(tensor.detach().cpu().float().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def peak_vram_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except ImportError:
        pass
    return 0.0


def read_documents(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ordered_ids(rows: list[dict[str, Any]], row_order: str) -> list[str]:
    """The cache's row order. `sorted_concept_id` is the frozen 0072 contract."""
    if row_order == FROZEN_0072_ROW_ORDER:
        return sorted(r["concept_id"] for r in rows)
    return [r["concept_id"] for r in rows]


# ------------------------------------------------------------------------------ preflight


def preflight(args: argparse.Namespace) -> list[str]:
    """Every reason the run cannot proceed, checked before a model byte is loaded."""
    problems: list[str] = []
    for label, path in (
        ("frozen documents", args.frozen_documents),
        ("frozen dense index", args.frozen_vectors),
        ("structured jsonl", args.structured),
        ("RxNorm index", args.rxnorm_index),
        ("ICD v3 index", args.icd_v3_index),
        ("ICD v4.1 index", args.icd_v41_index),
    ):
        if not path.exists():
            problems.append(f"{label} not found: {path}")
    for dirname, model in (
        (S1_EMBEDDING_DIRNAME, S1_EMBEDDING_MODEL),
        (S1_RERANKER_DIRNAME, S1_RERANKER_MODEL),
    ):
        config = args.model_root / dirname / "config.json"
        if not config.is_file():
            problems.append(f"missing {model}: {config}")
    if problems:
        return problems

    rows = read_documents(args.frozen_documents)
    counts = {"ICD10": 0, "RXNORM": 0}
    for row in rows:
        counts[row.get("ontology", "")] = counts.get(row.get("ontology", ""), 0) + 1
    if len(rows) != EXPECTED_DOCUMENT_COUNT:
        problems.append(f"corpus has {len(rows):,} rows, expected {EXPECTED_DOCUMENT_COUNT:,}")
    if counts.get("ICD10") != EXPECTED_ICD_ROWS:
        problems.append(f"ICD rows {counts.get('ICD10')}, expected {EXPECTED_ICD_ROWS:,}")
    if counts.get("RXNORM") != EXPECTED_RXNORM_ROWS:
        problems.append(f"RxNorm rows {counts.get('RXNORM')}, expected {EXPECTED_RXNORM_ROWS:,}")

    try:
        import torch

        vectors = torch.load(args.frozen_vectors, map_location="cpu")
        if int(vectors.shape[0]) != len(rows):
            problems.append(
                f"frozen cache has {int(vectors.shape[0]):,} vectors but the corpus has "
                f"{len(rows):,} rows"
            )
        log(f"frozen cache shape {tuple(vectors.shape)} dtype {vectors.dtype}")
    except ImportError:
        problems.append("torch not installed; cannot validate the frozen cache")
    return problems


# ----------------------------------------------------------------------------- build-docs


def build_docs(args: argparse.Namespace) -> Path:
    """Structured RxNorm document text. ICD rows are copied through byte-identically."""
    rows = read_documents(args.frozen_documents)
    structured: dict[str, StructuredDrug] = {}
    with args.structured.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                drug = StructuredDrug.from_dict(json.loads(line))
                structured[drug.rxcui] = drug

    out = args.out / "documents_structured.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".jsonl.tmp")
    changed = icd_kept = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.get("ontology") == "RXNORM":
                drug = structured.get(row["concept_id"])
                if drug is not None and drug.has_structure:
                    new = dict(row)
                    new["text"] = semantic_text(drug)
                    if new["text"] != row["text"]:
                        changed += 1
                    row = new
            else:
                icd_kept += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(out)
    log(
        f"structured documents written: {changed:,} RxNorm rows changed, "
        f"{icd_kept:,} ICD rows byte-identical -> {out}"
    )
    return out


# ------------------------------------------------------------- rebuild-rxnorm-vectors


def encode(texts: list[str], args: argparse.Namespace, label: str) -> Any:
    """Encode with the pinned embedding model, then release it (Audit 0074 §13)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    source = str(args.model_root / S1_EMBEDDING_DIRNAME)
    tokenizer = AutoTokenizer.from_pretrained(source)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(source, torch_dtype=dtype).to(args.device).eval()
    log(f"embedding model loaded for {label}")

    chunks = []
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(texts), args.embed_batch_size):
            batch = texts[start : start + args.embed_batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(args.device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
            done = min(start + args.embed_batch_size, len(texts))
            if done % (args.embed_batch_size * 50) == 0 or done == len(texts):
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                eta = (len(texts) - done) / rate if rate else 0.0
                log(
                    f"  encode {label}: {done:,}/{len(texts):,} "
                    f"({rate:.1f} rows/s, ETA {eta / 60:.1f} min)"
                )
    del model, tokenizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    log(f"embedding model unloaded after {label}")
    return torch.cat(chunks) if chunks else torch.empty(0)


def rebuild_rxnorm_vectors(args: argparse.Namespace) -> Path:
    """Frozen ICD slice + newly encoded structured RxNorm slice, finalized atomically."""
    import torch

    documents = args.out / "documents_structured.jsonl"
    if not documents.is_file():
        raise SystemExit("run --stage build-docs first")
    rows = read_documents(documents)
    frozen_rows = read_documents(args.frozen_documents)
    order = ordered_ids(frozen_rows, args.row_order)
    position_of = {concept_id: index for index, concept_id in enumerate(order)}
    text_of = {r["concept_id"]: r["text"] for r in rows}
    ontology_of = {r["concept_id"]: r["ontology"] for r in frozen_rows}

    frozen = torch.load(args.frozen_vectors, map_location="cpu")
    if int(frozen.shape[0]) != len(order):
        raise SystemExit(
            f"frozen cache rows {int(frozen.shape[0]):,} != corpus rows {len(order):,}"
        )

    icd_positions = [position_of[c] for c in order if ontology_of[c] == "ICD10"]
    rx_positions = [position_of[c] for c in order if ontology_of[c] == "RXNORM"]
    log(
        f"row order '{args.row_order}': ICD {len(icd_positions):,} rows, "
        f"RxNorm {len(rx_positions):,} rows"
    )
    icd_before = sha256_tensor(frozen[icd_positions])

    rx_texts = [text_of[order[p]] for p in rx_positions]
    rebuilt = encode(rx_texts, args, "RxNorm slice")
    if rebuilt.shape[0] != len(rx_positions):
        raise SystemExit("re-encoded RxNorm slice has the wrong row count")
    if rebuilt.shape[1] != frozen.shape[1]:
        raise SystemExit(
            f"dimension mismatch: rebuilt {rebuilt.shape[1]} vs frozen {frozen.shape[1]}"
        )

    combined = frozen.clone()
    combined[rx_positions] = rebuilt.to(combined.dtype)
    icd_after = sha256_tensor(combined[icd_positions])
    if icd_after != icd_before:
        raise SystemExit(
            "ICD slice changed during the rebuild; only the RxNorm slice may be replaced"
        )

    target = args.out / "cache" / "S1_structured_rxnorm_doc_vectors.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not args.overwrite:
        raise SystemExit(f"{target} already exists; pass --overwrite to replace it")
    temporary = target.with_suffix(".pt.tmp")
    torch.save(combined, temporary)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "row_order": args.row_order,
        "row_order_contract": (
            "vector row i corresponds to ordered_ids[i]; for 'sorted_concept_id' that is "
            "sorted(concept_id) over the frozen corpus, which places every numeric RxCUI "
            "before every letter-prefixed ICD code"
        ),
        "document_count": len(order),
        "icd_rows": len(icd_positions),
        "rxnorm_rows": len(rx_positions),
        "frozen_documents_sha256": sha256_file(args.frozen_documents),
        "structured_documents_sha256": sha256_file(documents),
        "frozen_vectors_sha256": sha256_file(args.frozen_vectors),
        "frozen_icd_slice_sha256": icd_before,
        "rebuilt_rxnorm_slice_sha256": sha256_tensor(rebuilt),
        "final_tensor_sha256": sha256_tensor(combined),
        "icd_slice_unchanged": True,
        "embedding_model": S1_EMBEDDING_MODEL,
        "embedding_revision": S1_EMBEDDING_REVISION,
        "dimensions": int(combined.shape[1]),
        "dtype": str(combined.dtype),
        "training_performed": False,
        "null_mode": NULL_MODE_SHADOW,
        "contains_clinical_text": False,
    }
    manifest_path = target.with_suffix(".manifest.json")
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    # Atomic finalization: both files appear together or neither does.
    temporary.replace(target)
    manifest_temporary.replace(manifest_path)
    log(f"finalized {target}")
    log(f"  ICD slice sha256    {icd_before[:16]}… (unchanged)")
    log(f"  final tensor sha256 {manifest['final_tensor_sha256'][:16]}…")
    return target


def verify_cache(vectors_path: Path, documents: Path, frozen_documents: Path) -> list[str]:
    """Refuse a cache whose manifest does not describe it. Fail closed on any mismatch."""
    problems: list[str] = []
    manifest_path = vectors_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return [f"no manifest beside {vectors_path}; a cache without one is never valid"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        problems.append(f"manifest version {manifest.get('manifest_version')!r} not supported")
    if manifest.get("structured_documents_sha256") != sha256_file(documents):
        problems.append("structured documents changed since the cache was built")
    if manifest.get("frozen_documents_sha256") != sha256_file(frozen_documents):
        problems.append("frozen corpus changed since the cache was built")
    try:
        import torch

        vectors = torch.load(vectors_path, map_location="cpu")
        if int(vectors.shape[0]) != manifest.get("document_count"):
            problems.append("cache row count disagrees with its manifest")
        if sha256_tensor(vectors) != manifest.get("final_tensor_sha256"):
            problems.append("cache tensor hash disagrees with its manifest")
    except ImportError:
        problems.append("torch not installed; cannot verify the cache")
    return problems


# ------------------------------------------------------------------------------------ ab


def evaluate_arm(
    arm: str,
    args: argparse.Namespace,
    documents: Path,
    vectors_path: Path,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """One complete-system arm: lexical + dense union -> real Qwen3 reranker."""
    import torch

    from mednorm_vi.kb.indexing.retrieval import load_index
    from mednorm_vi.linking.semantic import rxnorm_structured as rs

    started = time.time()
    rows = read_documents(documents)
    frozen_rows = read_documents(args.frozen_documents)
    order = ordered_ids(frozen_rows, args.row_order)
    ontology_of = {r["concept_id"]: r["ontology"] for r in frozen_rows}
    text_of = {r["concept_id"]: r["text"] for r in rows}
    rx_positions = [i for i, c in enumerate(order) if ontology_of[c] == "RXNORM"]
    rx_ids = [order[i] for i in rx_positions]

    vectors = torch.load(vectors_path, map_location="cpu")
    subset = vectors[rx_positions]
    rxnorm = load_index(args.rxnorm_index)

    structured: dict[str, StructuredDrug] = {}
    if arm == "B":
        with args.structured.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    drug = StructuredDrug.from_dict(json.loads(line))
                    structured[drug.rxcui] = drug

    queries = [SemanticQuery(c["query"], "THUỐC", "", "", "RXNORM").render() for c in cases]
    query_vectors = encode(queries, args, f"{arm} queries")

    backend = Qwen3Reranker(
        S1_RERANKER_MODEL,
        S1_RERANKER_REVISION,
        device=args.device,
        batch_size=args.rerank_batch_size,
        local_path=str(args.model_root / S1_RERANKER_DIRNAME),
    )
    log(f"{arm}: reranker loaded")

    results: dict[str, list[str]] = {}
    lexical_only: dict[str, list[str]] = {}
    dense_added = 0
    for index, case in enumerate(cases):
        scores = query_vectors[index] @ subset.T
        top = torch.topk(scores, min(args.dense_topk, len(rx_ids)))
        dense_hits = [
            (rx_ids[i], float(s))
            for i, s in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ]
        pool = build_pool(
            case["query"],
            v3_index=rxnorm,
            v41_index=None,
            dense_hits=dense_hits,
            governed_index=rxnorm,
            v3_topk=args.rxnorm_topk,
            dense_topk=args.dense_topk,
        )
        lexical_ids = [c.concept_id for c in pool if c.v3_rank]
        lexical_only[case["query"]] = lexical_ids
        if case["gold"] in [c for c, _ in dense_hits] and case["gold"] not in lexical_ids:
            dense_added += 1

        outcome = run_hybrid(
            case["query"],
            SemanticQuery(case["query"], "THUỐC", "", "", "RXNORM").render(with_instruction=False),
            pool,
            {c.concept_id: text_of.get(c.concept_id, c.concept_id) for c in pool},
            backend,
            ontology="RXNORM",
            icd_index=None,
            null_mode=NULL_MODE_SHADOW,
            limit=args.final_limit,
        )
        ranked = list(outcome.codes)
        if arm == "B":
            ranked = rs.reorder(ranked, case["query"], structured)
        results[case["query"]] = ranked
        if (index + 1) % args.progress == 0 or index + 1 == len(cases):
            elapsed = time.time() - started
            rate = (index + 1) / elapsed if elapsed else 0.0
            log(
                f"  {arm} eval {index + 1:,}/{len(cases):,} "
                f"({rate:.2f} q/s, ETA {(len(cases) - index - 1) / rate / 60:.1f} min)"
                if rate
                else f"  {arm} eval {index + 1:,}/{len(cases):,}"
            )
    backend.unload()
    log(f"{arm}: reranker unloaded")
    return {
        "ranked": results,
        "lexical_only": lexical_only,
        "dense_added_correct": dense_added,
        "runtime_seconds": round(time.time() - started, 1),
        "peak_vram_gib": peak_vram_gib(),
    }


def score(ranked: dict[str, list[str]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    at = {k: 0 for k in (1, 2, 5, 10)}
    reciprocal = 0.0
    covered = 0
    for case in cases:
        order = ranked[case["query"]]
        gold = case["gold"]
        if gold in order:
            covered += 1
            reciprocal += 1.0 / (order.index(gold) + 1)
        for k in at:
            if gold in order[:k]:
                at[k] += 1
    total = len(cases) or 1
    return {
        **{f"recall_at_{k}": round(at[k] / total, 4) for k in at},
        "mrr": round(reciprocal / total, 4),
        "coverage": round(covered / total, 4),
        "n": len(cases),
    }


def hard_negative_rate(
    ranked: dict[str, list[str]], cases: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    above = considered = 0
    for case in cases:
        siblings = [s for s in case.get(key, []) if s]
        order = ranked[case["query"]]
        if not siblings or case["gold"] not in order:
            continue
        considered += 1
        gold_at = order.index(case["gold"])
        if any(s in order and order.index(s) < gold_at for s in siblings):
            above += 1
    return {
        "considered": considered,
        "above_gold": above,
        "rate": round(above / considered, 4) if considered else 0.0,
    }


def run_ab(args: argparse.Namespace) -> dict[str, Any]:
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    log(f"benchmark cases {len(cases):,} (identical query set for A and B)")
    structured_documents = args.out / "documents_structured.jsonl"
    rebuilt = args.out / "cache" / "S1_structured_rxnorm_doc_vectors.pt"
    blockers = verify_cache(rebuilt, structured_documents, args.frozen_documents)
    if blockers:
        raise SystemExit("cache verification failed:\n  - " + "\n  - ".join(blockers))

    arm_a = evaluate_arm("A", args, args.frozen_documents, args.frozen_vectors, cases)
    arm_b = evaluate_arm("B", args, structured_documents, rebuilt, cases)

    fixes = regressions = 0
    for case in cases:
        gold = case["gold"]
        before = arm_a["ranked"][case["query"]][:1] == [gold]
        after = arm_b["ranked"][case["query"]][:1] == [gold]
        fixes += after and not before
        regressions += before and not after

    def subset(ranked: dict[str, list[str]], key: str) -> dict[str, Any]:
        chosen = [c for c in cases if c.get(key)]
        return score(ranked, chosen) if chosen else {"n": 0}

    report: dict[str, Any] = {
        "complete_system": True,
        "reranker": S1_RERANKER_MODEL,
        "reranker_revision": S1_RERANKER_REVISION,
        "embedding": S1_EMBEDDING_MODEL,
        "embedding_revision": S1_EMBEDDING_REVISION,
        "null_mode": NULL_MODE_SHADOW,
        "training_performed": False,
        "A_frozen_0073": score(arm_a["ranked"], cases),
        "B_structured_rxnorm": score(arm_b["ranked"], cases),
        "subsets": {
            name: {"A": subset(arm_a["ranked"], name), "B": subset(arm_b["ranked"], name)}
            for name in ("states_strength", "states_form", "brand")
        },
        "hard_negatives": {
            name: {
                "A": hard_negative_rate(arm_a["ranked"], cases, name),
                "B": hard_negative_rate(arm_b["ranked"], cases, name),
            }
            for name in ("wrong_strength_siblings", "wrong_form_siblings")
        },
        "reranker_fixes_top1": fixes,
        "reranker_regressions_top1": regressions,
        "dense_added_correct": {
            "A": arm_a["dense_added_correct"],
            "B": arm_b["dense_added_correct"],
        },
        "runtime_seconds": {"A": arm_a["runtime_seconds"], "B": arm_b["runtime_seconds"]},
        "peak_vram_gib": max(arm_a["peak_vram_gib"], arm_b["peak_vram_gib"]),
        "governed_candidate_ids_only": all(
            all(rxcui.isdigit() for rxcui in order) for order in arm_b["ranked"].values()
        ),
        "icd_evaluated": False,
        "icd_invariant_by_construction": "ICD documents and ICD vector rows are untouched",
    }
    a, b = report["A_frozen_0073"], report["B_structured_rxnorm"]
    report["gate"] = {
        "no_material_overall_regression": b["recall_at_1"] >= a["recall_at_1"] - 0.005
        and b["mrr"] >= a["mrr"] - 0.005,
        "recall_at_1_or_mrr_improves": b["recall_at_1"] > a["recall_at_1"] or b["mrr"] > a["mrr"],
        "structured_subsets_improved": all(
            report["hard_negatives"][name]["B"]["rate"]
            <= report["hard_negatives"][name]["A"]["rate"]
            for name in report["hard_negatives"]
        ),
        "reranker_regressions_do_not_erase_gains": regressions <= fixes,
        "candidate_ids_governed": report["governed_candidate_ids_only"],
    }
    report["verdict"] = (
        "COMPLETE_SYSTEM_ACCEPTED"
        if all(report["gate"].values())
        else "COMPLETE_SYSTEM_NOT_ACCEPTED"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "complete_system_ab.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "subsets"}, indent=2, sort_keys=True))
    return report


def package(args: argparse.Namespace) -> None:
    target = args.out / "0074-complete-system-results.tgz"
    staging = args.out / "_package"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ("complete_system_ab.json",):
        source = args.out / name
        if source.is_file():
            shutil.copy2(source, staging / name)
    cache = args.out / "cache" / "S1_structured_rxnorm_doc_vectors.manifest.json"
    if cache.is_file():
        shutil.copy2(cache, staging / "dense_cache_manifest.json")
    shutil.make_archive(str(target.with_suffix("")), "gztar", root_dir=staging)
    shutil.rmtree(staging)
    log(f"packaged -> {target}  sha256 {sha256_file(target)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preflight", "build-docs", "rebuild-rxnorm-vectors", "ab", "package", "all"],
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        required=True,
        help="frozen 0072 root, e.g. /content/drive/MyDrive/MedNorm-VI/0072",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="0074 output root, e.g. /content/drive/MyDrive/MedNorm-VI/0074",
    )
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--frozen-documents", type=Path, default=None)
    parser.add_argument("--frozen-vectors", type=Path, default=None)
    parser.add_argument(
        "--structured",
        type=Path,
        default=REPO / "artifacts/rxnorm_structured/0074/structured.jsonl",
    )
    parser.add_argument(
        "--cases", type=Path, default=REPO / "artifacts/rxnorm_structured/0074/cases.json"
    )
    parser.add_argument(
        "--rxnorm-index",
        type=Path,
        default=REPO / "indices/candidate/rxnorm/competition-v3/index.json",
    )
    parser.add_argument(
        "--icd-v3-index",
        type=Path,
        default=REPO / "indices/candidate/icd10_vi/competition-v3/index.json",
    )
    parser.add_argument(
        "--icd-v41-index",
        type=Path,
        default=REPO / "indices/candidate/icd10_vi/competition-v4.1/index.json",
    )
    parser.add_argument("--row-order", default=FROZEN_0072_ROW_ORDER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--rxnorm-topk", type=int, default=20)
    parser.add_argument("--dense-topk", type=int, default=50)
    parser.add_argument("--final-limit", type=int, default=20)
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.model_root = args.model_root or args.frozen_root / "models"
    args.frozen_documents = args.frozen_documents or args.frozen_root / "artifacts/documents.jsonl"
    args.frozen_vectors = args.frozen_vectors or args.frozen_root / "cache/S1_doc_vectors.pt"

    if args.stage in ("preflight", "all"):
        problems = preflight(args)
        if problems:
            print("BLOCKED - preflight failed:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2
        log("preflight OK")
        if args.stage == "preflight":
            return 0
    if args.stage in ("build-docs", "all"):
        build_docs(args)
        if args.stage == "build-docs":
            return 0
    if args.stage in ("rebuild-rxnorm-vectors", "all"):
        rebuild_rxnorm_vectors(args)
        if args.stage == "rebuild-rxnorm-vectors":
            return 0
    if args.stage in ("ab", "all"):
        run_ab(args)
        if args.stage == "ab":
            return 0
    if args.stage in ("package", "all"):
        package(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
