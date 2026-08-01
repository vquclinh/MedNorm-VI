#!/usr/bin/env python3
"""Benchmark the candidate linker against human-adjudicated gold (Audit 0065 §8).

**This evaluator refuses to report final metrics below the minimum reviewed counts** - 50
ICD and 30 RxNorm records. That refusal is the point. Audit 0063 had to report that
Recall@k was *undefined* because no gold codes existed; the failure mode this replaces is
reporting a confident number from a handful of annotations, which is worse than reporting
none because it looks like evidence.

Two separations are enforced rather than left to the caller:

* **ICD and RxNorm are never collapsed into one metric.** They fail differently - Audit
  0063 measured ICD as retrieval-shallow (8.8% verbatim alias coverage) and RxNorm as a
  ranking problem (46.2% exact score ties) - and a combined figure would hide both.
* **Only human gold counts.** `SILVER_UNIQUE_EXACT` is excluded by
  `linker_gold.is_human_gold`, so an automatic exact-alias match can never inflate a
  reported recall.

Usage::

    python scripts/evaluate_linker_gold.py
    python scripts/evaluate_linker_gold.py --pack ... --output runs/diagnostics/0065_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.annotation.linker_gold import (  # noqa: E402
    ONTOLOGY_ICD10,
    ONTOLOGY_RXNORM,
    is_human_gold,
    load_records,
)

DEFAULT_PACK = REPO / "data" / "private_linker_gold" / "0065" / "pack.jsonl"

MIN_REVIEWED = {ONTOLOGY_ICD10: 50, ONTOLOGY_RXNORM: 30}
RECALL_KS = (1, 2, 5, 10)

# §9 decision rules, as data so the audit and the code cannot disagree.
DECISION_ICD_RETRIEVAL = "ICD_SYNONYM_OR_RETRIEVAL_REPAIR"
DECISION_ICD_RERANK = "ICD_HIERARCHY_AWARE_RERANKER"
DECISION_RX_RERANK = "RXNORM_STRUCTURED_RERANKER"
DECISION_SHARED_RERANK = "SHARED_LEARNED_RERANKER"
DECISION_MORE_REVIEW = "MORE_HUMAN_ADJUDICATION_REQUIRED"

RECALL10_LOW = 0.70
TOP1_LOW = 0.50


def ranked_codes(record: dict[str, Any]) -> list[str]:
    """Candidates in the LINKER's order, not the shuffled display order."""
    return [
        candidate["code"]
        for candidate in sorted(
            record.get("offered_candidates") or [], key=lambda c: c["linker_rank"]
        )
    ]


def best_rank(record: dict[str, Any]) -> int | None:
    """1-based rank of the highest-ranked correct code, or None if none is offered."""
    gold = set(record["selected_codes"])
    if not gold:
        return None
    for position, code in enumerate(ranked_codes(record), start=1):
        if code in gold:
            return position
    return None


def subset_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recall@k / MRR / top-1 over records that HAVE a correct code to find.

    Records the reviewer marked `no_valid_candidate` are excluded from recall - there is
    nothing to retrieve - and counted separately, because that rate is itself a finding.
    """
    total = len(records)
    if not total:
        return {"records": 0}
    no_valid = [r for r in records if r["no_valid_candidate"]]
    answerable = [r for r in records if not r["no_valid_candidate"] and r["selected_codes"]]
    ranks = [best_rank(r) for r in answerable]
    found = [r for r in ranks if r is not None]

    recall = {
        f"recall_at_{k}": (sum(1 for r in found if r <= k) / len(answerable) if answerable else 0.0)
        for k in RECALL_KS
    }
    return {
        "records": total,
        "answerable": len(answerable),
        "no_valid_candidate": len(no_valid),
        "no_valid_candidate_rate": len(no_valid) / total,
        "empty_offered_set": sum(1 for r in records if not r.get("offered_candidates")),
        "empty_offered_set_rate": (
            sum(1 for r in records if not r.get("offered_candidates")) / total
        ),
        "kb_coverage": len(found) / len(answerable) if answerable else 0.0,
        **recall,
        "mrr": (sum(1.0 / r for r in found) / len(answerable)) if answerable else 0.0,
        "top1_exact_accuracy": (
            sum(1 for r in found if r == 1) / len(answerable) if answerable else 0.0
        ),
        "mean_candidate_set_size": (
            sum(len(r.get("offered_candidates") or []) for r in records) / total
        ),
        "candidate_set_size_distribution": dict(
            sorted(Counter(len(r.get("offered_candidates") or []) for r in records).items())
        ),
        "multi_code_gold": sum(1 for r in records if len(r["selected_codes"]) > 1),
    }


def stratify(records: list[dict[str, Any]], key) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(key(record)), []).append(record)
    return {name: subset_metrics(rows) for name, rows in sorted(buckets.items())}


def icd_specific(records: list[dict[str, Any]]) -> dict[str, Any]:
    """ICD-only diagnostics: specificity behaviour and parent/child confusion."""
    answerable = [r for r in records if not r["no_valid_candidate"] and r["selected_codes"]]
    by_class: Counter[str] = Counter()
    top1_class: Counter[str] = Counter()
    parent_child = 0
    vague_top1_when_specific_gold = 0
    for record in answerable:
        ranked = ranked_codes(record)
        gold = set(record["selected_codes"])
        structures = {c["code"]: c["structure"] for c in record["offered_candidates"]}
        for code in gold:
            if code in structures:
                by_class[structures[code].get("specificity_class", "?")] += 1
        if ranked:
            top = ranked[0]
            top_class = structures.get(top, {}).get("specificity_class", "?")
            top1_class[top_class] += 1
            if top not in gold:
                # Did the linker pick an ancestor or descendant of a correct code?
                if any(top.startswith(g[:3]) and top != g for g in gold):
                    parent_child += 1
                if top_class in ("category", "unspecified", "other") and any(
                    structures.get(g, {}).get("specificity_class") == "specific" for g in gold
                ):
                    vague_top1_when_specific_gold += 1
    return {
        "gold_specificity_distribution": dict(by_class),
        "top1_specificity_distribution": dict(top1_class),
        "parent_child_confusions": parent_child,
        "vague_top1_when_gold_is_specific": vague_top1_when_specific_gold,
        "vague_top1_overuse_rate": (
            vague_top1_when_specific_gold / len(answerable) if answerable else 0.0
        ),
        "alias_language_coverage": stratify(
            records,
            lambda r: "exact_alias" if r["provenance"].get("exact_alias_present") else "fuzzy_only",
        ),
    }


def rxnorm_specific(records: list[dict[str, Any]]) -> dict[str, Any]:
    """RxNorm-only diagnostics: TTY, ingredient/strength/form and tie-breaking."""
    answerable = [r for r in records if not r["no_valid_candidate"] and r["selected_codes"]]
    tty_correct = tty_total = 0
    confusion: Counter[str] = Counter()
    field_hits = {"ingredient": [0, 0], "strength": [0, 0], "dose_form": [0, 0]}
    for record in answerable:
        ranked = ranked_codes(record)
        gold = set(record["selected_codes"])
        structures = {c["code"]: c["structure"] for c in record["offered_candidates"]}
        if not ranked:
            continue
        top = ranked[0]
        gold_ttys = {structures.get(g, {}).get("tty") for g in gold if g in structures}
        top_tty = structures.get(top, {}).get("tty")
        tty_total += 1
        if top_tty in gold_ttys:
            tty_correct += 1
        else:
            confusion[f"{sorted(str(t) for t in gold_ttys)}->{top_tty}"] += 1
        for field in field_hits:
            gold_values = {structures.get(g, {}).get(field) for g in gold if g in structures}
            field_hits[field][1] += 1
            if structures.get(top, {}).get(field) in gold_values:
                field_hits[field][0] += 1
    return {
        "tty_accuracy": tty_correct / tty_total if tty_total else 0.0,
        "tty_confusions": dict(confusion.most_common(10)),
        "field_accuracy": {
            name: (hits / total if total else 0.0) for name, (hits, total) in field_hits.items()
        },
        "saturated_set_metrics": subset_metrics(
            [r for r in records if len(r.get("offered_candidates") or []) >= 10]
        ),
        "tie_metrics": subset_metrics(
            [r for r in records if "rx_exact_ranking_tie" in (r["provenance"].get("facets") or [])]
        ),
    }


def decide(icd: dict[str, Any], rx: dict[str, Any], ready: bool) -> tuple[str, str]:
    """§9 decision rules applied to measured evidence."""
    if not ready:
        return DECISION_MORE_REVIEW, (
            "reviewed counts are below the minimum; no retrieval or ranking claim is "
            "supportable yet"
        )
    icd_r10 = icd.get("recall_at_10", 0.0)
    rx_r10 = rx.get("recall_at_10", 0.0)
    icd_top1 = icd.get("top1_exact_accuracy", 0.0)
    rx_top1 = rx.get("top1_exact_accuracy", 0.0)
    if icd_r10 < RECALL10_LOW:
        return DECISION_ICD_RETRIEVAL, f"ICD Recall@10 {icd_r10:.3f} < {RECALL10_LOW}"
    if (
        icd_r10 >= RECALL10_LOW
        and rx_r10 >= RECALL10_LOW
        and icd_top1 < TOP1_LOW
        and rx_top1 < TOP1_LOW
    ):
        return DECISION_SHARED_RERANK, "both ontologies retrieve well and rank poorly"
    if rx_r10 >= RECALL10_LOW and rx_top1 < TOP1_LOW:
        return DECISION_RX_RERANK, f"RxNorm Recall@10 {rx_r10:.3f}, top-1 {rx_top1:.3f}"
    if icd_top1 < TOP1_LOW:
        return DECISION_ICD_RERANK, f"ICD Recall@10 {icd_r10:.3f}, top-1 {icd_top1:.3f}"
    return DECISION_MORE_REVIEW, "no rule fired decisively"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument(
        "--output", type=Path, default=REPO / "runs" / "diagnostics" / "0065_benchmark"
    )
    args = parser.parse_args(argv)

    if not args.pack.is_file():
        print(f"pack not found: {args.pack}", file=sys.stderr)
        return 2
    records = load_records(args.pack)
    gold = [r for r in records if is_human_gold(r)]
    by_ontology = {
        name: [r for r in gold if r["ontology"] == name]
        for name in (ONTOLOGY_ICD10, ONTOLOGY_RXNORM)
    }
    counts = {name: len(rows) for name, rows in by_ontology.items()}
    shortfalls = {
        name: MIN_REVIEWED[name] - counts[name]
        for name in MIN_REVIEWED
        if counts[name] < MIN_REVIEWED[name]
    }
    ready = not shortfalls

    print("=== reviewed human gold ===")
    for name in (ONTOLOGY_ICD10, ONTOLOGY_RXNORM):
        print(f"  {name:<8} {counts[name]:>4} reviewed / {MIN_REVIEWED[name]} required")
    print(f"  total pack records: {len(records)}  (human gold: {len(gold)})")

    payload: dict[str, Any] = {
        "pack": str(args.pack),
        "pack_records": len(records),
        "human_gold_records": len(gold),
        "reviewed_by_ontology": counts,
        "minimum_required": MIN_REVIEWED,
        "reportable": ready,
        "shortfall": shortfalls,
    }

    if not ready:
        print("\n" + "=" * 74)
        print("REFUSING TO REPORT BENCHMARK METRICS.")
        for name, missing in sorted(shortfalls.items()):
            print(f"  {name}: {missing} more reviewed record(s) required")
        print(
            "\nThis refusal is deliberate. Audit 0063 could not report Recall@k because no\n"
            "gold codes existed; reporting one now from too few annotations would be worse,\n"
            "because it would look like evidence. Run the adjudication CLI and come back."
        )
        print("=" * 74)
        decision, reason = decide({}, {}, ready=False)
        payload["decision"] = {"next_intervention": decision, "reason": reason}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "linker_benchmark.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nnext intervention: {decision}")
        print(f"status written to {args.output / 'linker_benchmark.json'}")
        return 0

    icd = subset_metrics(by_ontology[ONTOLOGY_ICD10])
    rx = subset_metrics(by_ontology[ONTOLOGY_RXNORM])
    payload["icd10"] = {
        **icd,
        "by_stratum": stratify(by_ontology[ONTOLOGY_ICD10], lambda r: r["provenance"]["stratum"]),
        "by_confidence": stratify(by_ontology[ONTOLOGY_ICD10], lambda r: r["reviewer_confidence"]),
        "by_length": stratify(
            by_ontology[ONTOLOGY_ICD10],
            lambda r: "short" if r["provenance"].get("surface_words", 0) <= 3 else "long",
        ),
        "icd_specific": icd_specific(by_ontology[ONTOLOGY_ICD10]),
    }
    payload["rxnorm"] = {
        **rx,
        "by_stratum": stratify(by_ontology[ONTOLOGY_RXNORM], lambda r: r["provenance"]["stratum"]),
        "by_confidence": stratify(by_ontology[ONTOLOGY_RXNORM], lambda r: r["reviewer_confidence"]),
        "rxnorm_specific": rxnorm_specific(by_ontology[ONTOLOGY_RXNORM]),
    }
    decision, reason = decide(icd, rx, ready=True)
    payload["decision"] = {"next_intervention": decision, "reason": reason}

    for name, block in (("ICD-10", icd), ("RxNorm", rx)):
        print(f"\n=== {name} ===")
        print(
            f"  answerable {block['answerable']}  no-valid-candidate {block['no_valid_candidate']}"
        )
        print(f"  KB coverage {block['kb_coverage']:.4f}")
        for k in RECALL_KS:
            print(f"  Recall@{k:<3} {block[f'recall_at_{k}']:.4f}")
        print(f"  MRR         {block['mrr']:.4f}")
        print(f"  top-1       {block['top1_exact_accuracy']:.4f}")
    print(f"\nnext intervention: {decision}  ({reason})")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "linker_benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"written to {args.output / 'linker_benchmark.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
