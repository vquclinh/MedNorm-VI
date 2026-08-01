#!/usr/bin/env python3
"""Prioritised human-review queues over the AI drafts (Audit 0066 §8).

The owner's first target is 50 reviewed ICD + 30 reviewed RxNorm. These queues decide
WHICH 80 to review first, so the earliest human effort lands where it is worth most.

`queue_minimum_benchmark` is stratified across failure classes rather than taken in id
order: reviewing 50 easy exact-alias records would satisfy the count and teach nothing.
The other queues target the records where the AI is least trustworthy or most informative
- disagreement with the production linker, abstention, low confidence - plus a
deterministic audit sample of HIGH-confidence drafts, because an unaudited high-confidence
claim is exactly the one that would quietly become a weak training label.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.annotation.linker_gold import (  # noqa: E402
    ONTOLOGY_ICD10,
    ONTOLOGY_RXNORM,
    load_records,
)

DRAFTS = REPO / "data" / "private_linker_gold" / "0065" / "ai_drafts" / "claude_code_v1.jsonl"
OUT = REPO / "data" / "private_linker_gold" / "0065" / "ai_drafts" / "queues"
TARGET = {ONTOLOGY_ICD10: 50, ONTOLOGY_RXNORM: 30}


def stratified(drafts: list[dict], ontology: str, target: int, rng: random.Random) -> list[str]:
    """Round-robin across ambiguity flags so every failure class is represented."""
    pools: dict[str, list[dict]] = defaultdict(list)
    for draft in drafts:
        if draft["ontology"] == ontology:
            pools[draft["ambiguity_reason"]].append(draft)
    for pool in pools.values():
        pool.sort(key=lambda d: d["annotation_id"])
        rng.shuffle(pool)
    chosen: list[str] = []
    names = sorted(pools)
    position = 0
    while len(chosen) < target and any(pools[n] for n in names):
        name = names[position % len(names)]
        position += 1
        if pools[name]:
            chosen.append(pools[name].pop()["annotation_id"])
    return sorted(chosen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, default=DRAFTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args(argv)

    drafts = load_records(args.drafts)
    rng = random.Random(args.seed)

    minimum: list[str] = []
    for ontology, target in TARGET.items():
        minimum.extend(stratified(drafts, ontology, target, rng))

    queues = {
        "queue_minimum_benchmark": sorted(minimum),
        "queue_ai_disagreement": sorted(
            d["annotation_id"]
            for d in drafts
            if d["linker_top1"] and not d["agrees_with_linker_top1"]
        ),
        "queue_ai_abstain": sorted(
            d["annotation_id"] for d in drafts if d["ai_decision"] == "ABSTAIN"
        ),
        "queue_ai_low_confidence": sorted(
            d["annotation_id"] for d in drafts if d["ai_confidence"] == "LOW"
        ),
        "queue_high_confidence_audit": sorted(
            rng.sample(
                sorted(d["annotation_id"] for d in drafts if d["ai_confidence"] == "HIGH"),
                min(20, sum(1 for d in drafts if d["ai_confidence"] == "HIGH")),
            )
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, ids in queues.items():
        (args.out / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    by_ontology = {d["annotation_id"]: d["ontology"] for d in drafts}
    summary = {
        name: {
            "size": len(ids),
            "by_ontology": dict(Counter(by_ontology[i] for i in ids)),
        }
        for name, ids in queues.items()
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    for name, block in sorted(summary.items()):
        print(f"  {name:<32}{block['size']:>4}  {block['by_ontology']}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
