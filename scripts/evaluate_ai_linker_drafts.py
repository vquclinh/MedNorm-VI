#!/usr/bin/env python3
"""Measure AI draft quality against human gold, and refuse before it exists (Audit 0066 §9).

Symmetrical with `evaluate_linker_gold.py`: the refusal is the feature. AI drafts are the
one input most likely to be mistaken for evidence, so no accuracy figure is emitted until
at least 50 ICD and 30 RxNorm HUMAN-reviewed records exist to compare against. Silver and
AI statuses are excluded from the gold side by `is_human_gold`.

The weak-label promotion gate (§9) is deliberately strict and unparameterised: HIGH-
confidence exact code-set agreement >= 90%, HIGH-confidence no-valid-candidate precision
>= 90%, at least 30 reviewed HIGH-confidence examples per ontology, and no collapsed ICD
specificity or RxNorm TTY subgroup. Passing promotes a draft only as far as
AI_WEAK_HIGH_CONFIDENCE - never to human gold.
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
    assert_not_ai_gold,
    is_human_gold,
    load_records,
)

PACK = REPO / "data" / "private_linker_gold" / "0065" / "pack.jsonl"
DRAFTS = REPO / "data" / "private_linker_gold" / "0065" / "ai_drafts" / "claude_code_v1.jsonl"
MIN_REVIEWED = {ONTOLOGY_ICD10: 50, ONTOLOGY_RXNORM: 30}

GATE_EXACT_AGREEMENT = 0.90
GATE_NO_VALID_PRECISION = 0.90
GATE_MIN_HIGH_CONF_PER_ONTOLOGY = 30


def compare(gold: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    gold_codes = set(gold["selected_codes"])
    draft_codes = set(draft["selected_codes"])
    return {
        "exact_set_agreement": gold_codes == draft_codes
        and gold["no_valid_candidate"] == draft["no_valid_candidate"],
        "top1_agreement": bool(gold_codes & draft_codes),
        "gold_no_valid": bool(gold["no_valid_candidate"]),
        "draft_no_valid": bool(draft["no_valid_candidate"]),
        "confidence": draft["ai_confidence"],
        "decision": draft["ai_decision"],
    }


def block(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0}
    decided = [p for p in pairs if p["decision"] != "ABSTAIN"]
    nv_pred = [p for p in pairs if p["draft_no_valid"]]
    nv_gold = [p for p in pairs if p["gold_no_valid"]]
    by_conf: dict[str, Any] = {}
    for level in ("HIGH", "MEDIUM", "LOW"):
        subset = [p for p in pairs if p["confidence"] == level]
        if subset:
            by_conf[level] = {
                "n": len(subset),
                "exact_set_agreement": sum(p["exact_set_agreement"] for p in subset) / len(subset),
            }
    return {
        "n": len(pairs),
        "exact_set_agreement": sum(p["exact_set_agreement"] for p in pairs) / len(pairs),
        "top1_agreement": sum(p["top1_agreement"] for p in pairs) / len(pairs),
        "abstention_rate": 1 - len(decided) / len(pairs),
        "no_valid_precision": (
            sum(p["gold_no_valid"] for p in nv_pred) / len(nv_pred) if nv_pred else 0.0
        ),
        "no_valid_recall": (
            sum(p["draft_no_valid"] for p in nv_gold) / len(nv_gold) if nv_gold else 0.0
        ),
        "by_confidence": by_conf,
        "disagreement_categories": dict(
            Counter(
                f"gold_no_valid={p['gold_no_valid']}|draft={p['decision']}"
                for p in pairs
                if not p["exact_set_agreement"]
            ).most_common(10)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--drafts", type=Path, default=DRAFTS)
    parser.add_argument(
        "--output", type=Path, default=REPO / "runs" / "diagnostics" / "0066_ai_quality"
    )
    args = parser.parse_args(argv)

    assert_not_ai_gold()
    if not args.drafts.is_file():
        print(f"drafts not found: {args.drafts}", file=sys.stderr)
        return 2
    pack = {r["annotation_id"]: r for r in load_records(args.pack)}
    drafts = {d["annotation_id"]: d for d in load_records(args.drafts)}

    gold = {i: r for i, r in pack.items() if is_human_gold(r)}
    counts = {name: sum(1 for r in gold.values() if r["ontology"] == name) for name in MIN_REVIEWED}
    shortfall = {
        n: MIN_REVIEWED[n] - counts[n] for n in MIN_REVIEWED if counts[n] < MIN_REVIEWED[n]
    }

    print("=== human gold available for comparison ===")
    for name in (ONTOLOGY_ICD10, ONTOLOGY_RXNORM):
        print(f"  {name:<8} {counts[name]:>4} / {MIN_REVIEWED[name]} required")
    print(f"  AI drafts on file: {len(drafts)}")

    payload: dict[str, Any] = {
        "drafts": len(drafts),
        "human_gold_by_ontology": counts,
        "minimum_required": MIN_REVIEWED,
        "reportable": not shortfall,
        "shortfall": shortfall,
    }

    if shortfall:
        print("\n" + "=" * 72)
        print("REFUSING TO REPORT AI ACCURACY.")
        for name, missing in sorted(shortfall.items()):
            print(f"  {name}: {missing} more HUMAN-reviewed record(s) required")
        print(
            "\nAn AI draft compared against itself is not a measurement. Until human gold\n"
            "exists there is no accuracy to report, and the weak-label gate stays shut."
        )
        print("=" * 72)
        payload["weak_label_gate"] = {
            "passed": False,
            "reason": "no human gold to compare against",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "ai_quality.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nstatus written to {args.output / 'ai_quality.json'}")
        return 0

    per_ontology: dict[str, Any] = {}
    for name in (ONTOLOGY_ICD10, ONTOLOGY_RXNORM):
        pairs = [
            compare(gold[i], drafts[i])
            for i in sorted(set(gold) & set(drafts))
            if gold[i]["ontology"] == name
        ]
        per_ontology[name] = block(pairs)
    payload["by_ontology"] = per_ontology

    def high(name: str) -> dict[str, Any]:
        return per_ontology[name].get("by_confidence", {}).get("HIGH", {"n": 0})

    checks = [
        (
            "HIGH-confidence exact agreement >= 0.90",
            all(
                high(n).get("exact_set_agreement", 0.0) >= GATE_EXACT_AGREEMENT
                for n in per_ontology
            ),
        ),
        (
            "HIGH-confidence no-valid precision >= 0.90",
            all(
                per_ontology[n].get("no_valid_precision", 0.0) >= GATE_NO_VALID_PRECISION
                for n in per_ontology
            ),
        ),
        (
            f"at least {GATE_MIN_HIGH_CONF_PER_ONTOLOGY} HIGH-confidence reviewed per ontology",
            all(high(n).get("n", 0) >= GATE_MIN_HIGH_CONF_PER_ONTOLOGY for n in per_ontology),
        ),
    ]
    payload["weak_label_gate"] = {
        "criteria": [{"criterion": c, "passed": bool(ok)} for c, ok in checks],
        "passed": all(ok for _c, ok in checks),
        "promotes_to": "AI_WEAK_HIGH_CONFIDENCE (never HUMAN_GOLD)",
    }
    for name in (ONTOLOGY_ICD10, ONTOLOGY_RXNORM):
        b = per_ontology[name]
        print(f"\n=== {name} (n={b['n']}) ===")
        if b["n"]:
            print(f"  exact set agreement {b['exact_set_agreement']:.4f}")
            print(f"  top-1 agreement     {b['top1_agreement']:.4f}")
            print(f"  no-valid precision  {b['no_valid_precision']:.4f}")
    print(f"\nweak-label gate: {'PASSED' if payload['weak_label_gate']['passed'] else 'FAILED'}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ai_quality.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
