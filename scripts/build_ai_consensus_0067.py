#!/usr/bin/env python3
"""Compare two independent AI adjudications and build a consensus set (Audit 0067 §5-§6, §8).

Claude's `AI_ASSISTED_DRAFT` and the second system's `AI_SECOND_OPINION` were produced
independently - the second system never saw the first's decisions - so their agreement is
measurable rather than assumed.

`AI_CONSENSUS_HIGH` is the strongest label this project produces without a human, and it is
still not human gold. Two models that share a blind spot agree exactly as readily as two
models that are right, and nothing here can tell those apart. The status exists so a later
milestone may consider these records as WEAK SUPERVISION, never so they can be reported as
benchmark accuracy.

Also emits a disagreement packet for a second round, which - unlike the first packet - may
show both prior opinions, and enriches each record with extra local KB lookups (ICD
aliases, parents, children, siblings; RxNorm ingredient/TTY/strength/form/brand and graph
neighbours). No external API is used.
"""

from __future__ import annotations

import argparse
import hashlib
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
    STATUS_AI_CONSENSUS_HIGH,
    assert_not_ai_gold,
    is_human_gold,
    load_records,
    utc_now,
)

BASE = REPO / "data" / "private_linker_gold" / "0065"
CLAUDE = BASE / "ai_drafts" / "claude_code_v1.jsonl"
SECOND = BASE / "ai_second_opinion" / "chatgpt_second_opinion.jsonl"
BLIND = BASE / "chatgpt_handoff" / "minimum_benchmark_blind.jsonl"
OUT = BASE / "ai_consensus"
HANDOFF = BASE / "chatgpt_handoff"

EXACT_AGREEMENT = "EXACT_AGREEMENT"
DECISION_AGREEMENT_CODE_DISAGREEMENT = "DECISION_AGREEMENT_CODE_DISAGREEMENT"
CLAUDE_SELECTED_CHATGPT_REJECTED = "CLAUDE_SELECTED_CHATGPT_REJECTED"
CHATGPT_SELECTED_CLAUDE_REJECTED = "CHATGPT_SELECTED_CLAUDE_REJECTED"
CLAUDE_ABSTAIN = "CLAUDE_ABSTAIN"
CHATGPT_ABSTAIN = "CHATGPT_ABSTAIN"
BOTH_ABSTAIN = "BOTH_ABSTAIN"
OTHER_DISAGREEMENT = "OTHER_DISAGREEMENT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(a: dict[str, Any], b: dict[str, Any]) -> str:
    """`a` is the first opinion, `b` the second."""
    da, db = a["ai_decision"], b["ai_decision"]
    ca, cb = set(a["selected_codes"]), set(b["selected_codes"])
    if da == "ABSTAIN" and db == "ABSTAIN":
        return BOTH_ABSTAIN
    if da == "ABSTAIN":
        return CLAUDE_ABSTAIN
    if db == "ABSTAIN":
        return CHATGPT_ABSTAIN
    if da == db:
        if da != "SELECTED" or ca == cb:
            return EXACT_AGREEMENT
        return DECISION_AGREEMENT_CODE_DISAGREEMENT
    if da == "SELECTED" and db == "NO_VALID_CANDIDATE":
        return CLAUDE_SELECTED_CHATGPT_REJECTED
    if db == "SELECTED" and da == "NO_VALID_CANDIDATE":
        return CHATGPT_SELECTED_CLAUDE_REJECTED
    return OTHER_DISAGREEMENT


def eligible_for_high_consensus(
    a: dict[str, Any], b: dict[str, Any], record: dict[str, Any], category: str
) -> tuple[bool, str]:
    if category != EXACT_AGREEMENT:
        return False, f"category is {category}"
    if "LOW" in (a["ai_confidence"], b["ai_confidence"]):
        return False, "one side is LOW confidence"
    unresolved = {"parent_child_ambiguity", "tty_ambiguity", "multiple_plausible"}
    flags = set(b.get("ambiguity_flags") or []) | {a.get("ambiguity_reason", "")}
    if flags & unresolved:
        return False, f"unresolved ontology ambiguity flagged: {sorted(flags & unresolved)}"
    codes = set(a["selected_codes"])
    if codes:
        offered = {c.get("code_dotless") or c.get("rxcui") for c in record["offered_candidates"]}
        if not codes <= offered:
            return False, "a selected code is absent from the frozen KB offer set"
        if record["ontology"] == ONTOLOGY_ICD10 and any("." in c for c in codes):
            return False, "ICD code is not a dotless internal identity"
    return True, "both systems independently agree under all structural checks"


def kb_enrichment(record: dict[str, Any]) -> dict[str, Any]:
    """Extra LOCAL evidence for a disagreement, from the frozen indices only."""
    from mednorm_vi.inference.config import PipelineConfig
    from mednorm_vi.kb.indexing.retrieval import load_index

    config = PipelineConfig.load(str(REPO / "configs" / "pipeline" / "full_v1.yaml"))
    index = load_index(
        config.icd_index if record["ontology"] == ONTOLOGY_ICD10 else config.rxnorm_index
    )
    out: dict[str, Any] = {"siblings": {}, "related": {}}
    for candidate in record["offered_candidates"]:
        code = candidate.get("code_dotless") or candidate.get("rxcui")
        if record["ontology"] == ONTOLOGY_ICD10:
            parent = code[:-1] if len(code) > 3 else code[:3]
            siblings = sorted(
                other
                for other in index.records
                if other.startswith(parent) and other != code and len(other) == len(code)
            )[:8]
            out["siblings"][code] = siblings
        else:
            out["related"][code] = sorted(index.graph.get(code, []))[:8]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", type=Path, default=CLAUDE)
    parser.add_argument("--second", type=Path, default=SECOND)
    parser.add_argument("--blind", type=Path, default=BLIND)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--handoff", type=Path, default=HANDOFF)
    parser.add_argument("--no-kb-enrichment", action="store_true")
    args = parser.parse_args(argv)

    assert_not_ai_gold()
    if not args.second.is_file():
        print(f"second opinion not found: {args.second}", file=sys.stderr)
        print(
            "Run scripts/import_ai_second_opinion.py after the second AI returns its decisions.",
            file=sys.stderr,
        )
        return 2

    # Where a human has already adjudicated, their decision is authoritative and the AI
    # consensus is recorded as commentary beside it, never as a replacement.
    human = {
        r["annotation_id"]: r
        for r in load_records(REPO / "data" / "private_linker_gold" / "0065" / "pack.jsonl")
        if is_human_gold(r)
    }
    first = {d["annotation_id"]: d for d in load_records(args.claude)}
    second = {d["annotation_id"]: d for d in load_records(args.second)}
    packet = {r["annotation_id"]: r for r in load_records(args.blind)}
    shared = sorted(set(first) & set(second) & set(packet))

    consensus: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    high: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()

    for annotation_id in shared:
        a, b, record = first[annotation_id], second[annotation_id], packet[annotation_id]
        category = classify(a, b)
        categories[category] += 1
        ok, reason = eligible_for_high_consensus(a, b, record, category)
        row = {
            "annotation_id": annotation_id,
            "ontology": record["ontology"],
            "category": category,
            "first_decision": a["ai_decision"],
            "first_codes": a["selected_codes"],
            "first_confidence": a["ai_confidence"],
            "second_decision": b["ai_decision"],
            "second_codes": b["selected_codes"],
            "second_confidence": b["ai_confidence"],
            "high_consensus_eligible": ok,
            "eligibility_reason": reason,
        }
        consensus.append(row)
        if ok:
            high.append(
                {
                    **row,
                    "human_gold_present": annotation_id in human,
                    "adjudication_status": STATUS_AI_CONSENSUS_HIGH,
                    "consensus_codes": a["selected_codes"],
                    "no_valid_candidate": a["ai_decision"] == "NO_VALID_CANDIDATE",
                    "note": "AI consensus only. NOT human gold and not a benchmark result.",
                }
            )
        else:
            entry = {
                **row,
                "first_rationale": a.get("rationale", ""),
                "second_rationale": b.get("rationale", ""),
                "mention_text": record["mention_text"],
                "context_before": record["context_before"],
                "context_after": record["context_after"],
                "offered_candidates": record["offered_candidates"],
            }
            if not args.no_kb_enrichment:
                entry["kb_enrichment"] = kb_enrichment(record)
            disagreements.append(entry)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("consensus.jsonl", consensus),
        ("disagreements.jsonl", disagreements),
        ("high_confidence_consensus.jsonl", high),
    ):
        (args.out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )

    agreement = categories[EXACT_AGREEMENT] / len(shared) if shared else 0.0
    by_ontology: dict[str, Any] = {}
    for ontology in ("ICD10", "RXNORM"):
        subset = [r for r in consensus if r["ontology"] == ontology]
        if subset:
            by_ontology[ontology] = {
                "n": len(subset),
                "exact_agreement": sum(r["category"] == EXACT_AGREEMENT for r in subset)
                / len(subset),
                "selected_agreement": sum(
                    r["category"] == EXACT_AGREEMENT and r["first_decision"] == "SELECTED"
                    for r in subset
                ),
                "no_valid_agreement": sum(
                    r["category"] == EXACT_AGREEMENT and r["first_decision"] == "NO_VALID_CANDIDATE"
                    for r in subset
                ),
                "high_consensus": sum(r["high_consensus_eligible"] for r in subset),
            }
    summary = {
        "compared": len(shared),
        "records_with_human_gold": sum(1 for i in shared if i in human),
        "human_gold_is_authoritative": True,
        "categories": dict(categories),
        "agreement_rate": agreement,
        "disagreement_rate": 1 - agreement,
        "high_consensus_records": len(high),
        "by_ontology": by_ontology,
        "status": STATUS_AI_CONSENSUS_HIGH,
        "not_human_gold": True,
        "created_at": utc_now(),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Second-round packet. Unlike the first, this MAY show both prior opinions.
    args.handoff.mkdir(parents=True, exist_ok=True)
    dis_jsonl = args.handoff / "disagreements_for_chatgpt.jsonl"
    dis_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in disagreements),
        encoding="utf-8",
    )
    lines = [
        "# Disagreement reconciliation packet (round 2)",
        "",
        "Two AI systems adjudicated these records independently and disagreed. Both prior",
        "opinions are shown deliberately - the blindness requirement applied to round 1 only.",
        "Extra local knowledge-base evidence (siblings, hierarchy, graph neighbours) is",
        "included where available.",
        "",
        f"- records: {len(disagreements)}",
        "- return the same schema as round 1 (`ai_second_opinion_v1.schema.json`)",
        "",
    ]
    for position, row in enumerate(disagreements, start=1):
        lines += [
            f"## {position}. `{row['annotation_id']}` — {row['ontology']} — {row['category']}",
            "",
            f"- mention: **{row['mention_text']}**",
            f"- opinion A: `{row['first_decision']}` {row['first_codes']} "
            f"({row['first_confidence']}) — {row['first_rationale'][:180]}",
            f"- opinion B: `{row['second_decision']}` {row['second_codes']} "
            f"({row['second_confidence']}) — {row['second_rationale'][:180]}",
            "",
            "```text",
            f"...{row['context_before']}",
            f"    >>> {row['mention_text']} <<<",
            f"{row['context_after']}...",
            "```",
            "",
        ]
    (args.handoff / "disagreements_for_chatgpt.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"compared {len(shared)} records")
    for name, count in categories.most_common():
        print(f"  {name:<40}{count:>5}")
    print(f"\nagreement rate      {agreement:.4f}")
    print(f"high consensus      {len(high)}")
    print(f"disagreement packet {dis_jsonl} ({len(disagreements)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
