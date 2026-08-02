#!/usr/bin/env python3
"""Experiment 0079: safe-bridge filter over completed 0078 merges. CPU only, no model.

Rebuilds output **from the E3 seed** and reapplies only the 0078 merges whose inter-fragment
bridges pass a deterministic linguistic rule. Starting from the seed rather than unpicking
0078's output means a rejected merge restores the originals exactly, with no residue.

The only decision available is ACCEPT an existing 0078 merge or REVERT to E3. No merge is ever
invented, so type, assertion, single-proposal and boundary-expansion counts are structurally
zero.
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

from mednorm_vi.reasoner.safe_bridge import (  # noqa: E402
    SAFE_BRIDGE_VERSION,
    evaluate_merge,
)
from mednorm_vi.reasoner.validator import ASSERTION_TYPES, CANDIDATE_TYPES  # noqa: E402


def log(message: str) -> None:
    print(f"[0079] {message}", flush=True)


def load_note(path: Path) -> str:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("text", "content", "note"):
                if key in payload:
                    return str(payload[key])
        return str(payload)
    return path.read_text(encoding="utf-8")


def load_merges(run_dir: Path) -> list[dict[str, Any]]:
    """0078 merge decisions, from the full list when present, else the summary."""
    full = run_dir / "merges-full.json"
    if full.is_file():
        return json.loads(full.read_text(encoding="utf-8"))
    summary = run_dir / "diagnostic-summary.json"
    if not summary.is_file():
        raise SystemExit(f"no 0078 artifacts under {run_dir}")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    merges = payload.get("merges") or []
    if payload.get("merges_truncated"):
        raise SystemExit(
            f"{summary} holds a truncated merge list and {full} is absent; "
            "the complete decision artifact is required"
        )
    return merges


def finalize(entity: dict[str, Any]) -> dict[str, Any]:
    row = {"text": entity["text"], "type": entity["type"], "position": entity["position"]}
    if entity["type"] in ASSERTION_TYPES:
        row["assertions"] = list(entity.get("assertions") or [])
    if entity["type"] in CANDIDATE_TYPES:
        row["candidates"] = []
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--seed-entities", type=Path, required=True)
    parser.add_argument("--merge-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs/safe_bridge_0079")
    parser.add_argument("--expected-documents", type=int, default=100)
    args = parser.parse_args(argv)

    merges = load_merges(args.merge_run_dir)
    log(f"0078 selected merges: {len(merges)}")

    by_document: dict[str, list[dict[str, Any]]] = {}
    for merge in merges:
        by_document.setdefault(str(merge["document"]), []).append(merge)

    notes = sorted(p for p in args.input_dir.glob("*") if p.is_file())
    out_dir = args.run_dir / "output_raw_dotless"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    accepted_by_reason: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    accepted_list: list[dict[str, Any]] = []

    for path in notes:
        note = load_note(path)
        seed_path = args.seed_entities / f"{path.stem}.json"
        seed = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else []
        stats["entities_in_e3"] += len(seed)

        consumed: set[tuple[int, int]] = set()
        replacements: list[dict[str, Any]] = []
        for merge in by_document.get(path.stem, []):
            offsets = [list(map(int, o)) for o in merge["original_offsets"]]
            verdict = evaluate_merge(note, offsets)
            record = {
                "document": path.stem,
                "original_texts": merge["original_texts"],
                "original_offsets": offsets,
                "union_text": merge["union_text"],
                "union_offsets": merge["union_offsets"],
                "type": merge["type"],
                **verdict.as_dict(),
            }
            records.append(record)
            if verdict.accepted:
                accepted_by_reason[verdict.reason] += 1
                stats["accepted"] += 1
                accepted_list.append(record)
                start, end = int(merge["union_offsets"][0]), int(merge["union_offsets"][1])
                # Byte/offset-identical to the 0078 union.
                assert note[start:end] == merge["union_text"], record
                consumed.update(tuple(o) for o in offsets)
                replacements.append(
                    {
                        "text": merge["union_text"],
                        "type": merge["type"],
                        "position": [start, end],
                        "assertions": list(merge.get("assertions") or []),
                    }
                )
                stats["fragments_removed_0079"] += len(offsets) - 1
            else:
                rejected_by_reason[verdict.reason] += 1
                stats["rejected"] += 1

        # Rebuild FROM THE SEED: anything not consumed by an accepted merge is restored.
        entities = [
            finalize(e)
            for e in seed
            if (int(e["position"][0]), int(e["position"][1])) not in consumed
        ]
        entities.extend(finalize(r) for r in replacements)
        entities.sort(key=lambda e: (e["position"][0], e["position"][1], e["type"]))
        stats["entities_out_0079"] += len(entities)
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(entities, ensure_ascii=False), encoding="utf-8"
        )

    report = {
        "safe_bridge_version": SAFE_BRIDGE_VERSION,
        "documents": len(notes),
        "0078_selected_merges": len(merges),
        "0079_accepted_merges": stats["accepted"],
        "0079_rejected_merges": stats["rejected"],
        "entities_in_e3": stats["entities_in_e3"],
        "entities_out_0079": stats["entities_out_0079"],
        "fragments_removed_0079": stats["fragments_removed_0079"],
        "accepted_by_reason": dict(sorted(accepted_by_reason.items())),
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        # Structurally zero: 0079 can only ACCEPT an existing merge or REVERT to E3.
        "boundary_expansions": 0,
        "single_proposal_changes": 0,
        "type_changes": 0,
        "assertion_changes": 0,
        "candidate_policy": "forced_all_null",
        "merge_records": records,
        "model_used": None,
        "gpu_used": False,
        "contains_clinical_text": bool(records),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "diagnostic-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    log(
        f"E3 {stats['entities_in_e3']} -> out {stats['entities_out_0079']} "
        f"(0078 was 1085) | accepted {stats['accepted']} rejected {stats['rejected']} "
        f"| fragments removed {stats['fragments_removed_0079']}"
    )
    log(f"rejected_by_reason: {dict(rejected_by_reason)}")
    log("ACCEPTED MERGES:")
    for record in accepted_list:
        log(
            f"  [{record['document']}] {record['original_texts']} -> "
            f"'{record['union_text']}' {record['union_offsets']} "
            f"[{record['type']}] ({record['reason']})"
        )
    if len(sorted(out_dir.glob("*.json"))) != args.expected_documents:
        print("BLOCKED: document count mismatch", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
