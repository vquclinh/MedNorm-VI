#!/usr/bin/env python3
"""Local CLI for human adjudication of linker-gold records (Audit 0065 §6).

One record at a time, one decision at a time, saved atomically after each. Nothing is
transmitted anywhere and the whole corpus is never printed - the reviewer sees the mention
under review and nothing else.

Two anti-bias properties are worth naming because they are easy to lose:

* **Candidates are displayed in a deterministically shuffled order.** The production
  linker's rank is kept in the record but is NOT shown while deciding. A reviewer shown
  "rank 1" tends to ratify it, and the benchmark would then measure agreement with the
  system it exists to judge. `--reveal-rank` exists for post-hoc inspection only.
* **"No valid candidate" is a first-class answer.** Audit 0063 measured a 5.4% empty-set
  rate for ICD on gold mentions; if the tool forced a choice, that population would be
  silently converted into wrong gold.

Commands inside the review loop::

    1 3 5   select those displayed candidates      n   no valid candidate
    u       undo the previous saved decision       s   skip (leave UNLABELLED)
    ?       uncertain (HUMAN_UNCERTAIN)            q   save and quit
    c<h|m|l> set confidence                        !<text>  add a note

Usage::

    python scripts/adjudicate_linker_gold.py --reviewer-id <you> --ontology ICD10 --resume
    python scripts/adjudicate_linker_gold.py --status
    python scripts/adjudicate_linker_gold.py --export-summary
    python scripts/adjudicate_linker_gold.py --reviewer-id <you> --second-review
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.annotation.linker_gold import (  # noqa: E402
    ONTOLOGY_ICD10,
    ONTOLOGY_RXNORM,
    STATUS_HUMAN_GOLD,
    STATUS_HUMAN_UNCERTAIN,
    STATUS_SECOND_REVIEW_AGREED,
    STATUS_SECOND_REVIEW_DISAGREED,
    STATUS_UNLABELLED,
    is_human_gold,
    load_records,
    utc_now,
    validate_record,
)

DEFAULT_PACK = REPO / "data" / "private_linker_gold" / "0065" / "pack.jsonl"


def write_atomically(path: Path, records: list[dict[str, Any]]) -> None:
    """Write to a sibling temp file then rename, so an interrupt cannot truncate the pack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), prefix=".adjudicate-", delete=False
    )
    try:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def bar(width: int = 78) -> str:
    return "─" * width


def render_candidate(index: int, candidate: dict[str, Any], *, reveal_rank: bool) -> str:
    structure = candidate.get("structure", {})
    lines = [f"  [{index}] {candidate.get('canonical_name', '') or '(no canonical name)'}"]
    if structure.get("ontology") == ONTOLOGY_ICD10:
        lines.append(
            f"      code {candidate['code']}  (display {candidate.get('display_code', '-')})"
            f"  class={structure.get('specificity_class')}"
            f"  depth={structure.get('hierarchy_depth')}"
        )
        if structure.get("parent_code"):
            lines.append(f"      parent {structure['parent_code']}")
        if structure.get("child_codes"):
            lines.append(f"      children {', '.join(structure['child_codes'][:6])}")
    else:
        lines.append(
            f"      RxCUI {candidate['code']}  TTY={structure.get('tty')}"
            f"  ingredient={structure.get('ingredient')}"
            f"  strength={structure.get('strength')}  form={structure.get('dose_form')}"
        )
        if structure.get("brand"):
            lines.append(f"      brand {structure['brand']}")
        if structure.get("graph_relations"):
            lines.append(f"      graph {', '.join(structure['graph_relations'][:3])}")
        if structure.get("structured_match"):
            lines.append(f"      matched fields {', '.join(structure['structured_match'][:6])}")
    aliases = candidate.get("aliases") or []
    if aliases:
        lines.append(f"      aliases {', '.join(aliases[:6])}")
    channels = ", ".join(candidate.get("retrieval_channels") or []) or "none"
    rank = f"  linker_rank={candidate.get('linker_rank')}" if reveal_rank else ""
    lines.append(f"      channels {channels}  score {candidate.get('score', 0.0):.3f}{rank}")
    return "\n".join(lines)


def render_record(record: dict[str, Any], position: int, total: int, *, reveal_rank: bool) -> str:
    lines = [
        bar(),
        f"record {position}/{total}   {record['annotation_id']}   "
        f"{record['entity_type']} -> {record['ontology']}",
        f"source {record['source_dataset']}   stratum {record['provenance'].get('stratum')}",
        bar(),
        "context:",
        f"  …{record['context_before']}",
        f"  >>> {record['mention_text']} <<<",
        f"  {record['context_after']}…",
        bar(),
    ]
    candidates = record.get("offered_candidates") or []
    if not candidates:
        lines.append(
            "  NO CANDIDATES WERE OFFERED for this mention. This is itself a finding:\n"
            "  answer 'n' (no valid candidate) unless you know a code the linker missed."
        )
    else:
        lines.append(f"candidates ({len(candidates)}, displayed in randomized order):")
        for index, candidate in enumerate(candidates, start=1):
            lines.append(render_candidate(index, candidate, reveal_rank=reveal_rank))
    lines.append(bar())
    return "\n".join(lines)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(r["adjudication_status"] for r in records)
    by_ontology = Counter(r["ontology"] for r in records)
    reviewed = Counter(r["ontology"] for r in records if is_human_gold(r))
    no_valid = sum(1 for r in records if is_human_gold(r) and r["no_valid_candidate"])
    multi = sum(1 for r in records if is_human_gold(r) and len(r["selected_codes"]) > 1)
    return {
        "records": len(records),
        "by_status": dict(sorted(status.items())),
        "by_ontology": dict(sorted(by_ontology.items())),
        "human_gold_by_ontology": dict(sorted(reviewed.items())),
        "human_gold_total": sum(reviewed.values()),
        "no_valid_candidate": no_valid,
        "multi_code_selections": multi,
        "confidence": dict(
            sorted(
                Counter(r["reviewer_confidence"] for r in records if is_human_gold(r)).items(),
                key=lambda kv: str(kv[0]),
            )
        ),
    }


def print_status(records: list[dict[str, Any]]) -> None:
    summary = summarize(records)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    icd = summary["human_gold_by_ontology"].get(ONTOLOGY_ICD10, 0)
    rx = summary["human_gold_by_ontology"].get(ONTOLOGY_RXNORM, 0)
    print(
        f"\nbenchmark minimums (Audit 0065 §8): ICD {icd}/50, RxNorm {rx}/30 -> "
        f"{'READY' if icd >= 50 and rx >= 30 else 'NOT YET REPORTABLE'}"
    )


def review_loop(
    records: list[dict[str, Any]],
    pack: Path,
    *,
    reviewer: str,
    ontology: str | None,
    resume: bool,
    reveal_rank: bool,
    second_review: bool,
) -> int:
    queue = [
        index
        for index, record in enumerate(records)
        if (ontology is None or record["ontology"] == ontology)
        and (
            is_human_gold(record)
            if second_review
            else (not resume or record["adjudication_status"] == STATUS_UNLABELLED)
        )
    ]
    if not queue:
        print("nothing to review with these filters.")
        return 0

    print(
        f"{len(queue)} record(s) queued. Commands: '1 3' select, 'n' none valid, 's' skip, "
        f"'?' uncertain, 'u' undo, 'ch|cm|cl' confidence, '!note', 'q' quit."
    )
    confidence = "medium"
    history: list[tuple[int, dict[str, Any]]] = []
    position = 0
    while position < len(queue):
        index = queue[position]
        record = records[index]
        print("\n" + render_record(record, position + 1, len(queue), reveal_rank=reveal_rank))
        if second_review:
            print(
                f"  first review selected: {record['selected_codes']} "
                f"(no_valid={record['no_valid_candidate']})"
            )
        try:
            answer = input(f"[{confidence}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ninterrupted; progress up to the last saved record is intact.")
            return 0

        if answer == "q":
            print("saved and exiting.")
            return 0
        if answer == "s":
            position += 1
            continue
        if answer == "u":
            if not history:
                print("nothing to undo.")
                continue
            previous_index, snapshot = history.pop()
            records[previous_index] = snapshot
            write_atomically(pack, records)
            position = max(0, position - 1)
            print(f"undid {snapshot['annotation_id']}; it is queued again.")
            continue
        if answer.startswith("c") and answer[1:] in ("h", "m", "l"):
            confidence = {"h": "high", "m": "medium", "l": "low"}[answer[1:]]
            print(f"confidence set to {confidence}")
            continue
        if answer.startswith("!"):
            record["adjudication_notes"] = (record["adjudication_notes"] + " " + answer[1:]).strip()
            write_atomically(pack, records)
            print("note saved.")
            continue

        snapshot = json.loads(json.dumps(record))
        candidates = record.get("offered_candidates") or []
        if answer == "n":
            selected: list[str] = []
            no_valid = True
            status = STATUS_HUMAN_GOLD
        elif answer == "?":
            selected, no_valid = [], False
            status = STATUS_HUMAN_UNCERTAIN
        else:
            try:
                picks = [int(token) for token in answer.split()]
            except ValueError:
                print("unrecognised input; nothing recorded.")
                continue
            if not picks or any(p < 1 or p > len(candidates) for p in picks):
                print(f"pick numbers between 1 and {len(candidates)}; nothing recorded.")
                continue
            selected = [candidates[p - 1]["code"] for p in picks]
            no_valid = False
            status = STATUS_HUMAN_GOLD

        if second_review:
            agrees = (
                sorted(selected) == sorted(record["selected_codes"])
                and no_valid == record["no_valid_candidate"]
            )
            record["second_review"] = {
                "reviewer_id": reviewer,
                "selected_codes": selected,
                "no_valid_candidate": no_valid,
                "agrees_with_first": agrees,
                "notes": "",
                "reviewed_at": utc_now(),
            }
            record["adjudication_status"] = (
                STATUS_SECOND_REVIEW_AGREED if agrees else STATUS_SECOND_REVIEW_DISAGREED
            )
        else:
            record["selected_codes"] = selected
            record["rejected_codes"] = (
                [c["code"] for c in candidates if c["code"] not in selected]
                if (selected or no_valid)
                else []
            )
            record["no_valid_candidate"] = no_valid
            record["adjudication_status"] = status
            record["reviewer_id"] = reviewer
            record["reviewer_confidence"] = confidence
        record["updated_at"] = utc_now()

        outcome = validate_record(record)
        if not outcome.valid:
            records[index] = snapshot
            print(f"REFUSED: {outcome.problems[:3]}\n  nothing was saved for this record.")
            continue

        history.append((index, snapshot))
        write_atomically(pack, records)
        position += 1
    print("\nqueue complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--reviewer-id", default="")
    parser.add_argument("--ontology", choices=[ONTOLOGY_ICD10, ONTOLOGY_RXNORM], default=None)
    parser.add_argument("--resume", action="store_true", help="skip records already decided")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--export-summary", type=Path, nargs="?", const=Path("-"), default=None)
    parser.add_argument("--second-review", action="store_true")
    parser.add_argument(
        "--reveal-rank",
        action="store_true",
        help="show the linker's rank; for post-hoc inspection, NOT for deciding",
    )
    args = parser.parse_args(argv)

    if not args.pack.is_file():
        print(f"pack not found: {args.pack}", file=sys.stderr)
        return 2
    records = load_records(args.pack)

    if args.status:
        print_status(records)
        return 0
    if args.export_summary is not None:
        summary = summarize(records)
        payload = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
        if str(args.export_summary) == "-":
            print(payload)
        else:
            args.export_summary.parent.mkdir(parents=True, exist_ok=True)
            args.export_summary.write_text(payload + "\n", encoding="utf-8")
            print(f"summary written to {args.export_summary}")
        return 0

    if not args.reviewer_id:
        print("--reviewer-id is required to record a decision", file=sys.stderr)
        return 2
    return review_loop(
        records,
        args.pack,
        reviewer=args.reviewer_id,
        ontology=args.ontology,
        resume=args.resume,
        reveal_rank=args.reveal_rank,
        second_review=args.second_review,
    )


if __name__ == "__main__":
    sys.exit(main())
