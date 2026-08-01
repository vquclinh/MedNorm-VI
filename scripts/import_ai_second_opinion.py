#!/usr/bin/env python3
"""Validate and import a second AI's adjudications (Audit 0067 §4).

Every check here exists because the alternative is a consensus number built on a file
nobody verified. A returned decision is refused unless it names a record from the packet,
names it once, keeps the record's ontology, selects only codes that record actually
offered, and keeps ICD identity dotless.

`pack.jsonl` is **never** written. Imported results land in their own directory with
`adjudication_status = AI_SECOND_OPINION`, which is outside `HUMAN_GOLD_STATUSES`, so no
amount of second-opinion data can unlock the human benchmark.
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
    AI_CONFIDENCES,
    AI_DECISION_SELECTED,
    AI_DECISIONS,
    ONTOLOGY_ICD10,
    STATUS_AI_SECOND_OPINION,
    LinkerGoldError,
    assert_not_ai_gold,
    load_records,
    utc_now,
)

BASE = REPO / "data" / "private_linker_gold" / "0065"
BLIND = BASE / "chatgpt_handoff" / "minimum_benchmark_blind.jsonl"
DEFAULT_INPUT = BASE / "chatgpt_handoff" / "chatgpt_decisions.jsonl"
OUT = BASE / "ai_second_opinion"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    rows: list[dict[str, Any]], packet: dict[str, dict[str, Any]], *, allow_partial: bool
) -> list[dict[str, Any]]:
    problems: list[str] = []
    seen: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []

    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            problems.append(f"line {number}: not a JSON object")
            continue
        annotation_id = str(row.get("annotation_id", ""))
        record = packet.get(annotation_id)
        if record is None:
            problems.append(f"line {number}: unknown annotation_id {annotation_id!r}")
            continue
        seen[annotation_id] += 1
        if seen[annotation_id] > 1:
            problems.append(f"line {number}: duplicate decision for {annotation_id}")
            continue

        for field in (
            "ontology",
            "decision",
            "selected_codes",
            "confidence",
            "rationale",
            "evidence_used",
            "competing_codes",
            "ambiguity_flags",
        ):
            if field not in row:
                problems.append(f"{annotation_id}: missing required field {field!r}")
        if any(f"{annotation_id}: missing" in p for p in problems[-8:]):
            continue

        if row["ontology"] != record["ontology"]:
            problems.append(
                f"{annotation_id}: ontology {row['ontology']!r} does not match the record's "
                f"{record['ontology']!r}"
            )
        if row["decision"] not in AI_DECISIONS:
            problems.append(f"{annotation_id}: unknown decision {row['decision']!r}")
        if row["confidence"] not in AI_CONFIDENCES:
            problems.append(f"{annotation_id}: unknown confidence {row['confidence']!r}")

        codes = list(row["selected_codes"])
        if row["decision"] == AI_DECISION_SELECTED and not codes:
            problems.append(f"{annotation_id}: SELECTED with no codes")
        if row["decision"] != AI_DECISION_SELECTED and codes:
            problems.append(f"{annotation_id}: {row['decision']} must not carry selected codes")

        offered = {c.get("code_dotless") or c.get("rxcui") for c in record["offered_candidates"]}
        unknown = [c for c in codes if c not in offered]
        if unknown:
            problems.append(f"{annotation_id}: codes were never offered: {unknown}")
        if record["ontology"] == ONTOLOGY_ICD10 and any("." in c for c in codes):
            problems.append(f"{annotation_id}: ICD codes must be dotless internal identities")

        accepted.append(
            {
                "annotation_id": annotation_id,
                "ontology": record["ontology"],
                "adjudication_status": STATUS_AI_SECOND_OPINION,
                "ai_decision": row["decision"],
                "selected_codes": codes,
                "no_valid_candidate": row["decision"] == "NO_VALID_CANDIDATE",
                "ai_confidence": row["confidence"],
                "rationale": str(row["rationale"]),
                "evidence_used": list(row["evidence_used"]),
                "competing_codes": list(row["competing_codes"]),
                "ambiguity_flags": list(row["ambiguity_flags"]),
                "source": "chatgpt_second_opinion",
                "imported_at": utc_now(),
            }
        )

    missing = sorted(set(packet) - set(seen))
    if missing and not allow_partial:
        problems.append(
            f"{len(missing)} of {len(packet)} expected records have no decision "
            f"(first few: {missing[:3]}); pass --allow-partial to accept an incomplete return"
        )
    if problems:
        raise LinkerGoldError(
            f"{len(problems)} problem(s) in the returned decisions:\n  - "
            + "\n  - ".join(problems[:20])
        )
    return accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--blind", type=Path, default=BLIND)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    assert_not_ai_gold()
    if not args.decisions.is_file():
        print(f"decisions file not found: {args.decisions}", file=sys.stderr)
        print(
            "Upload the blind packet to the second AI, save its reply as that file, then re-run.",
            file=sys.stderr,
        )
        return 2
    packet = {r["annotation_id"]: r for r in load_records(args.blind)}
    rows = load_records(args.decisions)
    accepted = validate(rows, packet, allow_partial=args.allow_partial)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "chatgpt_second_opinion.jsonl"
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in accepted),
        encoding="utf-8",
    )
    manifest = {
        "imported_at": utc_now(),
        "source_file": str(args.decisions),
        "source_sha256": sha256_file(args.decisions),
        "blind_packet_sha256": sha256_file(args.blind),
        "expected_records": len(packet),
        "imported_records": len(accepted),
        "partial": bool(args.allow_partial),
        "status": STATUS_AI_SECOND_OPINION,
        "by_ontology": dict(Counter(r["ontology"] for r in accepted)),
        "decisions": dict(Counter(r["ai_decision"] for r in accepted)),
        "confidence": dict(Counter(r["ai_confidence"] for r in accepted)),
        "output_sha256": sha256_file(out_path),
        "pack_never_modified": True,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"imported {len(accepted)}/{len(packet)} -> {out_path}")
    print(f"  by ontology {manifest['by_ontology']}")
    print(f"  decisions   {manifest['decisions']}")
    print(f"  confidence  {manifest['confidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
