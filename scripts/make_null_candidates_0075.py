#!/usr/bin/env python3
"""Submission #3 - empty every candidate list, change nothing else (sprint 0075).

A direct leaderboard measurement of what candidate refusal is worth. `text`, `type`,
`position` and `assertions` are copied byte-for-byte; only `candidates` becomes `[]`, and
only on the two types that may carry it.

Requires E3 only: run the deterministic `full_v1.yaml` pipeline first, then apply this.
No embedding model, no reranker, no 8B. Deployed parameters: 135,002,117.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.reasoner.validator import CANDIDATE_TYPES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int, default=100)
    args = parser.parse_args(argv)

    files = sorted(args.source_dir.glob("*.json"))
    if len(files) != args.expected_documents:
        print(
            f"BLOCKED: found {len(files)} documents, expected {args.expected_documents}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emptied: Counter[str] = Counter()
    untouched = codes_dropped = entities = 0
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for entity in rows:
            entities += 1
            row = dict(entity)
            if row.get("type") in CANDIDATE_TYPES and "candidates" in row:
                codes_dropped += len(row["candidates"])
                row["candidates"] = []
                emptied[row["type"]] += 1
            else:
                untouched += 1
            out.append(row)
        (args.output_dir / path.name).write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8"
        )

    print(f"documents {len(files)} | entities {entities} (unchanged)")
    print(f"candidate lists emptied: {dict(emptied)} ({codes_dropped} codes removed)")
    print(f"entities untouched: {untouched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
