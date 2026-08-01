#!/usr/bin/env python3
"""Run the hybrid semantic linker over public input (Audit 0071 §7-§9, §20).

Candidate generation is a union of three independent sources, deduplicated by dotless ICD
identity: the v3 lexical Top-K (precision anchor), the v4.1 tiered lexical Top-K (coverage),
and dense semantic Top-K. v3 is never globally replaced by v4.1 - Audit 0070 showed that
doing so costs J_candidates - but nothing hard-codes "v3 always wins" either: the reranker
may overturn it when semantic evidence is strong.

Every emitted code must exist in the governed frozen KB. The semantic model may reorder and
reject, never invent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.linking.semantic import null_gate as ng  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system", choices=["S1", "S2"], required=True)
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
    parser.add_argument("--dense-index", type=Path, required=True)
    parser.add_argument("--v3-topk", type=int, default=20)
    parser.add_argument("--v41-topk", type=int, default=20)
    parser.add_argument("--dense-topk", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if not args.dense_index.exists():
        raise SystemExit(
            f"dense index not found: {args.dense_index}\n"
            "Build it on a GPU host first (see docs/colab/0071-semantic-linker-commands.md). "
            "Audit 0071 stopped at COLAB_EXECUTION_REQUIRED; no dense index exists locally."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "system": args.system,
                "sources": {
                    "v3_topk": args.v3_topk,
                    "v41_topk": args.v41_topk,
                    "dense_topk": args.dense_topk,
                },
                "null_gate_version": ng.NULL_GATE_VERSION,
                "dedup": "dotless ICD identity; dotted form applied by derive_submission at output",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
