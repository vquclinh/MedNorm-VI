"""CLI for local KB index builders."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from .builders import build_icd_index, build_rxnorm_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    icd = sub.add_parser("build-icd", help="build ICD index from normalized CSV")
    icd.add_argument("--normalized-csv", required=True)
    icd.add_argument("--output-dir", required=True)
    icd.add_argument("--snapshot-id", required=True)
    icd.add_argument("--source-hash", default="")
    rxn = sub.add_parser("build-rxnorm", help="build RxNorm index from an RRF root")
    rxn.add_argument("--rrf-root", required=True)
    rxn.add_argument("--output-dir", required=True)
    rxn.add_argument("--snapshot-id", required=True)
    rxn.add_argument("--source-hash", default="")
    args = parser.parse_args(argv)
    if args.command == "build-icd":
        meta = build_icd_index(
            args.normalized_csv,
            args.output_dir,
            source_snapshot_id=args.snapshot_id,
            source_hash=args.source_hash,
        )
    elif args.command == "build-rxnorm":
        meta = build_rxnorm_index(
            args.rrf_root,
            args.output_dir,
            source_snapshot_id=args.snapshot_id,
            source_hash=args.source_hash,
        )
    else:
        return 2
    print(json.dumps(asdict(meta), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
