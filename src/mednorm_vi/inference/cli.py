"""MedNorm-VI full-pipeline inference CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import PipelineConfig
from .pipeline import run_input_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run L1-L9 and package output.zip")
    run.add_argument("--input-dir", required=True)
    run.add_argument("--output-zip", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--mode", choices=("deterministic", "specialist", "full"), required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        config = PipelineConfig.load(args.config)
        try:
            results = run_input_dir(
                args.input_dir,
                output_zip=args.output_zip,
                config=config,
                mode=args.mode,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"processed_documents: {len(results)}")
        print(f"output_zip: {args.output_zip}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
