"""MedNorm-VI full-pipeline inference CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace

from ..kb.competition.topk import resolve_decoding_profile
from .config import PipelineConfig
from .pipeline import FinalValidationError, KbMembershipViolation, run_input_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run L1-L9 and package output.zip")
    run.add_argument("--input-dir", required=True)
    run.add_argument("--output-zip", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--mode", choices=("deterministic", "specialist", "full"), required=True)
    run.add_argument(
        "--decoding-profile",
        default=None,
        help=(
            "override the config's governed decoding profile "
            "(competition_top1 | competition_adaptive)"
        ),
    )
    run.add_argument(
        "--run-manifest",
        default=None,
        metavar="PATH",
        help=(
            "write a deterministic, clinical-text-free run manifest to PATH. "
            "Opt-in: nothing is written unless this flag is given."
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        config = PipelineConfig.load(args.config)
        if args.decoding_profile:
            # Validate eagerly so an unknown profile fails before any inference runs.
            resolve_decoding_profile(args.decoding_profile)
            config = replace(config, decoding_profile=args.decoding_profile)
        try:
            results = run_input_dir(
                args.input_dir,
                output_zip=args.output_zip,
                config=config,
                mode=args.mode,
                run_manifest_path=args.run_manifest,
            )
        except (FinalValidationError, KbMembershipViolation) as exc:
            # L9 stopped the run. When a manifest was requested it has already been
            # written with `l9_stopped_the_run: true`, so the refusal is recorded.
            # Nothing was written: no output/ directory and no output.zip exist.
            print(f"error: {exc}", file=sys.stderr)
            return 3
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"processed_documents: {len(results)}")
        print(f"output_zip: {args.output_zip}")
        if args.run_manifest:
            print(f"run_manifest: {args.run_manifest}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
