"""Training workflow CLI for dry-run planning and readiness checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from .stages import build_training_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="emit deterministic S0-S6 training plan")
    plan.add_argument("--artifact-root", default="models/checkpoints/full_v1")
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(asdict(build_training_plan(artifact_root=args.artifact_root)),
                         sort_keys=True, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
