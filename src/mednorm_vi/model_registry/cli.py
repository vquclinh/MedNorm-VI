"""Model-registry budget validation CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from .registry import load_registry, validate_profile_budget


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/model_registry/models_v1.yaml")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--require-local-paths", action="store_true")
    args = parser.parse_args(argv)
    budget = validate_profile_budget(
        load_registry(args.registry),
        profile=args.profile,
        require_local_paths=args.require_local_paths,
    )
    print(json.dumps(asdict(budget), sort_keys=True, indent=2))
    return 0 if budget.within_9b and not budget.missing_checkpoints else 2


if __name__ == "__main__":
    raise SystemExit(main())
