"""Round-2 descriptor comparison CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from .compare import compare_task_descriptors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--upgraded", required=True)
    args = parser.parse_args(argv)
    report = compare_task_descriptors(args.current, args.upgraded)
    print(json.dumps(asdict(report), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
