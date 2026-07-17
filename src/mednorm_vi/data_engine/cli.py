"""CLI for deterministic Data Engine dry runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .adapters import document_from_text
from .build import build_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-from-text-dir", help="build a manifest from text files")
    build.add_argument("--input-dir", required=True)
    build.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "build-from-text-dir":
        docs = tuple(
            document_from_text(path.stem, path.read_text(encoding="utf-8"), source_id="text_dir")
            for path in sorted(Path(args.input_dir).glob("*.txt"))
        )
        result = build_dataset(docs, folds=args.folds)
        print(json.dumps(asdict(result.manifest), ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if not result.errors else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
