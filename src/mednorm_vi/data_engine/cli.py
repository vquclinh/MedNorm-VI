"""CLI for deterministic Data Engine dry runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .adapters import document_from_text
from .build import build_dataset
from .corpus_build import DEFAULT_SEED, build_governed_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-from-text-dir", help="build a manifest from text files")
    build.add_argument("--input-dir", required=True)
    build.add_argument("--folds", type=int, default=5)
    gov = sub.add_parser("build-governed-corpus",
                         help="build the deterministic governed training corpus (v1)")
    gov.add_argument("--base-dir", default="data/external/public_ner")
    gov.add_argument("--out-dir", default="data/derived/training_corpora/mednorm_vi_training_v1")
    gov.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.command == "build-from-text-dir":
        docs = tuple(
            document_from_text(path.stem, path.read_text(encoding="utf-8"), source_id="text_dir")
            for path in sorted(Path(args.input_dir).glob("*.txt"))
        )
        result = build_dataset(docs, folds=args.folds)
        print(json.dumps(asdict(result.manifest), ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if not result.errors else 2
    if args.command == "build-governed-corpus":
        report = build_governed_corpus(args.base_dir, args.out_dir, seed=args.seed)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
