#!/usr/bin/env python3
"""Fine-tune the SELECTED semantic architecture (Audit 0071 §15).

Runs only after a zero-shot measurement has chosen one architecture. Training data is
ontology-native supervision from the concept-disjoint benchmark; AI consensus is never used
as ground truth here, and public-test material is never used at all.

Heavy dependencies are imported lazily so the file stays checkable without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="official repo id, e.g. Qwen/Qwen3-Embedding-4B"
    )
    parser.add_argument("--revision", required=True, help="immutable commit sha")
    parser.add_argument("--objective", choices=["contrastive", "reranker"], required=True)
    parser.add_argument(
        "--train", type=Path, default=REPO / "artifacts/semantic/0071/benchmark/train.jsonl"
    )
    parser.add_argument(
        "--dev", type=Path, default=REPO / "artifacts/semantic/0071/benchmark/dev.jsonl"
    )
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/semantic/0071/trained")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--seed", type=int, default=0, help="one config + one confirmation seed only"
    )
    parser.add_argument(
        "--null-fraction",
        type=float,
        default=0.2,
        help="share of NULL examples, so the rejection gate is trained too",
    )
    args = parser.parse_args(argv)

    import torch

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": args.model,
        "revision": args.revision,
        "objective": args.objective,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "null_fraction": args.null_fraction,
        "train_rows": sum(1 for _ in args.train.open(encoding="utf-8")),
        "dev_rows": sum(1 for _ in args.dev.open(encoding="utf-8")),
        "supervision": "ontology-native only; AI consensus is calibration, never ground truth",
        "public_test_used": False,
    }
    (args.out / "training-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    print(
        "\nTraining loop intentionally not executed here: Audit 0071 stopped at "
        "COLAB_EXECUTION_REQUIRED before any zero-shot measurement existed to select an "
        "architecture. Run evaluate_semantic_linker_0071.py first (§15 forbids fine-tuning "
        "before a zero-shot number)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
