"""Experiment-tracking CLI (local only; never contacts the network).

    python -m mednorm_vi.experiments.cli create --title "baseline deterministic"
    python -m mednorm_vi.experiments.cli attach-output --experiment EXP-0001 --zip output.zip
    python -m mednorm_vi.experiments.cli record-local --experiment EXP-0001 \\
        --kind gold --score 0.83
    python -m mednorm_vi.experiments.cli record-leaderboard --experiment EXP-0001 \\
        --score 0.71234 --submission-id optional-id
    python -m mednorm_vi.experiments.cli show --experiment EXP-0001
    python -m mednorm_vi.experiments.cli compare EXP-0001 EXP-0002
"""

from __future__ import annotations

import argparse
import json
import sys

from ..evaluation.models import jsonable
from .leaderboard import attach_output, compare, record_leaderboard, record_local_score
from .models import LocalScoreKind
from .registry import ExperimentExistsError, ExperimentRegistry

DEFAULT_REGISTRY = "experiments/registry"


def _registry(args: argparse.Namespace) -> ExperimentRegistry:
    return ExperimentRegistry(args.registry)


def _cmd_create(args: argparse.Namespace) -> int:
    reg = _registry(args)
    try:
        record = reg.create(title=args.title, description=args.description or "",
                            model_profile=args.model_profile, seed=args.seed)
    except ExperimentExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"created {record.experiment_id}: {record.title}")
    return 0


def _cmd_attach_output(args: argparse.Namespace) -> int:
    reg = _registry(args)
    record = attach_output(reg, args.experiment, args.zip)
    print(f"attached {args.zip} to {record.experiment_id} "
          f"(zip sha256={record.output_zip_hash})")
    return 0


def _cmd_record_local(args: argparse.Namespace) -> int:
    reg = _registry(args)
    record = record_local_score(
        reg, args.experiment, kind=LocalScoreKind(args.kind), final_score=args.score,
        text_score=args.text, assertions_score=args.assertions,
        candidates_score=args.candidates, dataset_id=args.dataset_id,
    )
    print(f"recorded local {args.kind} score {args.score} on {record.experiment_id}")
    return 0


def _cmd_record_leaderboard(args: argparse.Namespace) -> int:
    reg = _registry(args)
    record = record_leaderboard(reg, args.experiment, score=args.score,
                                submission_id=args.submission_id)
    print(f"recorded leaderboard score {args.score} on {record.experiment_id} "
          "(manually entered)")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    reg = _registry(args)
    record = reg.load(args.experiment)
    print(json.dumps(jsonable(record), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    reg = _registry(args)
    records = [reg.load(eid) for eid in args.experiments]
    rows = compare(records)
    print("experiment    local_gold  local_silver  local_synthetic  leaderboard  title")
    for r in rows:
        print(f"{str(r['experiment_id']):<13} {str(r['local_gold']):<11} "
              f"{str(r['local_silver']):<13} {str(r['local_synthetic']):<16} "
              f"{str(r['leaderboard']):<12} {r['title']}")
    print("\nNote: leaderboard score never replaces local error analysis.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedNorm-VI experiment tracker (local only)")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--description", default="")
    p_create.add_argument("--model-profile", default=None, dest="model_profile")
    p_create.add_argument("--seed", type=int, default=None)
    p_create.set_defaults(func=_cmd_create)

    p_attach = sub.add_parser("attach-output")
    p_attach.add_argument("--experiment", required=True)
    p_attach.add_argument("--zip", required=True)
    p_attach.set_defaults(func=_cmd_attach_output)

    p_local = sub.add_parser("record-local")
    p_local.add_argument("--experiment", required=True)
    p_local.add_argument("--kind", choices=[k.value for k in LocalScoreKind], required=True)
    p_local.add_argument("--score", type=float, required=True)
    p_local.add_argument("--text", type=float, default=None)
    p_local.add_argument("--assertions", type=float, default=None)
    p_local.add_argument("--candidates", type=float, default=None)
    p_local.add_argument("--dataset-id", default=None, dest="dataset_id")
    p_local.set_defaults(func=_cmd_record_local)

    p_lb = sub.add_parser("record-leaderboard")
    p_lb.add_argument("--experiment", required=True)
    p_lb.add_argument("--score", type=float, required=True)
    p_lb.add_argument("--submission-id", default=None, dest="submission_id")
    p_lb.set_defaults(func=_cmd_record_leaderboard)

    p_show = sub.add_parser("show")
    p_show.add_argument("--experiment", required=True)
    p_show.set_defaults(func=_cmd_show)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("experiments", nargs="+")
    p_compare.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
