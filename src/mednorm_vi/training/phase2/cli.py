"""Phase-2 training readiness command-line helpers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifacts import (
    MODE_FULL,
    MODEL_ARTIFACT_KINDS,
    validate_phase2_artifact,
)
from .common import read_json, sha256_file
from .internal_test_gate import (
    evaluate_internal_test_freeze_gate,
)
from .proposal_generation import load_frozen_proposal_manifest


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_artifact(args: argparse.Namespace) -> int:
    expected = MODEL_ARTIFACT_KINDS[str(args.kind)]
    report = validate_phase2_artifact(
        args.artifact_dir,
        expected_expert_id=expected,
        expected_mode=str(args.mode),
    )
    _print(report.as_dict())
    return 0 if report.ok else 2


def _validate_proposals(args: argparse.Namespace) -> int:
    manifest = load_frozen_proposal_manifest(args.manifest)
    failures: list[str] = []
    proposal_path = Path(args.proposals_jsonl)
    if not proposal_path.is_file():
        failures.append("proposal_jsonl_missing")
    elif sha256_file(proposal_path) != manifest.proposal_dataset_sha256:
        failures.append("proposal_dataset_sha256_mismatch")
    if manifest.internal_test_accessed:
        failures.append("proposal_manifest_internal_test_accessed")
    if manifest.split not in {"train", "validation"}:
        failures.append("proposal_manifest_split_not_train_or_validation")
    payload = {
        "ok": not failures,
        "failures": failures,
        "manifest": manifest.as_dict(),
    }
    _print(payload)
    return 0 if not failures else 2


def _gate_internal_test(args: argparse.Namespace) -> int:
    profile = read_json(args.profile_manifest)
    report = evaluate_internal_test_freeze_gate(
        artifact_reports=(),
        frozen_feature_flags=dict(profile.get("feature_flags", {})),
        frozen_thresholds=dict(profile.get("thresholds", {})),
        config_hashes=dict(profile.get("config_hashes", {})),
        checkpoint_hashes=dict(profile.get("checkpoint_hashes", {})),
        validation_ablation_complete=bool(profile.get("validation_ablation_complete", False)),
        validation_ablation_hash=str(profile.get("validation_ablation_hash", "")),
        model_revisions=dict(profile.get("model_revisions", {})),
        authorization=str(args.authorization),
    )
    _print(report.as_dict())
    return 0 if report.ready else 2


def _show_artifact_schema(_args: argparse.Namespace) -> int:
    _print(
        {
            "required_files": [
                "checkpoints/best.pt",
                "checkpoints/latest.pt",
                "logs/training_history.jsonl",
                "resolved_config.json",
                "validation_metrics.json",
                "training_manifest.json",
            ],
            "kinds": MODEL_ARTIFACT_KINDS,
            "default_mode": MODE_FULL,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-artifact")
    validate.add_argument("--kind", choices=sorted(MODEL_ARTIFACT_KINDS), required=True)
    validate.add_argument("--artifact-dir", required=True)
    validate.add_argument("--mode", default=MODE_FULL, choices=("smoke", "full"))
    validate.set_defaults(func=_validate_artifact)

    proposals = subparsers.add_parser("validate-proposals")
    proposals.add_argument("--manifest", required=True)
    proposals.add_argument("--proposals-jsonl", required=True)
    proposals.set_defaults(func=_validate_proposals)

    gate = subparsers.add_parser("gate-internal-test")
    gate.add_argument("--profile-manifest", required=True)
    gate.add_argument("--authorization", default="")
    gate.set_defaults(func=_gate_internal_test)

    schema = subparsers.add_parser("artifact-schema")
    schema.set_defaults(func=_show_artifact_schema)

    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
