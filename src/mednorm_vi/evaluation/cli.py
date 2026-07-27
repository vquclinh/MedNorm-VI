"""Command-line interface for the PROVISIONAL LOCAL EVALUATOR.

Usage::

    python -m mednorm_vi.evaluation.cli \\
      --ground-truth data/dev_gold/example \\
      --predictions outputs/example \\
      --config configs/evaluation/provisional_v1.yaml \\
      --report-dir reports/example

Fails fast (nonzero exit) on malformed inputs or organizer-test-as-ground-truth;
permits low-quality but structurally valid predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import BANNER, evaluate_corpus
from .loading import load_ground_truth, load_predictions
from .models import EvaluationConfig, EvaluationRunMetadata, Provenance, parse_provenance
from .replay import build_replay_manifest, git_state, utc_timestamp
from .reporting_html import write_html
from .reporting_json import write_data_reports, write_replay_manifest

DEFAULT_CONFIG = "configs/evaluation/provisional_v1.yaml"


def _load_config_mapping(path: str | Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return data


def _print_issues(title: str, issues: Any) -> None:
    print(f"[{title}]", file=sys.stderr)
    for issue in issues:
        loc = f" (doc={issue.document_id}, entity={issue.entity_index})" \
            if issue.document_id else ""
        print(f"  - {issue.severity.value}: {issue.code}: {issue.message}{loc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedNorm-VI provisional local evaluator")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--provenance", default=None,
                        help="override GT provenance (GOLD/SILVER/SYNTHETIC/...)")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args(argv)

    print(BANNER)
    print("Organizer competition test set has NO ground truth; evaluating team-owned "
          "labeled data only.")

    try:
        config_mapping = _load_config_mapping(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load config: {exc}", file=sys.stderr)
        return 2
    config: EvaluationConfig = EvaluationConfig.from_mapping(config_mapping)

    provenance_override: Provenance | None = None
    if args.provenance:
        try:
            provenance_override = parse_provenance(args.provenance)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    gt_docs, provenance, gt_result = load_ground_truth(
        args.ground_truth, provenance=provenance_override
    )
    if not gt_result.ok:
        _print_issues("ground-truth validation failed", gt_result.errors)
        return 2
    if not gt_docs:
        print("error: no ground-truth documents loaded", file=sys.stderr)
        return 2
    if provenance is not None and provenance.value not in config.allowed_provenance:
        print(f"error: provenance {provenance.value} not in allowed "
              f"{list(config.allowed_provenance)}", file=sys.stderr)
        return 2

    pred_docs, pred_result = load_predictions(args.predictions)
    if not pred_result.ok:
        _print_issues("prediction validation failed", pred_result.errors)
        return 3

    outcome = evaluate_corpus(gt_docs, pred_docs, config)

    timestamp = utc_timestamp()
    commit, dirty = git_state()
    run_metadata = EvaluationRunMetadata(
        evaluator_version=config.evaluator_version,
        timestamp_utc=timestamp,
        python_version=sys.version.split()[0],
        platform=sys.platform,
        git_commit=commit,
        git_dirty=dirty,
    )
    prov_str = provenance.value if provenance else "UNKNOWN"

    report_hashes = write_data_reports(args.report_dir, outcome, config_mapping)
    if config.emit_html:
        write_html(args.report_dir, outcome, run_metadata, prov_str)
    manifest = build_replay_manifest(
        config=config, config_mapping=config_mapping,
        ground_truth_dir=args.ground_truth, predictions_dir=args.predictions,
        ground_truth_provenance=prov_str, prediction_experiment_id=args.experiment_id,
        timestamp=timestamp,
    )
    manifest_with_hashes = _attach_report_hashes(manifest, report_hashes)
    write_replay_manifest(args.report_dir, manifest_with_hashes)

    c = outcome.corpus
    print("")
    print(f"final provisional score : {c.final_score:.6f}")
    print(f"  text score            : {c.text_score:.6f}")
    print(f"  assertion score       : {c.assertions_score:.6f}")
    print(f"  candidate score       : {c.candidates_score:.6f}")
    print(f"matching strategy       : {config.matching_strategy}")
    print(f"tokenization strategy   : {config.tokenization}")
    print(f"aggregation strategy    : {config.aggregation_policy}")
    print(f"data provenance         : {prov_str}")
    print(f"report path             : {Path(args.report_dir).resolve()}")
    return 0


def _attach_report_hashes(manifest: Any, report_hashes: dict[str, str]) -> Any:
    import dataclasses

    return dataclasses.replace(manifest, report_file_hashes=dict(sorted(report_hashes.items())))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
