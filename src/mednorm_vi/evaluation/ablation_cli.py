"""Run the exact four-way ablation on the governed internal-test split.

Usage::

    env PYTHONPATH=src python -m mednorm_vi.evaluation.ablation_cli \\
      --split data/derived/training_corpora/.../splits/internal_test.jsonl \\
      --checkpoint checkpoint/s1_mention_full_training_v1/best.pt \\
      --checkpoint-sha256 <64 hex> \\
      --pinned-revision <40 hex> \\
      --report-dir reports/l3_l4_ablation

Forward passes only. The checkpoint digest and mtime are captured before and
after inference and compared; a mismatch aborts. Reports land in the git-ignored
``reports/`` tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..deterministic_baseline.models import Phase1BConfig
from ..mention_factory.neural.decoding import NeuralSpan
from ..resolution.config_v1 import DEFAULT_CONFIG_PATH, load_resolver_v1_config
from .ablation import run_ablation
from .exact_mention import render_markdown
from .symptom_attribution import render_markdown as render_symptom_markdown

DEFAULT_ROUTER_CONFIG = "configs/case_router/base.yaml"
DEFAULT_MEDICATION_CONFIG = "configs/medication/grammar_v1.yaml"
DEFAULT_LABORATORY_CONFIG = "configs/laboratory/parser_v1.yaml"


def _load_cached_spans(path: str | Path) -> dict[str, tuple[NeuralSpan, ...]]:
    """Reload previously decoded E3 spans instead of re-running inference."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        example_id: tuple(
            NeuralSpan(int(row[0]), int(row[1]), str(row[2]), str(row[3]),
                       float(row[4]), int(row[5]))
            for row in rows)
        for example_id, rows in payload.items()
    }


def _dump_spans(spans: dict[str, tuple[NeuralSpan, ...]], path: Path) -> None:
    payload = {
        example_id: [[s.start, s.end, s.entity_type, s.text, s.score, s.token_count]
                     for s in rows]
        for example_id, rows in spans.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")


def _neural_spans(args: argparse.Namespace, examples: list[dict[str, Any]],
                  ) -> tuple[dict[str, tuple[NeuralSpan, ...]], dict[str, Any]]:
    """Decode E3 spans with the validated checkpoint, or reload a cache."""
    if args.neural_span_cache and Path(args.neural_span_cache).is_file():
        return _load_cached_spans(args.neural_span_cache), {
            "source": "cache", "path": args.neural_span_cache}

    from ..mention_factory.neural.runtime import (
        NeuralExpertConfig,
        build_segmenter,
        fingerprint_checkpoint,
        load_expert,
    )

    config = NeuralExpertConfig(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        pinned_model_revision=args.pinned_revision,
        model_cache_dir=args.model_cache_dir,
        vncorenlp_dir=args.vncorenlp_dir,
        batch_size=args.batch_size)
    before = fingerprint_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256)
    expert = load_expert(config, segmenter=build_segmenter(args.vncorenlp_dir))
    spans = expert.predict_spans(examples)
    after = expert.verify_checkpoint_unchanged()
    before.assert_unchanged(after)
    if args.neural_span_cache:
        _dump_spans(spans, Path(args.neural_span_cache))
    return spans, {
        "source": "checkpoint",
        "checkpoint_before": before.as_dict(),
        "checkpoint_after": after.as_dict(),
        "checkpoint_unchanged": True,
        "backend": expert.report.backend,
        "training_performed": False,
        "backward_called": False,
        "optimizer_constructed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MedNorm-VI exact four-way L3/L4 ablation")
    parser.add_argument("--split", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--pinned-revision", default="")
    parser.add_argument("--model-cache-dir", default="")
    parser.add_argument("--vncorenlp-dir", default="")
    parser.add_argument("--neural-span-cache", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resolver-config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--router-config", default=DEFAULT_ROUTER_CONFIG)
    parser.add_argument("--medication-config", default=DEFAULT_MEDICATION_CONFIG)
    parser.add_argument("--laboratory-config", default=DEFAULT_LABORATORY_CONFIG)
    args = parser.parse_args(argv)

    examples: list[dict[str, Any]] = []
    with Path(args.split).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
            if args.limit and len(examples) >= args.limit:
                break

    phase1b_config = Phase1BConfig.load(
        args.router_config, args.medication_config, args.laboratory_config)
    resolver_config = load_resolver_v1_config(args.resolver_config)

    spans, provenance = _neural_spans(args, examples)
    report = run_ablation(
        examples, phase1b_config, resolver_config,
        neural_spans_for=lambda example_id: spans.get(example_id, ()))
    report["neural_provenance"] = provenance
    report["split"] = args.split

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "four_way_ablation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    for arm, arm_report in report["arms"].items():
        (report_dir / f"{arm}.md").write_text(render_markdown(arm_report), encoding="utf-8")
    (report_dir / "symptom_attribution.md").write_text(
        render_symptom_markdown(report["symptom_attribution"]), encoding="utf-8")

    print(json.dumps({
        arm: {
            "precision": round(arm_report["micro"]["precision"], 4),
            "recall": round(arm_report["micro"]["recall"], 4),
            "f1": round(arm_report["micro"]["f1"], 4),
            "final_count": arm_report["final_count"],
            "boundary_errors": arm_report["boundary_error_total"],
            "delta_f1_vs_arm1": arm_report["delta_vs_arm1"]["f1"],
        }
        for arm, arm_report in report["arms"].items()
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
