#!/usr/bin/env python3
"""Evaluate any E3 span/type checkpoint against the governed acceptance gate (Audit 0062).

One evaluator for every recipe, so R0/R1/R2/R3 are never compared across two rules.
Metrics are exact CHARACTER spans decoded by the production decoder
(``mention_factory.neural.decoding.decode_type_runs``) and scored by
``evaluation.exact_mention`` - never token-level accuracy (brief §7).

The gate encoded in :func:`apply_gate` is the one the milestone brief specifies, at the
thresholds it specifies. It is not parameterised, because a gate a caller can loosen at
the command line is not a gate.

Usage::

    python scripts/evaluate_e3_checkpoint_0062.py \
        --checkpoint checkpoint/experiments/0062_e3_boundary_refinement/R1/best.pt \
        --label R1_lowlr --output runs/diagnostics/0062_recipes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.evaluation.exact_mention import Mention, evaluate_examples  # noqa: E402
from mednorm_vi.mention_factory.experts.e3_vihealthbert import (  # noqa: E402
    E3_PINNED_MODEL_REVISION,
)
from mednorm_vi.training.governed_splits import resolve_governed_splits  # noqa: E402
from mednorm_vi.training.s1_mention_smoke import (  # noqa: E402
    ENTITY_TYPE_ORDER,
    iter_jsonl,
    load_coverage,
)

# The Audit-0061 baseline every candidate is judged against, reproduced exactly in
# Audit 0062 §3 through this same evaluator.
#
# Stored as CONFUSION COUNTS, not as the rounded rates the audits print. Comparing a
# full-precision candidate against a 4-decimal constant makes the control fail its own
# gate by -0.00: 1064/1697 is 0.626989, which is not >= 0.6270. A gate that a
# reproduction of the baseline cannot pass is measuring rounding, not quality.
BASELINE_COUNTS = {
    "micro": (1064, 633, 927),
    "DIAGNOSIS": (670, 344, 632),
    "SYMPTOM": (297, 241, 236),
    "MEDICATION": (97, 31, 59),
}
BASELINE_BOUNDARY_ERRORS = 364


def _rates(counts: tuple[int, int, int]) -> tuple[float, float, float]:
    true_positive, false_positive, false_negative = counts
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


BASELINE = {
    "f1": _rates(BASELINE_COUNTS["micro"])[2],
    "precision": _rates(BASELINE_COUNTS["micro"])[0],
    "recall": _rates(BASELINE_COUNTS["micro"])[1],
    "DIAGNOSIS": _rates(BASELINE_COUNTS["DIAGNOSIS"])[2],
    "SYMPTOM": _rates(BASELINE_COUNTS["SYMPTOM"])[2],
    "MEDICATION": _rates(BASELINE_COUNTS["MEDICATION"])[2],
    "boundary_errors": BASELINE_BOUNDARY_ERRORS,
}

# Brief §8, verbatim thresholds. Absolute points.
GATE_MIN_F1_GAIN = 1.5
GATE_MAX_RECALL_DROP = 1.0
GATE_MAX_MEDICATION_DROP = 1.0
GATE_CONFIRMATION_MIN_F1_GAIN = 1.0
MAX_DEPLOYMENT_PARAMETERS = 9_000_000_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint(checkpoint: Path, torch: Any) -> dict[str, Any]:
    """Read the checkpoint's provenance fields without building a model."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("mode", "")) != "FULL_TRAINING":
        raise RuntimeError(f"checkpoint mode is {payload.get('mode')!r}, not FULL_TRAINING")
    if tuple(payload.get("entity_type_order", ())) != tuple(ENTITY_TYPE_ORDER):
        raise RuntimeError("checkpoint label space does not match this repository")
    state = payload["model_state_dict"]
    return {
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "seed": payload.get("seed"),
        "pinned_model_revision": (
            str(payload.get("pinned_model_revision", "")) or E3_PINNED_MODEL_REVISION
        ),
        "refinement_source_sha256": str(payload.get("refinement_source_sha256", "")),
        "refinement_run_id": str(payload.get("refinement_run_id", "")),
        "total_parameters": int(sum(int(t.numel()) for t in state.values())),
    }


def run_pipeline(
    checkpoint: Path,
    digest: str,
    examples: list[dict],
    *,
    threshold: float,
    batch_size: int,
    workdir: Path,
) -> tuple[dict[str, list[Mention]], int, float, int]:
    """Run L1->L4 with this checkpoint injected through the governed E3 settings.

    The gate's baseline (F1 0.5770) is an L1->L4 number, so a candidate has to be
    measured on the same path. Evaluating E3 in isolation would produce a number that
    looks comparable and is not.
    """
    import resource

    from mednorm_vi.deterministic_baseline.models import Phase1BConfig
    from mednorm_vi.deterministic_baseline.pipeline import run_phase1b
    from mednorm_vi.document_intelligence import analyze_document
    from mednorm_vi.document_intelligence.builder import load_config as load_l1_config
    from mednorm_vi.inference.config import PipelineConfig, flags_for_mode
    from mednorm_vi.lattice.builder import build_span_lattice, lattice_config_hash
    from mednorm_vi.mention_factory import experts as _experts  # noqa: F401  registers E3
    from mednorm_vi.mention_factory.registry import run_registered_experts
    from mednorm_vi.resolution.canonical import resolve_lattice_to_hypotheses
    from mednorm_vi.resolution.config_v1 import load_resolver_v1_config
    from mednorm_vi.schemas.constants import TYPE_BY_ORGANIZER_LABEL

    mode = "specialist"
    config = PipelineConfig.load(str(REPO / "configs" / "pipeline" / "full_v1.yaml"))
    l1_config, l1_lexicon = load_l1_config(config.l1_config)
    p1b_config = Phase1BConfig.load(
        config.router_config, config.medication_config, config.laboratory_config
    )
    resolver_config = load_resolver_v1_config(config.l4_config)
    scoped = flags_for_mode(config, mode)
    settings = dict(config.expert_settings)
    settings["e3_checkpoint_path"] = str(checkpoint)
    settings["e3_expected_checkpoint_sha256"] = digest
    settings["e3_decision_threshold"] = threshold
    settings["e3_batch_size"] = batch_size
    prepared: dict[str, Any] = {}

    workdir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[Mention]] = {}
    offset_violations = 0
    started = time.time()
    for index, example in enumerate(examples, start=1):
        example_id = str(example["example_id"])
        text = str(example["text"])
        document = workdir / f"{example_id}.txt"
        document.write_text(text, encoding="utf-8")
        graph = analyze_document(document, config=l1_config, lexicon=l1_lexicon)
        phase1b = run_phase1b(graph, p1b_config)
        expert_proposals, _records = run_registered_experts(
            graph, phase1b.routings, feature_flags=scoped, settings=settings, prepared=prepared
        )
        lattice = build_span_lattice(
            graph.document_id,
            graph.original_text,
            routings=phase1b.routings,
            specialist_proposals=phase1b.proposals,
            expert_spans=expert_proposals,
            relations=phase1b.relations,
            config_hash=lattice_config_hash(
                {
                    "mode": mode,
                    "feature_flags": dict(sorted(scoped.items())),
                    "l4_config": config.l4_config,
                }
            ),
        )
        resolved = resolve_lattice_to_hypotheses(
            lattice, resolver_config, relations=phase1b.relations
        )
        mentions = []
        for hypothesis in resolved.accepted():
            if text[hypothesis.start : hypothesis.end] != hypothesis.text:
                offset_violations += 1
            mentions.append(
                Mention(
                    hypothesis.start,
                    hypothesis.end,
                    TYPE_BY_ORGANIZER_LABEL.get(hypothesis.entity_type, hypothesis.entity_type),
                    hypothesis.text,
                )
            )
        predictions[example_id] = mentions
        document.unlink(missing_ok=True)
        if index % 200 == 0 or index == len(examples):
            print(f"  [{index}/{len(examples)}] {time.time() - started:.0f}s", flush=True)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return predictions, offset_violations, time.time() - started, peak_rss


def apply_gate(current: dict[str, Any], *, confirmation_f1: float | None) -> dict[str, Any]:
    """The brief §8 gate. Every criterion is reported, pass or fail."""
    f1 = current["f1"] * 100
    precision = current["precision"] * 100
    recall = current["recall"] * 100
    by_type = current["by_type"]

    def type_f1(name: str) -> float:
        return by_type.get(name, {}).get("f1", 0.0) * 100

    checks = [
        (
            "exact span+type micro-F1 >= +1.5",
            f1 - BASELINE["f1"] * 100 >= GATE_MIN_F1_GAIN,
            f"{f1 - BASELINE['f1'] * 100:+.2f}",
        ),
        (
            "precision must not decrease",
            precision >= BASELINE["precision"] * 100,
            f"{precision - BASELINE['precision'] * 100:+.2f}",
        ),
        (
            "recall drop <= 1.0",
            recall - BASELINE["recall"] * 100 >= -GATE_MAX_RECALL_DROP,
            f"{recall - BASELINE['recall'] * 100:+.2f}",
        ),
        (
            "DIAGNOSIS F1 must not decrease",
            type_f1("DIAGNOSIS") >= BASELINE["DIAGNOSIS"] * 100,
            f"{type_f1('DIAGNOSIS') - BASELINE['DIAGNOSIS'] * 100:+.2f}",
        ),
        (
            "SYMPTOM F1 must not decrease",
            type_f1("SYMPTOM") >= BASELINE["SYMPTOM"] * 100,
            f"{type_f1('SYMPTOM') - BASELINE['SYMPTOM'] * 100:+.2f}",
        ),
        (
            "MEDICATION F1 drop <= 1.0",
            type_f1("MEDICATION") - BASELINE["MEDICATION"] * 100 >= -GATE_MAX_MEDICATION_DROP,
            f"{type_f1('MEDICATION') - BASELINE['MEDICATION'] * 100:+.2f}",
        ),
        (
            "total boundary errors must decrease",
            current["boundary_errors"] < BASELINE["boundary_errors"],
            f"{current['boundary_errors']} vs {BASELINE['boundary_errors']}",
        ),
        (
            "offset violations == 0",
            current["offset_violations"] == 0,
            str(current["offset_violations"]),
        ),
        (
            "no entity-type collapse",
            all(
                current["predictions_by_type"].get(t, 0) > 0
                for t in ("DIAGNOSIS", "SYMPTOM", "MEDICATION")
            ),
            json.dumps(current["predictions_by_type"], sort_keys=True),
        ),
        (
            "parameter budget < 9B",
            current["total_parameters"] < MAX_DEPLOYMENT_PARAMETERS,
            f"{current['total_parameters']:,}",
        ),
    ]
    if confirmation_f1 is not None:
        checks.append(
            (
                "confirmation seed retains >= +1.0 F1",
                confirmation_f1 * 100 - BASELINE["f1"] * 100 >= GATE_CONFIRMATION_MIN_F1_GAIN,
                f"{confirmation_f1 * 100 - BASELINE['f1'] * 100:+.2f}",
            )
        )
    return {
        "criteria": [
            {"criterion": name, "passed": bool(ok), "observed": observed}
            for name, ok, observed in checks
        ],
        "passed": all(ok for _, ok, _ in checks),
        "confirmation_seed_evaluated": confirmation_f1 is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--output", type=Path, default=REPO / "runs" / "diagnostics" / "0062_recipes"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument(
        "--confirmation-f1",
        type=float,
        default=None,
        help="F1 of the confirmation-seed run, when judging the selected recipe",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/mednorm_0062_eval"),
        help="scratch directory for per-document L1 input",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    digest = sha256_file(args.checkpoint)
    print(
        f"=== evaluating {args.label} ===\ncheckpoint {args.checkpoint}\nsha256 {digest}",
        flush=True,
    )

    import torch

    corpus_dir = args.corpus_dir or (
        REPO / "data" / "derived" / "training_corpora" / "mednorm_vi_training_v1"
    )
    splits = resolve_governed_splits([corpus_dir], splits=("validation",))
    load_coverage(corpus_dir)  # fails closed when the governed manifest is absent
    examples = list(iter_jsonl(splits["validation"].path))
    print(
        f"validation {splits['validation'].path} sha256={splits['validation'].sha256} "
        f"({len(examples)} examples)",
        flush=True,
    )

    meta = inspect_checkpoint(args.checkpoint, torch)
    predictions, offset_violations, elapsed, peak_vram = run_pipeline(
        args.checkpoint,
        digest,
        examples,
        threshold=args.threshold,
        batch_size=args.batch_size,
        workdir=args.workdir,
    )

    report = evaluate_examples(examples, predictions=predictions, label=args.label)
    micro = report["micro"]
    categories = report.get("error_categories", {})
    boundary = sum(
        int(categories.get(k, 0)) for k in ("left_boundary", "right_boundary", "both_boundary")
    )
    by_type_counts: dict[str, int] = {}
    for mentions in predictions.values():
        for mention in mentions:
            by_type_counts[mention.entity_type] = by_type_counts.get(mention.entity_type, 0) + 1

    current = {
        "f1": micro["f1"],
        "precision": micro["precision"],
        "recall": micro["recall"],
        "by_type": report.get("by_type", {}),
        "boundary_errors": boundary,
        "offset_violations": offset_violations,
        "predictions_by_type": by_type_counts,
        "total_parameters": meta["total_parameters"],
    }
    gate = apply_gate(current, confirmation_f1=args.confirmation_f1)

    print(
        f"\nmicro span+type  P={micro['precision']:.4f} R={micro['recall']:.4f} "
        f"F1={micro['f1']:.4f}  TP={micro['true_positive']} FP={micro['false_positive']} "
        f"FN={micro['false_negative']}",
        flush=True,
    )
    for name, values in sorted(report.get("by_type", {}).items()):
        print(
            f"  {name:<12} P={values['precision']:.4f} R={values['recall']:.4f} "
            f"F1={values['f1']:.4f}"
        )
    print(f"error categories: {json.dumps(categories, sort_keys=True)}")
    print(
        f"predictions {sum(len(v) for v in predictions.values())}  "
        f"empty documents {sum(1 for v in predictions.values() if not v)}  "
        f"offset violations {offset_violations}"
    )
    print(f"\n--- ACCEPTANCE GATE ({args.label}) ---")
    for entry in gate["criteria"]:
        print(
            f"  [{'PASS' if entry['passed'] else 'FAIL'}] {entry['criterion']:<42} "
            f"{entry['observed']}"
        )
    print(f"  => {'PASSED' if gate['passed'] else 'NOT ACCEPTED'}")

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_meta": meta,
        "validation_split_sha256": splits["validation"].sha256,
        "validation_examples": len(examples),
        "decision_threshold": args.threshold,
        "micro": micro,
        "by_type": report.get("by_type", {}),
        "error_categories": categories,
        "boundary_errors": boundary,
        "prediction_count": sum(len(v) for v in predictions.values()),
        "predictions_by_type": by_type_counts,
        "empty_documents": sum(1 for v in predictions.values() if not v),
        "offset_violations": offset_violations,
        "total_parameters": meta["total_parameters"],
        "inference_seconds": round(elapsed, 1),
        "peak_rss_bytes": peak_vram,
        "gate": gate,
        "baseline": BASELINE,
    }
    (args.output / f"{args.label}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
