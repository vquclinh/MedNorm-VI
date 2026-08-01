#!/usr/bin/env python3
"""Evaluate an E3 checkpoint against the Milestone-4E acceptance gate (Audit 0064).

Extends the Audit-0062 evaluator with what 4E judges on and 0062 did not measure:
**per-source** metrics and a **worst-source** floor. Audit 0062 §4.6 recorded that the
governed validation split is not iid with training - 86.7% `vimedner` against 24.4% in
training - and a single micro-F1 can improve while the weakest source silently degrades.
The gate now refuses that trade.

Everything else is deliberately identical to the 0062 evaluator: the same L1->L4 path with
the candidate injected through governed E3 settings, the same `evaluation.exact_mention`
scorer, the same production decoder. A recipe comparison is only meaningful when the
measuring instrument does not move.

The gate is not parameterised. A gate a caller can loosen at the command line is not a
gate.

Usage::

    python scripts/evaluate_e3_checkpoint_0064.py \
        --checkpoint checkpoint/experiments/0064_.../R4/best.pt \
        --label R4_boundary --output runs/diagnostics/0064_recipes
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

# The Audit-0062 ACTIVE checkpoint, reproduced exactly in Audit 0064 §3 through this
# same path. Stored as confusion counts, not rounded rates: comparing full-precision
# candidates against 4-decimal constants makes the control fail its own gate by -0.00.
BASELINE_COUNTS = {
    "micro": (1408, 743, 583),
    "DIAGNOSIS": (952, 394, 350),
    "SYMPTOM": (351, 304, 182),
    "MEDICATION": (105, 28, 51),
}
BASELINE_BOUNDARY_ERRORS = 298
# Worst-source F1 of the ACTIVE baseline, measured in Audit 0064 §3 (vimedner, the
# source that is 86.7% of validation and only 24.4% of training).
BASELINE_WORST_SOURCE_F1 = 0.6765316718587747

# Brief §6, verbatim. Absolute points.
GATE_MIN_F1_GAIN = 2.0
GATE_MAX_PRECISION_DROP = 0.5
GATE_MAX_MEDICATION_DROP = 1.0
GATE_MAX_WORST_SOURCE_DROP = 1.0
GATE_CONFIRMATION_MIN_F1_GAIN = 1.5
MAX_DEPLOYMENT_PARAMETERS = 9_000_000_000


def _rates(counts: tuple[int, int, int]) -> tuple[float, float, float]:
    tp, fp, fn = counts
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint(checkpoint: Path, torch: Any) -> dict[str, Any]:
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
    limit: int = 0,
) -> tuple[dict[str, list[Mention]], int, float, int]:
    """L1->L4 with this checkpoint injected through the governed E3 settings."""
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
    subset = examples[:limit] if limit else examples
    for index, example in enumerate(subset, start=1):
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
        if index % 250 == 0 or index == len(subset):
            print(f"  [{index}/{len(subset)}] {time.time() - started:.0f}s", flush=True)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return predictions, offset_violations, time.time() - started, peak_rss


def per_source_metrics(
    examples: list[dict], predictions: dict[str, list[Mention]]
) -> dict[str, dict[str, float]]:
    """Micro span+type P/R/F1 for each governed source dataset.

    Audit 0062 §4.6: validation is dominated by one source. A single micro figure can rise
    while the weakest source falls, and 4E's gate refuses that trade - so it has to be
    measured, not assumed.
    """
    by_source: dict[str, list[dict]] = {}
    for example in examples:
        by_source.setdefault(str(example.get("source_dataset", "")), []).append(example)
    out: dict[str, dict[str, float]] = {}
    for source, rows in sorted(by_source.items()):
        scoped = {str(r["example_id"]): predictions.get(str(r["example_id"]), []) for r in rows}
        report = evaluate_examples(rows, predictions=scoped, label=f"source:{source}")
        micro = report["micro"]
        out[source] = {
            "examples": len(rows),
            "gold": micro["true_positive"] + micro["false_negative"],
            "precision": micro["precision"],
            "recall": micro["recall"],
            "f1": micro["f1"],
            "true_positive": micro["true_positive"],
            "false_positive": micro["false_positive"],
            "false_negative": micro["false_negative"],
        }
    return out


def span_only_metrics(
    examples: list[dict], predictions: dict[str, list[Mention]]
) -> dict[str, float]:
    """Exact character-span metrics IGNORING type, so typing error is separable."""
    tp = fp = fn = 0
    for example in examples:
        gold = {(int(e["start"]), int(e["end"])) for e in example.get("entities", [])}
        pred = {(m.start, m.end) for m in predictions.get(str(example["example_id"]), [])}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision, recall, f1 = _rates((tp, fp, fn))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def length_bucket_metrics(
    examples: list[dict], predictions: dict[str, list[Mention]]
) -> dict[str, dict[str, int]]:
    """Exact span+type hit rate by gold length, so short and long spans are visible."""
    buckets: dict[str, dict[str, int]] = {}
    for example in examples:
        pred = {
            (m.start, m.end, m.entity_type) for m in predictions.get(str(example["example_id"]), [])
        }
        for entity in example.get("entities", []):
            words = len(str(entity.get("text", "")).split())
            name = "1" if words == 1 else ("2-3" if words <= 3 else ("4-7" if words <= 7 else "8+"))
            slot = buckets.setdefault(name, {"gold": 0, "exact": 0})
            slot["gold"] += 1
            key = (int(entity["start"]), int(entity["end"]), str(entity["target_type"]))
            if key in pred:
                slot["exact"] += 1
    return buckets


def apply_gate(current: dict[str, Any], *, confirmation_f1: float | None) -> dict[str, Any]:
    """The brief §6 gate, at the thresholds the brief specifies."""
    f1 = current["f1"] * 100
    precision = current["precision"] * 100
    recall = current["recall"] * 100
    by_type = current["by_type"]

    def type_f1(name: str) -> float:
        return by_type.get(name, {}).get("f1", 0.0) * 100

    worst_now = current["worst_source_f1"] * 100
    worst_base = current["baseline_worst_source_f1"] * 100

    checks = [
        (
            "exact span+type micro-F1 >= +2.0",
            f1 - BASELINE["f1"] * 100 >= GATE_MIN_F1_GAIN,
            f"{f1 - BASELINE['f1'] * 100:+.2f}",
        ),
        (
            "precision drop <= 0.5",
            precision - BASELINE["precision"] * 100 >= -GATE_MAX_PRECISION_DROP,
            f"{precision - BASELINE['precision'] * 100:+.2f}",
        ),
        (
            "recall must not decrease",
            recall >= BASELINE["recall"] * 100,
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
            "total boundary errors < 298",
            current["boundary_errors"] < BASELINE["boundary_errors"],
            f"{current['boundary_errors']} vs {BASELINE['boundary_errors']}",
        ),
        (
            "worst-source F1 drop <= 1.0",
            worst_now - worst_base >= -GATE_MAX_WORST_SOURCE_DROP,
            f"{worst_now - worst_base:+.2f} ({current['worst_source']})",
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
                "confirmation seed >= +1.5 F1",
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
        "--output", type=Path, default=REPO / "runs" / "diagnostics" / "0064_recipes"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--confirmation-f1", type=float, default=None)
    parser.add_argument(
        "--repeat-check",
        type=int,
        default=120,
        help="documents re-run to prove deterministic repeat agreement",
    )
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/mednorm_0064_eval"))
    parser.add_argument(
        "--baseline-worst-source-f1",
        type=float,
        default=None,
        help="override the recorded ACTIVE-baseline worst-source F1 (diagnostic only)",
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
    load_coverage(corpus_dir)
    examples = list(iter_jsonl(splits["validation"].path))
    print(
        f"validation {splits['validation'].path} sha256={splits['validation'].sha256} "
        f"({len(examples)} examples)",
        flush=True,
    )

    meta = inspect_checkpoint(args.checkpoint, torch)
    predictions, offset_violations, elapsed, peak_rss = run_pipeline(
        args.checkpoint,
        digest,
        examples,
        threshold=args.threshold,
        batch_size=args.batch_size,
        workdir=args.workdir,
    )

    # Deterministic repeat agreement over a bounded prefix.
    repeat_agreement = None
    if args.repeat_check > 0:
        again, _v, _t, _r = run_pipeline(
            args.checkpoint,
            digest,
            examples,
            threshold=args.threshold,
            batch_size=args.batch_size,
            workdir=args.workdir,
            limit=args.repeat_check,
        )
        same = sum(
            1
            for k, v in again.items()
            if [(m.start, m.end, m.entity_type) for m in v]
            == [(m.start, m.end, m.entity_type) for m in predictions.get(k, [])]
        )
        repeat_agreement = {
            "documents": len(again),
            "identical": same,
            "agreement": same / len(again) if again else 0.0,
        }
        print(f"deterministic repeat agreement: {same}/{len(again)}", flush=True)

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

    sources = per_source_metrics(examples, predictions)
    worst_source = min(sources, key=lambda s: sources[s]["f1"]) if sources else ""
    worst_f1 = sources[worst_source]["f1"] if worst_source else 0.0
    spans = span_only_metrics(examples, predictions)
    lengths = length_bucket_metrics(examples, predictions)

    current = {
        "f1": micro["f1"],
        "precision": micro["precision"],
        "recall": micro["recall"],
        "by_type": report.get("by_type", {}),
        "boundary_errors": boundary,
        "offset_violations": offset_violations,
        "predictions_by_type": by_type_counts,
        "total_parameters": meta["total_parameters"],
        "worst_source": worst_source,
        "worst_source_f1": worst_f1,
        "baseline_worst_source_f1": (
            args.baseline_worst_source_f1 if args.baseline_worst_source_f1 is not None else worst_f1
        ),
    }
    gate = apply_gate(current, confirmation_f1=args.confirmation_f1)

    print(
        f"\nexact SPAN ONLY  P={spans['precision']:.4f} R={spans['recall']:.4f} "
        f"F1={spans['f1']:.4f}"
    )
    print(
        f"micro span+type  P={micro['precision']:.4f} R={micro['recall']:.4f} "
        f"F1={micro['f1']:.4f}  TP={micro['true_positive']} FP={micro['false_positive']} "
        f"FN={micro['false_negative']}"
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
    print("\nper-source micro span+type")
    for source, values in sources.items():
        flag = "  <- worst" if source == worst_source else ""
        print(
            f"  {source:<16} n={values['examples']:<5} gold={values['gold']:<5} "
            f"P={values['precision']:.4f} R={values['recall']:.4f} "
            f"F1={values['f1']:.4f}{flag}"
        )
    print("\nexact rate by gold length")
    for name in ("1", "2-3", "4-7", "8+"):
        if name in lengths:
            slot = lengths[name]
            print(
                f"  {name:<5} gold={slot['gold']:<6} exact={slot['exact']:<6} "
                f"{slot['exact'] / slot['gold'] * 100:5.1f}%"
            )

    print(f"\n--- ACCEPTANCE GATE 4E ({args.label}) ---")
    for entry in gate["criteria"]:
        print(
            f"  [{'PASS' if entry['passed'] else 'FAIL'}] {entry['criterion']:<38} "
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
        "span_only": spans,
        "by_type": report.get("by_type", {}),
        "per_source": sources,
        "worst_source": worst_source,
        "worst_source_f1": worst_f1,
        "length_buckets": lengths,
        "error_categories": categories,
        "boundary_errors": boundary,
        "prediction_count": sum(len(v) for v in predictions.values()),
        "predictions_by_type": by_type_counts,
        "empty_documents": sum(1 for v in predictions.values() if not v),
        "offset_violations": offset_violations,
        "deterministic_repeat": repeat_agreement,
        "total_parameters": meta["total_parameters"],
        "inference_seconds": round(elapsed, 1),
        "peak_rss_bytes": peak_rss,
        "gate": gate,
        "baseline": BASELINE,
    }
    (args.output / f"{args.label}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Per-example predictions, so every downstream error analysis is offline and no
    # recipe is ever re-run just to slice its results a different way.
    (args.output / f"{args.label}.predictions.json").write_text(
        json.dumps(
            {
                eid: [[m.start, m.end, m.entity_type] for m in mentions]
                for eid, mentions in sorted(predictions.items())
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
