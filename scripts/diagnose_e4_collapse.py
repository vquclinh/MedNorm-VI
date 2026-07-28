#!/usr/bin/env python3
"""Diagnose why the completed E4 full run predicts zero mentions (Audit 0043).

Usage::

    env PYTHONPATH=src python scripts/diagnose_e4_collapse.py \
        --artifact-dir local-artifacts/e4_phobert_w2ner_full_v1

Read-only. It never trains, never loads PhoBERT, never opens internal_test, and
never writes into the artifact directory. Use ``--out`` to save the JSON report
somewhere outside the artifact (and outside Git).

The checkpoint probe needs ``checkpoints/best.pt`` and ``checkpoints/latest.pt``
inside the artifact directory. When they are absent the probe is reported as
blocked with the exact evidence it would have produced, and the verdict stays
``ROOT_CAUSE_NOT_YET_PROVEN`` rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mednorm_vi.training.phase2.e4_collapse_diagnosis import (  # noqa: E402
    E4DiagnosisError,
    run_collapse_diagnosis,
)

DEFAULT_ARTIFACT_DIR = "local-artifacts/e4_phobert_w2ner_full_v1"
DEFAULT_SPLIT_ROOT = "data/derived/training_corpora/mednorm_vi_training_v1/splits"


def _render(diagnosis) -> str:  # noqa: ANN001 - CollapseDiagnosis, kept local
    lines: list[str] = []
    integrity = diagnosis.integrity
    lines.append("== A. artifact integrity ==")
    lines.append(f"  artifact_dir            {integrity.artifact_dir}")
    lines.append(f"  consistent              {integrity.ok}")
    lines.append(f"  checkpoints_present     {integrity.checkpoints_present}")
    for name in sorted(integrity.present_files):
        lines.append(f"  present   {name:34s} {integrity.present_files[name]}")
    for name in integrity.missing_files:
        lines.append(f"  MISSING   {name}")
    for problem in integrity.inconsistencies:
        lines.append(f"  INCONSISTENT  {problem}")

    lines.append("")
    lines.append("== B. epoch history ==")
    header = ("  epoch  train_loss  pred  gold    tp    fp    fn   "
              "precision     recall         f1  best")
    lines.append(header)
    for row in diagnosis.history.rows:
        lines.append(
            f"  {row.epoch:5} {_num(row.mean_training_loss, 11, 8)}"
            f" {_num(row.predicted_mentions, 5)} {_num(row.gold_mentions, 5)}"
            f" {_num(row.true_positives, 5)} {_num(row.false_positives, 5)}"
            f" {_num(row.false_negatives, 5)} {_num(row.exact_precision, 11, 8)}"
            f" {_num(row.exact_recall, 10, 8)} {_num(row.exact_f1, 10, 8)}"
            f"  {row.is_new_best}")
    lines.append(
        f"  peak predicted {diagnosis.history.peak_predicted_mentions} at epoch "
        f"{diagnosis.history.peak_prediction_epoch}; "
        f"first zero-prediction epoch after the peak "
        f"{diagnosis.history.first_epoch_with_zero_predictions}; "
        f"final {diagnosis.history.final_predicted_mentions}")
    if diagnosis.history.unavailable_fields:
        lines.append(
            f"  UNAVAILABLE fields: {', '.join(diagnosis.history.unavailable_fields)}")

    lines.append("")
    lines.append("== C. gold-grid round-trip (no model) ==")
    for report in diagnosis.round_trips:
        lines.append(
            f"  {report.split:11s} examples={report.examples_checked} "
            f"gold={report.gold_mentions} reconstructed={report.reconstructed_mentions} "
            f"tp={report.true_positives} fp={report.false_positives} "
            f"fn={report.false_negatives}")
        lines.append(
            f"              P={report.exact_precision:.6f} R={report.exact_recall:.6f} "
            f"F1={report.exact_f1:.6f} passes={report.passes}")
        if report.failures_by_entity_type:
            lines.append(f"              failures {report.failures_by_entity_type}")

    lines.append("")
    lines.append("== D. grid class distribution ==")
    for report in diagnosis.distributions:
        lines.append(
            f"  {report.split:11s} valid_cells={report.valid_grid_cells} "
            f"background={report.background_cells} positive={report.positive_cells} "
            f"ratio={report.positive_to_background_ratio:.8f}")
        lines.append(f"              cells_by_label {report.cells_by_label}")
        lines.append(
            f"              zero-positive examples "
            f"{report.examples_with_zero_positive_cells}/{report.examples}; "
            f"entities represented {report.entities_represented}")
        lines.append(
            f"              patterns {report.mentions_by_relation_pattern}")
        if report.labels_with_no_training_signal:
            lines.append(
                f"              NO SIGNAL for "
                f"{list(report.labels_with_no_training_signal)}")
        lines.append(
            f"              best constant-predictor loss "
            f"{report.constant_predictor_loss:.9f}")

    if diagnosis.constant_predictor_comparison:
        lines.append("")
        lines.append("== D2. converged loss vs an input-independent predictor ==")
        for key, value in sorted(diagnosis.constant_predictor_comparison.items()):
            lines.append(f"  {key:48s} {value}")

    lines.append("")
    lines.append("== D3. corpus composition in training file order ==")
    for report in diagnosis.compositions:
        lines.append(f"  {report.split}: sources {list(report.sources_in_file_order)}")
        for source in report.sources_in_file_order:
            lines.append(
                f"    {source:16s} examples={report.examples_by_source[source]:6d} "
                f"entities={report.entities_by_source[source]:6d} "
                f"first_row={report.first_row_index_by_source[source]:6d} "
                f"types={report.entity_types_by_source[source]}")
        lines.append(
            f"    longest consecutive zero-entity run "
            f"{report.longest_zero_entity_run} starting at row "
            f"{report.longest_zero_entity_run_start}")

    lines.append("")
    lines.append("== E. checkpoint probe ==")
    if diagnosis.probes:
        for probe in diagnosis.probes:
            lines.append(f"  {probe.role}: {probe.as_dict()}")
    else:
        lines.append("  BLOCKED")
        lines.append(f"  {diagnosis.probe_blocked_reason}")

    lines.append("")
    lines.append("== F. label and loss contract ==")
    lines.append(f"  label order  {list(diagnosis.label_trace.label_order)}")
    lines.append(f"  consistent   {diagnosis.label_trace.consistent}")
    for key, value in sorted(diagnosis.loss_contract.as_dict().items()):
        if key == "notes":
            continue
        lines.append(f"  {key:38s} {value}")
    for note in diagnosis.loss_contract.notes:
        lines.append(f"  note: {note}")

    lines.append("")
    lines.append("== H. verdict ==")
    lines.append(f"  {diagnosis.verdict.verdict}")
    if diagnosis.verdict.supported_hypothesis:
        lines.append(f"  leading hypothesis: {diagnosis.verdict.supported_hypothesis}")
    for item in diagnosis.verdict.ruled_out:
        lines.append(f"  ruled out: {item}")
    for item in diagnosis.verdict.supporting_evidence:
        lines.append(f"  evidence:  {item}")
    for item in diagnosis.verdict.missing_evidence:
        lines.append(f"  MISSING:   {item}")
    return "\n".join(lines)


def _num(value: object, width: int, precision: int = 0) -> str:
    if isinstance(value, float):
        return f"{value:{width}.{precision}f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:{width}d}"
    return f"{str(value):>{width}s}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E4 post-training collapse diagnosis")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="bound the governed rows scanned per split (default: whole split)")
    parser.add_argument("--max-words", type=int, default=256)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--out", default="", help="write the JSON report to this path")
    args = parser.parse_args(argv)

    split_root = Path(args.split_root)
    split_paths = {
        "train": split_root / "train.jsonl",
        "validation": split_root / "validation.jsonl",
    }
    try:
        diagnosis = run_collapse_diagnosis(
            artifact_dir=args.artifact_dir,
            split_paths=split_paths,
            max_words=args.max_words,
            limit=args.limit,
        )
    except E4DiagnosisError as error:
        print(f"E4 collapse diagnosis failed: {error}", file=sys.stderr)
        return 2

    payload = diagnosis.as_dict()
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(_render(diagnosis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
