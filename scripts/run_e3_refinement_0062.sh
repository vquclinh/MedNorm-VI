#!/usr/bin/env bash
# Controlled E3 refinement comparison (Audit 0062, Milestone 4C).
#
# Runs the predefined recipes and evaluates each one against the governed acceptance
# gate. There is no hyperparameter search here: R1 and R2 differ by exactly one scalar
# (focal_alpha), so any difference between them attributes to that scalar.
#
#   R0  the current checkpoint, no training (control)
#   R1  low-LR continued fine-tuning, loss unchanged        (alpha 0.25)
#   R2  R1 + class-balanced focal                           (alpha 0.75)
#   R3  conditional midpoint, run only on R1/R2 evidence    (alpha 0.50)
#
# EVIDENCE for R2 (Audit 0062 §4): the reproduced baseline makes 269 too-NARROW vs 124
# too-WIDE boundary errors and never proposes 534 of the gold entities at all, while
# `alpha_t = alpha*y + (1-alpha)*(1-y)` at alpha=0.25 weights the rare positive class
# (8.26% of supervised tokens) at one third of the negative class. The model under-fires
# and the loss is why.
#
# Usage:
#   bash scripts/run_e3_refinement_0062.sh [recipe ...]     # default: R0 R1 R2 R3
#
# Environment overrides:
#   OUTPUT_ROOT   where checkpoints are written
#   REPORT_DIR    where per-recipe gate reports are written
#   PYTHON        interpreter (default .venv/bin/python, else python3)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
CONFIG="${CONFIG:-configs/training/e3_boundary_refinement_0062.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoint/experiments/0062_e3_boundary_refinement}"
REPORT_DIR="${REPORT_DIR:-runs/diagnostics/0062_recipes}"
CONFIRM="RUN E3 BOUNDARY REFINEMENT"
SOURCE_CHECKPOINT="checkpoint/s1_mention_full_training_v1/best.pt"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

RECIPES=("$@")
if [ ${#RECIPES[@]} -eq 0 ]; then RECIPES=(R0 R1 R2 R3); fi

mkdir -p "$OUTPUT_ROOT" "$REPORT_DIR"

train() {  # train <run_id> <extra args...>
  local run_id="$1"; shift
  echo "=== TRAIN ${run_id} ==="
  "$PYTHON" scripts/train_e3_boundary_refinement_0062.py \
    --config "$CONFIG" --run-id "$run_id" --output-root "$OUTPUT_ROOT" \
    --confirm "$CONFIRM" "$@"
}

evaluate() {  # evaluate <label> <checkpoint> [extra args...]
  local label="$1"; local checkpoint="$2"; shift 2
  echo "=== EVALUATE ${label} ==="
  "$PYTHON" scripts/evaluate_e3_checkpoint_0062.py \
    --checkpoint "$checkpoint" --label "$label" --output "$REPORT_DIR" "$@"
}

for recipe in "${RECIPES[@]}"; do
  case "$recipe" in
    R0)
      # Control: no training. Establishes that the evaluator reproduces the baseline.
      evaluate R0_current_checkpoint "$SOURCE_CHECKPOINT"
      ;;
    R1)
      train R1_lowlr
      evaluate R1_lowlr "${OUTPUT_ROOT}/R1_lowlr/best.pt"
      ;;
    R2)
      train R2_class_balanced --focal-alpha 0.75
      evaluate R2_class_balanced "${OUTPUT_ROOT}/R2_class_balanced/best.pt"
      ;;
    R3)
      # Conditional recipe. R1 (alpha 0.25) and R2 (alpha 0.75) bracket the axis:
      # R1 passes the gate but still misses 408 entities, and R2 proves those are
      # recoverable (missed 537 -> 169) at a precision cost the gate forbids. R3 is the
      # single predefined midpoint between two measured points, not a search.
      train R3_alpha050 --focal-alpha 0.50
      evaluate R3_alpha050 "${OUTPUT_ROOT}/R3_alpha050/best.pt"
      ;;
    CONFIRM)
      # Only for the SELECTED recipe: a second seed, per brief §5 and §8.
      # ALPHA and RUN_ID must name the selected recipe.
      train "${RUN_ID:?set RUN_ID}_seed2" --focal-alpha "${ALPHA:?set ALPHA}" --seed 20260801
      evaluate "${RUN_ID}_seed2" "${OUTPUT_ROOT}/${RUN_ID}_seed2/best.pt"
      ;;
    *)
      echo "unknown recipe: ${recipe}" >&2
      exit 2
      ;;
  esac
done

echo
echo "=== SUMMARY ==="
"$PYTHON" - "$REPORT_DIR" <<'PY'
import json, sys
from pathlib import Path

reports = sorted(Path(sys.argv[1]).glob("*.json"))
if not reports:
    print("no reports")
    raise SystemExit(0)
head = f"{'recipe':<28}{'P':>8}{'R':>8}{'F1':>8}{'dF1':>8}{'bound':>7}{'gate':>14}"
print(head)
print("-" * len(head))
for path in reports:
    payload = json.loads(path.read_text(encoding="utf-8"))
    micro = payload["micro"]
    baseline = payload["baseline"]["f1"]
    print(f"{payload['label']:<28}{micro['precision']:>8.4f}{micro['recall']:>8.4f}"
          f"{micro['f1']:>8.4f}{(micro['f1'] - baseline) * 100:>+8.2f}"
          f"{payload['boundary_errors']:>7}"
          f"{('PASSED' if payload['gate']['passed'] else 'NOT ACCEPTED'):>14}")
PY
