#!/usr/bin/env bash
# Controlled E3 source-balanced / boundary-weighted comparison (Audit 0064, Milestone 4E).
#
# Every recipe continues the ACTIVE Audit-0062 checkpoint (scored 11.9188) and differs by
# exactly one switch, so any difference attributes cleanly:
#
#   R0   the active checkpoint, no training (control)
#   R4a  boundary-token reweighting, weight 2.0
#   R4b  boundary-token reweighting, weight 3.0
#   R5   source-balanced sampling (uniform by governed source)
#   R6   R4 best + R5, run ONLY if each showed a beneficial, interpretable effect
#
# EVIDENCE (Audit 0064 §3): 298 boundary errors remain with right outnumbering left 2:1,
# and the sources are type-disjoint - vimedner carries all DIAGNOSIS/SYMPTOM supervision
# at 24.4% of training while those types are 92.2% of validation entities.
#
# Usage:
#   bash scripts/run_e3_source_boundary_0064.sh [recipe ...]     # default: R0 R4a R4b R5
#   BOUNDARY_WEIGHT=2.0 bash scripts/run_e3_source_boundary_0064.sh R6
#   RUN_ID=R6_combined ALPHA_ARGS="--boundary-weight 2.0 --source-balanced" \
#     bash scripts/run_e3_source_boundary_0064.sh CONFIRM

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
CONFIG="${CONFIG:-configs/training/e3_source_boundary_0064.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoint/experiments/0064_e3_source_boundary_refinement}"
REPORT_DIR="${REPORT_DIR:-runs/diagnostics/0064_recipes}"
CONFIRM_PHRASE="RUN E3 BOUNDARY REFINEMENT"
ACTIVE_CHECKPOINT="checkpoint/e3_boundary_refinement_0062/best.pt"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-2.0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

RECIPES=("$@")
if [ ${#RECIPES[@]} -eq 0 ]; then RECIPES=(R0 R4a R4b R5); fi

mkdir -p "$OUTPUT_ROOT" "$REPORT_DIR"

train() {  # train <run_id> <extra args...>
  local run_id="$1"; shift
  echo "=== TRAIN ${run_id} ==="
  "$PYTHON" scripts/train_e3_source_boundary_0064.py \
    --config "$CONFIG" --run-id "$run_id" --output-root "$OUTPUT_ROOT" \
    --confirm "$CONFIRM_PHRASE" "$@"
}

evaluate() {  # evaluate <label> <checkpoint> [extra args...]
  local label="$1"; local checkpoint="$2"; shift 2
  echo "=== EVALUATE ${label} ==="
  "$PYTHON" scripts/evaluate_e3_checkpoint_0064.py \
    --checkpoint "$checkpoint" --label "$label" --output "$REPORT_DIR" "$@"
}

for recipe in "${RECIPES[@]}"; do
  case "$recipe" in
    R0)
      evaluate R0_active_baseline "$ACTIVE_CHECKPOINT"
      ;;
    R4a)
      train R4a_boundary_w2 --boundary-weight 2.0
      evaluate R4a_boundary_w2 "${OUTPUT_ROOT}/R4a_boundary_w2/best.pt"
      ;;
    R4b)
      train R4b_boundary_w3 --boundary-weight 3.0
      evaluate R4b_boundary_w3 "${OUTPUT_ROOT}/R4b_boundary_w3/best.pt"
      ;;
    R5)
      train R5_source_balanced --source-balanced
      evaluate R5_source_balanced "${OUTPUT_ROOT}/R5_source_balanced/best.pt"
      ;;
    R6)
      # Conditional. Runs only when R4 and R5 each showed a beneficial and interpretable
      # effect; BOUNDARY_WEIGHT names the R4 arm that won.
      train R6_combined --boundary-weight "$BOUNDARY_WEIGHT" --source-balanced
      evaluate R6_combined "${OUTPUT_ROOT}/R6_combined/best.pt"
      ;;
    CONFIRM)
      # Independent confirmation seed, for the SELECTED recipe only (brief §4, §6).
      train "${RUN_ID:?set RUN_ID}_seed2" ${ALPHA_ARGS:-} --seed 20260802
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
import json
import sys
from pathlib import Path

reports = sorted(p for p in Path(sys.argv[1]).glob("*.json") if ".predictions" not in p.name)
if not reports:
    print("no reports")
    raise SystemExit(0)
head = (f"{'recipe':<26}{'P':>8}{'R':>8}{'F1':>8}{'dF1':>7}"
        f"{'bound':>7}{'worst':>8}{'gate':>14}")
print(head)
print("-" * len(head))
for path in reports:
    payload = json.loads(path.read_text(encoding="utf-8"))
    micro = payload["micro"]
    baseline = payload["baseline"]["f1"]
    print(f"{payload['label']:<26}{micro['precision']:>8.4f}{micro['recall']:>8.4f}"
          f"{micro['f1']:>8.4f}{(micro['f1'] - baseline) * 100:>+7.2f}"
          f"{payload['boundary_errors']:>7}{payload['worst_source_f1']:>8.4f}"
          f"{('PASSED' if payload['gate']['passed'] else 'NOT ACCEPTED'):>14}")
PY
