#!/usr/bin/env bash
set -euo pipefail

# Evaluate the base model (no adapter, no training) on all fairness datasets.
#
# This produces the zero-shot reference point required for every Pareto comparison.
# Run this ONCE per seed before any method comparison.
#
# Usage:
#   bash scripts/run_fairness_baselines.sh
#   GPU=1 SEED=123 EVAL_SIZE=500 MIN_GROUP_COUNT=20 bash scripts/run_fairness_baselines.sh
#
# Output structure:
#   results/fairness/<dataset>/base_model/<run_name>/run_summary.json

GPU="${GPU:-0}"
SEED="${SEED:-42}"
EVAL_SIZE="${EVAL_SIZE:-500}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-${EVAL_SIZE}}"
MIN_GROUP_COUNT="${MIN_GROUP_COUNT:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fairness}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-32}"
TASK_EVAL_MAX_INPUT_LENGTH="${TASK_EVAL_MAX_INPUT_LENGTH:-512}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TASKS=("bbq" "winogender" "difference_awareness")

echo "============================================================"
echo "Fairness base-model evaluation (zero-shot reference)"
echo "GPU              : ${GPU}"
echo "SEED             : ${SEED}"
echo "EVAL_SIZE        : ${EVAL_SIZE}"
echo "MIN_GROUP_COUNT  : ${MIN_GROUP_COUNT}"
echo "OUTPUT_ROOT      : ${OUTPUT_ROOT}"
echo "TASKS            : ${TASKS[*]}"
echo "============================================================"
echo ""

for TASK in "${TASKS[@]}"; do
    RUN_NAME="base_model_s${SEED}_${TIMESTAMP}"
    echo "------------------------------------------------------------"
    echo "Running: ${TASK} | base_model | seed=${SEED}"
    echo "Output : ${OUTPUT_ROOT}/${TASK}/base_model/${RUN_NAME}/run_summary.json"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="${GPU}" python -m cl_lora.fairness_benchmark \
        --task "${TASK}" \
        --base-model-eval \
        --run-name "${RUN_NAME}" \
        --output-root "${OUTPUT_ROOT}" \
        --seed "${SEED}" \
        --eval-size "${EVAL_SIZE}" \
        --task-eval-samples "${TASK_EVAL_SAMPLES}" \
        --task-eval-max-new-tokens "${TASK_EVAL_MAX_NEW_TOKENS}" \
        --task-eval-max-input-length "${TASK_EVAL_MAX_INPUT_LENGTH}" \
        --min-group-count "${MIN_GROUP_COUNT}"

    echo ""
done

echo "============================================================"
echo "All base-model evaluations complete."
echo "Results written to ${OUTPUT_ROOT}/<dataset>/base_model/"
echo "============================================================"
