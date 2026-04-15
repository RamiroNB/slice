#!/usr/bin/env bash
set -euo pipefail

# Standalone fairness-vs-accuracy benchmark runner.
#
# Supported tasks: bbq, winobias, difference_awareness
#
# Usage:
#   bash scripts/run_fairness.sh
#   GPU=0 TASK=bbq METHOD=vanilla RUN_NAME=smoke_b01 bash scripts/run_fairness.sh
#   GPU=0 TASK=difference_awareness METHOD=slice_proj_per_module EVAL_SIZE=500 MIN_GROUP_COUNT=20 bash scripts/run_fairness.sh
#
# Base-model baseline (no training):
#   GPU=0 TASK=bbq BASE_MODEL_EVAL=1 bash scripts/run_fairness.sh

GPU="${GPU:-0}"
TASK="${TASK:-bbq}"
METHOD="${METHOD:-vanilla}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fairness}"
RANK="${RANK:-128}"
SEED="${SEED:-42}"

EVAL_SIZE="${EVAL_SIZE:-500}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-500}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-32}"
MIN_GROUP_COUNT="${MIN_GROUP_COUNT:-20}"
BASE_MODEL_EVAL="${BASE_MODEL_EVAL:-0}"

EXTRA_ARGS=("$@")

CMD=(
  python -m cl_lora.fairness.benchmark
  --task "${TASK}"
  --output-root "${OUTPUT_ROOT}"
  --seed "${SEED}"
  --eval-size "${EVAL_SIZE}"
  --task-eval-samples "${TASK_EVAL_SAMPLES}"
  --task-eval-max-new-tokens "${TASK_EVAL_MAX_NEW_TOKENS}"
  --min-group-count "${MIN_GROUP_COUNT}"
)

if [[ "${BASE_MODEL_EVAL}" == "1" ]]; then
  CMD+=(--base-model-eval)
else
  CMD+=(--method "${METHOD}" --rank "${RANK}")
fi

if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "Fairness benchmark"
echo "TASK            : ${TASK}"
echo "METHOD          : ${BASE_MODEL_EVAL:+base_model}${BASE_MODEL_EVAL:-${METHOD}}"
echo "GPU             : ${GPU}"
echo "RANK            : ${BASE_MODEL_EVAL:+N/A}${BASE_MODEL_EVAL:-${RANK}}"
echo "SEED            : ${SEED}"
echo "MIN_GROUP_COUNT : ${MIN_GROUP_COUNT}"
echo "OUT ROOT        : ${OUTPUT_ROOT}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}"
