#!/usr/bin/env bash
set -euo pipefail

# Dev smoke run for WinoBias fairness benchmark.
# WinoBias (Zhao et al., 2018): gender-occupation pronoun coreference.
# ~3168 sentences split into pro/anti-stereotyped groups, no external data required.
#
# Usage:
#   bash scripts/dummy_fairness_winobias.sh
#   GPU=0 METHOD=slice bash scripts/dummy_fairness_winobias.sh

GPU="${GPU:-0}"
METHOD="${METHOD:-vanilla}"
RUN_NAME="${RUN_NAME:-dummy_fairness_winobias_dev01}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fairness_dev}"
SEED="${SEED:-42}"
RANK="${RANK:-16}"
EVAL_SIZE="${EVAL_SIZE:-32}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-32}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.03}"
TRAIN_BS="${TRAIN_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
TASK_EVAL_MAX_INPUT_LENGTH="${TASK_EVAL_MAX_INPUT_LENGTH:-256}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-16}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m cl_lora.fairness.benchmark \
  --task winobias \
  --method "${METHOD}" \
  --run-name "${RUN_NAME}" \
  --output-root "${OUTPUT_ROOT}" \
  --seed "${SEED}" \
  --rank "${RANK}" \
  --eval-size "${EVAL_SIZE}" \
  --task-eval-samples "${TASK_EVAL_SAMPLES}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --per-device-train-batch-size "${TRAIN_BS}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --task-eval-max-input-length "${TASK_EVAL_MAX_INPUT_LENGTH}" \
  --task-eval-max-new-tokens "${TASK_EVAL_MAX_NEW_TOKENS}" \
  --save-steps 1000000 \
  --eval-steps 1000000 \
  "$@"
