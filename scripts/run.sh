#!/usr/bin/env bash
set -euo pipefail

# Runs sequences back-to-back:
#   - NI-Seq-G1
#   - NI-Seq-G2
#   - TRACE
#
# Usage:
#   bash scripts/run.sh
#   GPU=0 RANK=32 RUN_PREFIX=vanilla2 RUN_SUFFIX=dev02 bash scripts/run.sh
#   bash scripts/run.sh --resume
#
# Any extra CLI args you pass to this script are forwarded to orchestrator.

GPU="${GPU:-1}"
RANK="${RANK:-64}"
GENERAL_EVAL_SET="${GENERAL_EVAL_SET:-core}"
EVAL_SIZE="${EVAL_SIZE:-10}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-5}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-32}"

# Used to build a per-sequence run-name that matches your example style.
RUN_PREFIX="${RUN_PREFIX:-vanilla}"
RUN_SUFFIX="${RUN_SUFFIX:-quick_eval}"

EXTRA_ARGS=("$@")

run_sequence() {
  local sequence_name="$1"
  local run_name="$2"

  echo "============================================================"
  echo "Sequence : ${sequence_name}"
  echo "Run name : ${run_name}"
  echo "GPU      : ${GPU}"
  echo "Rank     : ${RANK}"
  echo "Extra    : ${EXTRA_ARGS[*]:-(none)}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" \
    python -m cl_lora.orchestrator \
      --sequence "${sequence_name}" \
      --general-eval-set "${GENERAL_EVAL_SET}" \
      --eval-size "${EVAL_SIZE}" \
      --task-eval-samples "${TASK_EVAL_SAMPLES}" \
      --task-eval-max-new-tokens "${TASK_EVAL_MAX_NEW_TOKENS}" \
      --run-name "${run_name}" \
      --rank "${RANK}" \
      --quick-eval \
      "${EXTRA_ARGS[@]}"
}

run_sequence "NI-Seq-G1" "${RUN_PREFIX}_ni_seq_g1_${RUN_SUFFIX}"
run_sequence "NI-Seq-G2" "${RUN_PREFIX}_ni_seq_g2_${RUN_SUFFIX}"
run_sequence "TRACE" "debug_${RUN_PREFIX}_trace_r${RANK}_${RUN_SUFFIX}"
