#!/usr/bin/env bash
set -euo pipefail

# Combined full-eval entrypoint for both vanilla and slice runs.
# Keep the run lines explicit so each execution can be customized independently.
#
# Usage:
#   bash scripts/full_eval_both.sh
#   GPU=0 RANK=64 RUN_PREFIX=both RUN_SUFFIX=dev01 bash scripts/full_eval_both.sh
#   bash scripts/full_eval_both.sh --resume
#
# Any extra CLI args you pass to this script are forwarded to orchestrator.

GPU="${GPU:-1}"
RANK="${RANK:-64}"
GENERAL_EVAL_SET="${GENERAL_EVAL_SET:-core}"
EVAL_SIZE="${EVAL_SIZE:-10}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-5}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-32}"

# Used to build per-sequence run names.
VANILLA_RUN_PREFIX="${VANILLA_RUN_PREFIX:-vanilla}"
SLICE_RUN_PREFIX="${SLICE_RUN_PREFIX:-slice}"
RUN_SUFFIX="${RUN_SUFFIX:-full_eval}"

# Slice defaults (override via env if needed)
SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-slice_cache}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-64}"
SLICE_GRAD_PROJECTION_MODE="${SLICE_GRAD_PROJECTION_MODE:-global}"
SLICE_RETAIN_TASKS_PER_STEP="${SLICE_RETAIN_TASKS_PER_STEP:-1}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

EXTRA_ARGS=("$@")

run_vanilla_sequence() {
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
            --slice-max-steps 2 \
            --general-eval-strategy final_only \
            --seen-eval-strategy diagonal_final \
			"${EXTRA_ARGS[@]}"
}

run_slice_sequence() {
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
			--slice-init \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--slice-grad-project \
			--slice-grad-projection-mode "${SLICE_GRAD_PROJECTION_MODE}" \
			--slice-retain-batch-size-set each_task \
			--slice-max-steps 2 \
			--slice-grad-project \
			--slice-retain-batch-size-set each_task \
            --general-eval-strategy final_only \
            --seen-eval-strategy diagonal_final \
			"${EXTRA_ARGS[@]}"
}

# Vanilla executions
run_vanilla_sequence "NI-Seq-G1" "${VANILLA_RUN_PREFIX}_ni_seq_g1_${RUN_SUFFIX}"
run_vanilla_sequence "NI-Seq-G2" "${VANILLA_RUN_PREFIX}_ni_seq_g2_${RUN_SUFFIX}"
run_vanilla_sequence "TRACE" "debug_${VANILLA_RUN_PREFIX}_trace_r${RANK}_${RUN_SUFFIX}"

# Slice executions
run_slice_sequence "NI-Seq-G1" "full_eval_${SLICE_RUN_PREFIX}_ni_seq_g1_${RUN_SUFFIX}"
# run_slice_sequence "NI-Seq-G2" "each_task3_${SLICE_RUN_PREFIX}_ni_seq_g2_${RUN_SUFFIX}"
run_slice_sequence "TRACE" "full_eval_${SLICE_RUN_PREFIX}_trace_r${RANK}_${RUN_SUFFIX}"
