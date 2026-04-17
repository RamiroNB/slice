#!/usr/bin/env bash
set -euo pipefail

# Full-eval entrypoint for LoRA-GA and LoRAM baselines.
#
# LoRA-GA: SVD on forget gradients only (retain tasks ignored).
# LoRAM:   DST-based initialization, no gradient computation.
#
# Usage:
#   bash scripts/full_eval_baselines.sh
#   GPU=0 RANK=64 RUN_SUFFIX=dev01 bash scripts/full_eval_baselines.sh
#   bash scripts/full_eval_baselines.sh --resume
#
# Any extra CLI args passed to this script are forwarded to orchestrator.

GPU="${GPU:-1}"
RANK="${RANK:-64}"
GENERAL_EVAL_SET="${GENERAL_EVAL_SET:-core}"
EVAL_SIZE="${EVAL_SIZE:-20}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-16}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-32}"

LORA_GA_RUN_PREFIX="${LORA_GA_RUN_PREFIX:-lora_ga}"
SLICE_RUN_PREFIX="${SLICE_RUN_PREFIX:-slice}"
LORAM_RUN_PREFIX="${LORAM_RUN_PREFIX:-loram}"
RUN_SUFFIX="${RUN_SUFFIX:-full_eval}"

SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-slice_cache}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-64}"
SLICE_GRAD_PROJECTION_MODE="${SLICE_GRAD_PROJECTION_MODE:-global}"
SLICE_GRAD_PROJECT_ALWAYS="${SLICE_GRAD_PROJECT_ALWAYS:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

EXTRA_ARGS=("$@")

run_lora_ga_sequence() {
	local sequence_name="$1"
	local run_name="$2"

	echo "============================================================"
	echo "Baseline  : LoRA-GA"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}"
	echo "Rank      : ${RANK}"
	echo "Extra     : ${EXTRA_ARGS[*]:-(none)}"
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
			--slice-init-method lora_ga \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--general-eval-strategy final_only \
			--seen-eval-strategy diagonal_final \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

run_slice_sequence() {
	local sequence_name="$1"
	local run_name="$2"
	local ogd_always_flag=()
	if [[ "${SLICE_GRAD_PROJECT_ALWAYS}" == "1" ]]; then
		ogd_always_flag=(--slice-grad-project-always)
	fi

	echo "============================================================"
	echo "Baseline  : Slice"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}"
	echo "Rank      : ${RANK}"
	echo "Extra     : ${EXTRA_ARGS[*]:-(none)}"
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
			--slice-init-method slice \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--slice-grad-project \
			--slice-grad-projection-mode "${SLICE_GRAD_PROJECTION_MODE}" \
			"${ogd_always_flag[@]}" \
			--slice-retain-batch-size-set each_task \
			--general-eval-strategy final_only \
			--seen-eval-strategy diagonal_final \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

run_loram_sequence() {
	local sequence_name="$1"
	local run_name="$2"

	echo "============================================================"
	echo "Baseline  : LoRAM"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}"
	echo "Rank      : ${RANK}"
	echo "Extra     : ${EXTRA_ARGS[*]:-(none)}"
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
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--slice-init-method loram \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--general-eval-strategy final_only \
			--seen-eval-strategy diagonal_final \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

# LoRAM executions
# run_loram_sequence "NI-Seq-G1" "${LORAM_RUN_PREFIX}_ni_seq_g1_${RUN_SUFFIX}"
# run_loram_sequence "NI-Seq-G2" "${LORAM_RUN_PREFIX}_ni_seq_g2_${RUN_SUFFIX}"
# run_loram_sequence "TRACE"     "${LORAM_RUN_PREFIX}_trace_r${RANK}_${RUN_SUFFIX}"

# LoRA-GA executions
run_lora_ga_sequence "NI-Seq-G2" "fix_${LORA_GA_RUN_PREFIX}_ni_seq_g2_${RUN_SUFFIX}"
run_slice_sequence   "NI-Seq-G2" "${SLICE_RUN_PREFIX}_ni_seq_g2_${RUN_SUFFIX}"

run_lora_ga_sequence "TRACE"     "fix_${LORA_GA_RUN_PREFIX}_trace_r${RANK}_${RUN_SUFFIX}"
run_slice_sequence   "TRACE" "${SLICE_RUN_PREFIX}_trace_r${RANK}_${RUN_SUFFIX}"

run_lora_ga_sequence "NI-Seq-G1" "fix_${LORA_GA_RUN_PREFIX}_ni_seq_g1_${RUN_SUFFIX}"
run_slice_sequence   "NI-Seq-G1" "${SLICE_RUN_PREFIX}_ni_seq_g1_${RUN_SUFFIX}"

