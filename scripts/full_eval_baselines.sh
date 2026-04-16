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
EVAL_SIZE="${EVAL_SIZE:-10}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-5}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-32}"

LORA_GA_RUN_PREFIX="${LORA_GA_RUN_PREFIX:-lora_ga}"
LORAM_RUN_PREFIX="${LORAM_RUN_PREFIX:-loram}"
RUN_SUFFIX="${RUN_SUFFIX:-full_eval}"

SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-slice_cache}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-64}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

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
            --slice-max-steps 4 \
			--slice-init-method lora_ga \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
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
            --slice-max-steps 4 \
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
run_lora_ga_sequence "TRACE"     "fix_${LORA_GA_RUN_PREFIX}_trace_r${RANK}_${RUN_SUFFIX}"
run_lora_ga_sequence "NI-Seq-G1" "fix_${LORA_GA_RUN_PREFIX}_ni_seq_g1_${RUN_SUFFIX}"
