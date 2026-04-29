#!/usr/bin/env bash
set -euo pipefail

# Composition smoke test: every LoRA initialization × every CL method.
#
# Inits   : lora_vanilla, loram, lora_ga, slice
# CL meth.: vanilla, o_lora, inflora, sapt
# Total   : 4 × 4 = 16 runs per sequence.
#
# Train-only phase: this script runs --train-only (no evaluation). Evals
# are run on a separate machine afterwards from the persisted artifacts.
# Defaults target NI-Seq-G2 (2 stages); override env vars to point at a
# real sequence. Any extra positional args are forwarded to orchestrator
# (so you can pass --resume, etc).
#
# Usage:
#   bash scripts/test_init_x_cl_methods.sh
#   GPU=0 SEQUENCES="NI-Seq-G2" RUN_SUFFIX=smoke01 bash scripts/test_init_x_cl_methods.sh
#   ONLY_INITS="slice lora_ga" ONLY_CL="sapt" bash scripts/test_init_x_cl_methods.sh
#   FAIL_FAST=0 bash scripts/test_init_x_cl_methods.sh   # collect failures, do not stop

GPU="${GPU:-0}"
RANK="${RANK:-32}"
RUN_PREFIX="${RUN_PREFIX:-compose}"
RUN_SUFFIX="${RUN_SUFFIX:-smoke}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

SEQUENCES_RAW="${SEQUENCES:-NI-Seq-G2}"
read -r -a SEQUENCES <<< "${SEQUENCES_RAW}"

# Train-only phase: eval budgets are intentionally omitted (evals run later
# on a different machine). Slice cache dir/steps still apply to init.
SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-slice_cache}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-8}"

# CL-method-specific budgets (kept small for smoke).
O_LORA_LAMBDA="${O_LORA_LAMBDA:-0.5}"
INFLORA_NULLSPACE_RANK="${INFLORA_NULLSPACE_RANK:-32}"
INFLORA_MAX_COV_BATCHES="${INFLORA_MAX_COV_BATCHES:-4}"
INFLORA_COV_BATCH_SIZE="${INFLORA_COV_BATCH_SIZE:-4}"
SAPT_KEY_DIM="${SAPT_KEY_DIM:-32}"
SAPT_ARM_N_SAMPLES="${SAPT_ARM_N_SAMPLES:-8}"
SAPT_ARM_MAX_NEW_TOKENS="${SAPT_ARM_MAX_NEW_TOKENS:-16}"
SAPT_ARM_N_EPOCHS="${SAPT_ARM_N_EPOCHS:-1}"
SAPT_ARM_BATCH_SIZE="${SAPT_ARM_BATCH_SIZE:-2}"
SAPT_ARM_LR="${SAPT_ARM_LR:-1e-3}"
SAPT_SEED_PROMPTS_PER_TASK="${SAPT_SEED_PROMPTS_PER_TASK:-8}"

FAIL_FAST="${FAIL_FAST:-1}"
EXTRA_ARGS=("$@")

# Filters (space-separated). Empty = run everything.
ONLY_INITS_RAW="${ONLY_INITS:-}"
ONLY_CL_RAW="${ONLY_CL:-}"
read -r -a ONLY_INITS <<< "${ONLY_INITS_RAW}"
read -r -a ONLY_CL <<< "${ONLY_CL_RAW}"

# Per-init flag bundles. Keys must match `--cl-method` choices we care
# about and the slice CLI shape.
init_flags() {
	case "$1" in
		lora_vanilla) printf '%s\n' "" ;;
		loram)        printf '%s\n' "--slice-init --slice-init-method loram --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS}" ;;
		lora_ga)      printf '%s\n' "--slice-init --slice-init-method lora_ga --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS}" ;;
		slice)        printf '%s\n' "--slice-init --slice-init-method slice  --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS}" ;;
		*) echo "unknown init: $1" >&2; return 2 ;;
	esac
}

# Per-cl-method flag bundles.
cl_flags() {
	case "$1" in
		# vanilla) printf '%s\n' "--cl-method vanilla" ;;
		o_lora)  printf '%s\n' "--cl-method o_lora --cl-o-lora-lambda ${O_LORA_LAMBDA}" ;;
		inflora) printf '%s\n' "--cl-method inflora --cl-inflora-nullspace-rank ${INFLORA_NULLSPACE_RANK} --cl-inflora-max-cov-batches ${INFLORA_MAX_COV_BATCHES} --cl-inflora-cov-batch-size ${INFLORA_COV_BATCH_SIZE}" ;;
		sapt)    printf '%s\n' "--cl-method sapt --cl-sapt-key-dim ${SAPT_KEY_DIM} --cl-sapt-arm-n-samples ${SAPT_ARM_N_SAMPLES} --cl-sapt-arm-max-new-tokens ${SAPT_ARM_MAX_NEW_TOKENS} --cl-sapt-arm-n-epochs ${SAPT_ARM_N_EPOCHS} --cl-sapt-arm-batch-size ${SAPT_ARM_BATCH_SIZE} --cl-sapt-arm-learning-rate ${SAPT_ARM_LR} --cl-sapt-seed-prompts-per-task ${SAPT_SEED_PROMPTS_PER_TASK}" ;;
		*) echo "unknown cl_method: $1" >&2; return 2 ;;
	esac
}

filter_match() {
	# $1 = candidate, remaining = filter list; empty filter list ⇒ always match
	local cand="$1"; shift
	if [[ "$#" -eq 0 ]]; then return 0; fi
	for f in "$@"; do
		[[ "${f}" == "${cand}" ]] && return 0
	done
	return 1
}

INITS=(lora_vanilla loram lora_ga slice)
CL_METHODS=(o_lora inflora sapt)

run_combo() {
	local sequence_name="$1"
	local init_tag="$2"
	local cl_tag="$3"

	local init_flag_str
	init_flag_str="$(init_flags "${init_tag}")"
	local cl_flag_str
	cl_flag_str="$(cl_flags "${cl_tag}")"

	# shellcheck disable=SC2206
	local init_arr=(${init_flag_str})
	# shellcheck disable=SC2206
	local cl_arr=(${cl_flag_str})

	local seq_safe
	seq_safe="$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')"
	local run_name="${RUN_PREFIX}_${init_tag}_${cl_tag}_${seq_safe}_${RUN_SUFFIX}"

	echo "============================================================"
	echo "Init      : ${init_tag}"
	echo "CL method : ${cl_tag}"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}  | Rank: ${RANK}"
	echo "Init flags: ${init_arr[*]:-<none>}"
	echo "CL flags  : ${cl_arr[*]}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence "${sequence_name}" \
			--run-name "${run_name}" \
			--rank "${RANK}" \
			--train-only \
			--log-level "${LOG_LEVEL}" \
			"${init_arr[@]}" \
			"${cl_arr[@]}" \
			"${EXTRA_ARGS[@]}"
}

declare -a FAILED=()
declare -a OK=()

for sequence_name in "${SEQUENCES[@]}"; do
	for init_tag in "${INITS[@]}"; do
		filter_match "${init_tag}" "${ONLY_INITS[@]}" || continue
		for cl_tag in "${CL_METHODS[@]}"; do
			filter_match "${cl_tag}" "${ONLY_CL[@]}" || continue
			label="${sequence_name}|${init_tag}|${cl_tag}"
			if run_combo "${sequence_name}" "${init_tag}" "${cl_tag}"; then
				OK+=("${label}")
			else
				FAILED+=("${label}")
				if [[ "${FAIL_FAST}" == "1" ]]; then
					echo "FAIL_FAST=1 — stopping after first failure: ${label}" >&2
					break 3
				fi
			fi
		done
	done
done

echo
echo "============================================================"
echo "Summary"
echo "  ok     : ${#OK[@]}"
echo "  failed : ${#FAILED[@]}"
echo "============================================================"
if [[ "${#OK[@]}" -gt 0 ]]; then
	printf '  [OK] %s\n' "${OK[@]}"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
	printf '  [FAIL] %s\n' "${FAILED[@]}"
	exit 1
fi
