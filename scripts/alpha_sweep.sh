#!/usr/bin/env bash
set -euo pipefail

# rsLoRA alpha sweep: disentangle scale (alpha) from direction (init method)
# in CL-LoRA. Runs slice_var_global_cagrad_c050 and lora_ga (baseline) at
# alpha ∈ {1, 4} on NI-Seq-G2, NI-Seq-Opposite-v4, TRACE.
#
# The alpha=2 reference points are the existing *_projvariants runs.
#
# Usage:
#   bash scripts/alpha_sweep.sh
#   GPU=0 bash scripts/alpha_sweep.sh
#   ALPHAS="1 4" METHODS="cagrad lora_ga" SEQUENCES="NI-Seq-G2" bash scripts/alpha_sweep.sh
#   FAIL_FAST=0 bash scripts/alpha_sweep.sh        # collect failures, do not stop
#
# Any extra positional args are forwarded to the orchestrator
# (e.g. --resume, --keep-all-checkpoints, etc.).

GPU="${GPU:-4}"
RANK="${RANK:-64}"
RUN_SUFFIX="${RUN_SUFFIX:-alphasweep}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-slice_cache}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-8}"
SLICE_GRAD_PROJECTION_MODE="${SLICE_GRAD_PROJECTION_MODE:-global}"

ALPHAS_RAW="${ALPHAS:-1 2 4}"
read -r -a ALPHAS <<< "${ALPHAS_RAW}"

METHODS_RAW="${METHODS:-cagrad lora_ga}"
read -r -a METHODS <<< "${METHODS_RAW}"

SEQUENCES_RAW="${SEQUENCES:-NI-Seq-G2 NI-Seq-Opposite-v4 TRACE}"
read -r -a SEQUENCES <<< "${SEQUENCES_RAW}"

FAIL_FAST="${FAIL_FAST:-1}"
EXTRA_ARGS=("$@")

run_cagrad() {
	local sequence_name="$1"
	local alpha="$2"
	local seq_safe
	seq_safe="$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')"
	local run_name="slice_var_global_cagrad_c050_${seq_safe}_${RUN_SUFFIX}_a${alpha}"

	echo "============================================================"
	echo "Method    : slice_var_global_cagrad_c050"
	echo "Sequence  : ${sequence_name}"
	echo "Alpha     : ${alpha}    (rsLoRA scale = ${alpha}/sqrt(${RANK}))"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}      | Rank: ${RANK}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence "${sequence_name}" \
			--run-name "${run_name}" \
			--rank "${RANK}" \
			--lora-alpha "${alpha}" \
			--slice-init \
			--slice-init-method slice \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--slice-grad-project \
			--slice-grad-projection-mode "${SLICE_GRAD_PROJECTION_MODE}" \
			--slice-retain-batch-size-set each_task \
			--slice-projection-method cagrad \
			--slice-cagrad-c 0.50 \
			--train-only \
			--keep-all-checkpoints \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

run_lora_ga() {
	local sequence_name="$1"
	local alpha="$2"
	local seq_safe
	seq_safe="$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')"
	local run_name="lora_ga_lora_ga_${seq_safe}_${RUN_SUFFIX}_a${alpha}"

	echo "============================================================"
	echo "Method    : lora_ga (svd_selection=lora_ga)"
	echo "Sequence  : ${sequence_name}"
	echo "Alpha     : ${alpha}    (rsLoRA scale = ${alpha}/sqrt(${RANK}))"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}      | Rank: ${RANK}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence "${sequence_name}" \
			--run-name "${run_name}" \
			--rank "${RANK}" \
			--lora-alpha "${alpha}" \
			--slice-init \
			--slice-init-method lora_ga \
			--slice-svd-selection lora_ga \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--train-only \
			--keep-all-checkpoints \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

declare -a FAILED=()
declare -a OK=()

for sequence_name in "${SEQUENCES[@]}"; do
	for method in "${METHODS[@]}"; do
		for alpha in "${ALPHAS[@]}"; do
			label="${sequence_name}|${method}|alpha=${alpha}"
			case "${method}" in
				cagrad)
					if run_cagrad "${sequence_name}" "${alpha}"; then
						OK+=("${label}")
					else
						FAILED+=("${label}")
						if [[ "${FAIL_FAST}" == "1" ]]; then
							echo "FAIL_FAST=1 — stopping after first failure: ${label}" >&2
							break 3
						fi
					fi
					;;
				lora_ga)
					if run_lora_ga "${sequence_name}" "${alpha}"; then
						OK+=("${label}")
					else
						FAILED+=("${label}")
						if [[ "${FAIL_FAST}" == "1" ]]; then
							echo "FAIL_FAST=1 — stopping after first failure: ${label}" >&2
							break 3
						fi
					fi
					;;
				*)
					echo "Unknown method: ${method}" >&2
					exit 2
					;;
			esac
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
