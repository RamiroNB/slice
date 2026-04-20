#!/usr/bin/env bash
set -euo pipefail

# Companion to full_train_projection_variants.sh.
#
# Runs eval-only on checkpoints produced by the training sweep. Invokes
# eval_standalone per run directory: it reads each stage's eval_manifest.json
# (written by the orchestrator) and executes evaluate_all().
#
# Typical deployment: copy results/ directory from the training machine
# to this machine, then run:
#
#   bash scripts/full_eval_projection_variants.sh
#   GPU=0 RUN_SUFFIX=projvariants bash scripts/full_eval_projection_variants.sh
#
# Extra CLI args are forwarded to eval_standalone.

GPU="${GPU:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

SLICE_RUN_PREFIX="${SLICE_RUN_PREFIX:-slice_var}"
RUN_SUFFIX="${RUN_SUFFIX:-projvariants}"

EVAL_SIZE="${EVAL_SIZE:-20}"
TASK_EVAL_SAMPLES="${TASK_EVAL_SAMPLES:-16}"
TASK_EVAL_MAX_NEW_TOKENS="${TASK_EVAL_MAX_NEW_TOKENS:-32}"
GENERAL_EVAL_BATCH_SIZE="${GENERAL_EVAL_BATCH_SIZE:-8}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

SEQUENCES=("${SEQUENCES[@]:-NI-Seq-Opposite-v3 NI-Seq-Opposite-v2 NI-Seq-Opposite-v4 NI-Seq-G2 TRACE }")
# SEQUENCES=("${SEQUENCES[@]:-NI-Seq-G2 TRACE NI-Seq-G1}")

# Keep the VARIANT tag list in sync with full_train_projection_variants.sh.
VARIANT_TAGS=(
	cagrad_c025
	cagrad_c050
	cagrad_c075
	gradvac_phi0_b05
	costau_neg005
	costau_000
	costau_005
	costau_010
	perlayer_d000
	perlayer_d005
	nullspace_r8
	nullspace_r32
	magpreserve
	svd_topr_no_sigma
	combo_cagrad_mag_topr
)

EXTRA_ARGS=("$@")

run_eval_for_run_dir() {
	local run_dir="$1"
	if [[ ! -d "${run_dir}/stages" ]]; then
		echo "[skip] ${run_dir} has no stages/ directory"
		return 0
	fi

	echo "============================================================"
	echo "Eval run  : ${run_dir}"
	echo "GPU       : ${GPU}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.eval_standalone run \
			--run-dir "${run_dir}" \
			--eval-size "${EVAL_SIZE}" \
			--task-eval-samples "${TASK_EVAL_SAMPLES}" \
			--task-eval-max-new-tokens "${TASK_EVAL_MAX_NEW_TOKENS}" \
			--general-eval-batch-size "${GENERAL_EVAL_BATCH_SIZE}" \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

# Baseline run names produced by full_train_projection_variants.sh.
# Each entry is a run-name prefix; the full name is "<prefix>_<seq_slug>_<RUN_SUFFIX>".
BASELINE_PREFIXES=(
	vanilla
	loram
	lora_ga_lora_ga
	lora_ga_top_r_no_sigma
)

for sequence_name in ${SEQUENCES[@]}; do
	seq_slug=$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')

	# Baselines
	for prefix in "${BASELINE_PREFIXES[@]}"; do
		run_name="${prefix}_${seq_slug}_${RUN_SUFFIX}"
		run_dir="${RESULTS_ROOT}/${sequence_name}/${run_name}"
		if [[ ! -d "${run_dir}" ]]; then
			echo "[skip] missing run dir: ${run_dir}"
			continue
		fi
		run_eval_for_run_dir "${run_dir}"
	done

	# Projection / SVD-selection variants
	for tag in "${VARIANT_TAGS[@]}"; do
		run_name="${SLICE_RUN_PREFIX}_${tag}_${seq_slug}_${RUN_SUFFIX}"
		run_dir="${RESULTS_ROOT}/${sequence_name}/${run_name}"
		if [[ ! -d "${run_dir}" ]]; then
			echo "[skip] missing run dir: ${run_dir}"
			continue
		fi
		run_eval_for_run_dir "${run_dir}"
	done
done
