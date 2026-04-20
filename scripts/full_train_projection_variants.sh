#!/usr/bin/env bash
set -euo pipefail

# Train-only sweep over new projection/SVD-selection variants implemented
# from ideas_for_new_methods.md (A.1-A.6 and C.16 without sigma weighting).
#
# Follows the same pattern as scripts/full_eval_baselines.sh, but:
#   - --train-only (no eval)
#   - iterates through all new projection options
#
# Usage:
#   bash scripts/full_train_projection_variants.sh
#   GPU=0 RANK=64 RUN_SUFFIX=sweep01 bash scripts/full_train_projection_variants.sh
#   bash scripts/full_train_projection_variants.sh --resume
#
# Any extra CLI args are forwarded to orchestrator.

GPU="${GPU:-1}"
RANK="${RANK:-64}"
SLICE_RUN_PREFIX="${SLICE_RUN_PREFIX:-slice_var}"
RUN_SUFFIX="${RUN_SUFFIX:-projvariants}"

SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-slice_cache}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-8}"
SLICE_GRAD_PROJECTION_MODE="${SLICE_GRAD_PROJECTION_MODE:-global}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Sequences to sweep over.
SEQUENCES=("${SEQUENCES[@]:-NI-Seq-Opposite-v3 NI-Seq-Opposite-v2 NI-Seq-Opposite-v4 NI-Seq-G2 TRACE }")
# SEQUENCES=("${SEQUENCES[@]:-NI-Seq-G2 TRACE NI-Seq-G1}")

EXTRA_ARGS=("$@")

# $1 sequence, $2 variant tag, remaining = variant-specific flags
run_slice_variant() {
	local sequence_name="$1"; shift
	local variant_tag="$1"; shift
	local variant_flags=("$@")

	local run_name
	run_name="${SLICE_RUN_PREFIX}_${variant_tag}_$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')_${RUN_SUFFIX}"

	echo "============================================================"
	echo "Variant   : ${variant_tag}"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}"
	echo "Rank      : ${RANK}"
	echo "Flags     : ${variant_flags[*]}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence "${sequence_name}" \
			--run-name "${run_name}" \
			--rank "${RANK}" \
			--slice-init \
			--slice-init-method slice \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--slice-grad-project \
			--slice-grad-projection-mode "${SLICE_GRAD_PROJECTION_MODE}" \
			--slice-retain-batch-size-set each_task \
			--train-only \
			--keep-all-checkpoints \
			--log-level "${LOG_LEVEL}" \
			"${variant_flags[@]}" \
			"${EXTRA_ARGS[@]}"
}

# Vanilla LoRA baseline: no slice-init at all.
run_vanilla_baseline() {
	local sequence_name="$1"
	local run_name
	run_name="vanilla_$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')_${RUN_SUFFIX}"

	echo "============================================================"
	echo "Baseline  : vanilla LoRA"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}"
	echo "Rank      : ${RANK}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence "${sequence_name}" \
			--run-name "${run_name}" \
			--rank "${RANK}" \
			--train-only \
			--keep-all-checkpoints \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

# LoRAM baseline: DST-based init, no gradient computation.
run_loram_baseline() {
	local sequence_name="$1"
	local run_name
	run_name="loram_$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')_${RUN_SUFFIX}"

	echo "============================================================"
	echo "Baseline  : LoRAM"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}"
	echo "Rank      : ${RANK}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence "${sequence_name}" \
			--run-name "${run_name}" \
			--rank "${RANK}" \
			--slice-init \
			--slice-init-method loram \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--train-only \
			--keep-all-checkpoints \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

# LoRA-GA baseline: SVD on forget gradients only.
# $1 sequence, $2 svd_selection ("lora_ga" disjoint slices, or "top_r_no_sigma")
run_lora_ga_baseline() {
	local sequence_name="$1"
	local svd_selection="$2"
	local run_name
	run_name="lora_ga_${svd_selection}_$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')_${RUN_SUFFIX}"

	echo "============================================================"
	echo "Baseline  : LoRA-GA (svd-selection=${svd_selection})"
	echo "Sequence  : ${sequence_name}"
	echo "Run name  : ${run_name}"
	echo "GPU       : ${GPU}"
	echo "Rank      : ${RANK}"
	echo "============================================================"

	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence "${sequence_name}" \
			--run-name "${run_name}" \
			--rank "${RANK}" \
			--slice-init \
			--slice-init-method lora_ga \
			--slice-svd-selection "${svd_selection}" \
			--slice-cache-dir "${SLICE_CACHE_DIR}" \
			--slice-max-steps "${SLICE_MAX_STEPS}" \
			--train-only \
			--keep-all-checkpoints \
			--log-level "${LOG_LEVEL}" \
			"${EXTRA_ARGS[@]}"
}

# Variants: (tag, flags...)
# A.1 CAGrad at c in {0.25, 0.5, 0.75}
# A.2 GradVac (phi=0, beta=0.5)
# A.3 cosine-threshold sweep tau in {-0.05, 0.0, 0.05, 0.1}
# A.4 per-layer threshold with delta in {0.0, 0.05}
# A.5 null-space projection (rank 8, 32)
# A.6 magnitude-preserving (applied on top of pcgrad)
# C.16 top_r_no_sigma SVD selection (with default pcgrad)
# Plus a "combo" variant (cagrad + magnitude preserve + top_r_no_sigma)

VARIANTS=(
	# tag|flags
	"magpreserve|--slice-projection-method magnitude_preserving"
	"svd_topr_no_sigma|--slice-svd-selection top_r_no_sigma"
	"combo_cagrad_mag_topr|--slice-projection-method cagrad --slice-cagrad-c 0.5 --slice-magnitude-preserve --slice-svd-selection top_r_no_sigma"
	"cagrad_c025|--slice-projection-method cagrad --slice-cagrad-c 0.25"
	"cagrad_c050|--slice-projection-method cagrad --slice-cagrad-c 0.50"
	"cagrad_c075|--slice-projection-method cagrad --slice-cagrad-c 0.75"
	"gradvac_phi0_b05|--slice-projection-method gradvac --slice-gradvac-phi 0.0 --slice-gradvac-beta 0.5"
	"costau_neg005|--slice-cosine-threshold -0.05"
	"costau_000|--slice-cosine-threshold 0.0"
	"costau_005|--slice-cosine-threshold 0.05"
	"costau_010|--slice-cosine-threshold 0.10"
	"perlayer_d000|--slice-per-layer-threshold --slice-per-layer-threshold-delta 0.0"
	"perlayer_d005|--slice-per-layer-threshold --slice-per-layer-threshold-delta 0.05"
	"nullspace_r8|--slice-projection-method nullspace --slice-nullspace-rank 8"
	"nullspace_r32|--slice-projection-method nullspace --slice-nullspace-rank 32"
)

for sequence_name in ${SEQUENCES[@]}; do
	# Baselines
	run_vanilla_baseline "${sequence_name}"
	run_loram_baseline   "${sequence_name}"
	run_lora_ga_baseline "${sequence_name}" "lora_ga"
	run_lora_ga_baseline "${sequence_name}" "top_r_no_sigma"

	# Projection / SVD-selection variants
	for entry in "${VARIANTS[@]}"; do
		tag="${entry%%|*}"
		flags_str="${entry#*|}"
		# shellcheck disable=SC2206
		flags_arr=(${flags_str})
		run_slice_variant "${sequence_name}" "${tag}" "${flags_arr[@]}"
	done
done
