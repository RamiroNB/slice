#!/usr/bin/env bash
set -euo pipefail

# O-LoRA baseline training sweep: all four LoRA initializations x 3 seeds.
#
#   Method : --cl-method o_lora  (merge-based; adds an orthogonality penalty
#            lambda * sum_{t'<t} ||A_t A_t'^T||_F^2 between the current task's
#            LoRA A matrices and every previous task's)
#   Inits  : lora_vanilla, loram, lora_ga, slice   (the four init baselines)
#   Seeds  : 42, 242, 1042
#   Warmup : 0.01  (--warmup-ratio)
#   GPU    : 3
#
# Same setup as scripts/train_sd_lora_baselines.sh (train-only, rank 64,
# alpha 2, warmup 0.01, same four inits and seeds) so the two baselines are
# directly comparable; evaluation is run separately (scripts/parallel_eval.sh).
#
# O-LoRA hyperparameter. The only knob is the orthogonality weight lambda.
# Default 0.5 is the value from Wang et al. 2023 ("Orthogonal Subspace Learning
# for Language Model Continual Learning") and the codebase default -- a sane,
# likely-to-work starting point. To actually search it, pass a space-separated
# list, e.g.  O_LORA_LAMBDA="0.25 0.5 1.0"  (multiplies the run count).
#
# Grid  = |SEQUENCES| x 4 inits x |O_LORA_LAMBDA| x |SEEDS|
#         (default 2 x 4 x 1 x 3 = 24 runs).
#
# Usage:
#   bash scripts/train_o_lora_baselines.sh
#   GPU=3 SEQUENCES="NI-Seq-Opposite-v2" bash scripts/train_o_lora_baselines.sh
#   O_LORA_LAMBDA="0.25 0.5 1.0" bash scripts/train_o_lora_baselines.sh   # lambda search
#   ONLY_INITS="slice lora_ga" ONLY_SEEDS="42" bash scripts/train_o_lora_baselines.sh
#   bash scripts/train_o_lora_baselines.sh --resume                      # extra args -> orchestrator
#
# Env overrides:
#   GPU              CUDA device id                      (default: 3)
#   MODEL_NAME       HF model id                         (default: Qwen/Qwen3-4B-Instruct-2507)
#   O_LORA_LAMBDA    space-separated orth-weight list    (default: 0.5)
#   SEEDS            space-separated seed list           (default: 42 242 1042)
#   SEQUENCES        space-separated sequence list       (default: NI-Seq-Opposite-v2 NI-Seq-Opposite-v4)
#   ONLY_INITS       restrict to a subset of init tags   (default: all four)
#   ONLY_SEEDS       restrict to a subset of seeds       (default: all)
#   RANK             LoRA rank                           (default: 64)
#   LORA_ALPHA       LoRA alpha (rsLoRA)                 (default: 2)
#   WARMUP_RATIO     LR warmup ratio                     (default: 0.01)
#   SLICE_MAX_STEPS  init-probe gradient steps           (default: 100)
#   PCGRAD_C         PCGrad-c strength for the slice arm (default: 1.0)
#   RUN_PREFIX       run-name prefix                     (default: baseline)
#   RUN_SUFFIX       run-name suffix                     (default: full)
#   REPO_ROOT        cl-lora checkout                    (default: this repo)
#   FAIL_FAST        stop on first failure (0 disables)  (default: 1)

GPU="${GPU:-3}"
RANK="${RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-100}"
PCGRAD_C="${PCGRAD_C:-1.0}"
RUN_PREFIX="${RUN_PREFIX:-baseline}"
RUN_SUFFIX="${RUN_SUFFIX:-full}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
BASE_MODEL_CACHE="${BASE_MODEL_CACHE:-${REPO_ROOT}/outputs/base_models}"
SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-${REPO_ROOT}/slice_cache}"

O_LORA_LAMBDA_RAW="${O_LORA_LAMBDA:-0.5}"
read -r -a O_LORA_LAMBDAS <<< "${O_LORA_LAMBDA_RAW}"
SEEDS_RAW="${SEEDS:-42 242 1042}"
read -r -a SEEDS <<< "${SEEDS_RAW}"
SEQUENCES_RAW="${SEQUENCES:-NI-Seq-Opposite-v2 NI-Seq-Opposite-v4}"
read -r -a SEQUENCES <<< "${SEQUENCES_RAW}"

ONLY_INITS_RAW="${ONLY_INITS:-}"; read -r -a ONLY_INITS <<< "${ONLY_INITS_RAW}"
ONLY_SEEDS_RAW="${ONLY_SEEDS:-}"; read -r -a ONLY_SEEDS <<< "${ONLY_SEEDS_RAW}"

FAIL_FAST="${FAIL_FAST:-1}"
EXTRA_ARGS=("$@")

CL_METHOD="o_lora"

# Per-init flag bundle. lora_vanilla = plain LoRA (no slice init); the other
# three enable the corresponding --slice-init-method. O-LoRA composes with all
# of them (the init writes A/B; O-LoRA regularizes A toward orthogonality).
#
# The slice arm carries the full conflict-projection config, not just
# --slice-init-method slice: without --slice-grad-project the init degenerates
# to an SVD of (g_current + g_retain), which is not SLICE. The projection flags
# mirror scripts/tier1/00_train.sh so this arm is the paper's main-table
# configuration composed with O-LoRA.
init_flags() {
	case "$1" in
		lora_vanilla) echo "" ;;
		loram)   echo "--slice-init --slice-init-method loram   --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS}" ;;
		lora_ga) echo "--slice-init --slice-init-method lora_ga --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS}" ;;
		slice)   echo "--slice-init --slice-init-method slice   --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS} \
			--slice-grad-project --slice-grad-projection-mode global \
			--slice-retain-batch-size-set each_task \
			--slice-projection-method pcgrad_c --slice-pcgrad-c ${PCGRAD_C}" ;;
		*) echo "unknown init: $1" >&2; return 2 ;;
	esac
}

INITS=(lora_vanilla loram lora_ga slice)

filter_match() {
	local cand="$1"; shift
	[[ "$#" -eq 0 ]] && return 0
	for f in "$@"; do [[ "${f}" == "${cand}" ]] && return 0; done
	return 1
}

# --- sanity: right repo ---
if [[ ! -f "${REPO_ROOT}/cl_lora/orchestrator.py" ]]; then
	echo "REPO_ROOT does not look like a cl-lora checkout: ${REPO_ROOT}" >&2; exit 1
fi

echo "============================================================"
echo "O-LoRA baseline sweep"
echo "Sequences : ${SEQUENCES[*]}"
echo "Inits     : ${INITS[*]}"
echo "Lambdas   : ${O_LORA_LAMBDAS[*]}"
echo "Seeds     : ${SEEDS[*]}"
echo "Model     : ${MODEL_NAME}"
echo "GPU       : ${GPU}  | Rank: ${RANK}  | Alpha: ${LORA_ALPHA}  | Warmup: ${WARMUP_RATIO}"
echo "Slice steps: ${SLICE_MAX_STEPS}"
echo "Repo root : ${REPO_ROOT}"
echo "Runs      : $(( ${#SEQUENCES[@]} * ${#INITS[@]} * ${#O_LORA_LAMBDAS[@]} * ${#SEEDS[@]} )) (before ONLY_* filters)"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

run_combo() {
	local sequence_name="$1" init_tag="$2" lambda="$3" seed="$4"

	local init_flag_str; init_flag_str="$(init_flags "${init_tag}")"
	# Word-split intentional — init_flags emits a flat string of flags.
	# shellcheck disable=SC2206
	local init_arr=(${init_flag_str})

	local seq_safe; seq_safe="$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')"
	local lam_safe; lam_safe="lam$(echo "${lambda}" | tr '.' 'p')"
	local run_name="${RUN_PREFIX}_${init_tag}_${CL_METHOD}_${lam_safe}_${seq_safe}_seed${seed}_${RUN_SUFFIX}"

	echo ""
	echo "--- ${sequence_name} | ${init_tag} | ${CL_METHOD}(λ=${lambda}) | seed ${seed} -> ${run_name}"

	cd "${REPO_ROOT}"
	CUDA_VISIBLE_DEVICES="${GPU}" \
		python -m cl_lora.orchestrator \
			--sequence          "${sequence_name}" \
			--run-name          "${run_name}" \
			--model-name        "${MODEL_NAME}" \
			--seed              "${seed}" \
			--rank              "${RANK}" \
			--lora-alpha        "${LORA_ALPHA}" \
			--warmup-ratio      "${WARMUP_RATIO}" \
			--output-root       "${OUTPUT_ROOT}" \
			--train-output-root "${TRAIN_OUTPUT_ROOT}" \
			--base-model-cache  "${BASE_MODEL_CACHE}" \
			--train-only \
			--keep-all-checkpoints \
			--general-eval-strategy final_only \
			--log-level         "${LOG_LEVEL}" \
			--cl-method o_lora \
			--cl-o-lora-lambda  "${lambda}" \
			"${init_arr[@]}" \
			"${EXTRA_ARGS[@]}"
}

declare -a FAILED=() OK=()
for sequence_name in "${SEQUENCES[@]}"; do
	for init_tag in "${INITS[@]}"; do
		filter_match "${init_tag}" "${ONLY_INITS[@]}" || continue
		for lambda in "${O_LORA_LAMBDAS[@]}"; do
			for seed in "${SEEDS[@]}"; do
				filter_match "${seed}" "${ONLY_SEEDS[@]}" || continue
				label="${sequence_name}|${init_tag}|${CL_METHOD}|λ${lambda}|seed${seed}"
				if run_combo "${sequence_name}" "${init_tag}" "${lambda}" "${seed}"; then
					OK+=("${label}")
				else
					FAILED+=("${label}")
					if [[ "${FAIL_FAST}" == "1" ]]; then
						echo "FAIL_FAST=1 — stopping after first failure: ${label}" >&2
						break 4
					fi
				fi
			done
		done
	done
done

echo ""
echo "============================================================"
echo "Summary: ok=${#OK[@]} failed=${#FAILED[@]}"
echo "============================================================"
if [[ "${#OK[@]}" -gt 0 ]]; then printf '  [OK]   %s\n' "${OK[@]}"; fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then printf '  [FAIL] %s\n' "${FAILED[@]}"; exit 1; fi
exit 0
