#!/usr/bin/env bash
set -euo pipefail

# SD-LoRA baseline training sweep: all four LoRA initializations x 3 seeds.
#
#   Method : --cl-method sd_lora  (no-merge; every task's frozen net-update
#            direction stays live with a trainable per-task magnitude scalar)
#   Inits  : lora_vanilla, loram, lora_ga, slice   (the four init baselines)
#   Seeds  : 42, 242, 1042
#   Warmup : 0.01  (--warmup-ratio)
#   GPU    : 5
#
# Train-only, matching the house style of scripts/sapt.sh: evaluation is run
# separately from the persisted eval_manifest.json / checkpoints (see
# scripts/parallel_eval.sh, scripts/eval.sh). Results land in results/ and
# outputs/ as usual; the 4B base model is cached once and reused across runs.
#
# Grid  = |SEQUENCES| x 4 inits x |SEEDS|  (default 2 x 4 x 3 = 24 runs).
#
# Usage:
#   bash scripts/train_sd_lora_baselines.sh
#   GPU=5 SEQUENCES="NI-Seq-Opposite-v2" bash scripts/train_sd_lora_baselines.sh
#   ONLY_INITS="slice lora_ga" ONLY_SEEDS="42" bash scripts/train_sd_lora_baselines.sh
#   FAIL_FAST=0 bash scripts/train_sd_lora_baselines.sh          # collect failures
#   bash scripts/train_sd_lora_baselines.sh --resume             # extra args -> orchestrator
#
# Env overrides:
#   GPU              CUDA device id                      (default: 5)
#   MODEL_NAME       HF model id                         (default: Qwen/Qwen3-4B-Instruct-2507)
#   SEEDS            space-separated seed list           (default: 42 242 1042)
#   SEQUENCES        space-separated sequence list       (default: NI-Seq-Opposite-v2 NI-Seq-Opposite-v4)
#   ONLY_INITS       restrict to a subset of init tags   (default: all four)
#   ONLY_SEEDS       restrict to a subset of seeds       (default: all)
#   RANK             LoRA rank                           (default: 64)
#   LORA_ALPHA       LoRA alpha (rsLoRA)                 (default: 2)
#   WARMUP_RATIO     LR warmup ratio                     (default: 0.01)
#   SLICE_MAX_STEPS  init-probe gradient steps           (default: 100)
#   RUN_PREFIX       run-name prefix                     (default: baseline)
#   RUN_SUFFIX       run-name suffix                     (default: full)
#   REPO_ROOT        cl-lora checkout                    (default: this repo)
#   FAIL_FAST        stop on first failure (0 disables)  (default: 1)
#   SKIP_COMPLETED   skip combos already finished        (default: 1)
#   AUTO_RESUME      --resume combos left half-trained   (default: 1)
#   TRAIN_BATCH      per-device train batch              (default: 8)
#   GRAD_ACCUM       gradient accumulation steps         (default: 4)
#   PROBE_BATCH      init-probe batch, slice inits only  (default: 16)
#   PCGRAD_C         PCGrad-c strength for the slice arm (default: 1.0)
#
# Restartability. The sweep is idempotent: SKIP_COMPLETED skips any combo whose
# run dir already has run_summary.json, and AUTO_RESUME passes --resume to one
# that has a stage_records.partial.json but no run_summary.json. So re-running
# this script after an interruption picks up exactly what is left, and combos
# whose checkpoints were deleted after evaluation are not retrained.
#
# Memory. TRAIN_BATCH x GRAD_ACCUM defaults to 8 x 4, not the orchestrator's
# 16 x 2. The effective batch (32), the step count and the LR schedule are
# identical, but the micro-batch is halved -- at 16 the last stage of a sequence
# OOMs on a 48 GB card once several frozen SD-LoRA blocks are live (the fatal
# allocation is the fp32 logits cast in ForCausalLMLoss). PROBE_BATCH pins the
# init probe at 16 so that lowering the training batch does not change the
# gradient probe or invalidate cached slice inits -- see the comment on
# per_device_batch_size in cl_lora/train.py.

GPU="${GPU:-5}"
RANK="${RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-100}"
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

SEEDS_RAW="${SEEDS:-42 242 1042}"
read -r -a SEEDS <<< "${SEEDS_RAW}"
SEQUENCES_RAW="${SEQUENCES:-NI-Seq-Opposite-v2 NI-Seq-Opposite-v4}"
read -r -a SEQUENCES <<< "${SEQUENCES_RAW}"

ONLY_INITS_RAW="${ONLY_INITS:-}"; read -r -a ONLY_INITS <<< "${ONLY_INITS_RAW}"
ONLY_SEEDS_RAW="${ONLY_SEEDS:-}"; read -r -a ONLY_SEEDS <<< "${ONLY_SEEDS_RAW}"

FAIL_FAST="${FAIL_FAST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
PROBE_BATCH="${PROBE_BATCH:-16}"
PCGRAD_C="${PCGRAD_C:-1.0}"
EXTRA_ARGS=("$@")

CL_METHOD="sd_lora"
CL_FLAGS=( --cl-method sd_lora )

# Per-init flag bundle. lora_vanilla = plain LoRA (no slice init); the other
# three enable the corresponding --slice-init-method. SD-LoRA composes with all
# of them (the init writes A/B; SD-LoRA freezes each task's net update).
#
# The three slice-init variants also pin --slice-probe-batch-size: the probe
# batch defaults to the training batch, which we lowered for memory, and it
# feeds both the init gradient estimate and the slice cache key.
#
# The slice arm carries the full conflict-projection config, not just
# --slice-init-method slice: without --slice-grad-project the init degenerates
# to an SVD of (g_current + g_retain), which is not SLICE. The projection flags
# mirror scripts/tier1/00_train.sh so this arm is the paper's main-table
# configuration composed with SD-LoRA.
init_flags() {
	local probe="--slice-probe-batch-size ${PROBE_BATCH}"
	case "$1" in
		lora_vanilla) echo "" ;;
		loram)   echo "--slice-init --slice-init-method loram   --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS} ${probe}" ;;
		lora_ga) echo "--slice-init --slice-init-method lora_ga --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS} ${probe}" ;;
		slice)   echo "--slice-init --slice-init-method slice   --slice-cache-dir ${SLICE_CACHE_DIR} --slice-max-steps ${SLICE_MAX_STEPS} ${probe} \
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

# --- sanity: right repo, right branch (sd_lora must exist) ---
if [[ ! -f "${REPO_ROOT}/cl_lora/orchestrator.py" ]]; then
	echo "REPO_ROOT does not look like a cl-lora checkout: ${REPO_ROOT}" >&2; exit 1
fi
if [[ ! -f "${REPO_ROOT}/cl_lora/cl_methods/sd_lora.py" ]]; then
	echo "sd_lora method not found under ${REPO_ROOT} -- wrong branch? (need add_sd_lora / add_sapt)" >&2; exit 1
fi

echo "============================================================"
echo "SD-LoRA baseline sweep"
echo "Sequences : ${SEQUENCES[*]}"
echo "Inits     : ${INITS[*]}"
echo "Seeds     : ${SEEDS[*]}"
echo "Model     : ${MODEL_NAME}"
echo "GPU       : ${GPU}  | Rank: ${RANK}  | Alpha: ${LORA_ALPHA}  | Warmup: ${WARMUP_RATIO}"
echo "Slice steps: ${SLICE_MAX_STEPS}"
echo "Repo root : ${REPO_ROOT}"
echo "Runs      : $(( ${#SEQUENCES[@]} * ${#INITS[@]} * ${#SEEDS[@]} )) (before ONLY_* filters)"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

run_combo() {
	local sequence_name="$1" init_tag="$2" seed="$3"

	local init_flag_str; init_flag_str="$(init_flags "${init_tag}")"
	# Word-split intentional — init_flags emits a flat string of flags.
	# shellcheck disable=SC2206
	local init_arr=(${init_flag_str})

	local seq_safe; seq_safe="$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')"
	local run_name="${RUN_PREFIX}_${init_tag}_${CL_METHOD}_${seq_safe}_seed${seed}_${RUN_SUFFIX}"
	local run_dir="${OUTPUT_ROOT}/${sequence_name}/${run_name}"

	# Already finished: run_summary.json is written only after the last stage.
	# Reported as SKIP, not OK, so the tally distinguishes "ran now" from
	# "was already there".
	if [[ "${SKIP_COMPLETED}" == "1" && -f "${run_dir}/run_summary.json" ]]; then
		echo ""
		echo "--- ${sequence_name} | ${init_tag} | seed ${seed} -> SKIP (run_summary.json present)"
		return 100
	fi

	# Half-trained: the orchestrator refuses to start on top of an existing
	# partial state unless --resume is given.
	local resume_arr=()
	if [[ "${AUTO_RESUME}" == "1" && -f "${run_dir}/stage_records.partial.json" ]]; then
		resume_arr=(--resume)
	fi

	echo ""
	echo "--- ${sequence_name} | ${init_tag} | ${CL_METHOD} | seed ${seed} -> ${run_name}${resume_arr[0]:+ (resume)}"

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
			--per-device-train-batch-size "${TRAIN_BATCH}" \
			--gradient-accumulation-steps "${GRAD_ACCUM}" \
			--output-root       "${OUTPUT_ROOT}" \
			--train-output-root "${TRAIN_OUTPUT_ROOT}" \
			--base-model-cache  "${BASE_MODEL_CACHE}" \
			--train-only \
			--keep-all-checkpoints \
			--general-eval-strategy final_only \
			--log-level         "${LOG_LEVEL}" \
			"${CL_FLAGS[@]}" \
			"${init_arr[@]}" \
			"${resume_arr[@]+"${resume_arr[@]}"}" \
			"${EXTRA_ARGS[@]}"
}

declare -a FAILED=() OK=() SKIPPED=()
for sequence_name in "${SEQUENCES[@]}"; do
	for init_tag in "${INITS[@]}"; do
		filter_match "${init_tag}" "${ONLY_INITS[@]}" || continue
		for seed in "${SEEDS[@]}"; do
			filter_match "${seed}" "${ONLY_SEEDS[@]}" || continue
			label="${sequence_name}|${init_tag}|${CL_METHOD}|seed${seed}"
			# `|| rc=$?` and not a bare call: set -e would abort the sweep on
			# the first non-zero return, including the SKIP sentinel.
			rc=0
			run_combo "${sequence_name}" "${init_tag}" "${seed}" || rc=$?
			if [[ "${rc}" -eq 0 ]]; then
				OK+=("${label}")
			elif [[ "${rc}" -eq 100 ]]; then
				SKIPPED+=("${label}")
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

echo ""
echo "============================================================"
echo "Summary: ok=${#OK[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"
echo "============================================================"
if [[ "${#SKIPPED[@]}" -gt 0 ]]; then printf '  [SKIP] %s\n' "${SKIPPED[@]}"; fi
if [[ "${#OK[@]}" -gt 0 ]]; then printf '  [OK]   %s\n' "${OK[@]}"; fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then printf '  [FAIL] %s\n' "${FAILED[@]}"; exit 1; fi
exit 0
