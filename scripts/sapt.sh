#!/usr/bin/env bash
set -euo pipefail

# Full SAPT training sweep — NI-Seq-G2, TRACE, NI-Seq-Opposite-v4.
#
# Inits × SAPT:
#   lora_vanilla, loram, lora_ga, slice_cagrad_050, slice_cagrad_075
#
# Runs the fixed orchestrator from FIXED_REPO so per-stage router snapshots
# (router_stage_NN.pt) are written at every stage. This enables correct
# full-matrix AP when eval_trace.sh later evaluates these runs.
# Results and checkpoints land in cl-baselines/cl-lora/ as usual.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_sapt_full.sh
#   SEQUENCES="TRACE" bash scripts/train_sapt_full.sh
#   ONLY_INITS="lora_vanilla loram" bash scripts/train_sapt_full.sh
#   bash scripts/train_sapt_full.sh --resume
#
# Any extra positional args (e.g. --resume) are forwarded to orchestrator.
#
# Env overrides:
#   GPU                 CUDA device id (default: 0)
#   RANK                LoRA rank (default: 64)
#   RUN_PREFIX          run name prefix (default: compose)
#   RUN_SUFFIX          run name suffix (default: full)
#   SEQUENCES           space-separated list (default: all three)
#   ONLY_INITS          restrict to subset of init tags (default: all)
#   FAIL_FAST           stop on first failure, 0 to disable (default: 1)
#   FIXED_REPO          repo with fixed orchestrator (default: /mnt/E-SSD/fix-cl-lora/cl-lora)
#   PYTHON_BIN          python binary (default: cl-lora conda env)
#   SLICE_MAX_STEPS     steps for slice/lora_ga/loram init (default: 100)
#   SAPT_KEY_DIM        router key dimension (default: 64)
#   SAPT_ARM_N_SAMPLES  pseudo-samples per task for ARM (default: 64)
#   SAPT_ARM_N_EPOCHS   ARM router training epochs (default: 3)
#   SAPT_ARM_BATCH_SIZE ARM batch size (default: 4)
#   SAPT_ARM_LR         ARM AdamW learning rate (default: 1e-3)
#   SAPT_SEED_PROMPTS   seed prompts cached per task (default: 32)

GPU="${GPU:-0}"
RANK="${RANK:-64}"
RUN_PREFIX="${RUN_PREFIX:-compose}"
RUN_SUFFIX="${RUN_SUFFIX:-full}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIS_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXED_REPO="${FIXED_REPO:-/mnt/E-SSD/fix-cl-lora/cl-lora}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/E-SSD/jmpasquali/cache/conda/envs/cl-lora/bin/python}"

# Absolute output dirs so results land in cl-baselines regardless of cwd.
OUTPUT_ROOT="${THIS_REPO}/results"
TRAIN_OUTPUT_ROOT="${THIS_REPO}/outputs"
BASE_MODEL_CACHE="${THIS_REPO}/outputs/base_models"
SLICE_CACHE_DIR="${THIS_REPO}/slice_cache"

SEQUENCES_RAW="${SEQUENCES:-NI-Seq-G2 TRACE NI-Seq-Opposite-v4}"
read -r -a SEQUENCES <<< "${SEQUENCES_RAW}"

ONLY_INITS_RAW="${ONLY_INITS:-}"
read -r -a ONLY_INITS <<< "${ONLY_INITS_RAW}"

FAIL_FAST="${FAIL_FAST:-1}"
EXTRA_ARGS=("$@")

# Slice init hyperparameters.
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-8}"

# SAPT / ARM hyperparameters — "real" scale (much larger than smoke).
SAPT_KEY_DIM="${SAPT_KEY_DIM:-64}"
SAPT_ARM_N_SAMPLES="${SAPT_ARM_N_SAMPLES:-64}"
SAPT_ARM_MAX_NEW_TOKENS="${SAPT_ARM_MAX_NEW_TOKENS:-32}"
SAPT_ARM_MAX_INPUT_LENGTH="${SAPT_ARM_MAX_INPUT_LENGTH:-128}"
SAPT_ARM_N_EPOCHS="${SAPT_ARM_N_EPOCHS:-3}"
SAPT_ARM_BATCH_SIZE="${SAPT_ARM_BATCH_SIZE:-4}"
SAPT_ARM_LR="${SAPT_ARM_LR:-1e-3}"
SAPT_SEED_PROMPTS="${SAPT_SEED_PROMPTS:-32}"

SAPT_FLAGS=(
    --cl-method sapt
    --cl-sapt-key-dim             "${SAPT_KEY_DIM}"
    --cl-sapt-arm-n-samples       "${SAPT_ARM_N_SAMPLES}"
    --cl-sapt-arm-max-new-tokens  "${SAPT_ARM_MAX_NEW_TOKENS}"
    --cl-sapt-arm-max-input-length "${SAPT_ARM_MAX_INPUT_LENGTH}"
    --cl-sapt-arm-n-epochs        "${SAPT_ARM_N_EPOCHS}"
    --cl-sapt-arm-batch-size      "${SAPT_ARM_BATCH_SIZE}"
    --cl-sapt-arm-learning-rate   "${SAPT_ARM_LR}"
    --cl-sapt-seed-prompts-per-task "${SAPT_SEED_PROMPTS}"
)

# ---------------------------------------------------------------------------
# Init flag bundles. Each function prints the flags for one init variant.
# ---------------------------------------------------------------------------
init_flags() {
    case "$1" in
        lora_vanilla)
            echo ""
            ;;
        loram)
            echo "--slice-init --slice-init-method loram \
                  --slice-cache-dir ${SLICE_CACHE_DIR} \
                  --slice-max-steps ${SLICE_MAX_STEPS}"
            ;;
        lora_ga)
            echo "--slice-init --slice-init-method lora_ga \
                  --slice-cache-dir ${SLICE_CACHE_DIR} \
                  --slice-max-steps ${SLICE_MAX_STEPS}"
            ;;
        slice_cagrad_050)
            echo "--slice-init --slice-init-method slice \
                  --slice-cache-dir ${SLICE_CACHE_DIR} \
                  --slice-max-steps ${SLICE_MAX_STEPS} \
                  --slice-grad-project \
                  --slice-projection-method cagrad \
                  --slice-cagrad-c 0.50 \
                  --slice-grad-projection-mode global \
                  --slice-retain-batch-size-set each_task"
            ;;
        slice_cagrad_075)
            echo "--slice-init --slice-init-method slice \
                  --slice-cache-dir ${SLICE_CACHE_DIR} \
                  --slice-max-steps ${SLICE_MAX_STEPS} \
                  --slice-grad-project \
                  --slice-projection-method cagrad \
                  --slice-cagrad-c 0.75 \
                  --slice-grad-projection-mode global \
                  --slice-retain-batch-size-set each_task"
            ;;
        *)
            echo "unknown init: $1" >&2; return 2 ;;
    esac
}

filter_match() {
    local cand="$1"; shift
    [[ "$#" -eq 0 ]] && return 0
    for f in "$@"; do [[ "${f}" == "${cand}" ]] && return 0; done
    return 1
}

INITS=(lora_vanilla loram lora_ga slice_cagrad_050 slice_cagrad_075)

# ---------------------------------------------------------------------------
# Validate env before queuing any work.
# ---------------------------------------------------------------------------
if [[ ! -d "${FIXED_REPO}/cl_lora/sapt" ]]; then
    echo "FIXED_REPO does not look like a cl-lora checkout: ${FIXED_REPO}" >&2
    exit 1
fi

echo "============================================================"
echo "SAPT full training sweep"
echo "Sequences    : ${SEQUENCES[*]}"
echo "Inits        : ${INITS[*]}"
echo "GPU          : ${GPU}  | Rank: ${RANK}"
echo "Fixed repo   : ${FIXED_REPO}"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Train root   : ${TRAIN_OUTPUT_ROOT}"
echo "Base cache   : ${BASE_MODEL_CACHE}"
echo "Slice cache  : ${SLICE_CACHE_DIR}"
echo "Slice steps  : ${SLICE_MAX_STEPS}"
echo "ARM samples  : ${SAPT_ARM_N_SAMPLES}  epochs: ${SAPT_ARM_N_EPOCHS}  bs: ${SAPT_ARM_BATCH_SIZE}"
echo "ARM tokens   : ${SAPT_ARM_MAX_NEW_TOKENS}  seed prompts: ${SAPT_SEED_PROMPTS}"
echo "Key dim      : ${SAPT_KEY_DIM}"
echo "Run suffix   : ${RUN_SUFFIX}"
echo "Extra args   : ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------
run_combo() {
    local sequence_name="$1"
    local init_tag="$2"

    local init_flag_str
    init_flag_str="$(init_flags "${init_tag}")"
    # Word-split intentional — init_flags emits a flat string of flags.
    # shellcheck disable=SC2206
    local init_arr=(${init_flag_str})

    local seq_safe
    seq_safe="$(echo "${sequence_name}" | tr '[:upper:]-' '[:lower:]_')"
    local run_name="${RUN_PREFIX}_${init_tag}_sapt_${seq_safe}_${RUN_SUFFIX}"

    echo ""
    echo "============================================================"
    echo "Init       : ${init_tag}"
    echo "Sequence   : ${sequence_name}"
    echo "Run name   : ${run_name}"
    echo "============================================================"

    # cd into FIXED_REPO so the patched cl_lora package (with per-stage router
    # snapshots) is used for training. Output paths are absolute.
    cd "${FIXED_REPO}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PYTHON_BIN}" -m cl_lora.orchestrator \
            --sequence             "${sequence_name}" \
            --run-name             "${run_name}" \
            --rank                 "${RANK}" \
            --output-root          "${OUTPUT_ROOT}" \
            --train-output-root    "${TRAIN_OUTPUT_ROOT}" \
            --base-model-cache     "${BASE_MODEL_CACHE}" \
            --train-only \
            --keep-all-checkpoints \
            --general-eval-strategy final_only \
            --log-level            "${LOG_LEVEL}" \
            "${init_arr[@]}" \
            "${SAPT_FLAGS[@]}" \
            "${EXTRA_ARGS[@]}"
}

declare -a FAILED=()
declare -a OK=()

for sequence_name in "${SEQUENCES[@]}"; do
    for init_tag in "${INITS[@]}"; do
        filter_match "${init_tag}" "${ONLY_INITS[@]}" || continue
        label="${sequence_name}|${init_tag}|sapt"
        if run_combo "${sequence_name}" "${init_tag}"; then
            OK+=("${label}")
        else
            FAILED+=("${label}")
            if [[ "${FAIL_FAST}" == "1" ]]; then
                echo "FAIL_FAST=1 — stopping after first failure: ${label}" >&2
                break 2
            fi
        fi
    done
done

echo ""
echo "============================================================"
echo "Summary"
printf '  ok     : %d\n' "${#OK[@]}"
printf '  failed : %d\n' "${#FAILED[@]}"
echo "============================================================"
[[ "${#OK[@]}"     -gt 0 ]] && printf '  [OK]   %s\n' "${OK[@]}"
[[ "${#FAILED[@]}" -gt 0 ]] && { printf '  [FAIL] %s\n' "${FAILED[@]}"; exit 1; }
