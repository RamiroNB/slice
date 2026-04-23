#!/usr/bin/env bash
# Single-GPU parallel evaluation driver.
#
# Enumerates every stage directory under every run directory in RUNS_ROOT and
# runs `cl_lora.eval_standalone stage --stage-dir <sd>` concurrently via
# xargs -P. After stages finish, `cl_lora.eval_standalone summary` is called
# once per run dir to rebuild results_matrix.json / metrics.json.
#
# No Python changes required; relies entirely on the existing `stage` and
# `summary` subcommands.
#
# Env overrides:
#   CUDA_VISIBLE_DEVICES  GPU id (default 1, matching eval.sh)
#   RUNS_ROOT             directory containing run subdirectories to evaluate
#   PYTHON_BIN            python executable
#   PARALLEL              number of concurrent workers (default 3)
#   GENERAL_BATCH_SIZE    lm-eval batch size per worker (default 4)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-/mnt/E-SSD/dev-cl-lora/cl-lora/results/NI-Seq-Opposite-v4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PARALLEL="${PARALLEL:-2}"
GENERAL_BATCH_SIZE="${GENERAL_BATCH_SIZE:-4}"
EXTRA_ARGS=("$@")

if [[ ! -d "${RUNS_ROOT}" ]]; then
    echo "Runs root not found: ${RUNS_ROOT}" >&2
    exit 1
fi

mapfile -t RUN_DIRS < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    echo "No run directories found under: ${RUNS_ROOT}"
    exit 1
fi

# Collect every stage dir that has an eval_manifest.json across all runs.
STAGE_LIST_FILE="$(mktemp)"
trap 'rm -f "${STAGE_LIST_FILE}"' EXIT
# Propagate Ctrl-C / SIGTERM to the whole process group so xargs' bash
# workers (and the Python children they spawned) die with us instead of
# being orphaned and continuing to hold the GPU.
trap 'trap - INT TERM; kill 0' INT TERM

for run_dir in "${RUN_DIRS[@]}"; do
    find "${run_dir}" -path "*/stages/stage_*/eval_manifest.json" -print \
        | sed 's#/eval_manifest.json$##' \
        | sort >> "${STAGE_LIST_FILE}"
done

n_stages=$(wc -l < "${STAGE_LIST_FILE}" | tr -d ' ')

echo "============================================================"
echo "Repo root        : ${REPO_ROOT}"
echo "Runs root        : ${RUNS_ROOT}"
echo "Python           : ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEV : ${CUDA_VISIBLE_DEVICES}"
echo "Parallel workers : ${PARALLEL}"
echo "lm-eval batch    : ${GENERAL_BATCH_SIZE}"
echo "Run count        : ${#RUN_DIRS[@]}"
echo "Stage count      : ${n_stages}"
echo "Extra args       : ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

if [[ "${n_stages}" -eq 0 ]]; then
    echo "No stages with eval_manifest.json found."
    exit 1
fi

cd "${REPO_ROOT}"

# Worker: invoked once per stage dir by xargs. Output goes to a per-stage log.
run_one_stage() {
    local stage_dir="$1"
    local log_file="${stage_dir}/parallel_eval.log"
    # Give each concurrent worker its own torch inductor cache to avoid
    # rare races when multiple processes compile the same graph at once.
    # $$ is this bash worker's PID, so concurrent workers get distinct dirs.
    export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_cl_lora_$$"
    echo "[$(date +%H:%M:%S)] START ${stage_dir}"
    if "${PYTHON_BIN}" -m cl_lora.eval_standalone stage \
            --stage-dir "${stage_dir}" \
            --general-eval-batch-size "${GENERAL_BATCH_SIZE}" \
            "${EXTRA_ARGS[@]}" > "${log_file}" 2>&1; then
        echo "[$(date +%H:%M:%S)] OK    ${stage_dir}"
    else
        echo "[$(date +%H:%M:%S)] FAIL  ${stage_dir}  (see ${log_file})" >&2
        return 1
    fi
}
export -f run_one_stage
export PYTHON_BIN GENERAL_BATCH_SIZE
# xargs can't expand bash arrays; pass EXTRA_ARGS through as a single string
# that run_one_stage re-splits via its positional args.
export EXTRA_ARGS_STR="${EXTRA_ARGS[*]:-}"

# Re-export a wrapper that forwards EXTRA_ARGS_STR as individual args.
run_one_stage_wrapper() {
    # shellcheck disable=SC2086
    run_one_stage "$1" ${EXTRA_ARGS_STR}
}
export -f run_one_stage_wrapper

set +e
xargs -a "${STAGE_LIST_FILE}" -n 1 -P "${PARALLEL}" -I {} \
    bash -c 'run_one_stage_wrapper "$@"' _ {}
stage_status=$?
set -e

echo "============================================================"
echo "Rebuilding run-level summaries..."
summary_failures=0
for run_dir in "${RUN_DIRS[@]}"; do
    if ls "${run_dir}"/stages/stage_*/stage_record.json >/dev/null 2>&1; then
        if ! "${PYTHON_BIN}" -m cl_lora.eval_standalone summary --run-dir "${run_dir}"; then
            summary_failures=$((summary_failures + 1))
        fi
    fi
done

echo "============================================================"
if [[ "${stage_status}" -ne 0 ]]; then
    echo "One or more stage evaluations failed (xargs exit ${stage_status})."
    echo "Per-stage logs: <stage_dir>/parallel_eval.log"
    exit "${stage_status}"
fi
if [[ "${summary_failures}" -ne 0 ]]; then
    echo "Stages OK but ${summary_failures} run summaries failed."
    exit 1
fi
echo "All stages completed successfully."
