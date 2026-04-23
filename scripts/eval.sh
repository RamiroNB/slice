
#!/usr/bin/env bash
set -euo pipefail

# SET CUDA VISIBLE DEVICES TO 1
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
# RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/adaptors_eval/NI-Seq-Opposite-v3}"
RUNS_ROOT="/mnt/E-SSD/dev-cl-lora/cl-lora/results/NI-Seq-Opposite-v4"
PYTHON_BIN="${PYTHON_BIN:-python}"
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

echo "Repo root : ${REPO_ROOT}"
echo "Runs root : ${RUNS_ROOT}"
echo "Python    : ${PYTHON_BIN}"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "Run count : ${#RUN_DIRS[@]}"

failed_runs=()

for run_dir in "${RUN_DIRS[@]}"; do
    manifests=$(find "${run_dir}" -path "*/stages/stage_*/eval_manifest.json" | wc -l | tr -d ' ')
    merged_models=$(find "${run_dir}" -path "*/checkpoints/stage_*/merged_model" -type d | wc -l | tr -d ' ')
    adapters=$(find "${run_dir}" -path "*/checkpoints/stage_*/adapter" -type d | wc -l | tr -d ' ')
    checkpoints=$((merged_models + adapters))
    log_file="${run_dir}/standalone_eval.log"

    echo "============================================================"
    echo "Run dir       : ${run_dir}"
    echo "Eval manifests: ${manifests}"
    echo "Merged models : ${merged_models}"
    echo "Adapters      : ${adapters}"
    echo "Log file      : ${log_file}"

    if [[ "${manifests}" -eq 0 ]]; then
        echo "Skipping (no eval_manifest.json files found)."
        failed_runs+=("${run_dir} (no manifests)")
        continue
    fi

    if [[ "${checkpoints}" -eq 0 ]]; then
        echo "Skipping (no merged_model or adapter checkpoints found)."
        failed_runs+=("${run_dir} (no checkpoints)")
        continue
    fi

    set +e
    (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" -m cl_lora.eval_standalone run --run-dir "${run_dir}" "${EXTRA_ARGS[@]}"
    ) 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "${status}" -ne 0 ]]; then
        echo "FAILED: ${run_dir}"
        failed_runs+=("${run_dir}")
    else
        echo "OK: ${run_dir}"
    fi
done

echo "============================================================"
if [[ "${#failed_runs[@]}" -gt 0 ]]; then
    echo "Completed with failures (${#failed_runs[@]}):"
    for run in "${failed_runs[@]}"; do
        echo "  - ${run}"
    done
    exit 1
fi
echo "All runs completed successfully."
