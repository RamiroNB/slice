#!/usr/bin/env bash
# Roll up per-stage stage_record.json files into whole-sequence metrics
# (AP / FP / Forget / GP / IP) for every run directory under RUNS_ROOT.
#
# CPU-only: no GPU, no re-evaluation. Just calls
# `cl_lora.eval_standalone summary --run-dir <d>` for each run, which
# runs compute_cl_metrics() and writes:
#   <run_dir>/metrics.json
#   <run_dir>/results_matrix.json
#
# Env overrides:
#   RUNS_ROOT   directory containing run subdirectories to summarize
#               (default: the completed-runs folder on E-SSD)
#   PYTHON_BIN  python executable (default: python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-/mnt/E-SSD/dev-cl-lora/cl-lora/results/completed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${RUNS_ROOT}" ]]; then
    echo "Runs root not found: ${RUNS_ROOT}" >&2
    exit 1
fi

mapfile -t RUN_DIRS < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    echo "No run directories found under: ${RUNS_ROOT}"
    exit 1
fi

echo "============================================================"
echo "Repo root  : ${REPO_ROOT}"
echo "Runs root  : ${RUNS_ROOT}"
echo "Python     : ${PYTHON_BIN}"
echo "Run count  : ${#RUN_DIRS[@]}"
echo "============================================================"

cd "${REPO_ROOT}"

failures=0
skipped=0
for run_dir in "${RUN_DIRS[@]}"; do
    if ! ls "${run_dir}"/stages/stage_*/stage_record.json >/dev/null 2>&1; then
        echo "[SKIP] ${run_dir}  (no stages/stage_*/stage_record.json)"
        skipped=$((skipped + 1))
        continue
    fi
    echo "------------------------------------------------------------"
    echo "[RUN ] ${run_dir}"
    if ! "${PYTHON_BIN}" -m cl_lora.eval_standalone summary --run-dir "${run_dir}"; then
        echo "[FAIL] ${run_dir}" >&2
        failures=$((failures + 1))
    fi
done

echo "============================================================"
echo "Done.  ok=$((${#RUN_DIRS[@]} - failures - skipped))  skipped=${skipped}  failed=${failures}"
if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
