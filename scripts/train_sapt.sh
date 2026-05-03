#!/bin/bash

#SBATCH --job-name=sapt_train
#SBATCH --output=/home/joanapasquali/Sout/%j%x.out
#SBATCH --error=/home/joanapasquali/Sout/%j%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --gpus=3

# CIARS SLURM driver for the SAPT training sweep (all inits × all sequences).
#
# Runs sapt.sh (5 inits, fully sequential) on each of 3 sequences in parallel,
# one sequence per GPU (NI-Seq-G2 → GPU 0, TRACE → GPU 1, v4 → GPU 2).
# Results land in <FIXED_REPO>/results/, outputs in <FIXED_REPO>/outputs/.
#
# Override env vars from the sbatch command line, e.g.:
#   sbatch --export=ALL,RUN_SUFFIX=ciars01,ONLY_INITS="lora_vanilla loram" \
#          scripts/train_sapt.sh

set -euo pipefail

echo "=========================================="
echo "SAPT training sweep — $(date)"
echo "Host: $(hostname)"
echo "=========================================="

conda init
source "$(conda info --base)/etc/profile.d/conda.sh"

CONDA_ENV="${CONDA_ENV:-cl_lora}"
conda activate "${CONDA_ENV}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXED_REPO="${FIXED_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(conda run -n "${CONDA_ENV}" which python)}"

TRAIN_SCRIPT="${SCRIPT_DIR}/sapt.sh"

echo "FIXED_REPO   : ${FIXED_REPO}"
echo "PYTHON_BIN   : ${PYTHON_BIN}"
echo "TRAIN_SCRIPT : ${TRAIN_SCRIPT}"
echo "RUN_SUFFIX   : ${RUN_SUFFIX:-full}"
echo "ONLY_INITS   : ${ONLY_INITS:-(all)}"
echo "=========================================="

declare -A PIDS

for seq_gpu in "NI-Seq-G2:0" "TRACE:1" "NI-Seq-Opposite-v4:2"; do
    seq="${seq_gpu%%:*}"
    gpu="${seq_gpu##*:}"
    log_safe="$(echo "${seq}" | tr '[:upper:]-' '[:lower:]_')"
    log_file="${FIXED_REPO}/logs/train_${log_safe}_gpu${gpu}.log"
    mkdir -p "${FIXED_REPO}/logs"

    echo "[$(date +%H:%M:%S)] Launching  seq=${seq}  GPU=${gpu}  log=${log_file}"

    SEQUENCES="${seq}" \
    GPU="${gpu}" \
    FIXED_REPO="${FIXED_REPO}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    FAIL_FAST="${FAIL_FAST:-0}" \
        bash "${TRAIN_SCRIPT}" "$@" > "${log_file}" 2>&1 &

    PIDS["${seq}"]=$!
done

echo "[$(date +%H:%M:%S)] All three workers launched — waiting..."

EXIT=0
for seq in "NI-Seq-G2" "TRACE" "NI-Seq-Opposite-v4"; do
    pid="${PIDS[${seq}]}"
    if wait "${pid}"; then
        echo "[$(date +%H:%M:%S)] OK    ${seq}"
    else
        echo "[$(date +%H:%M:%S)] FAIL  ${seq}  (see ${FIXED_REPO}/logs/)" >&2
        EXIT=1
    fi
done

echo "=========================================="
if [[ "${EXIT}" -eq 0 ]]; then
    echo "All sequences completed successfully."
else
    echo "One or more sequences failed — check logs in ${FIXED_REPO}/logs/"
fi
echo "=========================================="
exit "${EXIT}"
