#!/bin/bash

#SBATCH --job-name=cl_lora_lean
#SBATCH --output=/home/joanapasquali/Sout/%j%x.out
#SBATCH --error=/home/joanapasquali/Sout/%j%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=42G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=3

# CIARS SLURM driver for the memory-economy LoRA × CL-method sweep.
#
# Wraps scripts/test_init_x_cl_methods_lean.sh. Pins one init group per GPU
# so concurrent workers cannot race on the same slice_cache/<key> entry
# (slice cache keys depend on init_method, not cl_method, so combos that
# share an init must stay on the same GPU and run sequentially). The base
# model cache is process-safe via atomic rename, so all GPUs may share it.
#
# Override env vars from the sbatch command line, e.g.:
#   sbatch --export=ALL,SEQUENCES="NI-Seq-G2 TRACE",RUN_SUFFIX=ciars01 \
#          scripts/ciars_lean_sweep.sh

set -euo pipefail

echo "Starting the execution"

# --- runtime env ----------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

REPO_ROOT="${REPO_ROOT:-/home/joanapasquali/cl-lora}"
CONDA_ENV="${CONDA_ENV:-cl_lora}"

cd "${REPO_ROOT}"
conda activate "${CONDA_ENV}"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Repo root: ${REPO_ROOT}"

# --- memory-economy paths -------------------------------------------------
# Shared base model lives once per (cache_dir, model_name); each run's
# checkpoints/base_model becomes a symlink into this cache. Slice-cache
# inits/*.pt are deleted in-process after apply (handled by the Python
# side of initialize_lora_with_slice).
export BASE_MODEL_CACHE="${BASE_MODEL_CACHE:-${REPO_ROOT}/outputs/base_models}"
export SLICE_CACHE_DIR="${SLICE_CACHE_DIR:-${REPO_ROOT}/slice_cache}"
mkdir -p "${BASE_MODEL_CACHE}" "${SLICE_CACHE_DIR}"

# --- sweep config (override via env vars) ---------------------------------
SEQUENCES_RAW="${SEQUENCES:-NI-Seq-G2}"
INITS_RAW="${ONLY_INITS:-lora_vanilla loram lora_ga slice}"
ONLY_CL_RAW="${ONLY_CL:-o_lora inflora sapt}"
RUN_PREFIX="${RUN_PREFIX:-compose}"
RUN_SUFFIX="${RUN_SUFFIX:-ciars}"
RANK="${RANK:-32}"
SLICE_MAX_STEPS="${SLICE_MAX_STEPS:-8}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Number of GPUs we may use. Prefer the SLURM-allocated count.
NGPU="${NGPU:-${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-3}}}"

# Where per-GPU stdout/stderr logs land.
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ciars/${SLURM_JOB_ID:-local}}"
mkdir -p "${LOG_DIR}"

read -r -a INITS <<< "${INITS_RAW}"

echo "Sequences      : ${SEQUENCES_RAW}"
echo "Inits          : ${INITS[*]}"
echo "CL methods     : ${ONLY_CL_RAW}"
echo "GPUs           : ${NGPU}"
echo "Base model cache: ${BASE_MODEL_CACHE}"
echo "Slice cache dir : ${SLICE_CACHE_DIR}"
echo "Per-GPU logs    : ${LOG_DIR}"

# --- shard inits across GPUs ---------------------------------------------
declare -a GPU_INITS
for ((g=0; g<NGPU; g++)); do GPU_INITS[$g]=""; done
for ((i=0; i<${#INITS[@]}; i++)); do
	g=$((i % NGPU))
	GPU_INITS[$g]+="${INITS[$i]} "
done

# --- launcher: one GPU's worth of work runs the lean script serially -----
launch_gpu_workload() {
	local gpu="$1"; shift
	local inits_for_gpu="$*"
	local log="${LOG_DIR}/gpu${gpu}.log"
	echo "[dispatch] GPU ${gpu} <- inits: ${inits_for_gpu}  log=${log}"
	GPU="${gpu}" \
	RANK="${RANK}" \
	SLICE_MAX_STEPS="${SLICE_MAX_STEPS}" \
	LOG_LEVEL="${LOG_LEVEL}" \
	ONLY_INITS="${inits_for_gpu}" \
	ONLY_CL="${ONLY_CL_RAW}" \
	SEQUENCES="${SEQUENCES_RAW}" \
	RUN_PREFIX="${RUN_PREFIX}" \
	RUN_SUFFIX="${RUN_SUFFIX}" \
	BASE_MODEL_CACHE="${BASE_MODEL_CACHE}" \
	SLICE_CACHE_DIR="${SLICE_CACHE_DIR}" \
	FAIL_FAST=0 \
		bash "${REPO_ROOT}/scripts/test_init_x_cl_methods_lean.sh" \
		>"${log}" 2>&1
}

# Propagate Ctrl-C / SIGTERM (sbatch scancel) to children.
trap 'trap - INT TERM; kill 0' INT TERM

# --- spawn one worker per GPU, wait for all ------------------------------
pids=()
for ((g=0; g<NGPU; g++)); do
	# Skip GPUs that received no inits (e.g. fewer inits than GPUs).
	if [[ -z "${GPU_INITS[$g]// }" ]]; then continue; fi
	launch_gpu_workload "${g}" ${GPU_INITS[$g]} &
	pids+=($!)
done

failed=0
for p in "${pids[@]}"; do
	if ! wait "${p}"; then failed=$((failed+1)); fi
done

echo "============================================================"
echo "Per-GPU log tails"
echo "============================================================"
for f in "${LOG_DIR}"/gpu*.log; do
	echo "----- ${f} -----"
	tail -n 25 "${f}" || true
done

if [[ "${failed}" -ne 0 ]]; then
	echo "Finished with ${failed} failed worker(s)."
	exit 1
fi

echo "Finished execution"
