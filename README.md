# Low-Rank Adapters Initialization for Pareto-Competing Multi-Objective Losses

This repository runs continual learning experiments with LoRA adapters on instruction-tuning tasks.
The main loop is:

1. Train a LoRA adapter on the current task.
2. Merge adapter weights into the model.
3. Evaluate on seen tasks and general benchmarks.
4. Continue with the next task in a sequence.

## Project Structure

- cl_lora/orchestrator.py: Multi-stage continual training and evaluation pipeline.
- cl_lora/train.py: Single-task LoRA train + merge logic.
- cl_lora/eval.py: General evaluation (GP/IP) and seen-task evaluation.
- cl_lora/load_dataset.py: SuperNI and TRACE dataset loading and formatting.
- cl_lora/task_sequences.py: Sequence definitions (NI and TRACE).
- cl_lora/metrics.py: AP, FP, Forget, GP, IP computation.
- recompute_metrics.py: Utility script to recompute metrics from saved stage records.



Install dependencies:
```
pip install -r requirements.txt
```

## Environment Variables

Set your Hugging Face token:

	export HUGGING_TOKEN="your_hf_token"

and login with the Hugging Face CLI:

	hf auth longin

TRACE local data:

	"/mnt/C-SSD/ramiro/data/TRACE-Benchmark"


## Quick Start

Run a small 2-stage dummy sequence:

	CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
	  --sequence NI-Seq-Dummy \
	  --general-eval-set core \
	  --eval-size 10 \
	  --task-eval-samples 5 \
	  --task-eval-max-new-tokens 32 \
	  --run-name dummy_dev01

Resume an interrupted run:

	CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
	  --sequence NI-Seq-Dummy \
	  --general-eval-set core \
	  --eval-size 10 \
	  --task-eval-samples 5 \
	  --task-eval-max-new-tokens 32 \
	  --run-name dummy_dev01 \
	  --resume

## TRACE Dummy Run

	CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
	  --sequence TRACE-Dummy \
	  --general-eval-set core \
	  --eval-size 10 \
	  --task-eval-samples 5 \
	  --task-eval-max-new-tokens 32 \
	  --run-name trace_dummy_dev01

## Single-Task Training

	python -m cl_lora.train \
	  --task task363_sst2_polarity_classification \
	  --output-dir outputs/single_task_demo \
	  --save-merged-model

## LoRA Initialization Methods

The `--slice-init` flag enables gradient-based initialization. Three methods are available:

### Slice (default gradient-based)
Performs backward passes over current task data and all previously seen tasks (retain tasks) to seed LoRA A/B matrices.
At task t, the retain gradient is constructed using data from all tasks < t.
Initializations are cached per task set and config.

### LoRA-GA (gradient-aware baseline)
Uses SVD on forget gradients only, ignoring retain tasks. Faster than Slice but may not handle competing task objectives.

### LoRAM (DST-based baseline)
Lightweight initialization without explicit gradient computation. Best for quick comparisons.

### Retain batch size modes

- `--slice-retain-batch-size`: Batch size for retain gradient computation. Defaults to the training batch size.
- `--slice-retain-grad-accum`: Max gradient accumulation steps for retain. Defaults to `--slice-max-steps`.
- `--slice-retain-batch-size-set`: Controls how batch size is distributed across retain tasks:
  - `all_tasks` (default): All retain task datasets are concatenated into one dataloader with total batch size = `--slice-retain-batch-size`.
  - `each_task`: A separate dataloader is built per retain task, each with batch size = `--slice-retain-batch-size`. Effective total batch size = `batch_size * n_retain_tasks`.
- `--slice-single-retain-task-mode`: Only use the most recent previous task (t−1) for retain gradients, with the same batch size and steps as forget. Useful for ablation against the full multi-task retain.
- `--slice-grad-project-always`: OGD-style projection. When enabled with `--slice-grad-project`, the retain-direction component is always removed (no ReLU/max conflict gate).

Example (continual run with slice-no-proj init enabled):

```bash
CUDA_VISIBLE_DEVICES=1 python -m cl_lora.orchestrator \
	--sequence NI-Seq-Dummy \
	--general-eval-set core \
	--eval-size 10 \
	--task-eval-samples 5 \
	--task-eval-max-new-tokens 32 \
	--run-name dummy_slice_dev01 \
	--slice-init \
	--slice-cache-dir slice_cache \
	--slice-max-steps 64 \
	--slice-retain-scale 1.0
```
Example (continual run with slice-projected init, per-task retain batching):

```bash
CUDA_VISIBLE_DEVICES=1 python -m cl_lora.orchestrator \
	--sequence NI-Seq-Dummy \
	--general-eval-set core \
	--eval-size 10 \
	--task-eval-samples 5 \
	--task-eval-max-new-tokens 32 \
	--run-name dummy_slice_dev01 \
	--slice-init \
	--slice-cache-dir slice_cache \
	--slice-max-steps 64 \
	--slice-grad-project \
	--slice-grad-projection-mode per_module \
	--slice-retain-batch-size 32 \
	--slice-retain-grad-accum 50 \
	--slice-retain-batch-size-set each_task \
	--log-level DEBUG
```

## Evaluation Strategies

By default the orchestrator runs full evaluation (seen tasks + general GP/IP benchmarks) at every stage. For faster runs, two strategies skip work that doesn't affect the final reported metrics:

- `--general-eval-strategy final_only`: Skip GP/IP lm-eval benchmarks at intermediate stages. Only the final stage's GP/IP scores are used in the reported metrics, so intermediate runs are purely diagnostic. Saves (T−1) × 5 expensive evaluation passes for a T-task sequence.
- `--seen-eval-strategy diagonal_final`: At intermediate stages, evaluate only the just-trained task (diagonal score needed for AP). At the final stage, evaluate all seen tasks (needed for FP). Saves T(T−1)/2 − (T−1) generation-based evaluations. Tradeoff: intermediate off-diagonal entries in the results matrix will be None.

Both flags can be combined. Defaults (`every_stage` / `full_matrix`) preserve existing behaviour.

```bash
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
  --sequence NI-Seq-G1 \
  --general-eval-strategy final_only \
  --seen-eval-strategy diagonal_final \
  --run-name fast_run_01
```

## Checkpoint Management

By default, only the most recent stage checkpoint is kept on disk (previous ones are deleted after the next stage completes). This avoids accumulating ~6 GB per stage for large models.

- `--keep-all-checkpoints`: Preserve all intermediate stage checkpoints.

## Dataset Caching

Downloaded SuperNI and TRACE datasets are cached locally to avoid repeated HTTP requests within and across runs. The default cache directory is `datasets_cache/` in the project root. Override with the `CL_LORA_DATASET_CACHE` environment variable:

```bash
export CL_LORA_DATASET_CACHE=/path/to/shared/cache
```

## Output Layout

Training artifacts:

- outputs/<sequence>/stage_xx_<task_name>/
- outputs/<sequence>/stage_xx_<task_name>/adapter/
- outputs/<sequence>/stage_xx_<task_name>/checkpoint-*/

Experiment outputs:

- results/<sequence>/<run_name>/stages/stage_xx_<task_name>/stage_record.json
- results/<sequence>/<run_name>/checkpoints/stage_xx_<task_name>/merged_model/
- results/<sequence>/<run_name>/stage_records.partial.json
- results/<sequence>/<run_name>/results_matrix.json
- results/<sequence>/<run_name>/metrics.json
- results/<sequence>/<run_name>/run_summary.json

## Metrics

Computed in cl_lora/metrics.py:

- AP: Mean diagonal task score (score immediately after training each task).
- FP: Mean final-stage score over trained tasks.
- Forget: AP - FP.
- GP: Mean 0-shot general benchmark score.
- IP: Mean few-shot in-context benchmark score.

## Running Baseline Comparisons

Use the provided evaluation scripts to run and compare baseline initialization methods:

```bash
# Run LoRA-GA, Slice, and LoRAM baselines on specified sequences
bash scripts/full_eval_baselines.sh

# Run both vanilla (no init) and Slice methods
bash scripts/full_eval_both.sh

# Customize with environment variables:
GPU=0 RANK=64 RUN_SUFFIX=exp01 bash scripts/full_eval_baselines.sh
```

Specify initialization method via command line:
```bash
--slice-init-method lora_ga    # Use LoRA-GA initialization
--slice-init-method slice      # Use Slice initialization (default)
--slice-init-method loram      # Use LoRAM initialization
```

Control baseline behavior with environment variables:
- `SLICE_GRAD_PROJECT_ALWAYS`: Enable OGD-style projection (always remove retain direction) when using Slice method.
- `SLICE_GRAD_PROJECTION_MODE`: Projection granularity (`global` or `per_module`).
- `SLICE_MAX_STEPS`: Gradient accumulation steps for initialization.

## Advanced Projection Methods

Beyond plain PCGrad (dot-sign conflict removal), the Slice initialization
supports several gradient-surgery variants (ideas A.1–A.6 from
`ideas_for_new_methods.md`). Select one via `--slice-projection-method` plus
the relevant sub-flags:

| Method                | Flag                                        | Extra flags                                                                 |
| --------------------- | ------------------------------------------- | --------------------------------------------------------------------------- |
| `pcgrad` (default)    | —                                           | `--slice-grad-project-always` for OGD-style unconditional removal           |
| `cagrad`              | `--slice-projection-method cagrad`          | `--slice-cagrad-c <float in [0,1]>` (0 = vanilla, 1 = PCGrad)               |
| `gradvac`             | `--slice-projection-method gradvac`         | `--slice-gradvac-phi <target_cos>`, `--slice-gradvac-beta <EMA rate>`       |
| `nullspace`           | `--slice-projection-method nullspace`       | `--slice-nullspace-rank <k>`, `--slice-nullspace-sv-threshold <ratio>`      |
| `magnitude_preserving`| `--slice-projection-method magnitude_preserving` | rescales PCGrad output to match the original `‖g_f‖`                    |

Orthogonal knobs (stack on top of any method):

- `--slice-cosine-threshold <τ>`: only project when `cos(g_f, g_r) < τ` (replaces the default dot-sign gate).
- `--slice-per-layer-threshold` / `--slice-per-layer-threshold-delta <δ>`: use `median(cos across modules) − δ` as the threshold.
- `--slice-magnitude-preserve`: rescale post-projection gradient to preserve `‖g_f‖` per module.
- `--slice-svd-selection {lora_ga,top_r_no_sigma}`: `lora_ga` (default) uses disjoint slices `B=U[:,:r], A=V[r:2r,:]ᵀ`; `top_r_no_sigma` uses the top-r singular vectors without σ weighting (`B=U[:,:r], A=V[:,:r]ᵀ`).

### Global vs. Per-Module Projection

`--slice-grad-projection-mode {global,per_module}` controls projection granularity.

- **Global** computes one scalar γ from the summed dot products and squared retain-gradient norms across all modules, then applies `g_f_i ← g_f_i + γ · g_r_i` uniformly. It uses the distributive property — no flattened whole-model vector is ever built, so memory stays per-module.
- **Per-module** decides and applies γ independently for each target matrix.

Compatibility matrix:

| Method / option             | `global` supported? |
| --------------------------- | :-----------------: |
| `pcgrad`                    | ✓                   |
| `cagrad`                    | ✓                   |
| `magnitude_preserving`      | ✓ (rescale applied per module after the global projection) |
| `--slice-cosine-threshold`  | ✓ (single global `cos` gate)                              |
| `nullspace`                 | ✗ — SVD of a 2D retain matrix has no meaningful global analog |
| `gradvac`                   | ✗ — per-layer φ EMA is the mechanism                        |
| `--slice-per-layer-threshold` | ✗ — contradictory (threshold is per-layer by definition)  |

Incompatible combinations **raise a `ValueError` at init-time** rather than
silently running per-module, so run metadata always matches what actually
executed. `scripts/full_train_projection_variants.sh` therefore forces
`--slice-grad-projection-mode per_module` for the `nullspace`, `gradvac`,
and `per_layer_*` tags even when the top-level `SLICE_GRAD_PROJECTION_MODE`
is `global`.

### Projection Variant Sweep

```bash
# Train-only sweep over all variants (baselines + A.1–A.6 + C.16):
GPU=0 RANK=64 RUN_SUFFIX=projvariants bash scripts/full_train_projection_variants.sh

# Eval-only (run on a different machine, same results/ directory):
GPU=0 RUN_SUFFIX=projvariants bash scripts/full_eval_projection_variants.sh
```

## Available Sequences

Defined in cl_lora/task_sequences.py:

**Benign sequences:**
- NI-Seq-C1, NI-Seq-C2, NI-Seq-G1, NI-Seq-G2, NI-Seq-M1, NI-Seq-M2
- TRACE, TRACE-Dummy, NI-Seq-Dummy

**Competing/catastrophic sequences** (opposite task pairs for testing method robustness):
- NI-Seq-Opposite-v1 through NI-Seq-Opposite-v7


## Quick Eval

Perplexity-only mode — skips generation-based seen-task metrics and GP/IP benchmarks:

```
CUDA_VISIBLE_DEVICES=1 python -m cl_lora.orchestrator \
  --sequence NI-Seq-Dummy \
  --general-eval-set core \
  --eval-size 10 \
  --task-eval-samples 5 \
  --run-name quick_eval_dev01 \
  --quick-eval
```
