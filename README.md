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

Checkpoints use an adapter-only format to save disk space. Instead of saving a full merged model (~6 GB) per stage, only the base model is saved once and each stage saves its LoRA adapter weights (~400 MB):

```
results/<sequence>/<run_name>/checkpoints/
├── base_model/                          # saved once (~6 GB)
├── stage_01_<task>/adapter/             # LoRA weights only (~400 MB)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── init_correction.pt              # only present when --slice-init is used
├── stage_02_<task>/adapter/
└── ...
```

At eval time, `eval_standalone` reconstructs the model for stage k by loading the base model and merging adapters 1..k in order — this is numerically identical to the in-memory merged model from training.

For runs using `--slice-init`, each adapter directory also contains `init_correction.pt` which stores the LoRA init matrices. This is required to correctly replay the weight absorption step during reconstruction.

To separate training from evaluation (recommended for long sequences):

```bash
# Train only — saves checkpoints and eval manifests, skips evaluation
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
  --sequence NI-Seq-G1 \
  --run-name my_run \
  --train-only

# Evaluate all stages from saved checkpoints
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.eval_standalone run \
  --run-dir results/NI-Seq-G1/my_run \
  --task-eval-samples 64 \
  --skip-general-eval
```

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
- results/<sequence>/<run_name>/checkpoints/base_model/
- results/<sequence>/<run_name>/checkpoints/stage_xx_<task_name>/adapter/
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

## Continual-Learning Methods (`--cl-method`)

The `--cl-method` flag selects a training-time continual-learning strategy
that composes with **any** LoRA initialization (`lora_vanilla`, `loram`,
`lora_ga`, `slice`). Init writes the starting A/B; the CL method then runs
its own hooks during the per-stage train+merge loop.

| Method     | Mechanism                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| `vanilla`  | Default. Per-stage train + merge with no extra mechanics.                                              |
| `o_lora`   | Adds `lambda * sum_{t' < t} ||A_t @ A_{t'}^T||_F^2` to each step's loss. Snapshots A's after each stage. |
| `inflora`  | Accumulates input-feature covariance per LoRA target across stages; projects new A onto the null-space of accumulated covariance before training. |
| `sapt`     | Faithful SAPT (Zhao et al., ACL 2024). Keeps each task's LoRA as a parallel named adapter (no merging across stages); after each stage runs ARM (Attentive Reflection) to train a shared-attention router on pseudo-samples generated by every adapter. Inference routes per-input through `base + Σ α_i(x) · B_i A_i x`. |

For all CL methods the retain set is automatically *all tasks before the
current stage* (the same retain set used by slice init). Per-method state
is persisted to `results/<seq>/<run>/cl_state/` and reloaded on `--resume`.

Examples — `slice + O-LoRA`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
    --sequence NI-Seq-Dummy \
    --run-name slice_olora_dev01 \
    --slice-init --slice-init-method slice --slice-max-steps 64 \
    --cl-method o_lora --cl-o-lora-lambda 0.5
```

`lora_ga + InfLoRA`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
    --sequence NI-Seq-Dummy \
    --run-name lora_ga_inflora_dev01 \
    --slice-init --slice-init-method lora_ga --slice-max-steps 64 \
    --cl-method inflora \
    --cl-inflora-nullspace-rank 64 \
    --cl-inflora-max-cov-batches 32 \
    --cl-inflora-cov-batch-size 8
```

`lora_vanilla` (no init) + O-LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
    --sequence NI-Seq-Dummy \
    --run-name vanilla_olora_dev01 \
    --cl-method o_lora --cl-o-lora-lambda 0.5
```

`loram + InfLoRA`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
    --sequence NI-Seq-Dummy \
    --run-name loram_inflora_dev01 \
    --slice-init --slice-init-method loram \
    --cl-method inflora --cl-inflora-nullspace-rank 32
```

Tunables:

- `--cl-o-lora-lambda <float>`: orthogonality regularizer weight (default `0.5`).
- `--cl-inflora-nullspace-rank <int>`: top-k principal directions of past-task input covariance to project A out of (default `64`).
- `--cl-inflora-max-cov-batches <int>`: forward batches per stage used to estimate input covariance (default `32`).
- `--cl-inflora-cov-batch-size <int>`: batch size for those forward passes (default `8`).
- `--cl-sapt-key-dim <int>`: SAPT router key/query dim (default `64`).
- `--cl-sapt-arm-n-samples <int>`: pseudo-samples per adapter generated during ARM (default `64`).
- `--cl-sapt-arm-max-new-tokens <int>`: max tokens per pseudo-sample (default `32`).
- `--cl-sapt-arm-n-epochs <int>`: router training epochs at each stage (default `3`).
- `--cl-sapt-arm-batch-size <int>`: router optimization batch size (default `4`).
- `--cl-sapt-arm-learning-rate <float>`: router AdamW lr (default `1e-3`).
- `--cl-sapt-seed-prompts-per-task <int>`: training prompts cached per task as ARM seeds (default `32`).

### SAPT specifics

SAPT replaces the train+merge pipeline with parallel-adapter training:

- **No merge between stages.** Each stage's LoRA is loaded as a *named PEFT adapter* (`task_NN`) that stays live for the rest of the run. Base weights are never updated.
- **Slice absorption is disabled** under `--cl-method sapt`. With multiple parallel adapters sharing one base, repeated absorption would compound across stages. Slice still computes A/B from gradients (so `--slice-init`, `--slice-init-method slice`, etc. still work) but does not subtract from base. `init_correction.pt` is not written.
- **ARM (Attentive Reflection)** runs after each stage. Cached seed prompts from each prior task are run through that task's adapter (via `set_adapter("task_NN")`) to generate pseudo-samples; the router is trained with cross-entropy to route each sample to its source adapter.
- **Inference (orchestrator + `eval_standalone`)** wraps the parallel-adapter PEFT model in a `SAPTWrapper`. Forward / generate compute one routing distribution per input from the prompt embedding, then mix every adapter's `B_i A_i x` weighted by `α_i`. The wrapper monkey-patches `peft.tuners.lora.Linear.forward` once globally; the patch falls back to the default PEFT behavior unless a SAPT routing context is active, so non-SAPT runs are unaffected.
- **Checkpoint layout** under SAPT:

  ```
  results/<seq>/<run>/
  ├── checkpoints/
  │   ├── base_model/
  │   └── stage_NN_<task>/adapter/        # PEFT named adapter (task_NN)
  └── cl_state/sapt/
      ├── sapt_state.pt                    # adapter_names, seed_prompts, last_arm_stats
      └── router.pt                        # cumulative SAPTRouter state
  ```

  The `eval_manifest.json` written each stage gains `cl_method` and `sapt_router_path` fields; `eval_standalone` switches to `load_sapt_model` whenever those are present.

Composes with all four init choices. Examples:

```bash
# vanilla init + SAPT
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
    --sequence NI-Seq-Dummy --run-name sapt_vanilla_dev01 \
    --cl-method sapt --cl-sapt-arm-n-samples 32

# slice + SAPT (skip_absorption forced under SAPT)
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
    --sequence NI-Seq-Dummy --run-name sapt_slice_dev01 \
    --slice-init --slice-init-method slice --slice-max-steps 64 \
    --cl-method sapt

# lora_ga + SAPT
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
    --sequence NI-Seq-Dummy --run-name sapt_lora_ga_dev01 \
    --slice-init --slice-init-method lora_ga --slice-max-steps 64 \
    --cl-method sapt
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
