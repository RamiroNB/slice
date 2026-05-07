# Low-Rank Adapters Initialization via Gradient Surgery for Continual Learning

This repository contains the official, **anonymized** implementation accompanying
the NeurIPS submission *"Low-Rank Adapters Initialization via Gradient Surgery
for Continual Learning"* (paper ID withheld for double-blind review). It
reproduces the SLICE initializer, the LoRA initialization baselines (vanilla
LoRA, LoRA-GA, LoRAM), the optional training-time continual-learning methods
(O-LoRA, InfLoRA, SAPT), and the adversarial **NI-Seq-Opposite** task
sequences introduced in the paper.

> **Anonymity notice.** No author names, affiliations, GitHub handles, project
> pages, or external links that could de-anonymize the authors are included in
> this repository. All paths and run names are generic. Reviewers should
> consider all internal scripts and code comments as the authoritative
> description of the method.

## Repository Layout

```
cl_lora/
├── orchestrator.py        # End-to-end CL pipeline: train → merge → eval per stage
├── train.py               # Single-task LoRA training + adapter merging
├── eval.py                # Seen-task and general (GP/IP) evaluation
├── eval_standalone.py     # Re-run evaluation from saved checkpoints
├── load_dataset.py        # Super-NI / TRACE dataset loaders
├── task_sequences.py      # NI-Seq-{C,G,M,Opposite} and TRACE sequence definitions
├── metrics.py             # AP / FP / Forget / GP / IP computation
├── find_conflicting_seq.py# Mining script that produced NI-Seq-Opposite
├── slice/                 # SLICE init: gradient capture, projection, SVD, apply
│   ├── compute.py         #   - main entry point (compute / cache inits)
│   ├── gradients.py       #   - per-module gradient accumulation
│   ├── projections.py     #   - PCGrad / CAGrad / GradVac / Nullspace operators
│   ├── decompose.py       #   - truncated SVD & magnitude rescaling
│   └── apply.py           #   - inject (A, B) into LoRA layers
├── cl_methods/            # Composable training-time CL strategies
│   ├── vanilla.py         #   - baseline (train + merge per stage)
│   ├── o_lora.py          #   - O-LoRA orthogonality regularizer
│   ├── inflora.py         #   - InfLoRA covariance-nullspace projection
│   └── sapt.py            #   - SAPT parallel-adapter routing
└── repro.py               # Global RNG seeding

scripts/                   # Bash wrappers for the experiments in the paper
├── full_train_projection_variants.sh   # SLICE projection-variant sweep (Tab. 1)
├── test_init_x_cl_methods_lean.sh      # Init × CL-method composition matrix
├── alpha_sweep.sh                      # rsLoRA α ∈ {1, 2, 4} sweep (Appx. C)
├── eval.sh, parallel_eval.sh, ...      # Evaluation drivers
└── compute_sequence_metrics.sh         # NI-Seq-Opposite sequence mining

# Analysis / figures (read run artifacts under results/)
plot.py, plot_gp_curve.py, alpha_sweep_analysis.py,
results_analysis.py, tables_script.py, recompute_metrics.py
```

## Requirements

The code targets Python ≥ 3.10 and a CUDA 12.x-capable GPU (experiments in the
paper used a single 24 GB GPU per run). Install dependencies with:

```setup
pip install -r requirements.txt
```

`requirements.txt` pins the PyTorch CUDA 12.6 wheels and pulls
`transformers >= 4.46`, `peft`, `accelerate`, `datasets`, and the
`lm-evaluation-harness` package directly from its public Git repository (used
to compute GP / IP).

A reference Conda environment file (`environment.yml`) is also provided
for reproducibility.

## Training

### Quick smoke test

A two-stage dummy sequence runs in a few minutes and exercises the full
pipeline:

```train-smoke
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
  --sequence NI-Seq-Dummy \
  --general-eval-set core \
  --eval-size 10 \
  --task-eval-samples 5 \
  --task-eval-max-new-tokens 32 \
  --run-name smoke_test
```

### Full training with SLICE initialization

The command below reproduces a SLICE (PCGrad, c=1.0) run on
`NI-Seq-Opposite-v2`, using the global gradient-projection mode and per-task
retain batching that produced the FP and Forget gains reported in Table 1
of the paper:

```train-slice
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
  --sequence NI-Seq-Opposite-v2 \
  --run-name slice_pcgrad_c100_oppv2 \
  --rank 64 \
  --lora-alpha 2 \
  --slice-init \
  --slice-init-method slice \
  --slice-cache-dir slice_cache \
  --slice-max-steps 8 \
  --slice-grad-project \
  --slice-grad-projection-mode global \
  --slice-retain-batch-size-set each_task \
  --slice-projection-method pcgrad
```

Switch the projection operator with `--slice-projection-method` and the
PCGrad/CAGrad strength with `--slice-cagrad-c` (e.g. `--slice-projection-method
cagrad --slice-cagrad-c 0.50` for the c=0.5 variant in the main table).

### Baseline initializations

Replace `--slice-init-method` to run the LoRA baselines compared in the paper:

```train-baselines
# Vanilla LoRA (random Gaussian A, zero B) — drop --slice-init entirely
python -m cl_lora.orchestrator --sequence NI-Seq-Opposite-v2 --run-name vanilla_oppv2 --rank 64 --lora-alpha 2

# LoRA-GA (SVD of the current-task gradient only)
python -m cl_lora.orchestrator --sequence NI-Seq-Opposite-v2 --run-name lora_ga_oppv2 --rank 64 --lora-alpha 2 \
    --slice-init --slice-init-method lora_ga --slice-svd-selection lora_ga \
    --slice-cache-dir slice_cache --slice-max-steps 8

# LoRAM (DST-based deterministic initialization)
python -m cl_lora.orchestrator --sequence NI-Seq-Opposite-v2 --run-name loram_oppv2 --rank 64 --lora-alpha 2 \
    --slice-init --slice-init-method loram --slice-cache-dir slice_cache --slice-max-steps 8
```

All comparisons in the paper use **variance-matched magnitude rescaling** by
default (`--slice-init` automatically scales `(A, B)` so that
`Var(BA) = Var(W₀) · (log_m r)`, controlling for the magnitude confound noted
in the related work).

### Reproducing the paper sweeps

The following scripts reproduce the experiments end-to-end. They iterate
sequences × variants and write artifacts under `results/<sequence>/<run>/`:

```sweeps
# Table 1 (rank 64) — projection-variant sweep across G1, G2, TRACE, Opp-v2..v4
GPU=0 RANK=64 RUN_SUFFIX=projvariants \
    bash scripts/full_train_projection_variants.sh

# Appendix C — rsLoRA α ∈ {1, 2, 4} sweep
GPU=0 ALPHAS="1 2 4" METHODS="cagrad lora_ga" \
    SEQUENCES="NI-Seq-G2 NI-Seq-Opposite-v4 TRACE" \
    bash scripts/alpha_sweep.sh

# Composition matrix: every init × every CL training method
GPU=0 SEQUENCES="NI-Seq-G2" RUN_SUFFIX=lean01 \
    bash scripts/test_init_x_cl_methods_lean.sh
```

Each script accepts `--resume` (passed through to the orchestrator) so
interrupted runs can be picked up where they left off.

### Adversarial sequence mining (NI-Seq-Opposite)

To regenerate the adversarial 5-task sequences described in Appendix B:

```mine
bash scripts/compute_sequence_metrics.sh
# or directly:
python -m cl_lora.find_conflicting_seq --pool-size 46 --sequence-length 5
```

This script computes per-task gradients for the 46-task Super-NI candidate
pool, sketches them with a CountSketch (k = 200,000), and exhaustively scores
all `C(46, 5) = 1,370,754` length-5 subsets by mean pairwise gradient cosine.

### Default hyperparameters

| Hyperparameter | Default | Where set |
|---|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` | [cl_lora/train.py:57](cl_lora/train.py#L57) |
| LoRA rank `r` | 128 (paper main: 64) | `--rank` |
| LoRA `α` (rsLoRA scaling = α / √r) | 2 | `--lora-alpha` |
| Learning rate | 1e-4 | [cl_lora/train.py:342](cl_lora/train.py#L342) |
| Epochs per task | 3 | [cl_lora/train.py:343](cl_lora/train.py#L343) |
| Per-device train batch size | 16 | [cl_lora/train.py:344](cl_lora/train.py#L344) |
| SLICE accumulation steps `S_cur`, `S_prev` | 8 | `--slice-max-steps` |
| Random seed | 42 | `--seed` |

## Evaluation

Evaluation is integrated into the orchestrator: after each stage it runs
seen-task evaluation (exact-match / ROUGE-L on the held-out splits) and, by
default, a fixed set of general-capability benchmarks (HellaSwag,
CommonsenseQA, BBH-ObjectCounting, Alpaca) for **GP** (zero-shot) and **IP**
(few-shot, n=5; n=3 for BBH).

### Evaluating from saved checkpoints

The recommended workflow for paper experiments is to **train on a GPU machine
and evaluate separately** to free GPU time and parallelize:

```eval-saved
# 1. Train only (saves base model + per-stage adapters under results/)
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.orchestrator \
  --sequence NI-Seq-G2 --run-name slice_g2 \
  --slice-init --slice-init-method slice --train-only --keep-all-checkpoints

# 2. Evaluate every stage from the saved adapters
CUDA_VISIBLE_DEVICES=0 python -m cl_lora.eval_standalone run \
  --run-dir results/NI-Seq-G2/slice_g2 \
  --task-eval-samples 64
```

Standalone eval reconstructs the model at stage `k` by loading the base model
once and merging adapters `1..k` in order — numerically identical to the
in-memory merged model used during training. Adapters initialized with
`--slice-init` additionally store `init_correction.pt` so the absorption step
can be replayed faithfully.

For batch evaluation across many runs:

```eval-batch
GPU=0 bash scripts/parallel_eval.sh        # parallelize across runs
GPU=0 bash scripts/eval.sh <run-dir>       # evaluate a single run-dir
```

### Computing CL metrics

After evaluation, the orchestrator writes:

- `results/<seq>/<run>/results_matrix.json` — the lower-triangular `R[i, j]`
  matrix (score on task `j` after training through task `i`)
- `results/<seq>/<run>/metrics.json` — AP, FP, Forget, GP, IP

To recompute aggregate metrics from raw stage records (e.g. after re-running
just GP/IP):

```recompute
python recompute_metrics.py
```

The metric definitions are in [cl_lora/metrics.py](cl_lora/metrics.py) and
match the formulas in Appendix A of the paper.

## Pre-trained Models

**No pretrained adapters or full model checkpoints are released with this
supplementary submission.** Reasons:

1. The supplementary ZIP is capped at 100 MB; a single rank-64 adapter
   set across 5 stages exceeds that.
2. Releasing run artifacts could expose internal storage paths and reviewer
   metadata that risk de-anonymization.
3. The base model is licensed and must be downloaded directly from the Hugging
   Face Hub by each reviewer.

## Results

The headline results are reproduced below. The full tables (including the
`(c=0.5)`, `(c=0.75)`, `(c=1.0)` SLICE variants, GP / IP preservation, and
rank-128 numbers) are in the paper.

### Continual-learning metrics (rank 64, AP / FP / Forget)

Baseline rows report **absolute** AP / FP / Forget values; each indented
SLICE row reports the **Δ** relative to the baseline immediately above it.
↑ on AP / FP and ↓ on Fgt indicate the favorable direction. Reproduce by
running `scripts/full_train_projection_variants.sh` and aggregating with
`python tables_script.py`.

| Method | G1 AP ↑ | G1 FP ↑ | G1 Fgt ↓ | G2 AP ↑ | G2 FP ↑ | G2 Fgt ↓ | TRACE AP ↑ | TRACE FP ↑ | TRACE Fgt ↓ | Opp1 AP ↑ | Opp1 FP ↑ | Opp1 Fgt ↓ | Opp2 AP ↑ | Opp2 FP ↑ | Opp2 Fgt ↓ | Opp3 AP ↑ | Opp3 FP ↑ | Opp3 Fgt ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Vanilla LoRA** | 10.29 | 11.06 | −0.77 | 18.83 | 9.04 | 9.79 | 10.21 | 10.28 | −0.06 | 27.27 | 12.53 | 14.74 | 26.50 | 25.10 | 1.40 | 26.32 | 4.32 | 22.00 |
| *SLICE (c=0.50)* | −0.75 | −2.03 | +1.28 | **+16.87** | **+22.71** | **−5.84** | **+3.04** | **+3.35** | **−0.31** | −3.44 | **+10.98** | **−14.43** | −0.98 | −10.72 | +9.75 | −3.50 | **+5.69** | **−9.18** |
| *SLICE (c=0.75)* | −0.89 | −1.99 | +1.10 | **+18.62** | **+14.25** | +4.36 | **+6.44** | **+5.28** | +1.16 | −4.69 | **+9.04** | **−13.73** | **+0.46** | −8.33 | +8.79 | −2.51 | **+18.24** | **−20.75** |
| *SLICE (c=1.00)* | **+1.27** | −1.00 | +2.27 | **+18.48** | **+22.55** | **−4.07** | **+2.59** | **+4.29** | **−1.69** | −1.73 | **+16.97** | **−18.70** | −2.45 | −10.86 | +8.41 | −0.80 | **+12.25** | **−13.04** |
| **LoRAM** | 11.89 | 10.00 | 1.89 | 21.58 | 5.69 | 15.88 | 10.16 | 8.91 | 1.26 | 27.22 | 24.25 | 2.97 | 24.85 | 31.58 | −6.73 | 24.00 | 12.95 | 11.05 |
| *SLICE (c=0.50)* | −2.35 | −0.96 | **−1.38** | **+14.13** | **+26.06** | **−11.93** | **+3.09** | **+4.72** | **−1.63** | −3.40 | −0.74 | **−2.66** | **+0.67** | −17.20 | +17.88 | −1.17 | −2.95 | +1.77 |
| *SLICE (c=0.75)* | −2.48 | −0.92 | **−1.56** | **+15.88** | **+17.61** | **−1.73** | **+6.49** | **+6.65** | **−0.16** | −4.64 | −2.68 | **−1.96** | **+2.11** | −14.82 | +16.92 | −0.19 | **+9.61** | **−9.80** |
| *SLICE (c=1.00)* | −0.32 | **+0.07** | **−0.39** | **+15.74** | **+25.91** | **−10.16** | **+2.64** | **+5.65** | **−3.01** | −1.68 | **+5.24** | **−6.93** | −0.80 | −17.35 | +16.54 | **+1.53** | **+3.61** | **−2.09** |
| **LoRA-GA** | 11.75 | 10.91 | 0.84 | 36.99 | 32.81 | 4.19 | 14.48 | 14.47 | 0.02 | 22.76 | 17.62 | 5.14 | 24.87 | 16.51 | 8.37 | 21.29 | 14.52 | 6.77 |
| *SLICE (c=0.50)* | −2.21 | −1.88 | **−0.33** | −1.29 | −1.06 | **−0.23** | −1.23 | −0.84 | **−0.39** | **+1.06** | **+5.89** | **−4.83** | **+0.65** | −2.13 | +2.78 | **+1.54** | −4.52 | +6.05 |
| *SLICE (c=0.75)* | −2.35 | −1.84 | **−0.51** | **+0.46** | −9.51 | +9.97 | **+2.17** | **+1.09** | +1.08 | −0.18 | **+3.95** | **−4.12** | **+2.08** | **+0.26** | +1.82 | **+2.52** | **+8.04** | **−5.52** |
| *SLICE (c=1.00)* | −0.19 | −0.85 | +0.67 | **+0.33** | −1.21 | +1.54 | −1.68 | **+0.10** | **−1.78** | **+2.77** | **+11.87** | **−9.10** | −0.83 | −2.27 | +1.44 | **+4.23** | **+2.04** | +2.19 |

Reading the table:
- **Bold-positive ΔAP / ΔFP** and **bold-negative ΔFgt** entries are the
  improvements that a SLICE variant achieves over its base initializer
  (corresponding to the green cells in the LaTeX version of Table 1 in the
  paper).
- The most pronounced gains over Vanilla LoRA appear on the adversarial
  sequences `Opp1` and `Opp3`, where SLICE (c=1.00) improves FP by **+16.97**
  and reduces Forget by **−18.70** on Opp1, and improves FP by **+12.25** /
  reduces Forget by **−13.04** on Opp3.
- Even when stacked on the already-strong LoRA-GA baseline, SLICE recovers
  additional **+11.87 FP** / **−9.10 Fgt** on Opp1 (c=1.00).

## Available Sequences

Defined in [cl_lora/task_sequences.py](cl_lora/task_sequences.py):

- **Standard Super-NI** (Jiang et al. 2025 grouping):
  `NI-Seq-C1`, `NI-Seq-C2`, `NI-Seq-G1`, `NI-Seq-G2`, `NI-Seq-M1`, `NI-Seq-M2`
- **Adversarial Super-NI (ours, NI-Seq-Opposite)**:
  `NI-Seq-Opposite-v1` through `NI-Seq-Opposite-v7` — 5-task sequences mined
  to minimize mean pairwise gradient cosine across the 46-task pool. The
  paper's main experiments use `Opp-v2`, `Opp-v3`, `Opp-v4` (rows 1–3 of
  Table 3). The full task IDs of every Opposite sequence are listed in
  Appendix B of the paper.
- **TRACE benchmark**: `TRACE` (6 tasks: C-STANCE, FOMC, MeetingBank, Py150,
  ScienceQA, NumGLUE-cm). Requires a local TRACE download — see
  *External assets* above.
- **Smoke tests**: `NI-Seq-Dummy`, `TRACE-Dummy`.

## Reproducibility Checklist

- All entry points accept `--seed` (default `42`). Global RNGs are seeded in
  [cl_lora/repro.py](cl_lora/repro.py).
- Every run writes a `run_summary.json` with the resolved CLI namespace,
  resolved package versions, and a reduced-form configuration hash so
  configurations are auditable.
- The SLICE init cache (`slice_cache/`) is keyed on `(forget_task,
  retain_tasks, model, rank, projection method, c, …)`; reusing the same
  configuration across runs deterministically reuses the cached
  initialization rather than recomputing.
- A `repro.py` helper enforces deterministic CUDA settings where possible
  (note that some HF Trainer paths still introduce nondeterminism — re-runs
  will yield very small numerical differences).

## Contributing

This repository is provided **for review** and reproduction of the paper's
results. Until the review process completes, no external contributions are
solicited. After acceptance / decision, the de-anonymized release will adopt a
permissive open-source license, and contributions will be welcomed through
issues and pull requests.