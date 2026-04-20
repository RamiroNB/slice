# Running Training and Evaluation Separately

This guide explains how to decouple the training and evaluation phases of a
continual-learning run so that they can be executed independently — for
example on different machines, at different times, or with different eval
hyperparameters without re-running expensive training.

---

## How it works

The orchestrator (`cl_lora/orchestrator.py`) saves the merged model checkpoint
**before** running evaluation at each stage.  It also writes an
`eval_manifest.json` file that records everything needed to reproduce the
evaluation step.  The standalone eval script (`cl_lora/eval_standalone.py`)
reads that manifest and runs `evaluate_all()`, producing a `stage_record.json`
identical in format to what the orchestrator produces.

```
results/<sequence>/<run-name>/
├── run_config.json
├── stage_records.partial.json
├── checkpoints/
│   ├── stage_01_<task>/
│   │   └── merged_model/          ← saved BEFORE eval; needed by eval_standalone
│   └── stage_02_<task>/
│       └── merged_model/
├── stages/
│   ├── stage_01_<task>/
│   │   ├── eval_manifest.json     ← written BEFORE eval
│   │   └── stage_record.json      ← written AFTER eval
│   └── stage_02_<task>/
│       ├── eval_manifest.json
│       └── stage_record.json
└── ...
```

### Critical: keep all checkpoints

By default the orchestrator **deletes each stage's checkpoint** after the next
stage finishes training (to save disk space).  If you plan to run eval
separately you must pass `--keep-all-checkpoints` at training time, otherwise
only the most recent stage's checkpoint will be available:

```bash
# Without this flag, stage_01 checkpoint is deleted once stage_02 trains.
# eval_standalone run will then fail for stage_01.
--keep-all-checkpoints
```


---

## For dummies: copying files to an eval machine

After a `--train-only` run you need to transfer two directories to the eval
machine: the **checkpoints** (model weights) and the **stages** (eval
manifests).  The code itself also needs to be present on the eval machine.

### What to transfer

```
results/<sequence>/<run-name>/
├── checkpoints/          ← model weights (~GB per stage, transfer once)
└── stages/               ← eval_manifest.json files (tiny, KB total)
```

You do NOT need to transfer `run_config.json`, `stage_records.partial.json`,
or any training outputs under `outputs/`.

### Transfer with rsync (recommended)

rsync is faster than scp for large directories because it skips files that
are already present and shows progress.

```bash
# Run this on the TRAINING machine, pushing to the eval machine.
# Replace user@eval-host and /path/on/eval with your values.

TRAIN_RUN=results/NI-Seq-G1/my_train_run
EVAL_HOST=user@eval-host
EVAL_DEST=/path/on/eval/results/NI-Seq-G1/my_train_run

# Transfer checkpoints (large — model weights)
rsync -avz --progress \
    $TRAIN_RUN/checkpoints/ \
    $EVAL_HOST:$EVAL_DEST/checkpoints/

# Transfer stage manifests (small — JSON only)
rsync -avz --progress \
    $TRAIN_RUN/stages/ \
    $EVAL_HOST:$EVAL_DEST/stages/
```

Or transfer everything in one go:

```bash
rsync -avz --progress \
    --include="checkpoints/***" \
    --include="stages/***" \
    --exclude="*" \
    $TRAIN_RUN/ \
    $EVAL_HOST:$EVAL_DEST/
```

### Transfer with scp

scp is simpler but has no resume support — if the connection drops you start
over.  Use rsync for anything over a few GB.

```bash
TRAIN_RUN=results/NI-Seq-G1/my_train_run
EVAL_HOST=user@eval-host
EVAL_DEST=/path/on/eval/results/NI-Seq-G1/my_train_run

# -r copies recursively
scp -r $TRAIN_RUN/checkpoints $EVAL_HOST:$EVAL_DEST/
scp -r $TRAIN_RUN/stages      $EVAL_HOST:$EVAL_DEST/
```

### Transfer the code

The eval machine needs the same version of `cl_lora/` that was used for
training.  The simplest approach is to clone/pull the repo there.  If the
machines don't share a git remote you can rsync the source directly:

```bash
rsync -avz --progress \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    /mnt/E-SSD/dev-cl-lora/cl-lora/cl_lora/ \
    $EVAL_HOST:/path/on/eval/cl-lora/cl_lora/
```

### Run eval on the eval machine

```bash
# On the eval machine, from the repo root
python -m cl_lora.eval_standalone run \
    --run-dir /path/on/eval/results/NI-Seq-G1/my_train_run
```

> **Note about paths in eval_manifest.json:** The manifest stores the
> checkpoint path as an absolute path from the training machine.  On the eval
> machine that path won't exist, but `eval_standalone` handles this
> automatically: if the stored path is not found it derives the checkpoint
> location from the stage directory itself
> (`<run_dir>/checkpoints/<stage_name>/merged_model`), which is correct as
> long as the directory structure was preserved by rsync/scp.

---

## Workflow A — Training only, then eval later

### Step 1: Run training without any evaluation

Pass `--train-only` to skip all evaluation.  Also pass `--keep-all-checkpoints`
so every stage's merged model is preserved for the eval machine.

```bash
python -m cl_lora.orchestrator \
    --sequence NI-Seq-G1 \
    --run-name my_train_run \
    --train-only \
    --keep-all-checkpoints
```

At each stage the orchestrator saves:
- `checkpoints/stage_XX_<task>/merged_model/` — the merged model and tokenizer
- `stages/stage_XX_<task>/eval_manifest.json` — all parameters needed for eval

### Step 2: Re-run full evaluation on saved checkpoints

```bash
python -m cl_lora.eval_standalone run \
    --run-dir results/NI-Seq-G1/my_train_run
```

This iterates over every `stages/stage_*/` directory that has an
`eval_manifest.json`, loads the corresponding model, runs `evaluate_all()`,
writes `stage_record.json`, and finally rebuilds `results_matrix.json` and
`metrics.json`.

---

## Workflow B — Evaluating a single stage

```bash
python -m cl_lora.eval_standalone stage \
    --stage-dir results/NI-Seq-G1/my_train_run/stages/stage_02_task618_amazonreview_summary_text_generation
```

All parameters are read from `eval_manifest.json`.  Any can be overridden:

```bash
python -m cl_lora.eval_standalone stage \
    --stage-dir results/NI-Seq-G1/my_train_run/stages/stage_02_... \
    --skip-general-eval \
    --task-eval-samples 128 \
    --seed 0
```

> **Note:** If only the latest stage checkpoint exists (default cleanup
> behaviour), only that stage can be evaluated with this command.

---

## Workflow C — Train a single task with train.py, then eval

### Step 1: Train one task and save the merged model

```bash
python -m cl_lora.train \
    --task task363_sst2_polarity_classification \
    --output-dir outputs/my_single_task \
    --save-merged-model \
    --rank 128 \
    --seed 42
```

Outputs:
```
outputs/my_single_task/
├── adapter/               ← LoRA adapter weights
├── training_report.json   ← training metrics
├── trainer_state.json
└── merged_model/          ← merged model + tokenizer
```

### Step 2: Evaluate the merged model

```bash
python -m cl_lora.eval_standalone stage \
    --stage-dir my_eval_output \
    --model-path outputs/my_single_task/merged_model \
    --sequence NI-Seq-G1 \
    --eval-seen-tasks task363_sst2_polarity_classification \
    --seen-tasks task363_sst2_polarity_classification \
    --seed 42
```

Results are written to `my_eval_output/stage_record.json`.

---

## Workflow D — Recompute run-level summary metrics only

If all `stage_record.json` files already exist and you only need to rebuild
`results_matrix.json` / `metrics.json`:

```bash
python -m cl_lora.eval_standalone summary \
    --run-dir results/NI-Seq-G1/my_train_run
```

---

## Troubleshooting

### Only the latest stage checkpoint exists

Without `--keep-all-checkpoints`, the orchestrator deletes each stage's
checkpoint after the next stage trains.  For a completed 2-stage run:

```
checkpoints/
└── stage_02_<task>/merged_model/   ← exists
    stage_01_<task>/                ← deleted
```

You can only eval stage_02.  To eval all stages you must re-run training
with `--keep-all-checkpoints`.

---

## eval_standalone.py sub-commands reference

| Sub-command | Purpose |
|-------------|---------|
| `stage` | Evaluate one stage from its stage directory |
| `run`   | Evaluate all stages in a run directory |
| `summary` | Recompute metrics from existing stage_record.json files |

### Common options (stage and run)

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-general-eval` | false | Skip GP/IP lm-eval evaluation |
| `--quick-eval` | false | Perplexity-only seen-task eval, skips generation |
| `--eval-size` | from manifest | Dataset eval split size |
| `--task-eval-samples` | from manifest | Max samples per seen task |
| `--task-eval-max-new-tokens` | from manifest | Generation budget per sample |
| `--seed` | from manifest | RNG seed |
| `--general-eval-batch-size` | 8 | Batch size for lm-eval |

### stage-only options

| Flag | Description |
|------|-------------|
| `--stage-dir` | Path to the stage directory (required) |
| `--model-path` | Override the model path from the manifest |
| `--sequence` | Override the sequence name |
| `--seen-tasks` | Override the full seen-task list |
| `--eval-seen-tasks` | Override which seen tasks to actually evaluate |
| `--general-eval-keys` | Override the general eval task keys |

---

## File format: eval_manifest.json

Written by the orchestrator to `stages/stage_XX_<task>/eval_manifest.json`
before each evaluation step.  All paths are stored as absolute paths.

```json
{
  "stage": 1,
  "trained_task": "task363_sst2_polarity_classification",
  "sequence": "NI-Seq-G1",
  "seen_tasks": ["task363_sst2_polarity_classification"],
  "eval_seen_tasks": ["task363_sst2_polarity_classification"],
  "model_path": "/absolute/path/to/checkpoints/stage_01_.../merged_model",
  "general_eval_keys": ["hellaswag", "commonsenseqa", "bbh_object_counting"],
  "skip_general_eval": false,
  "quick_eval": false,
  "eval_size": 200,
  "task_eval_samples": 64,
  "task_eval_max_new_tokens": 64,
  "seed": 42,
  "train_output_dir": "/absolute/path/to/outputs/NI-Seq-G1/stage_01_..."
}
```

## File format: stage_record.json

Written by either the orchestrator or `eval_standalone stage/run`.

```json
{
  "stage": 1,
  "trained_task": "task363_sst2_polarity_classification",
  "train_report": { "task_name": "...", "train_metrics": {}, "eval_metrics": {} },
  "seen_tasks": {
    "task363_sst2_polarity_classification": {
      "score": 0.85,
      "primary_metric": "exact_match",
      "exact_match": 0.85,
      "rouge_l": 0.88,
      "n_samples": 64
    }
  },
  "general": {
    "gp": { "hellaswag": 0.72, "commonsenseqa": 0.65 },
    "ip": { "hellaswag": 0.74, "commonsenseqa": 0.67 },
    "gp_mean": 0.685,
    "ip_mean": 0.705
  }
}
```
