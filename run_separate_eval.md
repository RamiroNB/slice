# Running Training and Evaluation Separately

This guide explains how to decouple the training and evaluation phases of a
continual-learning run so that they can be executed independently — for
example on different machines, at different times, or with different eval
hyperparameters without re-running expensive training.

---

## How it works

The orchestrator (`cl_lora/orchestrator.py`) now saves the merged model
checkpoint **before** running evaluation at each stage.  It also writes an
`eval_manifest.json` file alongside the checkpoint that records everything
needed to reproduce the evaluation step:

```
results/<sequence>/<run-name>/
├── run_config.json
├── stage_records.partial.json
├── checkpoints/
│   └── stage_01_<task>/
│       └── merged_model/          ← Hugging Face model + tokenizer
├── stages/
│   └── stage_01_<task>/
│       ├── eval_manifest.json     ← written BEFORE eval
│       └── stage_record.json      ← written AFTER eval
└── ...
```

The standalone eval script (`cl_lora/eval_standalone.py`) reads
`eval_manifest.json` and runs `evaluate_all()`, producing a
`stage_record.json` that is identical in format to the one the orchestrator
produces, making it fully compatible with `recompute_metrics.py` and the
rest of the analysis pipeline.

---

## Workflow A — Training only, then eval later

### Step 1: Run training without any evaluation

Pass `--train-only` to skip all evaluation entirely.  The orchestrator will
still save a merged-model checkpoint and an `eval_manifest.json` at each
stage so that the eval machine has everything it needs.

```bash
python -m cl_lora.orchestrator \
    --sequence NI-Seq-G1 \
    --run-name my_train_run \
    --train-only
```

Because checkpoints and `eval_manifest.json` files are written **before** the
eval step would run, they are safe to use even if a run is interrupted.

### Step 2: Re-run full evaluation on saved checkpoints

```bash
python -m cl_lora.eval_standalone run \
    --run-dir results/NI-Seq-G1/my_train_run
```

This iterates over every `stages/stage_*/` directory that contains an
`eval_manifest.json`, loads the corresponding saved model from
`checkpoints/stage_*/merged_model/`, runs the full `evaluate_all()`, and
writes `stage_record.json`.  After all stages complete it rebuilds
`results_matrix.json` and `metrics.json`.

---

## Workflow B — Evaluating a single stage

```bash
python -m cl_lora.eval_standalone stage \
    --stage-dir results/NI-Seq-G1/my_train_run/stages/stage_03_task292_storycommonsense_character_text_generation
```

All parameters (model path, seen tasks, eval settings) are read from
`eval_manifest.json` in that directory.  Any parameter can be overridden:

```bash
python -m cl_lora.eval_standalone stage \
    --stage-dir results/NI-Seq-G1/my_train_run/stages/stage_03_... \
    --skip-general-eval \          # skip GP/IP for speed
    --task-eval-samples 128 \      # evaluate more samples
    --seed 0
```

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
└── merged_model/          ← merged model + tokenizer (created by --save-merged-model)
```

### Step 2: Evaluate the single merged model

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

If all `stage_record.json` files already exist and you only need to
recompute `results_matrix.json` / `metrics.json`:

```bash
python -m cl_lora.eval_standalone summary \
    --run-dir results/NI-Seq-G1/my_train_run
```

This is equivalent to `recompute_metrics.py` but operates on the full
stage-record structure including GP/IP.

---

## eval_standalone.py sub-commands reference

| Sub-command | Purpose |
|-------------|---------|
| `stage` | Evaluate one stage from its stage directory |
| `run`   | Evaluate all stages in a full run directory |
| `summary` | Recompute run-level metrics from existing stage_record.json files |

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
| `--model-path` | Override the model path stored in the manifest |
| `--sequence` | Override the sequence name |
| `--seen-tasks` | Override the full seen-task list |
| `--eval-seen-tasks` | Override which seen tasks to actually evaluate |
| `--general-eval-keys` | Override the general eval task keys |

---

## File format: eval_manifest.json

Written by the orchestrator to `stages/stage_XX_<task>/eval_manifest.json`
before each evaluation step.

```json
{
  "stage": 1,
  "trained_task": "task363_sst2_polarity_classification",
  "sequence": "NI-Seq-G1",
  "seen_tasks": ["task363_sst2_polarity_classification"],
  "eval_seen_tasks": ["task363_sst2_polarity_classification"],
  "model_path": "results/NI-Seq-G1/run_xyz/checkpoints/stage_01_.../merged_model",
  "general_eval_keys": ["hellaswag", "commonsenseqa", "bbh_object_counting"],
  "skip_general_eval": false,
  "quick_eval": false,
  "eval_size": 200,
  "task_eval_samples": 64,
  "task_eval_max_new_tokens": 64,
  "seed": 42,
  "train_output_dir": "outputs/NI-Seq-G1/stage_01_..."
}
```

## File format: stage_record.json

Written by either the orchestrator or `eval_standalone stage/run` to
`stages/stage_XX_<task>/stage_record.json`.

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
