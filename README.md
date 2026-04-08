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

## Slice LoRA Initialization

The slice initializer performs a backward pass over the current task data
and (optionally) the previous task to seed LoRA A/B matrices. It reuses the
same collator used in training and caches initializations per task pair.

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
Example (continual run with slice-projected init enabled):

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
	--slice-rank 8 \
	--slice-grad-project \
	--slice-grad-projection-mode per_module \
	--log-level DEBUG
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

## Available Sequences

Defined in cl_lora/task_sequences.py:

- NI-Seq-C1
- NI-Seq-C2
- NI-Seq-G1
- NI-Seq-G2
- NI-Seq-M1
- NI-Seq-M2
- NI-Seq-Dummy
- TRACE-Dummy
- TRACE


# TODO 
VERIFY...

Change stage checkpoint to not save model every time, or not save it on other path




quick eval: 
```
CUDA_VISIBLE_DEVICES=1 python -m cl_lora.orchestrator \
  --sequence NI-Seq-Dummy \
  --general-eval-set core \
  --eval-size 10 \
  --task-eval-samples 5 \
  --run-name quick_eval_dev01 \
  --quick-eval
```