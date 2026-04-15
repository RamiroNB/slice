from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .eval import evaluate_fairness_task
    from ..train import MODEL_NAME, HF_TOKEN, build_tokenizer, load_base_model, train_on_task
except ImportError:
    from eval import evaluate_fairness_task
    from train import MODEL_NAME, HF_TOKEN, build_tokenizer, load_base_model, train_on_task


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return str(value)


def _resolve_method(method: str) -> dict[str, Any]:
    normalized = method.strip().lower()
    if normalized == "vanilla":
        return {
            "slice_enabled": False,
            "slice_grad_project": False,
            "slice_grad_projection_mode": "per_module",
            "slice_add_retain_grad": False,
        }
    if normalized == "slice":
        return {
            "slice_enabled": True,
            "slice_grad_project": False,
            "slice_grad_projection_mode": "per_module",
            "slice_add_retain_grad": False,
        }
    if normalized == "slice_proj_per_module":
        return {
            "slice_enabled": True,
            "slice_grad_project": True,
            "slice_grad_projection_mode": "per_module",
            "slice_add_retain_grad": False,
        }
    if normalized == "slice_proj_global":
        return {
            "slice_enabled": True,
            "slice_grad_project": True,
            "slice_grad_projection_mode": "global",
            "slice_add_retain_grad": False,
        }
    raise ValueError(
        f"Unknown method '{method}'. "
        "Use one of: vanilla, slice, slice_proj_per_module, slice_proj_global"
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    base_model_eval: bool = getattr(args, "base_model_eval", False)
    method_name = "base_model" if base_model_eval else args.method
    method_cfg = {} if base_model_eval else _resolve_method(args.method)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.task}_{method_name}_s{args.seed}_{timestamp}"

    run_dir = Path(args.output_root) / args.task / method_name / run_name
    train_dir = run_dir / "train"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(model_name=args.model_name, hf_token=HF_TOKEN)
    model = load_base_model(model_name=args.model_name, hf_token=HF_TOKEN)

    train_report: dict[str, Any] | None = None
    if base_model_eval:
        # Evaluate the base model directly without any fine-tuning.
        eval_model = model
    else:
        eval_model, train_report = train_on_task(
            model=model,
            tokenizer=tokenizer,
            task=args.task,
            output_dir=str(train_dir),
            rank=args.rank,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            max_seq_length=args.max_seq_length,
            eval_size=args.eval_size,
            seed=args.seed,
            use_bf16=args.use_bf16,
            save_adapter=True,
            slice_enabled=method_cfg["slice_enabled"],
            slice_cache_dir=args.slice_cache_dir,
            slice_max_steps=args.slice_max_steps,
            slice_retain_scale=args.slice_retain_scale,
            slice_grad_project=method_cfg["slice_grad_project"],
            slice_grad_projection_mode=method_cfg["slice_grad_projection_mode"],
            slice_add_retain_grad=method_cfg["slice_add_retain_grad"],
            slice_retain_batch_size=args.slice_retain_batch_size,
            slice_retain_grad_accum=args.slice_retain_grad_accum,
            slice_retain_batch_size_set=args.slice_retain_batch_size_set,
            slice_single_retain_task_mode=args.slice_single_retain_task_mode,
        )

    predictions_path = run_dir / "predictions.jsonl"
    eval_report = evaluate_fairness_task(
        model=eval_model,
        tokenizer=tokenizer,
        task=args.task,
        eval_size=args.eval_size,
        task_eval_samples=args.task_eval_samples,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_new_tokens=args.task_eval_max_new_tokens,
        max_input_length=args.task_eval_max_input_length,
        seed=args.seed,
        min_group_count=args.min_group_count,
        predictions_output_path=str(predictions_path),
    )

    if not base_model_eval and args.save_merged_model:
        merged_dir = run_dir / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        eval_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    summary = {
        "task": args.task,
        "method": method_name,
        "seed": int(args.seed),
        "model_name": args.model_name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "train_output_dir": str(train_dir) if not base_model_eval else None,
        "predictions_path": str(predictions_path),
        "metrics": eval_report,
        "train_report": train_report,
        "config": {
            "rank": args.rank if not base_model_eval else None,
            "learning_rate": args.learning_rate if not base_model_eval else None,
            "num_train_epochs": args.num_train_epochs if not base_model_eval else None,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps if not base_model_eval else None,
            "max_seq_length": args.max_seq_length,
            "eval_size": args.eval_size,
            "task_eval_samples": args.task_eval_samples,
            "task_eval_max_new_tokens": args.task_eval_max_new_tokens,
            "task_eval_max_input_length": args.task_eval_max_input_length,
            "min_group_count": args.min_group_count,
            "base_model_eval": base_model_eval,
            "method_cfg": method_cfg if not base_model_eval else None,
        },
    }

    summary_path = run_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(summary), f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone fairness-vs-accuracy initialization benchmark runner."
    )
    parser.add_argument(
        "--task",
        default="bbq",
        choices=["bbq", "fairness_bbq", "winobias", "wino_bias", "difference_awareness"],
        help="Fairness task alias to train/evaluate.",
    )
    parser.add_argument(
        "--method",
        default="vanilla",
        choices=["vanilla", "slice", "slice_proj_per_module", "slice_proj_global"],
        help="Initialization strategy.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-root", default="results/fairness")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--task-eval-samples", type=int, default=128)
    parser.add_argument("--task-eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--task-eval-max-input-length", type=int, default=512)

    parser.add_argument("--slice-cache-dir", default="slice_cache")
    parser.add_argument("--slice-max-steps", type=int, default=100)
    parser.add_argument("--slice-retain-scale", type=float, default=1.0)
    parser.add_argument("--slice-retain-batch-size", type=int, default=None)
    parser.add_argument("--slice-retain-grad-accum", type=int, default=None)
    parser.add_argument(
        "--slice-retain-batch-size-set",
        choices=["all_tasks", "each_task"],
        default="all_tasks",
    )
    parser.add_argument("--slice-single-retain-task-mode", action="store_true")

    parser.add_argument(
        "--base-model-eval",
        action="store_true",
        help="Skip training and evaluate the base model directly (zero-shot baseline).",
    )
    parser.add_argument(
        "--min-group-count",
        type=int,
        default=1,
        help="Minimum group size to include in WGA and GAP computation. Use 20+ for real runs.",
    )
    parser.add_argument("--save-merged-model", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")

    args = parser.parse_args()
    args.use_bf16 = not args.no_bf16

    summary = run_benchmark(args)
    print(json.dumps(_to_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
