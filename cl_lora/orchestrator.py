from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import logging

try:
    from .eval import evaluate_all
    from .metrics import compute_cl_metrics
    from .task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from .train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model, train_on_task
except ImportError:
    from eval import evaluate_all
    from metrics import compute_cl_metrics
    from task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model, train_on_task


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_sequence(
    sequence_name: str,
    model_name: str,
    run_output_dir: Path,
    train_output_dir: Path,
    general_eval_keys: List[str],
    eval_size: int,
    task_eval_samples: int,
    task_eval_max_new_tokens: int,
    save_final_model: bool,
    resume: bool,
    slice_enabled: bool,
    slice_cache_dir: str,
    slice_max_steps: int,
    slice_retain_scale: float,
    slice_rank: int | None,
) -> Dict[str, Any]:
    sequence = get_sequence(sequence_name)
    task_order = [task.name for task in sequence.tasks]

    partial_path = run_output_dir / "stage_records.partial.json"
    checkpoint_root = run_output_dir / "checkpoints"

    if partial_path.exists() and not resume:
        raise ValueError(
            f"Found existing partial state at {partial_path}. "
            "Use --resume to continue this run or choose a different --run-name."
        )

    stage_records: List[Dict[str, Any]] = []
    start_stage = 1
    seen_tasks = []

    if resume and partial_path.exists():
        partial = _read_json(partial_path)
        if partial.get("sequence") != sequence_name:
            raise ValueError("Resume failed: sequence name does not match saved partial state.")
        if partial.get("task_order") != task_order:
            raise ValueError("Resume failed: task order does not match saved partial state.")

        stage_records = partial.get("stage_records", [])
        completed = len(stage_records)
        start_stage = completed + 1
        seen_tasks = sequence.tasks[:completed]

        if completed >= len(sequence.tasks):
            summary = compute_cl_metrics(stage_records=stage_records, task_order=task_order)
            final_payload = {
                "sequence": sequence_name,
                "description": sequence.description,
                "task_order": task_order,
                "general_eval_keys": general_eval_keys,
                "stage_records": stage_records,
                "summary": summary,
            }
            _write_json(run_output_dir / "results_matrix.json", summary["results_matrix"])
            _write_json(run_output_dir / "metrics.json", summary["metrics"])
            _write_json(run_output_dir / "run_summary.json", final_payload)
            return final_payload

        if completed > 0:
            last_task_name = sequence.tasks[completed - 1].name
            last_safe = _safe_name(last_task_name)
            checkpoint_dir = checkpoint_root / f"stage_{completed:02d}_{last_safe}" / "merged_model"
            if not checkpoint_dir.exists():
                raise FileNotFoundError(
                    f"Resume failed: missing checkpoint at {checkpoint_dir}."
                )
            tokenizer = build_tokenizer(model_name=str(checkpoint_dir), hf_token=HF_TOKEN)
            model = load_base_model(model_name=str(checkpoint_dir), hf_token=HF_TOKEN)
        else:
            tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
            model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
        model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)

    for idx in range(start_stage, len(sequence.tasks) + 1):
        task = sequence.tasks[idx - 1]
        task_name = task.name
        safe_task_name = _safe_name(task_name)

        stage_train_dir = train_output_dir / sequence_name / f"stage_{idx:02d}_{safe_task_name}"
        stage_eval_dir = run_output_dir / "stages" / f"stage_{idx:02d}_{safe_task_name}"

        print(f"\n=== Stage {idx}/{len(sequence.tasks)} | Training task: {task_name} ===")
        retain_task = sequence.tasks[idx - 2] if idx > 1 else None

        model, train_report = train_on_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            output_dir=str(stage_train_dir),
            eval_size=eval_size,
            retain_task=retain_task,
            slice_enabled=slice_enabled,
            slice_cache_dir=slice_cache_dir,
            slice_max_steps=slice_max_steps,
            slice_retain_scale=slice_retain_scale,
            slice_rank=slice_rank,
        )

        seen_tasks.append(task)
        evaluation = evaluate_all(
            model=model,
            tokenizer=tokenizer,
            seen_tasks=seen_tasks,
            output_dir=str(stage_eval_dir),
            general_eval_task_keys=general_eval_keys,
            eval_size=eval_size,
            task_eval_samples=task_eval_samples,
            task_eval_max_new_tokens=task_eval_max_new_tokens,
        )

        stage_record = {
            "stage": idx,
            "trained_task": task_name,
            "train_report": train_report,
            "seen_tasks": evaluation["seen_tasks"],
            "general": evaluation["general"],
        }
        stage_records.append(stage_record)

        _write_json(stage_eval_dir / "stage_record.json", stage_record)
        checkpoint_dir = checkpoint_root / f"stage_{idx:02d}_{safe_task_name}" / "merged_model"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))

        _write_json(
            partial_path,
            {
                "sequence": sequence_name,
                "task_order": task_order,
                "completed_stages": idx,
                "stage_records": stage_records,
            },
        )

    summary = compute_cl_metrics(stage_records=stage_records, task_order=task_order)
    final_payload = {
        "sequence": sequence_name,
        "description": sequence.description,
        "task_order": task_order,
        "general_eval_keys": general_eval_keys,
        "stage_records": stage_records,
        "summary": summary,
    }

    _write_json(run_output_dir / "results_matrix.json", summary["results_matrix"])
    _write_json(run_output_dir / "metrics.json", summary["metrics"])
    _write_json(run_output_dir / "run_summary.json", final_payload)

    if save_final_model:
        model_dir = run_output_dir / "final_merged_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual-learning LoRA train-merge-eval orchestrator.")
    parser.add_argument("--sequence", required=True, help="Sequence name (e.g., NI-Seq-G1, NI-Seq-C1, TRACE).")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--train-output-root", default="outputs")
    parser.add_argument(
        "--general-eval-set",
        choices=["core", "all"],
        default="core",
        help="Use the core 4 general tasks or all configured general tasks.",
    )
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--task-eval-samples", type=int, default=64)
    parser.add_argument("--task-eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument("--slice-init", action="store_true", help="Enable slice LoRA init.")
    parser.add_argument("--slice-cache-dir", default="slice_cache")
    parser.add_argument("--slice-max-steps", type=int, default=100)
    parser.add_argument("--slice-retain-scale", type=float, default=1.0)
    parser.add_argument("--slice-rank", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Configure logging so slice and cache logs are visible
    try:
        lvl = getattr(logging, args.log_level.upper(), logging.INFO)
    except Exception:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Reduce verbosity of very chatty third-party libraries by default
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)

    general_eval_keys = (
        CORE_EVAL_TASKS if args.general_eval_set == "core" else list(GENERAL_EVAL_TASKS.keys())
    )

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_root) / args.sequence / run_name
    train_output_dir = Path(args.train_output_root)

    print(f"Sequence: {args.sequence}")
    print(f"General eval tasks: {general_eval_keys}")
    print(f"Results dir: {run_output_dir}")

    payload = run_sequence(
        sequence_name=args.sequence,
        model_name=args.model_name,
        run_output_dir=run_output_dir,
        train_output_dir=train_output_dir,
        general_eval_keys=general_eval_keys,
        eval_size=args.eval_size,
        task_eval_samples=args.task_eval_samples,
        task_eval_max_new_tokens=args.task_eval_max_new_tokens,
        save_final_model=args.save_final_model,
        resume=args.resume,
        slice_enabled=args.slice_init,
        slice_cache_dir=args.slice_cache_dir,
        slice_max_steps=args.slice_max_steps,
        slice_retain_scale=args.slice_retain_scale,
        slice_rank=args.slice_rank,
    )

    print("\n=== Final Metrics ===")
    print(json.dumps(payload["summary"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
