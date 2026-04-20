from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from importlib import metadata as importlib_metadata  # py3.8+
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:
    from .eval import evaluate_all
    from .metrics import compute_cl_metrics
    from .repro import set_global_seed
    from .task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from .train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model, train_on_task
except ImportError:
    from eval import evaluate_all
    from metrics import compute_cl_metrics
    from repro import set_global_seed
    from task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model, train_on_task


def _collect_fn_defaults(fn) -> Dict[str, Any]:
    import inspect

    sig = inspect.signature(fn)
    out: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.default is inspect._empty:
            continue
        out[name] = param.default
    return out


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


def _pkg_version(dist_name: str) -> str | None:
    try:
        return str(importlib_metadata.version(dist_name))
    except Exception:
        return None


def _collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "packages": {
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "accelerate": _pkg_version("accelerate"),
            "peft": _pkg_version("peft"),
            "datasets": _pkg_version("datasets"),
        },
    }

    try:
        import torch

        info["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        pass

    return info


def _collect_model_info(model) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "class": model.__class__.__name__,
        "name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None)
        or getattr(model, "name_or_path", None),
    }
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "to_dict"):
        try:
            out["config"] = cfg.to_dict()
        except Exception:
            out["config"] = str(cfg)
    return out


def _collect_tokenizer_info(tokenizer) -> Dict[str, Any]:
    return {
        "class": tokenizer.__class__.__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }


def run_sequence(
    sequence_name: str,
    model_name: str,
    run_output_dir: Path,
    train_output_dir: Path,
    general_eval_keys: List[str],
    seed: int,
    eval_size: int,
    task_eval_samples: int,
    task_eval_max_new_tokens: int,
    quick_eval: bool,
    save_final_model: bool,
    resume: bool,
    rank: int,
    slice_enabled: bool,
    slice_cache_dir: str,
    slice_max_steps: int,
    slice_retain_scale: float,
    slice_grad_project: bool,
    slice_grad_projection_mode: str,
    slice_grad_project_always: bool,
    slice_add_retain_grad: bool,
    slice_retain_batch_size: int | None = None,
    slice_retain_grad_accum: int | None = None,
    slice_retain_batch_size_set: str = "all_tasks",
    slice_single_retain_task_mode: bool = False,
    slice_init_method: str = "slice",
    keep_all_checkpoints: bool = False,
    general_eval_strategy: str = "every_stage",
    seen_eval_strategy: str = "full_matrix",
    train_only: bool = False,
    orchestrator_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    set_global_seed(seed)
    run_output_dir = run_output_dir.resolve()
    train_output_dir = train_output_dir.resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    sequence = get_sequence(sequence_name)
    task_order = [task.name for task in sequence.tasks]

    resolved_cfg: Dict[str, Any] = {
        "sequence": sequence_name,
        "description": sequence.description,
        "task_order": task_order,
        "general_eval_keys": general_eval_keys,
        "seed": seed,
        "quick_eval": bool(quick_eval),
        "eval_size": int(eval_size),
        "task_eval_samples": int(task_eval_samples),
        "task_eval_max_new_tokens": int(task_eval_max_new_tokens),
        "rank": int(rank),
        "slice_enabled": bool(slice_enabled),
        "slice_cache_dir": slice_cache_dir,
        "slice_max_steps": int(slice_max_steps),
        "slice_retain_scale": float(slice_retain_scale),
        "slice_grad_project": bool(slice_grad_project),
        "slice_grad_projection_mode": str(slice_grad_projection_mode),
        "slice_grad_project_always": bool(slice_grad_project_always),
        "slice_add_retain_grad": bool(slice_add_retain_grad),
        "slice_retain_batch_size": slice_retain_batch_size,
        "slice_retain_grad_accum": slice_retain_grad_accum,
        "slice_retain_batch_size_set": slice_retain_batch_size_set,
        "slice_single_retain_task_mode": slice_single_retain_task_mode,
        "slice_init_method": slice_init_method,
        "keep_all_checkpoints": keep_all_checkpoints,
        "general_eval_strategy": general_eval_strategy,
        "seen_eval_strategy": seen_eval_strategy,
    }

    run_cfg_payload: Dict[str, Any] = {
        "orchestrator": orchestrator_config or {},
        "env": _collect_env_info(),
        "resolved": resolved_cfg,
        "model": None,
        "tokenizer": None,
        "notes": {
            "hf_token_present": bool(HF_TOKEN),
            "hf_token_redacted": True,
        },
    }
    _write_json(run_output_dir / "run_config.json", run_cfg_payload)

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

    run_cfg_payload["model"] = _collect_model_info(model)
    run_cfg_payload["tokenizer"] = _collect_tokenizer_info(tokenizer)
    _write_json(run_output_dir / "run_config.json", run_cfg_payload)

    for idx in range(start_stage, len(sequence.tasks) + 1):
        task = sequence.tasks[idx - 1]
        task_name = task.name
        safe_task_name = _safe_name(task_name)

        stage_train_dir = train_output_dir / sequence_name / f"stage_{idx:02d}_{safe_task_name}"
        stage_eval_dir = run_output_dir / "stages" / f"stage_{idx:02d}_{safe_task_name}"

        print(f"\n=== Stage {idx}/{len(sequence.tasks)} | Training task: {task_name} ===")
        retain_tasks = list(sequence.tasks[:idx - 1]) if idx > 1 else None

        if idx == 1:
            slice_cache_context = f"base_model:{model_name}"
        else:
            prev_task_name = sequence.tasks[idx - 2].name
            prev_safe = _safe_name(prev_task_name)
            prev_checkpoint_dir = checkpoint_root / f"stage_{idx - 1:02d}_{prev_safe}" / "merged_model"
            slice_cache_context = f"checkpoint:{prev_checkpoint_dir}"

        model, train_report = train_on_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            output_dir=str(stage_train_dir),
            eval_size=eval_size,
            seed=seed,
            retain_tasks=retain_tasks,
            rank=rank,
            slice_enabled=slice_enabled,
            slice_cache_dir=slice_cache_dir,
            slice_cache_context=slice_cache_context,
            slice_max_steps=slice_max_steps,
            slice_retain_scale=slice_retain_scale,
            slice_grad_project=slice_grad_project,
            slice_grad_projection_mode=slice_grad_projection_mode,
            slice_grad_project_always=slice_grad_project_always,
            slice_add_retain_grad=slice_add_retain_grad,
            slice_retain_batch_size=slice_retain_batch_size,
            slice_retain_grad_accum=slice_retain_grad_accum,
            slice_retain_batch_size_set=slice_retain_batch_size_set,
            slice_single_retain_task_mode=slice_single_retain_task_mode,
            slice_init_method=slice_init_method,
        )

        seen_tasks.append(task)

        is_final_stage = (idx == len(sequence.tasks))

        # Decide which seen tasks to evaluate at this stage.
        if seen_eval_strategy == "diagonal_final" and not is_final_stage:
            eval_seen = [task]
            print(f"  Seen-task eval: diagonal only ({task_name})")
        else:
            eval_seen = list(seen_tasks)

        # Decide whether to run general (GP/IP) evaluation.
        skip_general = (general_eval_strategy == "final_only" and not is_final_stage)
        if skip_general:
            print(f"  Skipping general eval (strategy=final_only)")

        # Save the merged checkpoint BEFORE evaluation so it is always available
        # even if evaluation is skipped or run separately later.
        checkpoint_dir = checkpoint_root / f"stage_{idx:02d}_{safe_task_name}" / "merged_model"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
        print(f"  Checkpoint saved: {checkpoint_dir}")

        # Write a manifest that a standalone eval pass can consume without
        # needing to re-run training.
        stage_eval_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            stage_eval_dir / "eval_manifest.json",
            {
                "stage": idx,
                "trained_task": task_name,
                "sequence": sequence_name,
                "seen_tasks": [getattr(t, "name", str(t)) for t in seen_tasks],
                "eval_seen_tasks": [getattr(t, "name", str(t)) for t in eval_seen],
                "model_path": str(checkpoint_dir.resolve()),
                "general_eval_keys": general_eval_keys,
                "skip_general_eval": bool(skip_general),
                "quick_eval": bool(quick_eval),
                "eval_size": int(eval_size),
                "task_eval_samples": int(task_eval_samples),
                "task_eval_max_new_tokens": int(task_eval_max_new_tokens),
                "seed": int(seed),
                "train_output_dir": str(stage_train_dir),
            },
        )

        if train_only:
            print("  Skipping evaluation (--train-only mode).")
            stage_record = {
                "stage": idx,
                "trained_task": task_name,
                "train_report": train_report,
                "seen_tasks": {},
                "general": {"gp": {}, "ip": {}, "gp_mean": None, "ip_mean": None, "mode": "skipped"},
            }
        else:
            evaluation = evaluate_all(
                model=model,
                tokenizer=tokenizer,
                seen_tasks=eval_seen,
                output_dir=str(stage_eval_dir),
                general_eval_task_keys=general_eval_keys,
                eval_size=eval_size,
                task_eval_samples=task_eval_samples,
                task_eval_max_new_tokens=task_eval_max_new_tokens,
                quick_eval=quick_eval,
                skip_general_eval=skip_general,
                seed=seed,
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

        _write_json(
            partial_path,
            {
                "sequence": sequence_name,
                "task_order": task_order,
                "completed_stages": idx,
                "stage_records": stage_records,
            },
        )

        # Remove the previous stage checkpoint to save disk space.
        # Only the most recent checkpoint is needed for --resume.
        if not keep_all_checkpoints and idx >= 2:
            prev_task = sequence.tasks[idx - 2]
            prev_safe = _safe_name(prev_task.name)
            old_checkpoint = checkpoint_root / f"stage_{idx - 1:02d}_{prev_safe}"
            if old_checkpoint.exists():
                shutil.rmtree(old_checkpoint)
                print(f"  Cleaned up old checkpoint: {old_checkpoint}")

    if train_only:
        final_payload = {
            "sequence": sequence_name,
            "description": sequence.description,
            "task_order": task_order,
            "stage_records": stage_records,
            "summary": None,
        }
        _write_json(run_output_dir / "run_summary.json", final_payload)
        print("\nTraining complete. Run eval separately with:")
        print(f"  python -m cl_lora.eval_standalone run --run-dir {run_output_dir}")
    else:
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
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip all evaluation. Saves checkpoints and eval_manifest.json at each stage "
             "so eval can be run later with: python -m cl_lora.eval_standalone run --run-dir <dir>",
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Perplexity-only seen-task evaluation (skips GP/IP and generation-based seen-task metrics).",
    )
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for reproducibility.")
    parser.add_argument("--task-eval-samples", type=int, default=64)
    parser.add_argument("--task-eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument("--slice-init", action="store_true", help="Enable slice LoRA init.")
    parser.add_argument("--slice-cache-dir", default="slice_cache")
    parser.add_argument("--slice-max-steps", type=int, default=100)
    parser.add_argument("--slice-retain-scale", type=float, default=1.0)
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help="LoRA rank (also used for slice init when --slice-init is enabled).",
    )
    parser.add_argument("--slice-grad-project", action="store_true", help="Project forget gradients against retain gradients for slice init.")
    parser.add_argument(
        "--slice-grad-projection-mode",
        choices=["per_module", "global"],
        default="per_module",
        help="Projection mode when --slice-grad-project is enabled.",
    )
    parser.add_argument(
        "--slice-grad-project-always",
        action="store_true",
        help="Use OGD-style projection: always remove retain-gradient component (no conflict gating).",
    )
    parser.add_argument(
        "--slice-add-retain-grad",
        action="store_true",
        help="Add retain gradient after projection when --slice-grad-project is enabled.",
    )
    parser.add_argument("--slice-retain-batch-size", type=int, default=None,
        help="Batch size for retain gradient computation. Defaults to training batch size.")
    parser.add_argument("--slice-retain-grad-accum", type=int, default=None,
        help="Max accumulation steps for retain gradient. Defaults to --slice-max-steps.")
    parser.add_argument("--slice-retain-batch-size-set", choices=["all_tasks", "each_task"],
        default="all_tasks",
        help="How retain batch size is applied: 'all_tasks' = total across all tasks, 'each_task' = per task.")
    parser.add_argument("--slice-single-retain-task-mode", action="store_true",
        help="Only use the most recent previous task for retain, with same batch size as forget.")
    parser.add_argument("--slice-init-method", choices=["slice", "lora_ga", "loram"],
        default="slice",
        help="Initialization method: 'slice' (default), 'lora_ga' (SVD on forget gradients only), "
             "or 'loram' (DST-based, no gradients).")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
        help="Keep all intermediate stage checkpoints. By default only the latest is kept.")
    parser.add_argument("--general-eval-strategy", choices=["every_stage", "final_only"],
        default="every_stage",
        help="When to run general (GP/IP) eval. 'final_only' skips it at intermediate stages.")
    parser.add_argument("--seen-eval-strategy", choices=["full_matrix", "diagonal_final"],
        default="full_matrix",
        help="Seen-task eval: 'diagonal_final' evaluates only the trained task at intermediate "
             "stages and all tasks at the final stage.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    set_global_seed(args.seed)

    # Configure logging so slice and cache logs are visible
    try:
        lvl = getattr(logging, args.log_level.upper(), logging.INFO)
    except Exception:
        lvl = logging.INFO
    # basicConfig is a no-op if handlers are already set up; force=True makes this reliable.
    try:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )
    except TypeError:
        logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Ensure our package loggers follow the requested level.
    logging.getLogger("cl_lora").setLevel(lvl)
    logging.getLogger("cl_lora.slice").setLevel(lvl)
    # Reduce verbosity of very chatty third-party libraries by default
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)

    general_eval_keys = (
        CORE_EVAL_TASKS if args.general_eval_set == "core" else list(GENERAL_EVAL_TASKS.keys())
    )

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_root) / args.sequence / run_name
    train_output_dir = Path(args.train_output_root)

    orchestrator_config = {
        "entrypoint": "cl_lora.orchestrator",
        "created_at": datetime.now().isoformat(),
        "cli_args": vars(args),
        "defaults": {
            "train_on_task": _collect_fn_defaults(train_on_task),
            "load_base_model": _collect_fn_defaults(load_base_model),
            "build_tokenizer": _collect_fn_defaults(build_tokenizer),
            "evaluate_all": _collect_fn_defaults(evaluate_all),
        },
    }

    print(f"Sequence: {args.sequence}")
    print(f"General eval tasks: {general_eval_keys}")
    if args.train_only:
        print("Eval mode: DISABLED (--train-only)")
    else:
        print(f"Quick eval mode: {'ON (perplexity-only)' if args.quick_eval else 'OFF'}")
    print(f"Results dir: {run_output_dir}")

    payload = run_sequence(
        sequence_name=args.sequence,
        model_name=args.model_name,
        run_output_dir=run_output_dir,
        train_output_dir=train_output_dir,
        general_eval_keys=general_eval_keys,
        seed=args.seed,
        eval_size=args.eval_size,
        task_eval_samples=args.task_eval_samples,
        task_eval_max_new_tokens=args.task_eval_max_new_tokens,
        quick_eval=args.quick_eval,
        save_final_model=args.save_final_model,
        resume=args.resume,
        rank=args.rank,
        slice_enabled=args.slice_init,
        slice_cache_dir=args.slice_cache_dir,
        slice_max_steps=args.slice_max_steps,
        slice_retain_scale=args.slice_retain_scale,
        slice_grad_project=args.slice_grad_project,
        slice_grad_projection_mode=args.slice_grad_projection_mode,
        slice_grad_project_always=args.slice_grad_project_always,
        slice_add_retain_grad=args.slice_add_retain_grad,
        slice_retain_batch_size=args.slice_retain_batch_size,
        slice_retain_grad_accum=args.slice_retain_grad_accum,
        slice_retain_batch_size_set=args.slice_retain_batch_size_set,
        slice_single_retain_task_mode=args.slice_single_retain_task_mode,
        slice_init_method=args.slice_init_method,
        keep_all_checkpoints=args.keep_all_checkpoints,
        general_eval_strategy=args.general_eval_strategy,
        seen_eval_strategy=args.seen_eval_strategy,
        train_only=args.train_only,
        orchestrator_config=orchestrator_config,
    )

    if not args.train_only:
        print("\n=== Final Metrics ===")
        print(json.dumps(payload["summary"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
