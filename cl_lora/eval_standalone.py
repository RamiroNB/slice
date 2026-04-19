"""Standalone evaluation script.

Loads a saved merged-model checkpoint produced by the orchestrator (or by
train.py --save-merged-model) and runs evaluate_all() on it, writing a
stage_record.json compatible with recompute_metrics.py and the rest of the
analysis pipeline.

Typical usage after a training run produced checkpoints:

    python -m cl_lora.eval_standalone \
        --stage-dir results/NI-Seq-G1/run_xyz/stages/stage_01_task363_sst2_polarity_classification

The script reads eval_manifest.json from the stage dir (written by the
orchestrator before each eval step) and uses the parameters stored there.
All parameters can be overridden on the command line.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .eval import evaluate_all
    from .metrics import compute_cl_metrics
    from .repro import set_global_seed
    from .task_sequences import get_sequence
    from .train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model
except ImportError:
    from eval import evaluate_all
    from metrics import compute_cl_metrics
    from repro import set_global_seed
    from task_sequences import get_sequence
    from train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_tasks(sequence_name: str, task_names: List[str]):
    """Return task objects for the given names from the named sequence."""
    seq = get_sequence(sequence_name)
    name_to_task = {task.name: task for task in seq.tasks}
    tasks = []
    for name in task_names:
        if name not in name_to_task:
            raise ValueError(
                f"Task '{name}' not found in sequence '{sequence_name}'. "
                f"Available: {list(name_to_task.keys())}"
            )
        tasks.append(name_to_task[name])
    return tasks


def run_eval_from_manifest(
    stage_dir: Path,
    *,
    model_path: Optional[str] = None,
    sequence: Optional[str] = None,
    seen_tasks: Optional[List[str]] = None,
    eval_seen_tasks: Optional[List[str]] = None,
    general_eval_keys: Optional[List[str]] = None,
    skip_general_eval: Optional[bool] = None,
    quick_eval: Optional[bool] = None,
    eval_size: Optional[int] = None,
    task_eval_samples: Optional[int] = None,
    task_eval_max_new_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    general_eval_batch_size: int = 8,
) -> Dict[str, Any]:
    """Run evaluation for one stage, writing stage_record.json to stage_dir.

    Parameters that are None are loaded from eval_manifest.json in stage_dir.
    Explicit values override what is stored in the manifest.
    """
    manifest_path = stage_dir / "eval_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        print(f"Loaded eval manifest: {manifest_path}")
    else:
        print(f"No eval_manifest.json found at {stage_dir}; using CLI arguments only.")

    def _get(key: str, cli_val, default=None):
        if cli_val is not None:
            return cli_val
        return manifest.get(key, default)

    resolved_model_path: str = _get("model_path", model_path)
    if not resolved_model_path:
        raise ValueError("--model-path is required when no eval_manifest.json is present.")

    resolved_sequence: str = _get("sequence", sequence)
    if not resolved_sequence:
        raise ValueError("--sequence is required when no eval_manifest.json is present.")

    resolved_seen_task_names: List[str] = _get("seen_tasks", seen_tasks, [])
    resolved_eval_seen_names: List[str] = _get("eval_seen_tasks", eval_seen_tasks) or resolved_seen_task_names
    resolved_general_keys: List[str] = _get("general_eval_keys", general_eval_keys, [])
    resolved_skip_general: bool = bool(_get("skip_general_eval", skip_general_eval, False))
    resolved_quick: bool = bool(_get("quick_eval", quick_eval, False))
    resolved_eval_size: int = int(_get("eval_size", eval_size, 200))
    resolved_samples: int = int(_get("task_eval_samples", task_eval_samples, 64))
    resolved_max_new: int = int(_get("task_eval_max_new_tokens", task_eval_max_new_tokens, 64))
    resolved_seed: int = int(_get("seed", seed, 42))
    stage: int = int(manifest.get("stage", 0))
    trained_task: str = manifest.get("trained_task", "")

    # Resolve the model path to absolute. If the manifest path doesn't exist
    # (e.g. this is a different machine than where training ran), fall back to
    # deriving it from the stage_dir: the checkpoint directory mirrors the
    # stage directory structure under ../checkpoints/<stage_name>/merged_model.
    abs_model_path = str(Path(resolved_model_path).resolve())
    if not Path(abs_model_path).is_dir():
        derived = stage_dir.parent.parent / "checkpoints" / stage_dir.name / "merged_model"
        if derived.is_dir():
            abs_model_path = str(derived.resolve())
            print(f"Manifest model path not found; using derived path: {abs_model_path}")
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at '{abs_model_path}' "
                f"or derived path '{derived}'. "
                f"Ensure the checkpoints/ directory was transferred to this machine."
            )
    print(f"Loading model from: {abs_model_path}")
    tokenizer = build_tokenizer(model_name=abs_model_path, hf_token=HF_TOKEN)
    model = load_base_model(model_name=abs_model_path, hf_token=HF_TOKEN)

    print(f"Resolving {len(resolved_eval_seen_names)} eval tasks from sequence '{resolved_sequence}'")
    eval_seen = _resolve_tasks(resolved_sequence, resolved_eval_seen_names)

    set_global_seed(resolved_seed)

    print(f"Running evaluate_all() for stage {stage} ({trained_task})")
    evaluation = evaluate_all(
        model=model,
        tokenizer=tokenizer,
        seen_tasks=eval_seen,
        output_dir=str(stage_dir),
        general_eval_task_keys=resolved_general_keys or None,
        general_eval_batch_size=general_eval_batch_size,
        eval_size=resolved_eval_size,
        task_eval_samples=resolved_samples,
        task_eval_max_new_tokens=resolved_max_new,
        quick_eval=resolved_quick,
        skip_general_eval=resolved_skip_general,
        seed=resolved_seed,
    )

    train_report: Dict[str, Any] = {}
    train_report_path = Path(manifest.get("train_output_dir", "")) / "training_report.json"
    if train_report_path.exists():
        train_report = _read_json(train_report_path)

    stage_record = {
        "stage": stage,
        "trained_task": trained_task,
        "train_report": train_report,
        "seen_tasks": evaluation["seen_tasks"],
        "general": evaluation["general"],
    }

    out_path = stage_dir / "stage_record.json"
    _write_json(out_path, stage_record)
    print(f"Saved stage_record.json: {out_path}")

    return stage_record


def recompute_run_summary(run_dir: Path) -> None:
    """Rebuild results_matrix.json and metrics.json from all stage_record.json files."""
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        print(f"No stages/ directory found in {run_dir}")
        return

    stage_dirs = sorted(stages_dir.glob("stage_*"))
    stage_records = []
    task_order = []
    for sd in stage_dirs:
        sr_path = sd / "stage_record.json"
        if sr_path.exists():
            sr = _read_json(sr_path)
            stage_records.append(sr)
            task_order.append(sr.get("trained_task", ""))

    if not stage_records:
        print("No stage_record.json files found; skipping summary.")
        return

    summary = compute_cl_metrics(stage_records=stage_records, task_order=task_order)
    _write_json(run_dir / "results_matrix.json", summary["results_matrix"])
    _write_json(run_dir / "metrics.json", summary["metrics"])
    print(f"Updated results_matrix.json and metrics.json in {run_dir}")
    print(json.dumps(summary["metrics"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone eval: load a saved checkpoint and run evaluate_all()."
    )

    mode = parser.add_subparsers(dest="mode")

    # ---- stage sub-command: evaluate a single stage ----
    stage_p = mode.add_parser("stage", help="Evaluate a single stage from its stage directory.")
    stage_p.add_argument(
        "--stage-dir",
        required=True,
        help="Path to the stage directory (contains eval_manifest.json).",
    )
    stage_p.add_argument("--model-path", default=None, help="Override model path from manifest.")
    stage_p.add_argument("--sequence", default=None, help="Override sequence name from manifest.")
    stage_p.add_argument(
        "--seen-tasks",
        nargs="+",
        default=None,
        help="Override full seen-task list (task names) from manifest.",
    )
    stage_p.add_argument(
        "--eval-seen-tasks",
        nargs="+",
        default=None,
        help="Override which seen tasks to evaluate (subset of --seen-tasks).",
    )
    stage_p.add_argument(
        "--general-eval-keys",
        nargs="+",
        default=None,
        help="Override general eval task keys from manifest.",
    )
    stage_p.add_argument("--skip-general-eval", action="store_true", default=None)
    stage_p.add_argument("--quick-eval", action="store_true", default=None)
    stage_p.add_argument("--eval-size", type=int, default=None)
    stage_p.add_argument("--task-eval-samples", type=int, default=None)
    stage_p.add_argument("--task-eval-max-new-tokens", type=int, default=None)
    stage_p.add_argument("--seed", type=int, default=None)
    stage_p.add_argument("--general-eval-batch-size", type=int, default=8)
    stage_p.add_argument("--log-level", default="INFO")

    # ---- run sub-command: evaluate all stages of a full run ----
    run_p = mode.add_parser(
        "run",
        help="Evaluate all stages in a run directory (reads each stage's eval_manifest.json).",
    )
    run_p.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run output directory (contains stages/ sub-directory).",
    )
    run_p.add_argument("--skip-general-eval", action="store_true", default=None)
    run_p.add_argument("--quick-eval", action="store_true", default=None)
    run_p.add_argument("--eval-size", type=int, default=None)
    run_p.add_argument("--task-eval-samples", type=int, default=None)
    run_p.add_argument("--task-eval-max-new-tokens", type=int, default=None)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--general-eval-batch-size", type=int, default=8)
    run_p.add_argument("--log-level", default="INFO")

    # ---- summary sub-command: recompute run-level metrics without re-evaluating ----
    summary_p = mode.add_parser(
        "summary",
        help="Recompute results_matrix.json and metrics.json from existing stage_record.json files.",
    )
    summary_p.add_argument("--run-dir", required=True)

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    lvl = getattr(logging, args.log_level.upper(), logging.INFO) if hasattr(args, "log_level") else logging.INFO
    try:
        logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    except TypeError:
        logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)

    if args.mode == "stage":
        skip_general = True if args.skip_general_eval else None
        quick = True if args.quick_eval else None
        run_eval_from_manifest(
            stage_dir=Path(args.stage_dir),
            model_path=args.model_path,
            sequence=args.sequence,
            seen_tasks=args.seen_tasks,
            eval_seen_tasks=args.eval_seen_tasks,
            general_eval_keys=args.general_eval_keys,
            skip_general_eval=skip_general,
            quick_eval=quick,
            eval_size=args.eval_size,
            task_eval_samples=args.task_eval_samples,
            task_eval_max_new_tokens=args.task_eval_max_new_tokens,
            seed=args.seed,
            general_eval_batch_size=args.general_eval_batch_size,
        )

    elif args.mode == "run":
        run_dir = Path(args.run_dir)
        stages_dir = run_dir / "stages"
        if not stages_dir.exists():
            raise FileNotFoundError(f"stages/ directory not found in {run_dir}")
        stage_dirs = sorted(stages_dir.glob("stage_*"))
        if not stage_dirs:
            raise FileNotFoundError(f"No stage_* directories found in {stages_dir}")

        skip_general = True if args.skip_general_eval else None
        quick = True if args.quick_eval else None

        for sd in stage_dirs:
            if not (sd / "eval_manifest.json").exists():
                print(f"Skipping {sd} (no eval_manifest.json)")
                continue
            print(f"\n=== Evaluating {sd.name} ===")
            run_eval_from_manifest(
                stage_dir=sd,
                skip_general_eval=skip_general,
                quick_eval=quick,
                eval_size=args.eval_size,
                task_eval_samples=args.task_eval_samples,
                task_eval_max_new_tokens=args.task_eval_max_new_tokens,
                seed=args.seed,
                general_eval_batch_size=args.general_eval_batch_size,
            )

        recompute_run_summary(run_dir)

    elif args.mode == "summary":
        recompute_run_summary(Path(args.run_dir))


if __name__ == "__main__":
    main()
