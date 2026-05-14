from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from importlib import metadata as importlib_metadata  # py3.8+
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:
    from .cl_methods import REGISTRY as CL_METHOD_REGISTRY, build_cl_method
    from .cl_methods.sapt import SAPTMethod
    from .eval import evaluate_all
    from .metrics import compute_cl_metrics
    from .repro import set_global_seed
    from .task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from .train import (
        HF_TOKEN, MODEL_NAME,
        build_tokenizer, load_base_model,
        load_model_with_adapters, load_sapt_model,
        train_on_task,
    )
except ImportError:
    from cl_methods import REGISTRY as CL_METHOD_REGISTRY, build_cl_method  # type: ignore[no-redef]
    from cl_methods.sapt import SAPTMethod  # type: ignore[no-redef]
    from eval import evaluate_all
    from metrics import compute_cl_metrics
    from repro import set_global_seed
    from task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS, get_sequence
    from train import (  # type: ignore[no-redef]
        HF_TOKEN, MODEL_NAME,
        build_tokenizer, load_base_model,
        load_model_with_adapters, load_sapt_model,
        train_on_task,
    )


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


def _safe_model_dir_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", str(model_name)).strip("_")


def _shared_base_model_is_complete(shared_dir: Path) -> bool:
    if not shared_dir.is_dir():
        return False
    # HF save_pretrained always writes config.json + at least one weight shard.
    if not (shared_dir / "config.json").is_file():
        return False
    has_weights = any(
        shared_dir.glob(pat)
        for pat in ("*.safetensors", "*.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    )
    return bool(has_weights)


def _save_shared_base_model(model, tokenizer, shared_dir: Path) -> None:
    """Save the base model + tokenizer once into the shared cache.

    Concurrency-safe: writes into a unique temp directory next to shared_dir
    and atomically renames it into place. If another process won the race and
    populated shared_dir first, the temp copy is discarded.
    """
    shared_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(
        prefix=f".{shared_dir.name}.tmp-", dir=str(shared_dir.parent)
    ))
    try:
        model.save_pretrained(str(tmp_dir))
        tokenizer.save_pretrained(str(tmp_dir))
        try:
            os.rename(str(tmp_dir), str(shared_dir))
        except OSError:
            if _shared_base_model_is_complete(shared_dir):
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            else:
                raise
    except Exception:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise


def _link_shared_base_model(run_link: Path, shared_dir: Path) -> None:
    """Create a symlink run_link → shared_dir.

    If run_link is a broken symlink (target missing — e.g. the shared cache
    was cleaned), the dangling link is removed and recreated. Existing real
    directories/files or symlinks with a valid target are left as-is, even
    if they point elsewhere.
    """
    if run_link.is_symlink():
        if run_link.exists():
            return
        run_link.unlink()
    elif run_link.exists():
        return
    run_link.parent.mkdir(parents=True, exist_ok=True)
    run_link.symlink_to(shared_dir.resolve(), target_is_directory=True)


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
    lora_alpha: int,
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
    slice_projection_method: str = "pcgrad",
    slice_cosine_threshold: float | None = None,
    slice_per_layer_threshold: bool = False,
    slice_per_layer_threshold_delta: float = 0.0,
    slice_cagrad_c: float = 0.5,
    slice_gradvac_phi: float = 0.0,
    slice_gradvac_beta: float = 0.5,
    slice_magnitude_preserve: bool = False,
    slice_nullspace_rank: int = 8,
    slice_nullspace_sv_threshold: float = 0.0,
    slice_svd_selection: str = "lora_ga",
    keep_all_checkpoints: bool = False,
    save_intermediate_checkpoints: bool = False,
    general_eval_strategy: str = "every_stage",
    seen_eval_strategy: str = "full_matrix",
    train_only: bool = False,
    cl_method_name: str = "vanilla",
    cl_method_kwargs: Dict[str, Any] | None = None,
    orchestrator_config: Dict[str, Any] | None = None,
    base_model_cache_dir: str | None = None,
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
        "lora_alpha": int(lora_alpha),
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
        "save_intermediate_checkpoints": save_intermediate_checkpoints,
        "general_eval_strategy": general_eval_strategy,
        "seen_eval_strategy": seen_eval_strategy,
        "cl_method": str(cl_method_name),
        "cl_method_kwargs": dict(cl_method_kwargs or {}),
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
            base_model_ckpt = checkpoint_root / "base_model"
            if not base_model_ckpt.exists():
                raise FileNotFoundError(
                    f"Resume failed: base model checkpoint not found at {base_model_ckpt}. "
                    "Runs started before the adapter-only checkpoint format was introduced cannot be resumed."
                )
            adapter_paths_resume = [
                str(checkpoint_root / f"stage_{i + 1:02d}_{_safe_name(sequence.tasks[i].name)}" / "adapter")
                for i in range(completed)
            ]
            for ap in adapter_paths_resume:
                if not Path(ap).exists():
                    raise FileNotFoundError(f"Resume failed: missing adapter checkpoint at {ap}.")
            tokenizer = build_tokenizer(model_name=str(base_model_ckpt), hf_token=HF_TOKEN)
            if str(cl_method_name).lower() == "sapt":
                # SAPT keeps adapters parallel — load every prior adapter as a
                # named adapter without merging. The router (if previously
                # trained) is loaded later via cl_method.load(state_dir).
                from peft import PeftModel

                base = load_base_model(model_name=str(base_model_ckpt), hf_token=HF_TOKEN)
                peft_model = None
                for i, ap in enumerate(adapter_paths_resume):
                    name = SAPTMethod.adapter_name_for_stage(i + 1)
                    named_subdir = Path(ap) / name
                    if named_subdir.is_dir() and (named_subdir / "adapter_config.json").exists():
                        ap = str(named_subdir)
                    if peft_model is None:
                        peft_model = PeftModel.from_pretrained(base, ap, adapter_name=name)
                    else:
                        peft_model.load_adapter(ap, adapter_name=name)
                model = peft_model
            else:
                model = load_model_with_adapters(str(base_model_ckpt), adapter_paths_resume)
        else:
            tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
            model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
        model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)

    run_cfg_payload["model"] = _collect_model_info(model)
    run_cfg_payload["tokenizer"] = _collect_tokenizer_info(tokenizer)
    _write_json(run_output_dir / "run_config.json", run_cfg_payload)

    # Save the unmodified base model once per (model_name, base_model_cache_dir)
    # and symlink it into every run. All stage checkpoints store only LoRA
    # adapters; the full model is reconstructed by merging adapters onto this
    # base at eval time.
    base_model_ckpt = checkpoint_root / "base_model"
    if base_model_cache_dir:
        shared_dir = (Path(base_model_cache_dir) / _safe_model_dir_name(model_name)).resolve()
        if start_stage == 1:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            if not _shared_base_model_is_complete(shared_dir):
                _save_shared_base_model(model, tokenizer, shared_dir)
                print(f"Base model saved to shared cache: {shared_dir}")
            else:
                print(f"Base model reused from shared cache: {shared_dir}")
            _link_shared_base_model(base_model_ckpt, shared_dir)
            print(f"Run base model link: {base_model_ckpt} -> {shared_dir}")
        else:
            if not base_model_ckpt.exists():
                # Resume started before the shared-cache layout; rebuild the link.
                if not _shared_base_model_is_complete(shared_dir):
                    raise FileNotFoundError(
                        f"Resuming at stage {start_stage} but base model not found at {base_model_ckpt} "
                        f"and shared cache {shared_dir} is empty/incomplete."
                    )
                _link_shared_base_model(base_model_ckpt, shared_dir)
    else:
        if start_stage == 1:
            base_model_ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(base_model_ckpt))
            tokenizer.save_pretrained(str(base_model_ckpt))
            print(f"Base model saved: {base_model_ckpt}")
        elif not base_model_ckpt.exists():
            raise FileNotFoundError(
                f"Resuming at stage {start_stage} but base model checkpoint not found at {base_model_ckpt}."
            )

    # Build the CL method object once per run. Persistent state (O-LoRA A
    # snapshots, InfLoRA covariance) lives under <run>/cl_state/ and is
    # reloaded on resume so the per-stage history carries across runs.
    cl_state_dir = run_output_dir / "cl_state"
    cl_state_dir.mkdir(parents=True, exist_ok=True)
    cl_method = build_cl_method(cl_method_name, **(cl_method_kwargs or {}))
    if start_stage > 1:
        cl_method.load(str(cl_state_dir))
    sapt_active = bool(getattr(cl_method, "requires_no_merge", False))
    print(f"CL method: {cl_method.name} | state dir: {cl_state_dir} | no_merge={sapt_active}")

    for idx in range(start_stage, len(sequence.tasks) + 1):
        task = sequence.tasks[idx - 1]
        task_name = task.name
        safe_task_name = _safe_name(task_name)

        # Namespace by run_name: shared paths let later runs clobber training_report.json that eval_standalone reads back into every run's stage_record.
        stage_train_dir = train_output_dir / sequence_name / run_output_dir.name / f"stage_{idx:02d}_{safe_task_name}"
        stage_eval_dir = run_output_dir / "stages" / f"stage_{idx:02d}_{safe_task_name}"

        print(f"\n=== Stage {idx}/{len(sequence.tasks)} | Training task: {task_name} ===")
        retain_tasks = list(sequence.tasks[:idx - 1]) if idx > 1 else None

        if idx == 1:
            slice_cache_context = f"base_model:{model_name}"
        else:
            prev_task_name = sequence.tasks[idx - 2].name
            prev_safe = _safe_name(prev_task_name)
            prev_adapter_dir = checkpoint_root / f"stage_{idx - 1:02d}_{prev_safe}" / "adapter"
            slice_cache_context = f"adapter:{prev_adapter_dir}"

        adapter_checkpoint_dir = checkpoint_root / f"stage_{idx:02d}_{safe_task_name}" / "adapter"

        sapt_adapter_name = SAPTMethod.adapter_name_for_stage(idx) if sapt_active else None

        model, train_report = train_on_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            output_dir=str(stage_train_dir),
            eval_size=eval_size,
            seed=seed,
            retain_tasks=retain_tasks,
            rank=rank,
            lora_alpha=lora_alpha,
            adapter_checkpoint_path=str(adapter_checkpoint_dir),
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
            save_intermediate_checkpoints=save_intermediate_checkpoints,
            slice_init_method=slice_init_method,
            slice_projection_method=slice_projection_method,
            slice_cosine_threshold=slice_cosine_threshold,
            slice_per_layer_threshold=slice_per_layer_threshold,
            slice_per_layer_threshold_delta=slice_per_layer_threshold_delta,
            slice_cagrad_c=slice_cagrad_c,
            slice_gradvac_phi=slice_gradvac_phi,
            slice_gradvac_beta=slice_gradvac_beta,
            slice_magnitude_preserve=slice_magnitude_preserve,
            slice_nullspace_rank=slice_nullspace_rank,
            slice_nullspace_sv_threshold=slice_nullspace_sv_threshold,
            slice_svd_selection=slice_svd_selection,
            cl_method=cl_method,
            stage_idx=idx,
            sapt_mode=sapt_active,
            sapt_adapter_name=sapt_adapter_name,
        )

        # SAPT post-stage: train (or grow + retrain) the shared-attention router
        # on pseudo-samples generated by every loaded adapter. Must run BEFORE
        # cl_method.save() so the router state is included in the persisted
        # cl_state/sapt/sapt_state.pt.
        if sapt_active and hasattr(cl_method, "run_arm"):
            try:
                arm_stats = cl_method.run_arm(model, tokenizer)
                print(f"  SAPT ARM stats: {arm_stats}")
            except Exception as exc:
                logging.getLogger("cl_lora.orchestrator.sapt").warning(
                    "ARM training failed at stage %d: %s", idx, exc,
                )

        cl_method.save(str(cl_state_dir))
        print(f"  Adapter checkpoint saved: {adapter_checkpoint_dir}")
        print(f"  CL-method state saved: {cl_state_dir} | {cl_method.metadata()}")

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

        # Build the ordered list of adapter checkpoint paths up to this stage.
        # eval_standalone uses these to reconstruct model+adapter_1+...+adapter_k.
        stage_adapter_paths = [
            str((checkpoint_root / f"stage_{i + 1:02d}_{_safe_name(sequence.tasks[i].name)}" / "adapter").resolve())
            for i in range(idx)
        ]

        # Write a manifest that a standalone eval pass can consume without
        # needing to re-run training.
        stage_eval_dir.mkdir(parents=True, exist_ok=True)
        sapt_router_path: str | None = None
        if sapt_active:
            # Prefer the per-stage snapshot (enables full-matrix AP evaluation).
            # Fall back to the canonical router.pt for backward compat.
            stage_router_file = cl_state_dir / "sapt" / f"router_stage_{idx:02d}.pt"
            router_file = cl_state_dir / "sapt" / "router.pt"
            if stage_router_file.exists():
                sapt_router_path = str(stage_router_file.resolve())
            elif router_file.exists():
                sapt_router_path = str(router_file.resolve())

        _write_json(
            stage_eval_dir / "eval_manifest.json",
            {
                "stage": idx,
                "trained_task": task_name,
                "sequence": sequence_name,
                "seen_tasks": [getattr(t, "name", str(t)) for t in seen_tasks],
                "eval_seen_tasks": [getattr(t, "name", str(t)) for t in eval_seen],
                "base_model_path": str(base_model_ckpt.resolve()),
                "adapter_paths": stage_adapter_paths,
                "general_eval_keys": general_eval_keys,
                "skip_general_eval": bool(skip_general),
                "quick_eval": bool(quick_eval),
                "eval_size": int(eval_size),
                "task_eval_samples": int(task_eval_samples),
                "task_eval_max_new_tokens": int(task_eval_max_new_tokens),
                "seed": int(seed),
                "train_output_dir": str(stage_train_dir),
                "cl_method": cl_method.name,
                "sapt_router_path": sapt_router_path,
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
            # SAPT: wrap the in-memory un-merged peft model in SAPTWrapper so
            # forward / generate pass through the shared-attention router.
            if sapt_active:
                from .sapt import SAPTRouter, SAPTWrapper

                router_packed = cl_method.get_router_packed()  # type: ignore[attr-defined]
                if router_packed is None:
                    print("  WARNING: SAPT eval requested but no router available; falling back to bare model.")
                    eval_model = model
                else:
                    router = SAPTRouter.from_packed(router_packed)
                    eval_model = SAPTWrapper(
                        model, router.to(next(model.parameters()).device),
                        cl_method.all_adapter_names(),  # type: ignore[attr-defined]
                    )
            else:
                eval_model = model

            evaluation = evaluate_all(
                model=eval_model,
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

        # Adapter checkpoints are kept for all stages — they are needed by
        # eval_standalone to reconstruct any cumulative model state.

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
        "--base-model-cache",
        default="outputs/base_models",
        help="Directory holding one shared copy of the base model+tokenizer per model name. "
             "Each run's checkpoints/base_model becomes a symlink into this cache. "
             "Pass an empty string to disable sharing and save a full copy under each run.",
    )
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
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=2,
        help="LoRA alpha (rsLoRA scaling = alpha / sqrt(r)). Defaults to 2.",
    )
    parser.add_argument("--slice-grad-project", action="store_true", help="Project current-task gradients against retain gradients for slice init.")
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
        help="Only use the most recent previous task for retain, with same batch size as current task.")
    parser.add_argument("--slice-projection-method",
        choices=["pcgrad", "cagrad", "gradvac", "nullspace", "magnitude_preserving"],
        default="pcgrad",
        help="Advanced projection method. 'pcgrad' preserves existing behavior.")
    parser.add_argument("--slice-cosine-threshold", type=float, default=None,
        help="Cosine threshold tau. Projection fires when cos(g_f,g_r) < tau. "
             "If unset, falls back to dot-sign gating.")
    parser.add_argument("--slice-per-layer-threshold", action="store_true",
        help="Use a per-module threshold computed as median(cos) - delta.")
    parser.add_argument("--slice-per-layer-threshold-delta", type=float, default=0.0)
    parser.add_argument("--slice-cagrad-c", type=float, default=0.5,
        help="CAGrad interpolation strength in [0,1].")
    parser.add_argument("--slice-gradvac-phi", type=float, default=0.0,
        help="GradVac target cosine (initial).")
    parser.add_argument("--slice-gradvac-beta", type=float, default=0.5,
        help="GradVac EMA beta for target cosine.")
    parser.add_argument("--slice-magnitude-preserve", action="store_true",
        help="Rescale projected gradient back to original norm (idea A.6).")
    parser.add_argument("--slice-nullspace-rank", type=int, default=8,
        help="Rank for null-space projection.")
    parser.add_argument("--slice-nullspace-sv-threshold", type=float, default=0.0,
        help="Relative singular-value cutoff for null-space projection.")
    parser.add_argument("--slice-svd-selection",
        choices=["lora_ga", "top_r_no_sigma"], default="lora_ga",
        help="SVD selection rule for LoRA A/B. 'top_r_no_sigma' uses the "
             "top-r singular vectors without sigma weighting (idea C.16 variant).")
    parser.add_argument("--slice-init-method", choices=["slice", "lora_ga", "loram"],
        default="slice",
        help="Initialization method: 'slice' (default), 'lora_ga' (SVD on current-task gradients only), "
             "or 'loram' (DST-based, no gradients).")
    parser.add_argument("--cl-method",
        choices=sorted(CL_METHOD_REGISTRY.keys()),
        default="vanilla",
        help="Continual-learning training method (composes with any LoRA init). "
             "'vanilla' (default) is the existing per-stage train+merge pipeline. "
             "'o_lora' adds an orthogonality regularizer between current and prior "
             "task A matrices. 'inflora' projects the new task's A onto the null "
             "space of past-task input feature covariance.")
    parser.add_argument("--cl-o-lora-lambda", type=float, default=0.5,
        help="O-LoRA orthogonality regularizer weight. Used only when --cl-method=o_lora.")
    parser.add_argument("--cl-inflora-nullspace-rank", type=int, default=64,
        help="Top-k subspace size of past-task input covariance to project A out of. "
             "Used only when --cl-method=inflora.")
    parser.add_argument("--cl-inflora-max-cov-batches", type=int, default=32,
        help="Max forward batches per stage to estimate input covariance. "
             "Used only when --cl-method=inflora.")
    parser.add_argument("--cl-inflora-cov-batch-size", type=int, default=8,
        help="Batch size used during InfLoRA covariance estimation forward passes.")
    parser.add_argument("--cl-sapt-key-dim", type=int, default=64,
        help="SAPT router key/query dimensionality.")
    parser.add_argument("--cl-sapt-arm-n-samples", type=int, default=64,
        help="Pseudo-samples per task generated during ARM router training.")
    parser.add_argument("--cl-sapt-arm-max-new-tokens", type=int, default=32,
        help="Max new tokens per pseudo-sample during ARM generation.")
    parser.add_argument("--cl-sapt-arm-max-input-length", type=int, default=128,
        help="Max input length when tokenizing seed prompts for ARM generation.")
    parser.add_argument("--cl-sapt-arm-n-epochs", type=int, default=3,
        help="Router training epochs over the ARM pseudo-sample pool.")
    parser.add_argument("--cl-sapt-arm-batch-size", type=int, default=4,
        help="Batch size for router optimization steps.")
    parser.add_argument("--cl-sapt-arm-learning-rate", type=float, default=1e-3,
        help="Router AdamW learning rate.")
    parser.add_argument("--cl-sapt-seed-prompts-per-task", type=int, default=32,
        help="How many training prompts to cache per task for ARM seeding.")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
        help="Keep all intermediate stage checkpoints. By default only the latest is kept.")
    parser.add_argument("--save-checkpoints", action="store_true",
        help="Save intermediate HuggingFace Trainer checkpoints (checkpoint-N dirs) during training. "
             "By default only the final adapter is saved to results.")
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
        lora_alpha=args.lora_alpha,
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
        slice_projection_method=args.slice_projection_method,
        slice_cosine_threshold=args.slice_cosine_threshold,
        slice_per_layer_threshold=args.slice_per_layer_threshold,
        slice_per_layer_threshold_delta=args.slice_per_layer_threshold_delta,
        slice_cagrad_c=args.slice_cagrad_c,
        slice_gradvac_phi=args.slice_gradvac_phi,
        slice_gradvac_beta=args.slice_gradvac_beta,
        slice_magnitude_preserve=args.slice_magnitude_preserve,
        slice_nullspace_rank=args.slice_nullspace_rank,
        slice_nullspace_sv_threshold=args.slice_nullspace_sv_threshold,
        slice_svd_selection=args.slice_svd_selection,
        keep_all_checkpoints=args.keep_all_checkpoints,
        save_intermediate_checkpoints=args.save_checkpoints,
        general_eval_strategy=args.general_eval_strategy,
        seen_eval_strategy=args.seen_eval_strategy,
        train_only=args.train_only,
        cl_method_name=args.cl_method,
        cl_method_kwargs={
            "lambda_orth": args.cl_o_lora_lambda,
            "nullspace_rank": args.cl_inflora_nullspace_rank,
            "max_cov_batches": args.cl_inflora_max_cov_batches,
            "cov_batch_size": args.cl_inflora_cov_batch_size,
            "max_seq_length": 256,
            "seed": args.seed,
            # SAPT-specific (filtered out by build_cl_method when not applicable):
            "key_dim": args.cl_sapt_key_dim,
            "arm_n_samples_per_task": args.cl_sapt_arm_n_samples,
            "arm_max_new_tokens": args.cl_sapt_arm_max_new_tokens,
            "arm_max_input_length": args.cl_sapt_arm_max_input_length,
            "arm_n_epochs": args.cl_sapt_arm_n_epochs,
            "arm_batch_size": args.cl_sapt_arm_batch_size,
            "arm_learning_rate": args.cl_sapt_arm_learning_rate,
            "seed_prompts_per_task": args.cl_sapt_seed_prompts_per_task,
        },
        orchestrator_config=orchestrator_config,
        base_model_cache_dir=(args.base_model_cache or None),
    )

    if not args.train_only:
        print("\n=== Final Metrics ===")
        print(json.dumps(payload["summary"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
