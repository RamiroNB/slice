from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List, Tuple

import torch
from datasets import concatenate_datasets

from ..lora_config import build_lora_config
from ..load_dataset import load_training_dataset
from .cache import (
    SliceCacheEntry,
    load_slice_cache,
    make_cache_key,
    save_ab_stats_csv,
    save_projection_stats_json,
    save_slice_cache,
)
from .config import SliceInitConfig
from .decompose import build_ab_from_gradient, build_ab_loram
from .gradients import accumulate_gradients, combine_grads, project_forget_gradients
from .utils import build_dataloader, model_device, target_weight_params, tokenize_dataset

logger = logging.getLogger("cl_lora.slice.compute")


def _lora_ga_incompatible_flags(config: SliceInitConfig) -> List[str]:
    invalid_flags: List[str] = []
    if bool(config.grad_project):
        invalid_flags.append("grad_project")
    if bool(config.add_retain_grad):
        invalid_flags.append("add_retain_grad")
    if config.retain_batch_size is not None:
        invalid_flags.append("retain_batch_size")
    if config.retain_grad_accum is not None:
        invalid_flags.append("retain_grad_accum")
    if str(config.retain_batch_size_set) != "all_tasks":
        invalid_flags.append("retain_batch_size_set")
    if bool(config.single_retain_task_mode):
        invalid_flags.append("single_retain_task_mode")
    if float(config.retain_scale) != 1.0:
        invalid_flags.append("retain_scale")
    return invalid_flags


def compute_loram_inits(
    model: torch.nn.Module,
    *,
    config: SliceInitConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute LoRAM initialization (DST-based, no gradients needed)."""
    lora_cfg = build_lora_config()
    target_params = target_weight_params(model, lora_cfg.target_modules)
    if not target_params:
        raise RuntimeError("No target modules matched for LoRAM initialization.")

    device = model_device(model)
    r_use = config.rank or int(getattr(lora_cfg, "r", 8))
    logger.info("Computing LoRAM inits: modules=%d rank=%d", len(target_params), r_use)

    inits = {}
    for name, param in target_params.items():
        d_out, d_in = param.shape
        weight_var = float(param.detach().float().var().item())
        ab = build_ab_loram(d_out, d_in, r_use, weight_var, device=device, dtype=param.dtype)
        logger.debug("LoRAM A/B for %s: A_shape=%s B_shape=%s weight_var=%.6g",
                      name, tuple(ab['A'].shape), tuple(ab['B'].shape), weight_var)
        inits[name] = ab
    return inits


def compute_slice_inits(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_tasks=None,
    *,
    config: SliceInitConfig,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]]:
    if config.init_method == "lora_ga":
        # Hard guard to avoid accidental retain/projection usage with LoRA-GA.
        invalid_flags = _lora_ga_incompatible_flags(config)

        if invalid_flags:
            raise ValueError(
                "init_method='lora_ga' is incompatible with retain/projection settings: "
                f"{', '.join(invalid_flags)}. "
                "Use init_method='slice' for retain-gradient projection."
            )

    # LoRA-GA baseline: ignore retain tasks entirely
    if config.init_method == "lora_ga":
        retain_tasks = []
        logger.info("LoRA-GA mode: ignoring retain tasks")
    else:
        retain_tasks = retain_tasks or []
    retain_names = [getattr(rt, "name", str(rt)) for rt in retain_tasks] or None
    logger.info(
        "Starting slice init (method=%s): forget=%s retain=%s max_steps=%s batch_size=%s",
        config.init_method,
        getattr(forget_task, "name", str(forget_task)),
        retain_names,
        config.max_steps,
        config.per_device_batch_size,
    )
    lora_cfg = build_lora_config()
    target_params = target_weight_params(model, lora_cfg.target_modules)
    if not target_params:
        logger.error("No target modules matched for slice initialization.")
        raise RuntimeError("No target modules matched for slice initialization.")

    logger.info("Matched %d target weight parameters for slice init", len(target_params))
    forget_ds, _ = load_training_dataset(task=forget_task, eval_size=1, seed=config.seed)
    forget_ds = tokenize_dataset(forget_ds, tokenizer=tokenizer, max_length=config.max_seq_length)
    logger.info("Building forget dataloader: dataset_size=%d batch_size=%d", len(forget_ds), config.per_device_batch_size)
    forget_loader = build_dataloader(
        forget_ds,
        tokenizer=tokenizer,
        batch_size=config.per_device_batch_size,
        seed=config.seed,
    )

    device = model_device(model)
    grads_f, steps_f = accumulate_gradients(
        model=model,
        dataloader=forget_loader,
        target_params=target_params,
        device=device,
        max_steps=config.max_steps,
    )
    logger.info("Collected forget gradients: steps=%d modules=%d", steps_f, len(grads_f))
    for i, (n, g) in enumerate(grads_f.items()):
        if i >= 5:
            break
        logger.debug("forget grad sample: module=%s norm=%.6g", n, float(g.norm().item()))

    grads_r = None
    steps_r = 0
    if retain_tasks:
        if config.single_retain_task_mode:
            retain_tasks = [retain_tasks[-1]]
            retain_names = [getattr(retain_tasks[0], "name", str(retain_tasks[0]))]
            retain_bs = config.per_device_batch_size
            retain_max_steps = config.max_steps
            logger.info(
                "Single retain task mode: task=%s batch_size=%d max_steps=%d",
                retain_names[0], retain_bs, retain_max_steps,
            )
        else:
            retain_bs = config.retain_batch_size if config.retain_batch_size is not None else config.per_device_batch_size
            retain_max_steps = config.retain_grad_accum if config.retain_grad_accum is not None else config.max_steps
            logger.info(
                "Retain tasks (%d): %s | mode=%s batch_size=%d max_steps=%d",
                len(retain_tasks), retain_names, config.retain_batch_size_set, retain_bs, retain_max_steps,
            )

        if config.retain_batch_size_set == "all_tasks":
            all_retain_ds = []
            for rt in retain_tasks:
                ds, _ = load_training_dataset(task=rt, eval_size=1, seed=config.seed)
                ds = tokenize_dataset(ds, tokenizer=tokenizer, max_length=config.max_seq_length)
                all_retain_ds.append(ds)
            combined_ds = concatenate_datasets(all_retain_ds)
            logger.info("Retain dataloader (all_tasks): %d total samples, batch_size=%d", len(combined_ds), retain_bs)
            retain_loader = build_dataloader(combined_ds, tokenizer=tokenizer, batch_size=retain_bs, seed=config.seed)
            grads_r, steps_r = accumulate_gradients(
                model=model, dataloader=retain_loader, target_params=target_params,
                device=device, max_steps=retain_max_steps,
            )
        elif config.retain_batch_size_set == "each_task":
            grads_r = {name: torch.zeros_like(param, device=device) for name, param in target_params.items()}
            steps_r = 0
            for rt in retain_tasks:
                rt_name = getattr(rt, "name", str(rt))
                ds, _ = load_training_dataset(task=rt, eval_size=1, seed=config.seed)
                ds = tokenize_dataset(ds, tokenizer=tokenizer, max_length=config.max_seq_length)
                logger.info("Retain dataloader (each_task): task=%s, %d samples, batch_size=%d", rt_name, len(ds), retain_bs)
                rt_loader = build_dataloader(ds, tokenizer=tokenizer, batch_size=retain_bs, seed=config.seed)
                grads_rt, steps_rt = accumulate_gradients(
                    model=model, dataloader=rt_loader, target_params=target_params,
                    device=device, max_steps=retain_max_steps,
                )
                for name in grads_r:
                    grads_r[name] = grads_r[name] + grads_rt[name]
                steps_r += steps_rt
                logger.info("Accumulated retain grads for task=%s: steps=%d", rt_name, steps_rt)
        else:
            raise ValueError(
                f"Unknown retain_batch_size_set: {config.retain_batch_size_set!r}. "
                "Expected 'all_tasks' or 'each_task'."
            )

        logger.info("Collected retain gradients: total_steps=%d modules=%d", steps_r, len(grads_r))
        for i, (n, g) in enumerate(grads_r.items()):
            if i >= 5:
                break
            logger.debug("retain grad sample: module=%s norm=%.6g", n, float(g.norm().item()))

    denom_f = max(1, steps_f)
    grads_f = {k: v / float(denom_f) for k, v in grads_f.items()}
    if grads_r is not None:
        denom_r = max(1, steps_r)
        grads_r = {k: v / float(denom_r) for k, v in grads_r.items()}

    if config.grad_project and grads_r is not None:
        global_projection = str(config.grad_projection_mode).lower() == "global"
        logger.info(
            "Projecting slice gradients (mode=%s, add_retain_grad=%s)",
            "global" if global_projection else "per_module",
            config.add_retain_grad,
        )
        combined, projection_stats = project_forget_gradients(
            grads_forget=grads_f,
            grads_retain=grads_r,
            global_projection=global_projection,
            add_retain_grad=config.add_retain_grad,
            return_stats=True,
        )
        projection_stats["applied"] = True
        logger.info("Built projected gradient matrix for %d modules", len(combined))
    elif config.grad_project and grads_r is None:
        logger.info("grad_project=True but no retain task provided; using forget gradients without projection")
        combined = grads_f
        projection_stats = {
            "applied": False,
            "reason": "grad_project_true_but_no_retain_grads",
            "mode": str(config.grad_projection_mode),
            "gamma": None,
        }
    else:
        combined = combine_grads(grads_f, grads_r, config.retain_scale)
        logger.info("Built combined gradient matrix for %d modules (retain_scale=%s)", len(combined), config.retain_scale)
        projection_stats = {
            "applied": False,
            "reason": "grad_project_disabled",
            "mode": "none",
            "gamma": None,
        }

    r_use = config.rank or int(getattr(lora_cfg, "r", 8))
    inits = {}
    for name, g in combined.items():
        logger.debug("Building A/B for module %s: G_shape=%s r=%d", name, tuple(g.shape), r_use)
        weight_var = float(target_params[name].detach().float().var().item())
        ab = build_ab_from_gradient(g, r=r_use, weight_var=weight_var)
        logger.debug("Built A/B for %s: A_shape=%s B_shape=%s", name, tuple(ab['A'].shape), tuple(ab['B'].shape))
        inits[name] = ab
    return inits, projection_stats


def _task_fingerprint(task_obj) -> Optional[Dict[str, object]]:
    if task_obj is None:
        return None
    fp: Dict[str, object] = {
        "type": task_obj.__class__.__name__,
        "name": getattr(task_obj, "name", str(task_obj)),
    }
    for k in ("ni_id", "hf_config", "source", "category"):
        if hasattr(task_obj, k):
            fp[k] = getattr(task_obj, k)
    for k in ("hf_dataset", "language", "metric"):
        if hasattr(task_obj, k):
            fp[k] = getattr(task_obj, k)
    return fp


def load_or_compute_slice_inits(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_tasks,
    *,
    config: SliceInitConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if config.init_method == "lora_ga":
        # Enforce guard before cache lookup so incompatible settings cannot be hidden by cache hits.
        invalid_flags = _lora_ga_incompatible_flags(config)
        if invalid_flags:
            raise ValueError(
                "init_method='lora_ga' is incompatible with retain/projection settings: "
                f"{', '.join(invalid_flags)}. "
                "Use init_method='slice' for retain-gradient projection."
            )

    lora_cfg = build_lora_config(r=int(config.rank or 128))
    lora_payload = {
        "r": int(getattr(lora_cfg, "r", 0) or 0),
        "lora_alpha": float(getattr(lora_cfg, "lora_alpha", 1.0)),
        "lora_dropout": float(getattr(lora_cfg, "lora_dropout", 0.0)),
        "bias": str(getattr(lora_cfg, "bias", "none")),
        "use_rslora": bool(getattr(lora_cfg, "use_rslora", False)) if hasattr(lora_cfg, "use_rslora") else None,
        "target_modules": list(getattr(lora_cfg, "target_modules", []) or []),
    }

    is_lora_ga = (config.init_method == "lora_ga")
    payload = {
        "init_method": config.init_method,
        "cache_context": config.cache_context,
        "forget_task": _task_fingerprint(forget_task),
        # Canonicalize LoRA-GA cache identity: retain tasks are ignored by design.
        "retain_tasks": None if is_lora_ga else ([_task_fingerprint(rt) for rt in (retain_tasks or [])] or None),
        "rank": config.rank,
        "seed": config.seed,
        "max_seq_length": config.max_seq_length,
        "max_steps": config.max_steps,
        "batch_size": config.per_device_batch_size,
        "retain_scale": 1.0 if is_lora_ga else config.retain_scale,
        "grad_project": False if is_lora_ga else config.grad_project,
        "grad_projection_mode": "per_module" if is_lora_ga else config.grad_projection_mode,
        "add_retain_grad": False if is_lora_ga else config.add_retain_grad,
        "retain_batch_size": None if is_lora_ga else config.retain_batch_size,
        "retain_grad_accum": None if is_lora_ga else config.retain_grad_accum,
        "retain_batch_size_set": "all_tasks" if is_lora_ga else config.retain_batch_size_set,
        "single_retain_task_mode": False if is_lora_ga else config.single_retain_task_mode,
        "lora": lora_payload,
        "model": {
            "class": model.__class__.__name__,
        },
    }
    cache_key = make_cache_key(payload)
    cached = load_slice_cache(config.cache_dir, cache_key, device=model_device(model))
    if cached is not None:
        save_ab_stats_csv(config.cache_dir, cache_key, cached.inits)
        logger.info("Slice cache hit: cache_dir=%s cache_key=%s modules=%d", config.cache_dir, cache_key, len(cached.inits))
        return cached.inits
    logger.info("Slice cache miss: will compute inits (cache_dir=%s cache_key=%s)", config.cache_dir, cache_key)

    if config.init_method == "loram":
        inits = compute_loram_inits(model=model, config=config)
        projection_stats = {
            "applied": False,
            "reason": "init_method_loram",
            "mode": "none",
            "gamma": None,
        }
    else:
        inits, projection_stats = compute_slice_inits(
            model=model,
            tokenizer=tokenizer,
            forget_task=forget_task,
            retain_tasks=retain_tasks,
            config=config,
        )

    save_slice_cache(
        config.cache_dir,
        cache_key,
        SliceCacheEntry(inits=inits),
        meta={"payload": payload},
    )
    save_ab_stats_csv(config.cache_dir, cache_key, inits)
    save_projection_stats_json(config.cache_dir, cache_key, projection_stats)
    logger.info("Saved slice cache: cache_dir=%s cache_key=%s modules=%d", config.cache_dir, cache_key, len(inits))
    return inits
