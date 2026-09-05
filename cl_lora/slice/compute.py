from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Optional, List, Tuple

import torch
from datasets import concatenate_datasets

from ..lora_config import build_lora_config
from ..load_dataset import load_training_dataset
from .cache import (
    load_slice_cache,
    make_cache_key,
    save_ab_stats_csv,
    save_projection_stats_json,
)
from .config import SliceInitConfig
from .decompose import build_ab_from_gradient, build_ab_loram
from .gradients import accumulate_gradients, combine_grads, project_current_gradients
from .projections import project_gradients_advanced
from .utils import build_dataloader, model_device, target_weight_params, tokenize_dataset

logger = logging.getLogger("cl_lora.slice.compute")


def _lora_ga_incompatible_flags(config: SliceInitConfig) -> List[str]:
    invalid_flags: List[str] = []
    if bool(config.grad_project):
        invalid_flags.append("grad_project")
    if bool(config.grad_project_always):
        invalid_flags.append("grad_project_always")
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
    if float(getattr(config, "dot_significance_k", 0.0)) != 0.0:
        invalid_flags.append("dot_significance_k")
    if getattr(config, "gamma_rel_cap", None) is not None:
        invalid_flags.append("gamma_rel_cap")
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
    current_task,
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
        "Starting slice init (method=%s): current=%s retain=%s max_steps=%s batch_size=%s",
        config.init_method,
        getattr(current_task, "name", str(current_task)),
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
    # The probe must draw from the same train side the trainer does. With
    # split_seed unset that is the legacy eval_size=1 behaviour (the partition
    # tracks `seed`, so there is no fixed eval set to protect); with it pinned we
    # carve out the identical held-out block, otherwise the init's gradients are
    # computed on the very questions the run is later scored on.
    probe_eval_size = 1 if config.split_seed is None else int(config.eval_size)
    if config.split_seed is not None:
        logger.info(
            "Slice probe honouring pinned split: split_seed=%d eval_size=%d "
            "(held-out questions excluded from the gradient probe)",
            int(config.split_seed), probe_eval_size,
        )
    current_ds, _ = load_training_dataset(
        task=current_task, eval_size=probe_eval_size, seed=config.seed,
        split_seed=config.split_seed,
    )
    current_ds = tokenize_dataset(
        current_ds, tokenizer=tokenizer, max_length=config.max_seq_length,
        completion_only=config.completion_only, log_context="slice/current",
    )
    logger.info("Building current-task dataloader: dataset_size=%d batch_size=%d", len(current_ds), config.per_device_batch_size)
    current_loader = build_dataloader(
        current_ds,
        tokenizer=tokenizer,
        batch_size=config.per_device_batch_size,
        seed=config.seed,
    )

    device = model_device(model)
    grads_current, steps_current = accumulate_gradients(
        model=model,
        dataloader=current_loader,
        target_params=target_params,
        device=device,
        max_steps=config.max_steps,
    )
    logger.info("Collected current-task gradients: steps=%d modules=%d", steps_current, len(grads_current))
    for i, (n, g) in enumerate(grads_current.items()):
        if i >= 5:
            break
        logger.debug("current grad sample: module=%s norm=%.6g", n, float(g.norm().item()))

    grads_r = None
    steps_r = 0
    # Per-retain-batch samples of <g_current_raw, g_retain_batch>; scaled after
    # normalization so their mean equals the global dot the projection gates on.
    raw_dot_samples: List[float] = []
    collect_dots = bool(config.grad_project)
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
                ds, _ = load_training_dataset(
                    task=rt, eval_size=probe_eval_size, seed=config.seed,
                    split_seed=config.split_seed,
                )
                ds = tokenize_dataset(
                    ds, tokenizer=tokenizer, max_length=config.max_seq_length,
                    completion_only=config.completion_only, log_context="slice/retain",
                )
                all_retain_ds.append(ds)
            combined_ds = concatenate_datasets(all_retain_ds)
            logger.info("Retain dataloader (all_tasks): %d total samples, batch_size=%d", len(combined_ds), retain_bs)
            retain_loader = build_dataloader(combined_ds, tokenizer=tokenizer, batch_size=retain_bs, seed=config.seed)
            grads_r, steps_r = accumulate_gradients(
                model=model, dataloader=retain_loader, target_params=target_params,
                device=device, max_steps=retain_max_steps,
                dot_reference=grads_current if collect_dots else None,
                dot_samples=raw_dot_samples if collect_dots else None,
            )
        elif config.retain_batch_size_set == "each_task":
            grads_r = {name: torch.zeros_like(param, device=device) for name, param in target_params.items()}
            steps_r = 0
            for rt in retain_tasks:
                rt_name = getattr(rt, "name", str(rt))
                ds, _ = load_training_dataset(
                    task=rt, eval_size=probe_eval_size, seed=config.seed,
                    split_seed=config.split_seed,
                )
                ds = tokenize_dataset(
                    ds, tokenizer=tokenizer, max_length=config.max_seq_length,
                    completion_only=config.completion_only, log_context="slice/retain",
                )
                logger.info("Retain dataloader (each_task): task=%s, %d samples, batch_size=%d", rt_name, len(ds), retain_bs)
                rt_loader = build_dataloader(ds, tokenizer=tokenizer, batch_size=retain_bs, seed=config.seed)
                # Accumulate this task's gradients straight into the shared retain
                # buffer (one task at a time) so we never hold a second full-size
                # gradient set. Summing in place is numerically identical to the
                # previous per-task buffer + add, preserving each_task's effective
                # batch size and per-task weighting.
                grads_r, steps_rt = accumulate_gradients(
                    model=model, dataloader=rt_loader, target_params=target_params,
                    device=device, max_steps=retain_max_steps, grads=grads_r,
                    dot_reference=grads_current if collect_dots else None,
                    dot_samples=raw_dot_samples if collect_dots else None,
                )
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

    denom_c = max(1, steps_current)
    grads_current = {k: v / float(denom_c) for k, v in grads_current.items()}
    if grads_r is not None:
        denom_r = max(1, steps_r)
        grads_r = {k: v / float(denom_r) for k, v in grads_r.items()}
    # Scale the raw per-batch dots so that mean(dot_samples) equals the global
    # dot computed on the normalized gradients: each raw sample used the raw
    # (un-normalized) current-gradient sum as its reference.
    dot_samples = [s / float(denom_c) for s in raw_dot_samples]
    if dot_samples:
        logger.info(
            "Collected %d per-batch conflict-dot samples: mean=%.6g",
            len(dot_samples), sum(dot_samples) / len(dot_samples),
        )

    if config.grad_project and grads_r is not None:
        method = str(config.projection_method).lower()
        global_projection = str(config.grad_projection_mode).lower() == "global"
        use_advanced = (
            method != "pcgrad"
            or config.cosine_threshold is not None
            or bool(config.per_layer_threshold)
            or bool(config.magnitude_preserve)
        )
        if use_advanced:
            if float(config.dot_significance_k) != 0.0 or config.gamma_rel_cap is not None:
                logger.warning(
                    "dot_significance_k / gamma_rel_cap are only implemented for the "
                    "basic pcgrad path and are IGNORED by the advanced projection "
                    "(method=%s cosine_threshold=%s).",
                    method, config.cosine_threshold,
                )
            logger.info(
                "Advanced projection: method=%s mode=%s cos_tau=%s per_layer=%s mag_preserve=%s",
                method,
                "global" if global_projection else "per_module",
                config.cosine_threshold,
                config.per_layer_threshold,
                config.magnitude_preserve,
            )
            combined, projection_stats = project_gradients_advanced(
                grads_current=grads_current,
                grads_retain=grads_r,
                method=method,
                cosine_threshold=config.cosine_threshold,
                per_layer_threshold=bool(config.per_layer_threshold),
                per_layer_threshold_delta=float(config.per_layer_threshold_delta),
                pcgrad_c=float(config.pcgrad_c),
                gradvac_phi=float(config.gradvac_phi),
                gradvac_beta=float(config.gradvac_beta),
                magnitude_preserve=bool(config.magnitude_preserve),
                nullspace_rank=int(config.nullspace_rank),
                nullspace_sv_threshold=float(config.nullspace_sv_threshold),
                always_project=bool(config.grad_project_always),
                add_retain_grad=bool(config.add_retain_grad),
                global_projection=global_projection,
            )
            logger.info("Built advanced projected gradient matrix for %d modules", len(combined))
        else:
            logger.info(
                "Projecting slice gradients (mode=%s, always_project=%s, add_retain_grad=%s)",
                "global" if global_projection else "per_module",
                config.grad_project_always,
                config.add_retain_grad,
            )
            combined, projection_stats = project_current_gradients(
                grads_current=grads_current,
                grads_retain=grads_r,
                global_projection=global_projection,
                always_project=config.grad_project_always,
                add_retain_grad=config.add_retain_grad,
                return_stats=True,
                dot_samples=dot_samples,
                significance_k=float(config.dot_significance_k),
                gamma_rel_cap=config.gamma_rel_cap,
            )
            projection_stats["applied"] = True
            logger.info("Built projected gradient matrix for %d modules", len(combined))
    elif config.grad_project and grads_r is None:
        logger.info("grad_project=True but no retain task provided; using current-task gradients without projection")
        combined = grads_current
        projection_stats = {
            "applied": False,
            "reason": "grad_project_true_but_no_retain_grads",
            "mode": str(config.grad_projection_mode),
            "always_project": bool(config.grad_project_always),
            "gamma": None,
        }
    else:
        combined = combine_grads(grads_current, grads_r, config.retain_scale)
        logger.info("Built combined gradient matrix for %d modules (retain_scale=%s)", len(combined), config.retain_scale)
        projection_stats = {
            "applied": False,
            "reason": "grad_project_disabled",
            "mode": "none",
            "always_project": bool(config.grad_project_always),
            "gamma": None,
        }

    r_use = config.rank or int(getattr(lora_cfg, "r", 8))
    inits = {}
    for name, g in combined.items():
        logger.debug("Building A/B for module %s: G_shape=%s r=%d", name, tuple(g.shape), r_use)
        weight_var = float(target_params[name].detach().float().var().item())
        ab = build_ab_from_gradient(
            g, r=r_use, weight_var=weight_var,
            svd_selection=str(config.svd_selection),
        )
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


def summarize_projection_stats(projection_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary of one stage's gradient projection, for the run's stage report.

    The full per-module `projection_stats` only lives in the slice_cache; this
    distills it to a handful of scalars that ride along next to `ba_norms` in the
    results folder, so the projection record survives cache pruning.

    Handles both stats formats:
      - OGD-style `project_current_gradients`: global has dot/denom/gamma.
      - advanced `project_gradients_advanced`: global has cos/do_project/gamma.

    `rel_change` is the global relative change ||g_proj - g_c|| / ||g_c||. For the
    single-global-gamma path (g_proj = g_c + gamma*g_r) this is exact:
    |gamma| * sqrt(sum ||g_r||^2) / sqrt(sum ||g_c||^2).
    """
    if not isinstance(projection_stats, dict):
        return {"applied": False, "reason": "no_stats", "fired": False}

    g = projection_stats.get("global") or {}
    mods = projection_stats.get("modules") or {}
    gamma = g.get("gamma")

    if "do_project" in g:
        fired = bool(g.get("do_project"))
    elif gamma is not None:
        fired = float(gamma) > 0.0
    else:
        fired = False

    sum_cur2 = 0.0
    sum_ret2 = 0.0
    n_conflict = 0
    n_total = 0
    for m in mods.values():
        if not isinstance(m, dict):
            continue
        n_total += 1
        cn = m.get("current_norm")
        rn = m.get("retain_norm")
        if cn:
            sum_cur2 += float(cn) ** 2
        if rn:
            sum_ret2 += float(rn) ** 2
        d = m.get("dot")
        if d is not None and float(d) < 0.0:
            n_conflict += 1

    rel_change = None
    if gamma is not None and sum_cur2 > 0.0:
        rel_change = abs(float(gamma)) * math.sqrt(sum_ret2) / math.sqrt(sum_cur2)

    return {
        "applied": bool(projection_stats.get("applied", False)),
        "reason": projection_stats.get("reason"),
        "method": projection_stats.get("method"),
        "mode": projection_stats.get("mode"),
        "fired": fired,
        "gamma": None if gamma is None else float(gamma),
        "gamma_raw": g.get("gamma_raw"),
        "cos": g.get("cos"),
        "dot": g.get("dot", g.get("sum_dot")),
        # Significance-gate readout (basic pcgrad global mode with per-batch
        # dot sampling; None on legacy stats).
        "dot_mean": g.get("dot_mean"),
        "dot_se": g.get("dot_se"),
        "n_dot_samples": g.get("n_dot_samples"),
        "significant": g.get("significant"),
        "gated_by_significance": g.get("gated_by_significance"),
        "significance_k": projection_stats.get("significance_k"),
        "gamma_rel_cap": projection_stats.get("gamma_rel_cap"),
        "rel_change": rel_change,
        "n_modules_conflict": n_conflict,
        "n_modules_total": n_total,
    }


def _load_projection_summary(cache_root: str) -> Optional[Dict[str, Any]]:
    """Summarize a stage's projection_stats.json from the cache dir.

    Used on a cache hit, where the full stats are not recomputed in memory but the
    JSON is present in the cache entry. Returns None if the file is absent.
    """
    path = os.path.join(cache_root, "projection_stats.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return summarize_projection_stats(json.load(f))
    except (OSError, ValueError):
        return None


def load_or_compute_slice_inits(
    model: torch.nn.Module,
    tokenizer,
    current_task,
    retain_tasks,
    *,
    config: SliceInitConfig,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], str, Optional[Dict[str, Any]]]:
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
        "current_task": _task_fingerprint(current_task),
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
        "grad_project_always": False if is_lora_ga else config.grad_project_always,
        "add_retain_grad": False if is_lora_ga else config.add_retain_grad,
        "retain_batch_size": None if is_lora_ga else config.retain_batch_size,
        "retain_grad_accum": None if is_lora_ga else config.retain_grad_accum,
        "retain_batch_size_set": "all_tasks" if is_lora_ga else config.retain_batch_size_set,
        "single_retain_task_mode": False if is_lora_ga else config.single_retain_task_mode,
        "projection_method": "pcgrad" if is_lora_ga else str(config.projection_method),
        "cosine_threshold": None if is_lora_ga else config.cosine_threshold,
        "per_layer_threshold": False if is_lora_ga else bool(config.per_layer_threshold),
        "per_layer_threshold_delta": 0.0 if is_lora_ga else float(config.per_layer_threshold_delta),
        "pcgrad_c": 0.0 if is_lora_ga else float(config.pcgrad_c),
        "gradvac_phi": 0.0 if is_lora_ga else float(config.gradvac_phi),
        "gradvac_beta": 0.0 if is_lora_ga else float(config.gradvac_beta),
        "magnitude_preserve": False if is_lora_ga else bool(config.magnitude_preserve),
        "nullspace_rank": 0 if is_lora_ga else int(config.nullspace_rank),
        "nullspace_sv_threshold": 0.0 if is_lora_ga else float(config.nullspace_sv_threshold),
        "svd_selection": str(config.svd_selection),
        # The significance gate and gamma cap change the projected gradient, so
        # they must key the cache. Injected only when set, so legacy entries
        # stay reachable under their existing keys. (Both are rejected for
        # lora_ga by the guard above, so no is_lora_ga canonicalization needed.)
        **(
            {"dot_significance_k": float(config.dot_significance_k)}
            if float(config.dot_significance_k) != 0.0
            else {}
        ),
        **(
            {"gamma_rel_cap": float(config.gamma_rel_cap)}
            if config.gamma_rel_cap is not None
            else {}
        ),
        # Completion-only masking changes the loss the probe differentiates, so it
        # must key the cache. Injected only when enabled, so inits already cached
        # under the legacy full-sequence objective stay reachable in legacy mode.
        **({"completion_only": True} if config.completion_only else {}),
        # A pinned split changes which examples the probe may differentiate, so
        # it must key the cache: without this a pinned-split run would silently
        # reuse an init computed on the legacy partition -- i.e. on gradients
        # that saw the run's own eval questions. Injected only when pinned, so
        # legacy-mode entries stay reachable under their existing keys.
        **(
            {"split_seed": int(config.split_seed), "probe_eval_size": int(config.eval_size)}
            if config.split_seed is not None
            else {}
        ),
        "lora": lora_payload,
        "model": {
            "class": model.__class__.__name__,
            # Bump when the prompt/target rendering changes so cached gradients
            # (which are computed from the rendered text but not hashed on it)
            # are invalidated. v2 = native chat template + real EOS termination.
            "prompt_format": "chat_template_v2",
        },
    }
    cache_key = make_cache_key(payload)
    cache_root = os.path.join(config.cache_dir, cache_key)
    cached = load_slice_cache(config.cache_dir, cache_key, device=model_device(model))
    if cached is not None:
        save_ab_stats_csv(config.cache_dir, cache_key, cached.inits)
        logger.info("Slice cache hit: cache_dir=%s cache_key=%s modules=%d", config.cache_dir, cache_key, len(cached.inits))
        return cached.inits, cache_root, _load_projection_summary(cache_root)
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
            current_task=current_task,
            retain_tasks=retain_tasks,
            config=config,
        )

    # Inits live only in memory: by design we do not persist <cache>/<key>/inits/*.pt.
    # The meta + stats files stay so the run record is preserved.
    os.makedirs(cache_root, exist_ok=True)
    with open(os.path.join(cache_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"payload": payload}, f, sort_keys=True, indent=2)
    save_ab_stats_csv(config.cache_dir, cache_key, inits)
    save_projection_stats_json(config.cache_dir, cache_key, projection_stats)
    logger.info("Computed slice inits (not persisted): cache_dir=%s cache_key=%s modules=%d", config.cache_dir, cache_key, len(inits))
    return inits, cache_root, summarize_projection_stats(projection_stats)
