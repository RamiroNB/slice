from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from transformers import DataCollatorForLanguageModeling
import logging

try:
    from .lora_config import build_lora_config
    from .load_dataset import load_training_dataset
    from .repro import set_global_seed
    from .slice_cache import SliceCacheEntry, load_slice_cache, make_cache_key, save_slice_cache
except ImportError:
    from lora_config import build_lora_config
    from load_dataset import load_training_dataset
    from repro import set_global_seed
    from slice_cache import SliceCacheEntry, load_slice_cache, make_cache_key, save_slice_cache


@dataclass
class SliceInitConfig:
    cache_dir: str = "slice_cache"
    cache_context: Optional[str] = None
    max_steps: int = 100
    per_device_batch_size: int = 64
    seed: int = 42
    retain_scale: float = 1.0
    grad_project: bool = False
    grad_projection_mode: str = "per_module"
    add_retain_grad: bool = False
    rank: Optional[int] = None
    max_seq_length: int = 256
    retain_batch_size: Optional[int] = None
    retain_grad_accum: Optional[int] = None
    retain_batch_size_set: str = "all_tasks"


logger = logging.getLogger("cl_lora.slice")


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    return dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length),
        remove_columns=dataset.column_names,
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _build_dataloader(dataset, tokenizer, batch_size: int, seed: int) -> DataLoader:
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )


def _target_weight_params(model: torch.nn.Module, target_modules: Iterable[str]) -> Dict[str, torch.nn.Parameter]:
    target_modules = list(target_modules)
    out: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        if not name.endswith(".weight"):
            continue
        if not any(tgt in name for tgt in target_modules):
            continue
        module_name = name[: -len(".weight")]
        out[module_name] = param
    return out


def _accumulate_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_params: Dict[str, torch.nn.Parameter],
    device: torch.device,
    max_steps: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    grads: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device) for name, param in target_params.items()
    }
    steps = 0
    model.train()
    for batch in dataloader:
        if max_steps and steps >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for name, param in target_params.items():
            if param.grad is None:
                continue
            grads[name] = grads[name] + param.grad.detach()
        model.zero_grad(set_to_none=True)
        steps += 1
    return grads, steps


def _build_ab_from_gradient(G: torch.Tensor, r: int) -> Dict[str, torch.Tensor]:
    device = G.device
    d_out, d_in = G.shape
    G32 = G.float()
    q = min(4 * r, min(G32.shape))
    if q <= 0:
        raise ValueError("Invalid rank for slice initialization")

    U, _, V = torch.svd_lowrank(G32, q=q, niter=4)

    Vt = V.t()
    B = U[:, :r]
    A = Vt[r : 2 * r, :]

    # recon = B @ A
    # eps = 1e-12
    # var_recon = float(torch.var(recon).item()) if torch.var(recon).item() != 0.0 else eps
    # factor = (1.0 / (3.0 * d_in) ** 0.25) / (var_recon ** 0.25)
    # A = factor * A
    # B = factor * B

    scale_ga = (d_out ** 0.25) / (64 ** 0.5)
    B = B * scale_ga
    A = A * scale_ga

    return {
        "A": A.to(device=device, dtype=G.dtype).contiguous(),
        "B": B.to(device=device, dtype=G.dtype).contiguous(),
    }


def _combine_grads(
    grads_forget: Dict[str, torch.Tensor],
    grads_retain: Optional[Dict[str, torch.Tensor]],
    retain_scale: float,
) -> Dict[str, torch.Tensor]:
    combined: Dict[str, torch.Tensor] = {}
    for name, g_f in grads_forget.items():
        g_r = grads_retain.get(name) if grads_retain is not None else None
        if g_r is None:
            combined[name] = g_f
        else:
            combined[name] = g_f - retain_scale * g_r
    return combined


def _project_forget_gradients(
    grads_forget: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    *,
    global_projection: bool = False,
    add_retain_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """Project forget gradients against retain gradients (LInMU-style)."""
    projected: Dict[str, torch.Tensor] = {}
    eps = 1e-12

    if not grads_forget:
        return projected

    if not global_projection:
        for name, g_f in grads_forget.items():
            g_r = grads_retain.get(name)
            if g_r is None:
                projected[name] = g_f
                continue

            original_shape = g_f.shape
            g_f_flat = g_f.float().view(-1).to(torch.float64)
            g_r_flat = g_r.float().view(-1).to(torch.float64)

            dot = torch.dot(g_f_flat, g_r_flat)
            denom = torch.dot(g_r_flat, g_r_flat)
            dot_clipped = torch.relu(-dot)
            gamma = dot_clipped / (denom + eps)

            g_f_new = (g_f_flat + gamma * g_r_flat).view(original_shape)
            if add_retain_grad:
                g_f_new = g_f_new + g_r.to(g_f_new.device)
            projected[name] = g_f_new.to(g_f.dtype)
    else:
        first_name = next(iter(grads_forget.keys()))
        device = grads_forget[first_name].device
        global_dot = torch.tensor(0.0, device=device)
        global_denom = torch.tensor(0.0, device=device)

        for name, g_f in grads_forget.items():
            g_r = grads_retain.get(name)
            if g_r is None:
                continue
            g_f_flat = g_f.float().view(-1).to(torch.float64)
            g_r_flat = g_r.float().view(-1).to(torch.float64)
            global_dot = global_dot + torch.dot(g_f_flat, g_r_flat)
            global_denom = global_denom + torch.dot(g_r_flat, g_r_flat)

        dot_clipped = torch.relu(-global_dot)
        gamma = dot_clipped / (global_denom + eps)

        for name, g_f in grads_forget.items():
            g_r = grads_retain.get(name)
            if g_r is None:
                projected[name] = g_f
                continue

            original_shape = g_f.shape
            g_f_flat = g_f.float().view(-1).to(torch.float64)
            g_r_flat = g_r.float().view(-1).to(torch.float64)
            g_f_new = (g_f_flat + gamma * g_r_flat).view(original_shape)
            if add_retain_grad:
                g_f_new = g_f_new + g_r.to(g_f_new.device)
            projected[name] = g_f_new.to(g_f.dtype)

    return projected


def compute_slice_inits(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_tasks=None,
    *,
    config: SliceInitConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    retain_tasks = retain_tasks or []
    retain_names = [getattr(rt, "name", str(rt)) for rt in retain_tasks] or None
    logger.info(
        "Starting slice init: forget=%s retain=%s max_steps=%s batch_size=%s",
        getattr(forget_task, "name", str(forget_task)),
        retain_names,
        config.max_steps,
        config.per_device_batch_size,
    )
    lora_cfg = build_lora_config()
    target_params = _target_weight_params(model, lora_cfg.target_modules)
    if not target_params:
        logger.error("No target modules matched for slice initialization.")
        raise RuntimeError("No target modules matched for slice initialization.")

    logger.info("Matched %d target weight parameters for slice init", len(target_params))
    forget_ds, _ = load_training_dataset(task=forget_task, eval_size=1, seed=config.seed)
    forget_ds = _tokenize_dataset(forget_ds, tokenizer=tokenizer, max_length=config.max_seq_length)
    logger.info("Building forget dataloader: dataset_size=%d batch_size=%d", len(forget_ds), config.per_device_batch_size)
    forget_loader = _build_dataloader(
        forget_ds,
        tokenizer=tokenizer,
        batch_size=config.per_device_batch_size,
        seed=config.seed,
    )

    device = _model_device(model)
    grads_f, steps_f = _accumulate_gradients(
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
                ds = _tokenize_dataset(ds, tokenizer=tokenizer, max_length=config.max_seq_length)
                all_retain_ds.append(ds)
            combined_ds = concatenate_datasets(all_retain_ds)
            logger.info("Retain dataloader (all_tasks): %d total samples, batch_size=%d", len(combined_ds), retain_bs)
            retain_loader = _build_dataloader(combined_ds, tokenizer=tokenizer, batch_size=retain_bs, seed=config.seed)
            grads_r, steps_r = _accumulate_gradients(
                model=model, dataloader=retain_loader, target_params=target_params,
                device=device, max_steps=retain_max_steps,
            )
        elif config.retain_batch_size_set == "each_task":
            grads_r = {name: torch.zeros_like(param, device=device) for name, param in target_params.items()}
            steps_r = 0
            for rt in retain_tasks:
                rt_name = getattr(rt, "name", str(rt))
                ds, _ = load_training_dataset(task=rt, eval_size=1, seed=config.seed)
                ds = _tokenize_dataset(ds, tokenizer=tokenizer, max_length=config.max_seq_length)
                logger.info("Retain dataloader (each_task): task=%s, %d samples, batch_size=%d", rt_name, len(ds), retain_bs)
                rt_loader = _build_dataloader(ds, tokenizer=tokenizer, batch_size=retain_bs, seed=config.seed)
                grads_rt, steps_rt = _accumulate_gradients(
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
        combined = _project_forget_gradients(
            grads_forget=grads_f,
            grads_retain=grads_r,
            global_projection=global_projection,
            add_retain_grad=config.add_retain_grad,
        )
        logger.info("Built projected gradient matrix for %d modules", len(combined))
    elif config.grad_project and grads_r is None:
        logger.info("grad_project=True but no retain task provided; using forget gradients without projection")
        combined = grads_f
    else:
        combined = _combine_grads(grads_f, grads_r, config.retain_scale)
        logger.info("Built combined gradient matrix for %d modules (retain_scale=%s)", len(combined), config.retain_scale)

    r_use = config.rank or int(getattr(lora_cfg, "r", 8))
    inits = {}
    for name, g in combined.items():
        logger.info("Building A/B for module %s: G_shape=%s r=%d", name, tuple(g.shape), r_use)
        ab = _build_ab_from_gradient(g, r=r_use)
        logger.debug("Built A/B for %s: A_shape=%s B_shape=%s", name, tuple(ab['A'].shape), tuple(ab['B'].shape))
        inits[name] = ab
    return inits


def load_or_compute_slice_inits(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_tasks,
    *,
    config: SliceInitConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    def _task_fingerprint(task_obj) -> Optional[Dict[str, object]]:
        if task_obj is None:
            return None
        fp: Dict[str, object] = {
            "type": task_obj.__class__.__name__,
            "name": getattr(task_obj, "name", str(task_obj)),
        }
        # SuperNI tasks
        for k in ("ni_id", "hf_config", "source", "category"):
            if hasattr(task_obj, k):
                fp[k] = getattr(task_obj, k)
        # TRACE tasks
        for k in ("hf_dataset", "language", "metric"):
            if hasattr(task_obj, k):
                fp[k] = getattr(task_obj, k)
        return fp

    # Include LoRA settings used to define which weights slice targets.
    # (If these change, cached inits must not be reused.)
    lora_cfg = build_lora_config(r=int(config.rank or 128))
    lora_payload = {
        "r": int(getattr(lora_cfg, "r", 0) or 0),
        "lora_alpha": float(getattr(lora_cfg, "lora_alpha", 1.0)),
        "lora_dropout": float(getattr(lora_cfg, "lora_dropout", 0.0)),
        "bias": str(getattr(lora_cfg, "bias", "none")),
        "use_rslora": bool(getattr(lora_cfg, "use_rslora", False)) if hasattr(lora_cfg, "use_rslora") else None,
        "target_modules": list(getattr(lora_cfg, "target_modules", []) or []),
    }

    payload = {
        "cache_context": config.cache_context,
        "forget_task": _task_fingerprint(forget_task),
        "retain_tasks": [_task_fingerprint(rt) for rt in (retain_tasks or [])] or None,
        "rank": config.rank,
        "seed": config.seed,
        "max_seq_length": config.max_seq_length,
        "max_steps": config.max_steps,
        "batch_size": config.per_device_batch_size,
        "retain_scale": config.retain_scale,
        "grad_project": config.grad_project,
        "grad_projection_mode": config.grad_projection_mode,
        "add_retain_grad": config.add_retain_grad,
        "retain_batch_size": config.retain_batch_size,
        "retain_grad_accum": config.retain_grad_accum,
        "retain_batch_size_set": config.retain_batch_size_set,
        "lora": lora_payload,
        "model": {
            "class": model.__class__.__name__,
        },
    }
    cache_key = make_cache_key(payload)
    cached = load_slice_cache(config.cache_dir, cache_key, device=_model_device(model))
    if cached is not None:
        logger.info("Slice cache hit: cache_dir=%s cache_key=%s modules=%d", config.cache_dir, cache_key, len(cached.inits))
        return cached.inits
    logger.info("Slice cache miss: will compute inits (cache_dir=%s cache_key=%s)", config.cache_dir, cache_key)

    inits = compute_slice_inits(
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
    logger.info("Saved slice cache: cache_dir=%s cache_key=%s modules=%d", config.cache_dir, cache_key, len(inits))
    return inits


def apply_slice_inits(
    peft_model: torch.nn.Module,
    inits: Dict[str, Dict[str, torch.Tensor]],
    *,
    lora_alpha: float = 1.0,
    r: Optional[int] = None,
    decomposition: Optional[str] = None,
) -> int:
    """Apply slice inits to a PEFT LoRA model with in-place absorption."""
    from peft.tuners.lora import Linear as LoraLinear

    def _tensor_mean_var(t: torch.Tensor) -> tuple[float, float]:
        t32 = t.detach().to(dtype=torch.float32)
        mean_val = float(t32.mean().item())
        var_val = float(t32.var(unbiased=False).item())
        return mean_val, var_val

    def _normalize_name(name: str) -> str:
        out = str(name)
        out = out.replace(".weight", "")
        out = out.replace(".base_layer", "")
        return out

    def _resolve_target_module(init_key: str, index: Dict[str, str]) -> Optional[str]:
        nk = _normalize_name(init_key)
        exact = index.get(nk)
        if exact is not None:
            return exact

        # Robust fallback when one side has extra prefixes.
        candidates = [
            real_name
            for norm_name, real_name in index.items()
            if norm_name.endswith(nk) or nk.endswith(norm_name)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    skip_absorption = decomposition in {
        "right_singular_vectors",
        "right_singular_vectors_kaiming",
        "right_svd_kaiming_random_basis",
    }

    if r is None or int(r) <= 0:
        raise RuntimeError("slice apply requires a valid LoRA rank `r` for absorption.")

    named_modules = dict(peft_model.named_modules())
    logger.info("Applying slice inits with in-place absorption: candidate_modules=%d", len(inits))

    lora_index: Dict[str, str] = {}
    lora_names: list[str] = []
    for module_name, mod in named_modules.items():
        if not isinstance(mod, LoraLinear):
            continue
        normalized = _normalize_name(module_name)
        lora_index[normalized] = module_name
        lora_names.append(module_name)

    sample_init_keys = list(inits.keys())[:5]
    sample_lora_names = lora_names[:5]
    logger.debug("Slice sample init keys: %s", sample_init_keys)
    logger.debug("Slice sample LoRA modules: %s", sample_lora_names)

    num_written = 0
    num_skipped = 0
    for init_key, ab in inits.items():
        target_name = _resolve_target_module(init_key, lora_index)
        if target_name is None:
            logger.debug("No PEFT LoRA module matched for init key %s; skipping", init_key)
            num_skipped += 1
            continue

        logger.debug("slice map: init_key=%s -> peft_module=%s", init_key, target_name)
        module = named_modules[target_name]
        A_tgt = module.lora_A["default"].weight
        B_tgt = module.lora_B["default"].weight

        if A_tgt.shape != ab["A"].shape or B_tgt.shape != ab["B"].shape:
            raise RuntimeError(
                f"Slice init shape mismatch for layer {init_key}: "
                f"A_tgt.shape={A_tgt.shape}, A_init.shape={ab['A'].shape}, "
                f"B_tgt.shape={B_tgt.shape}, B_init.shape={ab['B'].shape}"
            )

        with torch.no_grad():
            A_tgt.copy_(ab["A"].to(device=A_tgt.device, dtype=A_tgt.dtype))
            B_tgt.copy_(ab["B"].to(device=B_tgt.device, dtype=B_tgt.dtype))

            if logger.isEnabledFor(logging.DEBUG):
                a_mean, a_var = _tensor_mean_var(A_tgt)
                b_mean, b_var = _tensor_mean_var(B_tgt)
                logger.debug(
                    "[slice] final layer stats: layer=%s A(mean=%.6g,var=%.6g,shape=%s,dtype=%s) "
                    "B(mean=%.6g,var=%.6g,shape=%s,dtype=%s)",
                    init_key,
                    a_mean,
                    a_var,
                    tuple(A_tgt.shape),
                    str(A_tgt.dtype).replace("torch.", ""),
                    b_mean,
                    b_var,
                    tuple(B_tgt.shape),
                    str(B_tgt.dtype).replace("torch.", ""),
                )

            if not skip_absorption:
                base_layer = None
                if hasattr(module, "get_base_layer"):
                    try:
                        base_layer = module.get_base_layer()
                    except Exception as exc:
                        raise RuntimeError(f"Failed to get base layer for LoRA module {init_key}") from exc

                base_weight = getattr(base_layer, "weight", None) if base_layer is not None else None
                if isinstance(base_weight, torch.nn.Parameter):
                    scaling_val = float(lora_alpha) / float(r)
                    orig_dtype = base_weight.dtype

                    weight_orig32 = base_weight.data.to(torch.float32).clone()
                    B_mat = B_tgt.to(dtype=torch.float32)
                    A_mat = A_tgt.to(dtype=torch.float32)
                    offset32 = (B_mat @ A_mat) * float(scaling_val)

                    weight32 = weight_orig32 - offset32
                    base_weight.data.copy_(weight32.to(orig_dtype))

                    recon32 = weight32 + offset32
                    diff = (recon32 - weight_orig32).abs().max()
                    tol = 1e-4 if orig_dtype in (torch.float16, torch.bfloat16) else 1e-7
                    if diff > tol:
                        logger.warning(
                            "[slice] LoRA absorption diff for layer %s: max|W_frozen+DeltaW-W_orig|=%.3e (tol=%.3e)",
                            init_key,
                            diff.item(),
                            tol,
                        )
                    else:
                        logger.info(
                            "[slice] LoRA absorption OK for layer %s: max|W_frozen+DeltaW-W_orig|=%.3e (tol=%.3e)",
                            init_key,
                            diff.item(),
                            tol,
                        )
            else:
                logger.info(
                    "[slice] Skipping LoRA absorption for layer %s due to decomposition='%s'",
                    init_key,
                    decomposition,
                )

        num_written += 1

    logger.info("Applied slice A/B to %d LoRA modules (skipped=%d)", num_written, num_skipped)
    return num_written


def initialize_lora_with_slice(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_tasks,
    *,
    config: SliceInitConfig,
) -> int:
    set_global_seed(int(config.seed))
    inits = load_or_compute_slice_inits(
        model=model,
        tokenizer=tokenizer,
        forget_task=forget_task,
        retain_tasks=retain_tasks,
        config=config,
    )

    # Attempt to infer LoRA alpha and rank from the PEFT model if present
    lora_alpha = getattr(config, "lora_alpha", 1.0)
    r_val = getattr(config, "rank", None)
    # Delegate to apply_slice_inits which will try absorption when possible
    return apply_slice_inits(model, inits, lora_alpha=lora_alpha, r=r_val)
