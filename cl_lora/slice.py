from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

try:
    from .lora_config import build_lora_config
    from .load_dataset import load_training_dataset
    from .slice_cache import SliceCacheEntry, load_slice_cache, make_cache_key, save_slice_cache
except ImportError:
    from lora_config import build_lora_config
    from load_dataset import load_training_dataset
    from slice_cache import SliceCacheEntry, load_slice_cache, make_cache_key, save_slice_cache


@dataclass
class SliceInitConfig:
    cache_dir: str = "slice_cache"
    max_steps: int = 100
    per_device_batch_size: int = 4
    seed: int = 42
    retain_scale: float = 1.0
    rank: Optional[int] = None
    max_seq_length: int = 256


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
    G32 = G.float()
    q = min(4 * r, min(G32.shape))
    if q <= 0:
        raise ValueError("Invalid rank for slice initialization")

    U, S, V = torch.svd_lowrank(G32, q=q, niter=4)
    U_r = U[:, :r]
    S_r = S[:r]
    V_r_rows = V[:, :r].t()

    D = torch.diag(S_r.sqrt())
    B = U_r @ D
    A = D @ V_r_rows
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


def compute_slice_inits(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_task=None,
    *,
    config: SliceInitConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    lora_cfg = build_lora_config()
    target_params = _target_weight_params(model, lora_cfg.target_modules)
    if not target_params:
        raise RuntimeError("No target modules matched for slice initialization.")

    forget_ds, _ = load_training_dataset(task=forget_task, eval_size=1, seed=config.seed)
    forget_ds = _tokenize_dataset(forget_ds, tokenizer=tokenizer, max_length=config.max_seq_length)
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

    grads_r = None
    steps_r = 0
    if retain_task is not None:
        retain_ds, _ = load_training_dataset(task=retain_task, eval_size=1, seed=config.seed)
        retain_ds = _tokenize_dataset(
            retain_ds,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )
        retain_loader = _build_dataloader(
            retain_ds,
            tokenizer=tokenizer,
            batch_size=config.per_device_batch_size,
            seed=config.seed,
        )
        grads_r, steps_r = _accumulate_gradients(
            model=model,
            dataloader=retain_loader,
            target_params=target_params,
            device=device,
            max_steps=config.max_steps,
        )

    denom_f = max(1, steps_f)
    grads_f = {k: v / float(denom_f) for k, v in grads_f.items()}
    if grads_r is not None:
        denom_r = max(1, steps_r)
        grads_r = {k: v / float(denom_r) for k, v in grads_r.items()}

    combined = _combine_grads(grads_f, grads_r, config.retain_scale)

    r_use = config.rank or int(getattr(lora_cfg, "r", 8))
    inits = {name: _build_ab_from_gradient(g, r=r_use) for name, g in combined.items()}
    return inits


def load_or_compute_slice_inits(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_task,
    *,
    config: SliceInitConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    payload = {
        "forget_task": getattr(forget_task, "name", str(forget_task)),
        "retain_task": getattr(retain_task, "name", str(retain_task)) if retain_task else None,
        "rank": config.rank,
        "max_steps": config.max_steps,
        "batch_size": config.per_device_batch_size,
        "retain_scale": config.retain_scale,
    }
    cache_key = make_cache_key(payload)
    cached = load_slice_cache(config.cache_dir, cache_key, device=_model_device(model))
    if cached is not None:
        return cached.inits

    inits = compute_slice_inits(
        model=model,
        tokenizer=tokenizer,
        forget_task=forget_task,
        retain_task=retain_task,
        config=config,
    )

    save_slice_cache(
        config.cache_dir,
        cache_key,
        SliceCacheEntry(inits=inits),
        meta={"payload": payload},
    )
    return inits


def apply_slice_inits(
    peft_model: torch.nn.Module,
    inits: Dict[str, Dict[str, torch.Tensor]],
) -> int:
    from peft.tuners.lora import Linear as LoraLinear

    named_modules = dict(peft_model.named_modules())
    num_written = 0

    for module_name, ab in inits.items():
        target_module = None
        for name, mod in named_modules.items():
            if isinstance(mod, LoraLinear) and module_name in name:
                target_module = mod
                break
        if target_module is None:
            continue

        A_tgt = target_module.lora_A["default"].weight
        B_tgt = target_module.lora_B["default"].weight

        if A_tgt.shape != ab["A"].shape or B_tgt.shape != ab["B"].shape:
            raise RuntimeError(
                f"Slice init shape mismatch for {module_name}: "
                f"A_tgt={A_tgt.shape}, A_init={ab['A'].shape}, "
                f"B_tgt={B_tgt.shape}, B_init={ab['B'].shape}"
            )

        with torch.no_grad():
            A_tgt.copy_(ab["A"].to(device=A_tgt.device, dtype=A_tgt.dtype))
            B_tgt.copy_(ab["B"].to(device=B_tgt.device, dtype=B_tgt.dtype))
        num_written += 1

    return num_written


def initialize_lora_with_slice(
    model: torch.nn.Module,
    tokenizer,
    forget_task,
    retain_task,
    *,
    config: SliceInitConfig,
) -> int:
    inits = load_or_compute_slice_inits(
        model=model,
        tokenizer=tokenizer,
        forget_task=forget_task,
        retain_task=retain_task,
        config=config,
    )
    return apply_slice_inits(model, inits)
