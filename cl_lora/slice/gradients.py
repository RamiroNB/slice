from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("cl_lora.slice.gradients")


def accumulate_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_params: Dict[str, torch.nn.Parameter],
    device: torch.device,
    max_steps: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    grads: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device) for name, param in target_params.items()
    }

    # PEFT freezes base weights (requires_grad=False) so backward would
    # skip them and .grad would stay None.  Temporarily re-enable so we
    # can collect gradients, then restore the original state.
    saved_requires_grad = {name: p.requires_grad for name, p in target_params.items()}
    for p in target_params.values():
        p.requires_grad_(True)

    # Enable gradient checkpointing to reduce activation memory,
    # allowing larger batch sizes during gradient accumulation.
    _had_gc = getattr(model, "is_gradient_checkpointing", False)
    _use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if not _had_gc and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            model.gradient_checkpointing_enable()

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
                raise RuntimeError(
                    f"param.grad is None for {name} despite requires_grad=True. "
                    "This should not happen -- check model wiring."
                )
            grads[name] = grads[name] + param.grad.detach()
        model.zero_grad(set_to_none=True)
        steps += 1

    # Restore gradient checkpointing and use_cache state.
    if not _had_gc and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if _use_cache is not None and hasattr(model, "config"):
        model.config.use_cache = _use_cache

    for name, p in target_params.items():
        p.requires_grad_(saved_requires_grad[name])

    return grads, steps


def combine_grads(
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


def project_forget_gradients(
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
                g_f_new = g_f_new + g_r.to(device=g_f_new.device, dtype=g_f_new.dtype)
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
                g_f_new = g_f_new + g_r.to(device=g_f_new.device, dtype=g_f_new.dtype)
            projected[name] = g_f_new.to(g_f.dtype)

    return projected
