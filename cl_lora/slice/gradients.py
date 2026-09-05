from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("cl_lora.slice.gradients")


def accumulate_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_params: Dict[str, torch.nn.Parameter],
    device: torch.device,
    max_steps: int,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    dot_reference: Optional[Dict[str, torch.Tensor]] = None,
    dot_samples: Optional[List[float]] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    # When `grads` is provided, accumulate into it in place instead of allocating a
    # fresh full-size buffer. This lets a caller sum gradients across several
    # dataloaders (e.g. one retain task at a time) without ever holding more than
    # one retain-gradient set resident -- identical result to summing per-task
    # buffers, but one full gradient set (~model-sized) lighter at peak.
    #
    # When `dot_reference` is provided (a fixed gradient set, e.g. the already-
    # accumulated current-task gradients), each batch also appends
    # sum_modules <reference, batch_grad> to `dot_samples`. The caller gets one
    # scalar per batch, i.e. an i.i.d. sample of the conflict dot whose mean
    # (after the caller's normalization) equals the global dot — which is what
    # turns the single point-estimate gate into a significance test.
    if grads is None:
        grads = {
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
        batch_dot = 0.0
        for name, param in target_params.items():
            if param.grad is None:
                raise RuntimeError(
                    f"param.grad is None for {name} despite requires_grad=True. "
                    "This should not happen -- check model wiring."
                )
            g = param.grad.detach()
            grads[name] = grads[name] + g
            if dot_reference is not None and dot_samples is not None:
                ref = dot_reference.get(name)
                if ref is not None:
                    batch_dot += float(
                        torch.sum(ref.float() * g.float()).item()
                    )
        if dot_reference is not None and dot_samples is not None:
            dot_samples.append(batch_dot)
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
    grads_current: Dict[str, torch.Tensor],
    grads_retain: Optional[Dict[str, torch.Tensor]],
    retain_scale: float,
) -> Dict[str, torch.Tensor]:
    combined: Dict[str, torch.Tensor] = {}
    for name, g_c in grads_current.items():
        g_r = grads_retain.get(name) if grads_retain is not None else None
        if g_r is None:
            combined[name] = g_c
        else:
            combined[name] = g_c - retain_scale * g_r
    return combined


def project_current_gradients(
    grads_current: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    *,
    global_projection: bool = False,
    always_project: bool = False,
    add_retain_grad: bool = False,
    return_stats: bool = False,
    dot_samples: Optional[Sequence[float]] = None,
    significance_k: float = 0.0,
    gamma_rel_cap: Optional[float] = None,
) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Project current-task gradients against retain gradients (LInMU-style).

    `dot_samples` are per-retain-batch estimates of the global dot (already
    scaled so their mean equals the global dot on the normalized gradients).
    With `significance_k` > 0, the global-mode projection fires only when the
    dot is negative AND its magnitude exceeds k standard errors — the sign of a
    single noisy point estimate stops being the trigger.

    `gamma_rel_cap` bounds the projection strength: |gamma|*||g_r|| is clipped
    to at most cap*||g_c|| (globally, or per module in per-module mode).
    """
    projected: Dict[str, torch.Tensor] = {}
    eps = 1e-12
    stats: Dict[str, Any] = {
        "mode": "global" if global_projection else "per_module",
        "always_project": bool(always_project),
        "add_retain_grad": bool(add_retain_grad),
        "eps": float(eps),
        "significance_k": float(significance_k),
        "gamma_rel_cap": None if gamma_rel_cap is None else float(gamma_rel_cap),
        "modules": {},
    }

    if not grads_current:
        if return_stats:
            stats["status"] = "empty_current_grads"
            return projected, stats
        return projected

    if not global_projection:
        for name, g_c in grads_current.items():
            g_r = grads_retain.get(name)
            if g_r is None:
                projected[name] = g_c
                stats["modules"][name] = {
                    "status": "missing_retain_grad",
                }
                continue

            original_shape = g_c.shape
            g_c_flat = g_c.float().view(-1).to(torch.float64)
            g_r_flat = g_r.float().view(-1).to(torch.float64)

            dot = torch.dot(g_c_flat, g_r_flat)
            denom = torch.dot(g_r_flat, g_r_flat)
            # OGD-style: always remove the retain-direction component.
            projection_numerator = -dot if always_project else torch.relu(-dot)
            gamma = projection_numerator / (denom + eps)
            gamma_raw = float(gamma.item())
            if gamma_rel_cap is not None:
                cur_norm = g_c_flat.norm()
                ret_norm = g_r_flat.norm()
                gamma_max = float(gamma_rel_cap) * cur_norm / (ret_norm + eps)
                gamma = torch.clamp(gamma, min=-gamma_max, max=gamma_max)

            g_c_new = (g_c_flat + gamma * g_r_flat).view(original_shape)
            if add_retain_grad:
                g_c_new = g_c_new + g_r.to(device=g_c_new.device, dtype=g_c_new.dtype)
            projected[name] = g_c_new.to(g_c.dtype)
            stats["modules"][name] = {
                "status": "projected",
                "dot": float(dot.item()),
                "denom": float(denom.item()),
                "dot_clipped": float(projection_numerator.item()),
                "projection_numerator": float(projection_numerator.item()),
                "gamma": float(gamma.item()),
                "gamma_raw": gamma_raw,
                "current_norm": float(g_c_flat.norm().item()),
                "retain_norm": float(g_r_flat.norm().item()),
                "projected_norm": float(g_c_new.float().view(-1).norm().item()),
            }
    else:
        first_name = next(iter(grads_current.keys()))
        device = grads_current[first_name].device
        global_dot = torch.tensor(0.0, device=device)
        global_denom = torch.tensor(0.0, device=device)

        for name, g_c in grads_current.items():
            g_r = grads_retain.get(name)
            if g_r is None:
                continue
            g_c_flat = g_c.float().view(-1).to(torch.float64)
            g_r_flat = g_r.float().view(-1).to(torch.float64)
            global_dot = global_dot + torch.dot(g_c_flat, g_r_flat)
            global_denom = global_denom + torch.dot(g_r_flat, g_r_flat)

        # Significance gate: the global dot is a small difference of large sums
        # whose sign flips across probe draws. With per-batch samples available,
        # only treat the conflict as real when it clears k standard errors.
        dot_mean = None
        dot_se = None
        n_samples = 0
        significant = None
        gated_by_significance = False
        if dot_samples:
            n_samples = len(dot_samples)
            dot_mean = float(sum(dot_samples) / n_samples)
            if n_samples > 1:
                var = sum((s - dot_mean) ** 2 for s in dot_samples) / (n_samples - 1)
                dot_se = math.sqrt(max(var, 0.0) / n_samples)
        if (
            significance_k > 0.0
            and not always_project
            and dot_se is not None
        ):
            significant = bool(
                float(global_dot.item()) < 0.0
                and -float(global_dot.item()) > significance_k * dot_se
            )
            if not significant and float(global_dot.item()) < 0.0:
                gated_by_significance = True

        projection_numerator = -global_dot if always_project else torch.relu(-global_dot)
        gamma = projection_numerator / (global_denom + eps)
        if gated_by_significance or (significant is False):
            gamma = torch.zeros_like(gamma)
        gamma_raw = float(gamma.item())
        if gamma_rel_cap is not None:
            global_cur2 = torch.tensor(0.0, device=device, dtype=torch.float64)
            for name, g_c in grads_current.items():
                if grads_retain.get(name) is None:
                    continue
                g_c_flat = g_c.float().view(-1).to(torch.float64)
                global_cur2 = global_cur2 + torch.dot(g_c_flat, g_c_flat)
            gamma_max = (
                float(gamma_rel_cap)
                * torch.sqrt(global_cur2)
                / (torch.sqrt(global_denom) + eps)
            )
            gamma = torch.clamp(gamma, min=-float(gamma_max.item()), max=float(gamma_max.item()))
        stats["global"] = {
            "dot": float(global_dot.item()),
            "denom": float(global_denom.item()),
            "dot_clipped": float(projection_numerator.item()),
            "projection_numerator": float(projection_numerator.item()),
            "gamma": float(gamma.item()),
            "gamma_raw": gamma_raw,
            "dot_mean": dot_mean,
            "dot_se": dot_se,
            "n_dot_samples": int(n_samples),
            "dot_samples": [float(s) for s in (dot_samples or [])],
            "significant": significant,
            "gated_by_significance": bool(gated_by_significance),
        }

        for name, g_c in grads_current.items():
            g_r = grads_retain.get(name)
            if g_r is None:
                projected[name] = g_c
                stats["modules"][name] = {
                    "status": "missing_retain_grad",
                }
                continue

            original_shape = g_c.shape
            g_c_flat = g_c.float().view(-1).to(torch.float64)
            g_r_flat = g_r.float().view(-1).to(torch.float64)
            g_c_new = (g_c_flat + gamma * g_r_flat).view(original_shape)
            if add_retain_grad:
                g_c_new = g_c_new + g_r.to(device=g_c_new.device, dtype=g_c_new.dtype)
            projected[name] = g_c_new.to(g_c.dtype)
            stats["modules"][name] = {
                "status": "projected",
                "dot": float(torch.dot(g_c_flat, g_r_flat).item()),
                "denom": float(torch.dot(g_r_flat, g_r_flat).item()),
                "gamma": float(gamma.item()),
                "current_norm": float(g_c_flat.norm().item()),
                "retain_norm": float(g_r_flat.norm().item()),
                "projected_norm": float(g_c_new.float().view(-1).norm().item()),
            }

    if return_stats:
        return projected, stats
    return projected
