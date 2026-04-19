"""Advanced gradient-surgery projection methods.

Implements ideas A.1 (CAGrad), A.2 (GradVac), A.3 (cosine threshold),
A.4 (per-layer threshold), A.5 (null-space), A.6 (magnitude preserving)
from ideas_for_new_methods.md. Kept on the same device as input
gradients to avoid CPU offload of large tensors.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Tuple

import torch

logger = logging.getLogger("cl_lora.slice.projections")

_EPS = 1e-12


def _flat_f64(t: torch.Tensor) -> torch.Tensor:
    return t.float().view(-1).to(torch.float64)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm()
    nb = b.norm()
    if float(na.item()) < _EPS or float(nb.item()) < _EPS:
        return 0.0
    return float((torch.dot(a, b) / (na * nb)).item())


def _magnitude_preserve(g_new: torch.Tensor, g_orig_norm: float) -> torch.Tensor:
    n = float(g_new.float().norm().item())
    if n < _EPS:
        return g_new
    scale = g_orig_norm / n
    return g_new * scale


def _decide_cosine_thresholds(
    grads_forget: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    *,
    config_cos_tau,
    per_layer: bool,
    delta: float,
) -> Dict[str, float]:
    """Return per-module cosine threshold to compare against."""
    cos_map: Dict[str, float] = {}
    for name, g_f in grads_forget.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            continue
        cos_map[name] = _cosine(_flat_f64(g_f), _flat_f64(g_r))

    if per_layer and cos_map:
        vals = sorted(cos_map.values())
        median = vals[len(vals) // 2]
        tau = median - float(delta)
        logger.info("per_layer_threshold: median_cos=%.4f delta=%.4f tau=%.4f",
                    median, float(delta), tau)
        return {n: tau for n in cos_map}
    elif config_cos_tau is not None:
        tau = float(config_cos_tau)
        return {n: tau for n in cos_map}
    else:
        return {n: 0.0 for n in cos_map}  # dot-sign equivalent (cos threshold 0)


def _should_project(cos: float, tau: float, always: bool) -> bool:
    if always:
        return True
    return cos < tau


def _pcgrad_update(g_f: torch.Tensor, g_r: torch.Tensor) -> torch.Tensor:
    g_f_flat = _flat_f64(g_f)
    g_r_flat = _flat_f64(g_r)
    dot = torch.dot(g_f_flat, g_r_flat)
    denom = torch.dot(g_r_flat, g_r_flat) + _EPS
    gamma = -dot / denom
    return (g_f_flat + gamma * g_r_flat).view(g_f.shape)


def _cagrad_update(g_f: torch.Tensor, g_r: torch.Tensor, c: float) -> torch.Tensor:
    """Soft interpolation between vanilla (c=0) and PCGrad (c=1)."""
    g_f_flat = _flat_f64(g_f)
    g_r_flat = _flat_f64(g_r)
    dot = torch.dot(g_f_flat, g_r_flat)
    denom = torch.dot(g_r_flat, g_r_flat) + _EPS
    gamma_full = -dot / denom
    # Only apply damping to the correcting term; c=1 reproduces PCGrad.
    return (g_f_flat + float(c) * gamma_full * g_r_flat).view(g_f.shape)


def _gradvac_update(
    g_f: torch.Tensor, g_r: torch.Tensor, phi: float,
) -> Tuple[torch.Tensor, float]:
    """Rotate g_f so cos(new, g_r) = phi (closed form from GradVac)."""
    g_f_flat = _flat_f64(g_f)
    g_r_flat = _flat_f64(g_r)
    nf = float(g_f_flat.norm().item())
    nr = float(g_r_flat.norm().item())
    if nf < _EPS or nr < _EPS:
        return g_f, 0.0
    cos = float((torch.dot(g_f_flat, g_r_flat) / (nf * nr)).item())
    # If already aligned beyond phi, no change.
    if cos >= phi:
        return g_f, cos
    phi_c = max(min(phi, 0.999), -0.999)
    cos_c = max(min(cos, 0.999), -0.999)
    num = nf * (phi_c * math.sqrt(max(1.0 - cos_c * cos_c, 0.0))
                - cos_c * math.sqrt(max(1.0 - phi_c * phi_c, 0.0)))
    den = nr * math.sqrt(max(1.0 - phi_c * phi_c, 0.0)) + _EPS
    lam = num / den
    out = (g_f_flat + lam * g_r_flat).view(g_f.shape)
    return out, cos


def _nullspace_update(
    g_f: torch.Tensor, g_r: torch.Tensor, k: int, sv_thresh: float,
) -> torch.Tensor:
    """Project g_f out of the top-k left-singular subspace of g_r.

    Uses SVD of the retain matrix (not covariance) as a proxy for the
    preserve-feature subspace — Adam-NSCL style but gradient-based.
    Runs on the gradient's device (GPU).
    """
    d_out = g_r.shape[0]
    q = min(max(int(k), 1), min(g_r.shape))
    try:
        Ur, Sr, _ = torch.svd_lowrank(g_r.float(), q=q, niter=2)
    except Exception:
        return g_f
    if sv_thresh > 0.0:
        smax = float(Sr.max().item()) if Sr.numel() else 0.0
        keep = (Sr / max(smax, _EPS)) >= sv_thresh
        Ur = Ur[:, keep]
    if Ur.shape[1] == 0:
        return g_f
    # Project g_f's column space out of Ur.
    proj = Ur @ (Ur.t() @ g_f.float())
    out = g_f.float() - proj
    return out.to(g_f.dtype)


def project_gradients_advanced(
    grads_forget: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    *,
    method: str,
    cosine_threshold,
    per_layer_threshold: bool,
    per_layer_threshold_delta: float,
    cagrad_c: float,
    gradvac_phi: float,
    gradvac_beta: float,
    magnitude_preserve: bool,
    nullspace_rank: int,
    nullspace_sv_threshold: float,
    always_project: bool,
    add_retain_grad: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Dispatch per-module projection using the selected method."""
    projected: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Any] = {
        "applied": True,
        "method": method,
        "cosine_threshold": cosine_threshold,
        "per_layer_threshold": per_layer_threshold,
        "magnitude_preserve": bool(magnitude_preserve),
        "always_project": bool(always_project),
        "modules": {},
    }

    cos_tau = _decide_cosine_thresholds(
        grads_forget, grads_retain,
        config_cos_tau=cosine_threshold,
        per_layer=per_layer_threshold,
        delta=per_layer_threshold_delta,
    )

    # GradVac keeps a running EMA of target cosine per-module.
    phi_state: Dict[str, float] = {n: float(gradvac_phi) for n in grads_forget}

    for name, g_f in grads_forget.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            projected[name] = g_f
            stats["modules"][name] = {"status": "missing_retain_grad"}
            continue

        g_f_flat = _flat_f64(g_f)
        g_r_flat = _flat_f64(g_r)
        orig_norm = float(g_f_flat.norm().item())
        cos = _cosine(g_f_flat, g_r_flat)
        tau = cos_tau.get(name, 0.0)

        if method == "nullspace":
            # Null-space projection ignores the cosine gate by design.
            g_new = _nullspace_update(
                g_f, g_r, k=nullspace_rank, sv_thresh=nullspace_sv_threshold,
            )
            action = "nullspace"
        elif not _should_project(cos, tau, always_project):
            g_new = g_f
            action = "skipped"
        elif method == "pcgrad":
            g_new = _pcgrad_update(g_f, g_r).to(g_f.dtype)
            action = "pcgrad"
        elif method == "cagrad":
            g_new = _cagrad_update(g_f, g_r, c=cagrad_c).to(g_f.dtype)
            action = "cagrad"
        elif method == "gradvac":
            phi = phi_state.get(name, float(gradvac_phi))
            g_new_f64, observed_cos = _gradvac_update(g_f, g_r, phi=phi)
            g_new = g_new_f64.to(g_f.dtype) if isinstance(g_new_f64, torch.Tensor) else g_f
            # EMA update of target cosine.
            phi_state[name] = (1.0 - gradvac_beta) * phi + gradvac_beta * observed_cos
            action = "gradvac"
        elif method == "magnitude_preserving":
            g_new_flat = _pcgrad_update(g_f, g_r)
            g_new = _magnitude_preserve(g_new_flat, orig_norm).to(g_f.dtype)
            action = "mag_preserve_pcgrad"
        else:
            raise ValueError(f"Unknown projection method: {method!r}")

        if magnitude_preserve and method != "magnitude_preserving":
            g_new = _magnitude_preserve(g_new.float(), orig_norm).to(g_f.dtype)

        if add_retain_grad:
            g_new = g_new + g_r.to(device=g_new.device, dtype=g_new.dtype)

        projected[name] = g_new
        stats["modules"][name] = {
            "action": action,
            "cos": cos,
            "tau": tau,
            "forget_norm": orig_norm,
            "retain_norm": float(g_r_flat.norm().item()),
            "projected_norm": float(g_new.float().view(-1).norm().item()),
        }

    return projected, stats
