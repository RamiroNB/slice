"""Advanced gradient-surgery projection methods.

Implements ideas A.1 (PCGrad_c), A.2 (GradVac), A.3 (cosine threshold),
A.4 (per-layer threshold), A.5 (null-space), A.6 (magnitude preserving)
from ideas_for_new_methods.md. Kept on the same device as input
gradients to avoid CPU offload of large tensors.

Global mode:
  For methods whose projection math has a distributive form (pcgrad,
  pcgrad_c, magnitude_preserving, and pcgrad/pcgrad_c with the cosine-
  threshold gate), a single global gamma is computed by summing
  per-module dot products and retain-gradient norms -- no concatenated
  whole-model vector is ever built. Methods whose math is inherently
  per-module (nullspace, per-layer threshold, gradvac) raise a clear
  error if global projection is requested, rather than silently running
  per-module.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Tuple

import torch

logger = logging.getLogger("cl_lora.slice.projections")

_EPS = 1e-12

# Methods whose projection is well-defined globally via the distributive
# property (sum of per-module dot products / squared norms).
_GLOBAL_COMPATIBLE_METHODS = {"pcgrad", "pcgrad_c", "magnitude_preserving"}


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
    grads_current: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    *,
    config_cos_tau,
    per_layer: bool,
    delta: float,
) -> Dict[str, float]:
    """Return per-module cosine threshold to compare against."""
    cos_map: Dict[str, float] = {}
    for name, g_c in grads_current.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            continue
        cos_map[name] = _cosine(_flat_f64(g_c), _flat_f64(g_r))

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


def _pcgrad_c_update(g_f: torch.Tensor, g_r: torch.Tensor, c: float) -> torch.Tensor:
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
    """Rotate g_f so cos(new, g_r) = phi (closed form from GradVac).

    Returns (new_g_f_with_original_shape, observed_cos_before_update).
    If already aligned beyond phi, returns g_f unchanged.
    """
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
    out = (g_f_flat + lam * g_r_flat).view(g_f.shape).to(g_f.dtype)
    return out, cos


def _nullspace_update(
    g_f: torch.Tensor, g_r: torch.Tensor, k: int, sv_thresh: float,
) -> torch.Tensor:
    """Project g_f out of the top-k left-singular subspace of g_r.

    Uses SVD of the retain matrix (not covariance) as a proxy for the
    preserve-feature subspace -- Adam-NSCL style but gradient-based.
    Runs on the gradient's device (GPU).

    SVD failures propagate -- do not silently fall back to the untouched
    current-task gradient, because that would make the run indistinguishable
    from a no-projection run without any warning.
    """
    q = min(max(int(k), 1), min(g_r.shape))
    Ur, Sr, _ = torch.svd_lowrank(g_r.float(), q=q, niter=2)
    if sv_thresh > 0.0:
        smax = float(Sr.max().item()) if Sr.numel() else 0.0
        keep = (Sr / max(smax, _EPS)) >= sv_thresh
        Ur = Ur[:, keep]
    if Ur.shape[1] == 0:
        raise RuntimeError(
            "nullspace_update: no singular directions passed sv_threshold "
            f"({sv_thresh}); refusing to silently return unprojected g_f."
        )
    # Project g_f's column space out of Ur.
    proj = Ur @ (Ur.t() @ g_f.float())
    out = g_f.float() - proj
    return out.to(g_f.dtype)


def _global_accumulators(
    grads_current: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
) -> Tuple[float, float, float]:
    """Compute sum_dot, sum_denom_r, sum_denom_c across modules.

    Uses the distributive property: <concat(a_i), concat(b_i)> = sum_i <a_i, b_i>,
    so no whole-model flattened vector is ever materialised.
    """
    sum_dot = 0.0
    sum_denom_r = 0.0
    sum_denom_c = 0.0
    for name, g_c in grads_current.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            continue
        g_c_flat = _flat_f64(g_c)
        g_r_flat = _flat_f64(g_r)
        sum_dot += float(torch.dot(g_c_flat, g_r_flat).item())
        sum_denom_r += float(torch.dot(g_r_flat, g_r_flat).item())
        sum_denom_c += float(torch.dot(g_c_flat, g_c_flat).item())
    return sum_dot, sum_denom_r, sum_denom_c


def _global_projection(
    grads_current: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    *,
    method: str,
    cosine_threshold,
    pcgrad_c: float,
    magnitude_preserve: bool,
    always_project: bool,
    add_retain_grad: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Global variant for pcgrad / pcgrad_c / magnitude_preserving.

    Computes one scalar gamma from summed dot-products and retain-grad
    squared norms, then applies it uniformly: g_c_i <- g_c_i + gamma * g_r_i
    (for pcgrad_c, gamma *= c). Magnitude preservation, if requested, is
    applied per module afterwards (orthogonal to projection scope).
    """
    if method not in _GLOBAL_COMPATIBLE_METHODS:
        # Defensive -- caller should have validated.
        raise ValueError(
            f"_global_projection called with unsupported method {method!r}; "
            f"expected one of {sorted(_GLOBAL_COMPATIBLE_METHODS)}."
        )

    sum_dot, sum_denom_r, sum_denom_c = _global_accumulators(
        grads_current, grads_retain,
    )
    global_cos = 0.0
    if sum_denom_r > _EPS and sum_denom_c > _EPS:
        global_cos = sum_dot / math.sqrt(sum_denom_r * sum_denom_c)

    tau = float(cosine_threshold) if cosine_threshold is not None else 0.0
    do_project = always_project or (global_cos < tau)

    gamma_full = -sum_dot / (sum_denom_r + _EPS)
    if method == "pcgrad_c":
        gamma = float(pcgrad_c) * gamma_full
    else:
        # pcgrad and magnitude_preserving use full pcgrad gamma before
        # any per-module rescale.
        gamma = gamma_full

    projected: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Any] = {
        "applied": True,
        "method": method,
        "mode": "global",
        "cosine_threshold": tau,
        "magnitude_preserve": bool(magnitude_preserve),
        "always_project": bool(always_project),
        "global": {
            "sum_dot": sum_dot,
            "sum_denom_r": sum_denom_r,
            "sum_denom_c": sum_denom_c,
            "cos": global_cos,
            "gamma": gamma,
            "do_project": bool(do_project),
        },
        "modules": {},
    }

    for name, g_c in grads_current.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            projected[name] = g_c
            stats["modules"][name] = {"status": "missing_retain_grad"}
            continue

        orig_norm = float(_flat_f64(g_c).norm().item())

        if do_project:
            g_c_flat = _flat_f64(g_c)
            g_r_flat = _flat_f64(g_r)
            g_new = (g_c_flat + gamma * g_r_flat).view(g_c.shape).to(g_c.dtype)
            action = method
        else:
            g_new = g_c
            action = "skipped"

        if magnitude_preserve and do_project:
            g_new = _magnitude_preserve(g_new.float(), orig_norm).to(g_c.dtype)

        if add_retain_grad:
            g_new = g_new + g_r.to(device=g_new.device, dtype=g_new.dtype)

        projected[name] = g_new
        stats["modules"][name] = {
            "action": action,
            "current_norm": orig_norm,
            "retain_norm": float(_flat_f64(g_r).norm().item()),
            "projected_norm": float(g_new.float().view(-1).norm().item()),
        }

    return projected, stats


def project_gradients_advanced(
    grads_current: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    *,
    method: str,
    cosine_threshold,
    per_layer_threshold: bool,
    per_layer_threshold_delta: float,
    pcgrad_c: float,
    gradvac_phi: float,
    gradvac_beta: float,
    magnitude_preserve: bool,
    nullspace_rank: int,
    nullspace_sv_threshold: float,
    always_project: bool,
    add_retain_grad: bool,
    global_projection: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Dispatch projection using the selected method.

    If global_projection is True, methods in _GLOBAL_COMPATIBLE_METHODS are
    handled by _global_projection (single scalar gamma via distributive sum).
    Methods whose math has no meaningful global analog (nullspace, gradvac,
    per_layer_threshold) raise ValueError rather than running per-module
    silently.
    """
    if global_projection:
        reasons = []
        if method not in _GLOBAL_COMPATIBLE_METHODS:
            reasons.append(
                f"projection_method={method!r} is per-module by construction "
                f"(only {sorted(_GLOBAL_COMPATIBLE_METHODS)} support global)"
            )
        if per_layer_threshold:
            reasons.append(
                "per_layer_threshold requires a per-module cosine "
                "distribution and has no global analog"
            )
        if reasons:
            raise ValueError(
                "grad_projection_mode='global' is incompatible with the "
                "requested projection settings: " + "; ".join(reasons)
                + ". Either switch to grad_projection_mode='per_module' "
                "or change the projection settings."
            )
        logger.info(
            "Advanced projection (global): method=%s cos_tau=%s mag_preserve=%s",
            method, cosine_threshold, bool(magnitude_preserve),
        )
        return _global_projection(
            grads_current=grads_current,
            grads_retain=grads_retain,
            method=method,
            cosine_threshold=cosine_threshold,
            pcgrad_c=pcgrad_c,
            magnitude_preserve=magnitude_preserve,
            always_project=always_project,
            add_retain_grad=add_retain_grad,
        )

    projected: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Any] = {
        "applied": True,
        "method": method,
        "mode": "per_module",
        "cosine_threshold": cosine_threshold,
        "per_layer_threshold": per_layer_threshold,
        "magnitude_preserve": bool(magnitude_preserve),
        "always_project": bool(always_project),
        "modules": {},
    }

    cos_tau = _decide_cosine_thresholds(
        grads_current, grads_retain,
        config_cos_tau=cosine_threshold,
        per_layer=per_layer_threshold,
        delta=per_layer_threshold_delta,
    )

    # GradVac keeps a running EMA of target cosine per-module.
    phi_state: Dict[str, float] = {n: float(gradvac_phi) for n in grads_current}

    for name, g_c in grads_current.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            projected[name] = g_c
            stats["modules"][name] = {"status": "missing_retain_grad"}
            continue

        g_c_flat = _flat_f64(g_c)
        g_r_flat = _flat_f64(g_r)
        orig_norm = float(g_c_flat.norm().item())
        cos = _cosine(g_c_flat, g_r_flat)
        tau = cos_tau.get(name, 0.0)

        if method == "nullspace":
            # Null-space projection ignores the cosine gate by design.
            g_new = _nullspace_update(
                g_c, g_r, k=nullspace_rank, sv_thresh=nullspace_sv_threshold,
            )
            action = "nullspace"
        elif not _should_project(cos, tau, always_project):
            g_new = g_c
            action = "skipped"
        elif method == "pcgrad":
            g_new = _pcgrad_update(g_c, g_r).to(g_c.dtype)
            action = "pcgrad"
        elif method == "pcgrad_c":
            g_new = _pcgrad_c_update(g_c, g_r, c=pcgrad_c).to(g_c.dtype)
            action = "pcgrad_c"
        elif method == "gradvac":
            phi = phi_state.get(name, float(gradvac_phi))
            g_new, observed_cos = _gradvac_update(g_c, g_r, phi=phi)
            # EMA update of target cosine.
            phi_state[name] = (1.0 - gradvac_beta) * phi + gradvac_beta * observed_cos
            action = "gradvac"
        elif method == "magnitude_preserving":
            g_new_flat = _pcgrad_update(g_c, g_r)
            g_new = _magnitude_preserve(g_new_flat, orig_norm).to(g_c.dtype)
            action = "mag_preserve_pcgrad"
        else:
            raise ValueError(f"Unknown projection method: {method!r}")

        if magnitude_preserve and method != "magnitude_preserving" and action != "skipped":
            g_new = _magnitude_preserve(g_new.float(), orig_norm).to(g_c.dtype)

        if add_retain_grad:
            g_new = g_new + g_r.to(device=g_new.device, dtype=g_new.dtype)

        projected[name] = g_new
        stats["modules"][name] = {
            "action": action,
            "cos": cos,
            "tau": tau,
            "current_norm": orig_norm,
            "retain_norm": float(g_r_flat.norm().item()),
            "projected_norm": float(g_new.float().view(-1).norm().item()),
        }

    return projected, stats
