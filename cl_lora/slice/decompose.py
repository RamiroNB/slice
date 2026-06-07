from __future__ import annotations

import logging
import math
from typing import Dict

import torch

logger = logging.getLogger("cl_lora.slice.decompose")


def build_ab_from_gradient(
    G: torch.Tensor,
    r: int,
    weight_var: float,
    svd_selection: str = "lora_ga",
) -> Dict[str, torch.Tensor]:
    """Decompose a gradient matrix into LoRA A/B via low-rank SVD.

    svd_selection:
      - "lora_ga": B=U[:,:r], A=V[r:2r,:]^T (LoRA-GA disjoint slices, BA=0 at init).
      - "top_r_no_sigma": B=U[:,:r], A=V[:,:r]^T (top-r singular vectors, no
        sigma weighting — idea C.16 variant without magnitude weighting).
    """
    device = G.device
    d_out, d_in = G.shape
    # Safety net: zero any non-finite entries before SVD. The accumulator already
    # skips non-finite-loss batches and sanitizes per-step, but a residual NaN/Inf
    # here would make torch.linalg.svd fail to converge ("non-finite values").
    G32 = torch.nan_to_num(G.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if svd_selection == "lora_ga":
        q = min(4 * r, min(G32.shape))
    else:
        q = min(max(2 * r, r + 2), min(G32.shape))
    if q <= 0:
        raise ValueError("Invalid rank for slice initialization")

    try:
        U, _, V = torch.svd_lowrank(G32, q=q, niter=4)
        Vt = V.t()
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        # Randomized SVD (cuSOLVER) can fail to converge on ill-conditioned
        # gradient matrices; the failure depends on the random projection draw.
        # Fall back to a full, deterministic SVD on CPU (LAPACK), which yields the
        # EXACT top-q singular vectors that svd_lowrank was approximating. Only the
        # rare failing module takes this path; successful modules are unchanged.
        logger.warning(
            "svd_lowrank failed (%s); full CPU SVD fallback for shape=%s q=%d",
            type(e).__name__, tuple(G32.shape), q,
        )
        U_f, _, Vh_f = torch.linalg.svd(G32.detach().cpu(), full_matrices=False)
        U = U_f[:, :q].to(device)
        Vt = Vh_f[:q, :].to(device)
    if svd_selection == "lora_ga":
        B = U[:, :r]
        A = Vt[r : 2 * r, :]
    elif svd_selection == "top_r_no_sigma":
        # Use the correct top-r singular vectors but discard singular values,
        # so BA is a rank-r product of orthonormal factors (not the SVD
        # reconstruction). The variance-matched rescale below still applies.
        B = U[:, :r]
        A = Vt[:r, :]
    else:
        raise ValueError(f"Unknown svd_selection: {svd_selection!r}")

    # Match LoRAM rescaling: variance-matched scaling using rho/variance_ratio/beta.
    eps = 1e-12
    recon = B @ A
    var_recon = float(torch.var(recon).item()) if torch.var(recon).item() != 0.0 else eps
    variance_ratio = float(weight_var) / (var_recon + eps)
    min_dim = max(2, min(d_out, d_in))
    r_val = max(2, r)
    rho = math.log(r_val, min_dim)
    beta = math.pow(max(rho * variance_ratio, eps), 1.0 / 4.0)
    B = B * beta
    A = A * beta

    return {
        "A": A.to(device=device, dtype=G.dtype).contiguous(),
        "B": B.to(device=device, dtype=G.dtype).contiguous(),
    }


def fast_dst_matrix(m: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate an m x n DST-I matrix."""
    transpose = False
    if m < n:
        m, n = n, m
        transpose = True

    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # (n, 1)
    i = torch.arange(m, device=device, dtype=dtype).unsqueeze(0)  # (1, m)

    dst_basis = torch.sin((i + 1) * (k + 1) * torch.pi / (m + 1))
    scale_local = torch.sqrt(torch.tensor(2.0 / (m + 1), device=device, dtype=dtype))
    dst_matrix = (scale_local * dst_basis).t()

    if transpose:
        dst_matrix = dst_matrix.t()

    return dst_matrix.to(device=device, dtype=dtype)


def build_ab_loram(
    d_out: int, d_in: int, r: int, weight_var: float, device: torch.device, dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Build LoRAM A/B from DST matrices with variance-matched scaling."""
    A = fast_dst_matrix(r, d_in, device=device, dtype=torch.float32)
    B = fast_dst_matrix(d_out, r, device=device, dtype=torch.float32)

    eps = 1e-12
    recon = B @ A
    var_recon = float(torch.var(recon).item()) if torch.var(recon).item() != 0.0 else eps
    variance_ratio = float(weight_var) / (var_recon + eps)
    min_dim = max(2, min(d_out, d_in))
    r_val = max(2, r)
    rho = math.log(r_val, min_dim)
    beta = math.pow(max(rho * variance_ratio, eps), 1.0 / 4.0)
    B = B * beta
    A = A * beta

    return {
        "A": A.to(device=device, dtype=dtype).contiguous(),
        "B": B.to(device=device, dtype=dtype).contiguous(),
    }
