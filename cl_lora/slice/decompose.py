from __future__ import annotations

import math
from typing import Dict

import torch


def build_ab_from_gradient(G: torch.Tensor, r: int, weight_var: float) -> Dict[str, torch.Tensor]:
    """Decompose a gradient matrix into LoRA A/B via low-rank SVD (LoRA-GA style)."""
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
