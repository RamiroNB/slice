from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SliceInitConfig:
    cache_dir: str = "slice_cache"
    cache_context: Optional[str] = None
    max_steps: int = 8
    per_device_batch_size: int = 64
    seed: int = 42
    retain_scale: float = 1.0
    grad_project: bool = False
    grad_projection_mode: str = "per_module"
    grad_project_always: bool = False
    add_retain_grad: bool = False
    rank: Optional[int] = None
    max_seq_length: int = 256
    retain_batch_size: Optional[int] = None
    retain_grad_accum: Optional[int] = None
    retain_batch_size_set: str = "all_tasks"
    single_retain_task_mode: bool = False
    init_method: str = "slice"  # "slice" (default), "lora_ga", or "loram"

    # Advanced projection methods (ideas A.1-A.6 from ideas_for_new_methods.md).
    # projection_method: "pcgrad" (existing), "cagrad", "gradvac",
    #                    "nullspace", "magnitude_preserving"
    projection_method: str = "pcgrad"
    # Cosine-based conflict threshold (idea A.3). If not None, projection only
    # fires when cos(g_f, g_r) < cosine_threshold. Replaces the raw dot-sign rule.
    cosine_threshold: Optional[float] = None
    # Per-layer threshold mode (idea A.4). When True the threshold is
    # set to (median_cos across modules) - per_layer_threshold_delta.
    per_layer_threshold: bool = False
    per_layer_threshold_delta: float = 0.0
    # CAGrad strength c in [0,1]. 0 = vanilla (no projection), 1 = full PCGrad (idea A.1).
    cagrad_c: float = 0.5
    # GradVac target cosine and EMA beta (idea A.2).
    gradvac_phi: float = 0.0
    gradvac_beta: float = 0.5
    # Magnitude-preserving rescale after projection (idea A.6).
    magnitude_preserve: bool = False
    # Null-space projection rank / threshold (idea A.5).
    nullspace_rank: int = 8
    nullspace_sv_threshold: float = 0.0  # relative singular-value cutoff
    # SVD selection rule (idea C.16 variant): "lora_ga" (default LoRA-GA disjoint slices)
    # or "top_r_no_sigma" (B=U[:,:r], A=V[:,:r]^T without singular-value weighting).
    svd_selection: str = "lora_ga"
    # Skip in-place absorption of `B_init A_init * scaling` into the frozen
    # base weights. Required by SAPT (parallel adapters share one base, so
    # multiple absorptions would compound). When True, init_correction.pt is
    # not written and load_model_with_adapters does not need to replay it.
    skip_absorption: bool = False
