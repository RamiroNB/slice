from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


def assert_slice_flags_require_slice_init(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject slice-init settings passed without --slice-init.

    Training only builds an init when `slice_enabled` is set, so
    `--slice-init-method lora_ga` on its own trains a plain vanilla-init run
    (A=kaiming, B=0) under a lora_ga run name, with nothing in the logs or the
    results folder saying so.
    """
    if getattr(args, "slice_init", False):
        return
    explicit = sorted(
        dest for dest in vars(args)
        if dest.startswith("slice_")
        and dest != "slice_init"
        and getattr(args, dest) != parser.get_default(dest)
    )
    if explicit:
        flags = ", ".join("--" + dest.replace("_", "-") for dest in explicit)
        parser.error(
            f"{flags} set without --slice-init: the run would train from PEFT's "
            "vanilla init (B=0) and ignore every slice setting. Pass --slice-init, "
            "or drop these flags if a vanilla-init run is what you want."
        )


@dataclass
class SliceInitConfig:
    cache_dir: str = "slice_cache"
    cache_context: Optional[str] = None
    max_steps: int = 8
    per_device_batch_size: int = 64
    seed: int = 42
    # Pins the probe's train/eval partition to the run's, so the gradients the
    # init decomposes never touch a held-out eval question. Left None the probe
    # keeps its legacy behaviour (eval_size=1, partition follows `seed`), which
    # hands it essentially the whole dataset -- fine when the eval set moves with
    # the seed anyway, leakage once the eval set is pinned across runs. Both
    # fields enter the cache key when split_seed is set, so an init computed
    # under the legacy partition can never be reused for a pinned-split run.
    split_seed: Optional[int] = None
    eval_size: int = 200
    retain_scale: float = 1.0
    grad_project: bool = False
    grad_projection_mode: str = "per_module"
    grad_project_always: bool = False
    add_retain_grad: bool = False
    rank: Optional[int] = None
    max_seq_length: int = 1024
    # Mask prompt tokens out of the probe's loss so the gradients it decomposes
    # describe the task behaviour being scored, not the instruction text.
    # Must track train_task's completion_only_loss; it also keys the init cache.
    completion_only: bool = True
    retain_batch_size: Optional[int] = None
    retain_grad_accum: Optional[int] = None
    retain_batch_size_set: str = "all_tasks"
    single_retain_task_mode: bool = False
    init_method: str = "slice"  # "slice" (default), "lora_ga", or "loram"

    # Advanced projection methods (ideas A.1-A.6 from ideas_for_new_methods.md).
    # projection_method: "pcgrad" (existing), "pcgrad_c", "gradvac",
    #                    "nullspace", "magnitude_preserving"
    projection_method: str = "pcgrad"
    # Cosine-based conflict threshold (idea A.3). If not None, projection only
    # fires when cos(g_f, g_r) < cosine_threshold. Replaces the raw dot-sign rule.
    cosine_threshold: Optional[float] = None
    # Per-layer threshold mode (idea A.4). When True the threshold is
    # set to (median_cos across modules) - per_layer_threshold_delta.
    per_layer_threshold: bool = False
    per_layer_threshold_delta: float = 0.0
    # PCGrad_c strength c in [0,1]. 0 = vanilla (no projection), 1 = full PCGrad (idea A.1).
    pcgrad_c: float = 0.5
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
    skip_absorption: bool = False
