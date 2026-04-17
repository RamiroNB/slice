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
