"""Continual-learning method interface.

A CL method composes with any LoRA initialization (vanilla, lora_ga, loram,
slice). The init step writes the initial A/B; the CL method then runs three
hooks during the per-stage training pipeline:

    initialize_lora_with_slice(...)         # init A/B (or no-op for vanilla)
    cl_method.pre_train(lora_model, ...)    # post-init projection (e.g. InfLoRA)
    Trainer(... cl_method.aux_loss ...)     # extra loss term during training (O-LoRA)
    cl_method.post_train(lora_model, ...)   # snapshot state (A's, covariance)
    cl_method.save(state_dir)               # persist for next stage / resume

State is reloaded by `load_state(state_dir)` at the start of each stage so
resuming a run picks up the same per-stage history.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch


class CLMethod:
    """Default no-op CL method. Subclasses override the hooks they need."""

    name: str = "vanilla"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def pre_train(
        self,
        lora_model: torch.nn.Module,
        *,
        stage_idx: int,
        retain_tasks: Optional[List[Any]],
    ) -> None:
        """Run after init, before training. Default: no-op."""
        return None

    def aux_loss(self, lora_model: torch.nn.Module) -> Optional[torch.Tensor]:
        """Return an extra scalar loss to add during each training step, or None."""
        return None

    def post_train(
        self,
        lora_model: torch.nn.Module,
        *,
        tokenizer: Any,
        train_dataset: Any,
        device: torch.device,
        stage_idx: int,
        task_name: str,
    ) -> None:
        """Run after training (before merge). Default: no-op."""
        return None

    def save(self, state_dir: str) -> None:
        return None

    def load(self, state_dir: str) -> None:
        return None

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, **{k: _to_jsonable(v) for k, v in self.kwargs.items()}}


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
