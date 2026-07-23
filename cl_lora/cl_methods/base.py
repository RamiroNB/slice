"""Continual-learning method interface.

A CL method composes with any LoRA initialization (vanilla, lora_ga, loram,
slice). The init step writes the initial A/B; the CL method then runs four
hooks during the per-stage training pipeline:

    cl_method.before_init(lora_model, ...)  # install accumulated state pre-probe
    initialize_lora_with_slice(...)         # init A/B (or no-op for vanilla)
    cl_method.pre_train(lora_model, ...)    # post-init projection hook
    Trainer(... cl_method.aux_loss ...)     # extra loss term during training (O-LoRA)
    cl_method.post_train(lora_model, ...)   # snapshot state (A's, covariance)
    cl_method.save(state_dir)               # persist for next stage / resume

`before_init` exists for no-merge methods (sd_lora) whose previously-learned
tasks live in the CL method's state rather than in the base weights. Without it,
the init's gradient probe would run against a pristine W0 and measure gradients
at the wrong operating point -- in particular the retain gradient would describe
"learn the old tasks from scratch" instead of "where does retained performance
degrade". Merge-based methods leave it a no-op: their base already contains the
previous tasks.

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

    def before_init(
        self,
        lora_model: torch.nn.Module,
        *,
        stage_idx: int,
        retain_tasks: Optional[List[Any]],
    ) -> None:
        """Run before the LoRA init (and its gradient probe). Default: no-op.

        No-merge methods install their accumulated previous-task state here so
        the probe sees the model's true operating point.
        """
        return None

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
