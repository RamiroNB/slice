"""Continual-learning training methods (composable with any LoRA init).

Public entry points:
  - REGISTRY: name -> CLMethod class
  - build_cl_method(name, **kwargs): factory used by orchestrator/train.
"""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import CLMethod
from .inflora import InfLoRAMethod
from .o_lora import OLoRAMethod
from .sapt import SAPTMethod
from .vanilla import VanillaCLMethod


REGISTRY: Dict[str, Type[CLMethod]] = {
    "vanilla": VanillaCLMethod,
    "o_lora": OLoRAMethod,
    "inflora": InfLoRAMethod,
    "sapt": SAPTMethod,
}


def build_cl_method(name: str, **kwargs: Any) -> CLMethod:
    """Instantiate a CL method by registry name. Unknown kwargs are ignored
    by methods that don't accept them, so a single argparse namespace can be
    forwarded to any method without per-method dispatch in callers."""
    key = (name or "vanilla").lower()
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown CL method: {name!r}. Available: {sorted(REGISTRY.keys())}"
        )
    cls = REGISTRY[key]
    accepted = _accepted_kwargs(cls)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


def _accepted_kwargs(cls: Type[CLMethod]) -> set:
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {p for p in sig.parameters if p not in {"self", "args", "kwargs"}}


__all__ = [
    "CLMethod",
    "InfLoRAMethod",
    "OLoRAMethod",
    "REGISTRY",
    "SAPTMethod",
    "VanillaCLMethod",
    "build_cl_method",
]
