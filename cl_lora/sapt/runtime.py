"""SAPT forward-time mixing of multiple LoRA experts via attention weights.

Strategy. PEFT loads each task's LoRA as a *named* adapter on the same model
(`peft_model.load_adapter(path, adapter_name="task_NN")`). The default
`LoraLinear.forward` iterates `self.active_adapters` and adds each
`scaling * B(A(x))` to the base output unweighted. SAPT needs each expert
contribution multiplied by a per-input attention weight.

Approach. Monkey-patch `LoraLinear.forward` once globally. The patched
function falls back to the original PEFT behaviour unless a SAPT routing
context has been entered (via `sapt_routing(...)` context manager) — when
active, the patched forward computes:

    base(x) + Σ_i α_i(input) · scaling_i · B_i(A_i(x))

where `α_i` is read from the active context. This keeps non-SAPT runs
bit-for-bit identical to today's behavior.

The context is thread-local and stores: routing weights `(B, n_tasks)`,
the ordered adapter names corresponding to each column, and an optional
`per_token` flag (currently always False — routing is computed once per
sequence on the prompt embedding).
"""
from __future__ import annotations

import contextlib
import logging
import threading
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("cl_lora.sapt.runtime")


_TLS = threading.local()
_PATCHED = False
_ORIG_FORWARD = None


# ---------------------------------------------------------------------------
# Routing context
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def sapt_routing(
    weights: torch.Tensor,
    adapter_names: List[str],
):
    """Enter a SAPT routing context for the duration of a forward pass.

    weights: (B, n_tasks) — softmax routing probabilities for each row in the
        current batch. n_tasks must equal len(adapter_names).
    adapter_names: ordered list mapping column i -> adapter name on the model.
    """
    if weights.dim() != 2:
        raise ValueError(f"SAPT routing weights must be 2D (B, n_tasks); got {tuple(weights.shape)}")
    if weights.shape[1] != len(adapter_names):
        raise ValueError(
            f"SAPT routing weights have {weights.shape[1]} columns but "
            f"{len(adapter_names)} adapter names were provided."
        )
    prev = getattr(_TLS, "ctx", None)
    _TLS.ctx = {
        "weights": weights,
        "adapter_names": list(adapter_names),
    }
    try:
        yield
    finally:
        if prev is None:
            try:
                del _TLS.ctx
            except AttributeError:
                pass
        else:
            _TLS.ctx = prev


def _current_ctx():
    return getattr(_TLS, "ctx", None)


# ---------------------------------------------------------------------------
# LoraLinear.forward patch
# ---------------------------------------------------------------------------

def _patched_lora_forward(self, x, *args, **kwargs):
    ctx = _current_ctx()
    if ctx is None:
        return _ORIG_FORWARD(self, x, *args, **kwargs)

    weights: torch.Tensor = ctx["weights"]
    adapter_names: List[str] = ctx["adapter_names"]

    # Base layer output (unmodified path through frozen weights).
    result = self.base_layer(x, *args, **kwargs)

    if x.dim() == 3:
        # Standard transformer activation: (B, T, d_in). Routing is one alpha
        # per sequence in the batch, broadcast across tokens.
        alpha_view = (-1, 1, 1)
    elif x.dim() == 2:
        alpha_view = (-1, 1)
    else:
        # Fall back to original behavior for unexpected shapes.
        return _ORIG_FORWARD(self, x, *args, **kwargs)

    for col, name in enumerate(adapter_names):
        if name not in self.lora_A or name not in self.lora_B:
            continue
        lora_A = self.lora_A[name]
        lora_B = self.lora_B[name]
        scaling = float(self.scaling[name])
        dropout = self.lora_dropout.get(name, nn.Identity())
        delta = lora_B(lora_A(dropout(x.to(lora_A.weight.dtype)))) * scaling
        alpha_i = weights[:, col].view(*alpha_view).to(dtype=delta.dtype, device=delta.device)
        result = result + alpha_i * delta
    return result


def install_lora_forward_patch() -> None:
    """Idempotently patch LoraLinear.forward."""
    global _PATCHED, _ORIG_FORWARD
    if _PATCHED:
        return
    from peft.tuners.lora import Linear as LoraLinear

    _ORIG_FORWARD = LoraLinear.forward
    LoraLinear.forward = _patched_lora_forward  # type: ignore[assignment]
    _PATCHED = True
    logger.info("SAPT: LoraLinear.forward patched")


def uninstall_lora_forward_patch() -> None:
    global _PATCHED, _ORIG_FORWARD
    if not _PATCHED or _ORIG_FORWARD is None:
        return
    from peft.tuners.lora import Linear as LoraLinear

    LoraLinear.forward = _ORIG_FORWARD
    _ORIG_FORWARD = None
    _PATCHED = False


# ---------------------------------------------------------------------------
# SAPTWrapper
# ---------------------------------------------------------------------------

class SAPTWrapper(nn.Module):
    """Drop-in model wrapper that applies SAPT mixed-adapter forward.

    Wraps a PEFT model on which all stage adapters have been loaded as
    named adapters. Forward / generate compute routing weights once from the
    input prompt embedding and stash them in a thread-local context that the
    patched LoRA forward consumes.
    """

    def __init__(
        self,
        peft_model: nn.Module,
        router: nn.Module,
        adapter_names: List[str],
    ) -> None:
        super().__init__()
        if not adapter_names:
            raise ValueError("SAPTWrapper requires at least one adapter name.")
        install_lora_forward_patch()
        self.model = peft_model
        self.router = router
        self.adapter_names = list(adapter_names)
        # Activate every adapter so PEFT keeps their weights live in the
        # forward graph; the patched forward decides which to mix and how.
        try:
            peft_model.set_adapter(self.adapter_names)
        except Exception as exc:
            logger.warning("SAPTWrapper: could not set_adapter(all): %s", exc)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def config(self):
        return self.model.config

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _compute_weights(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        embed_layer = self.get_input_embeddings()
        with torch.no_grad():
            embeds = embed_layer(input_ids)
        return self.router(embeds, attention_mask=attention_mask)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        attention_mask = kwargs.get("attention_mask", None)
        if input_ids is None:
            return self.model(*args, **kwargs)
        weights = self._compute_weights(input_ids, attention_mask)
        with sapt_routing(weights.to(self.device), self.adapter_names):
            return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        attention_mask = kwargs.get("attention_mask", None)
        if input_ids is None:
            return self.model.generate(*args, **kwargs)
        weights = self._compute_weights(input_ids, attention_mask)
        # Routing computed once on the prompt; stays fixed across the
        # autoregressive expansion. Beam search would expand the batch by
        # num_beams; we replicate routing rows to match in that case.
        num_beams = int(kwargs.get("num_beams", 1) or 1)
        if num_beams > 1:
            weights = weights.repeat_interleave(num_beams, dim=0)
        with sapt_routing(weights.to(self.device), self.adapter_names):
            return self.model.generate(*args, **kwargs)


def list_lora_adapter_names(peft_model: nn.Module) -> List[str]:
    """All adapter names present on any LoRA target of the given PEFT model."""
    from peft.tuners.lora import Linear as LoraLinear

    seen: List[str] = []
    seen_set: set = set()
    for _, mod in peft_model.named_modules():
        if not isinstance(mod, LoraLinear):
            continue
        for name in getattr(mod, "lora_A", {}).keys():
            if name not in seen_set:
                seen.append(name)
                seen_set.add(name)
    return seen


def iter_lora_modules(peft_model: nn.Module) -> Iterable:
    """Yield (module_name, lora_module) pairs."""
    from peft.tuners.lora import Linear as LoraLinear

    for name, mod in peft_model.named_modules():
        if isinstance(mod, LoraLinear):
            yield name, mod


__all__ = [
    "SAPTWrapper",
    "install_lora_forward_patch",
    "iter_lora_modules",
    "list_lora_adapter_names",
    "sapt_routing",
    "uninstall_lora_forward_patch",
]
