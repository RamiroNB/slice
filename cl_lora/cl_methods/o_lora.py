"""O-LoRA: orthogonality regularizer between current and previous LoRA bases.

Wang et al., "Orthogonal Subspace Learning for Language Model Continual Learning"
(EMNLP Findings 2023, arXiv:2310.14152).

At task t, the trainable LoRA matrices A_t (one per target module) are kept
orthogonal to the row-spaces spanned by all previously-trained {A_t', t' < t}.
We add to the cross-entropy loss:

    L_orth = lambda * sum_{t' < t} sum_{module} ||A_t @ A_t'^T||_F^2

After training, snapshots of the current A matrices are saved and added to the
running history. The base-weight + adapter merge is unchanged from vanilla
(O-LoRA's original formulation also merges adapters between tasks; the prior A's
are kept only as a regularization anchor, not in the live forward pass).

Composes with any LoRA init (vanilla / loram / lora_ga / slice): init writes
the starting A/B, then training pulls A toward the orthogonal subspace.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

from .base import CLMethod

logger = logging.getLogger("cl_lora.cl_methods.o_lora")


def _iter_lora_a_matrices(lora_model: torch.nn.Module):
    """Yield (module_name, A_tensor) pairs for every active LoRA module.

    A_tensor is the live trainable nn.Parameter (shape: (r, d_in)).
    """
    from peft.tuners.lora import Linear as LoraLinear

    for name, mod in lora_model.named_modules():
        if not isinstance(mod, LoraLinear):
            continue
        a_dict = getattr(mod, "lora_A", None)
        if a_dict is None or "default" not in a_dict:
            continue
        yield name, mod.lora_A["default"].weight


class OLoRAMethod(CLMethod):
    """O-LoRA training-time orthogonality regularizer."""

    name = "o_lora"

    def __init__(
        self,
        *,
        lambda_orth: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(lambda_orth=lambda_orth, **kwargs)
        self.lambda_orth = float(lambda_orth)
        # snapshots[i] is dict[module_name -> frozen A tensor (r, d_in)] for task i.
        self._snapshots: List[Dict[str, torch.Tensor]] = []

    def pre_train(self, lora_model, *, stage_idx, retain_tasks) -> None:
        # Move snapshots onto the model device once per stage so the
        # regularizer doesn't pay a host->device copy on every forward.
        try:
            device = next(lora_model.parameters()).device
        except StopIteration:
            return
        for snap in self._snapshots:
            for k, v in snap.items():
                if v.device != device:
                    snap[k] = v.to(device)

    def aux_loss(self, lora_model: torch.nn.Module) -> Optional[torch.Tensor]:
        if not self._snapshots or self.lambda_orth == 0.0:
            return None

        device = None
        total: Optional[torch.Tensor] = None
        for name, A_curr in _iter_lora_a_matrices(lora_model):
            for snap in self._snapshots:
                A_prev = snap.get(name)
                if A_prev is None:
                    continue
                if device is None:
                    device = A_curr.device
                # Inner product in fp32 to keep the regularizer numerically sane
                # under bf16 training; cast the result back to A_curr.dtype only
                # when we attach it to the loss.
                A_c = A_curr.to(torch.float32)
                A_p = A_prev.to(device=A_c.device, dtype=torch.float32)
                term = (A_c @ A_p.t()).pow(2).sum()
                total = term if total is None else (total + term)

        if total is None:
            return None
        return self.lambda_orth * total.to(torch.float32)

    def post_train(
        self,
        lora_model,
        *,
        tokenizer,
        train_dataset,
        device,
        stage_idx,
        task_name,
    ) -> None:
        snapshot: Dict[str, torch.Tensor] = {}
        for name, A in _iter_lora_a_matrices(lora_model):
            snapshot[name] = A.detach().to(device="cpu", dtype=torch.float32).clone()
        logger.info(
            "O-LoRA snapshot captured: stage=%d task=%s modules=%d",
            stage_idx, task_name, len(snapshot),
        )
        self._snapshots.append(snapshot)

    def save(self, state_dir: str) -> None:
        os.makedirs(state_dir, exist_ok=True)
        payload = {
            "lambda_orth": self.lambda_orth,
            "num_snapshots": len(self._snapshots),
            "snapshots": [dict(s) for s in self._snapshots],
        }
        torch.save(payload, os.path.join(state_dir, "o_lora_state.pt"))

    def load(self, state_dir: str) -> None:
        path = os.path.join(state_dir, "o_lora_state.pt")
        if not os.path.exists(path):
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.lambda_orth = float(payload.get("lambda_orth", self.lambda_orth))
        snaps = payload.get("snapshots", [])
        self._snapshots = [dict(s) for s in snaps]
        logger.info("O-LoRA state loaded: snapshots=%d lambda=%s", len(self._snapshots), self.lambda_orth)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lambda_orth": float(self.lambda_orth),
            "num_snapshots": len(self._snapshots),
        }
