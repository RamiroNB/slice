"""InfLoRA: Interference-Free LoRA via input-feature null-space projection.

Liang & Li, "InfLoRA: Interference-Free Low-Rank Adaptation for Continual
Learning" (CVPR 2024, arXiv:2404.00228).

Idea. Each LoRA target module sees an input feature distribution. Across all
previously-trained tasks we accumulate the input feature covariance
    C_m = sum_t E_{x ~ task_t}[ x x^T ]   (one (d_in, d_in) matrix per module m).

Before training task t, the freshly-initialized A_m is projected onto the
*approximate null space* of C_m:
    A_m <- A_m @ (I - U U^T)
where U = top-k left singular vectors of C_m (the principal subspace of past
inputs). This makes the new LoRA update cause negligible change in the model's
response to past-task inputs while leaving capacity for the new task.

After each task we accumulate the input covariance from the just-merged base
model on the new task's data via forward hooks on the LoRA target modules.

Composes with any LoRA init (vanilla / loram / lora_ga / slice): init seeds
A/B, then InfLoRA's pre_train projection narrows A onto the safe subspace.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from .base import CLMethod

logger = logging.getLogger("cl_lora.cl_methods.inflora")


def _iter_lora_modules(lora_model: torch.nn.Module):
    """Yield (module_name, lora_module) pairs for active LoRA targets."""
    from peft.tuners.lora import Linear as LoraLinear

    for name, mod in lora_model.named_modules():
        if not isinstance(mod, LoraLinear):
            continue
        if "default" not in getattr(mod, "lora_A", {}):
            continue
        yield name, mod


def _flatten_input(x: torch.Tensor) -> torch.Tensor:
    """Reshape an input tensor of shape (..., d_in) to (N, d_in)."""
    if x.dim() < 2:
        raise ValueError(f"InfLoRA: unexpected input with dim={x.dim()}")
    return x.reshape(-1, x.shape[-1])


class InfLoRAMethod(CLMethod):
    """Null-space projection of LoRA-A using past-task input covariance."""

    name = "inflora"

    def __init__(
        self,
        *,
        nullspace_rank: int = 64,
        max_cov_batches: int = 32,
        cov_batch_size: int = 8,
        max_seq_length: int = 256,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            nullspace_rank=nullspace_rank,
            max_cov_batches=max_cov_batches,
            cov_batch_size=cov_batch_size,
            max_seq_length=max_seq_length,
            seed=seed,
            **kwargs,
        )
        self.nullspace_rank = int(nullspace_rank)
        self.max_cov_batches = int(max_cov_batches)
        self.cov_batch_size = int(cov_batch_size)
        self.max_seq_length = int(max_seq_length)
        self.seed = int(seed)
        # Per-module input feature covariance (d_in, d_in), CPU fp32 for stability.
        self._covariance: Dict[str, torch.Tensor] = {}
        self._num_stages_seen: int = 0

    # ---- pre_train: project A onto null space of past-task input covariance ----

    def pre_train(self, lora_model, *, stage_idx, retain_tasks) -> None:
        if not self._covariance:
            logger.info("InfLoRA pre_train: no past covariance yet (stage=%d), skipping projection", stage_idx)
            return

        device = next(lora_model.parameters()).device
        num_projected = 0
        num_skipped = 0
        for name, mod in _iter_lora_modules(lora_model):
            C = self._covariance.get(name)
            if C is None:
                num_skipped += 1
                continue
            A = mod.lora_A["default"].weight  # (r, d_in)
            d_in = A.shape[1]
            if C.shape != (d_in, d_in):
                logger.warning(
                    "InfLoRA: covariance shape %s does not match A.d_in=%d for %s; skipping",
                    tuple(C.shape), d_in, name,
                )
                num_skipped += 1
                continue
            U = self._top_k_left_singular(C, k=self.nullspace_rank, device=device)
            if U is None:
                num_skipped += 1
                continue
            with torch.no_grad():
                A_f = A.detach().to(torch.float32)
                # A_new = A @ (I - U U^T) = A - (A @ U) @ U^T
                AU = A_f @ U
                A_proj = A_f - AU @ U.t()
                A.data.copy_(A_proj.to(dtype=A.dtype))
            num_projected += 1
        logger.info(
            "InfLoRA pre_train: projected_modules=%d skipped=%d nullspace_rank=%d (stage=%d)",
            num_projected, num_skipped, self.nullspace_rank, stage_idx,
        )

    @staticmethod
    def _top_k_left_singular(
        C: torch.Tensor, k: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        d = C.shape[0]
        kk = max(1, min(int(k), d - 1))
        C_dev = C.to(device=device, dtype=torch.float32)
        # C is symmetric positive semi-definite. svd_lowrank gives top-q components.
        try:
            U, _S, _V = torch.svd_lowrank(C_dev, q=min(kk + 4, d), niter=4)
        except Exception as exc:
            logger.warning("InfLoRA: svd_lowrank failed (%s); skipping projection", exc)
            return None
        return U[:, :kk].contiguous()

    # ---- post_train: accumulate input feature covariance on this task's data ----

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
        # Note: at this point train_dataset is already tokenized (see train.py).
        from peft.tuners.lora import Linear as LoraLinear

        # Hooks capture the input to the *base linear* under each LoRA module.
        # That input is also the input to A and is the right thing to put a
        # null-space mask against.
        accumulators: Dict[str, torch.Tensor] = {}
        sample_counts: Dict[str, int] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def make_hook(module_name: str):
            def _hook(module, inputs, output):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                with torch.no_grad():
                    x_flat = _flatten_input(x).to(torch.float32)
                    contrib = x_flat.t() @ x_flat  # (d_in, d_in)
                    prev = accumulators.get(module_name)
                    accumulators[module_name] = contrib if prev is None else prev + contrib
                    sample_counts[module_name] = sample_counts.get(module_name, 0) + x_flat.shape[0]
            return _hook

        for name, mod in _iter_lora_modules(lora_model):
            base_layer = mod.get_base_layer() if hasattr(mod, "get_base_layer") else None
            if base_layer is None:
                continue
            h = base_layer.register_forward_hook(make_hook(name))
            hooks.append(h)

        if not hooks:
            logger.warning("InfLoRA post_train: no LoRA target modules found; covariance not updated")
            return

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        gen = torch.Generator()
        gen.manual_seed(self.seed + stage_idx)
        loader = DataLoader(
            train_dataset,
            batch_size=self.cov_batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=gen,
            num_workers=0,
            pin_memory=False,
        )

        was_training = lora_model.training
        lora_model.eval()
        steps = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    if steps >= self.max_cov_batches:
                        break
                    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    lora_model(**batch)
                    steps += 1
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                lora_model.train()

        # Move to CPU fp32, accumulate into the running covariance.
        added = 0
        for name, contrib in accumulators.items():
            cpu_contrib = contrib.detach().to(device="cpu", dtype=torch.float32)
            if name in self._covariance:
                self._covariance[name] = self._covariance[name] + cpu_contrib
            else:
                self._covariance[name] = cpu_contrib
            added += 1

        self._num_stages_seen += 1
        logger.info(
            "InfLoRA covariance updated: stage=%d task=%s modules=%d batches=%d total_modules_in_state=%d",
            stage_idx, task_name, added, steps, len(self._covariance),
        )

    # ---- persistence ----

    def save(self, state_dir: str) -> None:
        os.makedirs(state_dir, exist_ok=True)
        payload = {
            "nullspace_rank": self.nullspace_rank,
            "max_cov_batches": self.max_cov_batches,
            "cov_batch_size": self.cov_batch_size,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
            "num_stages_seen": self._num_stages_seen,
            "covariance": dict(self._covariance),
        }
        torch.save(payload, os.path.join(state_dir, "inflora_state.pt"))

    def load(self, state_dir: str) -> None:
        path = os.path.join(state_dir, "inflora_state.pt")
        if not os.path.exists(path):
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.nullspace_rank = int(payload.get("nullspace_rank", self.nullspace_rank))
        self.max_cov_batches = int(payload.get("max_cov_batches", self.max_cov_batches))
        self.cov_batch_size = int(payload.get("cov_batch_size", self.cov_batch_size))
        self.max_seq_length = int(payload.get("max_seq_length", self.max_seq_length))
        self.seed = int(payload.get("seed", self.seed))
        self._num_stages_seen = int(payload.get("num_stages_seen", 0))
        self._covariance = dict(payload.get("covariance", {}))
        logger.info(
            "InfLoRA state loaded: stages_seen=%d cov_modules=%d nullspace_rank=%d",
            self._num_stages_seen, len(self._covariance), self.nullspace_rank,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nullspace_rank": int(self.nullspace_rank),
            "max_cov_batches": int(self.max_cov_batches),
            "cov_batch_size": int(self.cov_batch_size),
            "num_stages_seen": int(self._num_stages_seen),
            "cov_modules": len(self._covariance),
        }
