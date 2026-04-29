"""SAPT shared-attention router.

For each input the router computes one set of attention weights over all
task experts (LoRA adapters). The query is a small projection of the
mean-pooled input embedding; the keys are learnable per-task vectors.

The router is grown one task at a time: at the start of stage t we add a
new task key initialized from N(0, key_init_std). The query projection
is shared across stages.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


class SAPTRouter(nn.Module):
    """Per-input attention over task experts.

    forward signature:
        embeds (FloatTensor): (B, T, D) input token embeddings
        attention_mask (LongTensor or None): (B, T) — used for masked mean pooling
    returns:
        weights (FloatTensor): (B, n_tasks), softmax over current tasks
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        n_tasks: int = 0,
        key_dim: int = 64,
        key_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.key_dim = int(key_dim)
        self.key_init_std = float(key_init_std)
        # Shared query projector over the mean-pooled input embedding.
        self.query_proj = nn.Linear(self.hidden_dim, self.key_dim, bias=False)
        # task_keys grows by one row each time add_task is called.
        # Stored as a Parameter so it is trained with the rest of the router.
        self._n_tasks = 0
        self.task_keys = nn.Parameter(torch.zeros(0, self.key_dim), requires_grad=True)
        for _ in range(int(n_tasks)):
            self.add_task()

    def add_task(self) -> int:
        """Append a fresh task key. Returns the new task index (0-based)."""
        new_row = torch.randn(1, self.key_dim) * self.key_init_std
        with torch.no_grad():
            grown = torch.cat(
                [self.task_keys.detach().to(new_row.device), new_row], dim=0,
            )
        new_param = nn.Parameter(grown, requires_grad=True)
        # Replace the parameter in-place so any optimizer holding a stale
        # reference is rebuilt by the caller.
        del self.task_keys
        self.register_parameter("task_keys", new_param)
        self._n_tasks += 1
        return self._n_tasks - 1

    @property
    def n_tasks(self) -> int:
        return int(self.task_keys.shape[0])

    def forward(
        self,
        embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Mean-pool over sequence (mask-aware).
        if embeds.dim() != 3:
            raise ValueError(f"SAPTRouter expects (B, T, D) embeds, got {tuple(embeds.shape)}")
        if attention_mask is not None:
            mask = attention_mask.to(dtype=embeds.dtype).unsqueeze(-1)  # (B, T, 1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (embeds * mask).sum(dim=1) / denom
        else:
            pooled = embeds.mean(dim=1)
        q = self.query_proj(pooled.to(self.query_proj.weight.dtype))  # (B, key_dim)
        scale = 1.0 / math.sqrt(self.key_dim)
        logits = q @ self.task_keys.t() * scale  # (B, n_tasks)
        return logits.softmax(dim=-1)

    def state_dict_packed(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "key_dim": self.key_dim,
            "key_init_std": self.key_init_std,
            "n_tasks": self.n_tasks,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_packed(cls, packed: dict) -> "SAPTRouter":
        router = cls(
            hidden_dim=int(packed["hidden_dim"]),
            n_tasks=int(packed["n_tasks"]),
            key_dim=int(packed["key_dim"]),
            key_init_std=float(packed.get("key_init_std", 0.02)),
        )
        router.load_state_dict(packed["state_dict"])
        return router

    @classmethod
    def load_from_path(cls, path: str, map_location: str = "cpu") -> "SAPTRouter":
        packed = torch.load(path, map_location=map_location, weights_only=False)
        return cls.from_packed(packed)

    def save(self, path: str) -> None:
        torch.save(self.state_dict_packed(), path)


__all__ = ["SAPTRouter"]
