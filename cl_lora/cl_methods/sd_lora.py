"""SD-LoRA + SLICE: decoupled magnitude/direction continual learning.

Wu et al., "SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class
Incremental Learning" (ICLR 2025, arXiv:2501.13198).
Reference: https://github.com/WuYichen-97/SD-Lora-CL

Idea. The accumulated model is

    W_eff = W0 + sum_i s_i * D_i,     D_i = M_i / ||M_i||_F,

where W0 is the frozen pretrained weight (pristine at every stage, no merge),
M_i is task i's *net* weight update, D_i is its unit-norm direction (frozen once
task i is done), and s_i is a trainable magnitude scalar. Every s_i stays
trainable while learning any later task, so the magnitudes of earlier tasks are
jointly re-tuned -- the "decoupled magnitude/direction" mechanism that lets the
run settle into a low-loss region shared by all tasks. gamma is the rsLoRA
scaling (alpha / sqrt(r)).

Combination with SLICE ("absorb + freeze net update" variant). Task t trains as
a completely normal *absorbed* SLICE run: SLICE picks the gradient-SVD subspace
and its usual variance-matched `beta` scaling of A/B, and the init is
output-invariant. At the end of the task its net update

    M_t = gamma * (B_t^f A_t^f  -  B_t^i A_t^i)      (final minus init, rank <= 2r)

is frozen as task t's SD-LoRA direction, and a trainable scalar (init ||M_t||_F,
so s_t*D_t == M_t at freeze) is handed to future tasks to re-tune. SLICE keeps
its original job (init/direction); the SD-LoRA scalar keeps its original job
(cross-task magnitude), cleanly separated.

Absorption is realized *in the forward*, not by mutating base weights: the
override computes base(x) on the pristine W0 and adds the current task's net
term gamma*(B_t A_t - B_t^i A_t^i). This is numerically identical to physically
subtracting gamma*B_t^i A_t^i from the base weight (the correction is constant,
so gradients on A_t, B_t are unchanged) but keeps W0 pristine across stages so
the no-merge accumulation W0 + sum s_i D_i holds exactly.

Lifecycle (driven by cl_lora.train / cl_lora.orchestrator):
  * pre_train installs the forward override on every LoRA module: snapshots the
    current SLICE init (B^i, A^i) as the absorption reference, restores the
    frozen previous task blocks, and registers a trainable scalar vector for the
    previous tasks (empty at stage 1 -> pure SLICE).
  * post_train freezes the current task's net block and appends its scalar
    (||M_t||_F), and records the re-tuned previous scalars.
  * save()/load() persist state across stages; a cumulative sd_lora_state.pt is
    also dropped next to each stage's adapter so standalone eval can bake
    W0 + sum s_i D_i via `bake_sdlora_into_model`.
"""
from __future__ import annotations

import logging
import os
import types
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import CLMethod

logger = logging.getLogger("cl_lora.cl_methods.sd_lora")

_EPS = 1e-12
_ADAPTER = "default"


def _iter_lora_linears(model: torch.nn.Module):
    """Yield (module_name, LoraLinear) for every active `default` LoRA module."""
    from peft.tuners.lora import Linear as LoraLinear

    for name, mod in model.named_modules():
        if isinstance(mod, LoraLinear) and _ADAPTER in getattr(mod, "lora_A", {}):
            yield name, mod


def _lowrank_net_fro_norm(
    B_f: torch.Tensor, A_f: torch.Tensor, B_i: torch.Tensor, A_i: torch.Tensor, gamma: float
) -> float:
    """||gamma * (B_f A_f - B_i A_i)||_F, computed without forming the dense d_out x d_in.

    Stacks B_cat = [B_f | -B_i] (out, 2r), A_cat = [A_f ; A_i] (2r, in) so the net
    update is gamma * B_cat @ A_cat, then uses ||B_cat A_cat||_F^2 = tr((B_cat^T B_cat)(A_cat A_cat^T)).
    """
    Bc = torch.cat([B_f, -B_i], dim=1).float()          # (out, 2r)
    Ac = torch.cat([A_f, A_i], dim=0).float()           # (2r, in)
    gram_b = Bc.t() @ Bc                                 # (2r, 2r)
    gram_a = Ac @ Ac.t()                                 # (2r, 2r)
    sq = float((gram_b * gram_a.t()).sum().item())       # tr(gram_b @ gram_a)
    sq = max(sq, 0.0)
    return float(gamma) * (sq ** 0.5)


def _sdlora_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Replacement forward bound onto each LoRA module (see module docstring).

    result = base(x)                                         # W0 (pristine)
           + gamma * (B_t A_t(dropout x) - B_t^i A_t^i x)    # current net (absorption in forward)
           + sum_{i<t} (s_i/||M_i||) * gamma * (B_i^f A_i^f x - B_i^i A_i^i x)
    """
    kwargs.pop("adapter_names", None)  # mixed-batch path not supported here
    state = self._sdlora
    result = self.base_layer(x, *args, **kwargs)

    if self.disable_adapters or self.merged:
        return result

    lora_A = self.lora_A[_ADAPTER]
    lora_B = self.lora_B[_ADAPTER]
    dropout = self.lora_dropout[_ADAPTER]
    gamma = self.scaling[_ADAPTER]

    rdtype = result.dtype
    xc = x.to(lora_A.weight.dtype)

    # Current task net term: live (B_t A_t) minus its own SLICE init (frozen).
    cur_live = lora_B(lora_A(dropout(xc)))
    init_A = state["cur_init_A"]
    init_B = state["cur_init_B"]
    cur_init = F.linear(F.linear(xc, init_A), init_B)
    result = result + (gamma * (cur_live - cur_init)).to(rdtype)

    # Previous frozen task blocks, each re-scaled by its trainable magnitude.
    scalars = state["scalars"]  # nn.Parameter (num_prev,) or None
    if scalars is not None:
        prev = state["prev"]  # list of dicts with A_f,B_f,A_i,B_i,norm
        for i, blk in enumerate(prev):
            net = F.linear(F.linear(xc, blk["A_f"]), blk["B_f"]) - F.linear(
                F.linear(xc, blk["A_i"]), blk["B_i"]
            )
            coeff = (scalars[i] / blk["norm"]) * gamma
            result = result + coeff.to(rdtype) * net.to(rdtype)

    return result.to(rdtype)


class SDLoRAMethod(CLMethod):
    """SD-LoRA continual-learning method (composes with SLICE / any LoRA init)."""

    name = "sd_lora"

    # The base weights must stay pristine (no physical absorption, no merge);
    # train.py reads these flags to adjust the per-stage lifecycle.
    requires_skip_absorption = True
    requires_no_merge = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Per-module frozen task blocks:
        #   name -> [ {"A_f","B_f","A_i","B_i","norm"}, ... ] (cpu tensors + float norm)
        self._blocks: Dict[str, List[Dict[str, Any]]] = {}
        # Per-module trainable scalar values (magnitudes), length = #frozen tasks.
        self._scalars: Dict[str, List[float]] = {}
        # rsLoRA scaling gamma = alpha / sqrt(r) (identical across target modules).
        self._scaling: Optional[float] = None
        self._num_tasks: int = 0
        self._installed: List[str] = []

    # ------------------------------------------------------------------ hooks
    def pre_train(self, lora_model, *, stage_idx, retain_tasks) -> None:
        """Install the SD-LoRA forward + trainable scalars on every LoRA module.

        Runs AFTER the SLICE init has written the current-task A/B, so the live
        adapter weights here are exactly the SLICE init (B^i, A^i) used as the
        in-forward absorption reference and, later, in the net-update block.
        """
        try:
            device = next(lora_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self._installed = []
        for name, module in _iter_lora_linears(lora_model):
            if self._scaling is None:
                self._scaling = float(module.scaling[_ADAPTER])

            prev_blocks = self._blocks.get(name, [])
            prev_scalars = list(self._scalars.get(name, []))
            if len(prev_scalars) != len(prev_blocks):
                logger.warning(
                    "sd_lora: %s has %d scalars but %d blocks; truncating",
                    name, len(prev_scalars), len(prev_blocks),
                )
                prev_scalars = prev_scalars[: len(prev_blocks)]

            self._install_module(module, prev_blocks, prev_scalars, device=device)
            self._installed.append(name)

        self._num_tasks = int(stage_idx)
        logger.info(
            "SD-LoRA installed on %d modules (stage=%d, frozen_tasks=%d, gamma=%.6g)",
            len(self._installed), int(stage_idx), int(stage_idx) - 1,
            self._scaling if self._scaling is not None else float("nan"),
        )

    def _install_module(
        self,
        module: torch.nn.Module,
        prev_blocks: List[Dict[str, Any]],
        prev_scalars: List[float],
        *,
        device: torch.device,
    ) -> None:
        A_w = module.lora_A[_ADAPTER].weight
        B_w = module.lora_B[_ADAPTER].weight
        dtype = A_w.dtype

        # Snapshot the SLICE init (frozen absorption reference for this task).
        module.register_buffer(
            "sdlora_cur_init_A", A_w.detach().clone().to(device=device, dtype=dtype),
            persistent=False,
        )
        module.register_buffer(
            "sdlora_cur_init_B", B_w.detach().clone().to(device=device, dtype=dtype),
            persistent=False,
        )

        prev: List[Dict[str, torch.Tensor]] = []
        for j, blk in enumerate(prev_blocks):
            entry: Dict[str, Any] = {}
            for key in ("A_f", "B_f", "A_i", "B_i"):
                t = blk[key].to(device=device, dtype=dtype)
                bufname = f"sdlora_prev{j}_{key}"
                module.register_buffer(bufname, t, persistent=False)
                entry[key] = getattr(module, bufname)
            entry["norm"] = max(float(blk["norm"]), _EPS)
            prev.append(entry)

        scalars: Optional[torch.nn.Parameter] = None
        if prev_scalars:
            scalars = torch.nn.Parameter(
                torch.tensor([float(s) for s in prev_scalars], dtype=torch.float32, device=device)
            )
            module.register_parameter("sdlora_scalars", scalars)

        module._sdlora = {
            "cur_init_A": getattr(module, "sdlora_cur_init_A"),
            "cur_init_B": getattr(module, "sdlora_cur_init_B"),
            "prev": prev,
            "scalars": scalars,
        }
        module.forward = types.MethodType(_sdlora_forward, module)

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
        """Freeze the current task's net block; record re-tuned + new scalars."""
        gamma = float(self._scaling) if self._scaling is not None else 1.0
        for name, module in _iter_lora_linears(lora_model):
            state = getattr(module, "_sdlora", None)
            if state is None:
                continue
            A_f = module.lora_A[_ADAPTER].weight.detach()
            B_f = module.lora_B[_ADAPTER].weight.detach()
            A_i = state["cur_init_A"].detach()
            B_i = state["cur_init_B"].detach()
            norm = _lowrank_net_fro_norm(B_f, A_f, B_i, A_i, gamma)

            block = {
                "A_f": A_f.to("cpu").clone(),
                "B_f": B_f.to("cpu").clone(),
                "A_i": A_i.to("cpu").clone(),
                "B_i": B_i.to("cpu").clone(),
                "norm": float(norm),
            }
            self._blocks.setdefault(name, []).append(block)

            # Record re-tuned previous scalars (if any), then append the new one.
            scalars_param = state["scalars"]
            if scalars_param is not None:
                self._scalars[name] = [float(s) for s in scalars_param.detach().cpu().tolist()]
            else:
                self._scalars.setdefault(name, [])
            # New task's magnitude scalar init = ||M_t||_F so s_t * D_t == M_t.
            self._scalars[name].append(float(norm))

        self._num_tasks = int(stage_idx)
        logger.info(
            "SD-LoRA freeze: stage=%d task=%s modules=%d blocks/module=%d",
            int(stage_idx), task_name, len(self._blocks), self._num_tasks,
        )

    # ------------------------------------------------------------- persistence
    def _state_payload(self) -> Dict[str, Any]:
        return {
            "blocks": self._blocks,
            "scalars": self._scalars,
            "scaling": self._scaling,
            "num_tasks": self._num_tasks,
        }

    def save(self, state_dir: str) -> None:
        os.makedirs(state_dir, exist_ok=True)
        torch.save(self._state_payload(), os.path.join(state_dir, "sd_lora_state.pt"))

    def load(self, state_dir: str) -> None:
        path = os.path.join(state_dir, "sd_lora_state.pt")
        if not os.path.exists(path):
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._blocks = payload.get("blocks", {})
        self._scalars = payload.get("scalars", {})
        self._scaling = payload.get("scaling", None)
        self._num_tasks = int(payload.get("num_tasks", 0))
        logger.info(
            "SD-LoRA state loaded: modules=%d tasks=%d", len(self._blocks), self._num_tasks
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_tasks": int(self._num_tasks),
            "num_modules": len(self._blocks),
            "scaling": None if self._scaling is None else float(self._scaling),
        }


# ---------------------------------------------------------------------------
# Standalone-eval reconstruction: bake W0 + sum s_i D_i into a plain HF model.
# ---------------------------------------------------------------------------
def _normalize_module_name(name: str) -> str:
    out = str(name)
    for suffix in (".weight", ".base_layer"):
        out = out.replace(suffix, "")
    for prefix in ("base_model.model.", "base_model."):
        if out.startswith(prefix):
            out = out[len(prefix):]
    return out


def _resolve_linear(norm_key: str, index: Dict[str, str]) -> Optional[str]:
    if norm_key in index:
        return index[norm_key]
    candidates = [
        real for norm, real in index.items()
        if norm.endswith(norm_key) or norm_key.endswith(norm)
    ]
    return candidates[0] if len(candidates) == 1 else None


def bake_sdlora_into_model(model: torch.nn.Module, state: Dict[str, Any]) -> int:
    """Add sum_i (s_i/||M_i||) * gamma * (B_i^f A_i^f - B_i^i A_i^i) into each weight.

    `model` must be a pristine base model (no adapters). `state` is the payload
    saved by `SDLoRAMethod.save`. Returns the number of modules modified.
    """
    blocks: Dict[str, List[Dict[str, Any]]] = state.get("blocks", {})
    scalars: Dict[str, List[float]] = state.get("scalars", {})
    gamma = float(state.get("scaling") or 0.0)
    if gamma == 0.0:
        raise ValueError("sd_lora reconstruction requires a non-zero 'scaling' in state.")

    linear_index: Dict[str, str] = {}
    named = dict(model.named_modules())
    for mod_name, mod in named.items():
        if isinstance(mod, torch.nn.Linear):
            linear_index[_normalize_module_name(mod_name)] = mod_name

    modified = 0
    for train_name, blks in blocks.items():
        target = _resolve_linear(_normalize_module_name(train_name), linear_index)
        if target is None:
            logger.warning("sd_lora bake: no base Linear matched for %s; skipping", train_name)
            continue
        module = named[target]
        weight = module.weight
        s_vec = scalars.get(train_name, [])
        if len(s_vec) < len(blks):
            raise ValueError(
                f"sd_lora bake: {train_name} has {len(blks)} blocks but {len(s_vec)} scalars."
            )

        orig_dtype = weight.dtype
        delta = torch.zeros_like(weight, dtype=torch.float32)
        for i, blk in enumerate(blks):
            A_f = blk["A_f"].to(device=weight.device, dtype=torch.float32)
            B_f = blk["B_f"].to(device=weight.device, dtype=torch.float32)
            A_i = blk["A_i"].to(device=weight.device, dtype=torch.float32)
            B_i = blk["B_i"].to(device=weight.device, dtype=torch.float32)
            norm = max(float(blk["norm"]), _EPS)
            coeff = (float(s_vec[i]) / norm) * gamma
            delta += coeff * (B_f @ A_f - B_i @ A_i)
        weight.data.copy_((weight.data.to(torch.float32) + delta).to(orig_dtype))
        modified += 1

    logger.info("sd_lora bake: modified %d/%d modules", modified, len(blocks))
    return modified
