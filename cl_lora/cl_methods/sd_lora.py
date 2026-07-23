"""SD-LoRA + SLICE: decoupled magnitude/direction continual learning.

Wu et al., "SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class
Incremental Learning" (ICLR 2025, arXiv:2501.13198).
Reference: https://github.com/WuYichen-97/SD-Lora-CL

Idea. The accumulated model is

    W_eff = W0 + sum_i s_i * D_i,     ||D_i||_F = 1,

where W0 is the frozen pretrained weight (pristine at every stage, no merge),
D_i is task i's unit-norm *net update* direction (frozen once task i is done),
and s_i is a trainable magnitude scalar. Every s_i stays trainable while
learning any later task, so the magnitudes of earlier tasks are jointly
re-tuned -- the "decoupled magnitude/direction" mechanism that lets the run
settle into a low-loss region shared by all tasks.

Combination with SLICE ("absorb + freeze net update" variant). Task t trains as
a completely normal *absorbed* SLICE run: SLICE picks the gradient-SVD subspace
with its usual variance-matched `beta` scaling of A/B, and the init is
output-invariant. At the end of the task its net update

    M_t = gamma * (B_t^f A_t^f  -  B_t^i A_t^i)      (final minus init)

is frozen as task t's direction (D_t = M_t/||M_t||_F, s_t init = ||M_t||_F, so
s_t D_t == M_t at freeze). M_t is exactly what the merge-based pipeline would
have added to the base for task t. SLICE keeps its original job (init /
direction); the SD-LoRA scalar keeps its original job (cross-task magnitude).

Absorption is realized *in the forward*, not by mutating base weights, so W0
stays pristine across stages and the no-merge accumulation holds exactly.

NUMERICAL PRECISION (this is why the code looks the way it does).
M_t is a *small* residual: per SLICE_init_investigation.md §6 the init barely
moves (subspace cos >= 0.996, ||BA|| change ~0.1%), and ||dW_f - dW_i||/||dW_i||
is only 2-6%. Naively evaluating gamma*(B^f A^f - B^i A^i) as written subtracts
two large, ~97%-identical quantities -- catastrophic cancellation. In bf16
(~0.4% relative precision) that leaves roughly one significant digit on the
learned signal. Two fixes, both exact:

  * Current (live) task: use the algebraic identity
        B_t A_t - B^i A^i  ==  dB @ A_t  +  B^i @ dA,
        dB = B_t - B^i,  dA = A_t - A^i
    Same cost (two low-rank products), but every term is proportional to a
    delta, so we ADD two small quantities instead of SUBTRACTING two big ones.
    No large intermediate is ever rounded. dB/dA are themselves exact even in
    bf16 (Sterbenz: the operands are within a factor of 2), and at init they are
    identically zero, so the invariant init is exact by construction.

  * Frozen tasks: collapse M_i ONCE, in fp32, at freeze time into clean
    orthonormal factors via `_collapse_net_update` -- an exact thin SVD of the
    rank-<=2r product that never forms the dense d_out x d_in matrix (QR both
    thin factors, SVD the small 2r x 2r core). The forward then evaluates a
    single well-conditioned low-rank product per task with NO subtraction, so
    the cancellation is resolved once in high precision instead of being redone
    every forward in bf16 for all T tasks. This also drops the rank from 2r back
    toward r (negligible singular directions are pruned) and halves cost.

Lifecycle (driven by cl_lora.train / cl_lora.orchestrator):
  * pre_train installs the forward override: snapshots the SLICE init (B^i, A^i)
    as the absorption reference, restores the frozen collapsed blocks, and
    registers a trainable scalar vector for the previous tasks (empty at stage 1
    -> pure SLICE).
  * post_train collapses+freezes the current task's net update and appends its
    scalar (||M_t||_F), and records the re-tuned previous scalars.
  * save()/load() persist state across stages; a cumulative sd_lora_state.pt is
    also dropped next to each stage's adapter so standalone eval can bake
    W0 + sum s_i D_i via `bake_sdlora_into_model`.
"""
from __future__ import annotations

import logging
import os
import types
from typing import Any, Dict, List, Optional, Tuple

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


def _collapse_net_update(
    B_f: torch.Tensor,
    A_f: torch.Tensor,
    B_i: torch.Tensor,
    A_i: torch.Tensor,
    gamma: float,
    rank_tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Exactly factor M = gamma*(B_f A_f - B_i A_i) into (B_d, A_d, ||M||_F).

    Returns fp32 factors with `B_d @ A_d == M / ||M||_F` (unit Frobenius norm),
    so the forward needs no subtraction and no rescaling.

    Done once, in fp32, at freeze time -- this is where the catastrophic
    cancellation is resolved. The dense d_out x d_in matrix is never formed:
    M is the rank-<=2r product Bc @ Ac with Bc = gamma*[B_f | -B_i] and
    Ac = [A_f ; A_i], so QR-ing both thin factors and SVD-ing the small
    2r x 2r core gives the exact thin SVD of M in O((d_out+d_in)(2r)^2).
    """
    Bc = torch.cat([B_f, -B_i], dim=1).to(torch.float32) * float(gamma)  # (out, 2r)
    Ac = torch.cat([A_f, A_i], dim=0).to(torch.float32)                  # (2r, in)

    Qb, Rb = torch.linalg.qr(Bc, mode="reduced")      # Bc = Qb Rb
    Qa, Ra = torch.linalg.qr(Ac.t(), mode="reduced")  # Ac^T = Qa Ra  =>  Ac = Ra^T Qa^T
    core = Rb @ Ra.t()                                # M = Qb @ core @ Qa^T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)

    norm = float(torch.linalg.vector_norm(S).item())
    if not (norm > _EPS):
        # Task learned (numerically) nothing; store an empty block.
        out_dim, in_dim = B_f.shape[0], A_f.shape[1]
        return (
            torch.zeros(out_dim, 0, dtype=torch.float32),
            torch.zeros(0, in_dim, dtype=torch.float32),
            0.0,
        )

    keep = S > (float(rank_tol) * S[0])
    U, S, Vh = U[:, keep], S[keep], Vh[keep, :]

    B_d = (Qb @ U) * (S / norm).unsqueeze(0)   # (out, k), columns scaled by S/||M||
    A_d = (Qa @ Vh.t()).t().contiguous()       # (k, in)
    return B_d.contiguous(), A_d, norm


def _sdlora_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Replacement forward bound onto each LoRA module (see module docstring).

    result = base(x)                                  # W0 (pristine)
           + gamma * (dB @ A_t + B^i @ dA)(dropout x)  # current net, cancellation-free
           + sum_{i<t} s_i * (B_d^i @ A_d^i) x         # frozen collapsed directions
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
    xc = dropout(x).to(lora_A.weight.dtype)

    # --- Current task, delta form: gamma*(dB @ A_t + B^i @ dA) -------------
    # Algebraically identical to gamma*(B_t A_t - B^i A^i) but every term is
    # proportional to a delta, so no large near-equal quantities are subtracted.
    # Skipped during the pre-init probe (has_current=False), where the current
    # task does not exist yet and must contribute exactly nothing.
    if state["has_current"]:
        A_t = lora_A.weight
        B_t = lora_B.weight
        dA = A_t - state["cur_init_A"]
        dB = B_t - state["cur_init_B"]
        cur = F.linear(F.linear(xc, A_t), dB) + F.linear(F.linear(xc, dA), state["cur_init_B"])
        result = result + (gamma * cur).to(rdtype)

    # --- Frozen tasks: ONE fused low-rank product for all of them ----------
    # sum_i s_i * (B_d^i @ A_d^i) x  ==  B_cat @ diag(w) @ A_cat @ x, with the
    # frozen factors pre-concatenated at install time and w the per-column
    # expansion of the trainable scalars. Written as a loop this allocated one
    # (tokens x out) temporary PER TASK -- all retained for backward, since the
    # scalars need gradients -- which is an OOM at Qwen widths. Fused, the only
    # large temporary is a single (tokens x out) output regardless of T; the
    # intermediate is just (tokens x sum_i k_i).
    scalars = state["scalars"]  # nn.Parameter (num_prev,) or None
    A_cat = state["prev_A_cat"]
    if scalars is not None and A_cat is not None:
        h = F.linear(xc, A_cat)                                   # (..., K)
        w = torch.repeat_interleave(scalars, state["prev_ks"])    # (K,)
        h = h * w.to(h.dtype)
        result = result + F.linear(h, state["prev_B_cat"]).to(rdtype)

    return result.to(rdtype)


class SDLoRAMethod(CLMethod):
    """SD-LoRA continual-learning method (composes with SLICE / any LoRA init)."""

    name = "sd_lora"

    # The base weights must stay pristine (no physical absorption, no merge);
    # train.py reads these flags to adjust the per-stage lifecycle.
    requires_skip_absorption = True
    requires_no_merge = True

    def __init__(self, rank_tol: float = 1e-6, block_fp32: bool = False, **kwargs: Any) -> None:
        """
        rank_tol: relative singular-value cutoff when collapsing a frozen block.
            Prunes numerically-negligible directions (lossless in practice) and
            lets the stored rank fall from 2r back toward r.
        block_fp32: keep frozen block factors in fp32 at forward time. Off by
            default -- after the fp32 collapse the factors are well-conditioned
            and unit-norm, so bf16 costs ~0.4% on that task's contribution with
            no cancellation amplification (same order as the base model itself).
        """
        super().__init__(rank_tol=rank_tol, block_fp32=block_fp32, **kwargs)
        self.rank_tol = float(rank_tol)
        self.block_fp32 = bool(block_fp32)
        # Per-module frozen blocks: name -> [ {"B_d","A_d","norm"}, ... ]
        self._blocks: Dict[str, List[Dict[str, Any]]] = {}
        # Per-module trainable magnitudes, length = #frozen tasks.
        self._scalars: Dict[str, List[float]] = {}
        self._scaling: Optional[float] = None  # rsLoRA gamma = alpha/sqrt(r)
        self._num_tasks: int = 0
        self._installed: List[str] = []

    # ------------------------------------------------------------------ hooks
    def before_init(self, lora_model, *, stage_idx, retain_tasks) -> None:
        """Install ONLY the frozen previous-task blocks, before the init probe.

        This is what puts SLICE's gradient probe at the model's real operating
        point W0 + sum_{i<t} s_i D_i. It matters most for the retain gradient:
        probed at a pristine W0 the model has not learned the previous tasks at
        all, so g_r would point toward *learning* them rather than describing
        where retained performance degrades -- making cos(g_c, g_r), and hence
        the whole conflict projection, measure the wrong thing.

        The current task contributes exactly nothing here (has_current=False).
        """
        self._install_all(lora_model, stage_idx=stage_idx, with_current=False)
        logger.info(
            "SD-LoRA pre-init: %d modules carry %d frozen task block(s); "
            "init probe will see W0 + sum s_i D_i",
            len(self._installed), max(0, int(stage_idx) - 1),
        )

    def pre_train(self, lora_model, *, stage_idx, retain_tasks) -> None:
        """Attach the current task and enable the trainable scalars.

        Runs AFTER the SLICE init has written the current-task A/B, so the live
        adapter weights here are exactly the SLICE init (B^i, A^i) used as the
        absorption reference and, later, in the net-update block.
        """
        self._install_all(lora_model, stage_idx=stage_idx, with_current=True)
        logger.info(
            "SD-LoRA installed on %d modules (stage=%d, frozen_tasks=%d, gamma=%.6g)",
            len(self._installed), int(stage_idx), int(stage_idx) - 1,
            self._scaling if self._scaling is not None else float("nan"),
        )

    def _install_all(self, lora_model, *, stage_idx: int, with_current: bool) -> None:
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

            self._install_module(
                module, prev_blocks, prev_scalars,
                device=device, with_current=with_current,
            )
            self._installed.append(name)

        self._num_tasks = int(stage_idx)

    def _install_module(
        self,
        module: torch.nn.Module,
        prev_blocks: List[Dict[str, Any]],
        prev_scalars: List[float],
        *,
        device: torch.device,
        with_current: bool,
    ) -> None:
        """Idempotent install.

        First call (from before_init) registers the frozen blocks and the scalar
        vector, with scalars frozen so the init probe doesn't build graph for
        them. The second call (from pre_train) snapshots the SLICE init as the
        absorption reference, switches the current task on, and unfreezes the
        scalars for training.
        """
        A_w = module.lora_A[_ADAPTER].weight
        dtype = A_w.dtype
        state = getattr(module, "_sdlora", None)

        if state is None:
            blk_dtype = torch.float32 if self.block_fp32 else dtype
            # Pre-concatenate the frozen factors once: A_cat (K, in), B_cat (out, K),
            # K = sum_i k_i. Lets the forward do a single fused matmul pair instead
            # of one per task (see _sdlora_forward).
            ks = [int(blk["A_d"].shape[0]) for blk in prev_blocks]
            A_cat_t = B_cat_t = None
            if prev_blocks and sum(ks) > 0:
                A_cat_t = torch.cat(
                    [blk["A_d"] for blk in prev_blocks if blk["A_d"].shape[0] > 0], dim=0
                ).to(device=device, dtype=blk_dtype).contiguous()
                B_cat_t = torch.cat(
                    [blk["B_d"] for blk in prev_blocks if blk["B_d"].shape[1] > 0], dim=1
                ).to(device=device, dtype=blk_dtype).contiguous()
                module.register_buffer("sdlora_prev_A_cat", A_cat_t, persistent=False)
                module.register_buffer("sdlora_prev_B_cat", B_cat_t, persistent=False)
                # repeat_interleave over per-task ranks expands the scalar vector to
                # one weight per concatenated column; zero-rank blocks drop out.
                module.register_buffer(
                    "sdlora_prev_ks",
                    torch.tensor(ks, dtype=torch.long, device=device),
                    persistent=False,
                )

            scalars: Optional[torch.nn.Parameter] = None
            if prev_scalars:
                scalars = torch.nn.Parameter(
                    torch.tensor(
                        [float(s) for s in prev_scalars], dtype=torch.float32, device=device
                    ),
                    requires_grad=False,
                )
                module.register_parameter("sdlora_scalars", scalars)

            state = {
                "cur_init_A": None,
                "cur_init_B": None,
                "has_current": False,
                "prev_A_cat": getattr(module, "sdlora_prev_A_cat", None),
                "prev_B_cat": getattr(module, "sdlora_prev_B_cat", None),
                "prev_ks": getattr(module, "sdlora_prev_ks", None),
                "scalars": scalars,
            }
            module._sdlora = state
            module.forward = types.MethodType(_sdlora_forward, module)

        if with_current and not state["has_current"]:
            B_w = module.lora_B[_ADAPTER].weight
            module.register_buffer(
                "sdlora_cur_init_A", A_w.detach().clone().to(device=device, dtype=dtype),
                persistent=False,
            )
            module.register_buffer(
                "sdlora_cur_init_B", B_w.detach().clone().to(device=device, dtype=dtype),
                persistent=False,
            )
            state["cur_init_A"] = getattr(module, "sdlora_cur_init_A")
            state["cur_init_B"] = getattr(module, "sdlora_cur_init_B")
            state["has_current"] = True
            if state["scalars"] is not None:
                state["scalars"].requires_grad_(True)

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
        """Collapse+freeze the current task's net block; record scalars."""
        gamma = float(self._scaling) if self._scaling is not None else 1.0
        ranks: List[int] = []
        for name, module in _iter_lora_linears(lora_model):
            state = getattr(module, "_sdlora", None)
            if state is None:
                continue
            A_f = module.lora_A[_ADAPTER].weight.detach()
            B_f = module.lora_B[_ADAPTER].weight.detach()
            A_i = state["cur_init_A"].detach()
            B_i = state["cur_init_B"].detach()

            B_d, A_d, norm = _collapse_net_update(
                B_f, A_f, B_i, A_i, gamma, rank_tol=self.rank_tol
            )
            ranks.append(int(A_d.shape[0]))

            # Store at the dtype the forward will actually use. Post-collapse the
            # factors are orthonormal and unit-norm, so the model dtype costs
            # ~0.4% on this task's contribution with no cancellation amplification
            # -- and it halves both the state files and host memory. fp32 is kept
            # only when the caller asked for fp32 blocks at forward time.
            store_dtype = torch.float32 if self.block_fp32 else A_f.dtype
            self._blocks.setdefault(name, []).append(
                {
                    "B_d": B_d.to(device="cpu", dtype=store_dtype),
                    "A_d": A_d.to(device="cpu", dtype=store_dtype),
                    "norm": float(norm),
                }
            )

            scalars_param = state["scalars"]
            if scalars_param is not None:
                self._scalars[name] = [float(s) for s in scalars_param.detach().cpu().tolist()]
            else:
                self._scalars.setdefault(name, [])
            # New task's magnitude scalar init = ||M_t||_F so s_t * D_t == M_t.
            self._scalars[name].append(float(norm))

        self._num_tasks = int(stage_idx)
        mean_rank = (sum(ranks) / len(ranks)) if ranks else 0.0
        logger.info(
            "SD-LoRA freeze: stage=%d task=%s modules=%d blocks/module=%d mean_block_rank=%.1f",
            int(stage_idx), task_name, len(self._blocks), self._num_tasks, mean_rank,
        )

    # ------------------------------------------------------------- persistence
    def _state_payload(self) -> Dict[str, Any]:
        return {
            "blocks": self._blocks,
            "scalars": self._scalars,
            "scaling": self._scaling,
            "num_tasks": self._num_tasks,
            "format": "collapsed_v1",
        }

    def save(self, state_dir: str) -> None:
        os.makedirs(state_dir, exist_ok=True)
        torch.save(self._state_payload(), os.path.join(state_dir, "sd_lora_state.pt"))

    def load(self, state_dir: str) -> None:
        path = os.path.join(state_dir, "sd_lora_state.pt")
        if not os.path.exists(path):
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        fmt = payload.get("format")
        if fmt != "collapsed_v1":
            raise ValueError(
                f"sd_lora state at {path} has format {fmt!r}, expected 'collapsed_v1'. "
                "States written before the fp32 net-update collapse are not compatible."
            )
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
            "rank_tol": float(self.rank_tol),
            "block_fp32": bool(self.block_fp32),
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
    """Add sum_i s_i * (B_d^i @ A_d^i) into each target weight, in fp32.

    `model` must be a pristine base model (no adapters). `state` is the payload
    saved by `SDLoRAMethod.save`. Returns the number of modules modified.
    """
    blocks: Dict[str, List[Dict[str, Any]]] = state.get("blocks", {})
    scalars: Dict[str, List[float]] = state.get("scalars", {})

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
            A_d = blk["A_d"].to(device=weight.device, dtype=torch.float32)
            if A_d.shape[0] == 0:
                continue
            B_d = blk["B_d"].to(device=weight.device, dtype=torch.float32)
            delta += float(s_vec[i]) * (B_d @ A_d)
        weight.data.copy_((weight.data.to(torch.float32) + delta).to(orig_dtype))
        modified += 1

    logger.info("sd_lora bake: modified %d/%d modules", modified, len(blocks))
    return modified
