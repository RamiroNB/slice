"""InfLoRA: Interference-Free LoRA via input-feature null-space projection.

Liang & Li, "InfLoRA: Interference-Free Low-Rank Adaptation for Continual
Learning" (CVPR 2024, arXiv:2404.00228).

Idea. Each LoRA target module sees an input feature distribution. Across all
previously-trained tasks we accumulate the input feature covariance
    C_m = sum_t E_{x ~ task_t}[ x x^T ]   (one (d_in, d_in) matrix per module m).

Before training task t, the freshly-initialized A_m is projected onto the
*approximate null space* of C_m:
    A_m <- A_m @ (I - U U^T)
where U = top-k principal directions of C_m (the subspace past-task inputs
actually live in). The new LoRA update then causes negligible change in the
model's response to past-task inputs while leaving capacity for the new task.

Choosing k (`nullspace_energy`, `nullspace_rank`). A fixed small k is not
meaningful across modules: d_in ranges from 2560 (q/k/v/gate/up) to 9728
(down_proj) on Qwen3-4B, and their covariance spectra differ accordingly, so the
same k protects wildly different fractions of the past-input energy. Following
Adam-NSCL / InfLoRA, k is therefore chosen *per module* from the spectrum: the
smallest k whose eigenvalues cover `nullspace_energy` of the accumulated
covariance trace, clamped to
    1 <= k_m <= min(nullspace_rank, d_in - r)
with r the LoRA rank read off A. The cap bounds cost (it is also the sketch's
rank, see below) and the `d_in - r` floor guarantees the projected A keeps at
least r free input directions, so the projection can never strip the new task's
capacity. Set `nullspace_energy` outside (0, 1) to fall back to a fixed
k_m = min(nullspace_rank, d_in - r) for every module.

After training each task we accumulate that task's input covariance with forward
hooks on the base linear inside every LoRA target module. The hook fires on the
still-LoRA-wrapped model *before* the merge, so the recorded features are the
ones the model will actually see once this task's adapter is folded into W.

Covariance representation (`cov_store`). The exact C_m is (d_in, d_in), which is
not affordable here: with the 7 target modules of `build_lora_config` on
Qwen3-4B, down_proj alone (d_in = 9728) costs 378 MB per layer and the whole
model's covariance is 20.8 GB -- held in RAM and re-serialized to cl_state after
every stage. Since the projection only ever consumes the top-`nullspace_rank`
subspace, the default `sketch` store keeps a factor F_m of that rank with
C_m ~= F_m^T F_m (a frequent-directions style sketch: subsample activation rows,
compress by truncated SVD, merge across stages). At the default cap of 256 that
is 980 MB for the same model, and it yields the same U up to the sketch error.
`cov_store="full"` materializes the exact dense covariance and is kept for small
models and for verifying the sketch.

Because the sketch only retains `nullspace_rank` directions, the energy
denominator cannot be read off the sketch itself. `_energy_total` therefore
accumulates the exact Frobenius energy of every row ever fed into a module's
sketch, so the truncation loss stays in the denominator and the energy rule
cannot be fooled into thinking a truncated spectrum is complete.

Merge-based, so `before_init` stays a no-op: previous tasks already live in the
base weights by the time the next stage's init runs.

Composes with any LoRA init (vanilla / loram / lora_ga / slice): init seeds
A/B, then InfLoRA's pre_train projection narrows A onto the safe subspace.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

try:
    from ..data import build_collator
except ImportError:  # script-style import (see train.py)
    from data import build_collator  # type: ignore[no-redef]

from .base import CLMethod

logger = logging.getLogger("cl_lora.cl_methods.inflora")

COV_STORES = ("sketch", "full")


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


def _compress_rows(
    F: torch.Tensor, k: int, *, device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Reduce a row-factor F (m, d_in) to (<=k, d_in) preserving F^T F's top-k.

    With F = U S V^T, F^T F = V S^2 V^T, so keeping (S_k V_k^T) keeps exactly the
    top-k eigenpairs of the covariance F represents and drops the rest. Returned
    on CPU in fp32; `device` only says where to run the SVD (the accelerator is
    ~100x faster here and the operands are a few hundred MB at most).
    """
    m, d_in = F.shape
    if m <= k:
        return F.to(device="cpu", dtype=torch.float32)
    work = F.to(device=(device or F.device), dtype=torch.float32)
    q = min(int(k) + 4, m, d_in)
    _U, S, V = torch.svd_lowrank(work, q=q, niter=4)
    kk = min(int(k), int(S.numel()))
    out = (V[:, :kk] * S[:kk]).t().contiguous()
    return out.to(device="cpu", dtype=torch.float32)


class InfLoRAMethod(CLMethod):
    """Null-space projection of LoRA-A using past-task input covariance."""

    name = "inflora"

    def __init__(
        self,
        *,
        nullspace_rank: int = 256,
        nullspace_energy: float = 0.99,
        max_cov_batches: int = 32,
        cov_batch_size: int = 8,
        cov_store: str = "sketch",
        cov_sample_rows: int = 32,
        max_seq_length: int = 1024,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            nullspace_rank=nullspace_rank,
            nullspace_energy=nullspace_energy,
            max_cov_batches=max_cov_batches,
            cov_batch_size=cov_batch_size,
            cov_store=cov_store,
            cov_sample_rows=cov_sample_rows,
            max_seq_length=max_seq_length,
            seed=seed,
            **kwargs,
        )
        store = str(cov_store).lower()
        if store not in COV_STORES:
            raise ValueError(f"InfLoRA: cov_store must be one of {COV_STORES}, got {cov_store!r}")
        # Cap on the projected-out subspace, and the sketch's rank.
        self.nullspace_rank = int(nullspace_rank)
        self.nullspace_energy = float(nullspace_energy)
        self.max_cov_batches = int(max_cov_batches)
        self.cov_batch_size = int(cov_batch_size)
        self.cov_store = store
        self.cov_sample_rows = int(cov_sample_rows)
        self.max_seq_length = int(max_seq_length)
        self.seed = int(seed)
        # Per-module accumulated past-task input covariance, CPU fp32 for stability.
        # cov_store="full":   (d_in, d_in) dense C_m.
        # cov_store="sketch": (m <= nullspace_rank, d_in) factor F_m, C_m ~= F_m^T F_m.
        self._covariance: Dict[str, torch.Tensor] = {}
        # Exact accumulated trace of C_m (sum of squared activation rows fed to the
        # sketch), which is what the energy rule divides by. Unused for cov_store
        # "full", where the dense C carries its own exact trace.
        self._energy_total: Dict[str, float] = {}
        self._num_stages_seen: int = 0

    # ---- pre_train: project A onto null space of past-task input covariance ----

    def pre_train(self, lora_model, *, stage_idx, retain_tasks) -> None:
        if not self._covariance:
            logger.info("InfLoRA pre_train: no past covariance yet (stage=%d), skipping projection", stage_idx)
            return

        device = next(lora_model.parameters()).device
        num_projected = 0
        num_skipped = 0
        removed_fracs: List[float] = []
        chosen_k: List[int] = []
        for name, mod in _iter_lora_modules(lora_model):
            A = mod.lora_A["default"].weight  # (r, d_in)
            rank, d_in = int(A.shape[0]), int(A.shape[1])
            U = self._projection_basis(name, d_in=d_in, rank=rank, device=device)
            if U is None:
                num_skipped += 1
                continue
            chosen_k.append(int(U.shape[1]))
            with torch.no_grad():
                A_f = A.detach().to(torch.float32)
                # A_new = A @ (I - U U^T) = A - (A @ U) @ U^T
                AU = A_f @ U
                A_proj = A_f - AU @ U.t()
                norm_before = float(A_f.norm())
                if norm_before > 0.0:
                    removed_fracs.append(1.0 - float(A_proj.norm()) / norm_before)
                A.data.copy_(A_proj.to(dtype=A.dtype))
            num_projected += 1
        mean_removed = (sum(removed_fracs) / len(removed_fracs)) if removed_fracs else 0.0
        logger.info(
            "InfLoRA pre_train: projected_modules=%d skipped=%d k=[%d..%d] mean_k=%.1f "
            "(cap=%d energy=%s) store=%s mean_A_norm_removed=%.4f (stage=%d)",
            num_projected, num_skipped,
            (min(chosen_k) if chosen_k else 0), (max(chosen_k) if chosen_k else 0),
            (sum(chosen_k) / len(chosen_k)) if chosen_k else 0.0,
            self.nullspace_rank, self.nullspace_energy, self.cov_store, mean_removed, stage_idx,
        )

    def _projection_basis(
        self, name: str, *, d_in: int, rank: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Principal directions of the accumulated covariance, as (d_in, k_m).

        k_m is chosen from the spectrum by the `nullspace_energy` rule, capped at
        `nullspace_rank` and at d_in - rank so A keeps >= rank free directions.
        """
        state = self._covariance.get(name)
        if state is None:
            return None
        if state.shape[-1] != d_in:
            logger.warning(
                "InfLoRA: covariance state shape %s does not match A.d_in=%d for %s; skipping",
                tuple(state.shape), d_in, name,
            )
            return None

        k_max = max(1, min(self.nullspace_rank, d_in - max(1, rank)))
        state_dev = state.to(device=device, dtype=torch.float32)

        if self.cov_store == "full":
            if state.shape[0] != d_in:
                logger.warning(
                    "InfLoRA: expected square covariance for %s, got %s; skipping",
                    name, tuple(state.shape),
                )
                return None
            # C is symmetric positive semi-definite, so its left singular vectors
            # are its eigenvectors and its singular values are its eigenvalues.
            try:
                U, S, _V = torch.svd_lowrank(state_dev, q=min(k_max + 4, d_in), niter=4)
            except Exception as exc:
                logger.warning("InfLoRA: svd_lowrank failed for %s (%s); skipping projection", name, exc)
                return None
            total = float(torch.diagonal(state_dev).sum())
            kk = self._select_rank(S, total_energy=total, k_max=k_max)
            return U[:, :kk].contiguous()

        # sketch: state is F with C ~= F^T F, so C's eigenvectors are F's right
        # singular vectors and its eigenvalues are F's squared singular values.
        m = int(state_dev.shape[0])
        k_max = min(k_max, m)
        try:
            _U, S, V = torch.svd_lowrank(state_dev, q=min(k_max + 4, m, d_in), niter=4)
        except Exception as exc:
            logger.warning("InfLoRA: svd_lowrank failed for %s (%s); skipping projection", name, exc)
            return None
        total = self._energy_total.get(name)
        if total is None:
            total = float(state_dev.pow(2).sum())
        kk = self._select_rank(S.pow(2), total_energy=total, k_max=k_max)
        return V[:, :kk].contiguous()

    def _select_rank(
        self, eigenvalues: torch.Tensor, *, total_energy: float, k_max: int,
    ) -> int:
        """Smallest k covering `nullspace_energy` of the covariance trace, <= k_max."""
        if not (0.0 < self.nullspace_energy < 1.0) or total_energy <= 0.0:
            return k_max
        target = self.nullspace_energy * total_energy
        csum = torch.cumsum(eigenvalues.to(torch.float64), dim=0)
        # First index whose cumulative energy reaches the target; if the retained
        # spectrum never gets there (sketch truncation dropped too much), spend the
        # whole cap rather than under-projecting. Counting entries below the target
        # keeps this device-agnostic -- searchsorted here would need the boundary as
        # a tensor on csum's device, which is the accelerator during training.
        idx = int((csum < target).sum().item())
        return max(1, min(idx + 1, k_max))

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
        # Hooks capture the input to the *base linear* under each LoRA module.
        # That input is also the input to A and is the right thing to put a
        # null-space mask against.
        rng = torch.Generator()
        rng.manual_seed(self.seed + int(stage_idx))

        dense: Dict[str, torch.Tensor] = {}          # cov_store="full"
        buffers: Dict[str, List[torch.Tensor]] = {}  # cov_store="sketch"
        buffered_rows: Dict[str, int] = {}
        stage_energy: Dict[str, float] = {}
        sample_counts: Dict[str, int] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        # Safety valve only: with the defaults the whole stage's sampled rows
        # (max_cov_batches * cov_sample_rows) fit under this, so the sketch pays a
        # single SVD per module at the end of the stage instead of one every few
        # batches. Unusual settings compress mid-stage rather than growing without
        # bound.
        compress_at = max(8 * self.nullspace_rank, 4 * self.cov_sample_rows)

        def make_hook(module_name: str):
            def _hook(module, inputs, output):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                with torch.no_grad():
                    x_flat = _flatten_input(x)
                    sample_counts[module_name] = sample_counts.get(module_name, 0) + x_flat.shape[0]
                    if self.cov_store == "full":
                        x32 = x_flat.to(torch.float32)
                        contrib = x32.t() @ x32  # (d_in, d_in)
                        prev = dense.get(module_name)
                        dense[module_name] = contrib if prev is None else prev + contrib
                        return
                    # sketch: keep a random row subsample, buffered on CPU in fp32
                    # so the buffer stays one dtype across mid-stage compressions.
                    x_flat = x_flat.to(torch.float32)
                    n = int(x_flat.shape[0])
                    if n > self.cov_sample_rows:
                        idx = torch.randperm(n, generator=rng)[: self.cov_sample_rows]
                        rows = x_flat[idx.to(x_flat.device)]
                    else:
                        rows = x_flat
                    # Exact energy of what enters the sketch: the denominator of the
                    # nullspace_energy rule, tracked before any truncation.
                    stage_energy[module_name] = (
                        stage_energy.get(module_name, 0.0) + float(rows.pow(2).sum())
                    )
                    buf = buffers.setdefault(module_name, [])
                    buf.append(rows.detach().to(device="cpu", dtype=torch.float32))
                    buffered_rows[module_name] = buffered_rows.get(module_name, 0) + int(rows.shape[0])
                    if buffered_rows[module_name] > compress_at:
                        merged = _compress_rows(
                            torch.cat(buf, dim=0), self.nullspace_rank, device=x.device,
                        )
                        buffers[module_name] = [merged]
                        buffered_rows[module_name] = int(merged.shape[0])
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

        # Shared collator: this loader replays the *training* batches to collect
        # input covariance, so it must pad them exactly as training does. (The
        # labels it emits are unused here — only activations are hooked.)
        collator = build_collator(tokenizer)
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

        # Fold this stage's contribution into the running state (CPU fp32).
        added = 0
        if self.cov_store == "full":
            for name, contrib in dense.items():
                cpu_contrib = contrib.detach().to(device="cpu", dtype=torch.float32)
                prev = self._covariance.get(name)
                self._covariance[name] = cpu_contrib if prev is None else prev + cpu_contrib
                added += 1
        else:
            for name, buf in buffers.items():
                if not buf:
                    continue
                stage_factor = torch.cat(buf, dim=0)
                prev = self._covariance.get(name)
                if prev is not None:
                    stage_factor = torch.cat([prev, stage_factor], dim=0)
                self._covariance[name] = _compress_rows(
                    stage_factor, self.nullspace_rank, device=device,
                )
                self._energy_total[name] = (
                    self._energy_total.get(name, 0.0) + stage_energy.get(name, 0.0)
                )
                added += 1
            buffers.clear()

        self._num_stages_seen += 1
        logger.info(
            "InfLoRA covariance updated: stage=%d task=%s store=%s modules=%d batches=%d "
            "rows_per_module=%d state_bytes=%.1fMB total_modules_in_state=%d",
            stage_idx, task_name, self.cov_store, added, steps,
            (max(sample_counts.values()) if sample_counts else 0),
            self._state_bytes() / 1e6, len(self._covariance),
        )

    def _state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._covariance.values())

    # ---- persistence ----

    def save(self, state_dir: str) -> None:
        os.makedirs(state_dir, exist_ok=True)
        payload = {
            "nullspace_rank": self.nullspace_rank,
            "nullspace_energy": self.nullspace_energy,
            "energy_total": dict(self._energy_total),
            "max_cov_batches": self.max_cov_batches,
            "cov_batch_size": self.cov_batch_size,
            "cov_store": self.cov_store,
            "cov_sample_rows": self.cov_sample_rows,
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
        self.nullspace_energy = float(payload.get("nullspace_energy", self.nullspace_energy))
        self._energy_total = dict(payload.get("energy_total", {}))
        self.max_cov_batches = int(payload.get("max_cov_batches", self.max_cov_batches))
        self.cov_batch_size = int(payload.get("cov_batch_size", self.cov_batch_size))
        self.cov_sample_rows = int(payload.get("cov_sample_rows", self.cov_sample_rows))
        self.max_seq_length = int(payload.get("max_seq_length", self.max_seq_length))
        self.seed = int(payload.get("seed", self.seed))
        self._num_stages_seen = int(payload.get("num_stages_seen", 0))
        self._covariance = dict(payload.get("covariance", {}))
        # The stored state's layout (dense C vs row factor F) decides how it can be
        # read back, so on resume it wins over the CLI flag.
        stored_store = str(payload.get("cov_store", self.cov_store)).lower()
        if stored_store in COV_STORES and stored_store != self.cov_store:
            logger.warning(
                "InfLoRA: resumed state was written with cov_store=%s but %s was requested; "
                "keeping %s to stay compatible with the saved covariance.",
                stored_store, self.cov_store, stored_store,
            )
        self.cov_store = stored_store if stored_store in COV_STORES else self.cov_store
        logger.info(
            "InfLoRA state loaded: stages_seen=%d cov_modules=%d nullspace_rank=%d store=%s",
            self._num_stages_seen, len(self._covariance), self.nullspace_rank, self.cov_store,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nullspace_rank": int(self.nullspace_rank),
            "nullspace_energy": float(self.nullspace_energy),
            "max_cov_batches": int(self.max_cov_batches),
            "cov_batch_size": int(self.cov_batch_size),
            "cov_store": self.cov_store,
            "cov_sample_rows": int(self.cov_sample_rows),
            "num_stages_seen": int(self._num_stages_seen),
            "cov_modules": len(self._covariance),
            "cov_state_mb": round(self._state_bytes() / 1e6, 3),
        }
