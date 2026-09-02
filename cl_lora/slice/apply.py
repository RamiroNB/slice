from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger("cl_lora.slice.apply")


def apply_slice_inits(
    peft_model: torch.nn.Module,
    inits: Dict[str, Dict[str, torch.Tensor]],
    *,
    lora_alpha: float = 2.0,
    r: Optional[int] = None,
    decomposition: Optional[str] = None,
    skip_absorption: Optional[bool] = None,
    adapter_name: str = "default",
) -> int:
    """Apply slice inits to a PEFT LoRA model with in-place absorption.

    `skip_absorption` (when not None) overrides the decomposition-based
    default. `adapter_name` selects which named LoRA adapter on each module
    receives the init (default `"default"`).
    """
    from peft.tuners.lora import Linear as LoraLinear

    def _tensor_mean_var(t: torch.Tensor) -> tuple[float, float]:
        t32 = t.detach().to(dtype=torch.float32)
        mean_val = float(t32.mean().item())
        var_val = float(t32.var(unbiased=False).item())
        return mean_val, var_val

    def _normalize_name(name: str) -> str:
        out = str(name)
        out = out.replace(".weight", "")
        out = out.replace(".base_layer", "")
        return out

    def _resolve_target_module(init_key: str, index: Dict[str, str]) -> Optional[str]:
        nk = _normalize_name(init_key)
        exact = index.get(nk)
        if exact is not None:
            return exact

        candidates = [
            real_name
            for norm_name, real_name in index.items()
            if norm_name.endswith(nk) or nk.endswith(norm_name)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    if skip_absorption is None:
        skip_absorption = decomposition in {
            "right_singular_vectors",
            "right_singular_vectors_kaiming",
            "right_svd_kaiming_random_basis",
        }
    else:
        skip_absorption = bool(skip_absorption)

    if r is None or int(r) <= 0:
        raise RuntimeError("slice apply requires a valid LoRA rank `r` for absorption.")

    named_modules = dict(peft_model.named_modules())
    logger.info("Applying slice inits with in-place absorption: candidate_modules=%d", len(inits))

    lora_index: Dict[str, str] = {}
    lora_names: list[str] = []
    for module_name, mod in named_modules.items():
        if not isinstance(mod, LoraLinear):
            continue
        normalized = _normalize_name(module_name)
        lora_index[normalized] = module_name
        lora_names.append(module_name)

    sample_init_keys = list(inits.keys())[:5]
    sample_lora_names = lora_names[:5]
    logger.debug("Slice sample init keys: %s", sample_init_keys)
    logger.debug("Slice sample LoRA modules: %s", sample_lora_names)

    if not inits:
        raise RuntimeError(
            "slice apply received an empty init dict: every LoRA module would keep "
            "PEFT's vanilla init (A=kaiming, B=0), producing a vanilla run under a "
            "slice/lora_ga/loram label."
        )

    num_written = 0
    unmatched: list[str] = []
    written_modules: set[str] = set()
    for init_key, ab in inits.items():
        target_name = _resolve_target_module(init_key, lora_index)
        if target_name is None:
            unmatched.append(init_key)
            continue

        logger.debug("slice map: init_key=%s -> peft_module=%s", init_key, target_name)
        module = named_modules[target_name]
        if adapter_name not in getattr(module, "lora_A", {}):
            raise RuntimeError(
                f"Adapter {adapter_name!r} not present on LoRA module {init_key} "
                f"(have {list(getattr(module, 'lora_A', {}).keys())}). Skipping it would "
                "leave that module at PEFT's vanilla init."
            )
        A_tgt = module.lora_A[adapter_name].weight
        B_tgt = module.lora_B[adapter_name].weight

        if A_tgt.shape != ab["A"].shape or B_tgt.shape != ab["B"].shape:
            raise RuntimeError(
                f"Slice init shape mismatch for layer {init_key}: "
                f"A_tgt.shape={A_tgt.shape}, A_init.shape={ab['A'].shape}, "
                f"B_tgt.shape={B_tgt.shape}, B_init.shape={ab['B'].shape}"
            )

        with torch.no_grad():
            A_tgt.copy_(ab["A"].to(device=A_tgt.device, dtype=A_tgt.dtype))
            B_tgt.copy_(ab["B"].to(device=B_tgt.device, dtype=B_tgt.dtype))

            # B=0 (with A kaiming) is exactly PEFT's vanilla init, so an all-zero
            # factor after the copy means this layer trains as vanilla LoRA no
            # matter what the run is labelled. Catches an all-zero gradient slice
            # and a dtype underflow in the cast above.
            if not bool(A_tgt.any()) or not bool(B_tgt.any()):
                raise RuntimeError(
                    f"Slice init wrote an all-zero factor for layer {init_key} "
                    f"(A_nonzero={bool(A_tgt.any())}, B_nonzero={bool(B_tgt.any())}); "
                    "B=0 is PEFT's vanilla init."
                )

            if logger.isEnabledFor(logging.DEBUG):
                a_mean, a_var = _tensor_mean_var(A_tgt)
                b_mean, b_var = _tensor_mean_var(B_tgt)
                logger.debug(
                    "[slice] final layer stats: layer=%s A(mean=%.6g,var=%.6g,shape=%s,dtype=%s) "
                    "B(mean=%.6g,var=%.6g,shape=%s,dtype=%s)",
                    init_key,
                    a_mean,
                    a_var,
                    tuple(A_tgt.shape),
                    str(A_tgt.dtype).replace("torch.", ""),
                    b_mean,
                    b_var,
                    tuple(B_tgt.shape),
                    str(B_tgt.dtype).replace("torch.", ""),
                )

            if not skip_absorption:
                base_layer = None
                if hasattr(module, "get_base_layer"):
                    base_layer = module.get_base_layer()
                if base_layer is None:
                    raise RuntimeError(
                        f"Cannot get base layer for LoRA module {init_key}. "
                        "Absorption requires access to the base layer weight."
                    )
                base_weight = getattr(base_layer, "weight", None)
                if not isinstance(base_weight, torch.nn.Parameter):
                    raise RuntimeError(
                        f"base_layer.weight is not a Parameter for {init_key} "
                        f"(got {type(base_weight)}). Cannot perform absorption."
                    )

                scaling_val = float(module.scaling[adapter_name])
                orig_dtype = base_weight.dtype

                weight_orig32 = base_weight.data.to(torch.float32).clone()
                B_mat = B_tgt.to(dtype=torch.float32)
                A_mat = A_tgt.to(dtype=torch.float32)
                offset32 = (B_mat @ A_mat) * float(scaling_val)

                weight32 = weight_orig32 - offset32
                base_weight.data.copy_(weight32.to(orig_dtype))

                weight_stored32 = base_weight.data.to(torch.float32)
                recon32 = weight_stored32 + offset32
                diff = (recon32 - weight_orig32).abs().max()
                tol = 1e-3 if orig_dtype in (torch.float16, torch.bfloat16) else 1e-7
                if diff > tol:
                    logger.warning(
                        "[slice] LoRA absorption diff for layer %s: max|W_frozen+DeltaW-W_orig|=%.3e (tol=%.3e)",
                        init_key,
                        diff.item(),
                        tol,
                    )
                else:
                    logger.info(
                        "[slice] LoRA absorption OK for layer %s: max|W_frozen+DeltaW-W_orig|=%.3e (tol=%.3e)",
                        init_key,
                        diff.item(),
                        tol,
                    )
            else:
                logger.info(
                    "[slice] Skipping LoRA absorption for layer %s due to decomposition='%s'",
                    init_key,
                    decomposition,
                )

        num_written += 1
        written_modules.add(target_name)

    if unmatched:
        raise RuntimeError(
            f"Slice init could not be mapped onto a PEFT LoRA module for "
            f"{len(unmatched)} of {len(inits)} init keys (e.g. {unmatched[:5]}); "
            f"sample LoRA module names: {lora_names[:5]}. Those modules would keep "
            "their vanilla init."
        )

    # Reverse direction: a LoRA module the init never reached keeps A=kaiming/B=0,
    # i.e. trains as vanilla LoRA inside a run labelled slice/lora_ga/loram. Only
    # reachable if the init's target-module set diverges from the LoRA config's.
    expected = {
        name for name in lora_names
        if adapter_name in getattr(named_modules[name], "lora_A", {})
    }
    missed = sorted(expected - written_modules)
    if missed:
        raise RuntimeError(
            f"{len(missed)} of {len(expected)} LoRA modules carrying adapter "
            f"{adapter_name!r} received no slice init (e.g. {missed[:5]}); they would "
            "train from PEFT's vanilla init."
        )
    if num_written == 0:
        raise RuntimeError(
            "Slice init wrote 0 LoRA modules; the whole run would be vanilla LoRA."
        )

    logger.info(
        "Applied slice A/B to %d/%d LoRA modules (adapter=%s); no vanilla factors left",
        num_written, len(expected), adapter_name,
    )
    return num_written
