from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import accelerate
import torch
from dotenv import load_dotenv
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import logging

try:
    from .cl_methods import CLMethod, VanillaCLMethod
    from .load_dataset import load_training_dataset
    from .lora_config import build_lora_config
    from .repro import set_global_seed
    from .slice import SliceInitConfig, initialize_lora_with_slice
except ImportError:
    from cl_methods import CLMethod, VanillaCLMethod  # type: ignore[no-redef]
    from load_dataset import load_training_dataset
    from lora_config import build_lora_config
    from repro import set_global_seed
    from slice import SliceInitConfig, initialize_lora_with_slice  # type: ignore[no-redef]


def _patch_accelerate_unwrap_model_compat() -> None:
    """Make older accelerate versions ignore keep_torch_compile from newer transformers."""
    unwrap = accelerate.Accelerator.unwrap_model
    params = inspect.signature(unwrap).parameters
    if "keep_torch_compile" in params:
        return

    @functools.wraps(unwrap)
    def _wrapped(self, model, *args, keep_torch_compile=None, **kwargs):
        return unwrap(self, model, *args, **kwargs)

    accelerate.Accelerator.unwrap_model = _wrapped


_patch_accelerate_unwrap_model_compat()

load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HUGGING_TOKEN")


def build_tokenizer(model_name: str = MODEL_NAME, hf_token: str | None = HF_TOKEN):
    local = Path(model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, local_files_only=local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str = MODEL_NAME,
    hf_token: str | None = HF_TOKEN,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    local = Path(model_name).is_dir()
    kwargs: dict = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
        local_files_only=local,
    )
    try:
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Report which attention implementation is actually in use.
    attn_used = getattr(model.config, "_attn_implementation", None)
    print(f"[load_base_model] attn_implementation={attn_used}")
    return model


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    return dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length),
        remove_columns=dataset.column_names,
    )


def _setup_peft_for_sapt(model, lora_cfg, adapter_name: str):
    """Prepare a PEFT model for SAPT stage training.

    If the input model is a fresh base, wrap it with a named adapter
    (`adapter_name`). If it is already a PEFT model from a previous SAPT
    stage, append a new named adapter and make only it trainable.
    """
    from peft import PeftModel, get_peft_model

    if isinstance(model, PeftModel):
        # Already wrapped: add the new adapter and activate only it.
        try:
            model.add_adapter(adapter_name, lora_cfg)
        except ValueError:
            # adapter_name already present (e.g. on resume) — recover by
            # selecting it; weights will be re-initialized by slice/init below.
            pass
        # Freeze all loaded adapters' params, then set the new one trainable.
        from peft.tuners.lora import Linear as LoraLinear

        for _, mod in model.named_modules():
            if not isinstance(mod, LoraLinear):
                continue
            for name in list(mod.lora_A.keys()):
                mod.lora_A[name].weight.requires_grad_(name == adapter_name)
            for name in list(mod.lora_B.keys()):
                mod.lora_B[name].weight.requires_grad_(name == adapter_name)
        model.set_adapter(adapter_name)
        return model, adapter_name

    # Fresh base: build a PEFT model with the adapter named explicitly.
    try:
        wrapped = get_peft_model(model, lora_cfg, adapter_name=adapter_name)
    except TypeError:
        # Older PEFT versions: get_peft_model has no adapter_name kwarg.
        # Create with default name then rename via add+delete.
        wrapped = get_peft_model(model, lora_cfg)
        if adapter_name != "default":
            wrapped.add_adapter(adapter_name, lora_cfg)
            wrapped.set_adapter(adapter_name)
            try:
                wrapped.delete_adapter("default")
            except Exception:
                pass
    return wrapped, adapter_name


class _CLAuxLossTrainer(Trainer):
    """Trainer that adds a CL-method auxiliary loss term on every step.

    The aux term is a scalar (or None). When None, behaves identically to the
    base Trainer. The aux term is left in fp32 and added to the LM loss; the
    backward is unchanged.
    """

    def __init__(self, *args, _cl_aux_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cl_aux_loss_fn = _cl_aux_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        loss, outputs = result
        if self._cl_aux_loss_fn is not None:
            aux = self._cl_aux_loss_fn(model)
            if aux is not None:
                loss = loss + aux.to(dtype=loss.dtype, device=loss.device)
        return (loss, outputs) if return_outputs else loss


def _apply_init_absorption(peft_model, init_correction: dict) -> None:
    """Replay the slice-init weight absorption step during model reconstruction.

    During slice init, frozen base weights are modified in-place:
      W_frozen -= B_init @ A_init * scale
    This keeps model output unchanged at t=0. When reconstructing a model from
    the saved adapter (which only stores the trained B/A), this step must be
    replayed so the reconstructed weights match the true merged weights.
    """
    from peft.tuners.lora import Linear as LoraLinear

    named_modules = dict(peft_model.named_modules())
    for module_name, ab in init_correction.items():
        module = named_modules.get(module_name)
        if module is None or not isinstance(module, LoraLinear):
            continue
        if not hasattr(module, "get_base_layer"):
            continue
        base_layer = module.get_base_layer()
        base_weight = getattr(base_layer, "weight", None)
        if not isinstance(base_weight, torch.nn.Parameter):
            continue

        scaling = float(module.scaling["default"])
        B = ab["B"].to(device=base_weight.device, dtype=torch.float32)
        A = ab["A"].to(device=base_weight.device, dtype=torch.float32)
        offset = (B @ A) * scaling

        orig_dtype = base_weight.dtype
        base_weight.data.copy_(
            (base_weight.data.to(torch.float32) - offset).to(orig_dtype)
        )


def load_sapt_model(
    base_model_path: str,
    adapter_paths: List[str],
    router_path: str,
    *,
    hf_token: str | None = HF_TOKEN,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """Load base + every stage adapter as parallel named adapters and wrap in SAPT.

    Returns a `SAPTWrapper` ready for `evaluate_all` (drop-in for an
    `nn.Module` exposing `forward`/`generate`). Adapter naming convention
    is "task_NN" for the i-th path (1-based) — matching what
    `SAPTMethod.adapter_name_for_stage` produces during training.
    """
    from peft import PeftModel

    from .sapt import SAPTRouter, SAPTWrapper

    base = load_base_model(
        base_model_path, hf_token=hf_token, torch_dtype=torch_dtype, device_map=device_map
    )
    if not adapter_paths:
        raise ValueError("load_sapt_model requires at least one adapter path.")

    adapter_names: List[str] = []
    peft_model = None
    for i, ap in enumerate(adapter_paths):
        name = f"task_{i + 1:02d}"
        adapter_names.append(name)
        # PEFT saves named adapters to {path}/{adapter_name}/ when using
        # selected_adapters. Resolve that subdirectory when present.
        named_subdir = Path(ap) / name
        if named_subdir.is_dir() and (named_subdir / "adapter_config.json").exists():
            ap = str(named_subdir)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, ap, adapter_name=name)
        else:
            peft_model.load_adapter(ap, adapter_name=name)

    router = SAPTRouter.load_from_path(router_path)
    return SAPTWrapper(peft_model, router, adapter_names)


def load_model_with_adapters(
    base_model_path: str,
    adapter_paths: List[str],
    hf_token: str | None = HF_TOKEN,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """Load base model and apply LoRA adapters sequentially, merging each one.

    Reconstructs the cumulative model state for stage k by loading the base
    model and merging adapter_1, adapter_2, ..., adapter_k in order.
    If an adapter was trained with slice init, replays the weight absorption
    step before merging so the result is numerically identical to the
    in-memory merged model produced during training.
    """
    from peft import PeftModel

    model = load_base_model(
        base_model_path, hf_token=hf_token, torch_dtype=torch_dtype, device_map=device_map
    )
    for adapter_path in adapter_paths:
        model = PeftModel.from_pretrained(model, adapter_path)
        init_correction_path = Path(adapter_path) / "init_correction.pt"
        if init_correction_path.exists():
            init_correction = torch.load(str(init_correction_path), map_location="cpu", weights_only=True)
            _apply_init_absorption(model, init_correction)
        model = model.merge_and_unload()
    return model


def _compute_lora_ba_norms(lora_model, lora_alpha: float, rank: int, *, use_rslora: bool = True) -> Dict[str, Any]:
    """Walk PEFT model and compute ||B @ A||_F per LoRA layer.

    Returns raw and effective (rsLoRA-scaled) norms plus aggregates.
    Effective ΔW per layer = (alpha / sqrt(r)) * B @ A under rsLoRA, or
    (alpha / r) * B @ A under classic LoRA.
    """
    import math
    raw_norms: Dict[str, float] = {}
    eff_norms: Dict[str, float] = {}
    scale = lora_alpha / (math.sqrt(rank) if use_rslora else rank)
    with torch.no_grad():
        for name, module in lora_model.named_modules():
            la = getattr(module, "lora_A", None)
            lb = getattr(module, "lora_B", None)
            if la is None or lb is None:
                continue
            if not (hasattr(la, "items") or hasattr(la, "keys")):
                continue
            for adapter_name in list(la.keys()):
                A = la[adapter_name].weight  # (r, in)
                B = lb[adapter_name].weight  # (out, r)
                # ||B @ A||_F as float
                ba_f = torch.linalg.matrix_norm(B.float() @ A.float(), ord="fro").item()
                key = f"{name}::{adapter_name}"
                raw_norms[key] = ba_f
                eff_norms[key] = scale * ba_f
    if not raw_norms:
        return {
            "raw_per_layer": {},
            "effective_per_layer": {},
            "scale": float(scale),
            "num_layers": 0,
        }
    raw_vals = list(raw_norms.values())
    eff_vals = list(eff_norms.values())
    return {
        "raw_per_layer": raw_norms,
        "effective_per_layer": eff_norms,
        "scale": float(scale),
        "num_layers": len(raw_norms),
        "raw_mean": float(sum(raw_vals) / len(raw_vals)),
        "raw_max": float(max(raw_vals)),
        "raw_min": float(min(raw_vals)),
        "raw_total": float(sum(raw_vals)),
        "effective_mean": float(sum(eff_vals) / len(eff_vals)),
        "effective_max": float(max(eff_vals)),
        "effective_min": float(min(eff_vals)),
        "effective_total": float(sum(eff_vals)),
    }


def train_on_task(
    model,
    tokenizer,
    task,
    output_dir: str,
    retain_tasks=None,
    rank: int = 64,
    lora_alpha: int = 2,
    learning_rate: float = 1e-4,
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: int = 500,
    max_seq_length: int = 256,
    eval_size: int = 200,
    seed: int = 42,
    use_bf16: bool = True,
    save_adapter: bool = True,
    save_intermediate_checkpoints: bool = False,
    adapter_checkpoint_path: str | None = None,
    slice_enabled: bool = False,
    slice_cache_dir: str = "slice_cache",
    slice_max_steps: int = 100,
    slice_retain_scale: float = 1.0,
    slice_grad_project: bool = False,
    slice_grad_projection_mode: str = "per_module",
    slice_grad_project_always: bool = False,
    slice_add_retain_grad: bool = False,
    slice_cache_context: str | None = None,
    slice_retain_batch_size: int | None = None,
    slice_retain_grad_accum: int | None = None,
    slice_retain_batch_size_set: str = "all_tasks",
    slice_single_retain_task_mode: bool = False,
    slice_init_method: str = "slice",
    slice_projection_method: str = "pcgrad",
    slice_cosine_threshold: float | None = None,
    slice_per_layer_threshold: bool = False,
    slice_per_layer_threshold_delta: float = 0.0,
    slice_cagrad_c: float = 0.5,
    slice_gradvac_phi: float = 0.0,
    slice_gradvac_beta: float = 0.5,
    slice_magnitude_preserve: bool = False,
    slice_nullspace_rank: int = 8,
    slice_nullspace_sv_threshold: float = 0.0,
    slice_svd_selection: str = "lora_ga",
    cl_method: CLMethod | None = None,
    stage_idx: int = 1,
    sapt_mode: bool = False,
    sapt_adapter_name: str | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a fresh LoRA adapter on one task.

    In standard mode (sapt_mode=False) the adapter is named ``"default"``,
    trained, then merged into base via ``merge_and_unload``; the returned
    object is the merged ``nn.Module``.

    In SAPT mode (sapt_mode=True) the input model is a PEFT model that may
    already carry adapters from previous stages. A new adapter named
    ``sapt_adapter_name`` is added, only it is trained, and the function
    returns the *un-merged* PEFT model with all adapters live in parallel.

    Returns:
        (model_after_stage, training_report)
    """
    set_global_seed(seed)
    train_dataset, eval_dataset = load_training_dataset(task=task, eval_size=eval_size, seed=seed)
    train_dataset = _tokenize_dataset(train_dataset, tokenizer=tokenizer, max_length=max_seq_length)
    eval_dataset = _tokenize_dataset(eval_dataset, tokenizer=tokenizer, max_length=max_seq_length)

    lora_cfg = build_lora_config(r=rank, lora_alpha=lora_alpha)
    if sapt_mode:
        if not sapt_adapter_name:
            raise ValueError("sapt_mode=True requires a non-empty sapt_adapter_name.")
        lora_model, active_adapter = _setup_peft_for_sapt(model, lora_cfg, sapt_adapter_name)
    else:
        lora_model = get_peft_model(model, lora_cfg)
        active_adapter = "default"
    lora_model.print_trainable_parameters()

    if slice_enabled:
        model_id = (
            getattr(getattr(model, "config", None), "_name_or_path", None)
            or getattr(model, "name_or_path", None)
        )
        slice_config = SliceInitConfig(
            cache_dir=slice_cache_dir,
            cache_context=slice_cache_context or (str(model_id) if model_id else None),
            max_steps=slice_max_steps,
            per_device_batch_size=per_device_train_batch_size,
            seed=seed,
            retain_scale=slice_retain_scale,
            grad_project=slice_grad_project,
            grad_projection_mode=slice_grad_projection_mode,
            grad_project_always=slice_grad_project_always,
            add_retain_grad=slice_add_retain_grad,
            rank=rank,
            max_seq_length=max_seq_length,
            retain_batch_size=slice_retain_batch_size,
            retain_grad_accum=slice_retain_grad_accum,
            retain_batch_size_set=slice_retain_batch_size_set,
            single_retain_task_mode=slice_single_retain_task_mode,
            init_method=slice_init_method,
            projection_method=slice_projection_method,
            cosine_threshold=slice_cosine_threshold,
            per_layer_threshold=slice_per_layer_threshold,
            per_layer_threshold_delta=slice_per_layer_threshold_delta,
            cagrad_c=slice_cagrad_c,
            gradvac_phi=slice_gradvac_phi,
            gradvac_beta=slice_gradvac_beta,
            magnitude_preserve=slice_magnitude_preserve,
            nullspace_rank=slice_nullspace_rank,
            nullspace_sv_threshold=slice_nullspace_sv_threshold,
            svd_selection=slice_svd_selection,
            skip_absorption=bool(sapt_mode),
        )
        # propagate PEFT lora settings into slice config when available
        try:
            setattr(slice_config, "lora_alpha", float(getattr(lora_cfg, "lora_alpha", 1.0)))
        except Exception:
            pass
        logger = logging.getLogger("cl_lora.train.slice")
        num_written = initialize_lora_with_slice(
            model=lora_model,
            tokenizer=tokenizer,
            current_task=task,
            retain_tasks=retain_tasks,
            config=slice_config,
            adapter_name=active_adapter,
        )
        logger.info(
            "Slice init applied: num_modules_written=%d adapter=%s skip_absorption=%s",
            int(num_written), active_adapter, bool(sapt_mode),
        )

        # Capture A/B at init time so load_model_with_adapters can replay
        # the absorption step. Skipped under SAPT — there is no absorption
        # (skip_absorption=True), so no replay is needed at eval time.
        from peft.tuners.lora import Linear as LoraLinear
        if sapt_mode:
            lora_init_correction: dict = {}
        else:
            lora_init_correction = {
                name: {
                    "A": mod.lora_A[active_adapter].weight.detach().cpu().clone(),
                    "B": mod.lora_B[active_adapter].weight.detach().cpu().clone(),
                }
                for name, mod in lora_model.named_modules()
                if isinstance(mod, LoraLinear) and active_adapter in getattr(mod, "lora_A", {})
            }
    else:
        lora_init_correction = {}

    # CL-method pre-training hook (fires AFTER init_correction capture so
    # absorption replay at eval time still reflects the actual A_init/B_init
    # used to modify the base weights). Methods like InfLoRA mutate A here;
    # methods like O-LoRA/vanilla are no-ops.
    cl_method = cl_method or VanillaCLMethod()
    cl_method.pre_train(
        lora_model,
        stage_idx=int(stage_idx),
        retain_tasks=retain_tasks,
    )

    ba_norms_init = _compute_lora_ba_norms(lora_model, lora_alpha=float(lora_alpha), rank=int(rank))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_strategy="steps" if save_intermediate_checkpoints else "no",
        save_steps=save_steps,
        warmup_ratio=0.01,
        save_total_limit=2 if save_intermediate_checkpoints else None,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=use_bf16,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=True,
        seed=seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = _CLAuxLossTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        _cl_aux_loss_fn=cl_method.aux_loss,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    ba_norms_final = _compute_lora_ba_norms(lora_model, lora_alpha=float(lora_alpha), rank=int(rank))

    # CL-method post-training hook (e.g. snapshot O-LoRA A's, accumulate
    # InfLoRA covariance). Runs on the still-LoRA-wrapped model BEFORE merge.
    try:
        cl_device = next(lora_model.parameters()).device
    except StopIteration:
        cl_device = torch.device("cpu")
    cl_method.post_train(
        lora_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        device=cl_device,
        stage_idx=int(stage_idx),
        task_name=getattr(task, "name", str(task)),
    )

    save_kwargs: Dict[str, Any] = {}
    if sapt_mode:
        # Save only the just-trained adapter so the directory is
        # round-trippable with `model.load_adapter(path, adapter_name=...)`.
        save_kwargs["selected_adapters"] = [active_adapter]
    if save_adapter:
        lora_model.save_pretrained(str(output_path / "adapter"), **save_kwargs)
        if lora_init_correction:
            torch.save(lora_init_correction, output_path / "adapter" / "init_correction.pt")

    if adapter_checkpoint_path:
        adapter_cp = Path(adapter_checkpoint_path)
        adapter_cp.mkdir(parents=True, exist_ok=True)
        lora_model.save_pretrained(str(adapter_cp), **save_kwargs)
        if lora_init_correction:
            torch.save(lora_init_correction, adapter_cp / "init_correction.pt")

    if sapt_mode:
        # SAPT: do NOT merge — keep parallel adapters live for routing.
        post_stage_model = lora_model
    else:
        merge_fn = getattr(lora_model, "merge_and_unload", None)
        post_stage_model = merge_fn() if callable(merge_fn) else lora_model
    trainer.save_state()

    def _maybe_to_dict(obj: Any) -> Any:
        if obj is None:
            return None
        if is_dataclass(obj):
            return asdict(obj)
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict()
            except Exception:
                return str(obj)
        return str(obj)

    model_cfg = getattr(model, "config", None)
    model_name_or_path = getattr(model_cfg, "_name_or_path", None) if model_cfg is not None else None
    if model_name_or_path is None:
        model_name_or_path = getattr(model, "name_or_path", None)

    def _to_serializable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): _to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_serializable(v) for v in value]
        return str(value)

    report = {
        "task_name": getattr(task, "name", str(task)),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "ba_norms": {
            "init": ba_norms_init,
            "final": ba_norms_final,
            "lora_alpha": float(lora_alpha),
            "rank": int(rank),
            "use_rslora": True,
        },
        "output_dir": str(output_path),
        "configs": {
            "seed": int(seed),
            "model": {
                "class": model.__class__.__name__,
                "name_or_path": model_name_or_path,
                "config": _maybe_to_dict(model_cfg),
            },
            "tokenizer": {
                "class": tokenizer.__class__.__name__,
                "name_or_path": getattr(tokenizer, "name_or_path", None),
                "pad_token": getattr(tokenizer, "pad_token", None),
                "eos_token": getattr(tokenizer, "eos_token", None),
                "padding_side": getattr(tokenizer, "padding_side", None),
            },
            "lora": _maybe_to_dict(lora_cfg),
            "slice": _maybe_to_dict(slice_config) if slice_enabled else None,
            "cl_method": cl_method.metadata(),
            "training_args": _maybe_to_dict(training_args),
        },
    }
    with open(output_path / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(report), f, indent=2)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return post_stage_model, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA on one task and merge into base model.")
    parser.add_argument(
        "--task",
        default="task363_sst2_polarity_classification",
        help="Task name (e.g., task363_sst2_polarity_classification or NI363).",
    )
    parser.add_argument("--retain-task", default=None, help="Optional retain task for slice init.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default="outputs/single_task")
    parser.add_argument("--save-merged-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for reproducibility.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help="LoRA rank (also used for slice init when --slice-init is enabled).",
    )
    parser.add_argument("--slice-init", action="store_true", help="Enable slice LoRA init.")
    parser.add_argument("--slice-cache-dir", default="slice_cache")
    parser.add_argument("--slice-max-steps", type=int, default=100)
    parser.add_argument("--slice-retain-scale", type=float, default=1.0)
    parser.add_argument("--slice-grad-project", action="store_true", help="Project current-task gradients against retain gradients for slice init.")
    parser.add_argument(
        "--slice-grad-projection-mode",
        choices=["per_module", "global"],
        default="per_module",
        help="Projection mode when --slice-grad-project is enabled.",
    )
    parser.add_argument(
        "--slice-grad-project-always",
        action="store_true",
        help="Use OGD-style projection: always remove retain-gradient component (no conflict gating).",
    )
    parser.add_argument(
        "--slice-add-retain-grad",
        action="store_true",
        help="Add retain gradient after projection when --slice-grad-project is enabled.",
    )
    parser.add_argument("--slice-retain-batch-size", type=int, default=None,
        help="Batch size for retain gradient computation. Defaults to training batch size.")
    parser.add_argument("--slice-retain-grad-accum", type=int, default=None,
        help="Max accumulation steps for retain gradient. Defaults to --slice-max-steps.")
    parser.add_argument("--slice-retain-batch-size-set", choices=["all_tasks", "each_task"],
        default="all_tasks",
        help="How retain batch size is applied: 'all_tasks' = total across all tasks, 'each_task' = per task.")
    parser.add_argument("--slice-single-retain-task-mode", action="store_true",
        help="Only use the most recent previous task for retain, with same batch size as current task.")
    parser.add_argument("--slice-init-method", choices=["slice", "lora_ga", "loram"],
        default="slice",
        help="Initialization method: 'slice' (default), 'lora_ga' (SVD on current-task gradients only), "
             "or 'loram' (DST-based, no gradients).")
    args = parser.parse_args()

    set_global_seed(args.seed)

    tokenizer = build_tokenizer(model_name=args.model_name, hf_token=HF_TOKEN)
    model = load_base_model(model_name=args.model_name, hf_token=HF_TOKEN)

    retain_tasks = [args.retain_task] if args.retain_task else None
    merged_model, report = train_on_task(
        model=model,
        tokenizer=tokenizer,
        task=args.task,
        output_dir=args.output_dir,
        retain_tasks=retain_tasks,
        seed=args.seed,
        rank=args.rank,
        slice_enabled=args.slice_init,
        slice_cache_dir=args.slice_cache_dir,
        slice_max_steps=args.slice_max_steps,
        slice_retain_scale=args.slice_retain_scale,
        slice_grad_project=args.slice_grad_project,
        slice_grad_projection_mode=args.slice_grad_projection_mode,
        slice_grad_project_always=args.slice_grad_project_always,
        slice_add_retain_grad=args.slice_add_retain_grad,
        slice_retain_batch_size=args.slice_retain_batch_size,
        slice_retain_grad_accum=args.slice_retain_grad_accum,
        slice_retain_batch_size_set=args.slice_retain_batch_size_set,
        slice_single_retain_task_mode=args.slice_single_retain_task_mode,
        slice_init_method=args.slice_init_method,
    )

    if args.save_merged_model:
        merged_dir = Path(args.output_dir) / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
