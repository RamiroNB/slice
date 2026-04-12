from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

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
    from .load_dataset import load_training_dataset
    from .lora_config import build_lora_config
    from .repro import set_global_seed
    from .slice import SliceInitConfig, initialize_lora_with_slice
except ImportError:
    from load_dataset import load_training_dataset
    from lora_config import build_lora_config
    from repro import set_global_seed
    from slice import SliceInitConfig, initialize_lora_with_slice


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
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
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
    kwargs: dict = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )
    try:
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    return dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length),
        remove_columns=dataset.column_names,
    )


def train_on_task(
    model,
    tokenizer,
    task,
    output_dir: str,
    retain_tasks=None,
    rank: int = 64,
    learning_rate: float = 1e-4,
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 8,
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
    slice_enabled: bool = False,
    slice_cache_dir: str = "slice_cache",
    slice_max_steps: int = 100,
    slice_retain_scale: float = 1.0,
    slice_grad_project: bool = False,
    slice_grad_projection_mode: str = "per_module",
    slice_add_retain_grad: bool = False,
    slice_cache_context: str | None = None,
    slice_retain_batch_size: int | None = None,
    slice_retain_grad_accum: int | None = None,
    slice_retain_batch_size_set: str = "all_tasks",
    slice_single_retain_task_mode: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a fresh LoRA adapter on one task, then merge it into the model.

    Returns:
        (merged_model, training_report)
    """
    set_global_seed(seed)
    train_dataset, eval_dataset = load_training_dataset(task=task, eval_size=eval_size, seed=seed)
    train_dataset = _tokenize_dataset(train_dataset, tokenizer=tokenizer, max_length=max_seq_length)
    eval_dataset = _tokenize_dataset(eval_dataset, tokenizer=tokenizer, max_length=max_seq_length)

    lora_cfg = build_lora_config(r=rank)
    lora_model = get_peft_model(model, lora_cfg)
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
            add_retain_grad=slice_add_retain_grad,
            rank=rank,
            max_seq_length=max_seq_length,
            retain_batch_size=slice_retain_batch_size,
            retain_grad_accum=slice_retain_grad_accum,
            retain_batch_size_set=slice_retain_batch_size_set,
            single_retain_task_mode=slice_single_retain_task_mode,
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
            forget_task=task,
            retain_tasks=retain_tasks,
            config=slice_config,
        )
        logger.info("Slice init applied: num_modules_written=%d", int(num_written))

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
        save_steps=save_steps,
        warmup_ratio=0.01,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=use_bf16,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=True,
        seed=seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    if save_adapter:
        lora_model.save_pretrained(str(output_path / "adapter"))

    merge_fn = getattr(lora_model, "merge_and_unload", None)
    merged_model = merge_fn() if callable(merge_fn) else lora_model
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
            "training_args": _maybe_to_dict(training_args),
        },
    }
    with open(output_path / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(report), f, indent=2)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_model, report


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
    parser.add_argument("--slice-grad-project", action="store_true", help="Project forget gradients against retain gradients for slice init.")
    parser.add_argument(
        "--slice-grad-projection-mode",
        choices=["per_module", "global"],
        default="per_module",
        help="Projection mode when --slice-grad-project is enabled.",
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
        help="Only use the most recent previous task for retain, with same batch size as forget.")
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
        slice_add_retain_grad=args.slice_add_retain_grad,
        slice_retain_batch_size=args.slice_retain_batch_size,
        slice_retain_grad_accum=args.slice_retain_grad_accum,
        slice_retain_batch_size_set=args.slice_retain_batch_size_set,
        slice_single_retain_task_mode=args.slice_single_retain_task_mode,
    )

    if args.save_merged_model:
        merged_dir = Path(args.output_dir) / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()