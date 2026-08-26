from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

try:
    from ..data import build_collator, tokenize_dataset as _tokenize_dataset
except ImportError:  # script-style import (see train.py)
    from data import build_collator, tokenize_dataset as _tokenize_dataset  # type: ignore[no-redef]


def tokenize_dataset(dataset, tokenizer, max_length: int, *, completion_only: bool = True,
                     log_context: str = ""):
    return _tokenize_dataset(
        dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        completion_only=completion_only,
        log_context=log_context,
    )


def model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def build_dataloader(dataset, tokenizer, batch_size: int, seed: int) -> DataLoader:
    collator = build_collator(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=4,
        pin_memory=True,
    )


def target_weight_params(model: torch.nn.Module, target_modules: Iterable[str]) -> Dict[str, torch.nn.Parameter]:
    target_modules = list(target_modules)
    out: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        if not name.endswith(".weight"):
            continue
        if not any(tgt in name for tgt in target_modules):
            continue
        module_name = name[: -len(".weight")]
        out[module_name] = param
    return out
