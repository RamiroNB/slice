from .apply import apply_slice_inits
from .compute import compute_loram_inits, compute_slice_inits, load_or_compute_slice_inits
from .config import SliceInitConfig

from ..repro import set_global_seed


def initialize_lora_with_slice(
    model,
    tokenizer,
    forget_task,
    retain_tasks,
    *,
    config: SliceInitConfig,
    adapter_name: str = "default",
) -> int:
    set_global_seed(int(config.seed))
    inits = load_or_compute_slice_inits(
        model=model,
        tokenizer=tokenizer,
        forget_task=forget_task,
        retain_tasks=retain_tasks,
        config=config,
    )

    lora_alpha = getattr(config, "lora_alpha", 1.0)
    r_val = getattr(config, "rank", None)
    skip_abs = bool(getattr(config, "skip_absorption", False))
    return apply_slice_inits(
        model,
        inits,
        lora_alpha=lora_alpha,
        r=r_val,
        skip_absorption=skip_abs,
        adapter_name=adapter_name,
    )


__all__ = [
    "SliceInitConfig",
    "apply_slice_inits",
    "compute_loram_inits",
    "compute_slice_inits",
    "initialize_lora_with_slice",
    "load_or_compute_slice_inits",
]
