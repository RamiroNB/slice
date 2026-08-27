from .apply import apply_slice_inits
from .compute import compute_loram_inits, compute_slice_inits, load_or_compute_slice_inits
from .config import SliceInitConfig, assert_slice_flags_require_slice_init

from ..repro import set_global_seed


def initialize_lora_with_slice(
    model,
    tokenizer,
    current_task,
    retain_tasks,
    *,
    config: SliceInitConfig,
    adapter_name: str = "default",
) -> tuple[int, dict | None]:
    """Apply slice init. Returns (num_modules_written, projection_summary).

    `projection_summary` is a compact per-stage record of the gradient projection
    (gamma / rel_change / fired / conflict counts) suitable for storing in the run's
    stage report so it survives cache pruning. May be None (e.g. cache hit whose
    stats file was pruned, or non-projecting init methods).
    """
    set_global_seed(int(config.seed))
    inits, _cache_root, projection_summary = load_or_compute_slice_inits(
        model=model,
        tokenizer=tokenizer,
        current_task=current_task,
        retain_tasks=retain_tasks,
        config=config,
    )

    lora_alpha = getattr(config, "lora_alpha", 1.0)
    r_val = getattr(config, "rank", None)
    skip_abs = bool(getattr(config, "skip_absorption", False))
    num_written = apply_slice_inits(
        model,
        inits,
        lora_alpha=lora_alpha,
        r=r_val,
        skip_absorption=skip_abs,
        adapter_name=adapter_name,
    )
    return num_written, projection_summary


__all__ = [
    "SliceInitConfig",
    "apply_slice_inits",
    "assert_slice_flags_require_slice_init",
    "compute_loram_inits",
    "compute_slice_inits",
    "initialize_lora_with_slice",
    "load_or_compute_slice_inits",
]
