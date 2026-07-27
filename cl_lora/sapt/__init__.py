"""SAPT runtime: shared-attention router + multi-adapter forward."""
from .arm import generate_pseudo_samples_for_task, train_router_arm
from .router import SAPTRouter
from .runtime import (
    SAPTWrapper,
    install_lora_forward_patch,
    iter_lora_modules,
    list_lora_adapter_names,
    sapt_routing,
    uninstall_lora_forward_patch,
)


__all__ = [
    "SAPTRouter",
    "SAPTWrapper",
    "generate_pseudo_samples_for_task",
    "install_lora_forward_patch",
    "iter_lora_modules",
    "list_lora_adapter_names",
    "sapt_routing",
    "train_router_arm",
    "uninstall_lora_forward_patch",
]
