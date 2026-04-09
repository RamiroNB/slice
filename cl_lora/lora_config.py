from peft import LoraConfig, TaskType

def build_lora_config(r: int = 64, lora_alpha: int = 2, lora_dropout: float = 0.0):
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=True,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # target_modules="all_linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    return config

