from peft import LoraConfig, TaskType

def build_lora_config():
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_rslora=True,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        # target_modules="all_linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    return config

