import os
import inspect
import functools
from dotenv import load_dotenv

import torch
import accelerate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import get_peft_model
from lora_config import build_lora_config
from load_dataset import load_training_dataset


def _patch_accelerate_unwrap_model_compat():
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
HF_TOKEN = os.getenv("HUGGING_TOKEN")


def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        token=HF_TOKEN
    )

    # Apply LoRA
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_training_dataset()

    def tokenize(example):
        text = example["text"]
        return tokenizer(
            text,
            truncation=True,
            # padding="max_length",
            max_length=256
        )

    train_dataset = train_dataset.map(
        tokenize,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        tokenize,
        remove_columns=eval_dataset.column_names
    )

    # training_args = TrainingArguments(
    #     output_dir="./outputs",
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #      gradient_accumulation_steps=2,
    #     # learning_rate=2e-4,
    #      learning_rate=1e-4,
    #     num_train_epochs=3,
    #     logging_steps=50,
    #     save_steps=500,
    #     eval_strategy="steps",
    #     eval_steps=500,
    #     # fp16=True,
    #      bf16=True,
    #     report_to="none",
    #     dataset_text_field="text",
    #     max_seq_length=512,
    # )
    training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    bf16=True,
    report_to="none",
    remove_unused_columns=True,
)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained("/mnt/C-SSD/ramiro/adapters/llama-3.2-3b-instruct-lora")


if __name__ == "__main__":
    main()