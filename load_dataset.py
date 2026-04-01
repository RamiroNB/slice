from datasets import get_dataset_config_names, load_dataset
import json
import requests
from datasets import Dataset
# One SuperNI task used in the paper (sentiment classification, task 363)
# This is from the natural-instructions collection on HuggingFace
SUPERNI_TASK = "task363_sst2_polarity_classification"

def format_instruction(example):
    """Format as instruction-response pair for causal LM training."""
    instruction = example.get("definition", "Classify the sentiment.")
    input_text = example["input"]
    output_text = example["output"][0] if isinstance(example["output"], list) else example["output"]

    # Standard Llama chat template format
    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}\n\nInput: {input_text}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output_text}<|eot_id|>"
    )
    return {"text": text}


def load_training_dataset():

    url = f"https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/{SUPERNI_TASK}.json"

    data = requests.get(url).json()

    instances = data["Instances"]

    dataset = Dataset.from_list(instances)

    train_dataset = dataset.map(format_instruction)
    eval_dataset = dataset.select(range(min(200, len(dataset)))).map(format_instruction)

    return train_dataset, eval_dataset
