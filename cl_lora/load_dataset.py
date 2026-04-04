from __future__ import annotations

import json
import re
from typing import Any, Tuple, Union

import requests
from datasets import Dataset, load_dataset

try:
    from .task_sequences import SuperNITask, TraceTask, all_superni_tasks
except ImportError:
    from task_sequences import SuperNITask, TraceTask, all_superni_tasks


SUPERNI_RAW_TEMPLATE = (
    "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/{task_name}.json"
)


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_to_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _resolve_superni_task_name(task_name: str) -> str:
    if task_name.startswith("task"):
        return task_name

    if re.fullmatch(r"NI\d+", task_name):
        for task in all_superni_tasks():
            if task.ni_id == task_name:
                return task.name

    raise ValueError(
        "Could not resolve SuperNI task name. "
        f"Received '{task_name}'. Expected a full task name like "
        "'task363_sst2_polarity_classification' or an NI id like 'NI363'."
    )


def _split_dataset(dataset: Dataset, eval_size: int, seed: int) -> Tuple[Dataset, Dataset]:
    dataset = dataset.shuffle(seed=seed)
    if len(dataset) <= 1:
        return dataset, dataset

    test_size = min(eval_size, len(dataset) - 1)
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def _build_chat_prompt(instruction_text: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction_text}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _build_chat_text(prompt_text: str, output_text: str) -> str:
    return f"{prompt_text}{output_text}<|eot_id|>"


def _format_superni_instance(example: dict, definition: str) -> dict:
    input_text = _to_text(example.get("input", ""))

    output = example.get("output", "")
    output_text = _to_text(output[0] if isinstance(output, list) and output else output)

    instruction = f"{definition}\n\nInput: {input_text}"
    prompt = _build_chat_prompt(instruction)
    text = _build_chat_text(prompt_text=prompt, output_text=output_text)
    return {
        "text": text,
        "prompt": prompt,
        "target": output_text,
    }


def _pick_trace_fields(example: dict) -> Tuple[str, str]:
    instruction_keys = ["instruction", "prompt", "question", "query", "context"]
    output_keys = ["output", "answer", "response", "label", "target", "completion"]

    instruction = ""
    for key in instruction_keys:
        if key in example and example[key] is not None:
            instruction = _to_text(example[key])
            break

    if not instruction:
        # Fall back to the first non-output field.
        for key, value in example.items():
            if key not in output_keys and value is not None:
                instruction = f"{key}: {_to_text(value)}"
                break

    output = ""
    for key in output_keys:
        if key in example and example[key] is not None:
            output = _to_text(example[key])
            break

    if not output:
        output = "N/A"

    return instruction, output


def _format_trace_instance(example: dict, task_name: str) -> dict:
    instruction, output = _pick_trace_fields(example)
    prompt = _build_chat_prompt(f"TRACE task: {task_name}\n\n{instruction}")
    text = _build_chat_text(prompt_text=prompt, output_text=output)
    return {
        "text": text,
        "prompt": prompt,
        "target": output,
    }


def _load_trace_hf_dataset(task_name: str, hf_dataset: str | None) -> Dataset:
    candidates = []
    if hf_dataset:
        candidates.append((hf_dataset, None))
    candidates.extend(
        [
            ("BeyonderXX/TRACE", task_name),
            ("BeyonderXX/TRACE", task_name.lower()),
        ]
    )

    last_error = None
    for path, config in candidates:
        try:
            ds = load_dataset(path, name=config)
            if isinstance(ds, Dataset):
                return ds
            if "train" in ds:
                return ds["train"]
            first_split = next(iter(ds.keys()))
            return ds[first_split]
        except Exception as exc:  # noqa: BLE001 - keep trying candidates
            last_error = exc

    raise RuntimeError(
        "Unable to load TRACE dataset from known HuggingFace candidates for "
        f"task '{task_name}'."
    ) from last_error


def load_superni_training_dataset(
    task_name: str,
    eval_size: int = 200,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    resolved_name = _resolve_superni_task_name(task_name)
    url = SUPERNI_RAW_TEMPLATE.format(task_name=resolved_name)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    instances = data.get("Instances", [])
    if not instances:
        raise ValueError(f"No instances found for SuperNI task '{resolved_name}'.")

    definition = data.get("Definition", "")
    if isinstance(definition, list):
        definition = definition[0] if definition else "Complete the task."
    definition = _to_text(definition) if definition else "Complete the task."

    dataset = Dataset.from_list(instances)
    train_raw, eval_raw = _split_dataset(dataset=dataset, eval_size=eval_size, seed=seed)

    train_dataset = train_raw.map(
        lambda ex: _format_superni_instance(ex, definition=definition),
        remove_columns=train_raw.column_names,
    )
    eval_dataset = eval_raw.map(
        lambda ex: _format_superni_instance(ex, definition=definition),
        remove_columns=eval_raw.column_names,
    )
    return train_dataset, eval_dataset


def load_trace_training_dataset(
    task_name: str,
    hf_dataset: str | None = None,
    eval_size: int = 200,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    dataset = _load_trace_hf_dataset(task_name=task_name, hf_dataset=hf_dataset)
    train_raw, eval_raw = _split_dataset(dataset=dataset, eval_size=eval_size, seed=seed)

    train_dataset = train_raw.map(
        lambda ex: _format_trace_instance(ex, task_name=task_name),
        remove_columns=train_raw.column_names,
    )
    eval_dataset = eval_raw.map(
        lambda ex: _format_trace_instance(ex, task_name=task_name),
        remove_columns=eval_raw.column_names,
    )
    return train_dataset, eval_dataset


def load_training_dataset(
    task: Union[str, SuperNITask, TraceTask],
    eval_size: int = 200,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Unified loader for SuperNI and TRACE tasks.

    Args:
        task: SuperNI task name/NI id, SuperNITask dataclass, TRACE task name,
            or TraceTask dataclass.
    """
    if hasattr(task, "ni_id"):
        return load_superni_training_dataset(task.name, eval_size=eval_size, seed=seed)

    if hasattr(task, "language") and hasattr(task, "metric"):
        return load_trace_training_dataset(
            task_name=task.name,
            hf_dataset=getattr(task, "hf_dataset", None),
            eval_size=eval_size,
            seed=seed,
        )

    task_name = str(task)
    if task_name.startswith("task") or re.fullmatch(r"NI\d+", task_name):
        return load_superni_training_dataset(task_name, eval_size=eval_size, seed=seed)
    return load_trace_training_dataset(task_name=task_name, eval_size=eval_size, seed=seed)
