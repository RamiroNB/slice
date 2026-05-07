from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Tuple, Union

import requests
from datasets import Dataset

logger = logging.getLogger("cl_lora.load_dataset")

try:
    from .task_sequences import SuperNITask, TraceTask, all_superni_tasks
except ImportError:
    from task_sequences import SuperNITask, TraceTask, all_superni_tasks


_DATASET_CACHE_DIR = Path(os.environ.get(
    "CL_LORA_DATASET_CACHE",
    # os.path.expanduser("~/.cache/cl_lora/datasets"),
    str(Path(__file__).resolve().parents[1] / "datasets_cache"),
))


def _cached_fetch_json(url: str, timeout: int = 30) -> Any:
    """Fetch JSON from *url* with local disk caching.

    Cached files are stored under ``$CL_LORA_DATASET_CACHE``
    (default ``~/.cache/cl_lora/datasets``).  Set the env-var to
    change the location.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", url.rsplit("/", 1)[-1])[:80]
    cache_path = _DATASET_CACHE_DIR / f"{safe_name}_{url_hash}.json"

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Dataset cache hit: %s", cache_path)
            return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted dataset cache entry %s — re-downloading", cache_path)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True)
    logger.debug("Dataset cached: %s -> %s", url, cache_path)

    return data


SUPERNI_RAW_TEMPLATE = (
    "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/{task_name}.json"
)

TRACE_RAW_TEMPLATE = (
    "https://raw.githubusercontent.com/BeyonderXX/TRACE/master/data/{task_folder}/{split}.json"
)

TRACE_FOLDER_MAP = {
    "C-STANCE": "C-STANCE",
    "FOMC": "FOMC",
    "MeetingBank": "MeetingBank",
    "Py150": "Py150",
    "ScienceQA": "ScienceQA",
    "NumGLUE-cm": "NumGLUE-cm",
}

TRACE_BENCHMARK_DIRS = [
    "LLM-CL-Benchmark_1000",
    "LLM-CL-Benchmark_500",
    "LLM-CL-Benchmark_5000",
    "LLM-CL-Benchmark_Reasoning",
]

DEFAULT_TRACE_ROOT = "/mnt/C-SSD/user/data/TRACE-Benchmark"


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


def _load_trace_local(task_name: str, split: str = "train") -> Dataset | None:
    folder = TRACE_FOLDER_MAP.get(task_name)
    if folder is None:
        raise ValueError(
            f"Unknown TRACE task '{task_name}'. "
            f"Known tasks: {list(TRACE_FOLDER_MAP.keys())}"
        )

    env_candidates = [
        os.getenv("TRACE_DATA_DIR"),
        os.getenv("TRACE_DATA_ROOT"),
    ]
    root_candidates = [
        Path(p) for p in env_candidates if p
    ] + [
        Path(DEFAULT_TRACE_ROOT),
    ]

    split_filename = f"{split}.json"
    file_candidates: list[Path] = []
    for root in root_candidates:
        if not root.exists():
            continue

        # Candidate already points to a benchmark directory.
        file_candidates.append(root / folder / split_filename)

        # Candidate points to TRACE root containing benchmark subdirectories.
        for bench_dir in TRACE_BENCHMARK_DIRS:
            file_candidates.append(root / bench_dir / folder / split_filename)

    for file_path in file_candidates:
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                raise ValueError(
                    f"Unexpected TRACE JSON format from {file_path}; expected a list of records."
                )
            return Dataset.from_list(records)

    return None


def _load_trace_raw(task_name: str, split: str = "train") -> Dataset:
    folder = TRACE_FOLDER_MAP.get(task_name)
    if folder is None:
        raise ValueError(
            f"Unknown TRACE task '{task_name}'. "
            f"Known tasks: {list(TRACE_FOLDER_MAP.keys())}"
        )

    url = TRACE_RAW_TEMPLATE.format(task_folder=folder, split=split)
    try:
        records = _cached_fetch_json(url)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise RuntimeError(
                f"TRACE file not found at {url}. If TRACE raw data files are not committed "
                "to GitHub for this split, clone TRACE data locally and load from local files."
            ) from exc
        raise
    if not isinstance(records, list):
        raise ValueError(f"Unexpected TRACE JSON format from {url}; expected a list of records.")

    return Dataset.from_list(records)


def load_superni_training_dataset(
    task_name: str,
    eval_size: int = 200,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    resolved_name = _resolve_superni_task_name(task_name)
    url = SUPERNI_RAW_TEMPLATE.format(task_name=resolved_name)
    data = _cached_fetch_json(url)

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
    # hf_dataset kept for compatibility but TRACE now loads from local TRACE benchmarks
    # first, then falls back to raw GitHub if needed.
    _ = hf_dataset
    dataset = _load_trace_local(task_name=task_name, split="train")
    if dataset is None:
        dataset = _load_trace_raw(task_name=task_name, split="train")
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
    if isinstance(task, SuperNITask):
        return load_superni_training_dataset(task.name, eval_size=eval_size, seed=seed)

    if isinstance(task, TraceTask):
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
