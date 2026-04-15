from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
import pickle
from typing import Any, Tuple, Union

import requests
from datasets import Dataset

logger = logging.getLogger("cl_lora.load_dataset")

FAIRNESS_TASK_ALIASES = {
    "bbq": "bbq",
    "fairness_bbq": "bbq",
    "difference_awareness": "difference_awareness",
    "fairness_difference_awareness": "difference_awareness",
    "winobias": "winobias",
    "wino_bias": "winobias",
    "fairness_winobias": "winobias",
    # Legacy aliases kept for backwards-compat (WinoGender was replaced by WinoBias).
    "winogender": "winobias",
    "wino_gender": "winobias",
    "fairness_winogender": "winobias",
}

# WinoBias (Zhao et al., 2018) — 8 files split by pro/anti-stereotyped × type1/type2 × dev/test.
# Each line: "N [The entity] ... [pronoun] ..." with two bracket spans.
_WINOBIAS_RAW_BASE = (
    "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data"
)
WINOBIAS_FILES = {
    "pro_stereo_type1_dev":  "pro_stereotyped_type1.txt.dev",
    "pro_stereo_type1_test": "pro_stereotyped_type1.txt.test",
    "pro_stereo_type2_dev":  "pro_stereotyped_type2.txt.dev",
    "pro_stereo_type2_test": "pro_stereotyped_type2.txt.test",
    "anti_stereo_type1_dev":  "anti_stereotyped_type1.txt.dev",
    "anti_stereo_type1_test": "anti_stereotyped_type1.txt.test",
    "anti_stereo_type2_dev":  "anti_stereotyped_type2.txt.dev",
    "anti_stereo_type2_test": "anti_stereotyped_type2.txt.test",
}

DIFF_AWARENESS_RAW_BASE = (
    "https://raw.githubusercontent.com/Angelina-Wang/difference_awareness/main/benchmark_suite"
)

DIFF_AWARENESS_FILES = [
    "D1_1k.pkl",
    "D2_1k.pkl",
    "D3_1k.pkl",
    "D4_1k.pkl",
    "N1_1k.pkl",
    "N2_1k.pkl",
    "N3_1k.pkl",
    "N4_1k.pkl",
]

BBQ_RAW_BASE = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data"
BBQ_CATEGORY_FILES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
    # Intersectional categories may exist in some mirrors.
    "Race_x_gender",
    "Race_x_SES",
]
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


def _cached_fetch_bytes(url: str, timeout: int = 30) -> bytes:
    """Fetch bytes from *url* with local disk caching."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", url.rsplit("/", 1)[-1])[:80]
    cache_path = _DATASET_CACHE_DIR / f"{safe_name}_{url_hash}.bin"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return f.read()
        except OSError:
            logger.warning("Corrupted dataset cache entry %s — re-downloading", cache_path)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.content

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(payload)
    logger.debug("Dataset cached: %s -> %s", url, cache_path)
    return payload


def _parse_jsonl_bytes(payload: bytes, source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    text = payload.decode("utf-8")
    for line_idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSONL line %d in %s: %s", line_idx, source_name, exc)
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


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

DEFAULT_TRACE_ROOT = "/mnt/C-SSD/ramiro/data/TRACE-Benchmark"


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


def _canonical_fairness_task_name(task_name: str) -> str | None:
    return FAIRNESS_TASK_ALIASES.get(task_name.strip().lower())


def _coerce_choice_label(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        mapping = {0: "A", 1: "B", 2: "C"}
        return mapping.get(value)

    text = _to_text(value).strip().lower()
    if not text:
        return None

    if text in {"a", "b", "c"}:
        return text.upper()

    if text in {"0", "1", "2"}:
        return {"0": "A", "1": "B", "2": "C"}[text]

    match = re.search(r"\b([abc])\b", text)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([012])\b", text)
    if match:
        return {"0": "A", "1": "B", "2": "C"}[match.group(1)]

    return None


def _extract_group_id_from_uid(uid: Any, fallback: str) -> str:
    uid_text = _to_text(uid).strip()
    if not uid_text:
        return fallback

    if "-" in uid_text:
        prefix = uid_text.split("-", 1)[0].strip()
        if prefix:
            return prefix
    return uid_text


def _format_fairness_record(
    *,
    task_name: str,
    question: str,
    options: list[str] | None,
    target_label: str,
    group_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if options:
        options_text = "\n".join(
            f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options) if opt
        )
        instruction = (
            f"Fairness task: {task_name}\n\n"
            f"{question}\n\n"
            f"Options:\n{options_text}\n\n"
            "Answer with only one option letter."
        )
    else:
        # Options are already embedded in the question text (e.g. difference_awareness).
        instruction = (
            f"Fairness task: {task_name}\n\n"
            f"{question}\n\n"
            "Answer with only one option letter (A for option a, B for option b, C for option c)."
        )
    prompt = _build_chat_prompt(instruction)
    text = _build_chat_text(prompt_text=prompt, output_text=target_label)
    return {
        "text": text,
        "prompt": prompt,
        "target": target_label,
        "group_id": group_id,
        "label_id": {"A": 0, "B": 1, "C": 2}.get(target_label),
        "source_dataset": task_name,
        "metadata": metadata or {},
    }


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


def _load_bbq_records() -> list[dict[str, Any]]:
    raw_examples: list[dict[str, Any]] = []

    local_root = os.getenv("BBQ_DATA_DIR") or os.getenv("BBQ_DIR")
    if local_root:
        root = Path(local_root)
        candidate_dirs = [root, root / "data", root / "BBQ_full" / "data"]
        local_files: list[Path] = []
        for candidate in candidate_dirs:
            if candidate.is_dir():
                local_files.extend(sorted(candidate.glob("*.jsonl")))
            elif candidate.is_file() and candidate.suffix == ".jsonl":
                local_files.append(candidate)

        for path in local_files:
            try:
                blob = path.read_bytes()
                raw_examples.extend(_parse_jsonl_bytes(blob, str(path)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping local BBQ file %s due to error: %s", path, exc)

    if not raw_examples:
        for category in BBQ_CATEGORY_FILES:
            url = f"{BBQ_RAW_BASE}/{category}.jsonl"
            try:
                blob = _cached_fetch_bytes(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping remote BBQ category %s due to error: %s", category, exc)
                continue
            raw_examples.extend(_parse_jsonl_bytes(blob, url))

    if not raw_examples:
        raise RuntimeError(
            "Could not load BBQ records from local files or GitHub raw JSONL. "
            "Tip: set BBQ_DATA_DIR to a BBQ data folder containing *.jsonl files."
        )

    records: list[dict[str, Any]] = []
    for example in raw_examples:
        context = _to_text(example.get("context", "")).strip()
        question = _to_text(example.get("question", "")).strip()
        ans0 = _to_text(example.get("ans0", "")).strip()
        ans1 = _to_text(example.get("ans1", "")).strip()
        ans2 = _to_text(example.get("ans2", "")).strip()
        label = _coerce_choice_label(example.get("label"))
        if not label:
            label = _coerce_choice_label(example.get("target"))
        if not label:
            continue

        additional_metadata = example.get("additional_metadata", {})
        stereotype_groups = None
        if isinstance(additional_metadata, dict):
            stereotype_groups = additional_metadata.get("stereotyped_groups")

        group_id = _to_text(example.get("category", "")).strip() or "bbq"
        if stereotype_groups:
            if isinstance(stereotype_groups, list) and stereotype_groups:
                group_id = f"{group_id}:{_to_text(stereotype_groups[0]).strip()}"
            elif isinstance(stereotype_groups, str):
                group_id = f"{group_id}:{stereotype_groups.strip()}"

        question_text = f"Context: {context}\nQuestion: {question}" if context else question
        # answer_info maps ans0/ans1/ans2 → [text, demographic_label].
        # The demographic_label is matched against stereotyped_groups to identify
        # which answer position represents the stereotyped choice (for sBBQ scoring).
        answer_info = example.get("answer_info", {})
        records.append(
            _format_fairness_record(
                task_name="bbq",
                question=question_text,
                options=[ans0, ans1, ans2],
                target_label=label,
                group_id=group_id,
                metadata={
                    "context_condition": example.get("context_condition"),
                    "question_polarity": example.get("question_polarity"),
                    "example_id": example.get("example_id"),
                    "category": example.get("category"),
                    "answer_info": answer_info,
                    "stereotyped_groups": stereotype_groups or [],
                },
            )
        )

    if not records:
        raise RuntimeError("Loaded BBQ dataset but produced zero normalized records.")
    return records


def _load_winobias_records() -> list[dict[str, Any]]:
    """Load WinoBias (Zhao et al., 2018) pronoun coreference dataset.

    Task: given a sentence with two entities and a pronoun, choose which entity
    the pronoun refers to.  The fairness signal is the accuracy gap between
    pro-stereotyped examples (pronoun gender matches occupational stereotype)
    and anti-stereotyped examples (pronoun gender opposes stereotype).

    Format: each line is "N [The entity1] ... [pronoun] ..." (or variant).
    Two bracket spans appear in every sentence:
      - The first ``[...]`` span is the coreference target entity.
      - The second ``[...]`` span contains the pronoun.
    File naming encodes pro/anti-stereotyped and type1/type2.

    Source: https://github.com/uclanlp/corefBias
    """
    local_root = os.getenv("WINOBIAS_DATA_DIR")
    records: list[dict[str, Any]] = []

    # Regex to find all [bracketed spans] in a line.
    bracket_re = re.compile(r"\[([^\]]+)\]")

    for split_key, filename in WINOBIAS_FILES.items():
        blob: bytes | None = None

        if local_root:
            local_path = Path(local_root) / filename
            if local_path.exists():
                blob = local_path.read_bytes()

        if blob is None:
            url = f"{_WINOBIAS_RAW_BASE}/{filename}"
            try:
                blob = _cached_fetch_bytes(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping WinoBias file %s due to error: %s", filename, exc)
                continue

        is_pro = split_key.startswith("pro_")
        group_id = "pro_stereo" if is_pro else "anti_stereo"

        for raw_line in blob.decode("utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            # Strip leading line number if present.
            line_text = re.sub(r"^\d+\s+", "", raw_line)

            spans = bracket_re.findall(line_text)
            if len(spans) < 2:
                logger.warning("WinoBias line missing two bracket spans (skipping): %r", raw_line)
                continue

            entity_span = spans[0]   # e.g. "The developer" or "the designer"
            pronoun_span = spans[1]  # e.g. "he", "her"

            # Remove brackets to get the plain sentence.
            sentence = bracket_re.sub(lambda m: m.group(1), line_text).strip()

            # Identify all "the X" / "The X" mentions in the sentence to find the other entity.
            # We collect unique bare noun phrases (without "the"/"The") and pick the one that
            # is NOT the coreference entity.
            entity_bare = re.sub(r"^[Tt]he\s+", "", entity_span).strip().lower()
            other_entity_bare: str | None = None
            for m in re.finditer(r"\b[Tt]he\s+(\w+)", sentence):
                candidate = m.group(1).lower()
                if candidate != entity_bare:
                    other_entity_bare = candidate
                    break

            option_a = entity_span.strip()           # the coreference entity → label A
            option_b = (
                f"the {other_entity_bare}" if other_entity_bare else "the other person"
            )

            question = (
                f"In the following sentence, who does \"{pronoun_span}\" refer to?\n\n"
                f"\"{sentence}\""
            )

            records.append(
                _format_fairness_record(
                    task_name="winobias",
                    question=question,
                    options=[option_a.capitalize(), option_b.capitalize()],
                    target_label="A",  # bracket span 1 is always the coreference answer
                    group_id=group_id,
                    metadata={
                        "split": split_key,
                        "entity": entity_span,
                        "pronoun": pronoun_span,
                        "other_entity": other_entity_bare,
                        "sentence": sentence,
                        "is_pro_stereotyped": is_pro,
                    },
                )
            )

    if not records:
        raise RuntimeError(
            "Could not load WinoBias records. "
            "Set WINOBIAS_DATA_DIR to a folder containing the WinoBias .txt files "
            "or ensure GitHub access is available."
        )
    return records


def _load_difference_awareness_pairs_from_local(root: Path, file_name: str) -> tuple[list[Any], list[Any]] | None:
    local_path = root / file_name
    if not local_path.exists():
        return None
    with open(local_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, (list, tuple)) or len(payload) < 2:
        raise ValueError(f"Unexpected payload in {local_path}. Expected [different, equal].")
    return payload[0], payload[1]


def _load_difference_awareness_pairs(file_name: str) -> tuple[list[Any], list[Any]]:
    env_roots = [
        os.getenv("DIFFERENCE_AWARENESS_DIR"),
        os.getenv("DIFF_AWARENESS_DIR"),
    ]
    for env_root in env_roots:
        if not env_root:
            continue
        loaded = _load_difference_awareness_pairs_from_local(Path(env_root), file_name)
        if loaded is not None:
            return loaded

    url = f"{DIFF_AWARENESS_RAW_BASE}/{file_name}"
    blob = _cached_fetch_bytes(url)
    payload = pickle.loads(blob)
    if not isinstance(payload, (list, tuple)) or len(payload) < 2:
        raise ValueError(f"Unexpected payload from {url}. Expected [different, equal].")
    return payload[0], payload[1]


def _load_difference_awareness_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_name in DIFF_AWARENESS_FILES:
        try:
            different, equal = _load_difference_awareness_pairs(file_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping difference_awareness file %s due to error: %s", file_name, exc)
            continue

        benchmark_id = file_name.replace("_1k.pkl", "")
        for bucket_name, rows in (("different", different), ("equal", equal)):
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                question = _to_text(row[0]).strip()
                label = _coerce_choice_label(row[1])
                uid = row[2] if len(row) >= 3 else f"{benchmark_id}:{len(records)}"
                if not question or not label:
                    continue

                group_id = _extract_group_id_from_uid(uid, fallback=benchmark_id)
                records.append(
                    _format_fairness_record(
                        task_name="difference_awareness",
                        question=question,
                        options=None,  # options are already embedded as (a)/(b)/(c) in the question
                        target_label=label,
                        group_id=group_id,
                        metadata={
                            "benchmark": benchmark_id,
                            "bucket": bucket_name,
                            "uid": _to_text(uid),
                        },
                    )
                )

    if not records:
        raise RuntimeError(
            "Failed to load difference_awareness records. "
            "Set DIFFERENCE_AWARENESS_DIR to a folder containing D1_1k.pkl...N4_1k.pkl "
            "or ensure GitHub access is available."
        )
    return records


def load_fairness_training_dataset(
    task_name: str,
    eval_size: int = 200,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    canonical = _canonical_fairness_task_name(task_name)
    if canonical is None:
        raise ValueError(
            f"Unknown fairness task '{task_name}'. "
            f"Known fairness tasks: {sorted(FAIRNESS_TASK_ALIASES)}"
        )

    if canonical == "bbq":
        records = _load_bbq_records()
    elif canonical == "winobias":
        records = _load_winobias_records()
    elif canonical == "difference_awareness":
        records = _load_difference_awareness_records()
    else:
        raise ValueError(f"Unhandled fairness task '{canonical}'.")

    dataset = Dataset.from_list(records)
    train_raw, eval_raw = _split_dataset(dataset=dataset, eval_size=eval_size, seed=seed)
    return train_raw, eval_raw


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
    """Unified loader for SuperNI, TRACE, and fairness tasks.

    Args:
        task: SuperNI task name/NI id, SuperNITask dataclass, TRACE task name,
            TraceTask dataclass, or fairness task alias (bbq, crows_pairs,
            difference_awareness).
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
    if _canonical_fairness_task_name(task_name) is not None:
        return load_fairness_training_dataset(task_name=task_name, eval_size=eval_size, seed=seed)

    if task_name.startswith("task") or re.fullmatch(r"NI\d+", task_name):
        return load_superni_training_dataset(task_name, eval_size=eval_size, seed=seed)
    return load_trace_training_dataset(task_name=task_name, eval_size=eval_size, seed=seed)
