from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from .load_dataset import load_training_dataset
    from .repro import set_global_seed
except ImportError:
    from load_dataset import load_training_dataset
    from repro import set_global_seed


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_choice_label(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = _normalize_text(text)
    if not normalized:
        return None

    if normalized in {"a", "b", "c"}:
        return normalized.upper()

    if normalized in {"0", "1", "2"}:
        return {"0": "A", "1": "B", "2": "C"}[normalized]

    match = re.search(r"\b([abc])\b", normalized)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([012])\b", normalized)
    if match:
        return {"0": "A", "1": "B", "2": "C"}[match.group(1)]

    return None


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        dev = getattr(model, "device")
        if isinstance(dev, torch.device):
            return dev
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def _left_padding_for_generation(tokenizer):
    prev_padding_side = getattr(tokenizer, "padding_side", "right")
    prev_pad_token = getattr(tokenizer, "pad_token", None)
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        yield
    finally:
        tokenizer.padding_side = prev_padding_side
        tokenizer.pad_token = prev_pad_token


def evaluate_fairness_task(
    model,
    tokenizer,
    task: Any,
    *,
    eval_size: int = 200,
    task_eval_samples: int = 128,
    per_device_eval_batch_size: int = 8,
    max_new_tokens: int = 32,
    max_input_length: int = 512,
    seed: int = 42,
    min_group_count: int = 1,
    predictions_output_path: str | None = None,
) -> Dict[str, Any]:
    """Evaluate a single task for utility and fairness metrics.

    Args:
        min_group_count: Minimum number of examples a group must have to be
            included in WGA and GAP computation. Groups below the threshold are
            still tracked in ``group_accuracy`` / ``group_counts`` but excluded
            from ``worst_group_accuracy`` and ``group_accuracy_gap``.
            Use 1 (default) for smoke runs and ≥ 20 for real experiments.

    Returns run-level metrics and optional per-example predictions.
    """
    set_global_seed(seed)

    _, eval_dataset = load_training_dataset(task=task, eval_size=eval_size, seed=seed)
    if task_eval_samples and len(eval_dataset) > task_eval_samples:
        eval_dataset = eval_dataset.select(range(task_eval_samples))

    prompts = eval_dataset["prompt"]
    targets = eval_dataset["target"]
    group_ids = eval_dataset["group_id"] if "group_id" in eval_dataset.column_names else ["all"] * len(prompts)
    metadata_list = eval_dataset["metadata"] if "metadata" in eval_dataset.column_names else [{}] * len(prompts)

    device = _model_device(model)
    model.eval()

    per_group_total: dict[str, int] = {}
    per_group_correct: dict[str, int] = {}
    # BBQ context-condition tracking: keyed by context_condition value ("ambiguous" / "disambiguated")
    per_condition_total: dict[str, int] = {}
    per_condition_correct: dict[str, int] = {}
    exact_correct = 0
    parsed_total = 0
    parsed_correct = 0
    predictions: list[dict[str, Any]] = []

    with _left_padding_for_generation(tokenizer):
        for start in range(0, len(prompts), per_device_eval_batch_size):
            batch_prompts = prompts[start : start + per_device_eval_batch_size]
            batch_targets = targets[start : start + per_device_eval_batch_size]
            batch_groups = group_ids[start : start + per_device_eval_batch_size]
            batch_metadata = metadata_list[start : start + per_device_eval_batch_size]

            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for i, (target, group_id, meta) in enumerate(zip(batch_targets, batch_groups, batch_metadata)):
                continuation_ids = outputs[i, int(prompt_lengths[i]) :]
                prediction_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
                reference_text = str(target).strip()

                pred_label = _extract_choice_label(prediction_text)
                ref_label = _extract_choice_label(reference_text)

                if ref_label is not None:
                    is_correct = pred_label == ref_label
                    parsed_total += 1
                    parsed_correct += int(is_correct)
                else:
                    is_correct = _normalize_text(prediction_text) == _normalize_text(reference_text)

                exact_correct += int(is_correct)

                group_key = str(group_id)
                per_group_total[group_key] = per_group_total.get(group_key, 0) + 1
                per_group_correct[group_key] = per_group_correct.get(group_key, 0) + int(is_correct)

                # BBQ context-condition split
                context_condition: str | None = None
                if isinstance(meta, dict):
                    context_condition = meta.get("context_condition")
                if context_condition:
                    per_condition_total[context_condition] = per_condition_total.get(context_condition, 0) + 1
                    per_condition_correct[context_condition] = per_condition_correct.get(context_condition, 0) + int(is_correct)

                predictions.append(
                    {
                        "group_id": group_key,
                        "context_condition": context_condition,
                        "target": reference_text,
                        "prediction": prediction_text,
                        "target_label": ref_label,
                        "predicted_label": pred_label,
                        "is_correct": bool(is_correct),
                    }
                )

    group_accuracy: dict[str, float] = {}
    for group_id, total in per_group_total.items():
        if total <= 0:
            continue
        group_accuracy[group_id] = per_group_correct.get(group_id, 0) / total

    # Restrict WGA and GAP to groups that meet the minimum count threshold.
    eligible_groups = {g: acc for g, acc in group_accuracy.items() if per_group_total[g] >= min_group_count}
    wga = min(eligible_groups.values()) if eligible_groups else None
    gap = (max(eligible_groups.values()) - min(eligible_groups.values())) if len(eligible_groups) >= 2 else 0.0

    overall_accuracy = exact_correct / max(1, len(predictions))
    parsed_accuracy = (parsed_correct / parsed_total) if parsed_total > 0 else None

    # BBQ context-condition summary (only populated when context_condition metadata is present)
    bbq_conditions: dict[str, Any] = {}
    if per_condition_total:
        for condition, total in per_condition_total.items():
            acc = per_condition_correct.get(condition, 0) / total if total > 0 else None
            bbq_conditions[condition] = {"accuracy": acc, "n": total}

    if predictions_output_path:
        output_path = Path(predictions_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    fairness: dict[str, Any] = {
        "group_accuracy": group_accuracy,
        "group_counts": per_group_total,
        "worst_group_accuracy": wga,
        "group_accuracy_gap": gap,
        "eligible_group_count": len(eligible_groups),
        "min_group_count_threshold": min_group_count,
    }
    if bbq_conditions:
        fairness["bbq_conditions"] = bbq_conditions

    return {
        "utility": {
            "accuracy": overall_accuracy,
            "parsed_choice_accuracy": parsed_accuracy,
            "n_samples": len(predictions),
        },
        "fairness": fairness,
    }
