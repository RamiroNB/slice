from __future__ import annotations

import argparse
import importlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

try:
    from .load_dataset import load_training_dataset
    from .task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS
except ImportError:
    from load_dataset import load_training_dataset
    from task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS


def _extract_primary_metric(task_result: Dict[str, float]) -> float | None:
    preferred = [
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "f1,none",
        "rougeL,none",
        "bleu,none",
    ]
    for key in preferred:
        if key in task_result:
            return float(task_result[key])

    for value in task_result.values():
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _import_lm_eval_modules():
    lm_eval = importlib.import_module("lm_eval")
    hflm_module = importlib.import_module("lm_eval.models.huggingface")
    return lm_eval, hflm_module.HFLM


def _build_hflm(model, tokenizer, device: str = "cuda", dtype: str = "bfloat16"):
    _, hflm_cls = _import_lm_eval_modules()
    return hflm_cls(pretrained=model, tokenizer=tokenizer, device=device, dtype=dtype)


def _parse_available_fewshot_from_assertion(exc: AssertionError) -> int | None:
    text = str(exc)
    match = re.search(r"exceeds the\s+(\d+)\s+that are available", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _safe_simple_evaluate_single_task(
    lm_eval,
    lm,
    task_name: str,
    num_fewshot: int,
    batch_size: int,
) -> tuple[Dict[str, Any], int]:
    """Evaluate one lm-eval task and auto-retry if requested few-shot exceeds availability."""
    try:
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
        )
        return out.get("results", {}), num_fewshot
    except AssertionError as exc:
        available = _parse_available_fewshot_from_assertion(exc)
        if available is None:
            raise
        warnings.warn(
            f"Task '{task_name}' supports at most {available} few-shot examples; "
            f"retrying from requested {num_fewshot}.",
            stacklevel=2,
        )
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=available,
            batch_size=batch_size,
        )
        return out.get("results", {}), available


def _ip_fewshot_for_task(task_name: str) -> int:
    """Use task-specific few-shot for IP pass.

    - BBH object counting: 3-shot (dataset supports 3 exemplar few-shots)
    - Other general lm-eval tasks: 5-shot
    """
    lowered = task_name.lower()
    if "bbh" in lowered and "object_counting" in lowered:
        return 3
    return 5


LM_EVAL_TASK_ALIASES = {
    "commonsenseqa": ["commonsenseqa", "commonsense_qa"],
    "openbookqa": ["openbookqa", "openbook_qa"],
    "bbh_object_counting": ["bbh_object_counting", "bbh_cot_fewshot_object_counting"],
    "bbh_cot_fewshot_object_counting": ["bbh_cot_fewshot_object_counting", "bbh_object_counting"],
    "alpaca_eval": ["alpaca_eval", "alpaca", "alpacaeval"],
    "lambada_openai": ["lambada_openai", "lambada", "lambada_standard"],
}

DISALLOWED_GROUP_FALLBACKS = {"bbh", "leaderboard_bbh"}


def _resolve_general_eval_tasks(task_names: list[str]) -> tuple[list[str], list[str]]:
    """Resolve task aliases against the currently installed lm-eval registry.

    Returns:
        (resolved_tasks, skipped_original_tasks)
    """
    _, _ = _import_lm_eval_modules()
    manager_module = importlib.import_module("lm_eval.tasks.manager")
    task_manager = manager_module.TaskManager()

    resolved: list[str] = []
    skipped: list[str] = []

    for task_name in task_names:
        candidates = LM_EVAL_TASK_ALIASES.get(task_name, [task_name])
        selected = None
        for candidate in candidates:
            if candidate in DISALLOWED_GROUP_FALLBACKS:
                continue
            try:
                task_manager.load([candidate])
                selected = candidate
                break
            except Exception:
                continue

        if selected is None:
            skipped.append(task_name)
            continue
        resolved.append(selected)

    return resolved, skipped


def _build_alpaca_prompt(instruction: str, input_text: str) -> str:
    if input_text.strip():
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:"
    )


@torch.no_grad()
def _evaluate_alpaca_rouge_l(
    model,
    tokenizer,
    num_fewshot: int = 0,
    n_samples: int = 190,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    max_input_length: int = 512,
    seed: int = 42,
) -> float | None:
    try:
        from datasets import load_dataset as hf_load_dataset
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Alpaca evaluation skipped (datasets import failed): {exc}", stacklevel=2)
        return None

    try:
        dataset = hf_load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Alpaca evaluation skipped (dataset unavailable): {exc}", stacklevel=2)
        return None

    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples + 5, len(dataset))))

    def _row(ds, idx: int) -> Dict[str, Any]:
        value = ds[idx]
        if isinstance(value, dict):
            return value
        try:
            return dict(value)
        except Exception:
            return {
                "instruction": "",
                "input": "",
                "output": "",
            }

    if num_fewshot > 0:
        icl_indices = range(min(num_fewshot, len(dataset)))
        test_indices = range(num_fewshot, min(n_samples + num_fewshot, len(dataset)))
        icl_examples = [_row(dataset, i) for i in icl_indices]
        test_examples = [_row(dataset, i) for i in test_indices]
    else:
        icl_examples = []
        test_examples = [_row(dataset, i) for i in range(min(n_samples, len(dataset)))]

    prefix = ""
    if num_fewshot > 0:
        for ex in icl_examples:
            prompt = _build_alpaca_prompt(str(ex.get("instruction", "")), str(ex.get("input", "")))
            prefix += prompt + " " + str(ex.get("output", "")).strip() + "\n\n"

    device = _model_device(model)
    rouge_scores: List[float] = []

    prompts = [
        prefix + _build_alpaca_prompt(str(ex.get("instruction", "")), str(ex.get("input", "")))
        for ex in test_examples
    ]
    references = [str(ex.get("output", "")).strip() for ex in test_examples]

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_refs = references[start : start + batch_size]

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
        for i, ref in enumerate(batch_refs):
            continuation_ids = outputs[i, int(prompt_lengths[i]) :]
            prediction = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            rouge_scores.append(_rouge_l_f1(prediction, ref))

    return _mean(rouge_scores)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    dp = [0] * (len(tokens_b) + 1)
    for a in tokens_a:
        prev = 0
        for j, b in enumerate(tokens_b, start=1):
            cur = dp[j]
            if a == b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(reference))


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        dev = getattr(model, "device")
        if isinstance(dev, torch.device):
            return dev
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _primary_metric_name(task: Any) -> str:
    if hasattr(task, "metric") and str(getattr(task, "metric", "")).lower() == "rouge-l":
        return "rouge_l"
    if hasattr(task, "category") and str(getattr(task, "category", "")).lower() == "classification":
        return "exact_match"
    if hasattr(task, "category") and str(getattr(task, "category", "")).lower() == "generation":
        return "rouge_l"
    return "exact_match"


@torch.no_grad()
def _evaluate_task_with_generation(
    model,
    tokenizer,
    eval_dataset,
    batch_size: int,
    max_new_tokens: int,
    max_input_length: int,
    primary_metric: str,
) -> Dict[str, Any]:
    device = _model_device(model)
    exact_scores: list[float] = []
    rouge_scores: list[float] = []

    prompts = eval_dataset["prompt"]
    targets = eval_dataset["target"]

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]

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
        for i, target in enumerate(batch_targets):
            continuation_ids = outputs[i, int(prompt_lengths[i]) :]
            prediction = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            ref = str(target).strip()

            exact_scores.append(_exact_match(prediction, ref))
            rouge_scores.append(_rouge_l_f1(prediction, ref))

    exact_match = _mean(exact_scores) or 0.0
    rouge_l = _mean(rouge_scores) or 0.0
    score = exact_match if primary_metric == "exact_match" else rouge_l

    return {
        "score": score,
        "primary_metric": primary_metric,
        "exact_match": exact_match,
        "rouge_l": rouge_l,
        "n_samples": len(prompts),
    }


def evaluate_general_tasks(
    model,
    tokenizer,
    eval_task_keys: list[str] | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "bfloat16",
    alpaca_n_samples: int = 190,
    alpaca_max_new_tokens: int = 128,
) -> Dict[str, Any]:
    eval_task_keys = eval_task_keys or CORE_EVAL_TASKS
    lm_eval, _ = _import_lm_eval_modules()
    lm = _build_hflm(model=model, tokenizer=tokenizer, device=device, dtype=dtype)

    lm_eval_keys = [k for k in eval_task_keys if k != "alpaca"]
    has_alpaca = "alpaca" in eval_task_keys

    lm_eval_task_names = [GENERAL_EVAL_TASKS[k]["lm_eval_name"] for k in lm_eval_keys]
    resolved_tasks, skipped_tasks = _resolve_general_eval_tasks(lm_eval_task_names)

    gp_scores: Dict[str, float | None] = {}
    ip_scores: Dict[str, float | None] = {}
    gp_results: Dict[str, Any] = {}
    ip_results: Dict[str, Any] = {}
    ip_fewshot_used: Dict[str, int] = {}

    if skipped_tasks:
        warnings.warn(
            "Skipping unavailable lm-eval tasks: " + ", ".join(skipped_tasks),
            stacklevel=2,
        )

    if resolved_tasks:
        for resolved_task in resolved_tasks:
            gp_task_results, _ = _safe_simple_evaluate_single_task(
                lm_eval=lm_eval,
                lm=lm,
                task_name=resolved_task,
                num_fewshot=0,
                batch_size=batch_size,
            )
            ip_fewshot = _ip_fewshot_for_task(resolved_task)
            ip_task_results, used = _safe_simple_evaluate_single_task(
                lm_eval=lm_eval,
                lm=lm,
                task_name=resolved_task,
                num_fewshot=ip_fewshot,
                batch_size=batch_size,
            )
            gp_results.update(gp_task_results)
            ip_results.update(ip_task_results)
            ip_fewshot_used[resolved_task] = used

        for key in lm_eval_keys:
            configured = GENERAL_EVAL_TASKS[key]["lm_eval_name"]
            candidates = LM_EVAL_TASK_ALIASES.get(configured, [configured])
            resolved_name = next((name for name in candidates if name in gp_results), None)
            gp_scores[key] = (
                _extract_primary_metric(gp_results[resolved_name]) if resolved_name else None
            )
            ip_scores[key] = (
                _extract_primary_metric(ip_results[resolved_name]) if resolved_name else None
            )

    if has_alpaca:
        gp_scores["alpaca"] = _evaluate_alpaca_rouge_l(
            model=model,
            tokenizer=tokenizer,
            num_fewshot=0,
            n_samples=alpaca_n_samples,
            batch_size=batch_size,
            max_new_tokens=alpaca_max_new_tokens,
        )
        ip_scores["alpaca"] = _evaluate_alpaca_rouge_l(
            model=model,
            tokenizer=tokenizer,
            num_fewshot=5,
            n_samples=alpaca_n_samples,
            batch_size=batch_size,
            max_new_tokens=alpaca_max_new_tokens,
        )

    return {
        "gp": gp_scores,
        "ip": ip_scores,
        "gp_mean": _mean(v for v in gp_scores.values() if v is not None),
        "ip_mean": _mean(v for v in ip_scores.values() if v is not None),
        "resolved_tasks": resolved_tasks,
        "skipped_tasks": skipped_tasks,
        "ip_fewshot_used": ip_fewshot_used,
        "raw": {
            "gp": gp_results,
            "ip": ip_results,
        },
    }


def evaluate_seen_tasks(
    model,
    tokenizer,
    seen_tasks,
    output_dir: str,
    eval_size: int = 200,
    max_input_length: int = 512,
    per_device_eval_batch_size: int = 8,
    max_new_tokens: int = 64,
    task_eval_samples: int = 64,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for task in seen_tasks:
        task_name = getattr(task, "name", str(task))
        _, eval_dataset = load_training_dataset(task=task, eval_size=eval_size, seed=seed)
        if task_eval_samples and len(eval_dataset) > task_eval_samples:
            eval_dataset = eval_dataset.select(range(task_eval_samples))

        task_metrics = _evaluate_task_with_generation(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            batch_size=per_device_eval_batch_size,
            max_new_tokens=max_new_tokens,
            max_input_length=max_input_length,
            primary_metric=_primary_metric_name(task),
        )
        out[task_name] = {
            **task_metrics,
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def evaluate_all(
    model,
    tokenizer,
    seen_tasks,
    output_dir: str,
    general_eval_task_keys: list[str] | None = None,
    general_eval_batch_size: int = 8,
    eval_size: int = 200,
    task_eval_samples: int = 64,
    task_eval_max_new_tokens: int = 64,
) -> Dict[str, Any]:
    eval_keys = general_eval_task_keys or CORE_EVAL_TASKS

    general = evaluate_general_tasks(
        model=model,
        tokenizer=tokenizer,
        eval_task_keys=eval_keys,
        batch_size=general_eval_batch_size,
    )
    seen = evaluate_seen_tasks(
        model=model,
        tokenizer=tokenizer,
        seen_tasks=seen_tasks,
        output_dir=output_dir,
        eval_size=eval_size,
        max_new_tokens=task_eval_max_new_tokens,
        task_eval_samples=task_eval_samples,
    )
    return {
        "general": general,
        "seen_tasks": seen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GP/IP on lm-eval tasks using a HF model.")
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--peft", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--tasks", nargs="+", default=["hellaswag"])
    parser.add_argument("--output", default="results")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    lm_eval, hflm_cls = _import_lm_eval_modules()
    model = hflm_cls(
        pretrained=args.model,
        peft=args.peft,
        device="cuda",
        dtype="bfloat16",
    )
    gp_raw = lm_eval.simple_evaluate(
        model=model,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
    )
    ip_raw = lm_eval.simple_evaluate(
        model=model,
        tasks=args.tasks,
        num_fewshot=5,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gp_results.json", "w", encoding="utf-8") as f:
        json.dump(gp_raw.get("results", {}), f, indent=2)
    with open(output_dir / "ip_results.json", "w", encoding="utf-8") as f:
        json.dump(ip_raw.get("results", {}), f, indent=2)

    print("Saved GP/IP results to", output_dir)


if __name__ == "__main__":
    main()