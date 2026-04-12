from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

try:
    from .load_dataset import load_training_dataset
    from .repro import set_global_seed
    from .task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS
except ImportError:
    from load_dataset import load_training_dataset
    from repro import set_global_seed
    from task_sequences import CORE_EVAL_TASKS, GENERAL_EVAL_TASKS


def _extract_primary_metric(task_result: Dict[str, float]) -> float | None:
    preferred = [
        "acc_norm,none",
        "acc,none",
        "exact_match,get-answer",
        "f1,none",
        "rougeL,none",
        "bleu,none",
    ]
    # Try preferred keys in priority order first.
    for key in preferred:
        if key in task_result and isinstance(task_result[key], float):
            return float(task_result[key])

    # Fallback: any exact_match or acc key (catches variant suffixes)
    for key, value in task_result.items():
        if isinstance(value, float) and (
            key.startswith("exact_match,") or key.startswith("acc,")
        ):
            return float(value)

    # Last resort: first float that isn't stderr or sample_len
    for key, value in task_result.items():
        if isinstance(value, float) and "stderr" not in key:
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


def _parse_available_fewshot_from_assertion(exc: AssertionError) -> int | None:
    text = str(exc)
    match = re.search(r"exceeds the\s+(\d+)\s+that are available", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _safe_simple_evaluate_group(
    lm_eval,
    lm,
    tasks: list[str],
    num_fewshot: int,
    batch_size: int,
) -> tuple[Dict[str, Any], Dict[str, int]]:
    """Evaluate a group of tasks in one lm-eval call.

    Keeps throughput high versus one-call-per-task. If few-shot exceeds the
    available examples for this task group, retry the full group with the
    parsed available value.
    """
    if not tasks:
        return {}, {}

    requested = num_fewshot
    try:
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=requested,
            batch_size=batch_size,
        )
        return out.get("results", {}), {task_name: requested for task_name in tasks}
    except AssertionError as exc:
        available = _parse_available_fewshot_from_assertion(exc)
        if available is None:
            raise
        if available >= requested:
            raise
        warnings.warn(
            "One or more tasks in this group do not support the requested "
            f"few-shot={requested}; retrying the group with few-shot={available}.",
            stacklevel=2,
        )
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=available,
            batch_size=batch_size,
        )
        return out.get("results", {}), {task_name: available for task_name in tasks}


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

    with _left_padding_for_generation(tokenizer):
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


def _tokenize_dataset_for_perplexity(dataset, tokenizer, max_length: int):
    return dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length),
        remove_columns=dataset.column_names,
    )


@torch.no_grad()
def _evaluate_task_perplexity(
    model,
    tokenizer,
    task,
    *,
    eval_size: int,
    max_seq_length: int,
    per_device_eval_batch_size: int,
    task_eval_samples: int,
    seed: int,
) -> Dict[str, Any]:
    _, eval_dataset = load_training_dataset(task=task, eval_size=eval_size, seed=seed)
    if task_eval_samples and len(eval_dataset) > task_eval_samples:
        eval_dataset = eval_dataset.select(range(task_eval_samples))
    eval_dataset = _tokenize_dataset_for_perplexity(
        eval_dataset,
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = _model_device(model)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        bs = int(batch["input_ids"].shape[0])
        total_loss += float(loss.item()) * bs
        total_examples += bs

    mean_loss = total_loss / max(1, total_examples)
    perplexity = float(math.exp(min(mean_loss, 20.0)))

    return {
        "score": None,
        "primary_metric": "perplexity",
        "eval_loss": float(mean_loss),
        "perplexity": perplexity,
        "n_samples": int(total_examples),
    }


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

    with _left_padding_for_generation(tokenizer):
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
    seed: int = 42,
) -> Dict[str, Any]:
    set_global_seed(seed)
    eval_task_keys = eval_task_keys or CORE_EVAL_TASKS
    lm_eval, _ = _import_lm_eval_modules()
    with _left_padding_for_generation(tokenizer):
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
            gp_results, _ = _safe_simple_evaluate_group(
                lm_eval=lm_eval,
                lm=lm,
                tasks=resolved_tasks,
                num_fewshot=0,
                batch_size=batch_size,
            )

            bbh_ip_tasks = [
                t for t in resolved_tasks if "bbh" in t and "object_counting" in t
            ]
            non_bbh_ip_tasks = [t for t in resolved_tasks if t not in bbh_ip_tasks]

            ip_results_non_bbh, ip_used_non_bbh = _safe_simple_evaluate_group(
                lm_eval=lm_eval,
                lm=lm,
                tasks=non_bbh_ip_tasks,
                num_fewshot=5,
                batch_size=batch_size,
            )
            ip_results_bbh, ip_used_bbh = _safe_simple_evaluate_group(
                lm_eval=lm_eval,
                lm=lm,
                tasks=bbh_ip_tasks,
                num_fewshot=3,
                batch_size=batch_size,
            )
            ip_results.update(ip_results_non_bbh)
            ip_results.update(ip_results_bbh)
            ip_fewshot_used.update(ip_used_non_bbh)
            ip_fewshot_used.update(ip_used_bbh)

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
                seed=seed,
            )
            ip_scores["alpaca"] = _evaluate_alpaca_rouge_l(
                model=model,
                tokenizer=tokenizer,
                num_fewshot=5,
                n_samples=alpaca_n_samples,
                batch_size=batch_size,
                max_new_tokens=alpaca_max_new_tokens,
                seed=seed,
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


def evaluate_seen_tasks_perplexity(
    model,
    tokenizer,
    seen_tasks,
    eval_size: int = 200,
    max_seq_length: int = 512,
    per_device_eval_batch_size: int = 8,
    task_eval_samples: int = 64,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for task in seen_tasks:
        task_name = getattr(task, "name", str(task))
        out[task_name] = _evaluate_task_perplexity(
            model=model,
            tokenizer=tokenizer,
            task=task,
            eval_size=eval_size,
            max_seq_length=max_seq_length,
            per_device_eval_batch_size=per_device_eval_batch_size,
            task_eval_samples=task_eval_samples,
            seed=seed,
        )

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
    quick_eval: bool = False,
    skip_general_eval: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    set_global_seed(seed)

    # -- seen-task evaluation --
    if quick_eval:
        seen = evaluate_seen_tasks_perplexity(
            model=model,
            tokenizer=tokenizer,
            seen_tasks=seen_tasks,
            eval_size=eval_size,
            task_eval_samples=task_eval_samples,
            seed=seed,
        )
    else:
        seen = evaluate_seen_tasks(
            model=model,
            tokenizer=tokenizer,
            seen_tasks=seen_tasks,
            output_dir=output_dir,
            eval_size=eval_size,
            max_new_tokens=task_eval_max_new_tokens,
            task_eval_samples=task_eval_samples,
            seed=seed,
        )

    # -- general evaluation --
    if quick_eval or skip_general_eval:
        general = {
            "gp": {},
            "ip": {},
            "gp_mean": None,
            "ip_mean": None,
            "mode": "quick_perplexity" if quick_eval else "skipped",
        }
    else:
        eval_keys = general_eval_task_keys or CORE_EVAL_TASKS
        general = evaluate_general_tasks(
            model=model,
            tokenizer=tokenizer,
            eval_task_keys=eval_keys,
            batch_size=general_eval_batch_size,
            seed=seed,
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