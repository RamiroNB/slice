"""Tokenization and collation for causal-LM fine-tuning.

Single source of truth for how a (prompt, target) pair becomes
(input_ids, attention_mask, labels). Shared by the trainer, the perplexity
eval, the SLICE gradient probe and the CL methods so that the loss the probe
measures is exactly the loss training optimizes.

Completion-only supervision
---------------------------
`load_dataset` emits ``text = prompt + target + eos`` together with the
``prompt`` and ``target`` pieces. Supervising every token of ``text`` makes the
objective overwhelmingly a language model over the *instruction*: SuperNI
definitions run hundreds of tokens while the target is often a single label
word, so <2% of the loss (median, across the tasks in this benchmark) lands on
the tokens eval actually scores. Masking the prompt with -100 restores the
correspondence between the training objective and the generate-and-match
metric in `eval.py`.

Truncation
----------
The completion is the only supervised signal, so it is the last thing to be
dropped: it is reserved out of the token budget first and the *prompt* is
truncated from the left (keeping any leading BOS). Right-truncating the joined
text instead — the legacy behaviour — silently deletes the target entirely for
long-prompt tasks, leaving those examples with no supervision at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

logger = logging.getLogger("cl_lora.data")

LABEL_IGNORE_INDEX = -100


def _split_prompt_completion(example: Dict[str, Any]) -> tuple[str, str]:
    """Split ``text`` into its prompt prefix and its completion suffix.

    Slicing the already-built ``text`` (rather than re-joining prompt+target)
    keeps the eos that `_build_chat_text` appended inside the completion, so
    the model is still supervised on when to stop.
    """
    text = example.get("text") or ""
    prompt = example.get("prompt") or ""
    if prompt and text.startswith(prompt):
        return prompt, text[len(prompt):]
    # No usable prompt boundary: supervise the whole sequence rather than
    # silently dropping the example.
    return "", text


def truncate_prompt_ids(prompt_ids: Sequence[int], budget: int, bos_token_id: int | None = None) -> List[int]:
    """Fit a prompt into ``budget`` tokens by dropping from the *left*.

    The prompt's tail carries the end of the input and the assistant header
    that cues generation, so right-truncation (the tokenizer default) removes
    exactly the part the model needs to answer. Any leading BOS is preserved.
    Shared by training and eval so both see the same truncation regime.
    """
    prompt_ids = list(prompt_ids)
    if len(prompt_ids) <= budget:
        return prompt_ids
    if budget <= 0:
        return []
    head: List[int] = []
    if bos_token_id is not None and prompt_ids and prompt_ids[0] == bos_token_id:
        head = [bos_token_id]
        budget -= 1
    return head + (prompt_ids[len(prompt_ids) - budget:] if budget > 0 else [])


def _encode_completion_only(example, tokenizer, max_length: int) -> Dict[str, Any]:
    prompt, completion = _split_prompt_completion(example)

    # add_special_tokens=False: text is already chat-templated and carries its
    # own BOS; letting the tokenizer add another double-BOSes it.
    prompt_ids: List[int] = (
        tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
    )
    completion_ids: List[int] = tokenizer(completion, add_special_tokens=False)["input_ids"]

    # Reserve the completion first, then spend what is left on the prompt.
    completion_ids = completion_ids[:max_length]
    budget = max_length - len(completion_ids)

    truncated = len(prompt_ids) > budget
    prompt_ids = truncate_prompt_ids(
        prompt_ids, budget, bos_token_id=getattr(tokenizer, "bos_token_id", None)
    )

    input_ids = prompt_ids + completion_ids
    labels = [LABEL_IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "_prompt_truncated": bool(truncated),
        "_has_supervision": bool(completion_ids),
    }


def _encode_full_sequence(example, tokenizer, max_length: int) -> Dict[str, Any]:
    encoded = tokenizer(
        example["text"], truncation=True, max_length=max_length, add_special_tokens=False
    )
    input_ids = encoded["input_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": encoded.get("attention_mask", [1] * len(input_ids)),
        "labels": list(input_ids),
        "_prompt_truncated": False,
        "_has_supervision": bool(input_ids),
    }


def tokenize_dataset(
    dataset,
    tokenizer,
    max_length: int,
    *,
    completion_only: bool = True,
    log_context: str = "",
):
    """Tokenize a prompt/target dataset into input_ids/attention_mask/labels.

    With ``completion_only`` (default) the prompt tokens are masked out of the
    labels; with it disabled every token is supervised, reproducing the legacy
    full-sequence objective.
    """
    encode = _encode_completion_only if completion_only else _encode_full_sequence
    tokenized = dataset.map(
        lambda ex: encode(ex, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )

    n_total = len(tokenized)
    supervised = tokenized["_has_supervision"]
    n_truncated = int(sum(tokenized["_prompt_truncated"]))
    n_dropped = int(sum(1 for ok in supervised if not ok))

    if n_dropped:
        # All-masked labels would make the loss a 0/0 nan.
        tokenized = tokenized.filter(lambda ex: ex["_has_supervision"])
    tokenized = tokenized.remove_columns(["_prompt_truncated", "_has_supervision"])

    prefix = f"[{log_context}] " if log_context else ""
    logger.info(
        "%stokenized %d examples (completion_only=%s, max_length=%d): "
        "%d prompt-truncated, %d dropped for having no supervised token",
        prefix, n_total, completion_only, max_length, n_truncated, n_dropped,
    )
    if completion_only and n_truncated:
        logger.warning(
            "%s%d/%d examples had their prompt left-truncated to fit max_length=%d; "
            "raise max_seq_length if this is a large fraction.",
            prefix, n_truncated, n_total, max_length,
        )
    return tokenized


@dataclass
class CausalLMCollator:
    """Pad a batch of input_ids/attention_mask/labels for causal-LM training.

    Padding positions are masked in the labels *positionally*, so — unlike
    `DataCollatorForLanguageModeling`, which masks by matching the pad token
    *value* — a pad id that collides with eos cannot delete real eos tokens
    from the training signal.
    """

    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None; cannot pad a batch.")

        width = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            width = ((width + m - 1) // m) * m

        pad_left = getattr(self.tokenizer, "padding_side", "right") == "left"
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f.get("attention_mask") or [1] * len(ids))
            lab = list(f["labels"]) if f.get("labels") is not None else list(ids)
            n_pad = width - len(ids)
            if pad_left:
                input_ids.append([pad_id] * n_pad + ids)
                attention_mask.append([0] * n_pad + mask)
                labels.append([LABEL_IGNORE_INDEX] * n_pad + lab)
            else:
                input_ids.append(ids + [pad_id] * n_pad)
                attention_mask.append(mask + [0] * n_pad)
                labels.append(lab + [LABEL_IGNORE_INDEX] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_collator(tokenizer, pad_to_multiple_of: int | None = None) -> CausalLMCollator:
    return CausalLMCollator(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)


def encode_prompts_for_generation(tokenizer, prompts: Sequence[str], max_length: int):
    """Tokenize eval prompts with the same left-truncation rule training uses.

    `tokenizer(..., truncation=True)` would cut the *end* of an over-long
    prompt, deleting the assistant header and leaving the model with a prompt
    that stops mid-input — so it continues the input instead of answering.
    Padding goes through `tokenizer.pad`, which honours the caller's
    padding_side (eval sets it to left for generation).
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    features = []
    for prompt in prompts:
        # add_special_tokens=False: prompts come from the chat template and
        # already carry their own BOS.
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = truncate_prompt_ids(ids, max_length, bos_token_id=bos_id)
        features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return tokenizer.pad(features, return_tensors="pt")
