"""
find_conflicting_seq.py

Measure gradient OPPOSITION (not just difference) across CL tasks.

SLICE's projection only fires when dot(g_forget, g_retain) < 0 — i.e. the
gradients point in genuinely opposite directions. Same-domain / same-output
tasks tend to be ALIGNED (cos > 0), and diverse tasks tend to be ORTHOGONAL
(cos ≈ 0); neither case exercises the projection. This module finds which
task pairs in the repo actually produce negative cosines, on the base model.

Registers a few candidate sequences at import time (SEQUENCES is mutated).

Two CLI modes:

    # Analyze one registered sequence pairwise:
    python -m cl_lora.find_conflicting_seq --sequence NI-Seq-Conflict-Rating

    # Search all pairs in a pool of tasks, rank most-opposite first:
    python -m cl_lora.find_conflicting_seq --search
    python -m cl_lora.find_conflicting_seq --search --pool NI363,NI195,NI618,NI589
"""
from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    import torch

from .task_sequences import (
    SEQUENCES,
    Sequence,
    SuperNITask,
    get_sequence,
    # classification tasks
    NI195,
    NI363,
    NI1292,
    NI1310,
    NI1343,
    # generation tasks
    NI024,
    NI511,
    NI589,
    NI618,
    NI1290,
    NI1357,
)

logger = logging.getLogger("cl_lora.find_conflicting_seq")


# ---------------------------------------------------------------------------
# Candidate sequences (hypotheses, to be validated empirically via --search)
#
# NOTE: "same-domain, same-task" sequences like "3 rating tasks in a row"
# produce ALIGNED gradients (cos > 0) — SLICE's projection does NOTHING on
# aligned gradients. For SLICE to fire you need OPPOSITE gradients (cos < 0),
# which typically requires cross-modality (classification <-> generation) or
# contradictory supervision on similar inputs.
# ---------------------------------------------------------------------------

# Cross-modality: classify vs generate on review-ish text. Hypothesis: the
# output-mode difference forces anti-aligned components on the decoder side.
NI_SEQ_CROSSMODE_REVIEWS = Sequence(
    name="NI-Seq-CrossMode-Reviews",
    task_type="mixed",
    tasks=[NI363, NI618, NI1310, NI589],
    description=(
        "Cross-modality on review text: SST2(classify) -> AmazonRev(generate) -> "
        "AmazonRating(classify) -> AmazonFood(generate). Output mode flips each stage."
    ),
)


# Cross-modality: binary classify -> QA generation.
NI_SEQ_CROSSMODE_QA = Sequence(
    name="NI-Seq-CrossMode-QA",
    task_type="mixed",
    tasks=[NI195, NI024, NI363, NI618],
    description=(
        "Classify -> QA-gen -> classify -> summarize. Forces repeated head swap."
    ),
)


# CONTROL GROUP — same-task-type runs, expected to be ALIGNED (cos > 0).
# SLICE should collapse into LoRA-GA on these. Included so --search can
# confirm the alignment prediction empirically.
NI_SEQ_CONTROL_RATING = Sequence(
    name="NI-Seq-Control-Rating",
    task_type="classification",
    tasks=[NI1343, NI1310, NI1292],
    description="Control: 3 rating tasks, expected cos>0 (SLICE inert).",
)


NI_SEQ_CONTROL_SUMM = Sequence(
    name="NI-Seq-Control-Summ",
    task_type="generation",
    tasks=[NI618, NI589, NI1290, NI511, NI1357],
    description="Control: 5 summarisation tasks, expected cos>0 (SLICE inert).",
)


_NEW_SEQUENCES = [
    NI_SEQ_CROSSMODE_REVIEWS,
    NI_SEQ_CROSSMODE_QA,
    NI_SEQ_CONTROL_RATING,
    NI_SEQ_CONTROL_SUMM,
]

_OUR_SEQUENCE_NAMES = {s.name for s in _NEW_SEQUENCES}
for _seq in _NEW_SEQUENCES:
    existing = SEQUENCES.get(_seq.name)
    if existing is not None and existing.name not in _OUR_SEQUENCE_NAMES:
        # A different module/user already registered this name; refuse to clobber.
        raise RuntimeError(
            f"Sequence name collision: {_seq.name!r} already registered "
            "in task_sequences.SEQUENCES by another module."
        )
    # Idempotent: overwriting our own entry is fine (e.g. double-import via
    # `python -m cl_lora.find_conflicting_seq` which loads the module twice).
    SEQUENCES[_seq.name] = _seq


# Default pool for --search: covers binary classify, rating classify,
# summarisation, and QA generation. Only uses known-good SuperNI IDs.
DEFAULT_SEARCH_POOL: List[SuperNITask] = [
    NI195,   # Sentiment140 - binary classification (tweets)
    NI363,   # SST2 - binary classification (movie reviews)
    NI1310,  # Amazon multi-rating - 5-star classification
    NI618,   # Amazon review - summary generation
    NI589,   # Amazon food review - summary generation
    NI1290,  # XSum - summary generation
    NI024,   # CosmosQA - QA generation
]


# ---------------------------------------------------------------------------
# Gradient conflict measurement
# ---------------------------------------------------------------------------

def _normalize_target_modules(target_modules: Any) -> List[str]:
    """Narrow peft.LoraConfig.target_modules (list[str] | str | None) to list[str]."""
    if target_modules is None:
        raise RuntimeError(
            "LoraConfig.target_modules is None; cannot determine which weights to grad."
        )
    if isinstance(target_modules, str):
        return [target_modules]
    return list(target_modules)


def compute_task_gradient(
    model: Any,
    tokenizer: Any,
    task: Any,
    *,
    max_steps: int = 4,
    batch_size: int = 4,
    max_seq_length: int = 256,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Average per-step gradient on one task's training data, base model only."""
    from .lora_config import build_lora_config
    from .load_dataset import load_training_dataset
    from .slice.gradients import accumulate_gradients
    from .slice.utils import (
        build_dataloader,
        model_device,
        target_weight_params,
        tokenize_dataset,
    )

    lora_cfg = build_lora_config()
    target_modules = _normalize_target_modules(lora_cfg.target_modules)
    target_params = target_weight_params(model, target_modules)
    ds, _ = load_training_dataset(task=task, eval_size=1, seed=seed)
    ds = tokenize_dataset(ds, tokenizer=tokenizer, max_length=max_seq_length)
    loader = build_dataloader(ds, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
    grads, steps = accumulate_gradients(
        model=model,
        dataloader=loader,
        target_params=target_params,
        device=model_device(model),
        max_steps=max_steps,
    )
    denom = max(1, steps)
    # Move to CPU so we can hold gradients for multiple tasks at once without
    # blowing up VRAM when iterating a task pool.
    return {k: (v / float(denom)).detach().to("cpu") for k, v in grads.items()}


def pair_conflict(
    grads_forget: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Conflict metrics treating `grads_forget` as new task, `grads_retain` as old.

    Reports what the SLICE projection would see:
      - global cosine similarity (sign indicates conflict)
      - pct_conflicting: % of modules with dot(g_f, g_r) < 0 (projection fires)
      - ratio_mean: mean ||G_tilde||/||G_forget|| across modules
    """
    import torch

    per_module_ratios = []
    per_module_cosines = []
    n_conflicting = 0
    n_modules = 0
    global_dot = 0.0
    global_nf_sq = 0.0
    global_nr_sq = 0.0
    eps = 1e-12

    for name, g_f in grads_forget.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            continue
        n_modules += 1
        f = g_f.detach().float().view(-1).double()
        r = g_r.detach().float().view(-1).double()
        dot = float(torch.dot(f, r).item())
        nf = float(f.norm().item())
        nr = float(r.norm().item())
        cos = dot / (nf * nr + eps)

        if dot < 0:
            n_conflicting += 1
            gamma = -dot / (nr * nr + eps)
            proj_norm = float((f + gamma * r).norm().item())
            ratio = proj_norm / (nf + eps)
        else:
            ratio = 1.0

        per_module_ratios.append(ratio)
        per_module_cosines.append(cos)
        global_dot += dot
        global_nf_sq += nf * nf
        global_nr_sq += nr * nr

    if n_modules == 0:
        return {"n_modules": 0}

    ratios = torch.tensor(per_module_ratios)
    cosines = torch.tensor(per_module_cosines)
    return {
        "n_modules": n_modules,
        "pct_conflicting": n_conflicting / n_modules * 100.0,
        "cosine_mean": float(cosines.mean()),
        "cosine_median": float(cosines.median()),
        "cosine_min": float(cosines.min()),
        "ratio_mean": float(ratios.mean()),
        "ratio_min": float(ratios.min()),
        "ratio_below_0.5_pct": float((ratios < 0.5).float().mean() * 100),
        "global_cosine": global_dot / ((global_nf_sq ** 0.5) * (global_nr_sq ** 0.5) + eps),
    }


def analyze_sequence(
    sequence: Sequence,
    *,
    model_name: str | None = None,
    max_steps: int = 4,
    batch_size: int = 4,
    max_seq_length: int = 256,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:  # keys are "forget|retain"
    """Load base model, compute per-task gradients, report all ordered pairs.

    Returns dict keyed by "forget|retain" with the pair_conflict stats.
    """
    from .train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model

    model_name = model_name or MODEL_NAME
    tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
    model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)

    logger.info(
        "Computing per-task gradients: n_tasks=%d  max_steps=%d  batch_size=%d",
        len(sequence.tasks), max_steps, batch_size,
    )
    task_grads: Dict[str, Dict[str, torch.Tensor]] = {}
    for task in sequence.tasks:
        name = getattr(task, "name", str(task))
        logger.info("  computing gradient for task=%s", name)
        task_grads[name] = compute_task_gradient(
            model, tokenizer, task,
            max_steps=max_steps, batch_size=batch_size,
            max_seq_length=max_seq_length, seed=seed,
        )

    task_names = list(task_grads)
    report: Dict[str, Dict[str, float]] = {}

    print()
    print("=" * 84)
    print(f"Pairwise gradient conflict — sequence: {sequence.name}")
    print(f"{sequence.description}")
    print(f"base model: {model_name}  (gradients on untrained base)")
    print("=" * 84)
    print(f"{'forget (new)':26s} {'retain (old)':26s} "
          f"{'glob_cos':>9s} {'mean_cos':>9s} {'ratio_mu':>9s} {'conf%':>6s}")
    print("-" * 84)
    for i, f_name in enumerate(task_names):
        for j, r_name in enumerate(task_names):
            if i == j:
                continue
            s = pair_conflict(task_grads[f_name], task_grads[r_name])
            report[f"{f_name}|{r_name}"] = s
            print(
                f"{f_name[:26]:26s} {r_name[:26]:26s} "
                f"{s['global_cosine']:+9.4f} {s['cosine_mean']:+9.4f} "
                f"{s['ratio_mean']:9.4f} {s['pct_conflicting']:5.1f}%"
            )
    print("-" * 84)
    print("glob_cos : global cosine sim (all modules flattened together)")
    print("mean_cos : mean per-module cosine sim")
    print("ratio_mu : mean ||G_tilde|| / ||G_forget|| (1.0 = projection is a no-op)")
    print("conf%    : % of modules where dot(g_forget, g_retain) < 0")
    print()
    return report


def _all_superni_tasks_safe() -> List[SuperNITask]:
    """Like task_sequences.all_superni_tasks but skips TraceTask entries.

    The upstream helper assumes every task in every registered sequence has
    `.ni_id`, which crashes on TRACE-Dummy (contains TraceTask). We filter.
    """
    seen: set = set()
    out: List[SuperNITask] = []
    for seq in SEQUENCES.values():
        for task in seq.tasks:
            nid = getattr(task, "ni_id", None)
            if nid is None or nid in seen:
                continue
            seen.add(nid)
            out.append(task)
    return out


def _resolve_task_pool(pool_arg: str | None) -> List[SuperNITask]:
    """Parse --pool arg into a list of SuperNITask objects.

    Accepts comma-separated NI-IDs ("NI195,NI363") or "all" (the full
    registered SuperNI catalogue) or None (the curated DEFAULT_SEARCH_POOL).
    """
    if pool_arg is None:
        return list(DEFAULT_SEARCH_POOL)
    if pool_arg.strip().lower() == "all":
        return _all_superni_tasks_safe()
    wanted = {tok.strip() for tok in pool_arg.split(",") if tok.strip()}
    by_id = {t.ni_id: t for t in _all_superni_tasks_safe()}
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(
            f"Unknown NI-IDs in --pool: {sorted(missing)}. "
            f"Known: {sorted(by_id)}"
        )
    return [by_id[nid] for nid in wanted]


def search_opposite_pairs(
    tasks: List[Any],
    *,
    model_name: str | None = None,
    max_steps: int = 4,
    batch_size: int = 4,
    max_seq_length: int = 256,
    seed: int = 42,
    top_k: int = 10,
) -> List[Tuple[str, str, Dict[str, float]]]:
    """Compute pairwise conflict across `tasks`, rank by most-opposite first.

    Returns all unordered pairs with their stats, sorted by global_cosine
    ascending (most negative = most opposite first). Also prints a top-K
    summary (most opposite AND most aligned, for contrast).
    """
    from .train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model

    model_name = model_name or MODEL_NAME
    tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
    model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)

    logger.info(
        "Search: n_tasks=%d  pairs=%d  max_steps=%d  batch_size=%d",
        len(tasks), len(tasks) * (len(tasks) - 1) // 2, max_steps, batch_size,
    )

    task_grads: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        name = getattr(task, "name", str(task))
        logger.info("  grad for task=%s", name)
        try:
            task_grads[name] = compute_task_gradient(
                model, tokenizer, task,
                max_steps=max_steps, batch_size=batch_size,
                max_seq_length=max_seq_length, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 — we want to skip broken tasks
            logger.warning("  SKIP %s (failed to load/compute): %s", name, exc)

    names = list(task_grads)
    pairs: List[Tuple[str, str, Dict[str, float]]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            s = pair_conflict(task_grads[a], task_grads[b])
            pairs.append((a, b, s))

    pairs.sort(key=lambda x: x[2].get("global_cosine", 0.0))

    print()
    print("=" * 96)
    print(f"Pair search — {len(names)} tasks, {len(pairs)} unordered pairs")
    print(f"base model: {model_name}  (gradients on untrained base)")
    print("=" * 96)
    header = (
        f"{'task_a':34s} {'task_b':34s} "
        f"{'glob_cos':>9s} {'mean_cos':>9s} {'conf%':>6s} {'ratio_mu':>9s}"
    )
    row = (
        "{:34s} {:34s} {:+9.4f} {:+9.4f} {:5.1f}% {:9.4f}"
    )

    print("\n[MOST OPPOSITE — SLICE projection fires here]")
    print(header)
    print("-" * 96)
    for a, b, s in pairs[:top_k]:
        print(row.format(
            a[:34], b[:34],
            s["global_cosine"], s["cosine_mean"],
            s["pct_conflicting"], s["ratio_mean"],
        ))

    print("\n[MOST ALIGNED — SLICE does nothing here]")
    print(header)
    print("-" * 96)
    for a, b, s in pairs[-top_k:][::-1]:
        print(row.format(
            a[:34], b[:34],
            s["global_cosine"], s["cosine_mean"],
            s["pct_conflicting"], s["ratio_mean"],
        ))
    print()
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure gradient opposition across CL tasks."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sequence",
        help="Analyse one registered sequence pairwise (ordered pairs).",
    )
    mode.add_argument(
        "--search", action="store_true",
        help="Search all unordered pairs in a task pool; rank by opposition.",
    )
    parser.add_argument(
        "--pool", default=None,
        help=(
            "For --search: comma-separated NI-IDs (e.g. NI195,NI363,NI618), "
            "'all' for the full catalogue, or omit for the default pool."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-name", default=None, help="Override base model.")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    lvl = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)

    if args.search:
        pool = _resolve_task_pool(args.pool)
        search_opposite_pairs(
            pool,
            model_name=args.model_name,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            top_k=args.top_k,
        )
    else:
        seq = get_sequence(args.sequence)
        analyze_sequence(
            seq,
            model_name=args.model_name,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )


__all__ = [
    "NI_SEQ_CROSSMODE_REVIEWS",
    "NI_SEQ_CROSSMODE_QA",
    "NI_SEQ_CONTROL_RATING",
    "NI_SEQ_CONTROL_SUMM",
    "DEFAULT_SEARCH_POOL",
    "compute_task_gradient",
    "pair_conflict",
    "analyze_sequence",
    "search_opposite_pairs",
]


if __name__ == "__main__":
    main()
